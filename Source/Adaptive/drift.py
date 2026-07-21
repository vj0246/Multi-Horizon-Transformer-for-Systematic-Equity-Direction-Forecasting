"""Concept-drift detectors - the FAST layer, which changes no model parameters.

This layer only ever observes and reports. It exists because the project has
already been bitten once by undetected drift: the frozen validation entry
threshold went degenerate on test when the signal's LEVEL shifted (validation
mean 0.00 vs test -0.79), so the rule stopped trading entirely. That was found by
accident. Detectors turn it into something measured.

Two complementary tests, both implemented here rather than pulled in from
`river` so the daily CI job keeps its small dependency set:

  ADWIN         - adaptive windowing (Bifet & Gavalda 2007). Keeps a window of
                  recent values, and whenever any split of that window has two
                  halves whose means differ by more than a Hoeffding-style
                  bound, declares drift and drops the stale half. Good at
                  locating gradual distribution change without a fixed window.
  Page-Hinkley  - cumulative-deviation test. Tracks the running sum of
                  (x - running_mean - delta) and alarms when it departs from its
                  own minimum by more than lambda. Good at abrupt level shifts,
                  which is exactly the failure already observed.

Both are one-pass and O(1) amortised per sample, so they cost nothing in the
daily job. Neither is used to trigger a parameter update automatically -
promotion decisions live in retrain.py behind a champion/challenger gate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PageHinkley:
    """Cumulative-deviation drift test for abrupt mean shifts.

    The input is STANDARDISED by a running standard deviation before it enters
    the cumulative sum. Without that, the statistic is an unnormalised random
    walk whose range grows like sqrt(T), so any fixed `lam` eventually trips on
    stationary noise - a false alarm that looks exactly like real drift. The
    default `lam` below was chosen empirically as the smallest value with zero
    false alarms over stationary N(0,1) input while still catching a 3-sigma
    shift (see test_drift_detectors_fire_on_shift_and_stay_quiet_when_stationary).
    """

    delta: float = 0.005            # slack: drift smaller than this is ignored
    lam: float = 80.0               # alarm threshold, in standardised units
    min_samples: int = 60

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0                 # Welford accumulator for the running variance
    cum: float = 0.0                # running sum of standardised deviations
    min_cum: float = 0.0
    _alarms: list[int] = field(default_factory=list)

    def update(self, x: float) -> bool:
        self.n += 1
        prev_mean = self.mean
        self.mean += (x - self.mean) / self.n                    # online mean
        self.m2 += (x - prev_mean) * (x - self.mean)             # online variance
        if self.n < 2:
            return False
        sd = math.sqrt(self.m2 / (self.n - 1)) or 1.0
        self.cum += (x - self.mean) / sd - self.delta            # scale-free
        self.min_cum = min(self.min_cum, self.cum)
        if self.n < self.min_samples:
            return False
        if self.cum - self.min_cum > self.lam:
            self._alarms.append(self.n)
            self.reset()
            return True
        return False

    def reset(self) -> None:
        self.n, self.mean, self.m2 = 0, 0.0, 0.0
        self.cum, self.min_cum = 0.0, 0.0


@dataclass
class ADWIN:
    """Adaptive windowing: shrink the window whenever its halves disagree.

    Simplified but faithful to the original test - exact ADWIN keeps exponential
    histogram buckets for O(log n) memory; here the raw window is retained, which
    is fine at daily frequency (a few thousand points at most) and keeps the cut
    logic readable.
    """

    delta: float = 0.002            # confidence; smaller = more conservative
    min_samples: int = 60
    max_window: int = 2000

    window: list[float] = field(default_factory=list)
    _alarms: list[int] = field(default_factory=list)
    total_seen: int = 0

    def _eps_cut(self, n0: int, n1: int) -> float:
        """Hoeffding-style bound on the tolerable difference of two sub-means."""
        m = 1.0 / (1.0 / n0 + 1.0 / n1)               # harmonic mean of the halves
        n = max(len(self.window), 2)
        d = self.delta / n                            # Bonferroni over split points
        var = _variance(self.window)
        return math.sqrt(2.0 / m * var * math.log(2.0 / d)) + 2.0 / (3.0 * m) * math.log(2.0 / d)

    def update(self, x: float) -> bool:
        self.total_seen += 1
        self.window.append(float(x))
        if len(self.window) > self.max_window:
            self.window.pop(0)
        if len(self.window) < self.min_samples:
            return False

        # scan split points; cut at the first one whose halves disagree
        for i in range(5, len(self.window) - 5):
            w0, w1 = self.window[:i], self.window[i:]
            m0 = sum(w0) / len(w0)
            m1 = sum(w1) / len(w1)
            if abs(m0 - m1) > self._eps_cut(len(w0), len(w1)):
                self.window = w1                       # drop the stale prefix
                self._alarms.append(self.total_seen)
                return True
        return False


def _variance(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return sum((v - m) ** 2 for v in xs) / (n - 1)


def scan(values, dates, cfg: dict) -> dict:
    """Run the configured detectors over a signal series -> drift report.

    `values` is whatever should be monitored: the ensemble signal itself catches
    level shifts, a rolling error series catches skill decay. Returns the alarm
    dates per detector plus the segment means either side of each alarm, so a
    reader can see the size of the shift rather than only that one occurred.
    """
    d = cfg.get("adaptive", {}).get("drift", {})
    names = d.get("detectors", ["adwin", "page_hinkley"])
    ms = int(d.get("min_samples", 60))

    dets = {}
    if "adwin" in names:
        dets["adwin"] = ADWIN(delta=float(d.get("adwin_delta", 0.002)), min_samples=ms)
    if "page_hinkley" in names:
        dets["page_hinkley"] = PageHinkley(delta=float(d.get("ph_delta", 0.005)),
                                           lam=float(d.get("ph_lambda", 80.0)),
                                           min_samples=ms)

    events = {k: [] for k in dets}
    for i, v in enumerate(values):
        for name, det in dets.items():
            if det.update(float(v)):
                lo = max(0, i - 60)
                before = values[lo:i]
                after = values[i:i + 60]
                events[name].append({
                    "date": str(dates[i]),
                    "index": int(i),
                    "mean_before": float(sum(before) / len(before)) if len(before) else None,
                    "mean_after": float(sum(after) / len(after)) if len(after) else None,
                })

    return {
        "n_observations": len(values),
        "detectors": {
            name: {"n_alarms": len(ev), "alarms": ev} for name, ev in events.items()
        },
        "note": (
            "Detectors observe only - they never trigger a parameter update on "
            "their own. An alarm is evidence to inspect, not permission to "
            "retrain: at ~250 observations a year, drift alarms and false alarms "
            "are hard to tell apart from a single firing."
        ),
    }
