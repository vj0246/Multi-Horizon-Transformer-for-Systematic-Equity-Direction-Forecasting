"""Thompson sampling over fixed rules - reinforcement learning, sized to the data.

Full RL (DQN/PPO) learns a policy over a state-action space and needs ~1e6
decisions. This book has made 13. A bandit is the same family of idea reduced to
what the data can actually support: it does not learn a policy, it only allocates
between a handful of ALREADY-VALIDATED rules, so the thing being estimated is a
few scalars rather than a network.

Thompson sampling specifically, rather than epsilon-greedy or UCB, because it
represents its own uncertainty as a posterior. With this little data that is the
whole point: the honest output is not "arm 2 is best", it is "the posteriors
overlap almost completely and the data cannot yet tell these arms apart", and a
Beta posterior states that directly instead of hiding it behind a point estimate.

The guard that matters is `overlap()`. Any bandit will eventually name a winner -
that is what argmax does, regardless of evidence. Reporting that winner without
reporting how much the posteriors overlap is how a bandit launders noise into a
decision.
"""
from __future__ import annotations

import numpy as np


class ThompsonBandit:
    """Beta-Bernoulli bandit over named arms (each arm = one fixed strategy rule).

    Reward is binary: did this arm's decision make money net of costs. Binary
    rather than continuous return because with ~13 observations a mean-return
    estimator is dominated by a single outlier, whereas a Beta posterior degrades
    gracefully and its width is interpretable.
    """

    def __init__(self, arms: list[str], prior_a: float = 1.0, prior_b: float = 1.0):
        self.arms = list(arms)
        self.a = {k: float(prior_a) for k in self.arms}     # successes + prior
        self.b = {k: float(prior_b) for k in self.arms}     # failures + prior
        self.pulls = {k: 0 for k in self.arms}
        self.history: list[dict] = []

    def select(self, rng: np.random.Generator) -> str:
        """Sample one draw per arm from its posterior; play the highest."""
        draws = {k: float(rng.beta(self.a[k], self.b[k])) for k in self.arms}
        return max(draws, key=draws.get)

    def update(self, arm: str, reward: int, date: str | None = None) -> None:
        if arm not in self.arms:
            raise KeyError(f"unknown arm {arm}")
        if reward:
            self.a[arm] += 1
        else:
            self.b[arm] += 1
        self.pulls[arm] += 1
        self.history.append({"date": date, "arm": arm, "reward": int(reward)})

    def posterior(self, arm: str) -> dict:
        a, b = self.a[arm], self.b[arm]
        mean = a / (a + b)
        var = a * b / ((a + b) ** 2 * (a + b + 1))
        sd = float(np.sqrt(var))
        from scipy import stats
        lo, hi = stats.beta.ppf([0.025, 0.975], a, b)
        return {"arm": arm, "pulls": self.pulls[arm], "alpha": a, "beta": b,
                "mean": float(mean), "sd": sd,
                "ci95": [float(lo), float(hi)]}

    def overlap(self, rng: np.random.Generator, n: int = 20000) -> dict:
        """P(each arm is best) by Monte Carlo over the posteriors.

        This is the honesty check. If the top arm's probability of being best is
        near 1/n_arms, the bandit has learned nothing and its argmax is noise.
        """
        samples = {k: rng.beta(self.a[k], self.b[k], size=n) for k in self.arms}
        stacked = np.vstack([samples[k] for k in self.arms])
        winners = np.argmax(stacked, axis=0)
        p_best = {k: float((winners == i).mean()) for i, k in enumerate(self.arms)}
        top = max(p_best, key=p_best.get)
        uniform = 1.0 / len(self.arms)
        return {
            "p_best": p_best,
            "top_arm": top,
            "p_top_is_best": p_best[top],
            "uniform_baseline": uniform,
            "separated": bool(p_best[top] > 0.95),
            "verdict": (
                f"{top} is best with probability {p_best[top]:.2f}"
                if p_best[top] > 0.95 else
                f"No arm is distinguishable: the leader ({top}) is best with only "
                f"probability {p_best[top]:.2f} against a {uniform:.2f} coin-flip "
                "baseline. The argmax here is noise, not a decision."
            ),
        }

    def report(self, rng: np.random.Generator) -> dict:
        return {
            "arms": [self.posterior(k) for k in self.arms],
            "total_pulls": int(sum(self.pulls.values())),
            "separation": self.overlap(rng),
            "note": (
                "A bandit allocates between fixed, already-validated rules; it "
                "does not learn a policy. That is deliberate - policy learning "
                "needs ~1e6 decisions and this book has made ~13. Read the "
                "separation verdict before the arm means: argmax always names a "
                "winner whether or not one exists."
            ),
        }


def replay(arm_returns: dict[str, list[float]], seed: int = 42) -> dict:
    """Offline replay: what a bandit WOULD have learned from historical arms.

    Strictly a diagnostic. Every arm's return is known for every period here, so
    this is not a live experiment and its regret is not an out-of-sample result -
    it only answers whether the arms are separable at all at this sample size.
    """
    arms = list(arm_returns)
    n = min(len(v) for v in arm_returns.values())
    if n == 0:
        return {"error": "no periods to replay"}

    rng = np.random.default_rng(seed)
    bandit = ThompsonBandit(arms)
    chosen_returns = []
    for t in range(n):
        arm = bandit.select(rng)
        r = arm_returns[arm][t]
        bandit.update(arm, int(r > 0))
        chosen_returns.append(r)

    best_fixed = max(arms, key=lambda k: float(np.sum(arm_returns[k][:n])))
    rep = bandit.report(rng)
    rep["replay"] = {
        "n_periods": int(n),
        "bandit_total_return": float(np.sum(chosen_returns)),
        "best_single_arm": best_fixed,
        "best_single_arm_return": float(np.sum(arm_returns[best_fixed][:n])),
        "regret_vs_best_fixed": float(np.sum(arm_returns[best_fixed][:n])
                                      - np.sum(chosen_returns)),
        "caveat": (
            "Offline replay with full knowledge of every arm's outcome. Not an "
            "out-of-sample result and not evidence the bandit would help live."
        ),
    }
    return rep
