"""Provider-abstracted LLM client with the guardrails this project requires.

Never import a vendor SDK into business logic. `complete()` takes a provider name
and returns text; swapping Groq for Anthropic is a config change, not a code
change.

Guardrails implemented here rather than left to callers:
  input     length cap + injection screening on anything interpolated
  output    caller-supplied schema validation, one retry, then a deterministic
            fallback - never a partial or hallucinated artifact
  failure   explicit timeout, bounded exponential backoff on 429/5xx
  cost      max_tokens always set, retries bounded, token usage returned

The whole layer is OPTIONAL and OFF by default. With no API key the pipeline
writes deterministic template commentary instead, so the site is never blocked on
a paid service and CI never needs a secret.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Callable

MAX_INPUT_CHARS = 12000
MAX_TOKENS = 900
TIMEOUT_S = 30
MAX_RETRIES = 3

# Patterns that try to escape the prompt's framing. This payload is generated
# from our own artifacts so injection is unlikely, but the artifacts are written
# by scripts a contributor could change, and screening is cheap.
_INJECTION = re.compile(
    r"(ignore (all |the )?(previous|prior|above)|disregard (the )?(system|instructions)"
    r"|you are now|new instructions?:|</?system>|<\|im_start\|>)",
    re.IGNORECASE,
)


class AdvisorError(RuntimeError):
    pass


def screen_input(text: str) -> str:
    """Cap length and reject prompt-injection attempts before interpolation."""
    if len(text) > MAX_INPUT_CHARS:
        raise AdvisorError(f"payload {len(text)} chars exceeds {MAX_INPUT_CHARS} cap")
    if _INJECTION.search(text):
        raise AdvisorError("payload contains prompt-injection-like text; refusing to send")
    return text


def _post_with_backoff(fn: Callable[[], object]) -> object:
    """Retry on rate limits and 5xx with bounded exponential backoff."""
    delay = 1.0
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as e:                       # noqa: BLE001 - provider SDKs vary
            last = e
            status = getattr(e, "status_code", None) or getattr(e, "status", None)
            retryable = status in (429, 500, 502, 503, 504) or "timeout" in str(e).lower()
            if not retryable or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise AdvisorError(f"exhausted retries: {last}")


def complete(system: str, user: str, provider: str = "groq",
             model: str | None = None) -> dict:
    """One completion. Returns {text, provider, model, usage}."""
    screen_input(user)

    if provider == "groq":
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise AdvisorError("GROQ_API_KEY not set")
        from groq import Groq
        c = Groq(api_key=key, timeout=TIMEOUT_S)
        mdl = model or "llama-3.3-70b-versatile"
        r = _post_with_backoff(lambda: c.chat.completions.create(
            model=mdl, max_tokens=MAX_TOKENS, temperature=0.2,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}]))
        return {"text": r.choices[0].message.content, "provider": provider, "model": mdl,
                "usage": {"input_tokens": r.usage.prompt_tokens,
                          "output_tokens": r.usage.completion_tokens}}

    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise AdvisorError("ANTHROPIC_API_KEY not set")
        import anthropic
        c = anthropic.Anthropic(api_key=key, timeout=TIMEOUT_S)
        mdl = model or "claude-sonnet-5"
        r = _post_with_backoff(lambda: c.messages.create(
            model=mdl, max_tokens=MAX_TOKENS, temperature=0.2, system=system,
            messages=[{"role": "user", "content": user}]))
        return {"text": r.content[0].text, "provider": provider, "model": mdl,
                "usage": {"input_tokens": r.usage.input_tokens,
                          "output_tokens": r.usage.output_tokens}}

    raise AdvisorError(f"unknown provider {provider!r}")


def complete_json(system: str, user: str, validate: Callable[[dict], None],
                  provider: str = "groq", model: str | None = None) -> dict:
    """Completion that must parse as JSON and satisfy `validate`.

    One retry on failure with the error fed back, then give up. Never returns a
    partially-valid object - the caller falls back to deterministic text instead,
    because a malformed artifact on a site about statistical honesty is worse
    than no artifact.
    """
    attempt_user = user
    last_err = None
    for _ in range(2):
        r = complete(system, attempt_user, provider=provider, model=model)
        raw = (r["text"] or "").strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        try:
            obj = json.loads(raw)
            validate(obj)
            return {**r, "json": obj}
        except Exception as e:                        # noqa: BLE001
            last_err = e
            attempt_user = (f"{user}\n\nYour previous reply was rejected: {e}. "
                            "Reply with valid JSON matching the schema exactly, "
                            "and nothing else.")
    raise AdvisorError(f"schema validation failed after retry: {last_err}")
