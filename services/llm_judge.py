"""LLM judge — per-step critic + goal-alignment check.

## Трассируемость
Feature: Telegram AI Platform v1.1
Requirements: FR-21, FR-22, NFR-14

Two judgement primitives the runtime calls after every step:

1. `critic(...)` — verifies that the step's actual_result matches its
   expected_result and acceptance criteria. Returns `Verdict` with
   `verdict ∈ {pass, fail}` and a short reason. Critic-fail is a hard
   veto: the runtime stops the graph and sets `replan_signal=critic_failed`.

2. `goal_alignment(...)` — measures how far the latest step has drifted
   from the original goal. Returns `Alignment` with `drift ∈ [0, 1]`
   and `should_replan: bool`. Drift-replan is softer than critic-fail:
   the run signals `replan_signal=goal_drift` so the modes layer can
   choose to ask the user before redesigning the plan.

Both primitives have an LLM path (gpt-4o-mini, JSON output) and a
deterministic stub fallback (NFR-14). The stub is also what tests use
for `__force_fail__` / `__force_drift__` markers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import config

logger = logging.getLogger(__name__)


# ── Public types ─────────────────────────────────────────────────


@dataclass
class Verdict:
    verdict: str  # "pass" | "fail"
    reason: str
    confidence: float = 0.0  # 0..1
    provider: str = "stub"


@dataclass
class Alignment:
    drift: float  # 0 == perfectly on goal, 1 == fully off
    should_replan: bool
    reason: str
    provider: str = "stub"


# Drift threshold above which alignment recommends a replan. The runtime
# can override per-call but for v1.1 a single global value is enough.
DRIFT_THRESHOLD = 0.6


# ── critic ───────────────────────────────────────────────────────


def critic(
    *,
    step_description: str,
    expected_result: str,
    actual_result: Any,
    goal: str,
) -> Verdict:
    """FR-21: judge a single step's output. LLM-backed when configured,
    stub fallback otherwise. Stub recognises two test markers:

    - `expected_result == "__force_fail__"` → always fail
    - empty / falsy actual_result on a step that expected non-empty → fail
    """
    if expected_result == "__force_fail__":
        return Verdict(verdict="fail", reason="forced fail (test marker)", provider="stub")

    if config.LLM_API_KEY:
        try:
            return _critic_llm(
                step_description=step_description,
                expected_result=expected_result,
                actual_result=actual_result,
                goal=goal,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("critic LLM failed, falling back to stub: %s", exc)

    return _critic_stub(
        step_description=step_description,
        expected_result=expected_result,
        actual_result=actual_result,
    )


_CRITIC_PROMPT = """\
You are a lenient-but-honest reviewer. Your job: decide whether the
actual result of a plan step plausibly contributes toward the user's
goal — not whether it's perfect or complete.

Return JSON ONLY, schema:
{"verdict": "pass" | "fail", "reason": "...", "confidence": 0.0..1.0}

DEFAULT TO PASS. A messy-but-relevant web_search result with real
hits PASSES. A plausible pdf extraction PASSES. A partially-filled
table PASSES. The synthesis / final step can always polish incomplete
data downstream.

FAIL only when the result is one of:
- empty / null / zero hits
- an error message or exception trace
- completely off-topic (doesn't mention the goal at all)
- explicitly wrong or contradictory to the acceptance_criteria

When in doubt → pass. Verbose but relevant > clean but empty.
"""


def _critic_llm(
    *, step_description: str, expected_result: str,
    actual_result: Any, goal: str,
) -> Verdict:
    from openai import OpenAI

    client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    user = (
        f"Goal: {goal}\n"
        f"Step: {step_description}\n"
        f"Expected: {expected_result}\n"
        f"Actual: {json.dumps(actual_result, ensure_ascii=False, default=str)[:2000]}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _CRITIC_PROMPT},
            {"role": "user", "content": user},
        ],
    )
    data = json.loads(resp.choices[0].message.content or "{}")
    verdict = data.get("verdict", "pass")
    if verdict not in ("pass", "fail"):
        verdict = "pass"
    return Verdict(
        verdict=verdict,
        reason=str(data.get("reason") or ""),
        confidence=float(data.get("confidence") or 0.5),
        provider="openai",
    )


def _critic_stub(
    *, step_description: str, expected_result: str, actual_result: Any,
) -> Verdict:
    """Lightweight heuristic critic for tests / offline runs."""
    if actual_result is None:
        return Verdict(verdict="fail", reason="no result", provider="stub")
    if isinstance(actual_result, dict):
        # Treat empty hits / empty text as a failure when expected says so.
        if "hits" in actual_result and not actual_result["hits"]:
            return Verdict(verdict="fail", reason="empty hits", provider="stub")
        if "text" in actual_result and not actual_result["text"]:
            return Verdict(verdict="fail", reason="empty text", provider="stub")
    if isinstance(actual_result, str) and not actual_result.strip():
        return Verdict(verdict="fail", reason="empty string", provider="stub")
    return Verdict(
        verdict="pass",
        reason=f"{step_description[:40]} produced output",
        confidence=0.6,
        provider="stub",
    )


# ── goal alignment ───────────────────────────────────────────────


def goal_alignment(
    *,
    step_description: str,
    actual_result: Any,
    goal: str,
) -> Alignment:
    """FR-22: how far has the latest step drifted from the original
    goal? Drift in [0, 1]; should_replan = drift > DRIFT_THRESHOLD."""
    if "__force_drift__" in (step_description or ""):
        return Alignment(
            drift=1.0,
            should_replan=True,
            reason="forced drift (test marker)",
            provider="stub",
        )

    if config.LLM_API_KEY:
        try:
            return _alignment_llm(
                step_description=step_description,
                actual_result=actual_result,
                goal=goal,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("alignment LLM failed, falling back to stub: %s", exc)

    return _alignment_stub(
        step_description=step_description,
        actual_result=actual_result,
        goal=goal,
    )


_ALIGNMENT_PROMPT = """\
Score how well the latest step contributes to the original goal.

Return JSON ONLY:
{"drift": 0.0..1.0, "reason": "..."}

drift = 0.0 → perfectly aligned with the goal
drift = 1.0 → completely off-topic, the run is going sideways
"""


def _alignment_llm(
    *, step_description: str, actual_result: Any, goal: str,
) -> Alignment:
    from openai import OpenAI

    client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    user = (
        f"Goal: {goal}\n"
        f"Step: {step_description}\n"
        f"Actual: {json.dumps(actual_result, ensure_ascii=False, default=str)[:2000]}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _ALIGNMENT_PROMPT},
            {"role": "user", "content": user},
        ],
    )
    data = json.loads(resp.choices[0].message.content or "{}")
    drift = float(data.get("drift") or 0.0)
    drift = max(0.0, min(1.0, drift))
    return Alignment(
        drift=drift,
        should_replan=drift > DRIFT_THRESHOLD,
        reason=str(data.get("reason") or ""),
        provider="openai",
    )


def _alignment_stub(
    *, step_description: str, actual_result: Any, goal: str,
) -> Alignment:
    """Cheap token-overlap heuristic — deterministic enough for tests
    but capped so the stub is *friendly* by default.

    Real drift detection needs an LLM (handled in `_alignment_llm`); the
    stub is mainly here so the runtime can boot without a key. To avoid
    spurious replans on legitimate runs, the stub never returns drift
    above `DRIFT_THRESHOLD` unless the caller embeds the explicit
    `__force_drift__` marker — that's already handled upstream in
    `goal_alignment()` before this fallback runs.
    """
    goal_tokens = set(_tokenise(goal))
    if not goal_tokens:
        return Alignment(drift=0.0, should_replan=False, reason="empty goal", provider="stub")

    actual_text = _stringify(actual_result) + " " + (step_description or "")
    actual_tokens = set(_tokenise(actual_text))
    if not actual_tokens:
        return Alignment(drift=0.3, should_replan=False, reason="empty result", provider="stub")

    overlap = len(goal_tokens & actual_tokens) / len(goal_tokens)
    raw_drift = 1.0 - overlap
    # Cap stub drift below the replan threshold so we don't fire false
    # alarms on normal runs. Real alignment lives in the LLM path.
    capped_drift = min(raw_drift, DRIFT_THRESHOLD - 0.1)
    return Alignment(
        drift=capped_drift,
        should_replan=False,
        reason=f"overlap={overlap:.2f} (stub capped)",
        provider="stub",
    )


def _tokenise(text: str) -> list[str]:
    import re
    return [t for t in re.split(r"\W+", (text or "").lower()) if len(t) > 2]


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return str(value)
