"""Tests for US-4 — LLM-driven planner + LangGraph runtime + critic loop.

## Трассируемость
Feature: Telegram AI Platform v1
Requirements: FR-18, FR-19, FR-20, FR-21, FR-22, NFR-14

These tests encode the v1.1 advanced-planner contract:

- FR-18: planner returns a `StructuredPlan` with steps + tool bindings
  built from a natural-language goal. With no LLM key the deterministic
  stub kicks in (NFR-14).
- FR-19: plan supports parallel groups, sequential `depends_on`, and
  conditional edges.
- FR-20: executor compiles the StructuredPlan into a LangGraph StateGraph
  and runs it. Falls back to a linear runtime when langgraph is missing.
- FR-21: critic runs after every step, returns pass/fail with reasoning,
  and can veto continuation.
- FR-22: alignment check measures drift from the original goal each step
  and triggers replan when drift exceeds threshold.

LLM and LangGraph are abstracted behind module-level globals so tests
can swap in stubs without touching the network or installing heavy deps.
"""

from __future__ import annotations

import pytest


def _import_advanced():
    """Lazy import — returns (planner, judge, runtime) or None if any
    isn't shipped yet, so tests degrade to skip rather than ImportError."""
    try:
        from services import llm_planner, llm_judge, graph_runtime  # noqa: WPS433
        return llm_planner, llm_judge, graph_runtime
    except Exception:  # noqa: BLE001
        return None


# ─────────────────────────────────────────────────────────────────
# FR-18 / NFR-14 — LLM-backed planner + deterministic stub fallback
# ─────────────────────────────────────────────────────────────────

def test_fr_18_planner_emits_structured_plan_with_tool_bindings():
    mods = _import_advanced()
    if mods is None:
        pytest.skip("TODO FR-18: services.llm_planner not implemented")
    planner, _judge, _runtime = mods

    plan = planner.build_plan(
        goal="Найти 5 свежих новостей про Claude Opus и собрать сводку",
        available_tools=("web_search", "pdf_parser"),
    )

    # Structural contract
    assert plan.plan_id
    assert plan.goal.startswith("Найти")
    assert plan.steps, "planner must produce at least one step"
    for step in plan.steps:
        assert step.id
        assert step.description
        # tool is None or one of the allowed tools — never a hallucinated name
        assert step.tool in (None, "web_search", "pdf_parser")
        assert step.expected_result, "FR-15: each step has expected_result"
        assert isinstance(step.depends_on, list)


def test_nfr_14_planner_falls_back_to_stub_without_llm(monkeypatch):
    """NFR-14: when no OpenAI key is configured, planner returns a
    deterministic stub plan rather than crashing."""
    mods = _import_advanced()
    if mods is None:
        pytest.skip("TODO NFR-14: services.llm_planner not implemented")
    planner, _judge, _runtime = mods

    import config
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    plan = planner.build_plan(goal="anything", available_tools=("web_search",))
    assert plan.steps
    assert plan.metadata.get("provider") == "stub"


# ─────────────────────────────────────────────────────────────────
# FR-19 — plan supports parallel + sequential + conditional edges
# ─────────────────────────────────────────────────────────────────

def test_fr_19_plan_supports_parallel_and_sequential():
    mods = _import_advanced()
    if mods is None:
        pytest.skip("TODO FR-19: structured plan not implemented")
    planner, _judge, _runtime = mods

    plan = planner.build_plan(
        goal="Сравни две статьи через web search и собери таблицу",
        available_tools=("web_search",),
    )
    # Sequential dependency: each step (except entry) has at least one parent
    has_seq = any(step.depends_on for step in plan.steps)
    has_parallel = bool(plan.parallel_groups)
    assert has_seq or has_parallel, "must support at least one shape"


def test_fr_19_plan_can_carry_conditional_edges():
    mods = _import_advanced()
    if mods is None:
        pytest.skip("TODO FR-19: conditional edges not implemented")
    planner, _judge, _runtime = mods
    # The stub plan exposes the conditional_edges list (possibly empty);
    # the public contract is "the field exists and is iterable".
    plan = planner.build_plan(goal="any goal", available_tools=())
    assert hasattr(plan, "conditional_edges")
    assert isinstance(plan.conditional_edges, list)


# ─────────────────────────────────────────────────────────────────
# FR-20 — Executor compiles plan into LangGraph and runs it
# ─────────────────────────────────────────────────────────────────

def test_fr_20_executor_compiles_plan_to_runtime():
    mods = _import_advanced()
    if mods is None:
        pytest.skip("TODO FR-20: graph_runtime not implemented")
    planner, _judge, runtime = mods

    plan = planner.build_plan(goal="простая задача", available_tools=("web_search",))
    compiled = runtime.compile_plan(plan)
    assert compiled is not None
    # Whichever backend was chosen, it must announce itself.
    backend = runtime.runtime_backend()
    assert backend in ("langgraph", "linear")


def test_fr_20_compiled_graph_is_invokable():
    mods = _import_advanced()
    if mods is None:
        pytest.skip("TODO FR-20: graph runtime not implemented")
    planner, _judge, runtime = mods

    plan = planner.build_plan(goal="invoke me", available_tools=("web_search",))
    compiled = runtime.compile_plan(plan)
    final_state = runtime.run(compiled, goal=plan.goal)
    assert final_state.results, "runtime must produce per-step results"
    assert final_state.replan_signal in (None, "critic_failed", "goal_drift")


# ─────────────────────────────────────────────────────────────────
# FR-21 — per-step critic
# ─────────────────────────────────────────────────────────────────

def test_fr_21_critic_returns_verdict_with_reason():
    mods = _import_advanced()
    if mods is None:
        pytest.skip("TODO FR-21: llm_judge.critic not implemented")
    _planner, judge, _runtime = mods

    verdict = judge.critic(
        step_description="search for claude opus",
        expected_result="list of recent links",
        actual_result={"hits": [{"title": "Claude Opus 4.6 announced", "url": "x"}]},
        goal="research claude opus",
    )
    assert verdict.verdict in ("pass", "fail")
    assert verdict.reason


def test_fr_21_critic_can_veto_continuation():
    mods = _import_advanced()
    if mods is None:
        pytest.skip("TODO FR-21: critic veto path not implemented")
    _planner, judge, runtime = mods

    # Stub mode: empty result -> critic fails -> runtime sets replan_signal.
    plan = _planner_stub_with_failing_step(mods)
    compiled = runtime.compile_plan(plan)
    state = runtime.run(compiled, goal=plan.goal)
    # Critic should have flagged it
    failures = [t for t in state.trace if t.kind == "critic" and t.payload.get("verdict") == "fail"]
    assert failures, "critic should record at least one fail trace"


def _planner_stub_with_failing_step(mods):
    planner, _j, _r = mods
    # Cheat: build a normal plan then mark a step's expected_result as
    # something the deterministic stub critic will reject.
    plan = planner.build_plan(goal="test critic veto", available_tools=("web_search",))
    if plan.steps:
        plan.steps[0].expected_result = "__force_fail__"
    return plan


# ─────────────────────────────────────────────────────────────────
# FR-22 — goal alignment check + drift trigger
# ─────────────────────────────────────────────────────────────────

def test_fr_22_alignment_returns_drift_score():
    mods = _import_advanced()
    if mods is None:
        pytest.skip("TODO FR-22: alignment check not implemented")
    _planner, judge, _runtime = mods

    score = judge.goal_alignment(
        step_description="searched the web",
        actual_result={"hits": []},
        goal="research claude opus",
    )
    assert hasattr(score, "drift")
    assert 0.0 <= score.drift <= 1.0
    assert hasattr(score, "should_replan")
    assert isinstance(score.should_replan, bool)


def test_fr_22_high_drift_triggers_replan_signal():
    mods = _import_advanced()
    if mods is None:
        pytest.skip("TODO FR-22: alignment-driven replan not implemented")
    planner, _judge, runtime = mods

    plan = planner.build_plan(goal="aligned goal", available_tools=("web_search",))
    if plan.steps:
        plan.steps[0].description = "__force_drift__"
    compiled = runtime.compile_plan(plan)
    state = runtime.run(compiled, goal=plan.goal)
    drift_traces = [
        t for t in state.trace
        if t.kind == "alignment" and t.payload.get("should_replan") is True
    ]
    assert drift_traces, "alignment must record a drift trace"
    assert state.replan_signal == "goal_drift"


# ─────────────────────────────────────────────────────────────────
# NFR-14 — graph_runtime fallback works without langgraph installed
# ─────────────────────────────────────────────────────────────────

def test_nfr_14_runtime_runs_even_without_langgraph(monkeypatch):
    mods = _import_advanced()
    if mods is None:
        pytest.skip("TODO NFR-14: runtime not implemented")
    planner, _judge, runtime = mods
    monkeypatch.setattr(runtime, "_langgraph_available", lambda: False)

    plan = planner.build_plan(goal="no langgraph please", available_tools=("web_search",))
    compiled = runtime.compile_plan(plan)
    state = runtime.run(compiled, goal=plan.goal)
    assert runtime.runtime_backend() == "linear"
    assert state.results
