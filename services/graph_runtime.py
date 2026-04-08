"""LangGraph runtime — compile a StructuredPlan into an executable graph.

## Трассируемость
Feature: Telegram AI Platform v1.1
Requirements: FR-19, FR-20, FR-21, FR-22, NFR-10, NFR-12, NFR-14

Two backends, same public API:

1. **LangGraph backend** — when `langgraph` is importable, we build a
   real `StateGraph` with one node per plan step. Sequential edges
   come from `step.depends_on`. Nodes that share a parent and don't
   reference each other run in parallel automatically (LangGraph's
   default fan-out behaviour). Conditional edges from
   `plan.conditional_edges` are wired via `add_conditional_edges`.

2. **Linear backend** — graceful fallback when langgraph is missing.
   Topologically sorts the steps by `depends_on` and walks them.
   Loses parallelism but keeps every other contract intact (critic,
   alignment, traces, replan signals). NFR-14.

After every step the runtime invokes `llm_judge.critic()` and
`llm_judge.goal_alignment()`. A critic-fail or alignment-drift sets
`state.replan_signal` and stops further nodes — the modes layer reads
the signal and decides whether to redesign.

The runtime is intentionally synchronous: the bot's executor loop is
sync, and LangGraph's `compiled.invoke()` works fine in either mode.
"""

from __future__ import annotations

import logging
import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, TypedDict

from services import tools as tool_layer
from services import llm_judge
from services.llm_planner import StructuredPlan, PlanStep, ConditionalEdge

logger = logging.getLogger(__name__)


# ── Public state ─────────────────────────────────────────────────


@dataclass
class TraceEvent:
    kind: str  # "step" | "tool_call" | "critic" | "alignment" | "replan"
    node_id: str | None
    payload: dict = field(default_factory=dict)


@dataclass
class GraphState:
    """The blob threaded through every node. Pure data — no methods —
    so it round-trips through LangGraph's reducer cleanly."""
    goal: str
    results: dict[str, Any] = field(default_factory=dict)
    trace: list[TraceEvent] = field(default_factory=list)
    replan_signal: str | None = None  # None | "critic_failed" | "goal_drift"
    completed_step_ids: list[str] = field(default_factory=list)


# ── Backend detection ────────────────────────────────────────────


def _langgraph_available() -> bool:
    """Hookable from tests via monkeypatch — see test_nfr_14."""
    try:
        import langgraph  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def runtime_backend() -> str:
    return "langgraph" if _langgraph_available() else "linear"


# ── Compile + run ────────────────────────────────────────────────


@dataclass
class CompiledPlan:
    """Opaque handle returned by `compile_plan` and consumed by `run`.
    Carries the original plan + the chosen backend's runner closure."""
    plan: StructuredPlan
    backend: str
    runner: Callable[[GraphState], GraphState]


def compile_plan(plan: StructuredPlan) -> CompiledPlan:
    """FR-20: turn the StructuredPlan into something `run()` can drive."""
    if _langgraph_available():
        return _compile_langgraph(plan)
    return _compile_linear(plan)


def run(compiled: CompiledPlan, *, goal: str | None = None) -> GraphState:
    """Drive the compiled plan. Returns the final GraphState."""
    state = GraphState(goal=goal or compiled.plan.goal)
    return compiled.runner(state)


# ── Per-step executor (shared by both backends) ──────────────────


def _make_step_runner(step: PlanStep, plan: StructuredPlan) -> Callable[[GraphState], GraphState]:
    """Closure that executes one step + critic + alignment, mutates
    state in-place, and respects the replan_signal short-circuit."""

    def _runner(state: GraphState) -> GraphState:
        if state.replan_signal is not None:
            return state  # short-circuit: prior step already vetoed

        # 1. Execute the step
        result = _execute_step(step, state)
        state.results[step.id] = result
        state.trace.append(TraceEvent(
            kind="step",
            node_id=step.id,
            payload={"description": step.description, "tool": step.tool},
        ))

        # 2. Critic verdict
        verdict = llm_judge.critic(
            step_description=step.description,
            expected_result=step.expected_result,
            actual_result=result,
            goal=state.goal,
        )
        state.trace.append(TraceEvent(
            kind="critic",
            node_id=step.id,
            payload={
                "verdict": verdict.verdict,
                "reason": verdict.reason,
                "provider": verdict.provider,
            },
        ))
        if verdict.verdict == "fail":
            state.replan_signal = "critic_failed"
            return state

        # 3. Goal alignment
        align = llm_judge.goal_alignment(
            step_description=step.description,
            actual_result=result,
            goal=state.goal,
        )
        state.trace.append(TraceEvent(
            kind="alignment",
            node_id=step.id,
            payload={
                "drift": align.drift,
                "should_replan": align.should_replan,
                "reason": align.reason,
                "provider": align.provider,
            },
        ))
        if align.should_replan:
            state.replan_signal = "goal_drift"
            return state

        # 4. Step completed cleanly
        state.completed_step_ids.append(step.id)
        return state

    return _runner


def _execute_step(step: PlanStep, state: GraphState) -> Any:
    """Run the actual work of a step.

    - If the step binds a tool, route through `services.tools.call()`
      so FR-13 whitelist is enforced and tool failures degrade into a
      proper Result rather than an exception.
    - If no tool, treat the step as a synthesis / reasoning marker —
      we just record the description as the result. The critic will
      judge it on its own merits.
    """
    if step.tool:
        # Resolve any state references in tool_args (e.g. {"query": "$goal"})
        args = {
            k: state.goal if v == "$goal" else v
            for k, v in (step.tool_args or {}).items()
        }
        # Inject a default `query` for web_search when missing
        if step.tool == "web_search" and "query" not in args:
            args["query"] = state.goal
        try:
            result = tool_layer.call(step.tool, args)
        except tool_layer.DisallowedToolError as exc:
            return {"error": str(exc)}
        state.trace.append(TraceEvent(
            kind="tool_call",
            node_id=step.id,
            payload={
                "tool": step.tool,
                "status": result.status,
                "provider": result.metadata.get("provider"),
            },
        ))
        if result.status != "ok":
            return {"error": result.error or "tool failed", "tool": step.tool}
        return result.output

    # Pure reasoning step — no tool. Use the description as a placeholder
    # synthesis. A real LLM call would go here.
    return {"synthesis": step.description, "based_on": list(state.results.keys())}


# ── LangGraph backend ────────────────────────────────────────────


class _LGState(TypedDict, total=False):
    """LangGraph state schema. Lives at module level so `get_type_hints`
    can resolve `Annotated` / reducer references during compilation."""
    goal: str
    results: Annotated[dict, "_dict_merge"]
    trace: Annotated[list, operator.add]
    replan_signal: str
    completed_step_ids: Annotated[list, operator.add]


def _compile_langgraph(plan: StructuredPlan) -> CompiledPlan:
    """Build a real `langgraph.StateGraph` from the StructuredPlan.

    Parallel branches: when two nodes share a parent and don't depend on
    each other, LangGraph fans out automatically. We don't need to wire
    `parallel_groups` explicitly — the dependency edges already encode it.
    """
    from langgraph.graph import StateGraph, START, END

    # Replace the placeholder string reducer with the real callable
    # right before instantiation (LangGraph evaluates it after
    # get_type_hints during _add_schema).
    _LGState.__annotations__["results"] = Annotated[dict, _dict_merge]

    sg = StateGraph(_LGState)

    # Wrap each PlanStep runner so it can convert between LG dict and
    # our dataclass GraphState.
    for step in plan.steps:
        runner = _make_step_runner(step, plan)
        sg.add_node(step.id, _wrap_for_langgraph(runner))

    # Sequential edges from depends_on
    entries = [s for s in plan.steps if not s.depends_on]
    for entry in entries:
        sg.add_edge(START, entry.id)
    for step in plan.steps:
        for parent in step.depends_on:
            sg.add_edge(parent, step.id)

    # Terminal edges: any node with no children → END
    children: dict[str, list[str]] = {s.id: [] for s in plan.steps}
    for s in plan.steps:
        for parent in s.depends_on:
            children.setdefault(parent, []).append(s.id)
    leaves = [sid for sid, kids in children.items() if not kids]
    for leaf in leaves:
        sg.add_edge(leaf, END)

    # Conditional edges (override sequential where defined)
    for cond in plan.conditional_edges:
        sg.add_conditional_edges(
            cond.from_step,
            _make_conditional_router(cond),
            {"true": cond.true_target, "false": cond.false_target},
        )

    compiled = sg.compile()

    def _runner(state: GraphState) -> GraphState:
        initial = {
            "goal": state.goal,
            "results": {},
            "trace": [],
            "replan_signal": None,
            "completed_step_ids": [],
        }
        try:
            final = compiled.invoke(initial)
        except Exception as exc:  # noqa: BLE001
            logger.warning("langgraph invoke failed, falling back to linear: %s", exc)
            return _compile_linear(plan).runner(state)
        state.results = final.get("results") or {}
        state.trace = final.get("trace") or []
        state.replan_signal = final.get("replan_signal")
        state.completed_step_ids = final.get("completed_step_ids") or []
        return state

    return CompiledPlan(plan=plan, backend="langgraph", runner=_runner)


def _dict_merge(left: dict, right: dict) -> dict:
    """Reducer for the `results` field — last writer wins on conflicts,
    union otherwise. Lets parallel branches contribute their step
    results without clobbering each other."""
    out = dict(left or {})
    out.update(right or {})
    return out


def _wrap_for_langgraph(runner: Callable[[GraphState], GraphState]):
    """Adapter: LangGraph passes a TypedDict and expects a dict update;
    our runner takes/returns a GraphState dataclass. Bridge them."""
    def _adapter(state: dict) -> dict:
        gs = GraphState(
            goal=state.get("goal", ""),
            results=dict(state.get("results") or {}),
            trace=list(state.get("trace") or []),
            replan_signal=state.get("replan_signal"),
            completed_step_ids=list(state.get("completed_step_ids") or []),
        )
        prior_trace_len = len(gs.trace)
        prior_completed = set(gs.completed_step_ids)
        runner(gs)
        # Return only the *delta* so reducers union correctly.
        return {
            "results": gs.results,
            "trace": gs.trace[prior_trace_len:],
            "replan_signal": gs.replan_signal,
            "completed_step_ids": [
                sid for sid in gs.completed_step_ids if sid not in prior_completed
            ],
        }
    return _adapter


def _make_conditional_router(cond: ConditionalEdge):
    """LangGraph router function — inspects state.results[from_step] and
    returns "true" / "false". Stub implementation: a step is "true" if
    its result is non-empty / non-error."""
    def _route(state: dict) -> str:
        result = (state.get("results") or {}).get(cond.from_step)
        if isinstance(result, dict) and result.get("error"):
            return "false"
        if not result:
            return "false"
        return "true"
    return _route


# ── Linear backend (NFR-14 fallback) ─────────────────────────────


def _compile_linear(plan: StructuredPlan) -> CompiledPlan:
    """Topologically sort the plan and walk it sequentially."""
    order = _topological_sort(plan)

    def _runner(state: GraphState) -> GraphState:
        for sid in order:
            if state.replan_signal is not None:
                break
            step = next((s for s in plan.steps if s.id == sid), None)
            if step is None:
                continue
            _make_step_runner(step, plan)(state)
        return state

    return CompiledPlan(plan=plan, backend="linear", runner=_runner)


def _topological_sort(plan: StructuredPlan) -> list[str]:
    """Kahn's algorithm. Falls back to insertion order on cycles
    (which shouldn't happen — planner contract says it's a DAG)."""
    indeg: dict[str, int] = {s.id: 0 for s in plan.steps}
    edges: dict[str, list[str]] = {s.id: [] for s in plan.steps}
    for s in plan.steps:
        for parent in s.depends_on:
            if parent in indeg:
                indeg[s.id] += 1
                edges[parent].append(s.id)

    queue = [sid for sid, d in indeg.items() if d == 0]
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in edges[node]:
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
    if len(order) != len(plan.steps):
        # Cycle — fall back to original order
        return [s.id for s in plan.steps]
    return order
