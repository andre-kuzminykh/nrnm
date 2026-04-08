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


# ── Reducers for LangGraph TypedDict channels ────────────────────
# Defined at module scope BEFORE `_LGState` so the TypedDict can
# reference them directly. LangGraph inspects `Annotated` metadata
# looking for callables; string placeholders are silently ignored
# and the channel falls back to "single write per step", which then
# crashes on parallel fan-out.


def _dict_merge(left: dict, right: dict) -> dict:
    """Reducer for `results`. Shallow union; right wins on conflicts.
    Parallel branches contributing their own step ids never actually
    conflict because the adapter only emits new keys."""
    out = dict(left or {})
    out.update(right or {})
    return out


def _pick_signal(left: str | None, right: str | None) -> str | None:
    """Reducer for `replan_signal` under parallel writes.

    Semantics: first-set wins. If an earlier parallel branch already
    flagged `critic_failed` or `goal_drift`, later writes shouldn't
    clobber it — the replan decision is sticky. None / empty is
    treated as "no opinion" and never overrides a real signal.
    """
    if left:
        return left
    return right


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
    # FR-26: when the caller knows which user+domain the run belongs
    # to, tool calls dispatch via services.mcp_client with the
    # domain's registered MCP entry. When unset, falls back to the
    # legacy hardcoded tool_layer.call() path (keeps US-3 injection
    # tests working).
    tg_id: int | None = None
    active_domain: str | None = None


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
    Carries the original plan, the backend runner closure, and the
    (tg_id, active_domain) context that step runners capture via
    closure for MCP dispatch (FR-26)."""
    plan: StructuredPlan
    backend: str
    runner: Callable[[GraphState], GraphState]
    tg_id: int | None = None
    active_domain: str | None = None


def compile_plan(
    plan: StructuredPlan,
    *,
    tg_id: int | None = None,
    active_domain: str | None = None,
) -> CompiledPlan:
    """FR-20: turn the StructuredPlan into something `run()` can drive.

    `tg_id` + `active_domain` are optional — when supplied, step
    runners use them to look up tools in the per-domain MCP registry
    via `services.mcp_client`. Without them, tool calls fall back to
    the legacy hardcoded `tool_layer.call` path.
    """
    if _langgraph_available():
        compiled = _compile_langgraph(plan, tg_id, active_domain)
    else:
        compiled = _compile_linear(plan, tg_id, active_domain)
    logger.info(
        "runtime: compiled plan %s (%d steps) -> backend=%s | domain=%s",
        plan.plan_id, len(plan.steps), compiled.backend, active_domain or "-",
    )
    return compiled


def run(
    compiled: CompiledPlan,
    *,
    goal: str | None = None,
    tg_id: int | None = None,
    active_domain: str | None = None,
) -> GraphState:
    """Drive the compiled plan. Returns the final GraphState.

    `tg_id` / `active_domain` override whatever was baked in at
    compile time — useful when a reused compiled plan needs a
    different domain. Absent → fall back to the compile-time values.
    """
    state = GraphState(
        goal=goal or compiled.plan.goal,
        tg_id=tg_id if tg_id is not None else compiled.tg_id,
        active_domain=active_domain if active_domain is not None else compiled.active_domain,
    )
    logger.info(
        "runtime: invoke %s backend=%s | domain=%s | goal=%r",
        compiled.plan.plan_id, compiled.backend,
        state.active_domain or "-", (goal or compiled.plan.goal)[:80],
    )
    final = compiled.runner(state)
    logger.info(
        "runtime: done %s backend=%s | completed=%d/%d | replan=%s | trace=%d",
        compiled.plan.plan_id, compiled.backend,
        len(final.completed_step_ids), len(compiled.plan.steps),
        final.replan_signal, len(final.trace),
    )
    return final


# ── Per-step executor (shared by both backends) ──────────────────


def _make_step_runner(
    step: PlanStep,
    plan: StructuredPlan,
    tg_id: int | None = None,
    active_domain: str | None = None,
) -> Callable[[GraphState], GraphState]:
    """Closure that executes one step + critic + alignment, mutates
    state in-place, and respects the replan_signal short-circuit.

    `tg_id` / `active_domain` baked in here flow to `_execute_step`
    via the GraphState object — if the caller didn't already set
    them on state, we seed from the closure capture. This lets the
    LangGraph backend avoid threading domain context through the
    state schema (which would force us into reducer gymnastics).
    """

    def _runner(state: GraphState) -> GraphState:
        if state.tg_id is None:
            state.tg_id = tg_id
        if state.active_domain is None:
            state.active_domain = active_domain
        # LangGraph defaults missing Annotated[str, ...] channels to ""
        # rather than None. Treat empty string as "no signal" so the
        # short-circuit only fires on a real veto.
        if state.replan_signal:
            return state  # short-circuit: prior step already vetoed

        logger.info(
            "runtime: [%s] start | tool=%s | desc=%r",
            step.id, step.tool or "reasoning", step.description[:80],
        )

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
        logger.info(
            "runtime: [%s] critic(%s)=%s | reason=%r",
            step.id, verdict.provider, verdict.verdict, verdict.reason[:200],
        )
        if verdict.verdict == "fail":
            # Full reason on veto so ops can tell whether the critic
            # was right or too strict.
            logger.warning(
                "runtime: [%s] CRITIC VETO -> replan_signal=critic_failed | full_reason=%r",
                step.id, verdict.reason,
            )
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
        logger.info(
            "runtime: [%s] alignment(%s)=%.2f replan=%s",
            step.id, align.provider, align.drift, align.should_replan,
        )
        if align.should_replan:
            logger.warning(
                "runtime: [%s] DRIFT %.2f>threshold -> replan_signal=goal_drift",
                step.id, align.drift,
            )
            state.replan_signal = "goal_drift"
            return state

        # 4. Step completed cleanly
        state.completed_step_ids.append(step.id)
        logger.info("runtime: [%s] done", step.id)
        return state

    return _runner


def _execute_step(step: PlanStep, state: GraphState) -> Any:
    """Run the actual work of a step.

    - If the step binds a tool and the GraphState carries a
      (tg_id, active_domain) pair, route through the MCP registry:
      look up the entry by name and dispatch via `services.mcp_client`.
      This is the FR-26 path — tool calls are per-domain, transport
      is decided by the MCP URL scheme.
    - If no domain context, fall back to the legacy
      `services.tools.call()` path so existing US-3 tests (which
      don't set up a domain) keep working.
    - If no tool at all, treat the step as a synthesis / reasoning
      marker — record the description as the result. The critic
      judges it on its own merits.
    """
    if step.tool:
        # Resolve any state references in tool_args (e.g. {"query": "$goal"})
        args = {
            k: state.goal if v == "$goal" else v
            for k, v in (step.tool_args or {}).items()
        }
        # Inject a default `query` for search-y tools when missing
        if "query" not in args and ("search" in step.tool.lower()):
            args["query"] = state.goal

        result = _dispatch_tool(step.tool, args, state)

        provider = (result.metadata or {}).get("provider")
        mcp_url = (result.metadata or {}).get("mcp_url")
        state.trace.append(TraceEvent(
            kind="tool_call",
            node_id=step.id,
            payload={
                "tool": step.tool,
                "status": result.status,
                "provider": provider,
                "mcp_url": mcp_url,
            },
        ))
        if result.status != "ok":
            return {"error": result.error or "tool failed", "tool": step.tool}
        return result.output

    # Pure reasoning / synthesis step — call the LLM to combine prior
    # step results into real text. When there's no LLM key, fall back
    # to a deterministic stub so tests stay offline-clean (NFR-14).
    return _synthesize_step(step, state)


def _synthesize_step(step: PlanStep, state: GraphState) -> Any:
    """LLM-backed synthesis for tool-less steps.

    Takes the step description + every prior step's result + the
    original goal, asks GPT-4o-mini to produce a concise answer, and
    returns `{"text": <llm output>, "based_on": [step ids]}`.

    When LLM_API_KEY is empty, returns a deterministic stub carrying
    the same shape so the runtime contract stays stable for tests.
    """
    import config

    prior_ids = list(state.results.keys())

    if not config.LLM_API_KEY:
        return {
            "text": f"(stub synthesis) {step.description}",
            "based_on": prior_ids,
        }

    # Pack prior step outputs into a compact JSON-ish string so the
    # model sees everything that happened before without hitting token
    # limits. Each entry is trimmed to ~1500 chars.
    import json as _json

    def _pack(value: Any) -> str:
        try:
            s = _json.dumps(value, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001
            s = str(value)
        return s if len(s) <= 1500 else s[:1500] + "…"

    prior_block = "\n\n".join(
        f"[{sid}]\n{_pack(state.results.get(sid))}"
        for sid in prior_ids
    )

    system_msg = (
        "You are the final-step synthesiser in a multi-step agent. "
        "Combine the prior step outputs into a clean, useful answer "
        "for the user's goal. Write in the same language as the goal. "
        "Use markdown headers/bullets if helpful. Keep it concrete: "
        "cite filenames or source titles inline when they appear in "
        "the prior outputs. Do NOT add [1], [2] citation numbers. "
        "If prior outputs are thin, say so honestly and summarise "
        "what you do have."
    )
    user_msg = (
        f"Goal: {state.goal}\n\n"
        f"Step description: {step.description}\n"
        f"Expected result: {step.expected_result}\n\n"
        f"Prior step outputs:\n{prior_block if prior_block else '(none)'}"
    )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        text = (resp.choices[0].message.content or "").strip() or "(пусто)"
        logger.info(
            "runtime: [%s] synthesised %d chars from %d prior steps",
            step.id, len(text), len(prior_ids),
        )
        return {"text": text, "based_on": prior_ids}
    except Exception as exc:  # noqa: BLE001
        logger.warning("synthesis LLM failed for %s: %s", step.id, exc)
        return {
            "text": f"(synthesis failed: {exc})\n\nfallback: {step.description}",
            "based_on": prior_ids,
        }


def _dispatch_tool(tool_name: str, args: dict, state: GraphState):
    """Route a tool call: MCP registry if we have a domain context,
    else legacy tool_layer.call. Any failure returns a
    ToolCallResult(status="error") — never raises up to the runner."""
    # FR-26: prefer the per-domain MCP registry when we know which
    # domain the run belongs to.
    if state.tg_id is not None and state.active_domain:
        try:
            from services import mcp_registry
            from services import mcp_client

            entry = mcp_registry.get_mcp(state.tg_id, state.active_domain, tool_name)
            if entry is not None:
                return mcp_client.dispatch(entry, args)
            # Fall through to legacy if the tool isn't registered.
        except Exception as exc:  # noqa: BLE001
            logger.warning("mcp dispatch lookup failed: %s", exc)

    # Legacy fallback: hardcoded tool_layer whitelist. Still enforces
    # FR-13 (DisallowedToolError for anything outside web_search /
    # pdf_parser).
    try:
        return tool_layer.call(tool_name, args)
    except tool_layer.DisallowedToolError as exc:
        return tool_layer.ToolCallResult(
            status="error", error=str(exc),
            metadata={"tool": tool_name},
        )


# ── LangGraph backend ────────────────────────────────────────────


class _LGState(TypedDict, total=False):
    """LangGraph state schema. Every field that can be written by more
    than one parallel branch MUST have a reducer — LangGraph rejects
    concurrent writes to a non-reduced channel with:

        "At key 'X': Can receive only one value per step.
         Use an Annotated key to handle multiple values."

    So `results`, `trace`, `completed_step_ids`, AND `replan_signal`
    all carry reducers. `goal` is read-only after initial state, so
    it doesn't need one.

    NOTE on domain context (FR-26): `tg_id` + `active_domain` are
    NOT part of the LG state schema. They're thread-local to the
    runner and captured via closure in `_make_step_runner`, which
    means LangGraph never sees them as channels and there's no
    risk of concurrent-write conflicts or reducer plumbing.
    """
    goal: str
    results: Annotated[dict, _dict_merge]
    trace: Annotated[list, operator.add]
    replan_signal: Annotated[str, _pick_signal]
    completed_step_ids: Annotated[list, operator.add]


def _compile_langgraph(
    plan: StructuredPlan,
    tg_id: int | None = None,
    active_domain: str | None = None,
) -> CompiledPlan:
    """Build a real `langgraph.StateGraph` from the StructuredPlan.

    Parallel branches: when two nodes share a parent and don't depend on
    each other, LangGraph fans out automatically. We don't need to wire
    `parallel_groups` explicitly — the dependency edges already encode it.
    """
    from langgraph.graph import StateGraph, START, END

    sg = StateGraph(_LGState)

    # Wrap each PlanStep runner so it can convert between LG dict and
    # our dataclass GraphState.
    for step in plan.steps:
        runner = _make_step_runner(step, plan, tg_id, active_domain)
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

    # Bake domain context into the closure so _make_step_runner's
    # _execute_step can reach it via state (we stash it on GraphState
    # right before calling runner(gs) in the adapter).
    baked_tg_id = tg_id
    baked_active_domain = active_domain

    def _runner(state: GraphState) -> GraphState:
        # Propagate compile-time context onto the runtime state so
        # the per-step adapter copies it into the synthetic gs.
        if state.tg_id is None:
            state.tg_id = baked_tg_id
        if state.active_domain is None:
            state.active_domain = baked_active_domain
        initial: dict = {
            "goal": state.goal,
            "results": {},
            "trace": [],
            "completed_step_ids": [],
        }
        try:
            final = compiled.invoke(initial)
        except Exception as exc:  # noqa: BLE001
            logger.warning("langgraph invoke failed, falling back to linear: %s", exc)
            return _compile_linear(plan).runner(state)
        state.results = final.get("results") or {}
        state.trace = final.get("trace") or []
        state.replan_signal = final.get("replan_signal") or None
        state.completed_step_ids = final.get("completed_step_ids") or []
        return state

    return CompiledPlan(
        plan=plan, backend="langgraph", runner=_runner,
        tg_id=tg_id, active_domain=active_domain,
    )


def _wrap_for_langgraph(runner: Callable[[GraphState], GraphState]):
    """Adapter: LangGraph passes a TypedDict and expects a dict update;
    our runner takes/returns a GraphState dataclass. Bridge them.

    Critical: only include fields that *actually changed*. Writing
    `replan_signal: None` on every step caused concurrent-write failures
    under parallel fan-out because LangGraph treats every key in the
    return dict as a channel write — even if the value didn't really
    change. Minimal updates = no phantom conflicts.
    """
    def _adapter(state: dict) -> dict:
        # Domain context (tg_id / active_domain) is captured in the
        # runner closure, NOT threaded through LG state — see
        # `_LGState` docstring for why.
        gs = GraphState(
            goal=state.get("goal", ""),
            results=dict(state.get("results") or {}),
            trace=list(state.get("trace") or []),
            # LangGraph normalises missing Annotated[str, ...] to "",
            # not None — coerce back to None so our short-circuit
            # checks work.
            replan_signal=(state.get("replan_signal") or None),
            completed_step_ids=list(state.get("completed_step_ids") or []),
        )
        prior_trace_len = len(gs.trace)
        prior_completed = set(gs.completed_step_ids)
        runner(gs)

        # Only emit the delta — each field is skipped if it has nothing
        # new to contribute. Reducers then union cleanly across parallel
        # branches.
        update: dict = {}
        new_results = {
            k: v for k, v in gs.results.items()
            if (state.get("results") or {}).get(k) is not v
        }
        if new_results:
            update["results"] = new_results
        new_trace = gs.trace[prior_trace_len:]
        if new_trace:
            update["trace"] = new_trace
        new_completed = [
            sid for sid in gs.completed_step_ids if sid not in prior_completed
        ]
        if new_completed:
            update["completed_step_ids"] = new_completed
        # Only write replan_signal if we actually set one — never emit
        # a spurious None update (see _pick_signal docstring).
        if gs.replan_signal is not None:
            update["replan_signal"] = gs.replan_signal
        return update
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


def _compile_linear(
    plan: StructuredPlan,
    tg_id: int | None = None,
    active_domain: str | None = None,
) -> CompiledPlan:
    """Topologically sort the plan and walk it sequentially."""
    order = _topological_sort(plan)

    def _runner(state: GraphState) -> GraphState:
        for sid in order:
            if state.replan_signal is not None:
                break
            step = next((s for s in plan.steps if s.id == sid), None)
            if step is None:
                continue
            _make_step_runner(step, plan, tg_id, active_domain)(state)
        return state

    return CompiledPlan(
        plan=plan, backend="linear", runner=_runner,
        tg_id=tg_id, active_domain=active_domain,
    )


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
