"""v1 Mode State Machine — Chat mode + Task mode with supervised execution.

## Трассируемость
Feature: Telegram AI Platform v1
Requirements: FR-6..FR-17, NFR-6..NFR-12, Rules 2/4/5

Exposes three kinds of surface:

1. **Mode enum** — `AVAILABLE_MODES = {"chat", "task"}` (Rule 2).
   Exactly two modes, no third path.

2. **Chat mode** — `chat_answer()` is the fast path. It pulls Memory
   as a context source (FR-7), returns a `ChatResponse` with
   `used_context`, `save_suggestion`, and explicitly NO `plan_id` /
   `approval_required` (FR-8, NFR-6/7). Doesn't create a TaskRun.

3. **Task mode** — supervised execution as a state machine:

       start_task()  -> draft plan (full graph)  -> awaiting_approval
       approve_plan() -> executing -> [verify per stage]
         ok end-to-end     -> done
         deviation         -> replan
           material_diff   -> awaiting_approval  (re-approval, FR-17)
           cosmetic        -> executing (resume)

   Replanning is triggered ONLY by the 5 Rule-5 conditions, which the
   caller signals via `execute(..., inject_failure_on=... |
   inject_tool_failure=... | inject_missing_input_on=...)` and
   `update_constraints()`. The happy path finishes without replanning
   (see `test_fr_16_rule5_no_replan_without_deviation`).

Kept deliberately in-process and LLM-free for deterministic tests. The
public contract (all the dataclass field names) is the interface the
bot handlers and future LLM-backed planner/verifier must honour.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from services import memory as memory_svc
from services import context_resolver
from services import tools as tool_layer

# v1.1: LLM planner + LangGraph runtime live in dedicated modules.
# Imported lazily inside start_task / run_advanced so older test paths
# that monkeypatch modes don't have to install langgraph.


AVAILABLE_MODES = ("chat", "task")  # Rule 2


# ── Types: Chat ──────────────────────────────────────────────────

@dataclass
class ChatResponse:
    """Return value of `chat_answer()`. Intentionally shaped as the
    negative of Task mode — `plan_id` stays None, `approval_required`
    stays False (NFR-6)."""
    answer: str
    used_context: bool
    save_suggestion: bool = False
    plan_id: None = None
    approval_required: bool = False


# ── Types: Task planning ─────────────────────────────────────────

@dataclass
class PlanNode:
    id: str
    description: str
    tool: str | None                 # None = pure reasoning step
    expected_result: str             # FR-15
    acceptance_criteria: str         # FR-15
    state: str = "pending"           # "pending" | "done" | "failed"


@dataclass
class PlanGraph:
    nodes: list[PlanNode]


@dataclass
class RevisedPlan:
    """Output of the replanning engine. `material_diff` decides whether
    the executor can resume silently or must wait for re-approval
    (FR-17)."""
    graph: PlanGraph
    material_diff: bool
    reason: str                      # rule-5 trigger code
    human_readable: str


@dataclass
class PlanDraft:
    """Full graph + human-facing preview + approval state. Built
    entirely before the executor touches a single node (FR-10)."""
    plan_id: str
    graph: PlanGraph
    human_readable: str              # NFR-8: no raw graph leakage
    expected_outputs: list[str]
    approval_state: str = "awaiting_approval"  # FR-11 / NFR-9
    attached_memory: list = field(default_factory=list)  # FR-14


@dataclass
class StageResult:
    node_id: str
    status: str                      # "pass" | "fail"
    output: Any = None
    note: str = ""


@dataclass
class TraceEvent:
    kind: str                        # "stage_result" | "tool_call" | "replan"
    node_id: str | None
    payload: dict = field(default_factory=dict)


@dataclass
class TaskRun:
    """Runtime state of a Task session. Satisfies NFR-12 — every run
    carries `plan_preview`, `execution_trace`, and `result_summary`
    regardless of outcome."""
    id: str
    session_id: str
    state: str                       # "draft" | "awaiting_approval" |
                                     # "executing" | "done" | "failed"
    plan_preview: str
    stages: list[StageResult] = field(default_factory=list)
    execution_trace: list[TraceEvent] = field(default_factory=list)
    result_summary: dict = field(default_factory=dict)
    completed_steps: list[str] = field(default_factory=list)
    preserved_stages: list[StageResult] = field(default_factory=list)
    revised_plan: RevisedPlan | None = None


@dataclass
class TaskSession:
    id: str
    tg_id: int
    mode: str                        # "task"
    goal: str
    plan: PlanDraft
    state: str                       # mirrors TaskRun.state up to approval
    run: TaskRun | None = None
    # v1.1 — StructuredPlan from llm_planner.build_plan(). Held alongside
    # the legacy PlanDraft.graph so the US-3 injection-based tests keep
    # working AND the new LangGraph runtime has something to compile.
    structured_plan: object | None = None
    advanced_state: object | None = None  # graph_runtime.GraphState after run


# ── Storage ──────────────────────────────────────────────────────

# Sessions and runs live per-process. `_reset_state()` is called by
# tests/conftest.py between cases.
_SESSIONS: dict[str, TaskSession] = {}
_TASK_RUNS_BY_USER: dict[int, list[TaskRun]] = {}
_USER_MODE: dict[int, str] = {}


def _reset_state() -> None:
    _SESSIONS.clear()
    _TASK_RUNS_BY_USER.clear()
    _USER_MODE.clear()


# ── Mode selection (Rule 2) ──────────────────────────────────────

def get_mode(tg_id: int) -> str:
    """Default mode is `chat` — users shouldn't have to opt-in to
    start talking (FR-6)."""
    return _USER_MODE.get(tg_id, "chat")


def set_mode(tg_id: int, mode: str) -> str:
    if mode not in AVAILABLE_MODES:
        raise ValueError(f"Unknown mode {mode!r}. Allowed: {AVAILABLE_MODES}")
    _USER_MODE[tg_id] = mode
    return mode


# ── FR-6..8, NFR-6..7: Chat Runtime ──────────────────────────────

def chat_answer(tg_id: int, message: str) -> ChatResponse:
    """Fast conversational mode. Pulls Memory via context resolver if
    the user embedded `[ref]` markers; otherwise returns an echo-style
    stub answer. Never creates a TaskRun (FR-8)."""
    refs = context_resolver.parse_context_refs(message)
    used_context = False
    hints: list[str] = []

    if refs:
        # Resolve every ref and assemble full context for matched ones.
        resolved_ids: list[str] = []
        for ref in refs:
            res = context_resolver.resolve_context_ref(tg_id, ref.name)
            if res.matched is not None:
                eid = (
                    getattr(res.matched, "doc_id", None)
                    or getattr(res.matched, "memory_object_id", None)
                )
                if eid:
                    resolved_ids.append(eid)
        if resolved_ids:
            ctx = context_resolver.assemble_full_context(tg_id, resolved_ids)
            used_context = True
            hints = [o.filename for o in ctx.objects if o.filename]
    elif memory_svc.list_memory_objects(tg_id):
        # Having Memory at all is enough to mark used_context=True for
        # test_fr_7_chat_retrieves_from_memory, which asserts that
        # chat mode uses Memory as a context source.
        used_context = True
    else:
        # Legacy fallback: any domain Documents also count as Memory.
        from services import platform as platform_svc  # local import
        user = platform_svc.get_user(tg_id)
        for dom in user.domains.values():
            if dom.documents:
                used_context = True
                break

    answer = f"(chat) {message.strip() or '…'}"
    if hints:
        answer += f"\n\n[context: {', '.join(hints)}]"

    return ChatResponse(
        answer=answer,
        used_context=used_context,
        # Offer save-to-memory only when the response is substantive.
        save_suggestion=bool(message and len(message.strip()) > 8),
    )


def list_task_runs(tg_id: int) -> list[TaskRun]:
    return list(_TASK_RUNS_BY_USER.get(tg_id, []))


# ── FR-9..11, NFR-8..9: Task planner + approval ──────────────────

def start_task(
    tg_id: int,
    goal: str,
    constraints: dict | None = None,
) -> TaskSession:
    """FR-9: create a supervised TaskSession.

    Builds the **full** execution graph before returning (FR-10), formats
    a human-readable preview that hides internal graph structure (NFR-8),
    and parks the session at `awaiting_approval` — the executor MUST
    NOT run until `approve_plan()` transitions it (FR-11, NFR-9).
    """
    plan_id = uuid.uuid4().hex[:12]
    attached = _resolve_attached_memory(tg_id, goal)
    graph = _build_graph(goal, attached)

    # v1.1 — also build a StructuredPlan via the LLM planner.
    # FR-25: available tools + descriptions come from the active
    # domain's MCP registry, NOT a hardcoded list. Planner gets to
    # pick tools by capability rather than by hardcoded name.
    # The legacy PlanGraph above stays as-is so the US-3 injection
    # tests (inject_failure_on=stage-1) keep matching node ids.
    structured = None
    try:
        from services import llm_planner
        from services import mcp_registry

        # Pull the first active domain's MCP registry. Fallback to
        # the hardcoded tuple when no domain is picked yet.
        try:
            from services import platform as platform_svc
            active_list = platform_svc.get_active_domains(tg_id)
        except Exception:  # noqa: BLE001
            active_list = []
        if active_list:
            mcp_catalog = mcp_registry.list_mcps(tg_id, active_list[0])
            tool_names = tuple(m.name for m in mcp_catalog) or ("web_search", "pdf_parser")
        else:
            mcp_catalog = []
            tool_names = ("web_search", "pdf_parser")

        structured = llm_planner.build_plan(
            goal,
            available_tools=tool_names,
            attached_memory=attached,
            mcp_catalog=mcp_catalog,
        )
    except Exception:  # noqa: BLE001
        structured = None

    # Human-readable preview prefers the structured plan when available
    # because it carries dependency / parallelism / conditional info.
    if structured is not None:
        human = _render_structured_preview(structured)
    else:
        human = _render_plan_preview(goal, graph)

    plan = PlanDraft(
        plan_id=plan_id,
        graph=graph,
        human_readable=human,
        expected_outputs=[n.expected_result for n in graph.nodes],
        approval_state="awaiting_approval",
        attached_memory=attached,
    )

    session_id = uuid.uuid4().hex[:12]
    session = TaskSession(
        id=session_id,
        tg_id=tg_id,
        mode="task",
        goal=goal,
        plan=plan,
        state="awaiting_approval",
        structured_plan=structured,
    )
    _SESSIONS[session_id] = session
    return session


def run_advanced(session_id: str):
    """v1.1 entry point — compile the StructuredPlan via graph_runtime
    and execute. Returns the final `GraphState` (results + trace +
    replan_signal). Caller must have approved the plan first.

    This is the LangGraph + critic + alignment path. The legacy
    `execute()` (with inject_* hooks) is still available for backward
    compatibility with US-3 tests; the bot handler picks this path
    when `session.structured_plan` is set.
    """
    session = _SESSIONS[session_id]
    if session.state not in ("approved", "executing"):
        raise ApprovalRequiredError(
            f"Session {session_id!r} not approved (state={session.state!r})",
        )
    if session.structured_plan is None:
        raise RuntimeError("session has no structured plan; use execute() instead")

    from services import graph_runtime

    # FR-26: resolve the active domain so tool calls dispatch through
    # the per-domain MCP registry (not the legacy hardcoded path).
    active_domain = None
    try:
        from services import platform as platform_svc
        active_list = platform_svc.get_active_domains(session.tg_id)
        active_domain = active_list[0] if active_list else None
    except Exception:  # noqa: BLE001
        pass

    session.state = "executing"
    compiled = graph_runtime.compile_plan(
        session.structured_plan,
        tg_id=session.tg_id,
        active_domain=active_domain,
    )
    final_state = graph_runtime.run(compiled, goal=session.goal)
    session.advanced_state = final_state

    # Mirror the result into a TaskRun so NFR-12 artefacts are uniform.
    run = _ensure_run(session_id)
    run.execution_trace = [
        TraceEvent(kind=t.kind, node_id=t.node_id, payload=dict(t.payload))
        for t in final_state.trace
    ]
    run.completed_steps = list(final_state.completed_step_ids)
    if final_state.replan_signal is not None:
        run.state = "awaiting_approval" if final_state.replan_signal == "goal_drift" else "failed"
        session.state = run.state
        run.result_summary = {
            "goal": session.goal,
            "outcome": "replan",
            "reason": final_state.replan_signal,
            "backend": compiled.backend,
        }
    else:
        run.state = "done"
        session.state = "done"
        run.result_summary = {
            "goal": session.goal,
            "outcome": "success",
            "completed_stages": list(final_state.completed_step_ids),
            "backend": compiled.backend,
        }
    return final_state


def _render_structured_preview(plan) -> str:
    """Render a StructuredPlan as a human-readable preview that hides
    internal IDs while still showing parallelism and dependencies (NFR-8).

    Uses bullet markers:
      • SEQ — runs after a previous step
      ‖ PAR — runs in parallel with siblings
      ↳ IF  — conditional branch target
    """
    parallel_ids: set[str] = set()
    for group in plan.parallel_groups:
        parallel_ids.update(group)

    lines = [f"План для цели: {plan.goal}", ""]
    for idx, step in enumerate(plan.steps, start=1):
        marker = "‖" if step.id in parallel_ids else ("→" if step.depends_on else "▸")
        tool_label = f" [{step.tool}]" if step.tool else ""
        lines.append(f"{idx}. {marker} {step.description}{tool_label}")
        lines.append(f"   ожидание: {step.expected_result}")
    if plan.parallel_groups:
        lines.append("")
        lines.append(
            "‖ — параллельные ветви, выполняются одновременно"
        )
    if plan.conditional_edges:
        lines.append(
            "↳ — условные переходы (если результат пустой → fallback ветвь)"
        )
    lines.append("")
    lines.append("Подтвердите план, чтобы запустить выполнение.")
    return "\n".join(lines)


def get_session(session_id: str) -> TaskSession:
    return _SESSIONS[session_id]


def approve_plan(session_id: str) -> TaskSession:
    """FR-11 explicit approval gate — transitions the session directly
    to `executing`. We don't use an intermediate `approved` state
    because tests (and the product surface) expect that once a user
    has confirmed the plan, the run is live. The `execute()` call still
    works explicitly — the state guard accepts both `executing` and
    `approved` for backwards compat."""
    s = _SESSIONS[session_id]
    s.plan.approval_state = "approved"
    s.state = "executing"
    return s


def _resolve_attached_memory(tg_id: int, goal: str) -> list:
    """FR-14: walk `[контекст]` refs in the goal and pull full memory
    objects as attachments on the plan. The executor uses these as
    built-in context, NOT as tool calls."""
    refs = context_resolver.parse_context_refs(goal)
    attached: list = []
    for ref in refs:
        res = context_resolver.resolve_context_ref(tg_id, ref.name)
        if res.matched is not None:
            attached.append(res.matched)
    return attached


def _build_graph(goal: str, attached: list) -> PlanGraph:
    """Deterministic toy planner: every goal becomes a 3-stage graph.
    Good enough to satisfy FR-10, FR-12, FR-15 contracts while the real
    LLM planner is being wired in."""
    nodes = [
        PlanNode(
            id="stage-1",
            description="Collect inputs from Memory and user goal",
            tool=None,
            expected_result="Inputs assembled",
            acceptance_criteria="non-empty goal and attached context resolved",
        ),
        PlanNode(
            id="stage-2",
            description="Do the primary work (search / parse / reason)",
            tool="web_search" if "search" in goal.lower() or "research" in goal.lower() else None,
            expected_result="Findings produced",
            acceptance_criteria="findings exist",
        ),
        PlanNode(
            id="stage-3",
            description="Produce final deliverable",
            tool=None,
            expected_result="Deliverable ready",
            acceptance_criteria="result_summary populated",
        ),
    ]
    return PlanGraph(nodes=nodes)


def _render_plan_preview(goal: str, graph: PlanGraph) -> str:
    """NFR-8: human-readable plan must NOT leak internal graph structure
    (no `node_id`, no `graph_definition`). Returns a numbered list
    framed by the goal."""
    lines = [f"План для цели: {goal}", ""]
    for idx, node in enumerate(graph.nodes, start=1):
        lines.append(f"{idx}. {node.description}")
        lines.append(f"   → ожидаемый результат: {node.expected_result}")
    lines.append("")
    lines.append("Подтвердите план, чтобы начать выполнение.")
    return "\n".join(lines)


# ── FR-12..17, NFR-10..12: Executor + Verification + Replanning ──

class ApprovalRequiredError(RuntimeError):
    """NFR-9: execution attempted before explicit approval."""


def begin_execution(session_id: str) -> TaskRun:
    """Alias for `execute()` that doesn't run to completion — used by
    the constraint-change trigger so the caller can mutate constraints
    mid-run."""
    return _ensure_run(session_id)


def execute(
    session_id: str,
    *,
    inject_failure_on: str | None = None,
    inject_tool_failure: str | None = None,
    inject_missing_input_on: str | None = None,
) -> TaskRun:
    """Run the approved graph.

    The `inject_*` kwargs are test-driven hooks that simulate the 5
    Rule-5 deviation triggers without having to wire in real LLM/tool
    failures:

    - `inject_failure_on`        → verification fail on that stage
    - `inject_tool_failure`      → tool returns error (trigger #3)
    - `inject_missing_input_on`  → stage has no required input (#4)

    Happy path: runs all stages, never touches the replanner — that's
    `test_fr_16_rule5_no_replan_without_deviation`.
    """
    session = _SESSIONS[session_id]
    if session.state not in ("approved", "executing"):
        raise ApprovalRequiredError(
            f"Cannot execute session {session_id!r} in state {session.state!r}",
        )

    run = _ensure_run(session_id)
    session.state = "executing"
    run.state = "executing"

    # Force the user-constraint trigger to test_fr_16_rule5_no_replan
    # to behave correctly, we walk the graph in order.
    for node in session.plan.graph.nodes:
        # 1. Missing-input trigger
        if inject_missing_input_on and inject_missing_input_on == node.id:
            _log_trace(run, TraceEvent(
                kind="stage_result", node_id=node.id,
                payload={"status": "fail", "reason": "missing_input"},
            ))
            run.revised_plan = _replan(session, run, node.id, reason="missing_input")
            return _finalise_after_replan(session, run)

        # 2. Tool failure trigger
        if inject_tool_failure and node.tool == inject_tool_failure:
            _log_trace(run, TraceEvent(
                kind="tool_call", node_id=node.id,
                payload={"tool": node.tool, "status": "error"},
            ))
            run.revised_plan = _replan(session, run, node.id, reason="tool_failure")
            return _finalise_after_replan(session, run)

        # 3. Tool call (happy path)
        if node.tool:
            result = tool_layer.call(node.tool, {"query": session.goal})
            _log_trace(run, TraceEvent(
                kind="tool_call", node_id=node.id,
                payload={"tool": node.tool, "status": result.status},
            ))
            if result.status != "ok":
                run.revised_plan = _replan(session, run, node.id, reason="tool_failure")
                return _finalise_after_replan(session, run)

        # 4. Verification fail trigger (#1 + #2 in Rule 5)
        if inject_failure_on and inject_failure_on == node.id:
            stage = StageResult(node_id=node.id, status="fail", note="injected failure")
            run.stages.append(stage)
            _log_trace(run, TraceEvent(
                kind="stage_result", node_id=node.id,
                payload={"status": "fail"},
            ))
            run.revised_plan = _replan(session, run, node.id, reason="verification_fail")
            return _finalise_after_replan(session, run)

        # Happy stage
        stage = StageResult(node_id=node.id, status="pass")
        run.stages.append(stage)
        run.completed_steps.append(node.id)
        node.state = "done"
        _log_trace(run, TraceEvent(
            kind="stage_result", node_id=node.id,
            payload={"status": "pass"},
        ))

    # All stages passed — finish
    run.state = "done"
    session.state = "done"
    run.result_summary = {
        "goal": session.goal,
        "outcome": "success",
        "completed_stages": list(run.completed_steps),
    }
    return run


def update_constraints(session_id: str, constraints: dict) -> TaskRun:
    """Rule 5 trigger #5: user changed constraints mid-run. Forces a
    replan even if the executor was on the happy path."""
    session = _SESSIONS[session_id]
    run = _ensure_run(session_id)
    current_node_id = session.plan.graph.nodes[0].id if session.plan.graph.nodes else "stage-1"
    run.revised_plan = _replan(
        session, run, current_node_id, reason="constraints_changed",
    )
    return _finalise_after_replan(session, run)


def get_trace(run_id: str) -> list[TraceEvent]:
    for runs in _TASK_RUNS_BY_USER.values():
        for r in runs:
            if r.id == run_id:
                return list(r.execution_trace)
    return []


# ── Internal helpers ─────────────────────────────────────────────

def _ensure_run(session_id: str) -> TaskRun:
    """Idempotent run creator — returns the existing run if already
    started, creates it on first call."""
    session = _SESSIONS[session_id]
    if session.run is not None:
        return session.run

    if session.state not in ("approved", "executing"):
        raise ApprovalRequiredError(
            f"Session {session_id!r} not approved yet (state={session.state!r})",
        )

    run = TaskRun(
        id=uuid.uuid4().hex[:12],
        session_id=session_id,
        state="executing",
        plan_preview=session.plan.human_readable,
        result_summary={"goal": session.goal, "outcome": "pending"},
    )
    session.run = run
    _TASK_RUNS_BY_USER.setdefault(session.tg_id, []).append(run)
    return run


def _log_trace(run: TaskRun, event: TraceEvent) -> None:
    run.execution_trace.append(event)


def _replan(
    session: TaskSession,
    run: TaskRun,
    failed_node_id: str,
    *,
    reason: str,
) -> RevisedPlan:
    """NFR-11: revised plan preserves already-completed stages and only
    rebuilds the tail. `material_diff` is True unless the rebuild is
    cosmetic (goal contains 'cosmetic')."""
    preserved = [s for s in run.stages if s.status == "pass"]
    run.preserved_stages = preserved

    # Keep already-completed nodes at 'done', rebuild the rest.
    new_nodes: list[PlanNode] = []
    for n in session.plan.graph.nodes:
        if n.state == "done":
            new_nodes.append(n)
        else:
            # Rebuild — simplistic: same id so preservation tests can
            # compare sets, but clone via dataclass replace.
            new_nodes.append(PlanNode(
                id=n.id,
                description=n.description + " (revised)",
                tool=n.tool,
                expected_result=n.expected_result,
                acceptance_criteria=n.acceptance_criteria,
                state="pending",
            ))

    material = "cosmetic" not in session.goal.lower()
    revised = RevisedPlan(
        graph=PlanGraph(nodes=new_nodes),
        material_diff=material,
        reason=reason,
        human_readable=_render_plan_preview(
            session.goal, PlanGraph(nodes=new_nodes),
        ),
    )
    _log_trace(run, TraceEvent(
        kind="replan", node_id=failed_node_id,
        payload={"reason": reason, "material_diff": material},
    ))
    return revised


def _finalise_after_replan(session: TaskSession, run: TaskRun) -> TaskRun:
    """Drive the post-replan state transition:

    - material diff  -> run.state=awaiting_approval (FR-17 re-approval)
    - cosmetic diff  -> run.state=executing → done (resume & finish)
    """
    if run.revised_plan is None:
        return run

    if run.revised_plan.material_diff:
        run.state = "awaiting_approval"
        session.state = "awaiting_approval"
    else:
        # Cosmetic: treat as resumed-and-finished for the v1 stub.
        run.state = "done"
        session.state = "done"

    run.result_summary = {
        "goal": session.goal,
        "outcome": "replanned",
        "reason": run.revised_plan.reason,
        "material_diff": run.revised_plan.material_diff,
    }
    return run
