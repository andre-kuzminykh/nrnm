"""US-3 — Supervised execution with basic tools (FR-12..FR-17, NFR-10..NFR-11).

Режим Задачи + tool layer + verification + replanning. На момент фиксации
v1-спеки ни одного из этих сервисов ещё нет в коде — тесты образуют
spec-driven roadmap для Task Executor / Tool Layer / Replanning Engine.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.spec


def _modes_api():
    try:
        import services.modes as modes  # type: ignore
    except ModuleNotFoundError:
        return None
    return modes


def _tools_api():
    try:
        import services.tools as tools  # type: ignore
    except ModuleNotFoundError:
        return None
    return tools


# ─────────────────────────────────────────────────────────────────
# FR-12 — execution follows approved graph
# ─────────────────────────────────────────────────────────────────

def test_fr_12_executor_follows_approved_graph_exactly(platform_svc, tg_id):
    """FR-12: executor должен проходить ровно по тем узлам, что были
    в approved graph — никаких ad-hoc шагов."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO FR-12: Task Executor not implemented")

    session = modes.start_task(tg_id, goal="Build 2-step plan")  # type: ignore[attr-defined]
    planned_node_ids = [n.id for n in session.plan.graph.nodes]
    modes.approve_plan(session.id)  # type: ignore[attr-defined]
    run = modes.execute(session.id)  # type: ignore[attr-defined]

    executed_ids = [stage.node_id for stage in run.stages]
    # Executor should touch ONLY planned nodes (+ maybe a subset, but never
    # unplanned nodes).
    assert set(executed_ids).issubset(set(planned_node_ids))


# ─────────────────────────────────────────────────────────────────
# FR-13 — only Web Search + PDF Parser as tools in v1
# ─────────────────────────────────────────────────────────────────

def test_fr_13_only_two_tools_registered_in_v1():
    """FR-13: в v1 в tool registry должны быть ровно два инструмента —
    `web_search` и `pdf_parser`. `[контекст]` — это не обычный tool, а
    встроенный механизм подключения Памяти (проверяется в FR-14)."""
    tools = _tools_api()
    if tools is None:
        pytest.skip("TODO FR-13: Tool layer not implemented")
    registered = set(tools.list_tools())  # type: ignore[attr-defined]
    assert registered == {"web_search", "pdf_parser"}


def test_fr_13_executor_rejects_disallowed_tool_call(platform_svc, tg_id):
    """FR-13: попытка вызвать tool не из whitelist должна блокироваться."""
    tools = _tools_api()
    if tools is None:
        pytest.skip("TODO FR-13: Tool layer not implemented")
    with pytest.raises(Exception):  # DisallowedToolError
        tools.call("browser_automation", {"url": "https://example.com"})  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────────
# FR-14 — `[контекст]` as built-in full file attach
# ─────────────────────────────────────────────────────────────────

def test_fr_14_context_mechanism_is_builtin_not_a_tool(platform_svc, tg_id):
    """FR-14: `[контекст]` не должен появляться в tool registry — это
    встроенный механизм планировщика, а не внешний инструмент."""
    tools = _tools_api()
    if tools is None:
        pytest.skip("TODO FR-14: context mechanism not implemented")
    assert "context" not in tools.list_tools()  # type: ignore[attr-defined]
    assert "контекст" not in tools.list_tools()  # type: ignore[attr-defined]


def test_fr_14_context_ref_attaches_full_file_to_run(platform_svc, tg_id):
    """FR-14: при планировании/выполнении `[file]` файл должен
    прикрепиться к execution context целиком."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO FR-14: context attachment not implemented")
    platform_svc.create_domain(tg_id, "default")
    platform_svc.register_document(tg_id, "default", "paper.pdf", num_chunks=20)

    session = modes.start_task(tg_id, goal="Summarise [paper.pdf]")  # type: ignore[attr-defined]
    attached = [o.filename for o in session.plan.attached_memory]
    assert "paper.pdf" in attached


# ─────────────────────────────────────────────────────────────────
# FR-15 — each stage has expected result + acceptance criteria
# ─────────────────────────────────────────────────────────────────

def test_fr_15_every_planned_stage_has_expected_result(platform_svc, tg_id):
    """FR-15: каждый узел плана должен нести expected_result и acceptance
    criteria — без них verifier работать не сможет."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO FR-15: Planner not implemented")
    session = modes.start_task(tg_id, goal="Do a multi-step task")  # type: ignore[attr-defined]
    for node in session.plan.graph.nodes:
        assert node.expected_result, f"node {node.id} missing expected_result"
        assert node.acceptance_criteria, f"node {node.id} missing acceptance_criteria"


# ─────────────────────────────────────────────────────────────────
# FR-16 — replan on deviation
# ─────────────────────────────────────────────────────────────────

def test_fr_16_failed_stage_triggers_replanning(platform_svc, tg_id):
    """FR-16 (Rule 5 trigger #1+#2): если stage не прошёл verification /
    acceptance criteria, должен сработать replanning engine и выдать
    revised plan."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO FR-16: Replanning engine not implemented")
    session = modes.start_task(tg_id, goal="simulate-failure")  # type: ignore[attr-defined]
    modes.approve_plan(session.id)  # type: ignore[attr-defined]
    run = modes.execute(session.id, inject_failure_on="stage-1")  # type: ignore[attr-defined]
    assert run.revised_plan is not None, "replan must produce a revised plan"


def test_fr_16_tool_failure_triggers_replanning(platform_svc, tg_id):
    """FR-16 (Rule 5 trigger #3): tool failure, блокирующий прогресс,
    должен инициировать replanning."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO FR-16: tool-failure replan not implemented")
    session = modes.start_task(tg_id, goal="use-web-search")  # type: ignore[attr-defined]
    modes.approve_plan(session.id)  # type: ignore[attr-defined]
    run = modes.execute(session.id, inject_tool_failure="web_search")  # type: ignore[attr-defined]
    assert run.revised_plan is not None
    assert run.revised_plan.reason == "tool_failure"


def test_fr_16_missing_input_triggers_replanning(platform_svc, tg_id):
    """FR-16 (Rule 5 trigger #4): отсутствие обязательного input у stage
    должно запускать replanning."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO FR-16: missing-input replan not implemented")
    session = modes.start_task(tg_id, goal="requires-missing-input")  # type: ignore[attr-defined]
    modes.approve_plan(session.id)  # type: ignore[attr-defined]
    run = modes.execute(session.id, inject_missing_input_on="stage-2")  # type: ignore[attr-defined]
    assert run.revised_plan is not None
    assert run.revised_plan.reason == "missing_input"


def test_fr_16_user_constraint_change_triggers_replanning(platform_svc, tg_id):
    """FR-16 (Rule 5 trigger #5): изменение constraints пользователем во
    время исполнения должно инициировать replanning."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO FR-16: user-constraint replan not implemented")
    session = modes.start_task(tg_id, goal="long-running")  # type: ignore[attr-defined]
    modes.approve_plan(session.id)  # type: ignore[attr-defined]
    modes.begin_execution(session.id)  # type: ignore[attr-defined]
    run = modes.update_constraints(  # type: ignore[attr-defined]
        session.id, constraints={"max_steps": 2},
    )
    assert run.revised_plan is not None
    assert run.revised_plan.reason == "constraints_changed"


def test_fr_16_rule5_no_replan_without_deviation(platform_svc, tg_id):
    """Rule 5 negative: если никаких из 5 триггеров не произошло —
    replanning engine не должен вообще активироваться, run завершается
    по исходному approved graph."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO Rule 5: negative replan path not implemented")
    session = modes.start_task(tg_id, goal="happy-path")  # type: ignore[attr-defined]
    modes.approve_plan(session.id)  # type: ignore[attr-defined]
    run = modes.execute(session.id)  # type: ignore[attr-defined]
    assert run.revised_plan is None
    assert run.state == "done"


# ─────────────────────────────────────────────────────────────────
# FR-17 — material replan requires re-approval
# ─────────────────────────────────────────────────────────────────

def test_fr_17_material_replan_requires_reapproval(platform_svc, tg_id):
    """FR-17: если revised plan materially меняет будущие действия,
    он не должен запускаться без нового approval от пользователя."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO FR-17: re-approval flow not implemented")
    session = modes.start_task(tg_id, goal="trigger-material-diff")  # type: ignore[attr-defined]
    modes.approve_plan(session.id)  # type: ignore[attr-defined]
    run = modes.execute(session.id, inject_failure_on="stage-1")  # type: ignore[attr-defined]

    assert run.state == "awaiting_approval"
    assert run.revised_plan.material_diff is True


def test_fr_17_cosmetic_replan_does_not_require_reapproval(platform_svc, tg_id):
    """FR-17 (negative): тривиальная перестройка графа (non-material)
    не должна заставлять пользователя подтверждать план повторно."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO FR-17: re-approval flow not implemented")
    session = modes.start_task(tg_id, goal="trigger-cosmetic-diff")  # type: ignore[attr-defined]
    modes.approve_plan(session.id)  # type: ignore[attr-defined]
    run = modes.execute(session.id, inject_failure_on="stage-1")  # type: ignore[attr-defined]
    assert run.state in ("executing", "done")
    assert run.revised_plan.material_diff is False


# ─────────────────────────────────────────────────────────────────
# NFR-10 — tool calls + node results are traceable
# ─────────────────────────────────────────────────────────────────

def test_nfr_10_tool_calls_and_node_results_are_logged(platform_svc, tg_id):
    """NFR-10: каждый tool call и каждый node result должны быть залогированы
    в TraceEvent-поток."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO NFR-10: tracing not implemented")
    session = modes.start_task(tg_id, goal="Call web_search once")  # type: ignore[attr-defined]
    modes.approve_plan(session.id)  # type: ignore[attr-defined]
    run = modes.execute(session.id)  # type: ignore[attr-defined]

    trace = modes.get_trace(run.id)  # type: ignore[attr-defined]
    kinds = {e.kind for e in trace}
    assert "tool_call" in kinds or "stage_result" in kinds
    # Every tool call must be tied back to a node.
    for event in trace:
        if event.kind == "tool_call":
            assert event.node_id, "tool_call events must carry node_id for traceability"


# ─────────────────────────────────────────────────────────────────
# NFR-11 — replanning preserves already-valid completed stages
# ─────────────────────────────────────────────────────────────────

def test_nfr_11_replan_preserves_already_valid_stages(platform_svc, tg_id):
    """NFR-11: если stage-1 прошёл успешно, а упал stage-3 — revised
    plan не должен пересчитывать stage-1."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO NFR-11: incremental replan not implemented")
    session = modes.start_task(tg_id, goal="simulate-failure")  # type: ignore[attr-defined]
    modes.approve_plan(session.id)  # type: ignore[attr-defined]
    run = modes.execute(session.id, inject_failure_on="stage-3")  # type: ignore[attr-defined]

    preserved = {s.node_id for s in run.preserved_stages}
    replanned = {n.id for n in run.revised_plan.graph.nodes if n.state == "pending"}
    assert "stage-1" in preserved
    assert "stage-1" not in replanned


# ─────────────────────────────────────────────────────────────────
# NFR-12 — every task run has plan preview + execution trace + result summary
# ─────────────────────────────────────────────────────────────────

def test_nfr_12_task_run_has_plan_preview_trace_and_summary(platform_svc, tg_id):
    """NFR-12 (из метрики §1): каждый task run должен содержать все три
    артефакта — plan preview, execution trace и result summary. Это
    100%-инвариант: даже упавший или отменённый run должен их иметь."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO NFR-12: task run artefacts not implemented")
    session = modes.start_task(tg_id, goal="produce-any-result")  # type: ignore[attr-defined]
    modes.approve_plan(session.id)  # type: ignore[attr-defined]
    run = modes.execute(session.id)  # type: ignore[attr-defined]

    assert run.plan_preview, "NFR-12: plan_preview missing from task run"
    assert run.execution_trace is not None, "NFR-12: execution_trace missing"
    assert run.result_summary is not None, "NFR-12: result_summary missing"


def test_nfr_12_failed_run_still_has_artefacts(platform_svc, tg_id):
    """NFR-12: даже если run упал — артефакты обязаны быть. Мы не хотим
    невидимых «фантомных» runs."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO NFR-12: task run artefacts not implemented")
    session = modes.start_task(tg_id, goal="will-fail")  # type: ignore[attr-defined]
    modes.approve_plan(session.id)  # type: ignore[attr-defined]
    run = modes.execute(session.id, inject_failure_on="stage-1")  # type: ignore[attr-defined]
    assert run.plan_preview
    assert run.execution_trace is not None
    assert run.result_summary is not None
