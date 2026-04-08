"""US-2 — Chat / Task modes (FR-6..FR-11, NFR-6..NFR-9).

Режимы «Чат» и «Задачи» в v1 — это разрез, который заменит legacy-поток,
где у бота был один «всегда RAG-чат». Большая часть тестов ниже
активируется, когда появится `services/modes.py` (или аналог) с явной
mode-state machine и Task Planner.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.spec


# Helper — пытается импортировать mode-aware API, с одним местом для skip.
def _modes_api():
    try:
        import services.modes as modes  # type: ignore
    except ModuleNotFoundError:
        return None
    return modes


# ─────────────────────────────────────────────────────────────────
# FR-6 — Chat mode как быстрый conversational mode
# ─────────────────────────────────────────────────────────────────

def test_fr_6_chat_mode_exists():
    """FR-6: в системе должен существовать режим Чат как отдельное
    mode-значение, которое можно включить."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO FR-6: mode state machine not implemented")
    assert "chat" in modes.AVAILABLE_MODES  # type: ignore[attr-defined]


def test_fr_6_only_two_modes_exist():
    """Rule 2 + FR-6/FR-9: режимов только два — chat и task."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO FR-6: mode state machine not implemented")
    assert set(modes.AVAILABLE_MODES) == {"chat", "task"}  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────────
# FR-7 — Chat uses Memory as context source
# ─────────────────────────────────────────────────────────────────

def test_fr_7_chat_retrieves_from_memory(platform_svc, tg_id):
    """FR-7: в режиме Чат ответ должен учитывать контекст из Памяти.

    Проверяем контракт Chat Runtime: вызов должен подтянуть список hits из
    Памяти по активным объектам (или по всем, если активных нет)."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO FR-7: Chat Runtime not implemented")
    platform_svc.create_domain(tg_id, "default")
    platform_svc.register_document(tg_id, "default", "facts.txt", num_chunks=3)
    result = modes.chat_answer(tg_id, "What do the facts say?")  # type: ignore[attr-defined]
    assert result.used_context, "Chat must use Memory as a context source"


# ─────────────────────────────────────────────────────────────────
# FR-8 — Chat must not build execution graph for simple queries
# ─────────────────────────────────────────────────────────────────

def test_fr_8_chat_does_not_build_execution_graph(platform_svc, tg_id):
    """FR-8: chat-ответ не должен создавать `TaskRun` / execution graph
    для простых запросов."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO FR-8: Chat Runtime not implemented")
    before = len(modes.list_task_runs(tg_id))  # type: ignore[attr-defined]
    modes.chat_answer(tg_id, "What is 2+2?")  # type: ignore[attr-defined]
    after = len(modes.list_task_runs(tg_id))  # type: ignore[attr-defined]
    assert before == after, "chat mode must not create a TaskRun"


# ─────────────────────────────────────────────────────────────────
# FR-9 — Task mode as supervised execution mode
# ─────────────────────────────────────────────────────────────────

def test_fr_9_task_mode_creates_supervised_task_session(platform_svc, tg_id):
    """FR-9: при постановке цели в режиме Задачи должна создаваться
    TaskSession, на которую навешан supervised-flow (draft → approval →
    execution)."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO FR-9: Task mode not implemented")
    session = modes.start_task(tg_id, goal="Write a short market report")  # type: ignore[attr-defined]
    assert session.mode == "task"
    assert session.state in ("draft", "awaiting_approval")


# ─────────────────────────────────────────────────────────────────
# FR-10 — full graph built before execution
# ─────────────────────────────────────────────────────────────────

def test_fr_10_full_graph_built_before_any_node_runs(platform_svc, tg_id):
    """FR-10: полный execution graph должен быть готов ещё до запуска
    первого этапа. Проверяем, что `plan.graph` содержит все заранее
    известные узлы, а `run.completed_steps` пуст."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO FR-10: full-graph planner not implemented")
    session = modes.start_task(tg_id, goal="Summarise [report.pdf]")  # type: ignore[attr-defined]
    plan = session.plan
    assert plan.graph is not None
    assert len(plan.graph.nodes) >= 1
    assert session.run is None or not session.run.completed_steps


# ─────────────────────────────────────────────────────────────────
# FR-11 — human-readable plan preview + explicit approval
# ─────────────────────────────────────────────────────────────────

def test_fr_11_plan_preview_is_presented_before_approval(platform_svc, tg_id):
    """FR-11: пользователь должен видеть текстовое представление плана
    и явно его подтверждать. До подтверждения session.state ≠ `executing`."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO FR-11: approval flow not implemented")
    session = modes.start_task(tg_id, goal="Research X and produce a note")  # type: ignore[attr-defined]
    assert session.plan.human_readable, "Plan preview must be rendered as text"
    assert session.state == "awaiting_approval"


def test_fr_11_explicit_approval_transitions_to_executing(platform_svc, tg_id):
    """FR-11: approval должен быть явным actor-ом, не авто-approve."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO FR-11: approval flow not implemented")
    session = modes.start_task(tg_id, goal="Do Y")  # type: ignore[attr-defined]
    modes.approve_plan(session.id)  # type: ignore[attr-defined]
    refreshed = modes.get_session(session.id)  # type: ignore[attr-defined]
    assert refreshed.state in ("executing", "done")


# ─────────────────────────────────────────────────────────────────
# NFR-6 — Chat ≤ Task latency at comparable complexity
# ─────────────────────────────────────────────────────────────────

def test_nfr_6_chat_mode_skips_planning_overhead(platform_svc, tg_id):
    """NFR-6: chat-ответ на простой вопрос не должен проходить через
    planner/approval — это необходимое условие того, что chat быстрее
    task для сопоставимой сложности."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO NFR-6: Chat Runtime not implemented")
    result = modes.chat_answer(tg_id, "hi")  # type: ignore[attr-defined]
    assert getattr(result, "plan_id", None) is None
    assert getattr(result, "approval_required", False) is False


# ─────────────────────────────────────────────────────────────────
# NFR-7 — Chat mode works without any prior planning
# ─────────────────────────────────────────────────────────────────

def test_nfr_7_chat_mode_needs_no_planning(platform_svc, tg_id):
    """NFR-7: chat должен работать без обязательного предварительного
    планирования — не требуется создавать PlanDraft."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO NFR-7: Chat Runtime not implemented")
    # Ни PlanDraft, ни TaskSession не должны создаваться.
    session_before = modes.list_task_runs(tg_id)  # type: ignore[attr-defined]
    modes.chat_answer(tg_id, "Just a quick question")  # type: ignore[attr-defined]
    session_after = modes.list_task_runs(tg_id)  # type: ignore[attr-defined]
    assert len(session_before) == len(session_after)


# ─────────────────────────────────────────────────────────────────
# NFR-8 — plan understandable, internal graph hidden
# ─────────────────────────────────────────────────────────────────

def test_nfr_8_plan_preview_hides_internal_graph_structure(platform_svc, tg_id):
    """NFR-8: пользовательский план не должен содержать сырого графа —
    только человекочитаемое описание шагов и expected outputs."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO NFR-8: Planner not implemented")
    session = modes.start_task(tg_id, goal="Draft an email")  # type: ignore[attr-defined]
    preview = session.plan.human_readable
    # Heuristics: plan preview must not leak graph node ids or DSL tokens.
    assert "node_id" not in preview
    assert "graph_definition" not in preview


# ─────────────────────────────────────────────────────────────────
# NFR-9 — no execution without explicit approval
# ─────────────────────────────────────────────────────────────────

def test_nfr_9_execution_blocked_before_approval(platform_svc, tg_id):
    """NFR-9: попытка запустить execution без approval должна упасть
    (или просто не выполнять ни одного узла)."""
    modes = _modes_api()
    if modes is None:
        pytest.skip("TODO NFR-9: approval gate not implemented")
    session = modes.start_task(tg_id, goal="Do Z")  # type: ignore[attr-defined]
    with pytest.raises(Exception):  # ApprovalRequiredError
        modes.execute(session.id)  # type: ignore[attr-defined]
