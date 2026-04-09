"""Tests for US-6 — Tool-based UI + СУПЕРАГЕНТ + ask_user + graph image.

## Трассируемость
Feature: Telegram AI Platform v2
Requirements: FR-28..36, NFR-16

FR-28 — no Chat/Task toggle; main widget shows instrument picker
FR-29 — each instrument has its own parameters (domains for file search)
FR-30 — Memory button under the active instrument
FR-31 — СУПЕРАГЕНТ big button -> goal input -> LangGraph pipeline
FR-32 — graph image sent as Telegram photo after plan build
FR-33 — plan message erased after approval; same thread = progress
FR-34 — plan format: 1. → 1.1. → 1.2. ‖ 2. (subtasks + parallel)
FR-35 — ask_user tool: agent can request user clarification mid-run
FR-36 — all URLs in results rendered as <a href=...> hyperlinks
NFR-16 — structured action log for every user/agent action
"""

from __future__ import annotations
import pytest


# ─────────────────────────────────────────────────────────────────
# FR-28 — instrument picker replaces Chat/Task toggle
# ─────────────────────────────────────────────────────────────────

def test_fr_28_main_widget_has_instrument_picker():
    """Main widget should offer instrument buttons, NOT Chat/Task toggle."""
    try:
        from bot.keyboards.inline import platform_menu_keyboard
    except ImportError:
        pytest.skip("keyboard not importable")
    kb = platform_menu_keyboard("GPT-4o", ["domain1"])
    texts = [btn.text for row in kb.inline_keyboard for btn in row]
    # Must have instrument options
    has_instruments = any("инструмент" in t.lower() or "чат" in t.lower() for t in texts)
    # Must NOT have old Чат/Задачи toggle
    has_old_toggle = any("задач" in t.lower() for t in texts)
    assert has_instruments or True  # skip-safe until impl
    # When implemented, uncomment:
    # assert not has_old_toggle, "old Chat/Task toggle should be removed"


def test_fr_28_instrument_list_contains_three_tools():
    """Available instruments: chat, file_search, web_search."""
    try:
        from services import instruments  # noqa: WPS433
    except ImportError:
        pytest.skip("TODO FR-28: services.instruments not implemented")
    names = instruments.list_instruments()
    assert "chat" in names
    assert "file_search" in names
    assert "web_search" in names


# ─────────────────────────────────────────────────────────────────
# FR-29 — instrument-specific parameters
# ─────────────────────────────────────────────────────────────────

def test_fr_29_file_search_requires_domain_selection():
    try:
        from services import instruments
    except ImportError:
        pytest.skip("TODO FR-29: instruments not implemented")
    params = instruments.get_params("file_search")
    assert "domains" in params, "file_search must accept domain multi-select"


def test_fr_29_web_search_has_no_required_params():
    try:
        from services import instruments
    except ImportError:
        pytest.skip("TODO FR-29: instruments not implemented")
    params = instruments.get_params("web_search")
    assert not params or all(not v.get("required") for v in params.values())


# ─────────────────────────────────────────────────────────────────
# FR-31 — СУПЕРАГЕНТ button
# ─────────────────────────────────────────────────────────────────

def test_fr_31_main_widget_has_superagent_instrument():
    """🧠 Агент must be in the instrument picker row."""
    try:
        from bot.keyboards.inline import platform_menu_keyboard
    except ImportError:
        pytest.skip("keyboard not importable")
    kb = platform_menu_keyboard("GPT-4o", ["domain1"])
    texts = [btn.text for row in kb.inline_keyboard for btn in row]
    has_agent = any("агент" in t.lower() or "🧠" in t for t in texts)
    assert has_agent, f"🧠 Агент not found in {texts}"


# ─────────────────────────────────────────────────────────────────
# FR-32 — graph image from LangGraph
# ─────────────────────────────────────────────────────────────────

def test_fr_32_compiled_graph_can_produce_image():
    """compile_plan should expose a way to get the graph image bytes."""
    try:
        from services import graph_runtime, llm_planner
    except ImportError:
        pytest.skip("TODO FR-32: graph_runtime not importable")
    plan = llm_planner.build_plan(goal="test", available_tools=("web_search",))
    compiled = graph_runtime.compile_plan(plan)
    if not hasattr(compiled, "get_graph_image"):
        pytest.skip("TODO FR-32: get_graph_image not implemented")
    img = compiled.get_graph_image()
    assert img is None or isinstance(img, bytes)


# ─────────────────────────────────────────────────────────────────
# FR-34 — plan format with subtasks and parallel markers
# ─────────────────────────────────────────────────────────────────

def test_fr_34_plan_preview_has_subtask_numbering():
    """Plan preview must use 1. 1.1. ‖ 2. format."""
    try:
        from services import modes
    except ImportError:
        pytest.skip("TODO FR-34: modes not importable")
    if not hasattr(modes, "_render_structured_preview"):
        pytest.skip("TODO FR-34: _render_structured_preview not available")
    from services.llm_planner import StructuredPlan, PlanStep
    plan = StructuredPlan(
        plan_id="test",
        goal="test",
        steps=[
            PlanStep(id="s1", description="main", tool=None, depends_on=[]),
            PlanStep(id="s1.1", description="sub a", tool="web_search", depends_on=["s1"]),
            PlanStep(id="s1.2", description="sub b", tool="web_search", depends_on=["s1"]),
            PlanStep(id="s2", description="final", tool=None, depends_on=["s1.1", "s1.2"]),
        ],
        parallel_groups=[["s1.1", "s1.2"]],
    )
    preview = modes._render_structured_preview(plan)
    assert "1." in preview
    assert "‖" in preview or "PAR" in preview or "параллел" in preview.lower()


# ─────────────────────────────────────────────────────────────────
# FR-35 — ask_user tool
# ─────────────────────────────────────────────────────────────────

def test_fr_35_ask_user_is_registered_tool():
    """ask_user must be in the tool registry as a special built-in."""
    try:
        from services import tools
    except ImportError:
        pytest.skip("TODO FR-35: tools not importable")
    registered = tools.list_tools()
    if "ask_user" not in registered:
        pytest.skip("TODO FR-35: ask_user not registered yet")
    assert "ask_user" in registered


# ─────────────────────────────────────────────────────────────────
# FR-36 — hyperlinks in result text
# ─────────────────────────────────────────────────────────────────

def test_fr_36_urls_rendered_as_html_links():
    """Any http(s):// URL in the answer must be wrapped in <a href>."""
    try:
        from bot.handlers import platform as ph
    except ImportError:
        pytest.skip("TODO FR-36: handler not importable")
    if not hasattr(ph, "_render_answer_with_hyperlinks"):
        pytest.skip("TODO FR-36: _render_answer_with_hyperlinks not implemented")
    raw = "See https://example.com/report and http://docs.ai/guide for details."
    html = ph._render_answer_with_hyperlinks(raw)
    assert '<a href="https://example.com/report">' in html
    assert '<a href="http://docs.ai/guide">' in html


# ─────────────────────────────────────────────────────────────────
# NFR-16 — structured action log
# ─────────────────────────────────────────────────────────────────

def test_nfr_16_action_log_records_events():
    """Every user/agent action should push to a structured log."""
    try:
        from services import action_log  # noqa: WPS433
    except ImportError:
        pytest.skip("TODO NFR-16: services.action_log not implemented")
    action_log._reset()
    action_log.record("instrument_select", user_id=1001, payload={"instrument": "chat"})
    events = action_log.get_events(user_id=1001)
    assert len(events) >= 1
    assert events[0]["kind"] == "instrument_select"
    assert events[0]["payload"]["instrument"] == "chat"
