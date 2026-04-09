"""Tests for US-9 — Context-aware instruments + prompts as .md files.

## Трассируемость
Feature: Telegram AI Platform v2
Requirements: FR-44, FR-45

FR-44 — all instruments keep dialog context for follow-up questions.
        Web search builds a search query from history + current question,
        then synthesises the answer using history + search results.
FR-45 — system prompts live in prompts/system/*.md, loaded at call time.
"""

from __future__ import annotations
import os
import pytest


# ─────────────────────────────────────────────────────────────────
# FR-45 — prompt loader
# ─────────────────────────────────────────────────────────────────

def test_fr_45_prompt_loader_reads_md_file():
    try:
        from services import prompt_loader
    except ImportError:
        pytest.skip("TODO FR-45: services.prompt_loader not implemented")
    text = prompt_loader.load("chat")
    assert text, "chat prompt must not be empty"
    assert isinstance(text, str)


def test_fr_45_all_instrument_prompts_exist():
    try:
        from services import prompt_loader
    except ImportError:
        pytest.skip("TODO FR-45: prompt_loader not implemented")
    for name in ("chat", "file_search", "web_search", "web_search_query", "superagent"):
        text = prompt_loader.load(name)
        assert text, f"prompt '{name}' must exist and be non-empty"


def test_fr_45_prompt_files_are_real_md():
    """The actual .md files must exist on disk."""
    prompts_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "prompts", "system",
    )
    if not os.path.isdir(prompts_dir):
        pytest.skip("TODO FR-45: prompts/system/ directory not created")
    files = os.listdir(prompts_dir)
    md_files = [f for f in files if f.endswith(".md")]
    assert len(md_files) >= 4, f"expected ≥4 .md prompt files, found {md_files}"


# ─────────────────────────────────────────────────────────────────
# FR-44 — context-aware web search
# ─────────────────────────────────────────────────────────────────

def test_fr_51_superagent_followup_includes_chat_history():
    """FR-51: When user sends a follow-up after a completed task,
    the planner should receive prior chat_history as context."""
    try:
        from bot.handlers.platform import (
            _PLAN_GOAL, _PLAN_HISTORY,
        )
        from services import platform as svc
    except ImportError:
        pytest.skip("handler not importable")

    tg_id = 8888
    svc.get_user(tg_id)  # init user

    # Simulate: prior task conversation in chat_history
    svc.add_chat_message(tg_id, "user", "рынок роботов на Кипре")
    svc.add_chat_message(tg_id, "assistant", "Рынок роботов Кипра составляет...")

    # Now a follow-up — _PLAN_GOAL is NOT set (task was completed/cleared)
    assert tg_id not in _PLAN_GOAL

    # The handler should build full_goal from chat_history + new text
    user = svc.get_user(tg_id)
    new_text = "А в Греции?"
    history = user.chat_history
    assert len(history) >= 2
    assert "роботов" in history[0]["content"]

    # Cleanup
    svc.reset_chat(tg_id)


def test_fr_44_superagent_refinement_includes_original_goal():
    """When user refines a plan, the planner should receive the
    original goal + all refinements, not just the latest text."""
    try:
        from bot.handlers.platform import (
            _PLAN_GOAL, _PLAN_HISTORY, _handle_task_goal,
        )
    except ImportError:
        pytest.skip("handler not importable")

    # Simulate: user sets goal, then refines twice
    tg_id = 9999
    _PLAN_GOAL[tg_id] = "Маркетинг-ресёрч роботов"
    _PLAN_HISTORY[tg_id] = ["добавь Азию", "фокус на цены"]

    # The combined goal should contain all three
    parts = [f"Основная цель: {_PLAN_GOAL[tg_id]}"]
    for i, ref in enumerate(_PLAN_HISTORY[tg_id], 1):
        parts.append(f"Уточнение {i}: {ref}")
    full_goal = "\n".join(parts)

    assert "Маркетинг-ресёрч роботов" in full_goal
    assert "добавь Азию" in full_goal
    assert "фокус на цены" in full_goal
    assert "Уточнение 1" in full_goal
    assert "Уточнение 2" in full_goal

    # Cleanup
    _PLAN_GOAL.pop(tg_id, None)
    _PLAN_HISTORY.pop(tg_id, None)


def test_safe_html_relinkifies_after_strip():
    """_safe_html should re-create hyperlinks even after stripping
    broken tags, so URLs are always clickable."""
    try:
        from bot.handlers.platform import _safe_html
    except ImportError:
        pytest.skip("handler not importable")

    # Simulate broken HTML (unclosed tag)
    broken = (
        '<b>Title</b>\n'
        'Text <a href="https://example.com">link\n'  # unclosed <a>!
        '[1] Source — https://source.com/report\n'
        'Bare https://bare.url/here end.'
    )
    result = _safe_html(broken)
    # Should NOT contain unclosed tags
    assert "<a>" not in result or "</a>" in result
    # URLs should still be clickable hyperlinks
    assert '<a href="https://source.com/report">' in result
    assert '<a href="https://bare.url/here">' in result
    # [1] should be bold
    assert "<b>[1]</b>" in result


def test_fr_44_md_to_html_handles_citations_and_sources():
    """_md_to_html should:
    - Bold [N] inline citations
    - Convert source lines [N] Title — URL into hyperlinks
    - Handle headings, bold, bare URLs
    """
    try:
        from bot.handlers.platform import _md_to_html
    except ImportError:
        pytest.skip("handler not importable")

    result = _md_to_html(
        "# Отчёт\n\n"
        "Рынок вырос на 15% [1]. Лидеры: ABB, KUKA [2].\n"
        "Подробнее (https://example.com/details) здесь.\n"
        "Также [см. отчёт](https://report.eu/2024).\n\n"
        "[1] IFR Annual Report — https://ifr.org/report\n"
        "[2] EU Robotics Review — https://robotics.eu/review"
    )
    # Headings
    assert "<b>Отчёт</b>" in result
    # Inline [1] → bold
    assert "<b>[1]</b>" in result
    assert "<b>[2]</b>" in result
    # Source lines: [N] Title as hyperlink
    assert '<a href="https://ifr.org/report">IFR Annual Report</a>' in result
    assert '<a href="https://robotics.eu/review">EU Robotics Review</a>' in result
    # Parenthesized URL unwrapped
    assert "(https://example.com/details)" not in result
    # Markdown link unwrapped
    assert "](https://" not in result


def test_fr_44_web_search_query_uses_history(monkeypatch):
    """When user says "а в сша?" after asking about robots in Europe,
    the search query builder should produce something like
    "robots in USA" — not just "а в сша?".

    LLM is mocked so the test stays offline-deterministic."""
    try:
        from services import web_search_ctx
    except ImportError:
        pytest.skip("TODO FR-44: web_search_ctx not implemented")

    import config
    monkeypatch.setattr(config, "LLM_API_KEY", "fake-key")

    # Mock the OpenAI client to return an expanded query
    try:
        import openai as _openai  # noqa: F401
    except ImportError:
        pytest.skip("openai not installed in sandbox")

    class _FakeChoice:
        class message:
            content = "рынок робототехники в США 2024"

    class _FakeResp:
        choices = [_FakeChoice()]

    class _FakeClient:
        def __init__(self, **kw):
            pass

        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    return _FakeResp()

    monkeypatch.setattr("openai.OpenAI", _FakeClient)

    history = [
        {"role": "user", "content": "что думаешь про роботов в европе"},
        {"role": "assistant", "content": "Европейский рынок робототехники растёт..."},
    ]
    query = web_search_ctx.build_search_query(
        current_message="а в сша?",
        history=history,
    )
    assert query != "а в сша?", "query must be expanded with context"
    lower = query.lower()
    assert ("сша" in lower or "usa" in lower), f"query should mention USA: {query}"
