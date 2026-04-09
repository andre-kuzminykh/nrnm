"""Tests for US-10 — Superagent expanded tools.

## Трассируемость
Feature: Telegram AI Platform v2
Requirements: FR-47, FR-48, FR-49, FR-50

FR-47 — ask_user: agent can pause and ask user a question mid-run
FR-48 — rag_search: agent can query the internal RAG database
FR-49 — file_open: agent can open a file, auto-summarize if too large
FR-50 — all tools available to planner with descriptions
"""

from __future__ import annotations
import pytest


# ─────────────────────────────────────────────────────────────────
# FR-47 — ask_user tool
# ─────────────────────────────────────────────────────────────────

def test_fr_47_ask_user_registered_as_builtin():
    from services import tools
    assert "ask_user" in tools.list_tools()


def test_fr_47_ask_user_returns_pending_status():
    """ask_user should return a special status that the runtime
    can interpret as 'pause and wait for user input'."""
    from services import tools
    result = tools.call("ask_user", {"question": "Какой регион интересует?"})
    assert result.status == "pending"
    assert result.output.get("question") == "Какой регион интересует?"


# ─────────────────────────────────────────────────────────────────
# FR-48 — rag_search tool
# ─────────────────────────────────────────────────────────────────

def test_fr_48_rag_search_registered():
    from services import tools
    assert "rag_search" in tools.list_tools()


def test_fr_48_rag_search_returns_result_shape():
    from services import tools
    result = tools.call("rag_search", {"query": "test query", "tg_id": 1001})
    assert result.status in ("ok", "error")
    if result.status == "ok":
        assert isinstance(result.output.get("hits"), list)


# ─────────────────────────────────────────────────────────────────
# FR-49 — file_open tool
# ─────────────────────────────────────────────────────────────────

def test_fr_49_file_open_registered():
    from services import tools
    assert "file_open" in tools.list_tools()


def test_fr_49_file_open_returns_content(platform_svc, tg_id):
    from services import tools, memory as mem, file_tree as ft

    ft.create_folder(tg_id, "/", "docs")
    ft.add_file(tg_id, "/docs", "report.txt", "d1", 3)
    mem.set_object_content("d1", "This is the full report content.")

    result = tools.call("file_open", {
        "filename": "report.txt",
        "tg_id": tg_id,
    })
    assert result.status == "ok"
    assert "full report content" in result.output.get("content", "")
    assert result.output.get("summarized") is False


def test_fr_49_file_open_summarizes_large_file(platform_svc, tg_id):
    from services import tools, memory as mem, file_tree as ft

    ft.create_folder(tg_id, "/", "big")
    ft.add_file(tg_id, "/big", "huge.txt", "d2", 500)
    huge = "Lorem ipsum dolor sit amet. " * 20000  # ~540k chars
    mem.set_object_content("d2", huge)

    result = tools.call("file_open", {
        "filename": "huge.txt",
        "tg_id": tg_id,
        "max_chars": 5000,
    })
    assert result.status == "ok"
    assert result.output.get("summarized") is True
    assert len(result.output.get("content", "")) <= 5000


# ─────────────────────────────────────────────────────────────────
# FR-50 — all tools available to planner
# ─────────────────────────────────────────────────────────────────

def test_fr_50_planner_sees_all_agent_tools():
    from services import tools
    all_tools = tools.list_tools()
    for name in ("web_search", "rag_search", "file_open", "ask_user"):
        assert name in all_tools, f"{name} missing from tool registry"
