"""Tests for US-5 — MCP registry + per-domain scoping + RAG hyperlinks.

## Трассируемость
Feature: Telegram AI Platform v1.1
Requirements: FR-23, FR-24, FR-25, FR-26, FR-27, NFR-15

Encodes the contracts:
- FR-23: MCP entries carry {name, url, token, description, timestamps}
  and support add/edit/delete.
- FR-24: Every domain owns its MCP registry. New domains auto-bootstrap
  a default `web_search` pointing at `builtin://serpapi`.
- FR-25: LLM planner gets the active domain's MCPs as available_tools.
- FR-26: mcp_client.dispatch routes `builtin://*` in-process and
  `http(s)://*` to an HTTP server with bearer auth.
- FR-27: RAG answer rendering — no numbered citations, no footer.
- NFR-15: HTTP MCP errors degrade into tool_failure ToolCallResult.
"""

from __future__ import annotations

import pytest


def _registry():
    try:
        from services import mcp_registry  # noqa: WPS433
        return mcp_registry
    except Exception:  # noqa: BLE001
        return None


def _client():
    try:
        from services import mcp_client  # noqa: WPS433
        return mcp_client
    except Exception:  # noqa: BLE001
        return None


# ─────────────────────────────────────────────────────────────────
# FR-23 — MCP entry shape + CRUD
# ─────────────────────────────────────────────────────────────────

def test_fr_23_mcp_entry_has_required_fields(platform_svc, tg_id):
    reg = _registry()
    if reg is None:
        pytest.skip("TODO FR-23: services.mcp_registry not implemented")
    platform_svc.create_domain(tg_id, "default")
    entry = reg.add_mcp(
        tg_id, "default",
        name="my_tool",
        url="https://mcp.example.com/v1",
        token="bearer-xyz",
        description="A custom tool for X",
    )
    assert entry.name == "my_tool"
    assert entry.url == "https://mcp.example.com/v1"
    assert entry.token == "bearer-xyz"
    assert entry.description.startswith("A custom")
    assert entry.created_at
    assert entry.updated_at


def test_fr_23_mcp_can_be_edited(platform_svc, tg_id):
    reg = _registry()
    if reg is None:
        pytest.skip("TODO FR-23: MCP edit not implemented")
    platform_svc.create_domain(tg_id, "default")
    entry = reg.add_mcp(
        tg_id, "default",
        name="my_tool", url="https://mcp.example.com/v1",
        token="old", description="old desc",
    )
    old_updated = entry.updated_at
    updated = reg.update_mcp(
        tg_id, "default", "my_tool",
        token="new", description="new desc",
    )
    assert updated.token == "new"
    assert updated.description == "new desc"
    # URL unchanged because we didn't pass it
    assert updated.url == "https://mcp.example.com/v1"
    assert updated.updated_at >= old_updated


def test_fr_23_mcp_can_be_deleted(platform_svc, tg_id):
    reg = _registry()
    if reg is None:
        pytest.skip("TODO FR-23: MCP delete not implemented")
    platform_svc.create_domain(tg_id, "default")
    reg.add_mcp(
        tg_id, "default",
        name="tmp", url="https://x", token="t", description="x",
    )
    assert reg.get_mcp(tg_id, "default", "tmp") is not None
    reg.delete_mcp(tg_id, "default", "tmp")
    assert reg.get_mcp(tg_id, "default", "tmp") is None


# ─────────────────────────────────────────────────────────────────
# FR-24 — per-domain scoping + default bootstrap
# ─────────────────────────────────────────────────────────────────

def test_fr_24_new_domain_has_default_web_search_mcp(platform_svc, tg_id):
    reg = _registry()
    if reg is None:
        pytest.skip("TODO FR-24: default MCP bootstrap not implemented")
    platform_svc.create_domain(tg_id, "research")
    mcps = reg.list_mcps(tg_id, "research")
    names = [m.name for m in mcps]
    assert "web_search" in names
    ws = reg.get_mcp(tg_id, "research", "web_search")
    assert ws.url.startswith("builtin://")
    assert "поиск" in ws.description.lower() or "search" in ws.description.lower()


def test_fr_24_mcp_registries_are_isolated_across_domains(platform_svc, tg_id):
    reg = _registry()
    if reg is None:
        pytest.skip("TODO FR-24: per-domain scoping not implemented")
    platform_svc.create_domain(tg_id, "dom-a")
    platform_svc.create_domain(tg_id, "dom-b")
    reg.add_mcp(
        tg_id, "dom-a",
        name="a_only", url="https://a", token="t", description="a",
    )
    names_a = [m.name for m in reg.list_mcps(tg_id, "dom-a")]
    names_b = [m.name for m in reg.list_mcps(tg_id, "dom-b")]
    assert "a_only" in names_a
    assert "a_only" not in names_b
    # Both should still have the default web_search
    assert "web_search" in names_a
    assert "web_search" in names_b


# ─────────────────────────────────────────────────────────────────
# FR-25 — planner receives MCP descriptions as available_tools
# ─────────────────────────────────────────────────────────────────

def test_fr_25_planner_receives_domain_mcps(platform_svc, tg_id):
    reg = _registry()
    if reg is None:
        pytest.skip("TODO FR-25: planner integration not implemented")
    try:
        from services import llm_planner  # noqa: WPS433
    except Exception:  # noqa: BLE001
        pytest.skip("llm_planner not available")

    platform_svc.create_domain(tg_id, "research")
    reg.add_mcp(
        tg_id, "research",
        name="github_search",
        url="https://mcp.example.com/github",
        token="t",
        description="Searches public GitHub repositories by keyword or language.",
    )
    mcps = reg.list_mcps(tg_id, "research")
    mcp_names = tuple(m.name for m in mcps)
    plan = llm_planner.build_plan(
        goal="Поищи репозитории про mcp",
        available_tools=mcp_names,
        mcp_catalog=mcps,
    )
    # Planner must not invent tool names outside what we passed.
    for step in plan.steps:
        assert step.tool in (None, *mcp_names)


# ─────────────────────────────────────────────────────────────────
# FR-26 — mcp_client dispatches builtin + http
# ─────────────────────────────────────────────────────────────────

def test_fr_26_builtin_scheme_runs_in_process(platform_svc, tg_id):
    client = _client()
    reg = _registry()
    if client is None or reg is None:
        pytest.skip("TODO FR-26: mcp_client not implemented")
    platform_svc.create_domain(tg_id, "default")
    ws = reg.get_mcp(tg_id, "default", "web_search")
    assert ws.url.startswith("builtin://")
    result = client.dispatch(ws, {"query": "unit test"})
    assert result.status in ("ok", "error")
    assert result.metadata.get("tool") == "web_search"


def test_fr_26_http_scheme_posts_to_server(monkeypatch):
    client = _client()
    if client is None:
        pytest.skip("TODO FR-26: http dispatch not implemented")
    import config

    captured: dict = {}

    class _FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True, "data": "hello from mcp"}

    class _FakeClient:
        def __init__(self, *a, **kw):
            captured["timeout"] = kw.get("timeout")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _FakeResp()

    monkeypatch.setattr("httpx.Client", _FakeClient)

    from services.mcp_registry import MCPEntry
    entry = MCPEntry(
        name="external",
        url="https://mcp.example.com/v1/call",
        token="secret-token",
        description="external tool",
        created_at="now",
        updated_at="now",
    )
    result = client.dispatch(entry, {"query": "hello"})
    assert result.status == "ok"
    assert captured["url"] == entry.url
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["json"]["tool"] == "external"
    assert captured["json"]["args"]["query"] == "hello"


def test_nfr_15_http_mcp_error_degrades_to_tool_failure(monkeypatch):
    client = _client()
    if client is None:
        pytest.skip("TODO NFR-15: http failure path not implemented")
    import httpx

    class _BrokenClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            raise httpx.ConnectError("boom")

    monkeypatch.setattr("httpx.Client", _BrokenClient)

    from services.mcp_registry import MCPEntry
    entry = MCPEntry(
        name="flaky", url="https://flaky.example/v1",
        token="t", description="x",
        created_at="now", updated_at="now",
    )
    result = client.dispatch(entry, {"query": "x"})
    assert result.status == "error"
    assert "boom" in (result.error or "") or "mcp" in (result.error or "").lower()


# ─────────────────────────────────────────────────────────────────
# FR-27 — RAG answer rendering helpers
# ─────────────────────────────────────────────────────────────────

def test_fr_27_rag_answer_has_no_numbered_citations():
    from bot.handlers.platform import _render_answer_with_inline_sources

    html = _render_answer_with_inline_sources(
        "Согласно report.pdf, продажи выросли [1]. Доп детали в notes.txt [2].",
        ["report.pdf", "notes.txt"],
    )
    assert "[1]" not in html
    assert "[2]" not in html
    assert "<b>report.pdf</b>" in html
    assert "<b>notes.txt</b>" in html


def test_fr_27_dedupe_sources_returns_ordered_list_without_idx():
    from bot.handlers.platform import _dedupe_sources

    sources = _dedupe_sources([
        {"filename": "a.pdf", "message_id": 1, "score": 0.9},
        {"filename": "b.pdf", "message_id": 2, "score": 0.8},
        {"filename": "a.pdf", "message_id": 1, "score": 0.7},  # dup
    ])
    assert [s["filename"] for s in sources] == ["a.pdf", "b.pdf"]
    # No idx field anywhere — that's the point of FR-27.
    for s in sources:
        assert "idx" not in s
