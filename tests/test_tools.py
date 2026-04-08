"""Tests for the v1 Tool Layer (services/tools.py).

## Трассируемость
Feature: Telegram AI Platform v1
Requirements: FR-13, FR-14, NFR-10

Covers:
- FR-13 whitelist enforcement (no tool outside {web_search, pdf_parser}).
- FR-13 web_search provider routing:
  * SerpAPI path when SERPAPI_API_KEY is set (httpx is monkey-patched
    so no network call happens in CI).
  * Stub fallback when key is empty.
- Empty query -> hard error (no tool call).
- SerpAPI HTTP failure -> status="error" (so the executor fires the
  Rule-5 tool_failure replan trigger, FR-16).
"""
from __future__ import annotations

import pytest

import config
from services import tools as tool_layer


# ── FR-13 whitelist ─────────────────────────────────────────────

def test_fr_13_disallowed_tool_raises():
    with pytest.raises(tool_layer.DisallowedToolError):
        tool_layer.call("rm_rf")


def test_fr_13_exactly_two_tools_in_v1():
    assert tool_layer.list_tools() == {"web_search", "pdf_parser"}


# ── Provider routing ────────────────────────────────────────────

def test_web_search_empty_query_is_error(monkeypatch):
    monkeypatch.setattr(config, "SERPAPI_API_KEY", "")
    res = tool_layer.call("web_search", {"query": ""})
    assert res.status == "error"
    assert "empty query" in (res.error or "")


def test_web_search_falls_back_to_stub_without_key(monkeypatch):
    """No SERPAPI_API_KEY -> deterministic stub (so offline dev + CI
    still exercise the executor pipeline)."""
    monkeypatch.setattr(config, "SERPAPI_API_KEY", "")
    res = tool_layer.call("web_search", {"query": "aiogram 3 tutorial"})
    assert res.status == "ok"
    assert res.metadata["provider"] == "stub"
    assert len(res.output["hits"]) >= 1


def test_web_search_uses_serpapi_when_key_present(monkeypatch):
    """With a key, call the SerpAPI endpoint. httpx is patched so no
    network traffic leaves the test — we just assert our code reaches
    the right URL with the right params and parses the response
    correctly."""
    monkeypatch.setattr(config, "SERPAPI_API_KEY", "fake-key")

    captured: dict = {}

    class _FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "organic_results": [
                    {
                        "title": "Claude docs",
                        "link": "https://docs.anthropic.com/",
                        "snippet": "Anthropic docs",
                    },
                    {
                        "title": "Claude API",
                        "link": "https://api.anthropic.com/",
                        "snippet": "API reference",
                    },
                ],
                "answer_box": {
                    "title": "Claude",
                    "link": "https://anthropic.com/",
                    "answer": "An AI assistant by Anthropic.",
                },
            }

    class _FakeClient:
        def __init__(self, timeout=None):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            captured["url"] = url
            captured["params"] = params
            return _FakeResp()

    import httpx as real_httpx  # noqa: F401 — ensure module importable
    monkeypatch.setattr("httpx.Client", _FakeClient)

    res = tool_layer.call("web_search", {"query": "Claude Opus"})

    assert res.status == "ok"
    assert res.metadata["provider"] == "serpapi"
    assert captured["url"] == config.SERPAPI_ENDPOINT
    assert captured["params"]["q"] == "Claude Opus"
    assert captured["params"]["api_key"] == "fake-key"
    assert captured["params"]["engine"] == config.SERPAPI_ENGINE

    # answer_box should be injected as hit 0, then the 2 organic hits.
    hits = res.output["hits"]
    assert len(hits) == 3
    assert hits[0]["snippet"].startswith("An AI assistant")
    assert hits[1]["url"] == "https://docs.anthropic.com/"


def test_web_search_serpapi_http_error_degrades_to_tool_failure(monkeypatch):
    """Rule-5 trigger #3: when the provider fails, we return an error
    result (NOT raise) so the executor can flip to replanning without
    crashing. This is the seam between FR-13 and FR-16."""
    monkeypatch.setattr(config, "SERPAPI_API_KEY", "fake-key")

    import httpx

    class _BrokenClient:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            raise httpx.ConnectError("boom")

    monkeypatch.setattr("httpx.Client", _BrokenClient)

    res = tool_layer.call("web_search", {"query": "anything"})
    assert res.status == "error"
    assert "serpapi" in (res.error or "").lower()
    assert res.metadata["provider"] == "serpapi"


# ── pdf_parser smoke ────────────────────────────────────────────

def test_pdf_parser_echoes_content():
    res = tool_layer.call("pdf_parser", {"content": "hello world"})
    assert res.status == "ok"
    assert res.output["text"] == "hello world"
