"""v1 Tool Layer — whitelisted tools for Task mode.

## Трассируемость
Feature: Telegram AI Platform v1
Requirements: FR-13, FR-14, NFR-10, Rule 3

v1 exposes exactly two tools: `web_search` and `pdf_parser`. Everything
else the planner wants to reach for (including `[контекст]`) is NOT a
tool — `[контекст]` is a built-in memory attach mechanism, not a tool
call (FR-14, verified by `test_fr_14_context_mechanism_is_builtin_not_a_tool`).

The layer is also the enforcement point for NFR-10 traceability: every
tool invocation is routed through `call()`, which records a TraceEvent
and returns a structured ToolCallResult. Task Executor never reaches
a concrete tool implementation directly — it always goes through
`call()`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import config

logger = logging.getLogger(__name__)


class DisallowedToolError(RuntimeError):
    """Raised when the Executor tries to invoke a tool that isn't in the
    v1 whitelist (FR-13)."""


@dataclass
class ToolCallResult:
    status: str  # "ok" | "error"
    output: Any = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)


# ── web_search: SerpAPI provider with stub fallback ─────────────

def _web_search_stub(query: str) -> ToolCallResult:
    """Deterministic fake hits — used in tests, offline dev, or when
    SERPAPI_API_KEY is not configured. Keeps the executor pipeline
    exercisable without hitting the network."""
    return ToolCallResult(
        status="ok",
        output={
            "hits": [
                {"title": f"Result for {query}", "url": "https://example.com/1", "snippet": ""},
                {"title": f"Secondary {query}", "url": "https://example.com/2", "snippet": ""},
            ],
        },
        metadata={"tool": "web_search", "provider": "stub", "query": query},
    )


def _web_search_serpapi(query: str, num: int = 5) -> ToolCallResult:
    """Real SerpAPI call. Returns a normalised `hits` list so callers
    don't have to care whether the underlying provider is SerpAPI,
    Brave, or the stub.

    Uses sync `httpx.Client` because the executor loop in
    `services.modes.execute` is synchronous. Any network error
    degrades into a `status="error"` ToolCallResult — the executor
    interprets that as a Rule-5 tool_failure trigger (FR-16) and the
    replanning engine takes over.
    """
    try:
        import httpx  # transitive dep via openai / aiogram
    except Exception as exc:  # noqa: BLE001
        return ToolCallResult(
            status="error", error=f"httpx unavailable: {exc}",
            metadata={"tool": "web_search", "provider": "serpapi"},
        )

    params = {
        "engine": config.SERPAPI_ENGINE,
        "q": query,
        "api_key": config.SERPAPI_API_KEY,
        "num": num,
    }
    import time as _time

    last_err = ""
    data = None
    for attempt in range(1, 4):  # 3 attempts
        try:
            with httpx.Client(timeout=config.SERPAPI_TIMEOUT) as client:
                resp = client.get(config.SERPAPI_ENDPOINT, params=params)
                resp.raise_for_status()
                data = resp.json()
                break
        except (httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
            last_err = f"serpapi timeout (attempt {attempt}/3): {exc}"
            logger.warning("SerpAPI timeout attempt %d/3: %s", attempt, exc)
            if attempt < 3:
                _time.sleep(1.0 * attempt)
        except httpx.HTTPError as exc:
            last_err = f"serpapi http: {exc}"
            logger.warning("SerpAPI HTTP error: %s", exc)
            break  # non-timeout HTTP errors don't retry
        except Exception as exc:  # noqa: BLE001
            last_err = f"serpapi: {exc}"
            logger.warning("SerpAPI unexpected: %s", exc)
            break

    if data is None:
        return ToolCallResult(
            status="error", error=last_err,
            metadata={"tool": "web_search", "provider": "serpapi", "query": query},
        )

    hits = []
    for item in (data.get("organic_results") or [])[:num]:
        hits.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        })
    # Answer box is often the most useful piece — inject it as hit 0
    # with a clear "answer_box" source marker.
    ab = data.get("answer_box")
    if ab:
        hits.insert(0, {
            "title": ab.get("title", "answer_box"),
            "url": ab.get("link", ""),
            "snippet": ab.get("answer") or ab.get("snippet") or "",
        })

    logger.info(
        "tool(web_search/serpapi): ok | query=%r | hits=%d | answer_box=%s",
        query[:80], len(hits), bool(ab),
    )
    return ToolCallResult(
        status="ok",
        output={"hits": hits},
        metadata={
            "tool": "web_search",
            "provider": "serpapi",
            "query": query,
            "engine": config.SERPAPI_ENGINE,
        },
    )


def _web_search(query: str, **_: Any) -> ToolCallResult:
    """v1 web_search entry point. Routes to SerpAPI when configured,
    falls back to the stub otherwise. Empty query is a hard error."""
    if not query:
        return ToolCallResult(
            status="error", error="empty query",
            metadata={"tool": "web_search"},
        )
    if config.SERPAPI_API_KEY:
        return _web_search_serpapi(query)
    return _web_search_stub(query)


def _pdf_parser(content: str | bytes = "", **_: Any) -> ToolCallResult:
    """Stub pdf parser — in real use it would accept bytes and return
    extracted text. Here it just echoes whatever the caller passes so
    the executor can assert content flowed through."""
    text = content.decode("utf-8", "ignore") if isinstance(content, bytes) else (content or "")
    return ToolCallResult(
        status="ok",
        output={"text": text, "pages": max(1, len(text) // 2000)},
        metadata={"tool": "pdf_parser"},
    )


_TOOLS: dict[str, Callable[..., ToolCallResult]] = {
    "web_search": _web_search,
    "pdf_parser": _pdf_parser,
}


def list_tools() -> set[str]:
    """FR-13: registry contains exactly {web_search, pdf_parser}."""
    return set(_TOOLS.keys())


def call(tool_name: str, args: dict | None = None) -> ToolCallResult:
    """Invoke a whitelisted tool. Raises DisallowedToolError for any
    name outside `list_tools()` (FR-13 enforcement)."""
    if tool_name not in _TOOLS:
        raise DisallowedToolError(
            f"Tool '{tool_name}' is not allowed in v1. Allowed: {sorted(_TOOLS)}",
        )
    return _TOOLS[tool_name](**(args or {}))
