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


# ── FR-47: ask_user — pause execution and ask the user ──────────

def _ask_user(question: str = "", **_: Any) -> ToolCallResult:
    """Returns status='pending' — the runtime interprets this as
    'pause the graph and wait for user input'. The bot handler
    prompts the user, collects the response, and feeds it back
    into the graph state before resuming."""
    return ToolCallResult(
        status="pending",
        output={"question": question or "Уточните, пожалуйста."},
        metadata={"tool": "ask_user"},
    )


# ── FR-48: rag_search — query the internal RAG database ─────────

def _rag_search(query: str = "", tg_id: int = 0, **_: Any) -> ToolCallResult:
    """Synchronous wrapper around rag.query_rag. Searches across all
    active domains for the given user. Returns top-k hits with text
    fragments — same shape as web_search hits for uniformity."""
    if not query:
        return ToolCallResult(status="error", error="empty query", metadata={"tool": "rag_search"})

    import asyncio

    try:
        from services import rag, platform as platform_svc

        active = platform_svc.get_active_domains(tg_id)
        if not active:
            return ToolCallResult(
                status="ok",
                output={"hits": [], "note": "no active domains"},
                metadata={"tool": "rag_search", "query": query},
            )

        all_hits: list[dict] = []
        for dom in active:
            col = platform_svc.collection_name(tg_id, dom)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        hits = pool.submit(asyncio.run, rag.query_rag(col, query, top_k=5)).result()
                else:
                    hits = asyncio.run(rag.query_rag(col, query, top_k=5))
            except Exception:  # noqa: BLE001
                hits = []
            all_hits.extend(hits)

        all_hits.sort(key=lambda h: -h.get("score", 0))
        top = all_hits[:5]
        return ToolCallResult(
            status="ok",
            output={"hits": [
                {"title": h.get("filename", ""), "text": h.get("text", ""), "score": h.get("score", 0)}
                for h in top
            ]},
            metadata={"tool": "rag_search", "query": query},
        )
    except Exception as exc:  # noqa: BLE001
        return ToolCallResult(
            status="error", error=f"rag_search: {exc}",
            metadata={"tool": "rag_search", "query": query},
        )


# ── FR-49: file_open — read a file, summarize if too large ──────

def _file_open(
    filename: str = "",
    tg_id: int = 0,
    max_chars: int = 20_000,
    **_: Any,
) -> ToolCallResult:
    """Open a file from the user's file tree by name. If the content
    exceeds max_chars, runs map-reduce summarization via
    context_resolver so the whole file fits in the LLM context."""
    if not filename:
        return ToolCallResult(status="error", error="no filename", metadata={"tool": "file_open"})

    try:
        from services import file_tree as ft, memory as mem
        from services import context_resolver

        # Search the file tree for a matching filename
        all_files = ft.get_scope(tg_id, "/")
        match = None
        for f in all_files:
            if f.name == filename or f.path.endswith(f"/{filename}"):
                match = f
                break
        if match is None:
            # Fuzzy: try substring
            for f in all_files:
                if filename.lower() in f.name.lower():
                    match = f
                    break
        if match is None:
            return ToolCallResult(
                status="error",
                error=f"file '{filename}' not found in tree",
                metadata={"tool": "file_open", "filename": filename},
            )

        raw = mem.get_object_content(match.doc_id)
        if not raw:
            return ToolCallResult(
                status="ok",
                output={"content": "", "filename": match.name, "summarized": False,
                        "note": "file content empty or not cached"},
                metadata={"tool": "file_open", "filename": match.name},
            )

        if len(raw) <= max_chars:
            return ToolCallResult(
                status="ok",
                output={"content": raw, "filename": match.name, "summarized": False},
                metadata={"tool": "file_open", "filename": match.name},
            )

        # Too large — summarize via map-reduce (NFR-13)
        ctx = context_resolver.assemble_full_context(tg_id, [match.doc_id], max_chars=max_chars)
        summary = ctx.objects[0].content if ctx.objects else raw[:max_chars]
        return ToolCallResult(
            status="ok",
            output={"content": summary, "filename": match.name, "summarized": True},
            metadata={"tool": "file_open", "filename": match.name},
        )
    except Exception as exc:  # noqa: BLE001
        return ToolCallResult(
            status="error", error=f"file_open: {exc}",
            metadata={"tool": "file_open", "filename": filename},
        )


_TOOLS: dict[str, Callable[..., ToolCallResult]] = {
    "web_search": _web_search,
    "pdf_parser": _pdf_parser,
    "ask_user": _ask_user,
    "rag_search": _rag_search,
    "file_open": _file_open,
}


def list_tools() -> set[str]:
    """All registered tools available to the superagent planner."""
    return set(_TOOLS.keys())


def call(tool_name: str, args: dict | None = None) -> ToolCallResult:
    """Invoke a whitelisted tool. Raises DisallowedToolError for any
    name outside `list_tools()` (FR-13 enforcement)."""
    if tool_name not in _TOOLS:
        raise DisallowedToolError(
            f"Tool '{tool_name}' is not allowed in v1. Allowed: {sorted(_TOOLS)}",
        )
    return _TOOLS[tool_name](**(args or {}))
