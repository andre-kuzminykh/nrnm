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

from dataclasses import dataclass, field
from typing import Any, Callable


class DisallowedToolError(RuntimeError):
    """Raised when the Executor tries to invoke a tool that isn't in the
    v1 whitelist (FR-13)."""


@dataclass
class ToolCallResult:
    status: str  # "ok" | "error"
    output: Any = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)


# ── Registry ─────────────────────────────────────────────────────

def _web_search(query: str, **_: Any) -> ToolCallResult:
    """Stub web search — returns a deterministic "ok" result with a fake
    hit list so the executor + verification pipeline can run end-to-end
    without hitting the network. Swap for a real SerpAPI/Brave client
    when ready."""
    if not query:
        return ToolCallResult(status="error", error="empty query")
    return ToolCallResult(
        status="ok",
        output={
            "hits": [
                {"title": f"Result for {query}", "url": "https://example.com/1"},
                {"title": f"Secondary {query}", "url": "https://example.com/2"},
            ],
        },
        metadata={"tool": "web_search", "query": query},
    )


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
