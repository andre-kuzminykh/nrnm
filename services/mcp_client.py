"""MCP client — dispatches a tool call to a specific MCP entry.

## Трассируемость
Feature: Telegram AI Platform v1.1
Requirements: FR-26, NFR-15

Takes an `MCPEntry` (from services.mcp_registry) and dispatches the
call based on the URL scheme:

- `builtin://serpapi`    → existing in-process _web_search_serpapi
- `builtin://pdf_parser` → existing in-process _pdf_parser
- `http://` / `https://` → POST `{tool, args}` to the MCP server
                           with `Authorization: Bearer <token>` header

All paths return a `tools.ToolCallResult` so the executor doesn't
care which transport was used.

NFR-15: HTTP / network / parse errors never raise out of this module.
They degrade into `ToolCallResult(status="error")` with a readable
error message — the executor treats that as a Rule-5 tool_failure
trigger and the replanning engine takes over (FR-16).
"""

from __future__ import annotations

import logging
from typing import Any

from services import tools as tool_layer
from services.mcp_registry import MCPEntry

logger = logging.getLogger(__name__)


# Map of builtin URLs to the in-process functions they proxy to.
# Keeping this as a lookup makes it easy to add more builtins
# (builtin://file_reader, builtin://calculator, ...) without
# touching the dispatch logic.
_BUILTIN_ROUTES: dict[str, str] = {
    "builtin://serpapi": "web_search",
    "builtin://pdf_parser": "pdf_parser",
}


def dispatch(entry: MCPEntry, args: dict[str, Any] | None = None) -> tool_layer.ToolCallResult:
    """Invoke the given MCP entry with the supplied args.

    Never raises — every failure path returns a ToolCallResult with
    status='error'. The executor interprets that as FR-16 trigger #3
    (tool_failure) and fires a replan.
    """
    args = dict(args or {})

    # 1. Builtin routing — map to the existing hardcoded tool_layer
    #    implementations so we don't duplicate SerpAPI / pdf code.
    if entry.url in _BUILTIN_ROUTES:
        legacy_name = _BUILTIN_ROUTES[entry.url]
        logger.info(
            "mcp: builtin dispatch %s -> %s | args=%s",
            entry.name, legacy_name, list(args.keys()),
        )
        try:
            result = tool_layer.call(legacy_name, args)
            # Tag the result with the MCP entry name so traces are
            # self-describing even when the backend is a builtin.
            new_meta = dict(result.metadata)
            new_meta["mcp_name"] = entry.name
            new_meta["mcp_url"] = entry.url
            return tool_layer.ToolCallResult(
                status=result.status,
                output=result.output,
                error=result.error,
                metadata=new_meta,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("mcp builtin %s failed", entry.name)
            return tool_layer.ToolCallResult(
                status="error",
                error=f"builtin {entry.name}: {exc}",
                metadata={"tool": entry.name, "mcp_name": entry.name, "mcp_url": entry.url},
            )

    # 2. HTTP dispatch
    if entry.url.startswith("http://") or entry.url.startswith("https://"):
        return _dispatch_http(entry, args)

    return tool_layer.ToolCallResult(
        status="error",
        error=f"unknown MCP scheme: {entry.url}",
        metadata={"tool": entry.name, "mcp_name": entry.name, "mcp_url": entry.url},
    )


def _dispatch_http(entry: MCPEntry, args: dict[str, Any]) -> tool_layer.ToolCallResult:
    """POST the tool call to an external MCP server.

    Wire protocol (v1, intentionally minimal):

        POST {entry.url}
        Authorization: Bearer {entry.token}
        Content-Type: application/json

        { "tool": "<entry.name>", "args": { ... } }

    Response (expected):
        200 OK
        { "ok": true, "data": <any>, ... }
        or
        { "ok": false, "error": "..." }

    A real MCP server conforming to the Anthropic Model Context
    Protocol JSON-RPC spec will need a small adapter — swap
    `_dispatch_http` for a JSON-RPC transport when the server side
    moves that way. The executor contract doesn't change.
    """
    try:
        import httpx
    except Exception as exc:  # noqa: BLE001
        return tool_layer.ToolCallResult(
            status="error",
            error=f"httpx unavailable: {exc}",
            metadata={"tool": entry.name, "mcp_name": entry.name, "mcp_url": entry.url},
        )

    headers = {"Content-Type": "application/json"}
    if entry.token:
        headers["Authorization"] = f"Bearer {entry.token}"

    payload = {"tool": entry.name, "args": args}

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(entry.url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        logger.warning("mcp http %s: %s", entry.name, exc)
        return tool_layer.ToolCallResult(
            status="error",
            error=f"mcp http {entry.name}: {exc}",
            metadata={"tool": entry.name, "mcp_name": entry.name, "mcp_url": entry.url},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("mcp unexpected %s: %s", entry.name, exc)
        return tool_layer.ToolCallResult(
            status="error",
            error=f"mcp {entry.name}: {exc}",
            metadata={"tool": entry.name, "mcp_name": entry.name, "mcp_url": entry.url},
        )

    ok = bool(data.get("ok", True))
    if not ok:
        return tool_layer.ToolCallResult(
            status="error",
            error=str(data.get("error") or "mcp reported failure"),
            metadata={"tool": entry.name, "mcp_name": entry.name, "mcp_url": entry.url},
        )

    logger.info(
        "mcp http %s: ok | fields=%s",
        entry.name, list(data.keys())[:10],
    )
    return tool_layer.ToolCallResult(
        status="ok",
        output=data.get("data", data),
        metadata={"tool": entry.name, "mcp_name": entry.name, "mcp_url": entry.url},
    )
