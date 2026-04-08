"""MCP Registry — CRUD for per-domain tool entries.

## Трассируемость
Feature: Telegram AI Platform v1.1
Requirements: FR-23, FR-24, FR-25

Thin facade over `services.platform.Domain.mcps`. The MCP list lives
on the Domain itself (so it persists via the same pickle as documents
and survives bot restarts without a separate store) — this module
just wraps it with an intentional, narrow API so the bot handlers and
planner don't poke at dataclass fields directly.

CRUD operations:
- `add_mcp()` — validated insert
- `update_mcp()` — partial update (None means "don't change")
- `delete_mcp()` — remove by name
- `get_mcp()` / `list_mcps()` — lookup

All mutations call `platform._persist()` so state survives restarts.
Validation is deliberately loose: names must be non-empty and alphanumeric-ish,
URLs must start with `builtin://` or `http(s)://`, descriptions are free text.
Stricter validation is a follow-up when we add the Qdrant-backed
semantic tool search.
"""

from __future__ import annotations

import re
from datetime import datetime

from services import platform as platform_svc
from services.platform import MCPEntry


# Re-export MCPEntry so callers can `from services.mcp_registry import MCPEntry`
__all__ = [
    "MCPEntry",
    "add_mcp",
    "update_mcp",
    "delete_mcp",
    "get_mcp",
    "list_mcps",
]


_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_\-]{0,39}$")


class MCPRegistryError(ValueError):
    """Raised on invalid MCP name / URL / duplicate registration."""


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _validate(name: str, url: str) -> None:
    if not _NAME_RE.match(name or ""):
        raise MCPRegistryError(
            f"Invalid MCP name {name!r}. 1-40 chars, letters/digits/_/-.",
        )
    if not (url.startswith("builtin://") or url.startswith("http://") or url.startswith("https://")):
        raise MCPRegistryError(
            f"Invalid MCP url {url!r}. Must start with builtin://, http://, or https://",
        )


def _get_domain(tg_id: int, domain_name: str):
    user = platform_svc.get_user(tg_id)
    dom = user.domains.get(domain_name)
    if dom is None:
        raise MCPRegistryError(f"Domain {domain_name!r} not found")
    # Belt-and-suspenders: if the domain predates FR-24, backfill its
    # mcps list so tests / UI never see `None` there.
    if not hasattr(dom, "mcps") or dom.mcps is None:
        dom.mcps = platform_svc._default_mcp_bootstrap()
        platform_svc._persist()
    return dom


def add_mcp(
    tg_id: int,
    domain_name: str,
    *,
    name: str,
    url: str,
    token: str,
    description: str,
) -> MCPEntry:
    """FR-23 create. Fails loudly on duplicate names within the domain."""
    _validate(name, url)
    dom = _get_domain(tg_id, domain_name)
    if any(m.name == name for m in dom.mcps):
        raise MCPRegistryError(
            f"MCP {name!r} already exists in domain {domain_name!r}",
        )
    now = _now()
    entry = MCPEntry(
        name=name,
        url=url,
        token=token or "",
        description=description or "",
        created_at=now,
        updated_at=now,
    )
    dom.mcps.append(entry)
    platform_svc._persist()
    return entry


def update_mcp(
    tg_id: int,
    domain_name: str,
    name: str,
    *,
    url: str | None = None,
    token: str | None = None,
    description: str | None = None,
) -> MCPEntry:
    """FR-23 edit. `None` on a field means "leave it alone"."""
    dom = _get_domain(tg_id, domain_name)
    for i, entry in enumerate(dom.mcps):
        if entry.name == name:
            new_url = url if url is not None else entry.url
            if url is not None:
                _validate(name, new_url)
            updated = MCPEntry(
                name=entry.name,
                url=new_url,
                token=token if token is not None else entry.token,
                description=description if description is not None else entry.description,
                created_at=entry.created_at,
                updated_at=_now(),
            )
            dom.mcps[i] = updated
            platform_svc._persist()
            return updated
    raise MCPRegistryError(
        f"MCP {name!r} not found in domain {domain_name!r}",
    )


def delete_mcp(tg_id: int, domain_name: str, name: str) -> bool:
    """FR-23 delete. Returns True if something was removed."""
    dom = _get_domain(tg_id, domain_name)
    before = len(dom.mcps)
    dom.mcps = [m for m in dom.mcps if m.name != name]
    if len(dom.mcps) < before:
        platform_svc._persist()
        return True
    return False


def get_mcp(tg_id: int, domain_name: str, name: str) -> MCPEntry | None:
    try:
        dom = _get_domain(tg_id, domain_name)
    except MCPRegistryError:
        return None
    for m in dom.mcps:
        if m.name == name:
            return m
    return None


def list_mcps(tg_id: int, domain_name: str) -> list[MCPEntry]:
    """Snapshot of the domain's MCP registry in insertion order."""
    try:
        dom = _get_domain(tg_id, domain_name)
    except MCPRegistryError:
        return []
    return list(dom.mcps)
