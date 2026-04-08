"""v1 Memory Service — unified, versioned memory objects.

## Трассируемость
Feature: Telegram AI Platform v1
Requirements: FR-1, FR-2, FR-3, NFR-1, NFR-2, NFR-3, NFR-5, Rule 6

Parallels to the legacy Domain/Document model in `services.platform`:
— legacy Documents live inside `PlatformUser.domains[*].documents` and
  back the RAG chat;
— v1 MemoryObjects live in this module's in-process store and add
  version history, idempotent saves, aliases, and content access for
  the new `[контекст]` machinery.

Both kinds are addressable by their own ID and both are first-class
citizens of the resolver in `services.context_resolver`. To the user
it looks like a single "Память" container (Rule 1 — Unified Memory).

Storage is intentionally in-process / pickle-free here: for the v1
cut we only need it to be consistent within a running bot and across
test runs (conftest.py wipes the module-level dicts). Persistence
across restarts is a follow-up.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable


# ── Types ────────────────────────────────────────────────────────

@dataclass
class MemoryObjectVersion:
    """One immutable version of a MemoryObject. `version` is `v1`, `v2`, ..."""
    version: str
    content: str
    created_at: str


@dataclass
class MemoryObject:
    """A versioned memory object. The whole chain of versions lives in
    `versions`; `version` / `content` always point at the latest one."""
    memory_object_id: str
    tg_id: int
    kind: str  # "note" | "file" | "answer"
    versions: list[MemoryObjectVersion]
    filename: str | None = None

    @property
    def version(self) -> str:
        return self.versions[-1].version

    @property
    def content(self) -> str:
        return self.versions[-1].content


# ── Storage ──────────────────────────────────────────────────────

# { tg_id -> { memory_object_id -> MemoryObject } }
_MEMORY_STORE: dict[int, dict[str, MemoryObject]] = {}

# { tg_id -> { alias -> entity_id } }   — entity_id is doc_id OR memory_object_id
_ALIASES: dict[int, dict[str, str]] = {}

# { (tg_id, client_request_id) -> memory_object_id } — NFR-1 idempotency
_IDEMPOTENCY: dict[tuple[int, str], str] = {}

# { entity_id -> full text }  — attached content for legacy Documents AND
# new MemoryObjects (FR-5 full-context assembly pulls from here).
# For Documents the ingest pipeline is free to skip this and keep chunks
# in Qdrant only; `assemble_full_context` degrades gracefully when empty.
_OBJECT_CONTENT: dict[str, str] = {}


def _reset_stores() -> None:
    """Used by conftest to wipe v1 state between tests."""
    _MEMORY_STORE.clear()
    _ALIASES.clear()
    _IDEMPOTENCY.clear()
    _OBJECT_CONTENT.clear()


# ── Helpers ──────────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _fingerprint(tg_id: int, kind: str, content: str) -> str:
    """Stable hash used as a fallback idempotency key for content-identical
    saves when the caller hasn't supplied a client_request_id."""
    h = hashlib.sha256()
    h.update(f"{tg_id}::{kind}::".encode("utf-8"))
    h.update(content.encode("utf-8"))
    return h.hexdigest()


# ── FR-3 / NFR-1 / NFR-2 / Rule 6 — create + version ─────────────

def create_memory_object(
    tg_id: int,
    *,
    kind: str,
    content: str,
    filename: str | None = None,
    client_request_id: str | None = None,
) -> MemoryObject:
    """Create a new MemoryObject at version `v1`.

    NFR-1 — if `client_request_id` is supplied and already seen for this
    user, return the previously created object instead of duplicating.
    """
    if client_request_id is not None:
        key = (tg_id, client_request_id)
        existing_id = _IDEMPOTENCY.get(key)
        if existing_id is not None:
            return _MEMORY_STORE[tg_id][existing_id]

    obj_id = uuid.uuid4().hex[:12]
    obj = MemoryObject(
        memory_object_id=obj_id,
        tg_id=tg_id,
        kind=kind,
        filename=filename,
        versions=[MemoryObjectVersion(version="v1", content=content, created_at=_now())],
    )
    _MEMORY_STORE.setdefault(tg_id, {})[obj_id] = obj
    _OBJECT_CONTENT[obj_id] = content

    if client_request_id is not None:
        _IDEMPOTENCY[(tg_id, client_request_id)] = obj_id
    return obj


def update_memory_object(memory_object_id: str, *, content: str) -> MemoryObject:
    """Append a new version to an existing MemoryObject (Rule 6 — old
    versions must stay readable)."""
    obj = _find_object(memory_object_id)
    next_version = f"v{len(obj.versions) + 1}"
    obj.versions.append(MemoryObjectVersion(
        version=next_version, content=content, created_at=_now(),
    ))
    _OBJECT_CONTENT[memory_object_id] = content
    return obj


def list_versions(memory_object_id: str) -> list[MemoryObjectVersion]:
    """Return the full version chain in insertion order (v1 first)."""
    obj = _find_object(memory_object_id)
    return list(obj.versions)


def read_memory_object(
    memory_object_id: str, *, version: str | None = None,
) -> MemoryObjectVersion:
    """NFR-5 — explicit `version` returns exactly that version; omitted
    returns latest."""
    obj = _find_object(memory_object_id)
    if version is None:
        return obj.versions[-1]
    for v in obj.versions:
        if v.version == version:
            return v
    raise KeyError(f"{memory_object_id}@{version}")


def _find_object(memory_object_id: str) -> MemoryObject:
    for user_objs in _MEMORY_STORE.values():
        if memory_object_id in user_objs:
            return user_objs[memory_object_id]
    raise KeyError(memory_object_id)


# ── Aliases — used by the resolver (NFR-4) ───────────────────────

def set_alias(tg_id: int, entity_id: str, alias: str) -> None:
    """Register an alias -> entity_id mapping for this user. `entity_id`
    can be either a MemoryObject id OR a legacy Document doc_id — the
    resolver handles both."""
    _ALIASES.setdefault(tg_id, {})[alias] = entity_id


def get_aliases(tg_id: int) -> dict[str, str]:
    return dict(_ALIASES.get(tg_id, {}))


# ── Object content accessors (FR-5 / NFR-13) ─────────────────────

def set_object_content(entity_id: str, content: str) -> None:
    """Attach full text for a legacy Document or a MemoryObject.

    Used by:
    - the ingest pipeline, to cache the extracted file body alongside
      Qdrant-chunked embeddings;
    - tests, to stage a deterministic body for assemble_full_context.
    """
    _OBJECT_CONTENT[entity_id] = content


def get_object_content(entity_id: str) -> str:
    return _OBJECT_CONTENT.get(entity_id, "")


def list_memory_objects(tg_id: int) -> list[MemoryObject]:
    return list(_MEMORY_STORE.get(tg_id, {}).values())
