"""ИИ-платформа: per-user state, domain & document registry.

FR-P1..P6 — see SPEC.md.

Data model:
    PlatformUser
      model_id: str                          # picked LLM model
      active_domain: str | None              # currently selected domain for RAG
      domains: dict[str, Domain]             # name → Domain

    Domain
      name: str
      documents: list[Document]

    Document
      doc_id: str                            # 12-hex
      filename: str                          # original filename
      num_chunks: int                        # how many chunks landed in Qdrant
      added_at: str                          # ISO timestamp

The whole per-user store is persisted to ``data-persist/platform_store.pkl``
so users don't lose their domains between bot restarts. Qdrant collection per
domain is named ``platform-{tg_user_id}-{domain_safe_name}``.
"""

from __future__ import annotations

import os
import pickle
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import config


# ── Dataclasses ──────────────────────────────────────────────────

@dataclass
class Document:
    doc_id: str
    filename: str
    num_chunks: int
    added_at: str
    message_id: int | None = None  # FR-P18: tg message id of the upload (for source links)


@dataclass
class Domain:
    name: str
    documents: list[Document] = field(default_factory=list)


@dataclass
class PlatformUser:
    model_id: str = ""
    active_domains: set[str] = field(default_factory=set)  # FR-P9: multi-select
    domains: dict[str, Domain] = field(default_factory=dict)
    chat_history: list[dict] = field(default_factory=list)  # FR-P11
    last_answer: str = ""  # FR-P12: last LLM answer for "save to memory"

    # Backwards compat for pre-FR-P9 snapshots that stored `active_domain` str
    @property
    def active_domain(self) -> str | None:
        return next(iter(self.active_domains), None) if self.active_domains else None


# ── Storage ──────────────────────────────────────────────────────

_PLATFORM_STORE: dict[int, PlatformUser] = {}

_PERSIST_DIR = os.environ.get("DATA_PERSIST_DIR", "/app/data-persist")
_PLATFORM_FILE = os.path.join(_PERSIST_DIR, "platform_store.pkl")


def _persist() -> None:
    try:
        os.makedirs(_PERSIST_DIR, exist_ok=True)
        tmp = _PLATFORM_FILE + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(_PLATFORM_STORE, f)
        os.replace(tmp, _PLATFORM_FILE)
    except Exception:  # noqa: BLE001
        pass


def _migrate_user(u):
    """Backfill fields that were added after the pickle was written (FR-P9/P11/P12).
    Old snapshots may be missing `active_domains`, `chat_history`, `last_answer`.
    Also absorbs legacy `active_domain: str` into the new set."""
    if not hasattr(u, "active_domains") or u.active_domains is None:
        legacy = getattr(u, "active_domain", None)
        try:
            u.active_domains = {legacy} if legacy else set()
        except Exception:  # noqa: BLE001
            u.active_domains = set()
    if not hasattr(u, "chat_history") or u.chat_history is None:
        u.chat_history = []
    if not hasattr(u, "last_answer") or u.last_answer is None:
        u.last_answer = ""
    if not hasattr(u, "domains") or u.domains is None:
        u.domains = {}
    if not hasattr(u, "model_id") or u.model_id is None:
        u.model_id = ""
    # FR-P18: backfill Document.message_id on legacy pickles
    try:
        for dom in u.domains.values():
            for doc in getattr(dom, "documents", []) or []:
                if not hasattr(doc, "message_id"):
                    doc.message_id = None
    except Exception:  # noqa: BLE001
        pass
    return u


def load_platform_from_disk() -> None:
    """FR-P6: restore `_PLATFORM_STORE` from disk on bot startup."""
    try:
        if os.path.isfile(_PLATFORM_FILE):
            with open(_PLATFORM_FILE, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                _PLATFORM_STORE.clear()
                for k, v in data.items():
                    _PLATFORM_STORE[k] = _migrate_user(v)
    except Exception:  # noqa: BLE001
        pass


def get_user(tg_id: int) -> PlatformUser:
    """FR-P1: fetch or create the per-user platform state."""
    u = _PLATFORM_STORE.get(tg_id)
    if u is None:
        u = PlatformUser(model_id=config.PLATFORM_MODELS[0][1] if config.PLATFORM_MODELS else "")
        _PLATFORM_STORE[tg_id] = u
        _persist()
    else:
        _migrate_user(u)  # backfill fields that didn't exist in old pickles
    return u


def set_model(tg_id: int, model_id: str) -> None:
    """FR-P2: change the active LLM for a user."""
    u = get_user(tg_id)
    u.model_id = model_id
    _persist()


# ── Domains ──────────────────────────────────────────────────────

_DOMAIN_NAME_RE = re.compile(r"^[A-Za-zА-Яа-я0-9 _-]{1,40}$")


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name.lower())[:32]


def collection_name(tg_id: int, domain_name: str) -> str:
    """FR-P3: Qdrant collection naming per (user, domain)."""
    return f"platform-{tg_id}-{_safe_name(domain_name)}"


def create_domain(tg_id: int, name: str) -> Domain:
    """FR-P3: create a new memory domain. Auto-selects it (FR-P10)."""
    name = name.strip()
    if not _DOMAIN_NAME_RE.match(name):
        raise ValueError("Invalid domain name (1-40 chars, letters/digits/space/_-)")
    u = get_user(tg_id)
    if name in u.domains:
        u.active_domains.add(name)
        _persist()
        return u.domains[name]
    u.domains[name] = Domain(name=name)
    u.active_domains.add(name)  # FR-P10: auto-select newly created domain
    _persist()
    return u.domains[name]


def delete_domain(tg_id: int, name: str) -> bool:
    u = get_user(tg_id)
    if name not in u.domains:
        return False
    del u.domains[name]
    u.active_domains.discard(name)
    _persist()
    return True


def toggle_active_domain(tg_id: int, name: str) -> bool:
    """FR-P9: toggle a domain in the multi-select active set. Returns new state."""
    u = get_user(tg_id)
    if name not in u.domains:
        return False
    if name in u.active_domains:
        u.active_domains.discard(name)
        state = False
    else:
        u.active_domains.add(name)
        state = True
    _persist()
    return state


def set_active_domain(tg_id: int, name: str) -> None:
    """Legacy single-select setter — kept for backwards compat with old tests."""
    u = get_user(tg_id)
    if name in u.domains:
        u.active_domains = {name}
        _persist()


def get_active_domains(tg_id: int) -> list[str]:
    """FR-P9: ordered list of currently active domains for a user."""
    u = get_user(tg_id)
    return [d for d in u.domains if d in u.active_domains]


# ── FR-P11: chat history ─────────────────────────────────────────

def add_chat_message(tg_id: int, role: str, content: str) -> None:
    u = get_user(tg_id)
    u.chat_history.append({"role": role, "content": content})
    if role == "assistant":
        u.last_answer = content
    # Keep last 30 messages (~15 turns)
    if len(u.chat_history) > 30:
        u.chat_history = u.chat_history[-30:]
    _persist()


def reset_chat(tg_id: int) -> None:
    u = get_user(tg_id)
    u.chat_history.clear()
    u.last_answer = ""
    _persist()


def list_domains(tg_id: int) -> list[Domain]:
    return list(get_user(tg_id).domains.values())


# ── Documents ────────────────────────────────────────────────────

def register_document(
    tg_id: int,
    domain_name: str,
    filename: str,
    num_chunks: int,
    message_id: int | None = None,
) -> Document:
    """FR-P5: record a new document attached to a domain.
    `message_id` (FR-P18) is the tg message_id of the user's original upload — used to
    surface a clickable source reference in RAG answers. None for synthetic docs
    (e.g. «💾 Сохранить в память» from FR-P12 has no upload message)."""
    u = get_user(tg_id)
    domain = u.domains.get(domain_name)
    if domain is None:
        raise ValueError(f"Domain '{domain_name}' not found")
    doc = Document(
        doc_id=uuid.uuid4().hex[:12],
        filename=filename,
        num_chunks=num_chunks,
        added_at=datetime.utcnow().isoformat(timespec="seconds"),
        message_id=message_id,
    )
    domain.documents.append(doc)
    _persist()
    return doc


def delete_document(tg_id: int, domain_name: str, doc_id: str) -> bool:
    u = get_user(tg_id)
    domain = u.domains.get(domain_name)
    if domain is None:
        return False
    before = len(domain.documents)
    domain.documents = [d for d in domain.documents if d.doc_id != doc_id]
    if len(domain.documents) < before:
        _persist()
        return True
    return False


# ── v1 re-exports ────────────────────────────────────────────────
#
# Tests import as `import services.platform as svc` and expect the v1
# Memory Service + context resolver surface on the same module. Keep
# these imports at the bottom so there's no circular init ordering.

from services.memory import (  # noqa: E402
    MemoryObject,
    MemoryObjectVersion,
    create_memory_object,
    update_memory_object,
    list_versions,
    read_memory_object,
    set_alias,
    set_object_content,
    get_object_content,
    list_memory_objects,
)
from services.context_resolver import (  # noqa: E402
    ContextRef,
    ResolvedRef,
    AssembledContext,
    AssembledContextObject,
    parse_context_refs,
    resolve_context_ref,
    assemble_full_context,
)
