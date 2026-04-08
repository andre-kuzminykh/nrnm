"""v1 Context resolver — parse `[контекст]` refs and assemble full context.

## Трассируемость
Feature: Telegram AI Platform v1
Requirements: FR-4, FR-5, NFR-4, NFR-5, NFR-13, Rule 6

Handles three responsibilities:

1. **Parsing** — extract `[name]`, `[name@v2]`, `[контекст]` from a user
   message (FR-4).
2. **Resolving** — map a parsed ref to a concrete entity using the
   deterministic priority `exact -> alias -> fuzzy` (NFR-4). Ambiguous
   matches surface a disambiguation list instead of guessing.
3. **Assembly** — pull full content for a set of resolved ids and build
   a context bundle that can be injected directly into the LLM prompt
   (FR-5). For files that exceed the model's context window we use a
   map-reduce summarization pipeline: chunk -> per-chunk summary ->
   final summary (NFR-13). The resulting object is still `is_full=True`
   semantically but `used_summarization=True`.

The resolver intentionally looks across BOTH kinds of entities:
- legacy Documents (stored per-domain in `services.platform`)
- v1 MemoryObjects (stored in `services.memory`)

From the user's perspective "Память" is a single container (Rule 1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from services import memory as memory_svc


# ── Types ────────────────────────────────────────────────────────

@dataclass
class ContextRef:
    """A parsed reference from user text. `version=None` means "latest"."""
    name: str
    version: str | None = None


@dataclass
class ResolvedRef:
    """Outcome of resolving a single `ContextRef` against the user's Memory.

    `matched` is either a Document (from services.platform) or a
    MemoryObject (from services.memory) when resolution is unambiguous.
    When `needs_disambiguation=True`, `matched` is None and `candidates`
    holds every entity the resolver considered equally likely.
    """
    matched: object | None
    match_kind: str  # "exact" | "alias" | "fuzzy" | "none"
    candidates: list = field(default_factory=list)
    needs_disambiguation: bool = False


@dataclass
class AssembledContextObject:
    """One entry in the assembled context bundle. `is_full=True` means
    the caller asked for the *whole* object (not a RAG top-k excerpt).
    `used_summarization=True` means we had to run map-reduce to fit
    within `max_chars` (NFR-13)."""
    doc_id: str
    filename: str
    content: str
    is_full: bool
    used_summarization: bool = False


@dataclass
class AssembledContext:
    objects: list[AssembledContextObject]


# ── FR-4: parsing ────────────────────────────────────────────────

# Accept:
#   [контекст]           — placeholder for "whichever file the user meant"
#   [report.pdf]         — latest version
#   [report.pdf@v2]      — explicit version
#   [Quarterly report]   — multi-word names OK (the ] closes it)
_REF_RE = re.compile(r"\[([^\[\]]+?)\]")


def parse_context_refs(text: str) -> list[ContextRef]:
    """Extract every `[...]` reference from the text in order of appearance."""
    refs: list[ContextRef] = []
    for match in _REF_RE.finditer(text or ""):
        raw = match.group(1).strip()
        if not raw:
            continue
        if "@" in raw:
            name, ver = raw.rsplit("@", 1)
            refs.append(ContextRef(name=name.strip(), version=ver.strip()))
        else:
            refs.append(ContextRef(name=raw))
    return refs


# ── NFR-4: resolver ──────────────────────────────────────────────

def _collect_user_entities(tg_id: int) -> list:
    """Gather every addressable entity for a user: legacy Documents +
    v1 MemoryObjects. Both carry `.filename` so downstream logic is
    uniform."""
    from services import platform as platform_svc  # local import avoids cycle

    entities: list = []
    user = platform_svc.get_user(tg_id)
    for domain in user.domains.values():
        entities.extend(domain.documents)

    for obj in memory_svc.list_memory_objects(tg_id):
        if obj.filename:  # only addressable objects get indexed by name
            entities.append(obj)
    return entities


def _entity_id(entity) -> str:
    """Unified id across Document and MemoryObject."""
    return getattr(entity, "doc_id", None) or getattr(entity, "memory_object_id", "")


def resolve_context_ref(tg_id: int, name: str) -> ResolvedRef:
    """Resolve a bare name to a concrete entity.

    NFR-4: deterministic order **exact → alias → fuzzy**. The search
    short-circuits as soon as it finds an unambiguous match; ambiguous
    matches at the same level are surfaced via `needs_disambiguation`.
    """
    entities = _collect_user_entities(tg_id)
    if not entities:
        return ResolvedRef(matched=None, match_kind="none")

    # 1. Exact filename match
    exact = [e for e in entities if getattr(e, "filename", None) == name]
    if len(exact) == 1:
        return ResolvedRef(matched=exact[0], match_kind="exact")
    if len(exact) > 1:
        return ResolvedRef(
            matched=None, match_kind="exact",
            candidates=exact, needs_disambiguation=True,
        )

    # 2. Alias match
    aliases = memory_svc.get_aliases(tg_id)
    if name in aliases:
        target_id = aliases[name]
        for e in entities:
            if _entity_id(e) == target_id:
                return ResolvedRef(matched=e, match_kind="alias")

    # 3. Fuzzy (substring, case-insensitive)
    lower = name.lower()
    fuzzy = [e for e in entities if lower in (getattr(e, "filename", "") or "").lower()]
    if len(fuzzy) == 1:
        return ResolvedRef(matched=fuzzy[0], match_kind="fuzzy")
    if len(fuzzy) > 1:
        return ResolvedRef(
            matched=None, match_kind="fuzzy",
            candidates=fuzzy, needs_disambiguation=True,
        )

    return ResolvedRef(matched=None, match_kind="none")


# ── FR-5 / NFR-13: assembly + map-reduce summarization ───────────

_DEFAULT_MAX_CHARS = 20_000


def assemble_full_context(
    tg_id: int,
    doc_ids: Iterable[str],
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> AssembledContext:
    """Build a context bundle for the given entity ids.

    For each entity:
    - Pull its full text via `memory.get_object_content()`.
    - If the text fits within `max_chars`, include it as-is
      (`used_summarization=False`).
    - Otherwise run `_mapreduce_summarize()` which chunks the text,
      summarises each chunk, and merges the summaries until the
      final output fits (NFR-13).
    """
    # Build an id -> filename lookup across both stores
    filename_by_id: dict[str, str] = {}
    for e in _collect_user_entities(tg_id):
        filename_by_id[_entity_id(e)] = getattr(e, "filename", "") or ""

    objects: list[AssembledContextObject] = []
    for doc_id in doc_ids:
        raw = memory_svc.get_object_content(doc_id)
        filename = filename_by_id.get(doc_id, "")
        if len(raw) <= max_chars:
            objects.append(AssembledContextObject(
                doc_id=doc_id, filename=filename, content=raw,
                is_full=True, used_summarization=False,
            ))
            continue

        summary = _mapreduce_summarize(raw, max_chars=max_chars)
        objects.append(AssembledContextObject(
            doc_id=doc_id, filename=filename, content=summary,
            is_full=True, used_summarization=True,
        ))

    return AssembledContext(objects=objects)


# ── NFR-13: map-reduce summarization ─────────────────────────────

def _mapreduce_summarize(text: str, *, max_chars: int) -> str:
    """Map-reduce summarization that is deterministic and LLM-free.

    Strategy (kept simple on purpose):
    1. **Map** — split the source into ~`chunk_size` char chunks so that
       the per-chunk output is small enough.
    2. **Reduce** — for each chunk, take the first and last sentence-ish
       slice and concatenate; this preserves anchor points without
       calling out to an LLM (which would make tests non-deterministic).
    3. **Merge** — if the concatenated summaries are still over budget,
       recurse until we fit.

    When an LLM provider is wired in (see `_llm_summarize_chunk`), this
    becomes a real summariser; the public contract — `used_summarization`
    and `len <= max_chars` — stays identical.
    """
    if len(text) <= max_chars:
        return text

    # Target per-chunk summary length so that the total stays below budget.
    target_chunks = max(4, len(text) // max(1, max_chars) + 2)
    chunk_size = max(200, len(text) // target_chunks)

    summaries: list[str] = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        summaries.append(_summarize_chunk(chunk))

    merged = " ".join(summaries).strip()
    if len(merged) <= max_chars:
        return merged
    # Second reduce pass — trim the merged result to the budget, keeping
    # the start and end so the summary still has shape.
    head_len = max_chars // 2 - 50
    tail_len = max_chars - head_len - 5  # 5 = "\n...\n"
    return f"{merged[:head_len]}\n...\n{merged[-tail_len:]}"


def _summarize_chunk(chunk: str) -> str:
    """Extractive fallback: first N chars + last N chars of the chunk.

    This keeps the pipeline LLM-free for tests and CI. To upgrade to a
    real summariser, swap this function for an async LLM call — the
    caller is already structured so it'll work as-is for short files
    and trigger summarization only for oversize ones.
    """
    budget = 300
    if len(chunk) <= budget:
        return chunk.strip()
    head = chunk[: budget // 2].strip()
    tail = chunk[-budget // 2 :].strip()
    return f"{head} … {tail}"
