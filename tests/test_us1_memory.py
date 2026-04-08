"""US-1 — Unified Memory (FR-1..FR-5, NFR-1..NFR-5).

Каждый тест закреплён за конкретным требованием из `SPEC.md`. Тесты,
требующие новых сервисов (Memory Service с version history, context
resolver для `[контекст]`), помечены `pytest.skip(...)` с указанием ID, —
они образуют spec-driven roadmap и автоматически зелёнеют, когда новый
модуль появится.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.spec


# ─────────────────────────────────────────────────────────────────
# FR-1 — unified Memory как единый контейнер
# ─────────────────────────────────────────────────────────────────

def test_fr_1_memory_is_a_single_container_per_user(platform_svc, tg_id):
    """FR-1: Память — один контейнер, из которого пользователь видит всё,
    что он когда-либо сохранил (файлы, заметки, сохранённые ответы)."""
    user = platform_svc.get_user(tg_id)
    # В legacy-модели один PlatformUser владеет ровно одним `domains`
    # неймспейсом, который и является «Памятью» для пользователя.
    assert hasattr(user, "domains"), "User must expose a single Memory namespace"
    assert user.domains == {}, "Fresh user starts with an empty Memory"


def test_fr_1_memory_holds_all_object_kinds(platform_svc, tg_id):
    """FR-1: Память должна хранить файлы и текстовые объекты вместе,
    без разделения на отдельные модули «файлы» / «домены» для пользователя."""
    platform_svc.create_domain(tg_id, "inbox")
    # файл
    platform_svc.register_document(tg_id, "inbox", "report.pdf", num_chunks=4)
    # «заметка» / сохранённый ответ — в legacy реализации тоже
    # приземляется в тот же контейнер как Document:
    platform_svc.register_document(tg_id, "inbox", "saved-note.txt", num_chunks=1)

    docs = platform_svc.get_user(tg_id).domains["inbox"].documents
    filenames = [d.filename for d in docs]
    assert "report.pdf" in filenames
    assert "saved-note.txt" in filenames


# ─────────────────────────────────────────────────────────────────
# FR-2 — загрузка файлов и сохранение текстов в Память
# ─────────────────────────────────────────────────────────────────

def test_fr_2_upload_file_to_memory(platform_svc, tg_id):
    """FR-2: пользователь может сохранить файл в Память."""
    platform_svc.create_domain(tg_id, "research")
    doc = platform_svc.register_document(
        tg_id, "research", "paper.pdf", num_chunks=12, message_id=42,
    )
    assert doc.filename == "paper.pdf"
    assert doc.num_chunks == 12
    assert doc.message_id == 42


def test_fr_2_save_text_object_to_memory(platform_svc, tg_id):
    """FR-2: пользователь может сохранить текстовый объект (ответ,
    заметку) в Память — API совпадает с файлами."""
    platform_svc.create_domain(tg_id, "notes")
    doc = platform_svc.register_document(tg_id, "notes", "idea-1.txt", num_chunks=1)
    assert doc.doc_id
    assert doc.added_at  # ISO timestamp


# ─────────────────────────────────────────────────────────────────
# FR-3 — автоматическое создание v1 при сохранении
# ─────────────────────────────────────────────────────────────────

def test_fr_3_first_save_creates_version_v1(platform_svc, tg_id):
    """FR-3: первая запись объекта должна автоматически получить
    `version = v1`. В текущей реализации нет MemoryObjectVersion, поэтому
    тест skip-нут до появления Memory Service."""
    if not hasattr(platform_svc, "create_memory_object"):
        pytest.skip("TODO FR-3: Memory Service with version history not implemented")
    obj = platform_svc.create_memory_object(  # type: ignore[attr-defined]
        tg_id, kind="note", content="hello",
    )
    assert obj.version == "v1"
    assert obj.memory_object_id


def test_fr_3_rule6_update_creates_new_version_preserving_old(platform_svc, tg_id):
    """FR-3 + Rule 6: обновление объекта должно создавать новую версию,
    при этом старая версия обязана остаться доступной (`list_versions`
    возвращает обе). Эта инвариантность — ключ к NFR-5 и к метрике
    «100% файлов в Памяти имеют version history»."""
    if not hasattr(platform_svc, "create_memory_object"):
        pytest.skip("TODO FR-3 / Rule 6: version-preserving update not implemented")
    obj = platform_svc.create_memory_object(tg_id, kind="note", content="v1-body")  # type: ignore[attr-defined]
    platform_svc.update_memory_object(obj.memory_object_id, content="v2-body")  # type: ignore[attr-defined]

    versions = platform_svc.list_versions(obj.memory_object_id)  # type: ignore[attr-defined]
    assert [v.version for v in versions] == ["v1", "v2"]
    # Старая версия должна быть читаема после апдейта.
    v1 = platform_svc.read_memory_object(obj.memory_object_id, version="v1")  # type: ignore[attr-defined]
    assert v1.content == "v1-body"


# ─────────────────────────────────────────────────────────────────
# FR-4 — явное подключение объектов через `[контекст]`
# ─────────────────────────────────────────────────────────────────

def test_fr_4_parse_context_reference_syntax(platform_svc):
    """FR-4: парсер должен извлекать `[контекст]`, `[file_name]`,
    `[file_name@v2]` из сообщения пользователя."""
    if not hasattr(platform_svc, "parse_context_refs"):
        pytest.skip("TODO FR-4: context ref parser not implemented")
    refs = platform_svc.parse_context_refs(  # type: ignore[attr-defined]
        "Summarise [report.pdf] using [notes.txt@v2] and the default [контекст]",
    )
    names = [r.name for r in refs]
    assert "report.pdf" in names
    assert "notes.txt" in names
    assert any(r.version == "v2" for r in refs)


def test_fr_4_unknown_ref_triggers_disambiguation(platform_svc, tg_id):
    """FR-4: если имя не совпадает однозначно, resolver должен
    возвращать список кандидатов для disambiguation."""
    if not hasattr(platform_svc, "resolve_context_ref"):
        pytest.skip("TODO FR-4: context ref resolver not implemented")
    platform_svc.create_domain(tg_id, "default")
    platform_svc.register_document(tg_id, "default", "report-q1.pdf", 1)
    platform_svc.register_document(tg_id, "default", "report-q2.pdf", 1)
    result = platform_svc.resolve_context_ref(tg_id, "report")  # type: ignore[attr-defined]
    assert result.needs_disambiguation is True
    assert len(result.candidates) == 2


# ─────────────────────────────────────────────────────────────────
# FR-5 — selected file used as full context
# ─────────────────────────────────────────────────────────────────

def test_fr_5_full_context_assembly_uses_whole_object(platform_svc, tg_id):
    """FR-5: при явном `[контекст]` система должна подключить файл
    целиком, а не отдавать случайные фрагменты RAG-поиска."""
    if not hasattr(platform_svc, "assemble_full_context"):
        pytest.skip("TODO FR-5: full-context assembly not implemented")
    platform_svc.create_domain(tg_id, "default")
    doc = platform_svc.register_document(tg_id, "default", "paper.pdf", num_chunks=10)
    platform_svc.set_object_content(doc.doc_id, "short body")  # type: ignore[attr-defined]
    ctx = platform_svc.assemble_full_context(  # type: ignore[attr-defined]
        tg_id, [doc.doc_id],
    )
    assert ctx.objects[0].doc_id == doc.doc_id
    assert ctx.objects[0].is_full is True
    assert ctx.objects[0].used_summarization is False


# ─────────────────────────────────────────────────────────────────
# NFR-13 — large-file context triggers map-reduce summarization
# ─────────────────────────────────────────────────────────────────

def test_nfr_13_oversize_file_triggers_chunked_summarization(platform_svc, tg_id):
    """NFR-13: если полный контент файла превышает лимит context window,
    assemble_full_context должен запустить map-reduce:
    чанки → suммари → финальный свод. В результате `is_full` остаётся True
    (семантически это весь файл), но `used_summarization=True` и
    `content` — это не исходник, а свод."""
    if not hasattr(platform_svc, "assemble_full_context"):
        pytest.skip("TODO NFR-13: full-context assembly not implemented")
    platform_svc.create_domain(tg_id, "default")
    doc = platform_svc.register_document(tg_id, "default", "huge.pdf", num_chunks=500)
    # Inject a giant body far above the assembly budget.
    huge = "lorem ipsum dolor sit amet. " * 20000  # ~540k chars
    platform_svc.set_object_content(doc.doc_id, huge)  # type: ignore[attr-defined]

    ctx = platform_svc.assemble_full_context(  # type: ignore[attr-defined]
        tg_id, [doc.doc_id], max_chars=20_000,
    )
    obj = ctx.objects[0]
    assert obj.doc_id == doc.doc_id
    assert obj.is_full is True, "семантически объект подключён целиком"
    assert obj.used_summarization is True, "должно быть map-reduce сжатие"
    assert len(obj.content) <= 20_000, "свод не должен превышать max_chars"
    assert obj.content, "свод не должен быть пустым"


def test_nfr_13_small_file_is_not_summarized(platform_svc, tg_id):
    """NFR-13 negative: маленький файл подключается напрямую, без summarization."""
    if not hasattr(platform_svc, "assemble_full_context"):
        pytest.skip("TODO NFR-13: full-context assembly not implemented")
    platform_svc.create_domain(tg_id, "default")
    doc = platform_svc.register_document(tg_id, "default", "tiny.txt", num_chunks=1)
    platform_svc.set_object_content(doc.doc_id, "короткий оригинал")  # type: ignore[attr-defined]

    ctx = platform_svc.assemble_full_context(tg_id, [doc.doc_id], max_chars=20_000)  # type: ignore[attr-defined]
    obj = ctx.objects[0]
    assert obj.used_summarization is False
    assert obj.content == "короткий оригинал"


# ─────────────────────────────────────────────────────────────────
# NFR-1 — идемпотентность повторного сохранения
# ─────────────────────────────────────────────────────────────────

def test_nfr_1_idempotent_save_on_reconfirm(platform_svc, tg_id):
    """NFR-1: повторное подтверждение сохранения не должно порождать
    дубли объектов. Нужен Memory Service с content hash — пока skip."""
    if not hasattr(platform_svc, "create_memory_object"):
        pytest.skip("TODO NFR-1: idempotent save requires Memory Service")
    a = platform_svc.create_memory_object(  # type: ignore[attr-defined]
        tg_id, kind="note", content="same body", client_request_id="req-1",
    )
    b = platform_svc.create_memory_object(  # type: ignore[attr-defined]
        tg_id, kind="note", content="same body", client_request_id="req-1",
    )
    assert a.memory_object_id == b.memory_object_id


# ─────────────────────────────────────────────────────────────────
# NFR-2 — ID, timestamps, metadata
# ─────────────────────────────────────────────────────────────────

def test_nfr_2_object_has_unique_id_and_timestamp(platform_svc, tg_id):
    """NFR-2: у каждого объекта Памяти есть уникальный ID и timestamp."""
    platform_svc.create_domain(tg_id, "default")
    a = platform_svc.register_document(tg_id, "default", "a.txt", 1)
    b = platform_svc.register_document(tg_id, "default", "b.txt", 1)
    assert a.doc_id != b.doc_id, "doc_id must be unique"
    assert a.added_at, "added_at must be populated"
    assert b.added_at


def test_nfr_2_object_metadata_preserved_on_read(platform_svc, tg_id):
    """NFR-2: сохранённый объект можно достать обратно с его метаданными."""
    platform_svc.create_domain(tg_id, "default")
    platform_svc.register_document(tg_id, "default", "x.pdf", num_chunks=7, message_id=11)
    docs = platform_svc.list_domains(tg_id)[0].documents
    assert docs[0].filename == "x.pdf"
    assert docs[0].num_chunks == 7
    assert docs[0].message_id == 11


# ─────────────────────────────────────────────────────────────────
# NFR-3 — любые объекты доступны для retrieval
# ─────────────────────────────────────────────────────────────────

def test_nfr_3_all_objects_retrievable_after_save(platform_svc, tg_id):
    """NFR-3: все сохранённые объекты доступны для последующего retrieval."""
    platform_svc.create_domain(tg_id, "default")
    for name in ("a.txt", "b.txt", "c.txt"):
        platform_svc.register_document(tg_id, "default", name, 1)

    domains = platform_svc.list_domains(tg_id)
    assert len(domains) == 1
    names = {d.filename for d in domains[0].documents}
    assert names == {"a.txt", "b.txt", "c.txt"}


def test_nfr_3_persistence_roundtrip_via_pickle(platform_svc, tg_id):
    """NFR-3: retrieval должен пережить рестарт процесса — state грузится
    из `platform_store.pkl`."""
    platform_svc.create_domain(tg_id, "default")
    platform_svc.register_document(tg_id, "default", "persist.txt", 2)

    # Simulate restart.
    platform_svc._PLATFORM_STORE.clear()
    platform_svc.load_platform_from_disk()

    docs = platform_svc.list_domains(tg_id)[0].documents
    assert docs[0].filename == "persist.txt"


# ─────────────────────────────────────────────────────────────────
# NFR-4 — resolver order: exact → alias → fuzzy
# ─────────────────────────────────────────────────────────────────

def test_nfr_4_resolver_exact_match_wins(platform_svc, tg_id):
    """NFR-4: точное совпадение имени должно выигрывать у fuzzy-кандидата."""
    if not hasattr(platform_svc, "resolve_context_ref"):
        pytest.skip("TODO NFR-4: context ref resolver not implemented")
    platform_svc.create_domain(tg_id, "default")
    platform_svc.register_document(tg_id, "default", "report.pdf", 1)
    platform_svc.register_document(tg_id, "default", "reports-index.pdf", 1)
    res = platform_svc.resolve_context_ref(tg_id, "report.pdf")  # type: ignore[attr-defined]
    assert res.matched.filename == "report.pdf"
    assert res.match_kind == "exact"


def test_nfr_4_resolver_alias_before_fuzzy(platform_svc, tg_id):
    """NFR-4: alias-совпадение должно обходить fuzzy-совпадение."""
    if not hasattr(platform_svc, "resolve_context_ref"):
        pytest.skip("TODO NFR-4: context ref resolver not implemented")
    platform_svc.create_domain(tg_id, "default")
    doc = platform_svc.register_document(tg_id, "default", "q4-revenue-v7.pdf", 1)
    platform_svc.set_alias(tg_id, doc.doc_id, "revenue")  # type: ignore[attr-defined]
    platform_svc.register_document(tg_id, "default", "revenue-forecast.pdf", 1)

    res = platform_svc.resolve_context_ref(tg_id, "revenue")  # type: ignore[attr-defined]
    assert res.match_kind == "alias"
    assert res.matched.filename == "q4-revenue-v7.pdf"


# ─────────────────────────────────────────────────────────────────
# NFR-5 — explicit version is honored; otherwise latest
# ─────────────────────────────────────────────────────────────────

def test_nfr_5_explicit_version_is_honored(platform_svc, tg_id):
    """NFR-5: `[file@v2]` должно возвращать ровно v2."""
    if not hasattr(platform_svc, "read_memory_object"):
        pytest.skip("TODO NFR-5: version-aware read not implemented")
    obj = platform_svc.create_memory_object(tg_id, kind="note", content="v1-body")  # type: ignore[attr-defined]
    platform_svc.update_memory_object(obj.memory_object_id, content="v2-body")  # type: ignore[attr-defined]
    platform_svc.update_memory_object(obj.memory_object_id, content="v3-body")  # type: ignore[attr-defined]

    v2 = platform_svc.read_memory_object(obj.memory_object_id, version="v2")  # type: ignore[attr-defined]
    assert v2.content == "v2-body"


def test_nfr_5_missing_version_returns_latest(platform_svc, tg_id):
    """NFR-5: запрос без версии должен вернуть latest."""
    if not hasattr(platform_svc, "read_memory_object"):
        pytest.skip("TODO NFR-5: version-aware read not implemented")
    obj = platform_svc.create_memory_object(tg_id, kind="note", content="v1")  # type: ignore[attr-defined]
    platform_svc.update_memory_object(obj.memory_object_id, content="v2")  # type: ignore[attr-defined]

    latest = platform_svc.read_memory_object(obj.memory_object_id)  # type: ignore[attr-defined]
    assert latest.content == "v2"
    assert latest.version == "v2"
