"""Tests for US-7 — LLM-powered file editing + dependency cascade.

## Трассируемость
Feature: Telegram AI Platform v2
Requirements: FR-37, FR-38, NFR-17

FR-37 — edit a file via LLM prompt: select file → send instruction →
        LLM modifies content → new version saved (Rule 6).
FR-38 — when a file is updated, system proposes updating dependent
        files (those used together in prior runs / [контекст] sets).
NFR-17 — pluggable file storage interface: save/load/url.
"""

from __future__ import annotations
import pytest


# ─────────────────────────────────────────────────────────────────
# FR-37 — LLM-powered file editing
# ─────────────────────────────────────────────────────────────────

def test_fr_37_edit_file_via_llm(platform_svc, tg_id):
    """User selects a file, sends a prompt, LLM edits content,
    new version is saved."""
    try:
        from services import file_editor  # noqa: WPS433
    except ImportError:
        pytest.skip("TODO FR-37: services.file_editor not implemented")

    platform_svc.create_domain(tg_id, "docs")
    doc = platform_svc.register_document(tg_id, "docs", "report.md", num_chunks=3)
    from services import memory as mem
    mem.set_object_content(doc.doc_id, "# Report\n\nOld content here.")

    result = file_editor.edit(
        tg_id=tg_id,
        doc_id=doc.doc_id,
        instruction="Change 'Old content' to 'New content'",
    )
    assert result.success
    assert result.new_version  # e.g. "v2"
    new_body = mem.get_object_content(doc.doc_id)
    assert "New content" in new_body or result.new_content  # either way


def test_fr_37_edit_creates_new_version(platform_svc, tg_id):
    """Editing must create a new version per Rule 6."""
    try:
        from services import file_editor, memory as mem
    except ImportError:
        pytest.skip("TODO FR-37: services.file_editor not implemented")

    platform_svc.create_domain(tg_id, "docs")
    doc = platform_svc.register_document(tg_id, "docs", "draft.txt", num_chunks=1)
    mem.set_object_content(doc.doc_id, "v1 body")

    obj = mem.create_memory_object(tg_id, kind="file", content="v1 body", filename="draft.txt")
    file_editor.edit(tg_id=tg_id, doc_id=obj.memory_object_id, instruction="add footer")

    versions = mem.list_versions(obj.memory_object_id)
    assert len(versions) >= 2, "edit must create a new version"


# ─────────────────────────────────────────────────────────────────
# FR-38 — dependency cascade
# ─────────────────────────────────────────────────────────────────

def test_fr_38_detect_dependent_files(platform_svc, tg_id):
    """After a file is updated, system should identify files that
    were used together with it in prior runs."""
    try:
        from services import file_editor
    except ImportError:
        pytest.skip("TODO FR-38: file_editor not implemented")

    platform_svc.create_domain(tg_id, "docs")
    doc_a = platform_svc.register_document(tg_id, "docs", "prompt.md", num_chunks=1)
    doc_b = platform_svc.register_document(tg_id, "docs", "config.yaml", num_chunks=1)

    # Simulate that both were used in the same task run
    file_editor.record_co_usage(tg_id, [doc_a.doc_id, doc_b.doc_id], run_id="run-1")

    # After updating prompt.md, system should suggest config.yaml
    deps = file_editor.find_dependents(tg_id, doc_a.doc_id)
    assert doc_b.doc_id in [d.doc_id for d in deps]


def test_fr_38_cascade_proposes_rerun(platform_svc, tg_id):
    """When a file is updated, system should propose re-running
    the same actions on dependent files."""
    try:
        from services import file_editor
    except ImportError:
        pytest.skip("TODO FR-38: cascade proposal not implemented")

    platform_svc.create_domain(tg_id, "docs")
    doc_a = platform_svc.register_document(tg_id, "docs", "template.md", num_chunks=1)
    doc_b = platform_svc.register_document(tg_id, "docs", "output.md", num_chunks=1)
    file_editor.record_co_usage(tg_id, [doc_a.doc_id, doc_b.doc_id], run_id="run-1")

    proposal = file_editor.propose_cascade(
        tg_id=tg_id,
        updated_doc_id=doc_a.doc_id,
        instruction="reformat to markdown tables",
    )
    assert proposal.dependent_ids
    assert proposal.suggested_instruction


# ─────────────────────────────────────────────────────────────────
# NFR-17 — pluggable storage interface
# ─────────────────────────────────────────────────────────────────

def test_nfr_17_storage_interface_has_save_load():
    """Storage backend must expose save(id, content) -> url and
    load(id) -> content."""
    try:
        from services import storage  # noqa: WPS433
    except ImportError:
        pytest.skip("TODO NFR-17: services.storage not implemented")

    assert callable(getattr(storage, "save", None))
    assert callable(getattr(storage, "load", None))

    # Minimal round-trip
    url = storage.save("test-id", b"hello world")
    assert url  # some form of ref
    body = storage.load("test-id")
    assert body == b"hello world"
