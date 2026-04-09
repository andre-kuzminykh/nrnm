"""Tests for US-8 — File tree memory with scoped RAG.

## Трассируемость
Feature: Telegram AI Platform v2
Requirements: FR-39, FR-40, FR-41, FR-42, FR-43

FR-39 — tree structure: folders of arbitrary depth + files at any level
FR-40 — RAG scoping: root = all files, folder = recursive, file = single
FR-41 — in-context chat at any tree level
FR-42 — "select all" at any folder level
FR-43 — create subfolder + upload file into current folder
"""

from __future__ import annotations
import pytest


def _tree():
    try:
        from services import file_tree  # noqa: WPS433
        return file_tree
    except Exception:  # noqa: BLE001
        return None


# ─────────────────────────────────────────────────────────────────
# FR-39 — tree CRUD
# ─────────────────────────────────────────────────────────────────

def test_fr_39_create_root_folder(platform_svc, tg_id):
    ft = _tree()
    if ft is None:
        pytest.skip("TODO FR-39: services.file_tree not implemented")
    node = ft.create_folder(tg_id, parent_path="/", name="research")
    assert node.name == "research"
    assert node.path == "/research"
    assert node.is_folder is True


def test_fr_39_create_nested_folders(platform_svc, tg_id):
    ft = _tree()
    if ft is None:
        pytest.skip("TODO FR-39: file_tree not implemented")
    ft.create_folder(tg_id, parent_path="/", name="projects")
    ft.create_folder(tg_id, parent_path="/projects", name="alpha")
    node = ft.create_folder(tg_id, parent_path="/projects/alpha", name="docs")
    assert node.path == "/projects/alpha/docs"


def test_fr_39_add_file_to_folder(platform_svc, tg_id):
    ft = _tree()
    if ft is None:
        pytest.skip("TODO FR-39: file_tree not implemented")
    ft.create_folder(tg_id, parent_path="/", name="data")
    f = ft.add_file(tg_id, folder_path="/data", filename="report.pdf",
                    doc_id="d001", num_chunks=5)
    assert f.name == "report.pdf"
    assert f.path == "/data/report.pdf"
    assert f.is_folder is False


def test_fr_39_list_children(platform_svc, tg_id):
    ft = _tree()
    if ft is None:
        pytest.skip("TODO FR-39: file_tree not implemented")
    ft.create_folder(tg_id, parent_path="/", name="mix")
    ft.create_folder(tg_id, parent_path="/mix", name="sub")
    ft.add_file(tg_id, folder_path="/mix", filename="a.txt",
                doc_id="d1", num_chunks=1)
    children = ft.list_children(tg_id, "/mix")
    names = [c.name for c in children]
    assert "sub" in names
    assert "a.txt" in names


def test_fr_39_delete_folder_recursive(platform_svc, tg_id):
    ft = _tree()
    if ft is None:
        pytest.skip("TODO FR-39: file_tree not implemented")
    ft.create_folder(tg_id, parent_path="/", name="tmp")
    ft.add_file(tg_id, folder_path="/tmp", filename="x.txt",
                doc_id="d1", num_chunks=1)
    ft.delete_node(tg_id, "/tmp")
    children = ft.list_children(tg_id, "/")
    assert all(c.name != "tmp" for c in children)


# ─────────────────────────────────────────────────────────────────
# FR-40 — scoped RAG
# ─────────────────────────────────────────────────────────────────

def test_fr_40_root_scope_returns_all_files(platform_svc, tg_id):
    ft = _tree()
    if ft is None:
        pytest.skip("TODO FR-40: file_tree not implemented")
    ft.create_folder(tg_id, parent_path="/", name="a")
    ft.create_folder(tg_id, parent_path="/", name="b")
    ft.add_file(tg_id, folder_path="/a", filename="f1.txt",
                doc_id="d1", num_chunks=2)
    ft.add_file(tg_id, folder_path="/b", filename="f2.txt",
                doc_id="d2", num_chunks=3)
    scope = ft.get_scope(tg_id, "/")
    doc_ids = [f.doc_id for f in scope]
    assert "d1" in doc_ids
    assert "d2" in doc_ids


def test_fr_40_folder_scope_is_recursive(platform_svc, tg_id):
    ft = _tree()
    if ft is None:
        pytest.skip("TODO FR-40: file_tree not implemented")
    ft.create_folder(tg_id, parent_path="/", name="proj")
    ft.create_folder(tg_id, parent_path="/proj", name="sub")
    ft.add_file(tg_id, folder_path="/proj", filename="top.txt",
                doc_id="d1", num_chunks=1)
    ft.add_file(tg_id, folder_path="/proj/sub", filename="deep.txt",
                doc_id="d2", num_chunks=1)
    scope = ft.get_scope(tg_id, "/proj")
    doc_ids = [f.doc_id for f in scope]
    assert "d1" in doc_ids
    assert "d2" in doc_ids


def test_fr_40_single_file_scope(platform_svc, tg_id):
    ft = _tree()
    if ft is None:
        pytest.skip("TODO FR-40: file_tree not implemented")
    ft.create_folder(tg_id, parent_path="/", name="docs")
    ft.add_file(tg_id, folder_path="/docs", filename="only.txt",
                doc_id="d1", num_chunks=1)
    scope = ft.get_scope(tg_id, "/docs/only.txt")
    assert len(scope) == 1
    assert scope[0].doc_id == "d1"


# ─────────────────────────────────────────────────────────────────
# FR-42 — select all
# ─────────────────────────────────────────────────────────────────

def test_fr_42_select_all_returns_recursive_file_list(platform_svc, tg_id):
    ft = _tree()
    if ft is None:
        pytest.skip("TODO FR-42: file_tree not implemented")
    ft.create_folder(tg_id, parent_path="/", name="all")
    ft.create_folder(tg_id, parent_path="/all", name="nested")
    ft.add_file(tg_id, folder_path="/all", filename="a.txt",
                doc_id="d1", num_chunks=1)
    ft.add_file(tg_id, folder_path="/all/nested", filename="b.txt",
                doc_id="d2", num_chunks=1)
    # select_all is the same as get_scope — it's the same concept
    scope = ft.get_scope(tg_id, "/all")
    assert len(scope) == 2


# ─────────────────────────────────────────────────────────────────
# FR-43 — create subfolder + upload into current folder
# ─────────────────────────────────────────────────────────────────

def test_fr_43_file_lands_in_current_folder(platform_svc, tg_id):
    ft = _tree()
    if ft is None:
        pytest.skip("TODO FR-43: file_tree not implemented")
    ft.create_folder(tg_id, parent_path="/", name="inbox")
    f = ft.add_file(tg_id, folder_path="/inbox", filename="new.pdf",
                    doc_id="d99", num_chunks=10)
    children = ft.list_children(tg_id, "/inbox")
    assert any(c.doc_id == "d99" for c in children if not c.is_folder)
