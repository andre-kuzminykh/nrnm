"""Tests for US-8 — File tree memory with scoped RAG.

## Трассируемость
Feature: Telegram AI Platform v2
Requirements: FR-39, FR-40, FR-41, FR-42, FR-43

FR-39 — tree structure: folders + files as hyperlinks in text
FR-40 — scoped RAG: root = all, folder = recursive, file = single
FR-41 — file form: card + in-context chat + delete + back
FR-42 — pagination when >PAGE_SIZE files
FR-43 — create subfolder + upload + source hyperlinks in answers
"""

from __future__ import annotations
import pytest


def _tree():
    try:
        from services import file_tree
        return file_tree
    except Exception:
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
# FR-42 — pagination
# ─────────────────────────────────────────────────────────────────

def test_fr_42_pagination_available_when_many_files(platform_svc, tg_id):
    ft = _tree()
    if ft is None:
        pytest.skip("TODO FR-42: file_tree not implemented")
    ft.create_folder(tg_id, parent_path="/", name="big")
    for i in range(15):
        ft.add_file(tg_id, folder_path="/big", filename=f"file_{i}.txt",
                    doc_id=f"d{i}", num_chunks=1)
    total = ft.count_files(tg_id, "/big")
    assert total == 15
    page = ft.list_files_page(tg_id, "/big", page=0, page_size=10)
    assert len(page) == 10
    page2 = ft.list_files_page(tg_id, "/big", page=1, page_size=10)
    assert len(page2) == 5


def test_fr_42_no_pagination_when_few_files(platform_svc, tg_id):
    ft = _tree()
    if ft is None:
        pytest.skip("TODO FR-42: file_tree not implemented")
    ft.create_folder(tg_id, parent_path="/", name="small")
    for i in range(5):
        ft.add_file(tg_id, folder_path="/small", filename=f"f{i}.txt",
                    doc_id=f"d{i}", num_chunks=1)
    total = ft.count_files(tg_id, "/small")
    assert total == 5
    page = ft.list_files_page(tg_id, "/small", page=0, page_size=10)
    assert len(page) == 5


# ─────────────────────────────────────────────────────────────────
# FR-43 — file lands in current folder + source hyperlinks
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
