"""File tree — hierarchical folder/file memory with scoped RAG.

## Трассируемость
Feature: Telegram AI Platform v2
Requirements: FR-39, FR-40, FR-41, FR-42, FR-43

Replaces the flat Domain model with a tree structure:

    /                           ← root (all user's files)
    /research/                  ← folder
    /research/report.pdf        ← file
    /research/deep/notes.txt    ← nested file

Core concepts:
- **TreeNode**: either a folder (children=[]) or a file (doc_id, num_chunks).
- **Path**: slash-separated, root = "/". Folders end with trailing / internally
  but the public API accepts both "/foo" and "/foo/" as equivalent.
- **Scope** (FR-40): `get_scope(tg_id, path)` returns all files at or below
  the given path. Root scope = every file the user owns. File path = that
  one file only.
- **Persistence**: piggybacks on the per-user pickle in `services.platform`
  via a module-level dict `_TREES`, wiped by conftest between tests.

Legacy Domain compat: the tree is a PARALLEL structure to the legacy
`PlatformUser.domains`. Both coexist — legacy RAG/platform code still
works untouched. The tree is the new v2 interface; migration is gradual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable


@dataclass
class TreeNode:
    """One node in the file tree — either a folder or a file."""
    name: str
    path: str           # e.g. "/research/report.pdf"
    is_folder: bool
    children: list["TreeNode"] = field(default_factory=list)
    # File-only fields:
    doc_id: str = ""
    num_chunks: int = 0
    created_at: str = ""

    def _find(self, parts: list[str]) -> "TreeNode | None":
        """Walk into the tree by path parts. Returns None if not found."""
        if not parts:
            return self
        target = parts[0]
        for child in self.children:
            if child.name == target:
                return child._find(parts[1:])
        return None


# ── Per-user tree storage ────────────────────────────────────────

_TREES: dict[int, TreeNode] = {}


def _reset() -> None:
    _TREES.clear()


def _root(tg_id: int) -> TreeNode:
    if tg_id not in _TREES:
        _TREES[tg_id] = TreeNode(name="", path="/", is_folder=True)
    return _TREES[tg_id]


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _normalise(path: str) -> str:
    """Strip trailing slashes (except root), collapse double slashes."""
    path = "/" + "/".join(p for p in path.split("/") if p)
    return path or "/"


def _split(path: str) -> list[str]:
    return [p for p in _normalise(path).split("/") if p]


def _resolve(tg_id: int, path: str) -> TreeNode | None:
    return _root(tg_id)._find(_split(path))


# ── FR-39: CRUD ──────────────────────────────────────────────────

def create_folder(tg_id: int, parent_path: str, name: str) -> TreeNode:
    """Create a folder inside `parent_path`. Returns the new node."""
    parent = _resolve(tg_id, parent_path)
    if parent is None or not parent.is_folder:
        raise ValueError(f"Parent path {parent_path!r} not found or not a folder")
    # Deduplicate
    for child in parent.children:
        if child.name == name and child.is_folder:
            return child
    new_path = _normalise(f"{parent.path}/{name}")
    node = TreeNode(name=name, path=new_path, is_folder=True, created_at=_now())
    parent.children.append(node)
    return node


def add_file(
    tg_id: int,
    folder_path: str,
    filename: str,
    doc_id: str,
    num_chunks: int,
) -> TreeNode:
    """Add a file reference into a folder. FR-43: file lands in the
    current open folder."""
    parent = _resolve(tg_id, folder_path)
    if parent is None or not parent.is_folder:
        raise ValueError(f"Folder {folder_path!r} not found")
    # Deduplicate by doc_id
    for child in parent.children:
        if not child.is_folder and child.doc_id == doc_id:
            return child
    new_path = _normalise(f"{parent.path}/{filename}")
    node = TreeNode(
        name=filename,
        path=new_path,
        is_folder=False,
        doc_id=doc_id,
        num_chunks=num_chunks,
        created_at=_now(),
    )
    parent.children.append(node)
    return node


def delete_node(tg_id: int, path: str) -> bool:
    """Delete a node (folder recursively, or single file)."""
    parts = _split(path)
    if not parts:
        return False  # can't delete root
    parent_parts = parts[:-1]
    target_name = parts[-1]
    parent = _root(tg_id)._find(parent_parts) if parent_parts else _root(tg_id)
    if parent is None:
        return False
    before = len(parent.children)
    parent.children = [c for c in parent.children if c.name != target_name]
    return len(parent.children) < before


def list_children(tg_id: int, path: str) -> list[TreeNode]:
    """List immediate children of a folder. FR-39."""
    node = _resolve(tg_id, path)
    if node is None or not node.is_folder:
        return []
    return list(node.children)


# ── FR-40 / FR-42: scoped file retrieval ──────────────────────────

def get_scope(tg_id: int, path: str) -> list[TreeNode]:
    """Return all FILE nodes at or below `path`.

    - Root ("/") → every file the user owns (FR-40 root = all).
    - Folder → all files recursively inside (FR-40 folder = recursive).
    - File path → that single file (FR-40 single file scope).
    - FR-42 "select all" is just get_scope on the current folder.
    """
    node = _resolve(tg_id, path)
    if node is None:
        return []
    if not node.is_folder:
        return [node]  # single file scope
    # Recursive collect
    result: list[TreeNode] = []
    _collect_files(node, result)
    return result


def _collect_files(node: TreeNode, out: list[TreeNode]) -> None:
    for child in node.children:
        if child.is_folder:
            _collect_files(child, out)
        else:
            out.append(child)


def get_scope_doc_ids(tg_id: int, path: str) -> list[str]:
    """Convenience: just the doc_ids from get_scope. Used by the RAG
    query layer to filter Qdrant searches to the right collections."""
    return [f.doc_id for f in get_scope(tg_id, path)]
