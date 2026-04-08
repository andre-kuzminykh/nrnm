"""Shared pytest fixtures for the v1 Telegram AI Platform spec tests.

Tests here are **spec-driven**: they encode the FR-*/NFR-* requirements from
`SPEC.md` directly. Behaviour that is not yet implemented is marked with
`pytest.skip(...)` pointing to a stable spec ID, so the same test activates
automatically once the corresponding service ships.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest


# Make sure `import services.platform` / `import config` resolve to the
# repo root regardless of where pytest is invoked from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@pytest.fixture(autouse=True)
def _isolated_platform_store(monkeypatch):
    """Redirect pickle persistence to a tmp dir and wipe the in-memory store
    for every test, so tests never see each other's state."""
    tmp = tempfile.mkdtemp(prefix="nrnm-test-")
    monkeypatch.setenv("DATA_PERSIST_DIR", tmp)

    # Reset the module-level state if already imported.
    try:
        import services.platform as platform_svc  # noqa: WPS433

        platform_svc._PLATFORM_STORE.clear()
        platform_svc._PERSIST_DIR = tmp
        platform_svc._PLATFORM_FILE = os.path.join(tmp, "platform_store.pkl")
    except Exception:  # noqa: BLE001
        pass
    yield


@pytest.fixture
def tg_id() -> int:
    return 1001


@pytest.fixture
def platform_svc():
    """Lazy import so conftest works even if the service module changes."""
    import services.platform as svc  # noqa: WPS433

    return svc
