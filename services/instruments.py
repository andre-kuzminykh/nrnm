"""Instrument registry — the three user-facing tools and the СУПЕРАГЕНТ.

## Трассируемость
Feature: Telegram AI Platform v2
Requirements: FR-28, FR-29, FR-30

Replaces the old Chat / Task mode toggle with a flat instrument picker.
Each instrument:
- has a name, icon, and description for the keyboard
- declares its required parameters (e.g. file_search needs domains)
- routes incoming text messages to the right handler

The СУПЕРАГЕНТ (FR-31) is NOT an instrument — it's a separate entry
point that launches the full LangGraph pipeline. It lives outside
the instrument picker as a standalone big button.

Per-user active instrument is stored in a module-level dict, same
pattern as modes._USER_MODE. Default = "chat".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InstrumentParam:
    """One parameter an instrument requires before it can run."""
    name: str
    label: str
    type: str  # "domain_multiselect" | "text" | "none"
    required: bool = False
    default: Any = None


@dataclass
class Instrument:
    name: str
    icon: str
    label: str
    description: str
    params: dict[str, InstrumentParam] = field(default_factory=dict)


# ── Registry ─────────────────────────────────────────────────────

_INSTRUMENTS: dict[str, Instrument] = {
    "chat": Instrument(
        name="chat",
        icon="💬",
        label="Чат",
        description="Быстрый разговор с LLM. Подключайте файлы через [имя_файла].",
        params={},
    ),
    "file_search": Instrument(
        name="file_search",
        icon="📁",
        label="Память",
        description="RAG-поиск по выбранным доменам.",
        params={
            "domains": InstrumentParam(
                name="domains",
                label="Домены",
                type="domain_multiselect",
                required=True,
            ),
        },
    ),
    "web_search": Instrument(
        name="web_search",
        icon="🌐",
        label="Веб",
        description="Поиск актуальной информации через SerpAPI.",
        params={},
    ),
    "superagent": Instrument(
        name="superagent",
        icon="🧠",
        label="Агент",
        description="СУПЕРАГЕНТ: план задачи → LangGraph → step-by-step.",
        params={},
    ),
}


def list_instruments() -> set[str]:
    """FR-28: returns the set of available instrument names."""
    return set(_INSTRUMENTS.keys())


def get_instrument(name: str) -> Instrument | None:
    return _INSTRUMENTS.get(name)


def get_params(name: str) -> dict[str, InstrumentParam]:
    """FR-29: instrument-specific parameters."""
    inst = _INSTRUMENTS.get(name)
    if inst is None:
        return {}
    return dict(inst.params)


# ── Per-user active instrument ───────────────────────────────────

_USER_INSTRUMENT: dict[int, str] = {}


def get_active(tg_id: int) -> str:
    return _USER_INSTRUMENT.get(tg_id, "superagent")


def set_active(tg_id: int, name: str) -> str:
    if name not in _INSTRUMENTS:
        raise ValueError(f"Unknown instrument {name!r}. Available: {sorted(_INSTRUMENTS)}")
    _USER_INSTRUMENT[tg_id] = name
    return name


def _reset() -> None:
    """Called by conftest between tests."""
    _USER_INSTRUMENT.clear()
