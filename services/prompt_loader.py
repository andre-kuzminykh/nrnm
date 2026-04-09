"""Prompt loader — reads system prompts from .md files.

## Трассируемость
Feature: Telegram AI Platform v2
Requirements: FR-45

Prompts live in `prompts/system/` as plain .md files. They are
re-read on every call (not cached) so you can edit them without
restarting the bot.

Usage:
    from services import prompt_loader
    system_msg = prompt_loader.load("chat")
"""

from __future__ import annotations

import os

_PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "prompts",
    "system",
)


def load(name: str) -> str:
    """Load a prompt by name (without .md extension).

    Returns the file content as a string. If the file doesn't exist,
    returns a minimal fallback so the bot doesn't crash on missing
    prompt files.
    """
    path = os.path.join(_PROMPTS_DIR, f"{name}.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return f"You are a helpful assistant. (prompt '{name}' not found)"
    except Exception:  # noqa: BLE001
        return f"You are a helpful assistant. (prompt '{name}' read error)"


def list_prompts() -> list[str]:
    """List all available prompt names."""
    try:
        return [
            f[:-3] for f in os.listdir(_PROMPTS_DIR)
            if f.endswith(".md")
        ]
    except FileNotFoundError:
        return []
