"""Context-aware web search — expands follow-up queries using dialog history.

## Трассируемость
Feature: Telegram AI Platform v2
Requirements: FR-44

When user says "а в сша?" after asking about robots in Europe, the raw
query "а в сша?" won't find useful results. This module:

1. Takes the current message + chat history
2. Calls LLM to produce an expanded, self-contained search query
3. Returns it for the tool layer to execute

Falls back to the raw message when no LLM is available (NFR-14).
"""

from __future__ import annotations

import logging
from typing import Any

import config
from services import prompt_loader

logger = logging.getLogger(__name__)


def build_search_query(
    current_message: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Expand a follow-up message into a self-contained search query
    using the dialog history.

    If LLM is unavailable or history is empty, returns the message
    as-is (still a valid search query, just not contextualised).
    """
    if not history or not config.LLM_API_KEY:
        return current_message.strip()

    # Only use the last 6 messages (3 turns) to keep it focused
    recent = history[-6:]

    system_prompt = prompt_loader.load("web_search_query")

    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:200]}"
        for m in recent
    )

    user_prompt = (
        f"История диалога:\n{history_text}\n\n"
        f"Текущее сообщение: {current_message}\n\n"
        "Поисковый запрос:"
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        query = (resp.choices[0].message.content or "").strip()
        if query:
            logger.info("web_search_ctx: %r -> %r", current_message, query)
            return query
    except Exception as exc:  # noqa: BLE001
        logger.warning("web_search_ctx LLM failed: %s", exc)

    return current_message.strip()
