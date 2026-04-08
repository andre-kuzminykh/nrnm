"""Configuration for the standalone ИИ-платформа bot.

All values come from environment variables (see `.env.example`). The only
required one is `BOT_TOKEN`; everything else has sensible defaults or is
optional (e.g. RAG works in graceful-noop mode without `QDRANT_URL`).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Telegram ─────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("BOT_TOKEN", "")

# ── LLM (OpenAI-compatible) ──────────────────────────────────────
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

# Models user can pick in «🤖 Модель». Each entry:
#   (display_name, openai-compatible model id)
PLATFORM_MODELS = [
    ("GPT-4o mini (быстрая)", "gpt-4o-mini"),
    ("GPT-4o (качество)", "gpt-4o"),
    ("GPT-4.1 nano (самая быстрая)", "gpt-4.1-nano"),
]
DEFAULT_MODEL = PLATFORM_MODELS[0][1]

# ── RAG (Qdrant + embeddings) ────────────────────────────────────
# If QDRANT_URL is empty, the platform still works for domain management /
# model selection, but ingest/query are no-ops. See services/rag.py::is_configured.
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
