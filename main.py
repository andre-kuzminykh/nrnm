"""Entry point for the standalone nrnm ИИ-платформа Telegram bot.

Alongside the aiogram long-polling loop we run a tiny aiohttp health server
on ``config.HEALTH_PORT`` (default 8003). Docker-compose and external probes
(GCE, uptime monitors) hit ``GET /health`` and expect a 200 JSON payload.
The health server is started *before* polling so containers become healthy
as soon as the process is up and reachable.
"""

import asyncio
import json
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiohttp import web

import config
from bot.handlers.start import router as start_router
from bot.handlers.platform import router as platform_router
from services.platform import load_platform_from_disk


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Health server ────────────────────────────────────────────────

async def _health(_request: web.Request) -> web.Response:
    return web.Response(
        text=json.dumps({"status": "ok", "service": "nrnm-platform-bot"}),
        content_type="application/json",
    )


async def _start_health_server() -> web.AppRunner:
    """Start the /health aiohttp server on ``config.HEALTH_PORT``.

    Returned runner is kept alive for the process lifetime and cleaned up
    in the finally-block of ``main()``.
    """
    app = web.Application()
    app.router.add_get("/health", _health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=config.HEALTH_PORT)
    await site.start()
    logger.info("Health server listening on :%s", config.HEALTH_PORT)
    return runner


async def main():
    if not config.BOT_TOKEN:
        logger.error("BOT_TOKEN is not set. Put it in .env or export it.")
        sys.exit(1)

    # FR-P6: restore platform store from disk before first message.
    load_platform_from_disk()

    health_runner = await _start_health_server()

    bot = Bot(
        token=config.BOT_TOKEN,
        default=DefaultBotProperties(parse_mode="HTML"),
    )

    dp = Dispatcher()
    # start_router MUST be registered first so /start is caught by
    # CommandStart before the platform router's F.text catch-all.
    dp.include_router(start_router)
    dp.include_router(platform_router)

    logger.info("Bot starting...")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
        await health_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
