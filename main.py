"""Entry point for the standalone nrnm ИИ-платформа Telegram bot."""

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties

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


async def main():
    if not config.BOT_TOKEN:
        logger.error("BOT_TOKEN is not set. Put it in .env or export it.")
        sys.exit(1)

    # FR-P6: restore platform store from disk before first message.
    load_platform_from_disk()

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


if __name__ == "__main__":
    asyncio.run(main())
