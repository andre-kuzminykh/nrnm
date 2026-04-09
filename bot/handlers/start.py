"""/start handler — opens the main menu directly, no intermediate screen."""

from aiogram import Router
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message

import config
from services import platform as platform_svc
from services import instruments as instruments_svc
from bot.keyboards.inline import platform_menu_keyboard

router = Router()


def _model_label(model_id: str) -> str:
    return next(
        (label for label, mid in config.PLATFORM_MODELS if mid == model_id),
        model_id or "не выбрана",
    )


@router.message(CommandStart())
async def cmd_start(message: Message):
    """Skip the old "🧠 ИИ-платформа" landing page — go straight to
    the main menu with instruments, СУПЕРАГЕНТ, and model picker."""
    tg_id = message.from_user.id
    user = platform_svc.get_user(tg_id)

    # Import here to set the wait state (lives in the platform handler)
    from bot.handlers.platform import _set_wait
    _set_wait(tg_id, "platform")

    await message.answer(
        "👋 <b>Добро пожаловать!</b>\n\n"
        "Выберите инструмент или запустите 🧠 СУПЕРАГЕНТ.",
        reply_markup=platform_menu_keyboard(
            _model_label(user.model_id),
            platform_svc.get_active_domains(tg_id),
            active_instrument=instruments_svc.get_active(tg_id),
        ),
        parse_mode=ParseMode.HTML,
    )
