"""Minimal /start + main_menu handler for the standalone ИИ-платформа bot."""

from aiogram import Router, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import CallbackQuery, Message

from bot.keyboards.inline import start_keyboard

router = Router()


WELCOME_TEXT = (
    "👋 <b>Добро пожаловать в ИИ-платформу!</b>\n\n"
    "Два режима работы:\n"
    "• 💬 <b>Чат</b> — быстрые вопросы с Памятью, файлы подключаются через "
    "<code>[имя_файла]</code> или <code>[имя@v2]</code>\n"
    "• 🎯 <b>Задачи</b> — ставите цель, бот строит полный план, "
    "показывает его, ждёт подтверждения и затем выполняет\n\n"
    "Команды: /chat, /task, /mode\n\n"
    "Нажмите кнопку ниже, чтобы начать."
)


@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        WELCOME_TEXT,
        reply_markup=start_keyboard(),
        parse_mode=ParseMode.HTML,
    )


@router.callback_query(F.data == "main_menu")
async def on_main_menu(callback: CallbackQuery):
    """Return to the main menu. Deletes the current widget and posts a
    fresh welcome message — matches the replace-widget pattern used
    throughout the platform handler (FR-P17)."""
    try:
        await callback.message.delete()
    except Exception:  # noqa: BLE001
        pass
    await callback.message.bot.send_message(
        chat_id=callback.message.chat.id,
        text=WELCOME_TEXT,
        reply_markup=start_keyboard(),
        parse_mode=ParseMode.HTML,
    )
    await callback.answer()
