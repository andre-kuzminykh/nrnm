"""Minimal /start + main_menu handler for the standalone ИИ-платформа bot."""

from aiogram import Router, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import CallbackQuery, Message

from bot.keyboards.inline import start_keyboard

router = Router()


WELCOME_TEXT = (
    "👋 <b>Добро пожаловать в ИИ-платформу!</b>\n\n"
    "Три инструмента:\n"
    "• 💬 <b>Чат</b> — быстрые вопросы, файлы через "
    "<code>[имя_файла]</code> / <code>[имя@v2]</code>\n"
    "• 🔍 <b>Поиск по файлам</b> — RAG по выбранным доменам\n"
    "• 🌐 <b>Веб-поиск</b> — актуальная информация из интернета\n\n"
    "🤖 <b>СУПЕРАГЕНТ</b> — ставите цель, бот строит план "
    "(LangGraph, параллелизм, условия), показывает его, ждёт "
    "подтверждения и выполняет step-by-step с критикой.\n\n"
    "Команды: /chat, /search, /web, /agent, /mode\n\n"
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
