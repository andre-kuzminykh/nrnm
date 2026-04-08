"""Minimal /start + main_menu handler for the standalone ИИ-платформа bot."""

from aiogram import Router, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import CallbackQuery, Message

from bot.keyboards.inline import start_keyboard

router = Router()


WELCOME_TEXT = (
    "👋 <b>Добро пожаловать в ИИ-платформу!</b>\n\n"
    "Это ваш персональный RAG-помощник:\n"
    "• 📁 Создавайте <b>домены</b> — изолированные базы знаний\n"
    "• 📎 Загружайте <b>файлы</b> (pdf, docx, txt, md) — они автоматически индексируются\n"
    "• 💬 Задавайте <b>вопросы</b> — бот отвечает с <b>[1]</b>, <b>[2]</b> ссылками на источники\n"
    "• 💾 Сохраняйте ответы в память как новые документы\n\n"
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
