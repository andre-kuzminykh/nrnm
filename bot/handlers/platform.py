"""ИИ-платформа handler — FR-P1..P14.

Поведение:
  * Widget = chat. `platform_menu` открывает сообщение-виджет с инструкцией и
    3 кнопками (модель, память, сброс контекста). Любое текстовое сообщение
    юзера от этого момента до выхода трактуется как RAG-вопрос и заносится в
    `chat_history`. Любой файл — как документ для индексации в первый
    активный домен.
  * Multi-select доменов: клик по доменной строке переключает активность.
    RAG query идёт поверх объединённых коллекций выбранных доменов.
  * Новый домен: при «➕ Новый домен» виджет прячет все кнопки, оставляя
    только «◀️ Отмена». Следующий текст становится именем — домен создаётся,
    сразу активен, виджет обновляется.
  * «💾 Сохранить в память» под каждым ответом LLM кладёт ответ в Qdrant
    первого активного домена как новый chunk.
  * «🔄 Сбросить контекст» чистит `chat_history`.
"""

from __future__ import annotations

import asyncio
import html as _html
import io
import logging
import queue as _queue
import time as _time
import uuid
from dataclasses import dataclass

from aiogram import Router, F
from aiogram.enums import ContentType, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

import config
from services import platform as platform_svc
from services import rag
from services import modes as modes_svc
from services import context_resolver
from services import memory as memory_svc
from services import mcp_registry
from services import instruments as instruments_svc
from services import file_tree as file_tree_svc
from services import prompt_loader
from services import web_search_ctx
from bot.keyboards.inline import (
    platform_menu_keyboard,
    platform_model_keyboard,
    platform_memory_keyboard,
    platform_new_domain_keyboard,
    platform_domain_keyboard,
    platform_doc_keyboard,
    platform_answer_keyboard,
    platform_mcp_list_keyboard,
    platform_mcp_view_keyboard,
    platform_mcp_cancel_keyboard,
    task_approval_keyboard,
    task_reapproval_keyboard,
    file_tree_keyboard,
    file_context_keyboard,
)

logger = logging.getLogger(__name__)
router = Router()


# Per-user wait state. Values:
#   "platform"    — on platform widget, any text = RAG question, any file = upload
#   "new_domain"  — waiting for new-domain name
_PLATFORM_WAIT: dict[int, str] = {}

# Per-user list of bot message IDs that should be deleted on "🔄 reset".
# Messages the user explicitly saved (💾 → ✅) are removed from this list
# so they survive the wipe. Everything else = disposable dialog.
_DISPOSABLE_MSGS: dict[int, list[int]] = {}


def _track_msg(tg_id: int, message_id: int) -> None:
    """Mark a message (bot OR user) as disposable (deleted on 🔄 reset)."""
    _DISPOSABLE_MSGS.setdefault(tg_id, []).append(message_id)
    if len(_DISPOSABLE_MSGS[tg_id]) > 400:
        _DISPOSABLE_MSGS[tg_id] = _DISPOSABLE_MSGS[tg_id][-400:]


def _untrack_msg(tg_id: int, message_id: int) -> None:
    """Remove a message from the disposable list (it was saved)."""
    msgs = _DISPOSABLE_MSGS.get(tg_id)
    if msgs:
        try:
            msgs.remove(message_id)
        except ValueError:
            pass


def _set_wait(tg_id: int, state: str | None) -> None:
    if state is None:
        _PLATFORM_WAIT.pop(tg_id, None)
    else:
        _PLATFORM_WAIT[tg_id] = state


def _get_wait(tg_id: int) -> str | None:
    return _PLATFORM_WAIT.get(tg_id)


async def _replace_widget(message, text, **kwargs):
    """FR-P17: delete the current bot widget and post a fresh one in its place.

    Used for *every* callback inside ИИ-платформа so the chat reads as a
    sequence of clean widgets after each click. User-uploaded files
    (`Message.document`) and LLM-answer messages are NEVER touched — only
    the navigation widget itself is replaced. Status messages emitted from
    `_ingest_file` / `_handle_rag_chat` keep using `edit_text` because they
    are progress updates on a single bot message, not user clicks.
    """
    try:
        await message.delete()
    except TelegramBadRequest:
        pass
    except Exception:  # noqa: BLE001
        logger.debug("widget delete failed", exc_info=True)
    sent = await message.bot.send_message(
        chat_id=message.chat.id, text=text, **kwargs,
    )
    # Track for the 🔄 reset wipe. We use chat_id as a proxy for
    # tg_id when the real user id isn't available in this helper.
    _track_msg(message.chat.id, sent.message_id)
    return sent


def _model_label(model_id: str) -> str:
    return next(
        (label for label, mid in config.PLATFORM_MODELS if mid == model_id),
        model_id or "не выбрана",
    )


def _mode_label(mode: str) -> str:
    return {"chat": "💬 Чат", "task": "🎯 Задачи"}.get(mode, mode)


def _instrument_hint(instrument: str) -> str:
    """Short contextual hint shown in the main widget."""
    hints = {
        "chat": (
            "Просто пишите — чистый LLM с историей.\n"
            "Файл целиком: <code>[имя]</code> / <code>[имя@v2]</code>."
        ),
        "file_search": (
            "Введите запрос — RAG по выбранным доменам.\n"
            "Выберите домены в «📁 Файлы» ниже."
        ),
        "web_search": (
            "Введите запрос — веб-поиск через SerpAPI.\n"
            "Ссылки будут кликабельные."
        ),
        "superagent": (
            "🧠 Опишите задачу — построю план и выполню.\n"
            "Файлы: <code>[имя]</code>. Можно уточнять план."
        ),
    }
    return hints.get(instrument, hints["chat"])


def _menu_text(user: platform_svc.PlatformUser, tg_id: int | None = None) -> str:
    active = platform_svc.get_active_domains_for(user)
    instrument = instruments_svc.get_active(tg_id) if tg_id is not None else "chat"
    inst_obj = instruments_svc.get_instrument(instrument)
    inst_label = f"{inst_obj.icon} {inst_obj.label}" if inst_obj else instrument
    text = (
        "🧠 <b>ИИ-платформа</b>\n\n"
        f"<b>Инструмент:</b> {inst_label}\n"
        f"<b>Модель:</b> {_html.escape(_model_label(user.model_id))}\n"
        f"<b>Домены:</b> {_html.escape(', '.join(active) or 'не выбраны')}\n\n"
        f"<i>{_instrument_hint(instrument)}</i>"
    )
    if not rag.is_configured() and instrument == "file_search":
        text += "\n\n<i>⚠️ RAG не сконфигурирован (QDRANT_URL).</i>"
    if not active and instrument == "file_search":
        text += "\n\n<i>ℹ️ Выберите домен в «💾 Память» чтобы начать RAG-поиск.</i>"
    return text


# Monkey-patch helper: platform.py exposes get_active_domains(tg_id) but we
# often already have the user object — use this to avoid extra lookups.
def _get_active_domains_for(user: platform_svc.PlatformUser) -> list[str]:
    return [d for d in user.domains if d in user.active_domains]


platform_svc.get_active_domains_for = _get_active_domains_for  # type: ignore[attr-defined]


# ── FR-P1 / FR-P11: platform widget ──────────────────────────────

@router.callback_query(F.data == "platform_menu")
async def on_platform_menu(callback: CallbackQuery):
    tg_id = callback.from_user.id
    user = platform_svc.get_user(tg_id)
    _set_wait(tg_id, "platform")
    await _replace_widget(
        callback.message, _menu_text(user, tg_id),
        reply_markup=platform_menu_keyboard(
            _model_label(user.model_id),
            platform_svc.get_active_domains(tg_id),
            active_instrument=instruments_svc.get_active(tg_id),
        ),
        parse_mode=ParseMode.HTML,
    )
    await callback.answer()


# ── FR-28: instrument picker ──────────────────────────────────────

@router.callback_query(F.data.startswith("platform_instrument:"))
async def on_platform_instrument_switch(callback: CallbackQuery):
    name = callback.data.split(":", 1)[1]
    try:
        instruments_svc.set_active(callback.from_user.id, name)
    except ValueError as e:
        await callback.answer(str(e), show_alert=True)
        return
    inst = instruments_svc.get_instrument(name)
    label = f"{inst.icon} {inst.label}" if inst else name
    await callback.answer(f"Инструмент: {label}")
    await on_platform_menu(callback)


# ── FR-31: СУПЕРАГЕНТ entry point ────────────────────────────────

@router.callback_query(F.data == "platform_superagent")
async def on_platform_superagent(callback: CallbackQuery):
    """Legacy: redirect cached keyboards with old callback_data."""
    instruments_svc.set_active(callback.from_user.id, "superagent")
    await callback.answer("🧠 Агент")
    await on_platform_menu(callback)


# ── Legacy command shortcuts (kept for keyboard convenience) ─────

@router.message(Command("chat"))
async def cmd_chat(message: Message):
    instruments_svc.set_active(message.from_user.id, "chat")
    await message.answer(
        "✅ Инструмент: <b>💬 Чат</b>\n\n"
        "Пишите вопрос — отвечу с учётом Памяти.",
        parse_mode=ParseMode.HTML,
    )


@router.message(Command("search"))
async def cmd_search(message: Message):
    instruments_svc.set_active(message.from_user.id, "file_search")
    await message.answer(
        "✅ Инструмент: <b>🔍 Поиск по файлам</b>\n\n"
        "Введите запрос — поищу в выбранных доменах.",
        parse_mode=ParseMode.HTML,
    )


@router.message(Command("web"))
async def cmd_web(message: Message):
    instruments_svc.set_active(message.from_user.id, "web_search")
    await message.answer(
        "✅ Инструмент: <b>🌐 Веб-поиск</b>\n\n"
        "Введите запрос — поищу актуальную информацию в вебе.",
        parse_mode=ParseMode.HTML,
    )


@router.message(Command("agent"))
async def cmd_agent(message: Message):
    instruments_svc.set_active(message.from_user.id, "superagent")
    _set_wait(message.from_user.id, "platform")
    await message.answer(
        "✅ Инструмент: <b>🧠 Агент</b>\n\n"
        "Опишите задачу — построю план и выполню.",
        parse_mode=ParseMode.HTML,
    )


@router.message(Command("mode"))
async def cmd_mode(message: Message):
    inst = instruments_svc.get_active(message.from_user.id)
    inst_obj = instruments_svc.get_instrument(inst)
    label = f"{inst_obj.icon} {inst_obj.label}" if inst_obj else inst
    await message.answer(
        f"Текущий инструмент: <b>{label}</b>\n\n"
        "Переключение: /chat, /search, /web, /agent",
        parse_mode=ParseMode.HTML,
    )


# ── FR-P2: model picker ──────────────────────────────────────────

@router.callback_query(F.data == "platform_model")
async def on_platform_model(callback: CallbackQuery):
    user = platform_svc.get_user(callback.from_user.id)
    await _replace_widget(
        callback.message,
        "🤖 <b>Выбор модели</b>\n\nВыберите LLM для чата:",
        reply_markup=platform_model_keyboard(config.PLATFORM_MODELS, user.model_id),
        parse_mode=ParseMode.HTML,
    )
    await callback.answer()


@router.callback_query(F.data.startswith("platform_model_pick:"))
async def on_platform_model_pick(callback: CallbackQuery):
    model_id = callback.data.split(":", 1)[1]
    platform_svc.set_model(callback.from_user.id, model_id)
    await callback.answer(f"Модель: {model_id}")
    await on_platform_menu(callback)


# ── FR-P3 / FR-P9: memory / multi-select domains ─────────────────

@router.callback_query(F.data == "platform_memory")
async def on_platform_memory(callback: CallbackQuery):
    tg_id = callback.from_user.id
    user = platform_svc.get_user(tg_id)
    doms = list(user.domains.values())
    text = "💾 <b>Память — выберите один или несколько доменов</b>\n\n"
    if not doms:
        text += "<i>У вас пока нет доменов. Создайте первый.</i>"
    else:
        text += "<i>Клик по домену — переключить активность для RAG.</i>"
    await _replace_widget(
        callback.message, text,
        reply_markup=platform_memory_keyboard(doms, user.active_domains),
        parse_mode=ParseMode.HTML,
    )
    await callback.answer()


@router.callback_query(F.data.startswith("platform_domain_pick:"))
async def on_platform_domain_pick(callback: CallbackQuery):
    """FR-P16: single-select — picking a domain replaces the active set."""
    tg_id = callback.from_user.id
    idx = int(callback.data.split(":", 1)[1])
    user = platform_svc.get_user(tg_id)
    doms = list(user.domains.values())
    if idx >= len(doms):
        await callback.answer("Не найдено")
        return
    name = doms[idx].name
    user.active_domains = {name}
    platform_svc._persist()
    await callback.answer(f"Выбран: {name}")
    await on_platform_domain_open(callback)


# Legacy callback kept for backwards compat with old messages still in chats
@router.callback_query(F.data.startswith("platform_domain_toggle:"))
async def on_platform_domain_toggle(callback: CallbackQuery):
    tg_id = callback.from_user.id
    idx = int(callback.data.split(":", 1)[1])
    user = platform_svc.get_user(tg_id)
    doms = list(user.domains.values())
    if idx >= len(doms):
        await callback.answer("Не найдено")
        return
    user.active_domains = {doms[idx].name}
    platform_svc._persist()
    await callback.answer(f"Выбран: {doms[idx].name}")
    await on_platform_memory(callback)


@router.callback_query(F.data == "platform_domain_new")
async def on_platform_domain_new(callback: CallbackQuery):
    tg_id = callback.from_user.id
    _set_wait(tg_id, "new_domain")
    await _replace_widget(
        callback.message,
        "➕ <b>Новый домен</b>\n\n"
        "Отправьте название одним сообщением. 1–40 символов, русский/латиница, цифры, пробел, _-.",
        reply_markup=platform_new_domain_keyboard(),
        parse_mode=ParseMode.HTML,
    )
    await callback.answer()


@router.callback_query(F.data.startswith("platform_domain_open:"))
async def on_platform_domain_open(callback: CallbackQuery):
    tg_id = callback.from_user.id
    idx = int(callback.data.split(":", 1)[1])
    user = platform_svc.get_user(tg_id)
    doms = list(user.domains.values())
    if idx >= len(doms):
        await callback.answer("Не найдено")
        return
    # Make sure platform wait state stays on so file uploads in this screen
    # land in the ingest path even before user picks the domain explicitly.
    _set_wait(tg_id, "platform")
    domain = doms[idx]
    is_active = domain.name in user.active_domains
    mcp_count = len(getattr(domain, "mcps", None) or [])
    text = (
        f"📁 <b>Домен:</b> {_html.escape(domain.name)}\n"
        f"<b>Документов:</b> {len(domain.documents)}\n"
        f"<b>Инструментов (MCP):</b> {mcp_count}\n"
        f"<b>Выбран для RAG:</b> {'да' if is_active else 'нет'}\n\n"
        "<i>📎 Чтобы добавить файлы — отправьте их прямо в чат "
        "(txt/md/pdf/docx). Они автоматически попадут в этот домен.\n"
        "🔧 MCP-инструменты — доступны Task mode планировщику для данного домена.</i>"
    )
    await _replace_widget(
        callback.message, text,
        reply_markup=platform_domain_keyboard(idx, domain.documents, is_active, mcp_count),
        parse_mode=ParseMode.HTML,
    )
    await callback.answer()


@router.callback_query(F.data.startswith("platform_domain_delete:"))
async def on_platform_domain_delete(callback: CallbackQuery):
    tg_id = callback.from_user.id
    idx = int(callback.data.split(":", 1)[1])
    user = platform_svc.get_user(tg_id)
    doms = list(user.domains.values())
    if idx >= len(doms):
        await callback.answer("Не найдено")
        return
    name = doms[idx].name
    platform_svc.delete_domain(tg_id, name)
    await callback.answer(f"Удалён: {name}")
    await on_platform_memory(callback)


# ── FR-P5: doc view / delete ─────────────────────────────────────

@router.callback_query(F.data.startswith("platform_doc_view:"))
async def on_platform_doc_view(callback: CallbackQuery):
    tg_id = callback.from_user.id
    _, dom_idx_s, doc_idx_s = callback.data.split(":", 2)
    dom_idx, doc_idx = int(dom_idx_s), int(doc_idx_s)
    user = platform_svc.get_user(tg_id)
    doms = list(user.domains.values())
    if dom_idx >= len(doms) or doc_idx >= len(doms[dom_idx].documents):
        await callback.answer("Не найдено")
        return
    doc = doms[dom_idx].documents[doc_idx]
    text = (
        f"📄 <b>{_html.escape(doc.filename)}</b>\n\n"
        f"ID: <code>{doc.doc_id}</code>\n"
        f"Фрагментов: {doc.num_chunks}\n"
        f"Добавлен: {doc.added_at}"
    )
    await _replace_widget(
        callback.message, text,
        reply_markup=platform_doc_keyboard(dom_idx, doc_idx),
        parse_mode=ParseMode.HTML,
    )
    await callback.answer()


@router.callback_query(F.data.startswith("platform_doc_delete:"))
async def on_platform_doc_delete(callback: CallbackQuery):
    tg_id = callback.from_user.id
    _, dom_idx_s, doc_idx_s = callback.data.split(":", 2)
    dom_idx, doc_idx = int(dom_idx_s), int(doc_idx_s)
    user = platform_svc.get_user(tg_id)
    doms = list(user.domains.values())
    if dom_idx >= len(doms) or doc_idx >= len(doms[dom_idx].documents):
        await callback.answer("Не найдено")
        return
    domain = doms[dom_idx]
    doc = domain.documents[doc_idx]
    try:
        await rag.delete_document_vectors(
            platform_svc.collection_name(tg_id, domain.name), doc.doc_id,
        )
    except Exception:  # noqa: BLE001
        logger.exception("rag delete failed")
    platform_svc.delete_document(tg_id, domain.name, doc.doc_id)
    await callback.answer("Удалено")
    await on_platform_domain_open(callback)


# ── FR-39..43: File tree navigation + scoped chat ────────────────

# Per-user current path in the file tree. When set, RAG is scoped to
# this path. None or "/" = all files.
_USER_TREE_PATH: dict[int, str] = {}


def _get_tree_path(tg_id: int) -> str:
    return _USER_TREE_PATH.get(tg_id, "/")


def _set_tree_path(tg_id: int, path: str) -> None:
    _USER_TREE_PATH[tg_id] = path


def _breadcrumb(path: str) -> str:
    """Render a path as a readable breadcrumb: / → 🏠, /a/b → 🏠 › a › b"""
    parts = [p for p in path.split("/") if p]
    if not parts:
        return "🏠 Корень"
    return "🏠 › " + " › ".join(parts)


def _render_folder_text(tg_id: int, path: str, page: int = 0) -> str:
    """Render a folder as text: breadcrumb + file hyperlinks.

    Files are shown as clickable callback-based "hyperlinks" using
    inline formatting — user taps a filename and gets the file form.
    Since Telegram inline buttons are the only way to make text
    actionable in bot messages, each file name is bold in the text
    and also has a corresponding button in the keyboard. BUT per the
    new spec, files are listed AS TEXT (hyperlink-style) and clicking
    them triggers a callback via the file's path.

    Actually, Telegram can't make arbitrary text clickable without
    a URL. So we use a numbered list where each file is a bold name,
    and the user taps the file via a small inline button that appears
    alongside. But to minimize buttons: files are rendered as text
    with their index, and there's ONE "open file" mechanism per tap.
    """
    page_files = file_tree_svc.list_files_page(tg_id, path, page=page)
    total = file_tree_svc.count_files(tg_id, path)

    lines = [f"📁 <b>{_breadcrumb(path)}</b>"]
    lines.append(f"Файлов: {total}\n")

    if page_files:
        offset = page * file_tree_svc.PAGE_SIZE
        for i, f in enumerate(page_files, start=offset + 1):
            # FR-46: [filename.pdf] format — user can copy-paste into
            # messages for [контекст] refs. Displayed as code so it's
            # visually distinct and easy to select/copy.
            lines.append(f"  <code>[{_html.escape(f.name)}]</code>")
    else:
        lines.append("<i>Файлов нет. Отправьте файл в чат.</i>")

    lines.append("")
    lines.append("<i>📎 Отправьте файл — попадёт сюда.\n💬 Напишите вопрос — RAG по этой папке.</i>")
    return "\n".join(lines)


@router.callback_query(F.data.startswith("ftree:"))
async def on_ftree_navigate(callback: CallbackQuery):
    """Navigate into a folder or open a file form."""
    tg_id = callback.from_user.id
    path = callback.data.split(":", 1)[1]
    node = file_tree_svc._resolve(tg_id, path)

    if node is None:
        path = "/"
        node = file_tree_svc._root(tg_id)

    _set_tree_path(tg_id, path)
    _set_wait(tg_id, "platform")

    if node.is_folder:
        await _show_folder(callback.message, tg_id, path, page=0)
    else:
        # File form (FR-41)
        parent_path = "/".join(path.rstrip("/").split("/")[:-1]) or "/"
        text = (
            f"📄 <b>{_html.escape(node.name)}</b>\n\n"
            f"Путь: <code>{_html.escape(path)}</code>\n"
            f"Фрагментов: {node.num_chunks}\n"
            f"ID: <code>{node.doc_id}</code>\n\n"
            "<i>💬 Напишите вопрос — отвечу только по этому файлу.</i>"
        )
        await _replace_widget(
            callback.message, text,
            reply_markup=file_context_keyboard(path, parent_path),
            parse_mode=ParseMode.HTML,
        )
    await callback.answer()


async def _show_folder(message, tg_id: int, path: str, page: int = 0):
    """Render a folder with file list + subfolders + pagination."""
    children = file_tree_svc.list_children(tg_id, path)
    subfolders = [c for c in children if c.is_folder]
    page_files = file_tree_svc.list_files_page(tg_id, path, page=page)
    tp = file_tree_svc.total_pages(tg_id, path)
    parent_path = "/".join(path.rstrip("/").split("/")[:-1]) or "/"
    if path == "/":
        parent_path = None

    text = _render_folder_text(tg_id, path, page)
    await _replace_widget(
        message, text,
        reply_markup=file_tree_keyboard(
            path, subfolders,
            page_files=page_files,
            page=page, total_pages=tp,
            parent_path=parent_path,
        ),
        parse_mode=ParseMode.HTML,
    )


@router.callback_query(F.data.startswith("ftree_page:"))
async def on_ftree_page(callback: CallbackQuery):
    """Pagination: switch to a different page of files."""
    tg_id = callback.from_user.id
    # Format: ftree_page:/path:N
    parts = callback.data.split(":")
    # Rejoin path parts (path may contain colons in theory, but ours don't)
    page = int(parts[-1])
    path = ":".join(parts[1:-1])
    _set_tree_path(tg_id, path)
    await _show_folder(callback.message, tg_id, path, page=page)
    await callback.answer()


@router.callback_query(F.data == "noop")
async def on_noop(callback: CallbackQuery):
    await callback.answer()


@router.callback_query(F.data.startswith("ftree_mkdir:"))
async def on_ftree_mkdir(callback: CallbackQuery):
    """Create a new subfolder — ask for the name."""
    tg_id = callback.from_user.id
    parent_path = callback.data.split(":", 1)[1]
    _set_wait(tg_id, "ftree_mkdir")
    _USER_TREE_PATH[tg_id] = parent_path
    try:
        await callback.message.delete()
    except Exception:  # noqa: BLE001
        pass
    sent = await callback.message.bot.send_message(
        chat_id=callback.message.chat.id,
        text=(
            f"📁 <b>{_breadcrumb(parent_path)}</b>\n\n"
            "Введите название:"
        ),
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀️ Отмена", callback_data=f"ftree:{parent_path}")],
        ]),
        parse_mode=ParseMode.HTML,
    )
    _track_msg(tg_id, sent.message_id)
    await callback.answer("Введите название")


@router.callback_query(F.data.startswith("ftree_delete:"))
async def on_ftree_delete(callback: CallbackQuery):
    tg_id = callback.from_user.id
    path = callback.data.split(":", 1)[1]
    name = path.rstrip("/").split("/")[-1]
    parent_path = "/".join(path.rstrip("/").split("/")[:-1]) or "/"
    file_tree_svc.delete_node(tg_id, path)
    _set_tree_path(tg_id, parent_path)
    await callback.answer(f"Удалено: {name}")
    callback.data = f"ftree:{parent_path}"
    await on_ftree_navigate(callback)


# ── FR-23/24: MCP management ────────────────────────────────────

def _mcp_wait_key(tg_id: int) -> str:
    """Key used in _PLATFORM_WAIT to stash the in-progress MCP FSM
    payload. We piggyback on the existing wait-state dict rather than
    adding a second FSM — the lifecycle is short and single-step per
    user so one slot is enough."""
    return f"mcp:{tg_id}"


_MCP_DRAFT: dict[int, dict] = {}


@router.callback_query(F.data.startswith("platform_mcp_list:"))
async def on_platform_mcp_list(callback: CallbackQuery):
    tg_id = callback.from_user.id
    idx = int(callback.data.split(":", 1)[1])
    user = platform_svc.get_user(tg_id)
    doms = list(user.domains.values())
    if idx >= len(doms):
        await callback.answer("Домен не найден", show_alert=True)
        return
    domain = doms[idx]
    mcps = mcp_registry.list_mcps(tg_id, domain.name)
    text = (
        f"🔧 <b>Инструменты домена «{_html.escape(domain.name)}»</b>\n\n"
        f"<i>Task-режим видит эти MCP как available tools. "
        f"По умолчанию есть web_search (SerpAPI). Можно добавить свои HTTP-MCP.</i>"
    )
    if not mcps:
        text += "\n\n<i>Инструментов нет. Нажмите ➕ чтобы добавить.</i>"
    await _replace_widget(
        callback.message, text,
        reply_markup=platform_mcp_list_keyboard(idx, mcps),
        parse_mode=ParseMode.HTML,
    )
    await callback.answer()


@router.callback_query(F.data.startswith("platform_mcp_view:"))
async def on_platform_mcp_view(callback: CallbackQuery):
    tg_id = callback.from_user.id
    _, dom_idx_s, mcp_idx_s = callback.data.split(":", 2)
    dom_idx = int(dom_idx_s)
    mcp_idx = int(mcp_idx_s)
    user = platform_svc.get_user(tg_id)
    doms = list(user.domains.values())
    if dom_idx >= len(doms):
        await callback.answer("Домен не найден")
        return
    mcps = mcp_registry.list_mcps(tg_id, doms[dom_idx].name)
    if mcp_idx >= len(mcps):
        await callback.answer("MCP не найден")
        return
    mcp = mcps[mcp_idx]
    kind = "builtin" if mcp.is_builtin else "http"
    text = (
        f"🔧 <b>{_html.escape(mcp.name)}</b>\n\n"
        f"<b>Тип:</b> {kind}\n"
        f"<b>URL:</b> <code>{_html.escape(mcp.url)}</code>\n"
        f"<b>Token:</b> <code>{'***' if mcp.token else '(нет)'}</code>\n"
        f"<b>Описание:</b>\n<i>{_html.escape(mcp.description or '—')}</i>\n\n"
        f"<b>Создан:</b> {mcp.created_at}\n"
        f"<b>Обновлён:</b> {mcp.updated_at}"
    )
    await _replace_widget(
        callback.message, text,
        reply_markup=platform_mcp_view_keyboard(dom_idx, mcp_idx),
        parse_mode=ParseMode.HTML,
    )
    await callback.answer()


@router.callback_query(F.data.startswith("platform_mcp_delete:"))
async def on_platform_mcp_delete(callback: CallbackQuery):
    tg_id = callback.from_user.id
    _, dom_idx_s, mcp_idx_s = callback.data.split(":", 2)
    dom_idx = int(dom_idx_s)
    mcp_idx = int(mcp_idx_s)
    user = platform_svc.get_user(tg_id)
    doms = list(user.domains.values())
    if dom_idx >= len(doms):
        await callback.answer("Домен не найден")
        return
    mcps = mcp_registry.list_mcps(tg_id, doms[dom_idx].name)
    if mcp_idx >= len(mcps):
        await callback.answer("MCP не найден")
        return
    name = mcps[mcp_idx].name
    mcp_registry.delete_mcp(tg_id, doms[dom_idx].name, name)
    await callback.answer(f"Удалён: {name}")
    # Rebuild as a new MCP list-callback — fake the new callback_data
    callback.data = f"platform_mcp_list:{dom_idx}"
    await on_platform_mcp_list(callback)


@router.callback_query(F.data.startswith("platform_mcp_new:"))
async def on_platform_mcp_new(callback: CallbackQuery):
    """Start the add-MCP FSM: first ask for name."""
    tg_id = callback.from_user.id
    dom_idx = int(callback.data.split(":", 1)[1])
    user = platform_svc.get_user(tg_id)
    doms = list(user.domains.values())
    if dom_idx >= len(doms):
        await callback.answer("Домен не найден")
        return
    _MCP_DRAFT[tg_id] = {"domain_idx": dom_idx, "stage": "name"}
    _set_wait(tg_id, _mcp_wait_key(tg_id))
    await _replace_widget(
        callback.message,
        "🔧 <b>Новый MCP</b>\n\n"
        "<b>Шаг 1/4 — Имя:</b>\n"
        "Отправьте короткое имя инструмента (англ. буквы, цифры, _ -). "
        "Например: <code>github_search</code>, <code>slack_notify</code>.",
        reply_markup=platform_mcp_cancel_keyboard(dom_idx),
        parse_mode=ParseMode.HTML,
    )
    await callback.answer()


@router.callback_query(F.data.startswith("platform_mcp_edit:"))
async def on_platform_mcp_edit(callback: CallbackQuery):
    """Start the edit-MCP FSM: pre-load the existing entry and ask
    for a new description first (most common edit). URL/token stay
    as-is unless the user sends `-` to reuse the current value."""
    tg_id = callback.from_user.id
    _, dom_idx_s, mcp_idx_s = callback.data.split(":", 2)
    dom_idx = int(dom_idx_s)
    mcp_idx = int(mcp_idx_s)
    user = platform_svc.get_user(tg_id)
    doms = list(user.domains.values())
    if dom_idx >= len(doms):
        await callback.answer("Домен не найден")
        return
    mcps = mcp_registry.list_mcps(tg_id, doms[dom_idx].name)
    if mcp_idx >= len(mcps):
        await callback.answer("MCP не найден")
        return
    existing = mcps[mcp_idx]
    _MCP_DRAFT[tg_id] = {
        "domain_idx": dom_idx,
        "stage": "edit_description",
        "edit_name": existing.name,
    }
    _set_wait(tg_id, _mcp_wait_key(tg_id))
    await _replace_widget(
        callback.message,
        f"✏️ <b>Редактирование MCP «{_html.escape(existing.name)}»</b>\n\n"
        f"Отправьте новое описание (или <code>-</code> чтобы оставить как есть).\n\n"
        f"<b>Текущее:</b>\n<i>{_html.escape(existing.description or '—')}</i>",
        reply_markup=platform_mcp_cancel_keyboard(dom_idx),
        parse_mode=ParseMode.HTML,
    )
    await callback.answer()


async def _handle_mcp_fsm(message: Message) -> bool:
    """Text handler for the MCP add/edit FSM. Returns True if handled."""
    tg_id = message.from_user.id
    draft = _MCP_DRAFT.get(tg_id)
    if not draft:
        return False
    text = (message.text or "").strip()
    stage = draft.get("stage")
    dom_idx = draft["domain_idx"]
    user = platform_svc.get_user(tg_id)
    doms = list(user.domains.values())
    if dom_idx >= len(doms):
        _MCP_DRAFT.pop(tg_id, None)
        _set_wait(tg_id, "platform")
        return False
    domain_name = doms[dom_idx].name

    # ── Add flow ──────────────────────────────────────────────
    if stage == "name":
        draft["name"] = text
        draft["stage"] = "url"
        await message.answer(
            "<b>Шаг 2/4 — URL:</b>\n"
            "Введите URL MCP-сервера. Для встроенного web_search — "
            "<code>builtin://serpapi</code>. Для внешнего — "
            "<code>https://...</code>.",
            parse_mode=ParseMode.HTML,
            reply_markup=platform_mcp_cancel_keyboard(dom_idx),
        )
        return True

    if stage == "url":
        draft["url"] = text
        draft["stage"] = "token"
        await message.answer(
            "<b>Шаг 3/4 — Token:</b>\n"
            "Bearer-токен для авторизации на MCP-сервере. "
            "Отправьте <code>-</code> если токен не нужен.",
            parse_mode=ParseMode.HTML,
            reply_markup=platform_mcp_cancel_keyboard(dom_idx),
        )
        return True

    if stage == "token":
        draft["token"] = "" if text == "-" else text
        draft["stage"] = "description"
        await message.answer(
            "<b>Шаг 4/4 — Описание:</b>\n"
            "Короткое описание того, что делает инструмент. "
            "LLM-планировщик читает это и решает, когда его вызывать.",
            parse_mode=ParseMode.HTML,
            reply_markup=platform_mcp_cancel_keyboard(dom_idx),
        )
        return True

    if stage == "description":
        try:
            entry = mcp_registry.add_mcp(
                tg_id, domain_name,
                name=draft["name"],
                url=draft["url"],
                token=draft.get("token", ""),
                description=text,
            )
        except mcp_registry.MCPRegistryError as exc:
            await message.answer(
                f"❌ {_html.escape(str(exc))}\nПопробуйте снова: отправьте имя или нажмите Отмена.",
                parse_mode=ParseMode.HTML,
                reply_markup=platform_mcp_cancel_keyboard(dom_idx),
            )
            draft["stage"] = "name"
            return True
        _MCP_DRAFT.pop(tg_id, None)
        _set_wait(tg_id, "platform")
        await message.answer(
            f"✅ MCP <b>{_html.escape(entry.name)}</b> добавлен в домен "
            f"<b>{_html.escape(domain_name)}</b>.",
            parse_mode=ParseMode.HTML,
        )
        return True

    # ── Edit flow ─────────────────────────────────────────────
    if stage == "edit_description":
        new_desc = None if text == "-" else text
        try:
            mcp_registry.update_mcp(
                tg_id, domain_name, draft["edit_name"],
                description=new_desc,
            )
        except mcp_registry.MCPRegistryError as exc:
            await message.answer(f"❌ {_html.escape(str(exc))}")
            _MCP_DRAFT.pop(tg_id, None)
            _set_wait(tg_id, "platform")
            return True
        _MCP_DRAFT.pop(tg_id, None)
        _set_wait(tg_id, "platform")
        await message.answer(
            f"✅ Описание MCP <b>{_html.escape(draft['edit_name'])}</b> обновлено.",
            parse_mode=ParseMode.HTML,
        )
        return True

    return False


# ── FR-P11: reset chat context ───────────────────────────────────

@router.callback_query(F.data == "platform_reset")
async def on_platform_reset(callback: CallbackQuery):
    """🔄 Обновить контекст — wipe the disposable dialog.

    1. Delete every tracked bot message (answers, status msgs, widgets)
       EXCEPT those the user saved via 💾 (removed from the list by
       _untrack_msg in on_platform_save_answer).
    2. Clear chat_history so LLM forgets prior turns.
    3. Send a FRESH main menu as a new message — the old widget is
       gone (deleted above), so this is the only live message left.
    """
    tg_id = callback.from_user.id
    chat_id = callback.message.chat.id
    bot = callback.message.bot

    # 1. Delete disposable messages
    msg_ids = list(_DISPOSABLE_MSGS.pop(tg_id, []))
    # Also delete the message that had the 🔄 button itself
    msg_ids.append(callback.message.message_id)
    for mid in msg_ids:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=mid)
        except Exception:  # noqa: BLE001
            pass  # already deleted or too old (>48h)

    # 2. Clear history
    platform_svc.reset_chat(tg_id)

    # 3. Fresh menu in a new message
    user = platform_svc.get_user(tg_id)
    _set_wait(tg_id, "platform")
    await bot.send_message(
        chat_id=chat_id,
        text=_menu_text(user, tg_id),
        reply_markup=platform_menu_keyboard(
            _model_label(user.model_id),
            platform_svc.get_active_domains(tg_id),
            active_instrument=instruments_svc.get_active(tg_id),
        ),
        parse_mode=ParseMode.HTML,
    )
    try:
        await callback.answer("🔄 Контекст очищен")
    except Exception:  # noqa: BLE001
        pass  # callback may be stale if message was already deleted


# ── FR-P12: save LLM answer to memory ────────────────────────────

@router.callback_query(F.data == "platform_save_answer")
async def on_platform_save_answer(callback: CallbackQuery):
    tg_id = callback.from_user.id
    user = platform_svc.get_user(tg_id)
    active = platform_svc.get_active_domains(tg_id)
    if not active:
        await callback.answer("Нет активного домена — выберите в 💾 Память", show_alert=True)
        return
    if not user.last_answer:
        await callback.answer("Нет ответа для сохранения", show_alert=True)
        return
    if not rag.is_configured():
        await callback.answer("RAG не настроен", show_alert=True)
        return
    domain_name = active[0]
    doc_id = uuid.uuid4().hex[:12]
    n = await rag.ingest_document(
        platform_svc.collection_name(tg_id, domain_name),
        doc_id, f"saved-{doc_id}.txt", user.last_answer,
    )
    if n == 0:
        await callback.answer("Не удалось сохранить", show_alert=True)
        return
    platform_svc.register_document(tg_id, domain_name, f"saved-{doc_id}.txt", n)
    await callback.answer(f"Сохранено в «{domain_name}» ({n} фр.)", show_alert=True)
    # FR-P19: swap the inline button to its «✅» state. Also untrack
    # the message so it survives the 🔄 reset wipe — saved = permanent.
    _untrack_msg(tg_id, callback.message.message_id)
    try:
        await callback.message.edit_reply_markup(
            reply_markup=platform_answer_keyboard(saved=True),
        )
    except TelegramBadRequest:
        pass


@router.callback_query(F.data == "platform_save_noop")
async def on_platform_save_noop(callback: CallbackQuery):
    """FR-P19: idle callback for the «✅ Сохранено» state — already saved."""
    await callback.answer("Уже сохранено")


# ── Document upload (no button — just send a file while on platform) ──

@router.message(F.document)
async def on_platform_document(message: Message):
    tg_id = message.from_user.id
    if _get_wait(tg_id) not in ("platform", "new_domain"):
        return  # not in platform mode — let other handlers take it
    _track_msg(tg_id, message.message_id)
    await _ingest_file(message)


async def _ingest_file(message: Message) -> None:
    """FR-P13: file sent inside platform widget → ingest into first active domain."""
    tg_id = message.from_user.id
    user = platform_svc.get_user(tg_id)
    active = platform_svc.get_active_domains(tg_id)
    if not active:
        await message.answer(
            "ℹ️ Сначала выберите домен в «💾 Память» или создайте новый.",
            parse_mode=ParseMode.HTML,
        )
        return
    domain_name = active[0]
    if not rag.is_configured():
        await message.answer("⚠️ RAG не настроен.")
        return

    filename = "message.txt"
    text_content = ""
    if message.document:
        filename = message.document.file_name or "document"
        try:
            file = await message.bot.get_file(message.document.file_id)
            buf = io.BytesIO()
            await message.bot.download_file(file.file_path, destination=buf)
            data = buf.getvalue()
            text_content = _extract_text(data, filename)
        except Exception as exc:  # noqa: BLE001
            await message.answer(f"❌ Не удалось прочитать файл: {exc}")
            return
    elif message.text:
        text_content = message.text
        filename = "inline-text.txt"

    if not text_content.strip():
        await message.answer(f"❌ Файл «{filename}» пустой или нечитаем.")
        return

    status = await message.answer(f"⏳ Индексирую <b>{_html.escape(filename)}</b>…",
                                  parse_mode=ParseMode.HTML)
    doc_id = uuid.uuid4().hex[:12]
    n_chunks = await rag.ingest_document(
        platform_svc.collection_name(tg_id, domain_name),
        doc_id, filename, text_content,
        message_id=message.message_id,  # FR-P18: track upload msg for source links
    )
    if n_chunks == 0:
        await status.edit_text(f"❌ Индексирование <b>{_html.escape(filename)}</b> не удалось.",
                               parse_mode=ParseMode.HTML)
        return
    doc = platform_svc.register_document(
        tg_id, domain_name, filename, n_chunks,
        message_id=message.message_id,  # FR-P18
    )
    # FR-5: stash the full extracted body so [контекст]/[file] can pull
    # the whole document later without re-downloading / re-parsing.
    memory_svc.set_object_content(doc.doc_id, text_content)
    # FR-43: also register the file in the tree at the user's current path
    tree_path = _get_tree_path(tg_id)
    try:
        file_tree_svc.add_file(tg_id, tree_path, filename, doc.doc_id, n_chunks)
    except ValueError:
        # Tree path doesn't exist — add to root
        file_tree_svc.add_file(tg_id, "/", filename, doc.doc_id, n_chunks)
    await status.edit_text(
        f"✅ <b>{_html.escape(filename)}</b> — {n_chunks} фрагментов "
        f"в <b>{_breadcrumb(tree_path)}</b>",
        parse_mode=ParseMode.HTML,
    )


# ── Main text dispatcher for platform wait states ────────────────

async def platform_handle_message(message: Message) -> bool:
    """Called from strategy/assessment text routers BEFORE their own logic.
    Returns True if handled by platform (new_domain input or RAG chat)."""
    tg_id = message.from_user.id
    wait = _get_wait(tg_id)
    if wait is None:
        return False

    # Track the user's message so it gets wiped on 🔄 reset too.
    _track_msg(tg_id, message.message_id)

    # 0. MCP add/edit FSM — routes back to _handle_mcp_fsm which owns
    # its own in-progress draft dict.
    if wait == _mcp_wait_key(tg_id):
        handled = await _handle_mcp_fsm(message)
        if handled:
            return True

    # 0b. File tree: mkdir name input
    if wait == "ftree_mkdir" and message.text:
        name = message.text.strip()
        parent_path = _get_tree_path(tg_id)
        try:
            node = file_tree_svc.create_folder(tg_id, parent_path, name)
        except ValueError as e:
            await message.answer(f"❌ {e}")
            return True
        _set_wait(tg_id, "platform")
        _set_tree_path(tg_id, parent_path)
        # Just re-show the folder — user sees the new subfolder appear
        await _show_folder(message, tg_id, parent_path, page=0)
        return True

    # 0c. СУПЕРАГЕНТ plan refinement — user can send text to refine
    if wait == "platform_superagent_refine" and message.text:
        await _handle_task_goal(message)  # regenerates plan with new text
        return True

    # 1. New domain name input
    if wait == "new_domain" and message.text:
        name = message.text.strip()
        try:
            platform_svc.create_domain(tg_id, name)
        except ValueError as e:
            await message.answer(f"❌ {e}. Попробуйте ещё раз.", parse_mode=ParseMode.HTML)
            return True
        _set_wait(tg_id, "platform")
        # Send updated memory widget — NEW message (old one kept)
        user = platform_svc.get_user(tg_id)
        doms = list(user.domains.values())
        await message.answer(
            f"✅ Домен <b>{_html.escape(name)}</b> создан и выбран.\n\n"
            "💾 <b>Память — выберите один или несколько доменов</b>",
            reply_markup=platform_memory_keyboard(doms, user.active_domains),
            parse_mode=ParseMode.HTML,
        )
        return True

    # 2. Platform — route by active INSTRUMENT (FR-28)
    if wait == "platform":
        if message.document:
            await _ingest_file(message)
            return True
        if not message.text:
            return True
        instrument = instruments_svc.get_active(tg_id)
        if instrument == "superagent":
            await _handle_task_goal(message)
        elif instrument == "web_search":
            await _handle_web_search(message)
        elif instrument == "file_search":
            await _handle_rag_chat(message)
        else:
            await _handle_pure_chat(message)
        return True

    return False


# ── FR-28: pure chat instrument (no RAG, just LLM + history) ────

async def _handle_pure_chat(message: Message) -> None:
    """Chat instrument — pure conversational LLM with dialog history.

    Does NOT search RAG/Qdrant. Does NOT require active domains.
    [filename] refs are still resolved and injected as full context.
    """
    tg_id = message.from_user.id
    user = platform_svc.get_user(tg_id)
    question = message.text or ""
    platform_svc.add_chat_message(tg_id, "user", question)

    # FR-4: resolve [filename] / [file@v2] refs and pull full content
    explicit_refs = context_resolver.parse_context_refs(question)
    explicit_context_block = ""
    if explicit_refs:
        resolved_ids: list[str] = []
        for ref in explicit_refs:
            res = context_resolver.resolve_context_ref(tg_id, ref.name)
            if res.matched is not None:
                eid = (
                    getattr(res.matched, "doc_id", None)
                    or getattr(res.matched, "memory_object_id", None)
                )
                if eid:
                    resolved_ids.append(eid)
        if resolved_ids:
            bundle = context_resolver.assemble_full_context(tg_id, resolved_ids)
            chunks: list[str] = []
            for obj in bundle.objects:
                if not obj.content:
                    continue
                marker = "[СВОД]" if obj.used_summarization else "[ПОЛНЫЙ ФАЙЛ]"
                chunks.append(f"{marker} {obj.filename}\n{obj.content}")
            if chunks:
                explicit_context_block = "\n\n---\n\n".join(chunks)

    messages = [{
        "role": "system",
        "content": prompt_loader.load("chat"),
    }]
    messages.extend(user.chat_history[-10:-1])

    user_content = question
    if explicit_context_block:
        user_content = f"Контекст:\n{explicit_context_block}\n\nВопрос: {question}"

    messages.append({"role": "user", "content": user_content})

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        resp = await client.chat.completions.create(
            model=user.model_id, messages=messages, temperature=0.3,
        )
        answer = resp.choices[0].message.content or "(пусто)"
    except Exception as exc:  # noqa: BLE001
        await message.answer(f"❌ Ошибка LLM: {exc}")
        return

    platform_svc.add_chat_message(tg_id, "assistant", answer)
    answer_html = _render_answer_with_hyperlinks(answer)
    sent = await message.answer(
        f"💬 <b>{_html.escape(_model_label(user.model_id))}</b>\n\n{answer_html}",
        reply_markup=platform_answer_keyboard(),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )
    _track_msg(tg_id, sent.message_id)


# ── FR-28: web_search instrument handler ─────────────────────────

async def _handle_web_search(message: Message) -> None:
    """Web search instrument — context-aware (FR-44).

    Flow:
    1. Save user message to history.
    2. Build a context-aware search query from history + current msg.
    3. Call SerpAPI via MCP.
    4. Synthesise an answer via LLM using search results + history.
    5. Render with hyperlinks (FR-36), save to history.
    """
    tg_id = message.from_user.id
    user = platform_svc.get_user(tg_id)
    raw_query = (message.text or "").strip()
    if not raw_query:
        return

    platform_svc.add_chat_message(tg_id, "user", raw_query)

    status = await message.answer("🌐 Ищу в вебе…")

    # FR-44: expand follow-up queries using dialog history
    search_query = web_search_ctx.build_search_query(
        current_message=raw_query,
        history=user.chat_history,
    )

    # Call SerpAPI
    try:
        from services import mcp_registry, mcp_client

        active = platform_svc.get_active_domains(tg_id)
        domain = active[0] if active else None
        entry = None
        if domain:
            entry = mcp_registry.get_mcp(tg_id, domain, "web_search")
        if entry is None:
            from services import tools
            result = tools.call("web_search", {"query": search_query})
        else:
            result = mcp_client.dispatch(entry, {"query": search_query})
    except Exception as exc:  # noqa: BLE001
        await status.edit_text(f"❌ Ошибка поиска: {_html.escape(str(exc)[:300])}")
        return

    if result.status != "ok":
        await status.edit_text(
            f"❌ Поиск не удался: {_html.escape(result.error or 'unknown')}",
        )
        return

    hits = (result.output or {}).get("hits") or []
    if not hits:
        await status.edit_text("Ничего не найдено.")
        return

    # Synthesise an answer using LLM + search results + history (FR-44)
    hits_text = "\n\n".join(
        f"[{hit.get('title', '')}]({hit.get('url', '')})\n{hit.get('snippet', '')}"
        for hit in hits[:6]
    )

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)

        llm_messages = [{"role": "system", "content": prompt_loader.load("web_search")}]
        llm_messages.extend(user.chat_history[-8:-1])
        llm_messages.append({
            "role": "user",
            "content": (
                f"Результаты веб-поиска по запросу «{search_query}»:\n\n"
                f"{hits_text}\n\n"
                f"Вопрос пользователя: {raw_query}"
            ),
        })

        resp = await client.chat.completions.create(
            model=user.model_id, messages=llm_messages, temperature=0.3,
        )
        answer = resp.choices[0].message.content or "(пусто)"
    except Exception as exc:  # noqa: BLE001
        # Fallback: just show raw hits
        answer = ""
        logger.warning("web_search LLM synthesis failed: %s", exc)

    platform_svc.add_chat_message(tg_id, "assistant", answer or hits_text)

    # Render: just the LLM answer with hyperlinks already inline.
    # No separate sources block — the LLM is prompted to embed URLs
    # directly in the text via the web_search.md system prompt.
    answer_html = _render_answer_with_hyperlinks(answer) if answer else ""

    parts = [f"🌐 <b>{_html.escape(_model_label(user.model_id))}</b>"]
    if search_query != raw_query:
        parts.append(f"<i>Запрос: {_html.escape(search_query)}</i>")
    if answer_html:
        parts.append(answer_html)
    text = "\n\n".join(parts)
    if len(text) > 3800:
        text = text[:3800] + "\n<i>…(обрезано)</i>"

    user.last_answer = text
    platform_svc._persist()
    await status.edit_text(
        text,
        reply_markup=platform_answer_keyboard(),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )
    _track_msg(tg_id, status.message_id)


# ── FR-9..11: Task mode — goal → plan preview → approval ─────────

async def _handle_task_goal(message: Message) -> None:
    """Task mode entry: build full plan, show preview.

    User can either:
    - ✅ Подтвердить → plan erased, progress bar starts
    - Send another message → plan regenerated with refinement
    """
    tg_id = message.from_user.id
    goal = (message.text or "").strip()
    if not goal:
        return

    status = await message.answer("🧠 <i>Строю план…</i>", parse_mode=ParseMode.HTML)
    _track_msg(tg_id, status.message_id)

    session = modes_svc.start_task(tg_id, goal)

    preview_html = _html.escape(session.plan.human_readable)
    text = (
        f"🧠 <b>План</b>\n\n"
        f"<b>Цель:</b> {_html.escape(goal)}\n\n"
        f"<pre>{preview_html}</pre>\n\n"
        "<i>✅ — запустить. Или напишите уточнение — перестрою план.</i>"
    )
    await status.edit_text(
        text,
        reply_markup=task_approval_keyboard(session.id),
        parse_mode=ParseMode.HTML,
    )
    # Allow refinement: next text message regenerates the plan
    _set_wait(tg_id, "platform")


async def _handle_rag_chat(message: Message) -> None:
    """FR-P11: platform widget chat with RAG + history + save-to-memory button."""
    tg_id = message.from_user.id
    user = platform_svc.get_user(tg_id)
    active = platform_svc.get_active_domains(tg_id)
    if not active:
        await message.answer("ℹ️ Выберите хотя бы один домен в «💾 Память».",
                             parse_mode=ParseMode.HTML)
        return
    if not rag.is_configured():
        await message.answer("⚠️ RAG не настроен (QDRANT_URL).")
        return

    question = message.text or ""
    platform_svc.add_chat_message(tg_id, "user", question)

    # FR-4 / FR-5 / NFR-13: `[контекст]` / `[file@v2]` preprocessing.
    # If the user embedded explicit refs, resolve them and pull the full
    # file(s) into the prompt via assemble_full_context — this bypasses
    # the top-k RAG path so the LLM sees the whole document (with
    # map-reduce summarization when oversize).
    explicit_refs = context_resolver.parse_context_refs(question)
    explicit_context_block = ""
    if explicit_refs:
        resolved_ids: list[str] = []
        for ref in explicit_refs:
            res = context_resolver.resolve_context_ref(tg_id, ref.name)
            if res.matched is not None:
                eid = (
                    getattr(res.matched, "doc_id", None)
                    or getattr(res.matched, "memory_object_id", None)
                )
                if eid:
                    resolved_ids.append(eid)
        if resolved_ids:
            bundle = context_resolver.assemble_full_context(tg_id, resolved_ids)
            chunks: list[str] = []
            for obj in bundle.objects:
                if not obj.content:
                    continue
                marker = "[СВОД]" if obj.used_summarization else "[ПОЛНЫЙ ФАЙЛ]"
                chunks.append(
                    f"{marker} {obj.filename}\n{obj.content}"
                )
            if chunks:
                explicit_context_block = "\n\n---\n\n".join(chunks)

    status = await message.answer("⏳ Ищу в памяти…")
    # FR-P9: query top-k across ALL active domains and merge by score
    all_hits: list[dict] = []
    for dom in active:
        hits = await rag.query_rag(platform_svc.collection_name(tg_id, dom), question, top_k=5)
        for h in hits:
            h["domain"] = dom
        all_hits.extend(hits)
    all_hits.sort(key=lambda h: -h["score"])
    top = all_hits[:5]
    if not top and not explicit_context_block:
        await status.edit_text("❌ В выбранных доменах ничего релевантного не нашёл.")
        return

    # FR-27: drop numbered [N] citations entirely. Keep a deduped source
    # list purely to pick the reply target (so Telegram's quote-preview
    # still jumps to the original file on tap) and to bold filenames if
    # the LLM mentions them inline.
    sources = _dedupe_sources(top) if top else []
    rag_context = "\n\n---\n\n".join(
        f"(файл: {h['filename']})\n{h['text']}"
        for h in top
    )

    # Merge the explicit full-file block (FR-5) with the top-k RAG excerpts.
    # Full files come first so the model sees them as the primary context.
    context_parts: list[str] = []
    if explicit_context_block:
        context_parts.append(explicit_context_block)
    if rag_context:
        context_parts.append(rag_context)
    context = "\n\n===\n\n".join(context_parts)

    # Build chat history for LLM (system + last 10 turns + current context)
    messages = [{
        "role": "system",
        "content": prompt_loader.load("file_search"),
    }]
    messages.extend(user.chat_history[-10:-1])  # prior turns
    messages.append({
        "role": "user",
        "content": f"Контекст:\n{context}\n\nВопрос: {question}",
    })

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        resp = await client.chat.completions.create(
            model=user.model_id, messages=messages, temperature=0.3,
        )
        answer = resp.choices[0].message.content or "(пусто)"
    except Exception as exc:  # noqa: BLE001
        await status.edit_text(f"❌ Ошибка LLM: {exc}")
        return

    platform_svc.add_chat_message(tg_id, "assistant", answer)
    answer_html = _render_answer_with_inline_sources(
        answer, [s["filename"] for s in sources],
    )
    final_text = (
        f"💬 <b>{_html.escape(_model_label(user.model_id))}</b>\n\n"
        f"{answer_html}"
    )

    # FR-27: reply to the top source's upload message — the quote-preview
    # IS the hyperlink. Tap the quoted filename → jump to the original.
    # No numbered citations, no footer — just the answer + reply anchor.
    reply_to = next((s["message_id"] for s in sources if s["message_id"]), None)
    try:
        sent = await message.bot.send_message(
            chat_id=message.chat.id,
            text=final_text,
            reply_markup=platform_answer_keyboard(),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_to_message_id=reply_to,
            allow_sending_without_reply=True,
        )
    except TelegramBadRequest:
        sent = await message.bot.send_message(
            chat_id=message.chat.id,
            text=final_text,
            reply_markup=platform_answer_keyboard(),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
    _track_msg(tg_id, sent.message_id)
    try:
        await status.delete()
    except Exception:  # noqa: BLE001
        pass


# ── FR-27: RAG source rendering (no numbers, no footer) ─────────

def _dedupe_sources(top: list[dict]) -> list[dict]:
    """Dedupe RAG hits by filename in score order. Returns an ordered
    list of `{filename, message_id, domain}` — no idx number anymore.
    First entry is the primary source (reply target)."""
    out: list[dict] = []
    seen: set[str] = set()
    for h in top:
        fn = h.get("filename") or "(без имени)"
        if fn in seen:
            continue
        seen.add(fn)
        out.append({
            "filename": fn,
            "message_id": h.get("message_id"),
            "domain": h.get("domain", ""),
        })
    return out


def _render_answer_with_hyperlinks(text: str) -> str:
    """FR-36: wrap every http(s):// URL in the text with an HTML <a>
    tag so Telegram renders it as a clickable hyperlink. Also strips
    any stray [N] markers the LLM might produce."""
    import re

    escaped = _html.escape(text)
    # Strip stray [N] markers
    escaped = re.sub(r"\s*\[\d+\]", "", escaped)
    # Linkify URLs — match http:// and https:// up to whitespace or <
    escaped = re.sub(
        r"(https?://[^\s<\"]+)",
        r'<a href="\1">\1</a>',
        escaped,
    )
    return escaped


def _render_answer_with_inline_sources(answer: str, filenames: list[str]) -> str:
    """HTML-escape the answer and bold any filename mentions so the user
    visually spots them. No numbered markers, no footer — the tappable
    quote-preview at the top of the message (via `reply_to_message_id`)
    IS the navigation affordance.

    If the model accidentally slips a `[1]` or `[2]` marker back in
    despite the system prompt, strip it so the UX stays clean.
    """
    import re

    text = _html.escape(answer)

    # Strip any stray [N] markers the model might have produced.
    text = re.sub(r"\s*\[\d+\]", "", text)

    # Bold every literal occurrence of a known filename so it stands out.
    # Sort by length descending so "report-final.pdf" bolds before "report".
    for fn in sorted(set(filenames), key=len, reverse=True):
        if not fn:
            continue
        esc_fn = re.escape(_html.escape(fn))
        text = re.sub(
            rf"(?<!<b>){esc_fn}(?!</b>)",
            f"<b>{_html.escape(fn)}</b>",
            text,
        )
    return text


def _extract_text(data: bytes, filename: str) -> str:
    """Best-effort text extraction from uploaded file bytes."""
    name_lower = filename.lower()
    if name_lower.endswith((".txt", ".md", ".json", ".csv", ".log", ".yaml", ".yml")):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            return ""
    if name_lower.endswith(".pdf"):
        try:
            from pypdf import PdfReader  # type: ignore
            reader = PdfReader(io.BytesIO(data))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("pdf extract failed: %s", exc)
            return ""
    if name_lower.endswith(".docx"):
        try:
            from docx import Document as DocxDoc  # type: ignore
            d = DocxDoc(io.BytesIO(data))
            return "\n".join(p.text for p in d.paragraphs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("docx extract failed: %s", exc)
            return ""
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return ""


# ── Live task progress rendering ────────────────────────────────


# Max number of progress lines we keep before trimming the oldest.
# Telegram max message length is 4096 chars; with ~60 chars per line
# this leaves room for the header and the final summary.
_MAX_PROGRESS_LINES = 40


def _format_progress_event(kind: str, node_id: str | None, payload: dict) -> str | None:
    """Render a single runtime progress event as a Telegram HTML line.

    Return None for events we don't want to surface (runtime_done is
    handled separately by the final summary).
    """
    if kind == "planner_done":
        return (
            f"📋 <b>Планировщик:</b> {payload.get('steps', '?')} шагов, "
            f"{payload.get('parallel_groups', 0)} параллель, "
            f"{payload.get('conditional_edges', 0)} условных | "
            f"backend=<code>{payload.get('backend', '?')}</code>"
        )
    if kind == "step_start":
        tool = payload.get("tool")
        tool_tag = f" 🔧 <code>{_html.escape(tool)}</code>" if tool else " 💡 reasoning"
        desc = (payload.get("description") or "")[:90]
        return f"▸ <b>{_html.escape(node_id or '?')}</b>: {_html.escape(desc)}{tool_tag}"
    if kind == "tool_call_done":
        status = payload.get("status")
        if status == "ok":
            hits = payload.get("hits")
            hits_tag = f" ({hits} hits)" if hits is not None else ""
            provider = payload.get("provider") or "?"
            return f"   ✅ tool <code>{_html.escape(str(payload.get('tool', '?')))}</code> via <i>{_html.escape(str(provider))}</i>{hits_tag}"
        err = (payload.get("error") or "")[:80]
        attempt = payload.get("attempt", 1)
        return f"   ⚠️ tool error (attempt {attempt}): <i>{_html.escape(err)}</i>"
    if kind == "tool_retry":
        attempt = payload.get("attempt", "?")
        m = payload.get("max", "?")
        backoff = payload.get("backoff_sec", 0)
        return f"   🔄 retry {attempt}/{m} (wait {backoff:.1f}s)"
    if kind == "critic":
        verdict = payload.get("verdict", "?")
        icon = "✅" if verdict == "pass" else "❌"
        reason = (payload.get("reason") or "")[:120]
        return f"   🧐 critic: {icon} {_html.escape(verdict)} — <i>{_html.escape(reason)}</i>"
    if kind == "process_critic":
        action = payload.get("action", "?")
        icon = "▶️" if action == "continue" else "🛑"
        reason = (payload.get("reason") or "")[:100]
        return f"   🧠 process critic: {icon} <b>{_html.escape(action)}</b> — <i>{_html.escape(reason)}</i>"
    if kind == "alignment":
        drift = float(payload.get("drift", 0))
        icon = "⚠️" if payload.get("should_replan") else "·"
        return f"   🎯 alignment: {icon} drift={drift:.2f}"
    if kind == "synthesising":
        prior = payload.get("prior_count", 0)
        return f"   💡 synthesising from {prior} prior steps…"
    if kind == "synthesis_done":
        chars = payload.get("chars", 0)
        return f"   ✅ synthesis: {chars} chars"
    if kind == "step_done":
        return "   ✅ <b>done</b>"
    if kind == "step_soft_pass":
        concern = (payload.get("concern") or "")[:100]
        return f"   🟡 soft-pass (concern: <i>{_html.escape(concern)}</i>)"
    if kind == "step_abort":
        reason = (payload.get("reason") or "")[:100]
        return f"   🛑 <b>abort</b>: <i>{_html.escape(reason)}</i>"
    return None


async def _run_with_live_progress(
    session_id: str,
    status_message: Message,
    advanced: bool,
) -> "modes_svc.TaskRun | None":
    """Run `run_advanced` (or legacy `execute`) in a worker thread while
    editing `status_message` with a live progress log.

    Architecture:
    - sync progress events go through a `queue.Queue` (thread-safe).
    - async loop polls the queue, batches events, and edits the
      status message at most once every 1.5s (Telegram edit rate).
    - when the worker thread finishes (done or exception), we do a
      final drain + let the caller render the summary.
    """
    progress_q: _queue.Queue = _queue.Queue()

    def _push(kind: str, node_id: str | None, payload: dict) -> None:
        try:
            progress_q.put_nowait((kind, node_id, payload))
        except Exception:  # noqa: BLE001
            pass

    # Kick off the worker thread
    if advanced:
        task = asyncio.create_task(
            asyncio.to_thread(modes_svc.run_advanced, session_id, _push)
        )
    else:
        task = asyncio.create_task(
            asyncio.to_thread(modes_svc.execute, session_id)
        )

    header = "🎯 <b>Выполнение плана</b>\n"
    lines: list[str] = []
    last_edit = 0.0
    min_edit_interval = 1.5  # seconds — Telegram rate limit protection

    async def _render() -> None:
        """Build the current progress text and edit the status message."""
        visible = lines[-_MAX_PROGRESS_LINES:]
        body = "\n".join(visible) if visible else "<i>подожди, запускаю…</i>"
        text = header + "\n" + body
        # Telegram max ~4096 chars. Leave safety margin.
        if len(text) > 3800:
            text = text[:3800] + "\n<i>…(обрезано)</i>"
        try:
            await status_message.edit_text(text, parse_mode=ParseMode.HTML)
        except TelegramBadRequest as e:
            # "message is not modified" fires when content hasn't
            # changed — ignore. Any other bad request: log + swallow.
            if "not modified" not in str(e).lower():
                logger.debug("status edit: %s", e)
        except Exception as exc:  # noqa: BLE001
            logger.debug("status edit unexpected: %s", exc)

    # Main drain loop — runs until the worker task finishes
    while not task.done():
        # Drain any pending events
        changed = False
        while True:
            try:
                kind, node_id, payload = progress_q.get_nowait()
            except _queue.Empty:
                break
            line = _format_progress_event(kind, node_id, payload)
            if line:
                lines.append(line)
                changed = True

        now = _time.monotonic()
        if changed and (now - last_edit) >= min_edit_interval:
            await _render()
            last_edit = now

        await asyncio.sleep(0.3)

    # Final drain once the worker is done
    while True:
        try:
            kind, node_id, payload = progress_q.get_nowait()
        except _queue.Empty:
            break
        line = _format_progress_event(kind, node_id, payload)
        if line:
            lines.append(line)

    # Show the last progress snapshot once more so nothing is lost
    if lines:
        await _render()

    # Await the task — re-raises any exception from the worker
    try:
        await task
    except Exception as exc:  # noqa: BLE001
        logger.exception("task worker failed")
        try:
            await status_message.edit_text(
                f"{header}\n\n❌ <b>Ошибка:</b> <code>{_html.escape(str(exc)[:500])}</code>",
                parse_mode=ParseMode.HTML,
            )
        except Exception:  # noqa: BLE001
            pass
        return None

    # Return the TaskRun from the session
    try:
        session = modes_svc.get_session(session_id)
        return session.run
    except Exception:  # noqa: BLE001
        return None


# ── FR-11 / FR-17 / NFR-9: task approval callbacks ──────────────

def _extract_synthesis_text(run) -> str | None:
    """Find the final synthesis text in the run's results dict.

    Walks the session's advanced_state (FR-20 GraphState) and looks
    for the last step whose output has a `text` key — that's the
    convention our LLM-backed synthesis uses in _synthesize_step.
    Falls back to any result dict that carries `text`, in whatever
    order they appear, so partial runs still surface something
    useful.
    """
    session = None
    try:
        for s in modes_svc._SESSIONS.values():
            if s.run is run:
                session = s
                break
    except Exception:  # noqa: BLE001
        return None
    if session is None or session.advanced_state is None:
        return None
    results = session.advanced_state.results or {}
    # Prefer the LAST step's output if it has text
    for sid in reversed(list(results.keys())):
        val = results.get(sid)
        if isinstance(val, dict) and isinstance(val.get("text"), str):
            return val["text"]
    return None


def _render_run_summary(run) -> str:
    """Compact human-readable summary of a finished / partial task run.

    Two-section layout:
    1. **The actual answer** — pulled from the last synthesis step
       (the one _synthesize_step wrote). This is what the user asked
       for; it goes first.
    2. **Diagnostic block** — critic verdicts, goal alignment drifts,
       trace count, backend, outcome. Collapsed after the answer so
       the user can inspect or ignore it.
    """
    summary = run.result_summary or {}
    outcome = summary.get("outcome", "—")
    reason = summary.get("reason", "")
    backend = summary.get("backend", "")

    parts: list[str] = []

    # 1. Actual synthesised answer — THIS is what the user wanted.
    synthesis = _extract_synthesis_text(run)
    if synthesis:
        # Telegram HTML max message length is 4096 chars; leave room
        # for the diagnostic footer.
        if len(synthesis) > 3500:
            synthesis = synthesis[:3500] + "\n\n<i>…(обрезано, полный ответ в логах)</i>"
        parts.append(f"📝 <b>Ответ:</b>\n\n{_html.escape(synthesis)}")

    # 2. Diagnostic footer
    stage_lines = [
        f"  • {s.node_id}: {'✅' if s.status == 'pass' else '❌'}"
        for s in run.stages
    ]

    critic_lines: list[str] = []
    drift_lines: list[str] = []
    for ev in run.execution_trace:
        if ev.kind == "critic":
            mark = "✅" if ev.payload.get("verdict") == "pass" else "❌"
            critic_lines.append(
                f"  {mark} {ev.node_id}: {_html.escape(str(ev.payload.get('reason', ''))[:120])}"
            )
        elif ev.kind == "alignment":
            drift = float(ev.payload.get("drift", 0.0))
            warn = "⚠️" if ev.payload.get("should_replan") else "·"
            drift_lines.append(
                f"  {warn} {ev.node_id}: drift={drift:.2f}"
            )

    diag: list[str] = []
    if backend:
        diag.append(f"<b>Runtime:</b> <code>{_html.escape(backend)}</code>")
    if stage_lines:
        diag.append("<b>Этапы:</b>\n" + "\n".join(stage_lines))
    if critic_lines:
        diag.append("<b>Critic verdicts:</b>\n" + "\n".join(critic_lines[:8]))
    if drift_lines:
        diag.append("<b>Goal alignment:</b>\n" + "\n".join(drift_lines[:8]))
    diag.append(f"<b>Trace events:</b> {len(run.execution_trace)}")
    diag.append(f"<b>Результат:</b> {_html.escape(str(outcome))}")
    if reason:
        diag.append(f"<b>Причина:</b> {_html.escape(str(reason))}")

    if parts:
        # Collapse the diagnostics behind a separator under the answer.
        parts.append("━━━━━━━━━━━━━━\n<i>Диагностика:</i>\n\n" + "\n\n".join(diag))
    else:
        # Nothing synthesised — diagnostics IS the message.
        parts.extend(diag)
    return "\n\n".join(parts)


@router.callback_query(F.data.startswith("task_approve:"))
async def on_task_approve(callback: CallbackQuery):
    """FR-11 approval gate with **live progress rendering**.

    The pipeline is synchronous (OpenAI + SerpAPI via httpx.Client),
    so we offload it to a worker thread and poll a `queue.Queue` of
    progress events. The status message gets edited at most once
    every 1.5s with the latest progress snapshot (Telegram edit
    rate limit). When the worker finishes we replace the progress
    log with the final summary (synthesis + critic verdicts +
    alignment drifts).

    UX guarantees:
    - `callback.answer()` fires in the first millisecond so the
      Telegram spinner dismisses and the 15s query timeout doesn't
      bite even for 30s+ runs.
    - `edit_reply_markup(None)` kills the approve keyboard so a
      second tap can't start a duplicate pipeline.
    - `asyncio.to_thread` keeps the event loop responsive —
      `/health` and other callbacks keep working during the run.
    """
    await callback.answer()  # dismiss spinner before 15s timeout

    session_id = callback.data.split(":", 1)[1]
    try:
        session = modes_svc.get_session(session_id)
    except KeyError:
        await callback.message.answer(
            "❌ Сессия не найдена или уже завершена",
            parse_mode=ParseMode.HTML,
        )
        return

    # Kill the approval keyboard so a second tap doesn't kick off a
    # duplicate run.
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except TelegramBadRequest:
        pass

    status = await callback.message.answer(
        "🎯 <b>Выполнение плана</b>\n\n<i>запускаю…</i>",
        parse_mode=ParseMode.HTML,
    )

    try:
        modes_svc.approve_plan(session_id)
    except modes_svc.ApprovalRequiredError as e:
        await status.edit_text(
            f"❌ <i>{_html.escape(str(e))}</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    # Run with live progress — uses queue.Queue + rate-limited edits.
    advanced = session.structured_plan is not None
    run = await _run_with_live_progress(session_id, status, advanced=advanced)
    if run is None:
        return  # error already rendered inside the helper

    # FR-17 / FR-22: material replan or goal drift — re-confirm
    if run.state == "awaiting_approval":
        if run.revised_plan is not None:
            revised_block = (
                f"<b>Причина:</b> {_html.escape(run.revised_plan.reason)}\n"
                f"<b>Material diff:</b> да\n\n"
                f"<pre>{_html.escape(run.revised_plan.human_readable)}</pre>"
            )
        else:
            reason = (run.result_summary or {}).get("reason", "goal_drift")
            revised_block = (
                f"<b>Причина:</b> {_html.escape(str(reason))}\n"
                "Goal alignment просела ниже порога — план надо переделать."
            )
        await status.edit_text(
            "🔁 <b>План требует пересборки</b>\n\n" + revised_block,
            reply_markup=task_reapproval_keyboard(session_id),
            parse_mode=ParseMode.HTML,
        )
        return

    icon = "✅" if run.state == "done" else "❌"
    final_text = f"{icon} <b>Задача завершена</b>\n\n{_render_run_summary(run)}"
    # Telegram hard limit 4096; we already trim in _render_run_summary
    # but the header adds ~30 chars — clip to 4000 just in case.
    if len(final_text) > 4000:
        final_text = final_text[:4000] + "\n<i>…(обрезано)</i>"
    try:
        await status.edit_text(final_text, parse_mode=ParseMode.HTML)
    except TelegramBadRequest as e:
        logger.warning("final edit failed: %s", e)
        # Fall back to sending as a new message
        await callback.message.answer(final_text, parse_mode=ParseMode.HTML)


@router.callback_query(F.data.startswith("task_reapprove:"))
async def on_task_reapprove(callback: CallbackQuery):
    """FR-17: user confirms the revised plan — resume execution.

    Same callback-timeout defence as on_task_approve: ACK first,
    disable the keyboard, show a status, offload the sync work.
    """
    await callback.answer()

    session_id = callback.data.split(":", 1)[1]
    try:
        session = modes_svc.get_session(session_id)
    except KeyError:
        await callback.message.answer("❌ Сессия не найдена")
        return

    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except TelegramBadRequest:
        pass

    status = await callback.message.answer(
        "⏳ <b>Запускаю пересобранный план…</b>",
        parse_mode=ParseMode.HTML,
    )

    # Swap the live plan with the revised one and re-arm
    if session.run and session.run.revised_plan:
        session.plan.graph = session.run.revised_plan.graph
        session.plan.human_readable = session.run.revised_plan.human_readable
    session.state = "executing"
    session.run = None  # fresh run against the revised graph

    try:
        run = await asyncio.to_thread(modes_svc.execute, session_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("task_reapprove pipeline failed")
        await status.edit_text(
            f"❌ Неожиданная ошибка:\n<code>{_html.escape(str(exc)[:400])}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    await status.edit_text(
        f"✅ <b>Задача выполнена (revised)</b>\n\n{_render_run_summary(run)}",
        parse_mode=ParseMode.HTML,
    )


@router.callback_query(F.data.startswith("task_reject:"))
async def on_task_reject(callback: CallbackQuery):
    session_id = callback.data.split(":", 1)[1]
    modes_svc._SESSIONS.pop(session_id, None)
    await callback.message.answer(
        "❌ Задача отменена. Можете поставить новую.",
        parse_mode=ParseMode.HTML,
    )
    await callback.answer()


# ── Standalone catch-all text handler ────────────────────────────
#
# In the nrnm fork there is no strategy/assessment text router to call
# `platform_handle_message` first — so we wire it up directly. Any text
# that isn't a command gets routed through platform_handle_message; if
# the user isn't in a platform wait state we nudge them to /start.

@router.message(F.text & ~F.text.startswith("/"))
async def on_platform_text(message: Message):
    handled = await platform_handle_message(message)
    if handled:
        return
    await message.answer(
        "👋 Откройте меню через /start и выберите «🧠 ИИ-платформа».",
        parse_mode=ParseMode.HTML,
    )
