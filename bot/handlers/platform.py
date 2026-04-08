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

import html as _html
import io
import logging
import uuid
from dataclasses import dataclass

from aiogram import Router, F
from aiogram.enums import ContentType, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import CallbackQuery, Message

import config
from services import platform as platform_svc
from services import rag
from bot.keyboards.inline import (
    platform_menu_keyboard,
    platform_model_keyboard,
    platform_memory_keyboard,
    platform_new_domain_keyboard,
    platform_domain_keyboard,
    platform_doc_keyboard,
    platform_answer_keyboard,
)

logger = logging.getLogger(__name__)
router = Router()


# Per-user wait state. Values:
#   "platform"    — on platform widget, any text = RAG question, any file = upload
#   "new_domain"  — waiting for new-domain name
_PLATFORM_WAIT: dict[int, str] = {}


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
    return await message.bot.send_message(
        chat_id=message.chat.id, text=text, **kwargs,
    )


def _model_label(model_id: str) -> str:
    return next(
        (label for label, mid in config.PLATFORM_MODELS if mid == model_id),
        model_id or "не выбрана",
    )


def _menu_text(user: platform_svc.PlatformUser) -> str:
    active = platform_svc.get_active_domains_for(user)
    text = (
        "🧠 <b>ИИ-платформа</b>\n\n"
        f"<b>Модель:</b> {_html.escape(_model_label(user.model_id))}\n"
        f"<b>Домены:</b> {_html.escape(', '.join(active) or 'не выбраны')}\n\n"
        "<i>Отправьте сообщение — я отвечу на основе выбранных доменов.\n"
        "Отправьте файл (txt/md/pdf/docx) — он будет добавлен в память.</i>"
    )
    if not rag.is_configured():
        text += "\n\n<i>⚠️ RAG не сконфигурирован (QDRANT_URL).</i>"
    if not active:
        text += "\n\n<i>ℹ️ Выберите домен в «💾 Память» чтобы начать диалог.</i>"
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
        callback.message, _menu_text(user),
        reply_markup=platform_menu_keyboard(
            _model_label(user.model_id), platform_svc.get_active_domains(tg_id)
        ),
        parse_mode=ParseMode.HTML,
    )
    await callback.answer()


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
    text = (
        f"📁 <b>Домен:</b> {_html.escape(domain.name)}\n"
        f"<b>Документов:</b> {len(domain.documents)}\n"
        f"<b>Выбран для RAG:</b> {'да' if is_active else 'нет'}\n\n"
        "<i>📎 Чтобы добавить файлы — отправьте их прямо в чат "
        "(txt/md/pdf/docx). Они автоматически попадут в этот домен "
        "(после нажатия «Выбрать для RAG»).</i>"
    )
    await _replace_widget(
        callback.message, text,
        reply_markup=platform_domain_keyboard(idx, domain.documents, is_active),
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


# ── FR-P11: reset chat context ───────────────────────────────────

@router.callback_query(F.data == "platform_reset")
async def on_platform_reset(callback: CallbackQuery):
    tg_id = callback.from_user.id
    platform_svc.reset_chat(tg_id)
    await callback.answer("Контекст очищен")
    await on_platform_menu(callback)


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
    # FR-P19: swap the inline button to its «✅ Сохранено» state so the user
    # gets a persistent visual confirmation under the answer message itself.
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
    platform_svc.register_document(
        tg_id, domain_name, filename, n_chunks,
        message_id=message.message_id,  # FR-P18
    )
    await status.edit_text(
        f"✅ <b>{_html.escape(filename)}</b> — {n_chunks} фрагментов в домене "
        f"<b>{_html.escape(domain_name)}</b>.",
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

    # 2. Platform chat — any text becomes a RAG question
    if wait == "platform":
        if message.document:
            await _ingest_file(message)
            return True
        if not message.text:
            return True
        await _handle_rag_chat(message)
        return True

    return False


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
    if not top:
        await status.edit_text("❌ В выбранных доменах ничего релевантного не нашёл.")
        return

    # FR-P18: build numbered source list deduped by filename so each unique
    # source gets a single [N] marker. Each chunk references its file's number,
    # which the LLM is instructed to use inline.
    sources, chunk_to_idx = _build_sources(top)
    context = "\n\n---\n\n".join(
        f"[{chunk_to_idx[i]}] ({h['filename']}): {h['text']}"
        for i, h in enumerate(top)
    )

    # Build chat history for LLM (system + last 10 turns + current context)
    messages = [{
        "role": "system",
        "content": (
            "Ты — помощник, отвечающий на основе контекста из базы знаний пользователя. "
            "Каждый фрагмент контекста начинается с маркера вида [1], [2] и т.д. "
            "Когда используешь информацию из фрагмента — обязательно ставь его номер "
            "в квадратных скобках сразу после соответствующего утверждения, например: "
            "«Apollo 11 высадился на Луне в 1969 году [2].» "
            "Если в контексте нет ответа — так и скажи, не выдумывай."
        ),
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
    answer_html = _format_answer_with_citations(answer, sources)
    sources_html = _render_sources_footer(sources)
    final_text = (
        f"💬 <b>{_html.escape(_model_label(user.model_id))}</b> "
        f"<i>(домены: {_html.escape(', '.join(active))})</i>\n\n"
        f"{answer_html}\n\n"
        f"<b>Источники:</b>\n{sources_html}"
    )

    # FR-P18: send the answer as a reply to the top source's upload message
    # so users can tap the quoted preview to jump straight to the source file.
    # Drop the «⏳ Ищу в памяти…» status message since the answer replaces it.
    reply_to = next((s["message_id"] for s in sources if s["message_id"]), None)
    try:
        await message.bot.send_message(
            chat_id=message.chat.id,
            text=final_text,
            reply_markup=platform_answer_keyboard(),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_to_message_id=reply_to,
            allow_sending_without_reply=True,
        )
    except TelegramBadRequest:
        # Reply target is gone — send without reply.
        await message.bot.send_message(
            chat_id=message.chat.id,
            text=final_text,
            reply_markup=platform_answer_keyboard(),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
    try:
        await status.delete()
    except Exception:  # noqa: BLE001
        pass


# ── FR-P18: source citation helpers ──────────────────────────────

def _build_sources(top: list[dict]) -> tuple[list[dict], list[int]]:
    """Dedupe top hits by filename. Returns:
       * `sources`  — ordered list of `{idx, filename, message_id, domain}`,
                      one per unique filename, idx is 1-based.
       * `chunk_to_idx` — list parallel to `top`, mapping each chunk to its
                          source idx so the LLM context can prefix each
                          fragment with its citation number.
    """
    sources: list[dict] = []
    seen: dict[str, int] = {}
    chunk_to_idx: list[int] = []
    for h in top:
        fn = h.get("filename") or "(без имени)"
        if fn not in seen:
            seen[fn] = len(sources) + 1
            sources.append({
                "idx": seen[fn],
                "filename": fn,
                "message_id": h.get("message_id"),
                "domain": h.get("domain", ""),
            })
        chunk_to_idx.append(seen[fn])
    return sources, chunk_to_idx


def _render_sources_footer(sources: list[dict]) -> str:
    """Render the «Источники» list as `[N] 📎 filename` lines.
    Filenames are wrapped in `<code>` for visual distinction. The clickable
    deep-link is provided by `reply_to_message_id` on the answer message
    itself (Telegram's native quote-preview), since custom `tg://` href
    schemes are not whitelisted by Bot API HTML mode."""
    lines: list[str] = []
    for s in sources:
        fn = _html.escape(s["filename"])
        lines.append(f"[{s['idx']}] 📎 <code>{fn}</code>")
    return "\n".join(lines)


def _format_answer_with_citations(answer: str, sources: list[dict]) -> str:
    """FR-P19: HTML-escape the LLM answer, **bold** every `[N]` citation, and
    if the model forgot to cite anything, append a tail like
    «(Источники: [1], [2])» referencing every available source so the link
    between the answer and the source list is preserved."""
    import re

    escaped = _html.escape(answer)
    # Find all valid [N] markers (where N matches a known source idx)
    valid_idxs = {s["idx"] for s in sources}
    found: set[int] = set()

    def repl(m: "re.Match[str]") -> str:
        n = int(m.group(1))
        if n in valid_idxs:
            found.add(n)
            return f"<b>[{n}]</b>"
        return m.group(0)  # leave [N] alone if it doesn't match a real source

    bolded = re.sub(r"\[(\d+)\]", repl, escaped)

    if not found and sources:
        # Model forgot to cite — append a fallback tail referencing all sources
        tail = ", ".join(f"<b>[{s['idx']}]</b>" for s in sources)
        bolded = f"{bolded}\n\n<i>(Источники: {tail})</i>"

    return bolded


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
