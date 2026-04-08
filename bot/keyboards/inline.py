"""Inline keyboards for the standalone ИИ-платформа bot.

Only the platform-related keyboards are here. `start_keyboard` is a
minimal main menu with a single «🧠 ИИ-платформа» button.
"""

from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton


def start_keyboard() -> InlineKeyboardMarkup:
    """Minimal main menu — just the ИИ-платформа entry point."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🧠 ИИ-платформа", callback_data="platform_menu")],
    ])


# ── ИИ-платформа keyboards (FR-P1..P19) ──────────────────────────

def platform_menu_keyboard(
    active_model: str,
    active_domains: list[str],
    active_mode: str = "chat",
) -> InlineKeyboardMarkup:
    """FR-P1 / FR-P11 / FR-6 / FR-9: main platform widget.

    Adds a v1 mode toggle at the very top — the two modes are Чат and
    Задачи (Rule 2). Active mode shown with ✅."""
    model_label = active_model or "не выбрана"
    doms = ", ".join(active_domains) if active_domains else "не выбран"
    chat_mark = "✅ " if active_mode == "chat" else ""
    task_mark = "✅ " if active_mode == "task" else ""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text=f"{chat_mark}💬 Чат", callback_data="platform_mode:chat"),
            InlineKeyboardButton(text=f"{task_mark}🎯 Задачи", callback_data="platform_mode:task"),
        ],
        [InlineKeyboardButton(text=f"🤖 {model_label}", callback_data="platform_model")],
        [InlineKeyboardButton(text=f"💾 Память: {doms}", callback_data="platform_memory")],
        [InlineKeyboardButton(text="🔄 Сбросить контекст", callback_data="platform_reset")],
        [InlineKeyboardButton(text="◀️ Главное меню", callback_data="main_menu")],
    ])


def task_approval_keyboard(session_id: str) -> InlineKeyboardMarkup:
    """FR-11 / NFR-9: two-button approval gate for a draft task plan.

    Shown under the plan preview. Accept -> `execute()`; Reject drops
    the session and returns the user to the platform widget.
    """
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Подтвердить план", callback_data=f"task_approve:{session_id}"),
            InlineKeyboardButton(text="❌ Отменить", callback_data=f"task_reject:{session_id}"),
        ],
    ])


def task_reapproval_keyboard(session_id: str) -> InlineKeyboardMarkup:
    """FR-17: material replan triggered — user must re-confirm the
    revised plan before the executor resumes."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Подтвердить новый план", callback_data=f"task_reapprove:{session_id}"),
            InlineKeyboardButton(text="❌ Прервать задачу", callback_data=f"task_reject:{session_id}"),
        ],
    ])


def platform_answer_keyboard(saved: bool = False) -> InlineKeyboardMarkup:
    """FR-P12 / FR-P19: per-answer «сохранить в память» button.
    When `saved=True` the button label becomes «✅ Сохранено в память» and
    its callback is `noop` so additional taps do nothing."""
    if saved:
        label = "✅ Сохранено в память"
        cb = "platform_save_noop"
    else:
        label = "💾 Сохранить в память"
        cb = "platform_save_answer"
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=label, callback_data=cb)],
    ])


def platform_model_keyboard(models: list[tuple[str, str]], active: str) -> InlineKeyboardMarkup:
    """FR-P2: pick one of the configured LLMs."""
    buttons = []
    for label, model_id in models:
        check = "✅ " if model_id == active else ""
        buttons.append([InlineKeyboardButton(
            text=f"{check}{label}", callback_data=f"platform_model_pick:{model_id}",
        )])
    buttons.append([InlineKeyboardButton(text="◀️ Назад", callback_data="platform_menu")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def platform_memory_keyboard(domains: list, active_domains: set) -> InlineKeyboardMarkup:
    """FR-P3 / FR-P15 / FR-P16: list of domains. One button per row — click
    OPENS the domain detail screen (where the user can pick / delete /
    upload files). The active domain is marked with `✅`."""
    buttons = []
    for i, d in enumerate(domains):
        check = "✅ " if d.name in active_domains else "▫️ "
        buttons.append([InlineKeyboardButton(
            text=f"{check}{d.name} ({len(d.documents)})",
            callback_data=f"platform_domain_open:{i}",
        )])
    buttons.append([InlineKeyboardButton(text="➕ Новый домен", callback_data="platform_domain_new")])
    buttons.append([InlineKeyboardButton(text="◀️ Назад", callback_data="platform_menu")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def platform_new_domain_keyboard() -> InlineKeyboardMarkup:
    """FR-P10: during new-domain input only a «Back» button is shown —
    all other controls hide until the user types the name."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Отмена", callback_data="platform_memory")],
    ])


def platform_domain_keyboard(domain_idx: int, documents: list, is_active: bool) -> InlineKeyboardMarkup:
    """FR-P16: single-domain detail screen.

    Layout:
      • «✅ Выбран» / «◻️ Выбрать» — single-select activation.
      • Per-document row (clickable to view/delete).
      • «🗑 Удалить домен» — full domain delete with all docs.
      • «◀️ К доменам» — back to memory list.
    Upload is done by sending a file directly into the chat (FR-P13)."""
    buttons = []
    toggle_label = "✅ Выбран для RAG" if is_active else "◻️ Выбрать для RAG"
    buttons.append([InlineKeyboardButton(
        text=toggle_label, callback_data=f"platform_domain_pick:{domain_idx}",
    )])
    for j, doc in enumerate(documents):
        buttons.append([
            InlineKeyboardButton(
                text=f"📄 {doc.filename} ({doc.num_chunks} фр.)",
                callback_data=f"platform_doc_view:{domain_idx}:{j}",
            ),
        ])
    buttons.append([InlineKeyboardButton(
        text="🗑 Удалить домен", callback_data=f"platform_domain_delete:{domain_idx}",
    )])
    buttons.append([InlineKeyboardButton(text="◀️ К доменам", callback_data="platform_memory")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def platform_doc_keyboard(domain_idx: int, doc_idx: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🗑 Удалить",
                              callback_data=f"platform_doc_delete:{domain_idx}:{doc_idx}")],
        [InlineKeyboardButton(text="◀️ Назад",
                              callback_data=f"platform_domain_open:{domain_idx}")],
    ])
