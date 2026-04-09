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
    active_instrument: str = "chat",
) -> InlineKeyboardMarkup:
    """Main platform widget.

    Layout (top to bottom):
    1. **🧠 СУПЕРАГЕНТ** — biggest, topmost (FR-31 + user request)
    2. **Instrument picker row** — 💬 Чат | 📁 Память | 🌐 Веб
    3. **🤖 Model picker**
    4. (if instrument = file_search) **📁 Домен(ы)** — domain selector

    No 🔄 Сбросить контекст here — it lives inline under each answer.
    """
    if active_instrument == "chat" and active_mode != "chat":
        active_instrument = active_mode

    model_label = active_model or "не выбрана"

    # Domain label for the memory instrument
    n_doms = len(active_domains)
    if n_doms == 0:
        dom_label = "не выбран"
    elif n_doms == 1:
        dom_label = active_domains[0]
    else:
        dom_label = ", ".join(active_domains)
    dom_button_text = f"📁 Домен: {dom_label}" if n_doms <= 1 else f"📁 Домены: {dom_label}"

    # Instrument picker row: Чат | Память | Веб
    instruments = [
        ("chat", "💬", "Чат"),
        ("file_search", "📁", "Память"),
        ("web_search", "🌐", "Веб"),
    ]
    inst_buttons = []
    for name, icon, label in instruments:
        mark = "✅ " if name == active_instrument else ""
        inst_buttons.append(
            InlineKeyboardButton(
                text=f"{mark}{icon} {label}",
                callback_data=f"platform_instrument:{name}",
            )
        )

    rows = [
        # 🧠 СУПЕРАГЕНТ — top, prominent
        [InlineKeyboardButton(
            text="🧠 СУПЕРАГЕНТ",
            callback_data="platform_superagent",
        )],
        inst_buttons,
        [InlineKeyboardButton(text=f"🤖 {model_label}", callback_data="platform_model")],
    ]

    # File tree selector only for the "Память" (file_search) instrument
    if active_instrument == "file_search":
        rows.append([InlineKeyboardButton(
            text="📁 Файлы",
            callback_data="ftree:/",
        )])

    return InlineKeyboardMarkup(inline_keyboard=rows)


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
    """Per-answer inline keyboard. Every response gets two buttons:

    1. 💾 Сохранить в память / ✅ Сохранено — save the answer as a
       new versioned document in the active domain.
    2. 🔄 Обновить контекст — clears chat history so subsequent
       messages start fresh. Replaces the old widget-level reset
       button — context reset belongs here, next to the content it
       affects, not on the static main menu.
    """
    if saved:
        save_btn = InlineKeyboardButton(text="✅", callback_data="platform_save_noop")
    else:
        save_btn = InlineKeyboardButton(text="💾", callback_data="platform_save_answer")
    reset_btn = InlineKeyboardButton(text="🔄", callback_data="platform_reset")
    return InlineKeyboardMarkup(inline_keyboard=[
        [save_btn, reset_btn],
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


def platform_domain_keyboard(
    domain_idx: int,
    documents: list,
    is_active: bool,
    mcp_count: int = 0,
) -> InlineKeyboardMarkup:
    """FR-P16 + FR-24 single-domain detail screen.

    Layout:
      • «✅ Выбран» / «◻️ Выбрать» — single-select activation.
      • 🔧 Инструменты (N MCP) — opens the MCP registry screen.
      • Per-document row (clickable to view/delete).
      • «🗑 Удалить домен» — full domain delete with all docs.
      • «◀️ К доменам» — back to memory list.
    Upload is done by sending a file directly into the chat (FR-P13)."""
    buttons = []
    toggle_label = "✅ Выбран для RAG" if is_active else "◻️ Выбрать для RAG"
    buttons.append([InlineKeyboardButton(
        text=toggle_label, callback_data=f"platform_domain_pick:{domain_idx}",
    )])
    buttons.append([InlineKeyboardButton(
        text=f"🔧 Инструменты ({mcp_count} MCP)",
        callback_data=f"platform_mcp_list:{domain_idx}",
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


# ── FR-39..43: File tree keyboards ────────────────────────────────

def file_tree_keyboard(
    path: str,
    children: list,
    parent_path: str | None = None,
) -> InlineKeyboardMarkup:
    """Navigation keyboard for a folder in the file tree.

    Shows:
    - 📂 subfolders (tap = enter)
    - 📄 files (tap = enter file context)
    - ➕ Папка — create a new subfolder
    - 📎 Выбрать все — select all files in this folder for RAG
    - ◀️ Назад — go up one level (or back to main menu if at root)
    """
    buttons = []

    # Children: folders first, then files
    folders = sorted([c for c in children if c.is_folder], key=lambda c: c.name)
    files = sorted([c for c in children if not c.is_folder], key=lambda c: c.name)

    for f in folders:
        buttons.append([InlineKeyboardButton(
            text=f"📂 {f.name}",
            callback_data=f"ftree:{f.path}",
        )])
    for f in files:
        label = f"📄 {f.name}"
        if f.num_chunks:
            label += f" ({f.num_chunks} фр.)"
        buttons.append([InlineKeyboardButton(
            text=label,
            callback_data=f"ftree:{f.path}",
        )])

    # Action buttons
    buttons.append([
        InlineKeyboardButton(text="➕ Папка", callback_data=f"ftree_mkdir:{path}"),
        InlineKeyboardButton(text="📎 Выбрать все", callback_data=f"ftree_scope:{path}"),
    ])

    # Back button
    if parent_path is not None:
        buttons.append([InlineKeyboardButton(
            text="◀️ Назад",
            callback_data=f"ftree:{parent_path}",
        )])
    else:
        buttons.append([InlineKeyboardButton(
            text="◀️ К меню",
            callback_data="platform_menu",
        )])

    return InlineKeyboardMarkup(inline_keyboard=buttons)


def file_context_keyboard(file_path: str, parent_path: str) -> InlineKeyboardMarkup:
    """Keyboard shown when user enters a single file context.
    They can chat within this file's scope or go back."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(
            text="🗑 Удалить файл",
            callback_data=f"ftree_delete:{file_path}",
        )],
        [InlineKeyboardButton(
            text="◀️ Назад",
            callback_data=f"ftree:{parent_path}",
        )],
    ])


# ── FR-23/24: MCP management keyboards ──────────────────────────

def platform_mcp_list_keyboard(domain_idx: int, mcps: list) -> InlineKeyboardMarkup:
    """List of MCPs inside a domain. Tap an entry to view / edit / delete.
    Tap ➕ to add a new one."""
    buttons = []
    for i, m in enumerate(mcps):
        icon = "🔌" if m.is_builtin else "🌐"
        buttons.append([InlineKeyboardButton(
            text=f"{icon} {m.name}",
            callback_data=f"platform_mcp_view:{domain_idx}:{i}",
        )])
    buttons.append([InlineKeyboardButton(
        text="➕ Добавить MCP",
        callback_data=f"platform_mcp_new:{domain_idx}",
    )])
    buttons.append([InlineKeyboardButton(
        text="◀️ К домену",
        callback_data=f"platform_domain_open:{domain_idx}",
    )])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def platform_mcp_view_keyboard(domain_idx: int, mcp_idx: int) -> InlineKeyboardMarkup:
    """Single MCP detail screen — edit / delete / back."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(
            text="✏️ Редактировать",
            callback_data=f"platform_mcp_edit:{domain_idx}:{mcp_idx}",
        )],
        [InlineKeyboardButton(
            text="🗑 Удалить",
            callback_data=f"platform_mcp_delete:{domain_idx}:{mcp_idx}",
        )],
        [InlineKeyboardButton(
            text="◀️ К инструментам",
            callback_data=f"platform_mcp_list:{domain_idx}",
        )],
    ])


def platform_mcp_cancel_keyboard(domain_idx: int) -> InlineKeyboardMarkup:
    """Shown during the add/edit FSM — single Cancel button."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(
            text="◀️ Отмена",
            callback_data=f"platform_mcp_list:{domain_idx}",
        )],
    ])
