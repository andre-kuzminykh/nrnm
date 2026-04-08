# nrnm — ИИ-платформа (RAG Telegram bot)

Standalone Telegram bot с функционалом ИИ-платформы из `andre-ai-maturity`:
— личные домены-базы-знаний, загрузка файлов (pdf/docx/txt/md), RAG-чат
через Qdrant + OpenAI с numbered-цитатами `[1]`, `[2]` и reply-to-source
linking.

## Возможности

- 🤖 **Выбор LLM** — GPT-4o/mini/nano из `config.PLATFORM_MODELS`
- 📁 **Изолированные домены** — каждый = отдельная Qdrant коллекция
- 📎 **Загрузка файлов** — pdf, docx, txt, md, json, csv, yaml
- 💬 **RAG-чат** — query поверх активного домена, history 30 turns
- 🔢 **Numbered citations** — `[1]`, `[2]` inline + блок «Источники» с filename
- 🔗 **Reply-to-source** — ответ приходит как reply на исходный файл,
  тап по quote-preview скроллит к файлу в чате
- 💾 **Сохранить в память** — ответы LLM можно ingest'ить как новые документы
- 🔄 **Сброс контекста** — очистка chat_history
- ♻️ **Replace-widget** — каждый клик создаёт свежее сообщение, старый widget
  удаляется (чат читается как чистый стрим)

## Быстрый старт

### 1. Склонируй репу

```bash
git clone https://github.com/andre-kuzminykh/nrnm.git
cd nrnm
```

### 2. Создай `.env`

```bash
cp .env.example .env
```

Заполни минимум:
- `BOT_TOKEN` — токен от @BotFather
- `LLM_API_KEY` — OpenAI API ключ

Опционально (но для RAG **обязательно**):
- `QDRANT_URL` — URL Qdrant Cloud cluster (https://xxx.qdrant.io)
- `QDRANT_API_KEY` — API ключ Qdrant

Без `QDRANT_URL` бот всё равно запустится: меню, выбор модели и управление
доменами работают, но ingest и query возвращают пустой результат, UI
показывает предупреждение «⚠️ RAG не сконфигурирован».

### 3. Запусти через Docker

```bash
docker compose up -d --build
docker compose logs -f bot
```

Или без докера:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### 4. Открой бот в Telegram

Нажми `/start`, нажми «🧠 ИИ-платформа» → создай домен → выбери его →
загрузи файл → задай вопрос.

## Qdrant Cloud setup

1. Зарегистрируйся на https://cloud.qdrant.io
2. Создай free-tier cluster (1 GB)
3. Скопируй URL (https://xxx-xxx.gcp.cloud.qdrant.io:6333) в `QDRANT_URL`
4. Создай API key в UI → копируй в `QDRANT_API_KEY`
5. Коллекции создаются автоматически при первом ingest,
   naming: `platform-{tg_user_id}-{safe_domain_name}`

## Структура проекта

```
nrnm/
├── main.py                      # entry point
├── config.py                    # env vars + PLATFORM_MODELS list
├── requirements.txt             # aiogram, openai, qdrant-client, pypdf, python-docx
├── Dockerfile                   # python:3.11-slim
├── docker-compose.yml           # bot service + data-persist volume
├── .env.example
├── bot/
│   ├── handlers/
│   │   ├── start.py             # /start + main_menu
│   │   └── platform.py          # все callbacks ИИ-платформы + text dispatcher
│   └── keyboards/
│       └── inline.py            # start_keyboard + platform_*_keyboard
└── services/
    ├── platform.py              # PlatformUser/Domain/Document + pickle-persist
    └── rag.py                   # Qdrant + OpenAI embeddings + chunking
```

## Архитектурные заметки

### Persistence
`services/platform.py::_PLATFORM_STORE` сериализуется через `pickle` в
`/app/data-persist/platform_store.pkl` (atomic write через `tmp + os.replace`).
Запись выполняется в каждом мутирующем вызове. На старте `main.py` вызывает
`load_platform_from_disk()`. Qdrant векторы живут в Qdrant Cloud, в pickle
НЕ попадают.

**⚠️ Volume mount обязателен** — без `bot-data` volume (или локальной bind-mount
директории) pickle будет стираться при каждом пересоздании контейнера.

### Text dispatch flow
1. `CommandStart()` в `start_router` — ловит `/start` первым
2. `F.text & ~F.text.startswith("/")` в `platform_router` — ловит всё
   остальное, делегирует в `platform_handle_message(message)`
3. Если юзер в wait-state `new_domain` — текст становится именем домена
4. Если в `platform` — текст идёт в RAG-чат
5. Иначе — показывается nudge "откройте /start"

### Replace-widget (FR-P17)
Все callback-хендлеры платформы используют `_replace_widget(message, text, ...)`
вместо `edit_text` — это удаляет старый widget и создаёт новый. Цель —
чат читается как стрим свежих widget'ов. Файлы пользователя и LLM-ответы
НЕ удаляются.

### Numbered citations (FR-P18/P19)
1. Document.message_id сохраняется при upload
2. Пробрасывается в Qdrant payload каждого чанка
3. `_build_sources(top)` дедуплицирует по filename, нумерует 1..N
4. System prompt инструктирует LLM ставить `[N]` inline
5. `_format_answer_with_citations` оборачивает валидные `[N]` в `<b>`
6. Если LLM забыл — fallback `(Источники: [1], [2])` в конец
7. Финальный ответ отправляется как reply на top source's message_id —
   Telegram quote-preview работает как deep-link на исходный файл

### Batched embeddings (FR-P5b)
`embed_texts` слайсит input на батчи по 500 chunks чтобы не упереться в
OpenAI лимит `max_tokens_per_request: 300000`. Большие PDF/DOCX (до ~10 MB)
ingestятся без ошибок.

### Graceful degradation без Qdrant
Если `QDRANT_URL` пустой, `rag._get_qdrant()` возвращает `None`, все
операции (`ingest_document`, `query_rag`, `delete_document_vectors`)
возвращают пустой результат, UI показывает предупреждение. Пользователь
всё ещё может создавать домены и выбирать модель.

## Env vars reference

| Var | Required | Default | Описание |
|---|---|---|---|
| `BOT_TOKEN` | ✅ | — | Токен от @BotFather |
| `LLM_API_KEY` | ✅ | — | OpenAI API ключ |
| `LLM_BASE_URL` | | `https://api.openai.com/v1` | Для совместимых прокси (openrouter etc.) |
| `QDRANT_URL` | | `""` | Без него RAG noop, но меню работает |
| `QDRANT_API_KEY` | | `""` | — |
| `EMBEDDING_MODEL` | | `text-embedding-3-small` | OpenAI embedding модель |
| `EMBEDDING_DIM` | | `1536` | Размерность эмбеддинга (3-small = 1536) |

## Кастомизация

### Сменить список доступных LLM
Отредактируй `config.py::PLATFORM_MODELS`:
```python
PLATFORM_MODELS = [
    ("Моя любимая модель", "my-model-id"),
    ("GPT-4o", "gpt-4o"),
]
```

### Сменить welcome-текст / меню
`bot/handlers/start.py::WELCOME_TEXT` и `bot/keyboards/inline.py::start_keyboard`.

### Добавить свои команды
Создай новый роутер в `bot/handlers/` и зарегистрируй в `main.py` ПЕРЕД
`platform_router` (иначе catch-all F.text перехватит).

## Origin

Этот код — порт ИИ-платформы из
[andre-ai-maturity](https://github.com/andre-kuzminykh/andre-ai-maturity),
где она является одним из модулей большого бота. В nrnm она выделена в
standalone-приложение для переиспользования с другим BOT_TOKEN.

Feature requirements: FR-P1..P19 (см. `SPEC.md` в исходной репе).
