# SPEC — Telegram AI Platform v1

> Статус: зафиксировано (v1).
> Источник: внутренняя продуктовая спека «Telegram AI Platform v1».
> Идентификаторы требований (FR-*, NFR-*) — стабильные, на них завязаны тесты в `tests/`.

---

## 1. Feature Context

**Feature**
Telegram AI Platform v1: unified Memory + Chat/Task modes + supervised super-agent + basic tools.

**Description (Goal / Scope)**
Создать Telegram-бота как AI-платформу, где:

* пользователь хранит все знания, файлы и результаты в едином разделе **Память**
* пользователь работает в одном из двух режимов:
  * **Чат**
  * **Задачи**
* в режиме **Чат** бот отвечает на вопросы с учётом Памяти
* в режиме **Задачи** бот строит **полный план/граф действий**, показывает его пользователю, получает подтверждение и затем выполняет
* инструменты v1 ограничены:
  * **Web Search**
  * **PDF Parser**
  * **`[контекст]`** как явное подключение полного файла из Памяти

**Client**
Power users, founders, researchers, operators.

**Problem**
Обычные чат-боты:

* не имеют единого пространства памяти
* плохо работают с файлами как с управляемым контекстом
* не разделяют режим «быстро поговорить» и режим «решить задачу»
* не согласуют полный план действий перед выполнением
* не сохраняют версионность контекста и файлов

**Solution**
Telegram-native AI workspace, где:

* все документы, файлы, заметки и артефакты находятся в разделе **Память**
* пользователь явно выбирает режим **Чат** или **Задачи**
* в режиме **Чат** бот использует Память как источник контекста
* в режиме **Задачи** бот:
  * формирует полный граф действий
  * показывает человеку понятный текстовый план
  * получает подтверждение
  * выполняет согласованный граф
  * при отклонении пересобирает граф и повторно согласует изменения
* пользователь может явно подключить файл через `[контекст]`

**Metrics**

* Time-to-first-response в режиме Чат < 10 sec
* Time-to-plan-preview в режиме Задачи < 15 sec
* ≥ 90% успешных retrieval-ответов из Памяти
* ≥ 85% корректного разрешения ссылок `[контекст]`
* 100% файлов в Памяти имеют version history
* 100% runs в режиме Задачи имеют plan preview + execution trace + result summary

---

## 2. User Stories and Use Cases

### User Story 1 — Unified Memory (US-1)

**Role**: Пользователь платформы.

**User Story**
As a user, I want all my files, notes, and saved results to live in one unified Memory space, so that I can reuse them as persistent context.

**UX / User Flow**
Пользователь открывает раздел **Память**, загружает файлы, создаёт папки/секции внутри Памяти, сохраняет ответы и заметки, потом использует их в чате и задачах.

#### Use Case UC-1.1 — сохранение объекта в Память

* **Given**: пользователь открыл раздел Память.
* **When**: загружает файл или сохраняет результат в Память.
* **Then**: система сохраняет объект, создаёт метаданные и версию.
* **Input**: файл / текст / ответ бота, optional имя, optional папка.
* **Output**: memory object created, object ID, version = v1, confirmation.
* **State**: объект сохранён, version history инициализирована, объект доступен для Чата и Задач.

Functional Requirements

| ID   | Requirement                                                                                                   |
| ---- | ------------------------------------------------------------------------------------------------------------- |
| FR-1 | Система должна поддерживать единый раздел Память как контейнер всех файлов, заметок и сохранённых результатов |
| FR-2 | Система должна позволять загружать файлы и сохранять текстовые объекты в Память                               |
| FR-3 | Система должна автоматически создавать первую версию объекта при сохранении                                   |

Non-Functional Requirements

| ID    | Requirement                                                                      |
| ----- | -------------------------------------------------------------------------------- |
| NFR-1 | Сохранение объекта в Память должно быть идемпотентным при повторном подтверждении |
| NFR-2 | Каждый объект Памяти должен иметь уникальный ID, timestamps и metadata            |
| NFR-3 | Все объекты Памяти должны быть доступны для последующего retrieval                |

#### Use Case UC-1.2 — `[контекст]` / явное подключение файла

* **Given**: в Памяти уже есть сохранённые файлы.
* **When**: пользователь указывает `[контекст]` или `[имя_файла]` в сообщении.
* **Then**: система подключает выбранный объект Памяти как полный контекст текущего запроса.
* **Input**: сообщение + `[контекст]` / `[file_name]` / `[file_name@v2]`.
* **Output**: answer with attached memory context, resolved list, optional disambiguation.
* **State**: context resolution залогирован, объект прикреплён к сессии/run.

Functional Requirements

| ID   | Requirement                                                                                                                                |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| FR-4 | Система должна поддерживать явное подключение объектов Памяти через ссылочный синтаксис                                                    |
| FR-5 | Система должна использовать выбранный файл как полный контекст, даже если под капотом он обрабатывается через parse/chunk/summary pipeline |

Non-Functional Requirements

| ID    | Requirement                                                                                 |
| ----- | ------------------------------------------------------------------------------------------- |
| NFR-4  | Resolver должен использовать порядок exact match → alias match → fuzzy match                                                                 |
| NFR-5  | При указании версии система должна использовать ровно указанную версию; без версии — latest                                                  |
| NFR-13 | Если полный контент файла превышает лимит context window, assemble должен запускать map-reduce summarization: чанки → суммари → финальный свод |

---

### User Story 2 — Chat / Task modes (US-2)

**Role**: пользователь, который хочет либо общаться, либо запускать задачу.

**User Story**
As a user, I want the bot to work in either Chat mode or Task mode, so that I can either ask quick questions or launch controlled task execution.

**UX / User Flow**
Пользователь выбирает режим:

* **Чат** — для быстрых ответов
* **Задачи** — для постановки целей и выполнения по плану

#### Use Case UC-2.1 — режим Чат

* **Given**: пользователь включил режим Чат.
* **When**: задаёт вопрос.
* **Then**: система отвечает как чат-бот, используя Память и явный `[контекст]`, если указан.
* **Input**: user message, optional memory refs, optional uploaded file.
* **Output**: answer, optional suggestion to save result to Memory.
* **State**: chat session active, task graph НЕ создаётся, retrieval trace залогирован если использована Память.

Functional Requirements

| ID   | Requirement                                                                 |
| ---- | --------------------------------------------------------------------------- |
| FR-6 | Система должна поддерживать режим Чат как быстрый conversational mode        |
| FR-7 | В режиме Чат система должна использовать Память как источник контекста       |
| FR-8 | В режиме Чат система не должна строить execution graph для простых запросов  |

Non-Functional Requirements

| ID    | Requirement                                                                                    |
| ----- | ---------------------------------------------------------------------------------------------- |
| NFR-6 | Ответы в режиме Чат должны приходить быстрее, чем в режиме Задачи, при сопоставимой сложности  |
| NFR-7 | Режим Чат должен работать без обязательного предварительного планирования                      |

#### Use Case UC-2.2 — режим Задачи

* **Given**: пользователь включил режим Задачи.
* **When**: формулирует цель естественным языком.
* **Then**: система строит полный граф действий, показывает текстовый план, ждёт согласования и только потом стартует.
* **Input**: goal, optional memory refs, optional constraints.
* **Output**: plan preview, planned steps, expected outputs, approval request.
* **State**: task run created, draft graph готов, execution не стартовал до approval.

Functional Requirements

| ID    | Requirement                                                                                               |
| ----- | --------------------------------------------------------------------------------------------------------- |
| FR-9  | Система должна поддерживать режим Задачи как supervised execution mode                                    |
| FR-10 | В режиме Задачи система должна строить полный граф выполнения до старта execution                         |
| FR-11 | Система должна показывать пользователю текстовое представление полного плана и ждать явного подтверждения |

Non-Functional Requirements

| ID    | Requirement                                                                    |
| ----- | ------------------------------------------------------------------------------ |
| NFR-8 | План должен быть понятен пользователю и не показывать внутренний граф напрямую |
| NFR-9 | Execution в режиме Задачи не должен стартовать без explicit approval           |

---

### User Story 3 — Supervised execution with basic tools (US-3)

**Role**: пользователь, который хочет решать задачи с инструментами.

**User Story**
As a user, I want the Task mode to use only the approved basic tools and follow the agreed graph, so that execution is predictable and controllable.

**UX / User Flow**
Пользователь ставит цель в режиме Задачи. Бот строит план, где может использовать:

* Web Search
* PDF Parser
* `[контекст]`

После подтверждения бот выполняет задачу по плану и пересобирает граф только если застрял или не получил ожидаемый результат.

#### Use Case UC-3.1 — выполнение approved graph

* **Given**: пользователь подтвердил план.
* **When**: система выполняет согласованный граф.
* **Then**: последовательно/параллельно проходит этапы, вызывает только доступные инструменты, сверяет результат с expected outputs, при необходимости инициирует replanning.
* **Input**: approved plan, memory context, optional explicit `[контекст]`, allowed tools = `web_search` + `pdf_parser`.
* **Output**: final result, execution summary, созданные memory objects, revised plan request если граф существенно изменился.
* **State**: approved graph executing, node/stage results logged, replan только после deviation detection.

Functional Requirements

| ID    | Requirement                                                                                                                                          |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| FR-12 | Система должна выполнять задачу строго по согласованному графу                                                                                       |
| FR-13 | Система должна использовать в v1 только инструменты Web Search и PDF Parser                                                                          |
| FR-14 | Система должна поддерживать `[контекст]` как встроенный механизм полного подключения файла из Памяти                                                 |
| FR-15 | Для каждого этапа система должна иметь expected result и acceptance criteria                                                                         |
| FR-16 | Если фактический результат не соответствует expected result, система должна пересобрать граф выполнения                                              |
| FR-17 | Если пересобранный граф materially меняет будущие действия, система должна повторно показать обновлённый план пользователю и получить подтверждение  |
| FR-18 | План должен генерироваться LLM из текстовой цели как структурированная последовательность шагов с привязкой к инструментам                           |
| FR-19 | План должен поддерживать параллельные ветви (fan-out/fan-in), последовательные цепочки и условные переходы (conditional edges)                       |
| FR-20 | Executor должен компилировать план в граф LangGraph и запускать его (parallel + sequential + conditional)                                            |
| FR-21 | После каждого шага запускается LLM-критик: верификация результата против expected_result, бинарный verdict pass/fail с обоснованием                  |
| FR-22 | После каждого шага запускается goal-alignment проверка: drift от исходной цели, при превышении порога — триггер replan/redesign                      |
| FR-23 | Инструменты регистрируются как MCP-записи: `{name, url, token, description, created_at, updated_at}`. Запись можно добавлять, редактировать, удалять |
| FR-24 | Каждый домен ведёт собственный MCP-реестр. При создании нового домена автоматически бутстрапится дефолтный `web_search` (builtin://serpapi)         |
| FR-25 | LLM-планировщик получает описания MCP активного домена и использует их как available_tools для привязки шагов                                         |
| FR-26 | Tool invocation идёт через `services/mcp_client.py`: `builtin://*` → в-процессная реализация, `http(s)://*` → POST на MCP-сервер с bearer token      |
| FR-27 | RAG-ответ показывает источники как tap-to-original через Telegram `reply_to_message_id`. Никаких `[N]` маркеров, никакого `Источники:` footer          |
| FR-28 | Убрать кнопки Чат/Задачи. В главном виджете — выбор **инструмента**: Чат, Поиск по файлам, Веб-поиск. Текстовое сообщение при выбранном инструменте уходит именно в него |
| FR-29 | Каждый инструмент имеет свой набор параметров. Поиск по файлам: мультивыбор доменов. Веб-поиск: нет параметров. Чат: нет параметров |
| FR-30 | Кнопка «💾 Память» располагается под виджетом текущего инструмента (а не в отдельном меню) |
| FR-31 | Отдельная большая кнопка **🤖 СУПЕРАГЕНТ** в главном виджете. При нажатии — ввод задачи → LangGraph pipeline → план → approval → execution |
| FR-32 | После построения плана система отправляет **картинку графа LangGraph** (`compiled.get_graph().draw_mermaid_png()`) как фото в Telegram |
| FR-33 | После согласования план-сообщение **стирается** и в том же thread'е начинается live progress log с трейсами каждого шага |
| FR-34 | План отображается в формате `1. → 1.1. → 1.2. ‖ 2.` — с подзадачами и маркерами параллелизма |
| FR-35 | Среди инструментов доступен **ask_user** — агент может запросить уточнение у пользователя mid-run (inline keyboard «ответить» + FSM ожидания текста) |
| FR-36 | Все URL в результатах (web search hits, ссылки на файлы) рендерятся как HTML `<a href="...">title</a>` гиперссылки прямо в тексте ответа |
| FR-37 | Пользователь может **редактировать файл через LLM**: выбрать файл из Памяти, отправить промт-инструкцию, LLM модифицирует содержимое, новая версия сохраняется (Rule 6) |
| FR-38 | При обновлении файла система **предлагает обновить зависимые файлы** — те, что ранее использовались вместе (same task run / same `[контекст]` набор). Пользователь подтверждает → LLM выполняет тот же набор действий с новым контекстом |

Non-Functional Requirements

| ID     | Requirement                                                                                                                                                              |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| NFR-10 | Все tool calls и node results должны быть traceable                                                                                                                      |
| NFR-11 | Replanning должен сохранять уже валидные завершённые этапы, если они остаются релевантными                                                                               |
| NFR-12 | Каждый task run в режиме Задачи должен содержать `plan_preview`, `execution_trace` и `result_summary` (100%)                                                             |
| NFR-14 | LLM-компоненты (planner, critic, alignment) должны иметь deterministic stub fallback на случай отсутствия API-ключа / langgraph — для offline dev и CI                   |
| NFR-15 | Ошибки MCP-клиента (HTTP / network / невалидный response) должны деградировать в `ToolCallResult(status="error")` и триггерить Rule-5 tool_failure replan (FR-16)          |
| NFR-16 | Все действия пользователя и агента (выбор инструмента, смена домена, tool calls, файловые операции, промты) пишутся в structured action log (JSON per event)               |
| NFR-17 | Файловое хранение должно быть pluggable: pickle-in-process (v1), S3/Postgres/GDrive/Git/Notion (v2+). Интерфейс: `save(id, content) -> url`, `load(id) -> content`       |

---

# User Story 4 — LLM-driven planner + LangGraph runtime + critic loop

**Role**: Пользователь, ставящий сложную задачу.

**Story**: Как пользователь, я хочу, чтобы бот сам разбирал мою цель на структурированный план шагов с явными вызовами инструментов, запускал их через настоящий LangGraph (с параллелизмом и условиями), и после каждого шага критически проверял результат и сверялся с целью — чтобы не уходить в дрейф.

**UX**: Пользователь в режиме Задачи пишет цель → бот строит план через LLM → показывает граф (с метками PARALLEL / IF / SEQ) → подтверждение → каждый шаг исполняется + критикуется + сверяется с целью → итоговый ответ либо replan при дрейфе.

#### Use Case UC-4.1 — LLM план + LangGraph + critic loop

* **Given**: пользователь в режиме Задачи, OpenAI API доступен.
* **When**: ставит цель.
* **Then**: planner вызывает LLM → возвращает `StructuredPlan` с шагами, `depends_on`, `parallel_groups`, `conditional_edges`. Executor компилирует в `langgraph.StateGraph`. После каждого ноды — critic + alignment. Критик-fail → `replan_signal=critic_failed`. Drift > порога → `replan_signal=goal_drift`. На выходе run несёт plan_preview, execution_trace, result_summary.
* **Input**: goal text, attached memory.
* **Output**: structured plan + run state + critic verdicts + alignment scores per step.
* **State**: LangGraph compiled and invoked; после каждой ноды — verdict trace event.

---

## 3. Architecture / Solution

### 3.1 Client Side

| Area                    | Value                                                                                               |
| ----------------------- | --------------------------------------------------------------------------------------------------- |
| Client Type             | Telegram bot                                                                                        |
| User Entry Points       | Текстовые сообщения, загрузка файлов, inline buttons, переключатель режима                          |
| Main Screens / Commands | `/start`, `/memory`, `/mode`, `/chat`, `/task`, `/save`, `/versions`                                |
| Input / Output Format   | NL input, file upload, `[контекст]`, plan preview, confirmation buttons, task summary               |

### 3.2 Backend Services

| Service            | Responsibility                                                        | API                                                                           |
| ------------------ | --------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Telegram Gateway   | Приём сообщений, файлов, callback actions, mode switching             | `Update → internal request`                                                   |
| Memory Service     | Хранение всех файлов, заметок, результатов; версии; retrieval         | `CreateMemoryObject`, `ReadMemoryObject`, `ResolveContextRef`, `ListVersions` |
| Chat Runtime       | Обработка запросов в режиме Чат                                        | `ChatRequest → ChatResponse`                                                  |
| Planner / Graph    | Полный граф в режиме Задачи                                            | `TaskRequest → PlanDraft`                                                     |
| Approval Controller| Согласование исходного и пересобранного плана                          | `PlanDraft → ApprovalDecision`                                                |
| Task Executor      | Исполнение approved graph                                              | `ApprovedPlan → TaskRun`                                                      |
| Tool Layer         | Web Search, PDF Parser (v1)                                            | `ToolCallRequest → ToolCallResult`                                            |
| Verification Layer | Проверка этапов against expected result                                | `StageOutput → VerificationVerdict`                                           |
| Replanning Engine  | Пересборка графа при deviation                                         | `RunState + FailureContext → RevisedPlanDraft`                                |

### 3.3 Data Architecture

**Main entities**: `User`, `MemoryFolder`, `MemoryObject`, `MemoryObjectVersion`, `ChatSession`, `TaskSession`, `PlanDraft`, `ApprovedPlan`, `TaskRun`, `StageRun`, `ToolCall`, `TraceEvent`.

**Relationships**

```
User 1:N MemoryFolder
MemoryFolder 1:N MemoryObject
MemoryObject 1:N MemoryObjectVersion
User 1:N ChatSession
User 1:N TaskSession
TaskSession 1:1 ApprovedPlan
TaskRun 1:N StageRun
StageRun 1:N ToolCall
```

**Data flow**

```
Telegram input
  → Mode Router
  → Chat Runtime | Planner
  → Approval
  → Executor
  → Verification
  → Replanning (if needed)
  → Result
  → Save to Memory
```

### 3.4 Infrastructure

* Telegram Bot API
* Python backend
* PostgreSQL (metadata)
* Object storage (Memory objects)
* Vector index (Memory retrieval)
* Redis (session state, queues)
* LLM provider
* Web Search integration — SerpAPI (`SERPAPI_API_KEY`). В v1 это
  единственный провайдер; при отсутствии ключа tool graceful-fallback'ает
  на детерминированный stub так, чтобы executor-pipeline оставался
  проходимым в offline dev / CI. HTTP-ошибки SerpAPI маппятся в
  `status="error"` и триггерят replan (FR-16, Rule 5 trigger #3).
* PDF Parser service
* Logging / tracing stack

---

## 4. Work Plan — Mapping Use Case → Tasks

| Use Case | Task ID | Task                                                           | Dependencies | DoD                                                                                             |
| -------- | ------- | -------------------------------------------------------------- | ------------ | ----------------------------------------------------------------------------------------------- |
| UC-1.1   | T-1     | Unified Memory + versioned objects                             | Storage      | Файлы и тексты сохраняются в Память с версиями                                                  |
| UC-1.2   | T-2     | `[контекст]` и context resolver                                | T-1          | Ссылки на файлы из Памяти корректно подключаются как полный контекст                            |
| UC-2.1   | T-3     | Режим Чат                                                      | T-1, T-2     | Чат-режим отвечает с учётом Памяти                                                              |
| UC-2.2   | T-4     | Режим Задачи с полным планом и approval flow                   | T-1, T-2     | Полный граф строится до старта и согласуется с пользователем                                    |
| UC-3.1   | T-5     | Task execution + tools + verification + replanning             | T-4          | Approved graph исполняется, использует web search/pdf parser, при сбое пересобирается           |

---

## 5. Product Rules

### Rule 1 — Unified Memory
«Память» — это один раздел, где лежит всё: файлы, PDF, заметки, сохранённые ответы, результаты задач. Отдельных пользовательских сущностей «домены» и «файлы» как разных модулей нет — они входят в Память как структура хранения.

### Rule 2 — Only Two Modes
В продукте есть только два режима: **Чат** и **Задачи**.

### Rule 3 — Limited Tools in v1
В v1 доступны только:

* **Web Search**
* **PDF Parser**
* **`[контекст]`** — встроенный механизм явного подключения файла из Памяти

### Rule 4 — Full Planning Before Execution
В режиме Задачи: сначала строится **весь граф**, потом показывается **весь план**, потом пользователь его **подтверждает**, только после этого начинается выполнение.

### Rule 5 — Replanning Only on Deviation
Graph re-evaluation запускается только если:

* не получен expected result
* этап не прошёл acceptance criteria
* tool failure blocks progress
* отсутствует нужный input
* пользователь изменил constraints

### Rule 6 — Versioning
Каждый объект Памяти versioned:

* обновление создаёт новую версию
* старая версия сохраняется
* `[file@v2]` закрепляет конкретную версию
* без версии используется latest

---

## 6. Requirement → Test Mapping

Тесты лежат в `tests/` и названы так, что каждый ID требования попадает в имя теста (`test_fr_1_*`, `test_nfr_4_*`, и т.д.).

| ID     | Test file                        |
| ------ | -------------------------------- |
| FR-1   | `tests/test_us1_memory.py`       |
| FR-2   | `tests/test_us1_memory.py`       |
| FR-3   | `tests/test_us1_memory.py`       |
| FR-4   | `tests/test_us1_memory.py`       |
| FR-5   | `tests/test_us1_memory.py`       |
| NFR-1  | `tests/test_us1_memory.py`       |
| NFR-2  | `tests/test_us1_memory.py`       |
| NFR-3  | `tests/test_us1_memory.py`       |
| NFR-4  | `tests/test_us1_memory.py`       |
| NFR-5  | `tests/test_us1_memory.py`       |
| FR-6   | `tests/test_us2_modes.py`        |
| FR-7   | `tests/test_us2_modes.py`        |
| FR-8   | `tests/test_us2_modes.py`        |
| FR-9   | `tests/test_us2_modes.py`        |
| FR-10  | `tests/test_us2_modes.py`        |
| FR-11  | `tests/test_us2_modes.py`        |
| NFR-6  | `tests/test_us2_modes.py`        |
| NFR-7  | `tests/test_us2_modes.py`        |
| NFR-8  | `tests/test_us2_modes.py`        |
| NFR-9  | `tests/test_us2_modes.py`        |
| FR-12  | `tests/test_us3_execution.py`    |
| FR-13  | `tests/test_us3_execution.py`    |
| FR-14  | `tests/test_us3_execution.py`    |
| FR-15  | `tests/test_us3_execution.py`    |
| FR-16  | `tests/test_us3_execution.py`    |
| FR-17  | `tests/test_us3_execution.py`    |
| NFR-10 | `tests/test_us3_execution.py`    |
| NFR-11 | `tests/test_us3_execution.py`    |
| NFR-12 | `tests/test_us3_execution.py`    |
| NFR-13 | `tests/test_us1_memory.py`       |
| FR-18  | `tests/test_us4_advanced_planner.py` |
| FR-19  | `tests/test_us4_advanced_planner.py` |
| FR-20  | `tests/test_us4_advanced_planner.py` |
| FR-21  | `tests/test_us4_advanced_planner.py` |
| FR-22  | `tests/test_us4_advanced_planner.py` |
| NFR-14 | `tests/test_us4_advanced_planner.py` |
| FR-23  | `tests/test_us5_mcp.py`              |
| FR-24  | `tests/test_us5_mcp.py`              |
| FR-25  | `tests/test_us5_mcp.py`              |
| FR-26  | `tests/test_us5_mcp.py`              |
| FR-27  | `tests/test_us5_mcp.py`              |
| NFR-15 | `tests/test_us5_mcp.py`              |
| FR-28  | `tests/test_us6_tool_ui.py`          |
| FR-29  | `tests/test_us6_tool_ui.py`          |
| FR-30  | `tests/test_us6_tool_ui.py`          |
| FR-31  | `tests/test_us6_tool_ui.py`          |
| FR-32  | `tests/test_us6_tool_ui.py`          |
| FR-33  | `tests/test_us6_tool_ui.py`          |
| FR-34  | `tests/test_us6_tool_ui.py`          |
| FR-35  | `tests/test_us6_tool_ui.py`          |
| FR-36  | `tests/test_us6_tool_ui.py`          |
| FR-37  | `tests/test_us7_file_edit.py`        |
| FR-38  | `tests/test_us7_file_edit.py`        |
| NFR-16 | `tests/test_us6_tool_ui.py`          |
| NFR-17 | `tests/test_us7_file_edit.py`        |

> На момент фиксации v1-спеки многие поведения ещё не реализованы в коде (`services/platform.py` описывает legacy domain-based API из FR-P1..P19). Тесты под ещё не реализованные требования помечены `pytest.skip(...)` с явным `TODO` — они образуют spec-driven roadmap и активируются по мере реализации соответствующих модулей.

### 6.1 Product Rule → Requirement mapping

| Rule                                          | Покрывают требования           |
| --------------------------------------------- | ------------------------------ |
| R1. Unified Memory                            | FR-1, FR-2, NFR-3              |
| R2. Only Two Modes                            | FR-6, FR-9                     |
| R3. Limited Tools in v1 (web / pdf / ctx)     | FR-13, FR-14                   |
| R4. Full Planning Before Execution            | FR-10, FR-11, NFR-9            |
| R5. Replanning Only on Deviation              | FR-16 (+ все 5 суб-триггеров)  |
| R6. Versioning (новая версия, старая живёт)   | FR-3, NFR-5                    |

### 6.2 Rule 5 — триггеры replanning

Replanning (FR-16) должен активироваться **только** при одном из 5 условий:

1. Не получен `expected_result` у stage.
2. Stage не прошёл `acceptance_criteria`.
3. Tool failure blocks progress.
4. Отсутствует нужный input.
5. Пользователь изменил constraints.

В остальных случаях replanning не запускается — покрыто negative-тестом.
