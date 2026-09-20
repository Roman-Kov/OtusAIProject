# rag-kb — локальная база знаний (MCP-сервер)

> **[REPORT.md](REPORT.md) — история создания проекта: процесс разработки с ИИ-инструментами, ключевые проблемы и их решения, разбор промпта.**

## Что это

MCP-сервер «база знаний»: индексирует локальную папку с документами (`.md`, `.txt`, `.py`, `.js`, `.ts`, `.json`, `.yaml`) и отвечает на вопросы по их содержимому. Внутри — Corrective RAG на LangGraph: переформулировка запроса, гибридный поиск (BM25 + векторы, слияние через Reciprocal Rank Fusion), LLM-оценка релевантности найденных фрагментов, генерация ответа только на их основе — с источниками.

Всё работает локально и без платных API: LLM — Ollama с моделью `qwen2.5:3b`, эмбеддинги — `nomic-embed-text` (или встроенная модель ChromaDB), векторное хранилище — персистентный ChromaDB. Любой MCP-совместимый агент подключается по HTTP.

Подробности устройства — в [ARCHITECTURE.md](ARCHITECTURE.md).

## Быстрый старт

Нужен только Docker:

```bash
git clone https://github.com/Roman-Kov/OtusAIProject.git
cd OtusAIProject
docker compose up
```

Что происходит при первом запуске:

- собирается образ `rag-kb`;
- сервис `ollama-init` скачивает модели: `qwen2.5:3b` (~1,9 ГБ) и `nomic-embed-text` (~0,3 ГБ), суммарно ~2,2 ГБ;
- после завершения скачивания поднимается MCP-сервер: <http://localhost:8000/mcp>.

Важно: индексация и ответы выполняются на CPU и занимают время (эмбеддинги всех чанков папки, вызовы LLM на каждый вопрос) — для локальной установки без GPU это нормально. Модели удерживаются в RAM (`keep_alive`), а при старте сервер прогревает их в фоне, поэтому первый вопрос после `docker compose up` обычно не ждёт загрузки. Поэтому `index_folder` запускает индексацию в фоне и отвечает мгновенно: следите за ходом через `index_status`, пока `indexing_in_progress` не станет `false`. Дальнейшая проверка — через инструменты ниже (например, `index_folder("./sample_docs")` и `ask_question`).

## Подключение к VSCode Copilot

Файл `.vscode/mcp.json` уже лежит в репозитории:

```json
{
  "servers": {
    "rag-kb": {
      "type": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

1. Откройте папку проекта в VS Code — workspace-конфигурация MCP подхватится автоматически.
2. Command Palette (`Ctrl+Shift+P`) → `MCP: List Servers` → `rag-kb` → Start. Инструменты появятся в агентном режиме Copilot Chat.

Если вы клонировали репозиторий в другую папку проекта — просто положите файл с этим содержимым в `<проект>/.vscode/mcp.json`.

Подойдёт и любой другой MCP-совместимый агент (Claude Desktop, opencode и т. п.): достаточно указать URL `http://localhost:8000/mcp`.

## Инструменты

| Инструмент | Вход | Что делает | LLM |
| --- | --- | --- | --- |
| `index_folder` | `path`, `pattern="**/*"` | Запускает индексацию в фоне и отвечает сразу: сканирует папку по поддерживаемым расширениям, режет документы на чанки (~1200 символов, перекрытие 200), считает эмбеддинги и сохраняет в ChromaDB, заменяя старую версию уже проиндексированных файлов; перестраивает BM25-индекс. Ход и итог — в `index_status`. Повторный вызов во время индексации безопасен (`already_running`). | нет (только эмбеддинги) |
| `ask_question` | `question` | Полный RAG-пайплайн: переформулировка запроса → гибридный поиск → LLM-оценка релевантности фрагментов → генерация ответа по релевантным; если релевантных нет — до 2 повторов с расширенной формулировкой. Локальная модель на CPU может думать минуты: ответ возвращается синхронно, но если он не успел подготовиться за `RAGKB_ASK_WAIT_SECONDS` (25 c, меньше типовых клиентских таймаутов), придёт `status="in_progress"` с `retry_after_seconds` — подождите столько секунд, повторите тот же вопрос, и готовый ответ придёт из кеша мгновенно. Ошибки не роняют инструмент: приходят `status="error"` с текстом причины. | да |
| `find_relevant_docs` | `query`, `top_k=5` (1–50) | Гибридный поиск без генерации ответа: возвращает сами фрагменты с файлом и позицией — когда нужны цитаты или точное место в файлах. | нет (эмбеддинг запроса) |
| `index_status` | — | Статистика базы: число файлов, число чанков, время последней индексации; ход фоновой индексации (`indexing_in_progress`, `progress` вида «3/17») и итог последнего запуска (`last_report`). | нет |

Пример сценария (данные из `sample_docs/`; числа могут немного отличаться):

```
index_status()
  → {"files": 0, "chunks": 0, "last_indexed_at": null,
     "indexing_in_progress": false, "progress": null, "last_report": null}

index_folder("./sample_docs")
  → {"status": "started", "path": "./sample_docs", "pattern": "**/*"}   # отвечает мгновенно

index_status()   # во время индексации — опрашивайте, пока indexing_in_progress не станет false
  → {"files": 5, "chunks": 210, "last_indexed_at": null,
     "indexing_in_progress": true, "progress": "5/17", "last_report": null}

index_status()   # после завершения
  → {"files": 17, "chunks": 795, "last_indexed_at": "2026-09-16T10:00:00+00:00",
     "indexing_in_progress": false, "progress": "17/17",
     "last_report": {"files": 17, "chunks": 795, "seconds": 123.4, "errors": []}}

ask_question("Кто правит городом Кальдера?")
  → {"answer": "Кальдерой правит Владычица Пепла Исольда — ей 127 лет…",
     "sources": ["…/sample_docs/README.md", "…/sample_docs/lore/towns/calderra.md"]}
```

## Проверочные факты

Демонстрационная база `sample_docs/` (внутренняя вики фанатского дополнения «Трон Пепла» к Heroes of Might and Magic III) собрана так, что каждый факт ниже гарантированно присутствует в текстах — формулировки сверены с `tests/test_sample_docs.py` (`VERIFICATION_FACTS`). По этой таблице удобно проверять качество ответов: задайте вопрос и сверьте ответ с фактом и источником.

| № | Факт | Где искать в sample_docs | Вопрос для проверки |
| --- | --- | --- | --- |
| 1 | Кальдерой правит Владычица Пепла Исольда | `README.md`, `lore/towns/calderra.md` | Кто правит городом Кальдера? |
| 2 | Исольде 127 лет (возраст продлевает огненная клятва) | `README.md`, `lore/heroes.md`, `lore/timeline.md` | Сколько лет Исольде? |
| 3 | Кальдера основана в 812 году | `README.md`, `lore/timeline.md`, `lore/towns/calderra.md` | Когда была основана Кальдера? |
| 4 | Магнус Чёрный Молот присягнул Кальдере в 897 году | `README.md`, `lore/heroes.md`, `lore/world-overview.md` | Когда Магнус Чёрный Молот присягнул Кальдере? |
| 5 | Битва у Трёх Кратеров — 14 октября 903 года | `README.md`, `lore/timeline.md`, `lore/lands.md` | Когда произошла Битва у Трёх Кратеров? |
| 6 | Пепельная война с некромантами Дейи — 889–891 годы | `lore/world-overview.md`, `lore/timeline.md`, `lore/towns/relations.md` | В какие годы шла Пепельная война? |
| 7 | Лиара Ветрокрылая открыла Пепельный проход в 907 году | `README.md`, `lore/heroes.md`, `lore/world-overview.md` | Кто и когда открыл Пепельный проход? |
| 8 | Лавовый Дракон — урон 40-63 | `README.md`, `lore/creatures.md`, `balance/creature-stats.md` | Какой урон у Лавового Дракона? |
| 9 | Обсидианный голем — ремонт 47 золота за ХП | `README.md`, `lore/creatures.md`, `lore/towns/calderra.md` | Сколько стоит ремонт Обсидианного голема? |
| 10 | Пепельная гарпия — «Пепельная завеса», 25% уклонения | `README.md`, `lore/creatures.md`, `balance/creature-stats.md` | Что даёт «Пепельная завеса» и сколько процентов уклонения? |
| 11 | «Сердце Кальдеры» — +7 к силе магии, +2 к знаниям | `README.md`, `balance/artifacts.md`, `guides/campaign.md` | Какие бонусы у «Сердца Кальдеры»? |
| 12 | «Регалии Пепла» — 4 предмета, полный комплект +15% огненного урона | `README.md`, `balance/artifacts.md` | Сколько предметов в «Регалиях Пепла» и что даёт полный комплект? |
| 13 | Кампания «Огонь и Прах» — 7 миссий (889–907 годы) | `README.md`, `guides/campaign.md` | Сколько миссий в кампании «Огонь и Прах»? |
| 14 | Население — 12 408 душ по переписи 909 года | `lore/timeline.md`, `lore/towns/calderra.md` | Каково население Кальдеры по переписи 909 года? |
| 15 | «Ночь Обсидиана» — ежегодно 30 ноября | `README.md`, `guides/lore-writing.md`, `lore/timeline.md` | Когда отмечается «Ночь Обсидиана»? |

Совет из самой базы (`sample_docs/README.md`, «Как задавать вопросы базе»): формулируйте вопрос с именами собственными и пишите числа цифрами — поиск по ключевым словам ищет точную форму («40-63», «25%»).

## Конфигурация

Все настройки — переменные окружения с префиксом `RAGKB_` (pydantic-settings, `src/rag_kb/config.py`). Основные:

| Переменная | По умолчанию | Описание |
| --- | --- | --- |
| `RAGKB_HOST` | `0.0.0.0` | Адрес MCP-сервера |
| `RAGKB_PORT` | `8000` | Порт MCP-сервера (endpoint — `/mcp`) |
| `RAGKB_OLLAMA_BASE_URL` | `http://localhost:11434` | Адрес Ollama |
| `RAGKB_LLM_MODEL` | `qwen2.5:3b` | Модель LLM для ответов и грейдинга |
| `RAGKB_OLLAMA_KEEP_ALIVE_SEC` | `2592000` | Удержание моделей Ollama в RAM, секунды (30 суток; короткое значение даёт холодный старт на первом вопросе) |
| `RAGKB_EMBEDDING_PROVIDER` | `chromadb` | Провайдер эмбеддингов: `chromadb` или `ollama` |
| `RAGKB_OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Модель эмбеддингов Ollama (при `RAGKB_EMBEDDING_PROVIDER=ollama`) |
| `RAGKB_CHROMA_DIR` | `data/chroma` | Каталог персистентного хранилища ChromaDB |
| `RAGKB_TOP_K` | `6` | Сколько чанков выдаёт гибридный поиск |
| `RAGKB_ASK_WAIT_SECONDS` | `25` | Сколько `ask_question` ждёт ответ до отдачи `in_progress` с подсказкой переспросить |

Пример — в `.env.example`. Полный список (чанкинг, RRF, параметры графа) — в [ARCHITECTURE.md](ARCHITECTURE.md).

Переключение эмбеддингов:

- `RAGKB_EMBEDDING_PROVIDER=chromadb` (по умолчанию) — встроенная функция ChromaDB (ONNX all-MiniLM-L6-v2), модель скачивается автоматически при первом запуске; отдельный Ollama для эмбеддингов не нужен.
- `RAGKB_EMBEDDING_PROVIDER=ollama` — `nomic-embed-text` через Ollama; так настроено в `docker-compose.yml`.

Эмбеддинги разных провайдеров несравнимы между собой: меняйте провайдер до первой индексации либо переиндексируйте папки заново.

## Тесты

```bash
uv sync
uv run pytest
```

100 тестов, сеть не нужна: unit-тесты узлов графа, RRF, BM25, чанкинга, загрузчиков, индексера и сторов — на `FakeLLM`/`FakeEmbedder` из `tests/conftest.py`; e2e-тест MCP-инструментов через `fastmcp.Client` с ChromaDB во временном каталоге; плюс проверки `sample_docs` (26 проверочных фактов, суммарный размер не меньше 500 КБ).

## Запуск без Docker

Нужен локальный Ollama; скачайте модели:

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

Затем:

```bash
uv sync
uv run python -m rag_kb.server
```

Сервер поднимется на <http://localhost:8000/mcp>. По умолчанию `RAGKB_EMBEDDING_PROVIDER=chromadb` — для эмбеддингов достаточно автоматически скачиваемой модели MiniLM; чтобы использовать `nomic-embed-text` из Ollama, задайте `RAGKB_EMBEDDING_PROVIDER=ollama`.

## Структура проекта

```text
OtusAIProject/
├── src/rag_kb/
│   ├── app.py            # FastMCP: регистрация 4 инструментов
│   ├── server.py         # входная точка: сборка зависимостей и запуск сервера
│   ├── config.py         # Settings (pydantic-settings, префикс RAGKB_)
│   ├── types.py          # LoadedDoc, Chunk, IndexReport, IndexStats
│   ├── llm.py            # протокол LLM + OllamaLLM (ChatOllama)
│   ├── indexing/         # loaders, chunking, embedder, indexer
│   ├── retrieval/        # stores (ChromaDB), bm25_store, hybrid (RRF)
│   └── graph/            # LangGraph: state, nodes, builder
├── tests/                # unit/ + e2e/, conftest.py (FakeLLM, FakeEmbedder)
├── sample_docs/          # демонстрационная база знаний «Трон Пепла»
├── .github/workflows/    # CI: ruff, pytest, docker build
├── docker-compose.yml    # ollama + ollama-init (pull моделей) + rag-kb
├── Dockerfile
├── ARCHITECTURE.md       # архитектура проекта
└── pyproject.toml
```
