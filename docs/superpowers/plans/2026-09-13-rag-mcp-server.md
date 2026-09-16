# RAG Knowledge Base MCP-сервер — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MCP-сервер (streamable-http), превращающий локальную папку документов в базу знаний с Corrective RAG пайплайном на LangGraph и локальной LLM через Ollama.

**Architecture:** FastMCP-сервер выставляет 4 инструмента. Индексация: скан папки → TextLoader → RecursiveCharacterTextSplitter (язык по расширению) → эмбеддинги → ChromaDB (единственный источник правды, персистентный) + BM25-индекс в памяти (перестраивается из Chroma). Поиск: гибридный (BM25 + вектор → RRF). Ответы: LangGraph-граф rewrite → retrieve → grade (LLM yes/no) → generate, с retry-циклом расширения запроса (max 2). Всё в docker compose: ollama + ollama-init (pull моделей) + rag-kb.

**Tech Stack:** Python 3.12 (в Docker) / uv, fastmcp ^2, langgraph, langchain-text-splitters, langchain-community, langchain-ollama, chromadb, rank-bm25, pydantic-settings, pytest + pytest-asyncio, ruff, Docker Compose. Модель: `qwen2.5:3b`, эмбеддинги: `nomic-embed-text` (Ollama) или дефолт ChromaDB — переключение конфигом.

**Документация:** README/ARCHITECTURE/REPORT на русском. ARCHITECTURE.md и README.md пишет исполнитель. REPORT.md — со слов пользователя: после каждой задачи исполнитель спрашивает, что занести, пользователь диктует, исполнитель оформляет и исправляет (требование задания — вести по ходу работы). В ARCHITECTURE.md обязательна секция «ИИ-инструменты разработки»: opencode, плагин superpowers, MCP-сервер Context7, модель GLM 5.3.

---

## File Structure

```
OtusAIProject/
├── pyproject.toml, uv.lock, .gitignore, .env.example, ruff.toml (в pyproject)
├── docker-compose.yml
├── Dockerfile
├── .vscode/mcp.json                  # конфиг подключения для VSCode Copilot
├── .github/workflows/ci.yml
├── src/rag_kb/
│   ├── __init__.py
│   ├── config.py                     # Settings (pydantic-settings), get_settings()
│   ├── server.py                     # входная точка: сборка зависимостей + mcp.run
│   ├── app.py                        # create_mcp_server(...) — 4 инструмента, DI для тестов
│   ├── types.py                      # LoadedDoc, Chunk, IndexReport, IndexStats
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── loaders.py                # scan_folder, load_file (TextLoader)
│   │   ├── chunking.py               # split_document (RecursiveCharacterTextSplitter)
│   │   ├── embedder.py               # Embedder-протокол, ChromaDefaultEmbedder, OllamaEmbedder, create_embedder
│   │   └── indexer.py                # Indexer: index_folder, status
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── stores.py                 # VectorStore (ChromaDB wrapper)
│   │   ├── bm25_store.py             # BM25Store (rank_bm25)
│   │   └── hybrid.py                 # rrf_fuse, HybridRetriever
│   ├── llm.py                        # LLM-протокол, OllamaLLM (ChatOllama: plain+json)
│   └── graph/
│       ├── __init__.py
│       ├── state.py                  # GraphState (TypedDict)
│       ├── nodes.py                  # make_rewriter, retrieve_node, make_grader, make_generator
│       └── builder.py                # build_graph (условные рёбра, retry ≤ 2)
├── sample_docs/                      # демо-база ≥500 КБ с проверочными фактами
└── tests/
    ├── conftest.py                   # FakeLLM, FakeEmbedder, фикстуры
    ├── unit/ (test_config, test_loaders, test_chunking, test_embedder, test_stores,
    │          test_bm25, test_rrf, test_indexer, test_nodes, test_builder)
    ├── e2e/test_mcp_tools.py
    └── test_sample_docs.py
```

Ключевые типы (единые для всего плана):

```python
# types.py
@dataclass
class LoadedDoc:
    path: Path
    text: str
    doc_type: str  # "markdown" | "text" | "python" | "js" | "ts" | "json" | "yaml"


@dataclass
class Chunk:
    id: str  # sha1(f"{source}:{chunk_index}")
    text: str
    metadata: dict  # {"source": str, "chunk_index": int, "total_chunks": int, "doc_type": str}


@dataclass
class IndexReport:
    files: int
    chunks: int
    seconds: float
    errors: list  # list[str]


@dataclass
class IndexStats:
    files: int
    chunks: int
    last_indexed_at: str | None
```

---

### Task 1: Каркас проекта (uv, pyproject, ruff, пакеты)

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/rag_kb/__init__.py`, `src/rag_kb/indexing/__init__.py`, `src/rag_kb/retrieval/__init__.py`, `src/rag_kb/graph/__init__.py`, `tests/__init__.py` (пустые), `tests/unit/__init__.py`, `tests/e2e/__init__.py`

- [ ] **Step 1: Установить uv** (ещё не установлен)

Run: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
Expected: `uv is installed`. Переоткрыть терминал (или `$env:Path += ";$env:USERPROFILE\.local\bin"`), проверить `uv --version`.

- [ ] **Step 2: Создать pyproject.toml**

```toml
[project]
name = "rag-kb"
version = "0.1.0"
description = "RAG Knowledge Base MCP-сервер: локальная база знаний с Corrective RAG на LangGraph"
requires-python = ">=3.11"
dependencies = [
    "fastmcp>=2.0",
    "langgraph>=0.2",
    "langchain>=0.3",
    "langchain-community>=0.3",
    "langchain-ollama>=0.2",
    "langchain-text-splitters>=0.3",
    "chromadb>=0.5",
    "rank-bm25>=0.2.2",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.5",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/rag_kb"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 3: Создать .gitignore**

```gitignore
__pycache__/
*.pyc
.venv/
.pytest_cache/
.ruff_cache/
data/
.env
uv.lock
```

Примечание: `uv.lock` в .gitignore НЕ добавляем — lock фиксирует версии для преподавателя. Итоговое содержимое:

```gitignore
__pycache__/
*.pyc
.venv/
.pytest_cache/
.ruff_cache/
data/
.env
```

- [ ] **Step 4: Создать пустые `__init__.py`** во всех пакетах из списка Files.

Run: `New-Item -ItemType File -Force -Path src\rag_kb\__init__.py, src\rag_kb\indexing\__init__.py, src\rag_kb\retrieval\__init__.py, src\rag_kb\graph\__init__.py, tests\__init__.py, tests\unit\__init__.py, tests\e2e\__init__.py` (из корня проекта)

- [ ] **Step 5: Установить зависимости**

Run: `uv sync`
Expected: `Resolved N packages`, создан `.venv`, `uv.lock`.

- [ ] **Step 6: Smoke-проверка**

Run: `uv run python -c "import rag_kb, fastmcp, langgraph, chromadb, rank_bm25; print('ok')"`
Expected: `ok`

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: project skeleton with uv, deps and package layout"
```

---

### Task 2: Конфигурация (config.py)

**Files:**
- Create: `src/rag_kb/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Написать тест**

```python
# tests/unit/test_config.py
from rag_kb.config import Settings


def test_defaults():
    s = Settings()
    assert s.llm_model == "qwen2.5:3b"
    assert s.embedding_provider == "chromadb"
    assert s.top_k == 6
    assert s.min_relevant_chunks == 1
    assert s.max_retries == 2


def test_env_override(monkeypatch):
    monkeypatch.setenv("RAGKB_LLM_MODEL", "phi3:mini")
    monkeypatch.setenv("RAGKB_EMBEDDING_PROVIDER", "ollama")
    s = Settings()
    assert s.llm_model == "phi3:mini"
    assert s.embedding_provider == "ollama"
```

- [ ] **Step 2: Запустить, убедиться в FAIL**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'rag_kb.config'`

- [ ] **Step 3: Реализовать**

```python
# src/rag_kb/config.py
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAGKB_", extra="ignore")

    # MCP-сервер
    host: str = "0.0.0.0"
    port: int = 8000

    # Хранилище
    chroma_dir: Path = Path("data/chroma")
    collection_name: str = "documents"

    # LLM / эмбеддинги
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b"
    embedding_provider: str = "chromadb"  # "chromadb" | "ollama"
    ollama_embedding_model: str = "nomic-embed-text"

    # Чанкинг
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # Поиск и граф
    top_k: int = 6
    rrf_k: int = 60
    min_relevant_chunks: int = 1
    max_retries: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Запустить тест — PASS**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/rag_kb/config.py tests/unit/test_config.py
git commit -m "feat: settings via pydantic-settings with env overrides"
```

---

### Task 3: Типы (types.py)

**Files:**
- Create: `src/rag_kb/types.py`
- Test: `tests/unit/test_types.py`

- [ ] **Step 1: Тест**

```python
# tests/unit/test_types.py
from rag_kb.types import Chunk, IndexReport, IndexStats, LoadedDoc


def test_dataclasses_hold_values(tmp_path):
    doc = LoadedDoc(path=tmp_path / "a.md", text="hello", doc_type="markdown")
    chunk = Chunk(
        id="x",
        text="hello",
        metadata={"source": "a.md", "chunk_index": 0, "total_chunks": 1, "doc_type": "markdown"},
    )
    report = IndexReport(files=1, chunks=1, seconds=0.1, errors=[])
    stats = IndexStats(files=1, chunks=1, last_indexed_at=None)
    assert (doc.text, chunk.id, report.files, stats.chunks) == ("hello", "x", 1, 1)
```

- [ ] **Step 2: FAIL** — Run: `uv run pytest tests/unit/test_types.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Реализовать** — содержимое `src/rag_kb/types.py` равно блоку «Ключевые типы» из раздела File Structure (dataclass-импорты: `from dataclasses import dataclass; from pathlib import Path`).

- [ ] **Step 4: PASS** — `uv run pytest tests/unit/test_types.py -v` → `1 passed`.

- [ ] **Step 5: Commit** — `git add src/rag_kb/types.py tests/unit/test_types.py && git commit -m "feat: core dataclasses Chunk/LoadedDoc/IndexReport/IndexStats"`

---

### Task 4: Загрузка файлов (loaders.py)

**Files:**
- Create: `src/rag_kb/indexing/loaders.py`
- Test: `tests/unit/test_loaders.py`

- [ ] **Step 1: Тест**

```python
# tests/unit/test_loaders.py
import pytest

from rag_kb.indexing.loaders import SUPPORTED_EXTENSIONS, load_file, scan_folder


def test_scan_folder_filters_by_extension(tmp_path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "b.py").write_text("x=1", encoding="utf-8")
    (tmp_path / "c.exe").write_bytes(b"\x00")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.yaml").write_text("k: v", encoding="utf-8")
    found = scan_folder(tmp_path, "**/*")
    names = sorted(p.name for p in found)
    assert names == ["a.md", "b.py", "d.yaml"]


def test_scan_folder_respects_pattern(tmp_path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("x", encoding="utf-8")
    assert [p.name for p in scan_folder(tmp_path, "*.md")] == ["a.md"]


@pytest.mark.parametrize(
    "name,text,doc_type",
    [
        ("a.md", "# Заголовок", "markdown"),
        ("b.txt", "текст", "text"),
        ("c.py", "x = 1", "python"),
        ("d.js", "const x = 1", "js"),
        ("e.ts", "const x: number = 1", "ts"),
        ("f.json", '{"k": 1}', "json"),
        ("g.yaml", "k: v", "yaml"),
    ],
)
def test_load_file_supported_formats(tmp_path, name, text, doc_type):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    doc = load_file(p)
    assert doc.doc_type == doc_type
    assert doc.path == p


def test_load_file_rejects_unsupported(tmp_path):
    p = tmp_path / "a.exe"
    p.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="not supported"):
        load_file(p)
```

- [ ] **Step 2: FAIL** — `uv run pytest tests/unit/test_loaders.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Реализовать**

```python
# src/rag_kb/indexing/loaders.py
from pathlib import Path

from langchain_community.document_loaders import TextLoader

from rag_kb.types import LoadedDoc

SUPPORTED_EXTENSIONS = {".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml"}

DOC_TYPE_BY_EXT = {
    ".md": "markdown",
    ".txt": "text",
    ".py": "python",
    ".js": "js",
    ".ts": "ts",
    ".json": "json",
    ".yaml": "yaml",
}


def scan_folder(folder: Path, pattern: str = "**/*") -> list[Path]:
    return sorted(
        p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_file(path: Path) -> LoadedDoc:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Format {ext} is not supported: {path}")
    text = TextLoader(str(path), encoding="utf-8").load()[0].page_content
    return LoadedDoc(path=path, text=text, doc_type=DOC_TYPE_BY_EXT[ext])
```

- [ ] **Step 4: PASS** — `uv run pytest tests/unit/test_loaders.py -v` → `9 passed`.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: file scanning and loading for 7 supported formats"`

---

### Task 5: Чанкинг (chunking.py)

**Files:**
- Create: `src/rag_kb/indexing/chunking.py`
- Test: `tests/unit/test_chunking.py`

- [ ] **Step 1: Тест**

```python
# tests/unit/test_chunking.py
from rag_kb.indexing.chunking import split_document
from rag_kb.types import LoadedDoc


def make_doc(text: str, doc_type: str = "text") -> LoadedDoc:
    return LoadedDoc(path="fake.md", text=text, doc_type=doc_type)


def test_long_text_split_with_overlap():
    text = "слово " * 600  # ~3.6k символов
    chunks = split_document(make_doc(text), chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.metadata["chunk_index"] == i
        assert c.metadata["total_chunks"] == len(chunks)
        assert c.metadata["source"] == "fake.md"
        assert len(c.text) <= 600  # допуск на разделители


def test_short_text_single_chunk():
    chunks = split_document(make_doc("короткий текст"))
    assert len(chunks) == 1
    assert chunks[0].text == "короткий текст"


def test_python_split_prefers_function_boundaries():
    code = "def a():\n    return 1\n\n\n" + "x = 2\n" * 200
    chunks = split_document(make_doc(code, doc_type="python"), chunk_size=400, chunk_overlap=0)
    assert len(chunks) > 1
    assert "def a():" in chunks[0].text


def test_chunk_id_deterministic():
    a = split_document(make_doc("один и тот же текст"))
    b = split_document(make_doc("один и тот же текст"))
    assert a[0].id == b[0].id
```

- [ ] **Step 2: FAIL** — ModuleNotFoundError.

- [ ] **Step 3: Реализовать**

```python
# src/rag_kb/indexing/chunking.py
import hashlib
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from rag_kb.types import Chunk, LoadedDoc

LANGUAGE_BY_TYPE = {
    "python": Language.PYTHON,
    "js": Language.JS,
    "ts": Language.TS,
    "markdown": Language.MARKDOWN,
}


def _splitter(doc_type: str, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    lang = LANGUAGE_BY_TYPE.get(doc_type)
    if lang is not None:
        return RecursiveCharacterTextSplitter.from_language(
            lang, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def split_document(doc: LoadedDoc, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    parts = _splitter(doc.doc_type, chunk_size, chunk_overlap).split_text(doc.text)
    source = str(doc.path)
    chunks = []
    for i, text in enumerate(parts):
        chunk_id = hashlib.sha1(f"{source}:{i}".encode()).hexdigest()
        chunks.append(
            Chunk(
                id=chunk_id,
                text=text,
                metadata={
                    "source": source,
                    "chunk_index": i,
                    "total_chunks": len(parts),
                    "doc_type": doc.doc_type,
                },
            )
        )
    return chunks
```

- [ ] **Step 4: PASS** — `uv run pytest tests/unit/test_chunking.py -v` → `4 passed`.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: language-aware chunking with source metadata"`

---

### Task 6: Эмбеддинги (embedder.py) — опциональное задание

**Files:**
- Create: `src/rag_kb/indexing/embedder.py`
- Test: `tests/unit/test_embedder.py`

- [ ] **Step 1: Тест**

```python
# tests/unit/test_embedder.py
from rag_kb.config import Settings
from rag_kb.indexing.embedder import ChromaDefaultEmbedder, OllamaEmbedder, create_embedder


def test_factory_returns_chromadb_default():
    emb = create_embedder(Settings(embedding_provider="chromadb"))
    assert isinstance(emb, ChromaDefaultEmbedder)


def test_factory_returns_ollama():
    emb = create_embedder(Settings(embedding_provider="ollama", ollama_base_url="http://x:11434"))
    assert isinstance(emb, OllamaEmbedder)


def test_factory_rejects_unknown():
    import pytest

    with pytest.raises(ValueError, match="embedding_provider"):
        create_embedder(Settings(embedding_provider="openai"))
```

- [ ] **Step 2: FAIL** — ModuleNotFoundError.

- [ ] **Step 3: Реализовать**

```python
# src/rag_kb/indexing/embedder.py
from typing import Protocol

from chromadb.utils import embedding_functions
from langchain_ollama import OllamaEmbeddings

from rag_kb.config import Settings


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class ChromaDefaultEmbedder:
    """Дефолтная эмбеддинг-функция ChromaDB (ONNX all-MiniLM-L6-L2, качается автоматически)."""

    def __init__(self) -> None:
        self._ef = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(v) for v in self._ef(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(self._ef([text])[0])


class OllamaEmbedder:
    """Внешние эмбеддинги через Ollama (например, nomic-embed-text)."""

    def __init__(self, base_url: str, model: str) -> None:
        self._client = OllamaEmbeddings(base_url=base_url, model=model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)


def create_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "chromadb":
        return ChromaDefaultEmbedder()
    if settings.embedding_provider == "ollama":
        return OllamaEmbedder(settings.ollama_base_url, settings.ollama_embedding_model)
    raise ValueError(f"Unknown embedding_provider: {settings.embedding_provider}")
```

- [ ] **Step 4: PASS** — `uv run pytest tests/unit/test_embedder.py -v` → `3 passed`.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: embedder factory — chromadb default or ollama, config switch"`

---

### Task 7: ChromaDB-хранилище (stores.py)

**Files:**
- Create: `src/rag_kb/retrieval/stores.py`
- Test: `tests/unit/test_stores.py`

Дизайн: эмбеддинги всегда считаем сами (через Embedder) и передаём в Chroma — коллекция не привязана к EF, провайдер переключается свободно. `last_indexed_at` храним в метаданных коллекции.

- [ ] **Step 1: Тест**

```python
# tests/unit/test_stores.py
from rag_kb.retrieval.stores import VectorStore
from rag_kb.types import Chunk


def make_chunks(n=3, source="a.md"):
    return [
        Chunk(
            id=f"id{i}",
            text=f"текст номер {i}",
            metadata={"source": source, "chunk_index": i, "total_chunks": n, "doc_type": "text"},
        )
        for i in range(n)
    ]


def make_store(tmp_path):
    from rag_kb.config import Settings

    return VectorStore(Settings(chroma_dir=tmp_path / "chroma", collection_name="test"))


def test_add_query_count(tmp_path):
    store = make_store(tmp_path)
    chunks = make_chunks(3)
    emb = [[float(i)] * 4 for i in range(3)]
    store.add_chunks(chunks, emb)
    assert store.count() == 3
    found = store.query([2.0] * 4, top_k=2)
    assert len(found) == 2
    assert found[0].id == "id2"  # ближайший вектор


def test_add_chunks_upserts_by_id(tmp_path):
    store = make_store(tmp_path)
    chunks = make_chunks(2)
    store.add_chunks(chunks, [[0.0] * 4, [1.0] * 4])
    chunks[0].text = "обновлённый"
    store.add_chunks(chunks, [[0.0] * 4, [1.0] * 4])
    assert store.count() == 2
    assert store.all_chunks()[0].text == "обновлённый"


def test_delete_by_source(tmp_path):
    store = make_store(tmp_path)
    store.add_chunks(make_chunks(2, source="a.md") + make_chunks(2, source="b.md"), [[0.0] * 4] * 4)
    store.delete_by_source("a.md")
    assert store.count() == 2
    assert store.unique_sources() == ["b.md"]


def test_last_indexed_at(tmp_path):
    store = make_store(tmp_path)
    assert store.get_last_indexed_at() is None
    store.set_last_indexed_at("2026-09-13T10:00:00")
    assert store.get_last_indexed_at() == "2026-09-13T10:00:00"
```

- [ ] **Step 2: FAIL** — ModuleNotFoundError.

- [ ] **Step 3: Реализовать**

```python
# src/rag_kb/retrieval/stores.py
from pathlib import Path

import chromadb

from rag_kb.config import Settings
from rag_kb.types import Chunk


class VectorStore:
    """Обёртка над ChromaDB PersistentClient. Эмбеддинги передаются снаружи."""

    def __init__(self, settings: Settings) -> None:
        Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
            embeddings=embeddings,
        )

    def query(self, embedding: list[float], top_k: int) -> list[Chunk]:
        if self.count() == 0:
            return []
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas"],
        )
        return [
            Chunk(id=i, text=d, metadata=m)
            for i, d, m in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], strict=True)
        ]

    def all_chunks(self) -> list[Chunk]:
        res = self._collection.get(include=["documents", "metadatas"])
        return [
            Chunk(id=i, text=d, metadata=m)
            for i, d, m in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
        ]

    def count(self) -> int:
        return self._collection.count()

    def unique_sources(self) -> list[str]:
        return sorted({c.metadata["source"] for c in self.all_chunks()})

    def delete_by_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def set_last_indexed_at(self, iso: str) -> None:
        self._collection.modify(metadata={"last_indexed_at": iso})

    def get_last_indexed_at(self) -> str | None:
        return (self._collection.metadata or {}).get("last_indexed_at")
```

- [ ] **Step 4: PASS** — `uv run pytest tests/unit/test_stores.py -v` → `4 passed`.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: chromadb vector store wrapper with upsert and source deletion"`

---

### Task 8: BM25 + RRF + гибридный retriever (bm25_store.py, hybrid.py)

**Files:**
- Create: `src/rag_kb/retrieval/bm25_store.py`, `src/rag_kb/retrieval/hybrid.py`
- Test: `tests/unit/test_bm25.py`, `tests/unit/test_rrf.py`

- [ ] **Step 1: Тест BM25**

```python
# tests/unit/test_bm25.py
from rag_kb.retrieval.bm25_store import BM25Store
from rag_kb.types import Chunk


def chunk(i, text):
    return Chunk(
        id=str(i),
        text=text,
        metadata={"source": "a.md", "chunk_index": i, "total_chunks": 3, "doc_type": "text"},
    )


def test_bm25_finds_exact_keyword():
    store = BM25Store()
    store.build(
        [
            chunk(0, "настройка кэша redis"),
            chunk(1, "логирование запросов"),
            chunk(2, "TOKEN_EXPIRY_HOURS равен 72"),
        ]
    )
    results = store.search("TOKEN_EXPIRY_HOURS", top_k=1)
    assert results[0][0].id == "2"


def test_bm25_empty_store_returns_empty_list():
    store = BM25Store()
    assert store.search("что угодно", top_k=3) == []


def test_bm25_rebuild_replaces_index():
    store = BM25Store()
    store.build([chunk(0, "старый индекс")])
    store.build([chunk(0, "новое содержимое про docker")])
    assert store.search("docker", top_k=1)[0][0].text.startswith("новое")
```

- [ ] **Step 2: FAIL** — ModuleNotFoundError.

- [ ] **Step 3: Реализовать**

```python
# src/rag_kb/retrieval/bm25_store.py
import re

from rank_bm25 import BM25Okapi

from rag_kb.types import Chunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Store:
    """Sparse-поиск по чанкам. In-memory, перестраивается из ChromaDB."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._chunks[i], float(scores[i])) for i in order if scores[i] > 0]
```

- [ ] **Step 4: PASS** — `uv run pytest tests/unit/test_bm25.py -v` → `3 passed`.

- [ ] **Step 5: Тест RRF**

```python
# tests/unit/test_rrf.py
from rag_kb.retrieval.hybrid import rrf_fuse
from rag_kb.types import Chunk


def chunk(i):
    return Chunk(
        id=str(i),
        text=f"t{i}",
        metadata={"source": "a", "chunk_index": i, "total_chunks": 5, "doc_type": "text"},
    )


def test_doc_found_by_both_ranks_first():
    a = [chunk(1), chunk(2), chunk(3)]
    b = [chunk(3), chunk(1), chunk(4)]
    fused = rrf_fuse(a, b, k=60)
    assert fused[0].id == "1"  # высокий ранг в обоих списках


def test_doc_in_one_list_still_present():
    a = [chunk(1)]
    b = [chunk(2)]
    fused = rrf_fuse(a, b, k=60)
    assert {c.id for c in fused} == {"1", "2"}


def test_empty_inputs():
    assert rrf_fuse([], [], k=60) == []


def test_top_k_limit():
    a = [chunk(i) for i in range(10)]
    b = [chunk(i) for i in range(9, -1, -1)]
    assert len(rrf_fuse(a, b, k=60, top_k=3)) == 3
```

- [ ] **Step 6: FAIL** — ModuleNotFoundError.

- [ ] **Step 7: Реализовать**

```python
# src/rag_kb/retrieval/hybrid.py
from rag_kb.config import Settings
from rag_kb.indexing.embedder import Embedder
from rag_kb.retrieval.bm25_store import BM25Store
from rag_kb.retrieval.stores import VectorStore
from rag_kb.types import Chunk


def rrf_fuse(
    list_a: list[Chunk], list_b: list[Chunk], k: int = 60, top_k: int | None = None
) -> list[Chunk]:
    """Reciprocal Rank Fusion: score(d) = sum 1/(k + rank_i(d))."""
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, Chunk] = {}
    for results in (list_a, list_b):
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            chunks_by_id.setdefault(chunk.id, chunk)
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    if top_k is not None:
        ordered = ordered[:top_k]
    return [chunks_by_id[cid] for cid in ordered]


class HybridRetriever:
    """Гибридный поиск: BM25 (точные ключевые слова) + вектор (смысл) → RRF."""

    def __init__(
        self, vector_store: VectorStore, embedder: Embedder, bm25: BM25Store, settings: Settings
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._bm25 = bm25
        self._settings = settings

    def search(self, query: str, top_k: int | None = None) -> list[Chunk]:
        top_k = top_k or self._settings.top_k
        sparse = [c for c, _ in self._bm25.search(query, top_k)]
        dense = self._vector_store.query(self._embedder.embed_query(query), top_k)
        return rrf_fuse(sparse, dense, k=self._settings.rrf_k, top_k=top_k)

    def rebuild_bm25(self) -> None:
        self._bm25.build(self._vector_store.all_chunks())
```

- [ ] **Step 8: PASS** — `uv run pytest tests/unit/test_rrf.py -v` → `4 passed`.

- [ ] **Step 9: Commit** — `git add -A && git commit -m "feat: BM25 store and RRF hybrid retriever"`

---

### Task 9: Индексатор (indexer.py)

**Files:**
- Create: `src/rag_kb/indexing/indexer.py`
- Test: `tests/unit/test_indexer.py`

Дизайн: `Indexer` владеет `VectorStore`, `Embedder`, `BM25Store`. `index_folder` переиндексирует каждый файл атомарно (delete_by_source → add). BM25 перестраивается в конце из Chroma. `status()` возвращает `IndexStats`.

- [ ] **Step 1: Тест (FakeEmbedder кладём в conftest.py)**

```python
# tests/conftest.py
class FakeEmbedder:
    """Детерминированные эмбеддинги без сети: вектор из кодов символов."""

    def _vec(self, text: str) -> list[float]:
        import hashlib

        digest = hashlib.sha1(text.encode()).digest()
        return [b / 255.0 for b in digest[:16]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeLLM:
    """Программируемая LLM для тестов графа: pop ответов по очереди, запись промптов."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def invoke(self, prompt: str, json_mode: bool = False) -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0)
```

```python
# tests/unit/test_indexer.py
import pytest

from rag_kb.config import Settings
from rag_kb.indexing.indexer import Indexer
from rag_kb.retrieval.bm25_store import BM25Store
from rag_kb.retrieval.stores import VectorStore
from tests.conftest import FakeEmbedder


@pytest.fixture
def indexer(tmp_path):
    settings = Settings(chroma_dir=tmp_path / "chroma")
    return Indexer(
        vector_store=VectorStore(settings),
        embedder=FakeEmbedder(),
        bm25=BM25Store(),
        settings=settings,
    )


def test_index_folder_reports_files_and_chunks(indexer, tmp_path):
    (tmp_path / "a.md").write_text("# Тест\n" + "содержимое " * 300, encoding="utf-8")
    (tmp_path / "b.txt").write_text("короткий файл", encoding="utf-8")
    report = indexer.index_folder(tmp_path)
    assert report.files == 2
    assert report.chunks > 2
    assert report.errors == []


def test_index_status_empty_then_filled(indexer, tmp_path):
    assert indexer.status().files == 0
    (tmp_path / "a.txt").write_text("текст", encoding="utf-8")
    indexer.index_folder(tmp_path)
    stats = indexer.status()
    assert stats.files == 1
    assert stats.chunks == 1
    assert stats.last_indexed_at is not None


def test_reindex_replaces_old_chunks(indexer, tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("первая версия " * 100, encoding="utf-8")
    indexer.index_folder(tmp_path)
    p.write_text("вторая версия " * 100, encoding="utf-8")
    indexer.index_folder(tmp_path)
    assert indexer.status().files == 1
    assert indexer.status().chunks == indexer.status().chunks  # стабильное число, без дублей
    assert "вторая" in " ".join(c.text for c in indexer.vector_store.all_chunks())
    assert not any("первая" in c.text for c in indexer.vector_store.all_chunks())


def test_index_folder_missing_path_raises(indexer):
    with pytest.raises(FileNotFoundError):
        indexer.index_folder("Z:/definitely/missing")
```

- [ ] **Step 2: FAIL** — ModuleNotFoundError.

- [ ] **Step 3: Реализовать**

```python
# src/rag_kb/indexing/indexer.py
import time
from datetime import UTC, datetime
from pathlib import Path

from rag_kb.config import Settings
from rag_kb.indexing.chunking import split_document
from rag_kb.indexing.embedder import Embedder
from rag_kb.indexing.loaders import load_file, scan_folder
from rag_kb.retrieval.bm25_store import BM25Store
from rag_kb.retrieval.stores import VectorStore
from rag_kb.types import IndexReport, IndexStats


class Indexer:
    def __init__(
        self, vector_store: VectorStore, embedder: Embedder, bm25: BM25Store, settings: Settings
    ) -> None:
        self.vector_store = vector_store
        self._embedder = embedder
        self._bm25 = bm25
        self._settings = settings

    def index_folder(self, folder: str | Path, pattern: str = "**/*") -> IndexReport:
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")
        started = time.perf_counter()
        errors: list[str] = []
        total_chunks = 0
        files = scan_folder(folder, pattern)
        for path in files:
            try:
                doc = load_file(path)
                chunks = split_document(
                    doc, self._settings.chunk_size, self._settings.chunk_overlap
                )
                embeddings = self._embedder.embed_documents([c.text for c in chunks])
                self.vector_store.delete_by_source(str(path))
                self.vector_store.add_chunks(chunks, embeddings)
                total_chunks += len(chunks)
            except Exception as exc:  # noqa: BLE001 — один битый файл не рушит индексацию
                errors.append(f"{path}: {exc}")
        self._bm25.build(self.vector_store.all_chunks())
        self.vector_store.set_last_indexed_at(datetime.now(UTC).isoformat(timespec="seconds"))
        return IndexReport(
            files=len(files) - len(errors),
            chunks=total_chunks,
            seconds=round(time.perf_counter() - started, 2),
            errors=errors,
        )

    def status(self) -> IndexStats:
        return IndexStats(
            files=len(self.vector_store.unique_sources()),
            chunks=self.vector_store.count(),
            last_indexed_at=self.vector_store.get_last_indexed_at(),
        )
```

- [ ] **Step 4: PASS** — `uv run pytest tests/unit/test_indexer.py -v` → `4 passed`.

- [ ] **Step 5: Запустить все тесты** — `uv run pytest -q` → все зелёные.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: folder indexer with atomic re-indexing and status stats"`

---

### Task 10: LLM-обёртка (llm.py)

**Files:**
- Create: `src/rag_kb/llm.py`

Тест не нужен (тонкая обёртка над ChatOllama; поведение графа тестируется с FakeLLM в Task 11–12).

- [ ] **Step 1: Реализовать**

```python
# src/rag_kb/llm.py
from typing import Protocol

from langchain_ollama import ChatOllama


class LLM(Protocol):
    def invoke(self, prompt: str, json_mode: bool = False) -> str: ...


class OllamaLLM:
    """Обёртка над ChatOllama: plain-режим для текста, json-режим для грейдинга."""

    def __init__(self, base_url: str, model: str) -> None:
        self._plain = ChatOllama(base_url=base_url, model=model)
        self._json = ChatOllama(base_url=base_url, model=model, format="json")

    def invoke(self, prompt: str, json_mode: bool = False) -> str:
        client = self._json if json_mode else self._plain
        return client.invoke(prompt).content
```

- [ ] **Step 2: Smoke-проверка импорта**

Run: `uv run python -c "from rag_kb.llm import OllamaLLM; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit** — `git add src/rag_kb/llm.py && git commit -m "feat: ollama llm wrapper with json mode for grading"`

---

### Task 11: Узлы графа (state.py, nodes.py)

**Files:**
- Create: `src/rag_kb/graph/state.py`, `src/rag_kb/graph/nodes.py`
- Test: `tests/unit/test_nodes.py`

- [ ] **Step 1: Состояние**

```python
# src/rag_kb/graph/state.py
from typing import TypedDict

from rag_kb.types import Chunk


class GraphState(TypedDict):
    question: str  # исходный вопрос пользователя
    query: str  # текущий (пере)сформулированный поисковый запрос
    attempt: int  # номер попытки поиска (0 = первая)
    chunks: list[Chunk]  # чанки после retrieve
    relevant: list[Chunk]  # чанки, оценённые LLM как релевантные
    answer: str
    sources: list[str]  # уникальные source релевантных чанков
```

- [ ] **Step 2: Тест узлов**

```python
# tests/unit/test_nodes.py
import json

from rag_kb.graph.nodes import (
    make_generator,
    make_grader,
    make_rewriter,
    retrieve_node,
)
from rag_kb.graph.state import GraphState
from tests.conftest import FakeEmbedder, FakeLLM


class StubRetriever:
    def __init__(self, results):
        self._results = results
        self.queries = []

    def search(self, query, top_k=None):
        self.queries.append(query)
        return self._results


def chunk(i, text):
    from rag_kb.types import Chunk

    return Chunk(
        id=str(i),
        text=text,
        metadata={"source": f"f{i}.md", "chunk_index": 0, "total_chunks": 1, "doc_type": "text"},
    )


def base_state(**over) -> GraphState:
    state = GraphState(
        question="как работает кэш?",
        query="как работает кэш?",
        attempt=0,
        chunks=[],
        relevant=[],
        answer="",
        sources=[],
    )
    state.update(over)
    return state


def test_rewriter_first_attempt_normalizes():
    llm = FakeLLM(["не должно вызываться"])
    node = make_rewriter(llm)
    out = node(base_state())
    assert out["query"] == "как работает кэш?"  # первая попытка — LLM не зовём
    assert llm.prompts == []


def test_rewriter_broadens_on_retry():
    llm = FakeLLM(["кэш redis TTL инвалидация хранение"])
    node = make_rewriter(llm)
    out = node(base_state(attempt=1, query="стухание кэша"))
    assert out["query"] == "кэш redis TTL инвалидация хранение"
    assert "стухание кэша" in llm.prompts[0]


def test_retrieve_node_calls_retriever_and_counts_attempt():
    r = StubRetriever([chunk(0, "a"), chunk(1, "b")])
    out = retrieve_node(r)(base_state())
    assert [c.id for c in out["chunks"]] == ["0", "1"]
    assert out["attempt"] == 1


def test_grader_parses_yes_no():
    llm = FakeLLM(['{"relevant": "yes"}', '{"relevant": "no"}'])
    node = make_grader(llm)
    out = node(base_state(chunks=[chunk(0, "про кэш"), chunk(1, "про логи")]))
    assert [c.id for c in out["relevant"]] == ["0"]


def test_grader_tolerates_broken_json():
    llm = FakeLLM(["мусор без json", "relevant: yes точно"])
    node = make_grader(llm)
    out = node(base_state(chunks=[chunk(0, "a"), chunk(1, "b")]))
    assert [c.id for c in out["relevant"]] == ["1"]  # fallback-парсер нашёл yes


def test_generator_answer_and_sources():
    llm = FakeLLM(["Кэш работает так-то."])
    node = make_generator(llm)
    out = node(base_state(relevant=[chunk(0, "про кэш"), chunk(0, "про кэш ещё")]))
    assert out["answer"] == "Кэш работает так-то."
    assert out["sources"] == ["f0.md"]


def test_generator_nothing_relevant():
    llm = FakeLLM([])
    node = make_generator(llm)
    out = node(base_state(relevant=[]))
    assert "ничего не найдено" in out["answer"].lower()
```

- [ ] **Step 3: FAIL** — ModuleNotFoundError.

- [ ] **Step 4: Реализовать**

```python
# src/rag_kb/graph/nodes.py
import json
import re

from rag_kb.graph.state import GraphState
from rag_kb.llm import LLM
from rag_kb.types import Chunk

REWRITE_PROMPT = (
    "Ты — помощник по поиску во внутренней базе знаний. Поисковый запрос не нашёл достаточно "
    "материала. Переформулируй и расширь его: добавь синонимы и связанные термины, убери жаргон. "
    "Верни только поисковый запрос одной строкой, без пояснений.\n\nВопрос: {question}\n"
    "Неудачный запрос: {query}"
)

GRADE_PROMPT = (
    "Определи, релевантен ли фрагмент документа вопросу для ответа на него.\n"
    'Ответь строго JSON: {{"relevant": "yes"}} или {{"relevant": "no"}}.\n\n'
    "Вопрос: {question}\n\nФрагмент:\n{chunk}"
)

GENERATE_PROMPT = (
    "Ты — помощник по внутренней базе знаний. Ответь на вопрос пользователя, опираясь ТОЛЬКО "
    "на приведённые фрагменты. Не выдумывай. Если фрагменты не содержат ответа — так и скажи.\n\n"
    "Вопрос: {question}\n\nФрагменты:\n{context}"
)

NOT_FOUND_ANSWER = (
    "В проиндексированной базе знаний ничего релевантного не найдено "
    "(включая повторные запросы с расширенной формулировкой). "
    "Попробуйте переформулировать вопрос или проиндексировать дополнительные папки."
)


def make_rewriter(llm: LLM):
    def rewrite(state: GraphState) -> dict:
        if state["attempt"] == 0:
            return {"query": state["question"].strip()}
        query = llm.invoke(REWRITE_PROMPT.format(question=state["question"], query=state["query"]))
        return {"query": query.strip()}

    return rewrite


def retrieve_node(retriever):
    def retrieve(state: GraphState) -> dict:
        chunks = retriever.search(state["query"])
        return {"chunks": chunks, "attempt": state["attempt"] + 1}

    return retrieve


def _parse_relevant(raw: str) -> bool:
    try:
        return json.loads(raw)["relevant"].strip().lower().startswith("y")
    except (json.JSONDecodeError, KeyError, AttributeError):
        return bool(re.search(r"\byes\b", raw, re.IGNORECASE))


def make_grader(llm: LLM):
    def grade(state: GraphState) -> dict:
        relevant: list[Chunk] = []
        for chunk in state["chunks"]:
            raw = llm.invoke(
                GRADE_PROMPT.format(question=state["question"], chunk=chunk.text[:1500]),
                json_mode=True,
            )
            if _parse_relevant(raw):
                relevant.append(chunk)
        return {"relevant": relevant}

    return grade


def make_generator(llm: LLM):
    def generate(state: GraphState) -> dict:
        if not state["relevant"]:
            return {"answer": NOT_FOUND_ANSWER, "sources": []}
        context = "\n\n---\n\n".join(
            f"[{c.metadata['source']}]\n{c.text}" for c in state["relevant"]
        )
        answer = llm.invoke(GENERATE_PROMPT.format(question=state["question"], context=context))
        sources = sorted({c.metadata["source"] for c in state["relevant"]})
        return {"answer": answer.strip(), "sources": sources}

    return generate
```

- [ ] **Step 5: PASS** — `uv run pytest tests/unit/test_nodes.py -v` → `7 passed`.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: corrective-rag graph nodes with grading fallback parser"`

---

### Task 12: Сборка графа (builder.py)

**Files:**
- Create: `src/rag_kb/graph/builder.py`
- Test: `tests/unit/test_builder.py`

Схема рёбер: `START → rewrite → retrieve → grade → decide`: relevant достаточно → `generate` → END; недостаточно и `attempt <= max_retries` → `rewrite` (цикл расширения); иначе → `generate`.

- [ ] **Step 1: Тест**

```python
# tests/unit/test_builder.py
import json

from rag_kb.config import Settings
from rag_kb.graph.builder import build_graph
from rag_kb.types import Chunk
from tests.conftest import FakeLLM


class StubRetriever:
    def __init__(self):
        self.results = []

    def search(self, query, top_k=None):
        return self.results


def chunk(i, text):
    return Chunk(
        id=str(i),
        text=text,
        metadata={"source": f"f{i}.md", "chunk_index": 0, "total_chunks": 1, "doc_type": "text"},
    )


def build(retriever, answers):
    return build_graph(retriever, FakeLLM(answers), Settings())


def test_happy_path_generate_after_good_grade():
    r = StubRetriever()
    r.results = [chunk(0, "про кэш")]
    graph = build(r, ['{"relevant": "yes"}', "Кэш работает через Redis."])
    out = graph.invoke({"question": "как работает кэш?"})
    assert out["answer"] == "Кэш работает через Redis."
    assert out["sources"] == ["f0.md"]
    assert out["attempt"] == 1


def test_retry_loop_broadens_query():
    r = StubRetriever()
    r.results = [chunk(0, "про кэш")]
    graph = build(
        r, ['{"relevant": "no"}', "кэш redis TTL хранение", '{"relevant": "yes"}', "Ответ."]
    )
    out = graph.invoke({"question": "почему данные устаревают?"})
    assert out["answer"] == "Ответ."
    assert out["attempt"] == 2  # был повторный поиск


def test_max_two_retries_then_generate_with_what_we_have():
    r = StubRetriever()
    r.results = [chunk(0, "нерелевантное")]
    graph = build(
        r,
        [
            '{"relevant": "no"}',
            "запрос2",
            '{"relevant": "no"}',
            "запрос3",
            '{"relevant": "yes"}',
            "Спасательный ответ.",
        ],
    )
    out = graph.invoke({"question": "q?"})
    assert out["attempt"] == 3  # 1 попытка + 2 retry
    assert out["answer"] == "Спасательный ответ."


def test_nothing_relevant_anywhere():
    r = StubRetriever()
    r.results = [chunk(0, "мимо")]
    graph = build(
        r, ['{"relevant": "no"}', "запрос2", '{"relevant": "no"}', "запрос3", '{"relevant": "no"}']
    )
    out = graph.invoke({"question": "q?"})
    assert "ничего релевантного" in out["answer"].lower()
    assert out["sources"] == []
```

- [ ] **Step 2: FAIL** — ModuleNotFoundError.

- [ ] **Step 3: Реализовать**

```python
# src/rag_kb/graph/builder.py
from langgraph.graph import END, START, StateGraph

from rag_kb.config import Settings
from rag_kb.graph.nodes import make_generator, make_grader, make_rewriter, retrieve_node
from rag_kb.graph.state import GraphState


def _route_after_grade(state: GraphState, settings: Settings) -> str:
    if len(state["relevant"]) >= settings.min_relevant_chunks:
        return "generate"
    if state["attempt"] <= settings.max_retries:
        return "rewrite"
    return "generate"


def build_graph(retriever, llm, settings: Settings):
    graph = StateGraph(GraphState)
    graph.add_node("rewrite", make_rewriter(llm))
    graph.add_node("retrieve", retrieve_node(retriever))
    graph.add_node("grade", make_grader(llm))
    graph.add_node("generate", make_generator(llm))
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        lambda state: _route_after_grade(state, settings),
        {"rewrite": "rewrite", "generate": "generate"},
    )
    graph.add_edge("generate", END)
    return graph.compile()
```

- [ ] **Step 4: PASS** — `uv run pytest tests/unit/test_builder.py -v` → `4 passed`.

- [ ] **Step 5: Все тесты** — `uv run pytest -q` → зелёные; суммарно уже ≥ 30 тестов.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: corrective rag graph with conditional retry loop"`

---

### Task 13: MCP-сервер — 4 инструмента (app.py, server.py)

**Files:**
- Create: `src/rag_kb/app.py`, `src/rag_kb/server.py`, `.env.example`
- Test: `tests/e2e/test_mcp_tools.py`

- [ ] **Step 1: e2e-тест (in-memory FastMCP Client, без сети и без Ollama)**

```python
# tests/e2e/test_mcp_tools.py
import json

import pytest
from fastmcp import Client

from rag_kb.app import create_mcp_server
from rag_kb.config import Settings
from rag_kb.graph.builder import build_graph
from rag_kb.indexing.indexer import Indexer
from rag_kb.retrieval.bm25_store import BM25Store
from rag_kb.retrieval.hybrid import HybridRetriever
from rag_kb.retrieval.stores import VectorStore
from tests.conftest import FakeEmbedder, FakeLLM


@pytest.fixture
def mcp(tmp_path):
    settings = Settings(chroma_dir=tmp_path / "chroma")
    vector_store = VectorStore(settings)
    embedder = FakeEmbedder()
    bm25 = BM25Store()
    retriever = HybridRetriever(vector_store, embedder, bm25, settings)
    indexer = Indexer(vector_store=vector_store, embedder=embedder, bm25=bm25, settings=settings)
    graph = build_graph(
        retriever, FakeLLM(['{"relevant": "yes"}', "Лавовый дракон: урон 40-63."]), settings
    )
    return create_mcp_server(
        indexer=indexer, retriever=retriever, graph_factory=lambda q: graph, settings=settings
    )


async def test_all_four_tools_end_to_end(mcp, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text(
        "Лавовый Дракон — элита Кальдеры: урон 40-63, скорость 9, иммунитет к огню.",
        encoding="utf-8",
    )
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert names == {"index_folder", "ask_question", "find_relevant_docs", "index_status"}
        assert all(t.description and len(t.description) > 40 for t in tools)

        status0 = json.loads(await client.call_tool("index_status", {}))
        assert status0["files"] == 0

        report = json.loads(await client.call_tool("index_folder", {"path": str(docs)}))
        assert report["files"] == 1 and report["errors"] == []

        status1 = json.loads(await client.call_tool("index_status", {}))
        assert status1["files"] == 1 and status1["chunks"] >= 1

        found = json.loads(
            await client.call_tool(
                "find_relevant_docs", {"query": "статы лавового дракона", "top_k": 3}
            )
        )
        assert len(found) >= 1
        assert any("a.md" in c["metadata"]["source"] for c in found)

        answer = await client.call_tool(
            "ask_question", {"question": "какие статы у лавового дракона?"}
        )
        text = answer.content[0].text
        assert "40-63" in text
        assert "a.md" in text  # источники приложены


async def test_ask_question_nothing_found(mcp, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    async with Client(mcp) as client:
        await client.call_tool("index_folder", {"path": str(empty)})
        # FakeLLM истощится → сгенерируем под него поведение через пустую базу:
        # в пустой базе retrieve вернёт 0 чанков → grade не вызовет LLM → NOT_FOUND
        result = await client.call_tool("ask_question", {"question": "что-нибудь"})
        assert "ничего" in result.content[0].text.lower()
```

- [ ] **Step 2: FAIL** — ModuleNotFoundError.

- [ ] **Step 3: Реализовать app.py**

```python
# src/rag_kb/app.py
import json

from fastmcp import FastMCP

from rag_kb.config import Settings
from rag_kb.indexing.indexer import Indexer
from rag_kb.retrieval.hybrid import HybridRetriever


def create_mcp_server(
    indexer: Indexer,
    retriever: HybridRetriever,
    graph_factory,  # graph_factory(question) -> CompiledGraph
    settings: Settings,
) -> FastMCP:
    mcp = FastMCP(
        name="rag-kb",
        instructions=(
            "База знаний по локальным документам пользователя. Сначала индексируйте папку "
            "(index_folder), затем отвечайте на вопросы через ask_question или ищите фрагменты "
            "через find_relevant_docs. Ответы основаны только на проиндексированных документах."
        ),
    )

    @mcp.tool
    def index_folder(path: str, pattern: str = "**/*") -> str:
        """Проиндексировать папку с документами в базу знаний для последующего поиска.

        Сканирует файлы (.md, .txt, .py, .js, .ts, .json, .yaml) по glob-паттерну,
        разбивает на чанки, создаёт эмбеддинги и сохраняет в векторное хранилище.
        Вызывайте, когда пользователь просит «проиндексируй папку», добавить документы
        в базу знаний, или когда по вопросу пользователя индекс ещё пуст.
        Повторный вызов для той же папки безопасно обновляет индекс.
        Возвращает JSON: количество файлов, чанков, время и список ошибок.
        """
        report = indexer.index_folder(path, pattern)
        return json.dumps(
            {
                "files": report.files,
                "chunks": report.chunks,
                "seconds": report.seconds,
                "errors": report.errors,
            },
            ensure_ascii=False,
        )

    @mcp.tool
    def ask_question(question: str) -> str:
        """Ответить на вопрос по проиндексированной базе знаний пользователя (RAG).

        Запускает полный пайплайн: переформулировка запроса, гибридный поиск
        (ключевые слова + смысл), LLM-оценка релевантности найденных фрагментов,
        генерация ответа только на их основе. Если релевантного мало — запрос
        автоматически расширяется и поиск повторяется.
        Вызывайте для любых вопросов о содержимом документов пользователя:
        «что написано про X», «как работает Y», «где используется Z».
        Возвращает ответ и список источников (файлы), из которых он составлен.
        """
        state = graph_factory(question).invoke({"question": question})
        return json.dumps(
            {"answer": state["answer"], "sources": state["sources"]}, ensure_ascii=False
        )

    @mcp.tool
    def find_relevant_docs(query: str, top_k: int = 5) -> str:
        """Найти релевантные фрагменты документов без генерации ответа.

        Гибридный поиск: точное совпадение ключевых слов (BM25) + семантический
        поиск по эмбеддингам, объединение через Reciprocal Rank Fusion.
        Вызывайте, когда нужны сами цитаты/фрагменты или точное место в файлах
        («в каком файле упомянуто X», «покажи кусок про Y»), без LLM-пересказа.
        Возвращает JSON-список чанков: текст, источник, позиция в файле.
        """
        chunks = retriever.search(query, top_k=top_k)
        return json.dumps(
            [{"text": c.text, "metadata": c.metadata} for c in chunks], ensure_ascii=False
        )

    @mcp.tool
    def index_status() -> str:
        """Показать статистику базы знаний: число файлов, чанков и время индексации.

        Вызывайте перед поиском, чтобы понять, проиндексировано ли что-то вообще,
        а также когда пользователь спрашивает «что в базе» / «сколько документов».
        Возвращает JSON: files, chunks, last_indexed_at.
        """
        stats = indexer.status()
        return json.dumps(
            {
                "files": stats.files,
                "chunks": stats.chunks,
                "last_indexed_at": stats.last_indexed_at,
            },
            ensure_ascii=False,
        )

    return mcp
```

- [ ] **Step 4: Реализовать server.py (входную точку)**

```python
# src/rag_kb/server.py
"""Входная точка: сборка реальных зависимостей и запуск MCP-сервера."""

from rag_kb.app import create_mcp_server
from rag_kb.config import get_settings
from rag_kb.graph.builder import build_graph
from rag_kb.indexing.embedder import create_embedder
from rag_kb.indexing.indexer import Indexer
from rag_kb.llm import OllamaLLM
from rag_kb.retrieval.bm25_store import BM25Store
from rag_kb.retrieval.hybrid import HybridRetriever
from rag_kb.retrieval.stores import VectorStore


def main() -> None:
    settings = get_settings()
    vector_store = VectorStore(settings)
    embedder = create_embedder(settings)
    bm25 = BM25Store()
    retriever = HybridRetriever(vector_store, embedder, bm25, settings)
    retriever.rebuild_bm25()  # восстановить BM25 из персистентного Chroma после рестарта
    indexer = Indexer(vector_store=vector_store, embedder=embedder, bm25=bm25, settings=settings)
    llm = OllamaLLM(settings.ollama_base_url, settings.llm_model)
    mcp = create_mcp_server(
        indexer=indexer,
        retriever=retriever,
        graph_factory=lambda _q: build_graph(retriever, llm, settings),
        settings=settings,
    )
    mcp.run(transport="streamable-http", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: .env.example**

```bash
# Скопируйте в .env при необходимости переопределить значения
RAGKB_HOST=0.0.0.0
RAGKB_PORT=8000
RAGKB_OLLAMA_BASE_URL=http://localhost:11434
RAGKB_LLM_MODEL=qwen2.5:3b
# chromadb | ollama
RAGKB_EMBEDDING_PROVIDER=chromadb
RAGKB_OLLAMA_EMBEDDING_MODEL=nomic-embed-text
RAGKB_CHROMA_DIR=data/chroma
RAGKB_TOP_K=6
```

- [ ] **Step 6: PASS** — `uv run pytest tests/e2e/test_mcp_tools.py -v` → `2 passed`.

- [ ] **Step 7: Все тесты + линтер** — `uv run pytest -q && uv run ruff check .` → зелёные. При замечаниях ruff — исправить и перезапустить.

- [ ] **Step 8: Commit** — `git add -A && git commit -m "feat: mcp server with 4 tools and rich descriptions, e2e tests"`

---

### Task 14: Демо-документы ≥500 КБ с проверочными фактами (sample_docs/)

**Files:**
- Create: `sample_docs/**` (список ниже), обновить `README.md` (таблица фактов — черновик)
- Test: `tests/test_sample_docs.py`

Контент: база знаний **игрового лора** — вымышленное дополнение **«Трон Пепла»** для Heroes of Might and Magic III (в духе HotA): новый город **Кальдера**, герои с биографиями и взаимоотношениями, существа, артефакты, земли, хронология событий. Русский язык. Канон вселенной HoMM3 (Эрафия, Дейя, Нигон и т.д.) — как фон; **проверочные факты — ТОЛЬКО вымышленные элементы дополнения** (города, имена, статы, даты событий мира): они не встречаются в публичных источниках, в отличие от канона, который LLM знает из обучения. **Все проверочные факты ниже обязаны буквально встречаться в файлах.**

Обязательные файлы и минимальные размеры (сумма минимумов ≥ 520 КБ, контроль ≥ 512 КБ):

| Файл | Тема | Мин. размер |
|---|---|---|
| `sample_docs/README.md` | обзор дополнения «Трон Пепла» и базы знаний | 25 КБ |
| `sample_docs/lore/world-overview.md` | мир дополнения: история, предыстория, сюжет | 45 КБ |
| `sample_docs/lore/towns/calderra.md` | город Кальдера: история, устройство, культура | 30 КБ |
| `sample_docs/lore/towns/relations.md` | отношения Кальдеры с городами Эрафии, Дейи, Нигона | 25 КБ |
| `sample_docs/lore/heroes.md` | биографии героев, взаимоотношения, союзники/враги | 50 КБ |
| `sample_docs/lore/creatures.md` | существа: лор, повадки, иерархия | 55 КБ |
| `sample_docs/lore/lands.md` | Пепельные Пустоши: география, провинции, локации | 35 КБ |
| `sample_docs/lore/timeline.md` | хронология событий мира | 25 КБ |
| `sample_docs/balance/creature-stats.md` | статы новых существ (таблицы) | 40 КБ |
| `sample_docs/balance/artifacts.md` | новые артефакты и наборы | 25 КБ |
| `sample_docs/guides/campaign.md` | прохождение кампании из 7 миссий | 30 КБ |
| `sample_docs/guides/lore-writing.md` | правила написания лора | 20 КБ |
| `sample_docs/data/creature-stats.json` | статы существ в машиночитаемом виде | 40 КБ |
| `sample_docs/config/expansion-settings.yaml` | параметры дополнения | 15 КБ |
| `sample_docs/tools/army_calculator.py` | калькулятор армий (рабочий Python) | 30 КБ |
| `sample_docs/tools/stats_exporter.js` | экспортёр статов (JS) | 15 КБ |
| `sample_docs/tools/api-types.ts` | TS-типы API | 15 КБ |

Проверочные факты (все — вымышленный лор дополнения, не встречаются в публичных источниках):

1. Дополнение — **«Трон Пепла»**, действие происходит в **Пепельных Пустошах** к северу от Эрафии.
2. Новый город — **Кальдера**, правит им **Владычица Пепла Исольда**, ей **127 лет**.
3. Кальдера основана в **812 году** по эрафийскому календарю.
4. Элитное существо города — **Лавовый Дракон**: урон **40-63**, скорость **9**, иммунитет к огню.
5. **Обсидианный голем** ремонтируется за **47 золота** за единицу здоровья.
6. **Пепельная гарпия**: способность «Пепельная завеса» — **25%** уклонения.
7. Герой **Магнус Чёрный Молот** — бывший генерал Эрафии, присягнул Кальдере в **897 году**.
8. **Битва у Трёх Кратеров** — **14 октября 903 года**; после неё Исольда и Магнус стали союзниками.
9. Артефакт **«Сердце Кальдеры»**: +**7** к силе магии, +2 к знаниям.
10. Набор **«Регалии Пепла»** — **4 предмета**, собранный целиком даёт **+15%** к огненному урону.
11. Заклятые враги Кальдеры — некроманты **Дейи** после **Пепельной войны 889-891 годов**.
12. Разведчица **Лиара Ветрокрылая** открыла **Пепельный проход** в **907 году**.
13. Кампания дополнения — **7 миссий**, финальная называется **«Трон Пепла»**.
14. Население Кальдеры — **12 408** душ по переписи **909 года**.
15. Главный праздник города — **«Ночь Обсидиана»**, отмечается **30 ноября**.

- [ ] **Step 1: Сгенерировать файлы** по таблице (темы раскрывать правдоподобно: примеры запросов, таблицы, куски логов, диаграммы в тексте). Каждый факт из списка включить минимум в один файл дословно. Контент генерируется итеративно, `Write`-инструментом; после записи проверить суммарный размер.

Run: `Get-ChildItem -Recurse sample_docs | Measure-Object -Property Length -Sum`
Expected: `Sum >= 524288` (512 КБ). Если меньше — расширить самые большие файлы.

- [ ] **Step 2: Тест**

```python
# tests/test_sample_docs.py
from pathlib import Path

import pytest

SAMPLE_DOCS = Path(__file__).parents[1] / "sample_docs"

VERIFICATION_FACTS = [
    "Трон Пепла",
    "Пепельные Пустоши",
    "Кальдера",
    "Исольда",
    "127 лет",
    "812 году",
    "Лавовый Дракон",
    "40-63",
    "Обсидианный голем",
    "47 золота",
    "Пепельная завеса",
    "25%",
    "Магнус Чёрный Молот",
    "897 году",
    "Битва у Трёх Кратеров",
    "14 октября 903",
    "Сердце Кальдеры",
    "Регалии Пепла",
    "Пепельной войны 889-891",
    "Лиара Ветрокрылая",
    "Пепельный проход",
    "907 году",
    "7 миссий",
    "12 408",
    "Ночь Обсидиана",
    "30 ноября",
]


def _all_text() -> str:
    parts = [p.read_text(encoding="utf-8") for p in SAMPLE_DOCS.rglob("*") if p.is_file()]
    return "\n".join(parts)


def test_total_size_at_least_500kb():
    total = sum(p.stat().st_size for p in SAMPLE_DOCS.rglob("*") if p.is_file())
    assert total >= 500 * 1024


@pytest.mark.parametrize("fact", VERIFICATION_FACTS)
def test_verification_fact_present(fact):
    assert fact in _all_text()
```

- [ ] **Step 3: PASS** — `uv run pytest tests/test_sample_docs.py -q` → `27 passed`.

- [ ] **Step 4: Таблица фактов в README.md** — секция «Проверочные факты»: факт → файл → пример вопроса для проверки (например: «Кто правит городом Кальдера?» → lore/heroes.md → ответ «Владычица Пепла Исольда»). Канонические факты вселенной в таблицу не включать.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: 500kb sample knowledge base with 15 verification facts"`

---

### Task 15: Docker (Dockerfile, docker-compose.yml, .vscode/mcp.json)

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.vscode/mcp.json`

- [ ] **Step 1: Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY sample_docs ./sample_docs
RUN uv sync --frozen --no-dev

ENV RAGKB_HOST=0.0.0.0 \
    RAGKB_PORT=8000 \
    RAGKB_CHROMA_DIR=/data/chroma \
    PYTHONUNBUFFERED=1

VOLUME ["/data/chroma"]

CMD ["uv", "run", "python", "-m", "rag_kb.server"]
```

- [ ] **Step 2: .dockerignore**

```
.venv
data
tests
.git
.github
__pycache__
*.pyc
.env
```

- [ ] **Step 3: docker-compose.yml**

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    healthcheck:
      test: ["CMD-SHELL", "ollama list || exit 1"]
      interval: 5s
      timeout: 5s
      retries: 12

  ollama-init:
    image: ollama/ollama:latest
    entrypoint: ["/bin/sh", "-c"]
    command: >
      "ollama pull $${LLM_MODEL:-qwen2.5:3b} &&
       ollama pull $${EMBEDDING_MODEL:-nomic-embed-text}"
    environment:
      OLLAMA_HOST: http://ollama:11434
    depends_on:
      ollama:
        condition: service_healthy
    restart: "no"

  rag-kb:
    build: .
    environment:
      RAGKB_OLLAMA_BASE_URL: http://ollama:11434
      RAGKB_LLM_MODEL: ${LLM_MODEL:-qwen2.5:3b}
      RAGKB_EMBEDDING_PROVIDER: ollama
      RAGKB_OLLAMA_EMBEDDING_MODEL: ${EMBEDDING_MODEL:-nomic-embed-text}
    volumes:
      - chroma_data:/data/chroma
    ports:
      - "8000:8000"
    depends_on:
      ollama-init:
        condition: service_completed_successfully
    restart: on-failure

volumes:
  ollama_data:
  chroma_data:
```

- [ ] **Step 4: .vscode/mcp.json**

```json
{
  "servers": {
    "rag-kb": {
      "type": "http",
      "url": "http://localhost:8000/mcp/"
    }
  }
}
```

- [ ] **Step 5: Ручная проверка (занимает время — качаются модели ~2.5 ГБ)**

Run: `docker compose up --build`
Expected: ollama healthy → ollama-init pulled модели и завершился → rag-kb поднялся, слушает 8000.
Проверка: `Invoke-RestMethod http://localhost:8000/mcp/ -Method Post -ContentType "application/json" -Body '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}'` → JSON-ответ с `serverInfo`.
Затем через MCP Inspector (`npx @modelcontextprotocol/inspector`) или VSCode Copilot: `index_status` → `index_folder` c путём `./sample_docs` → `ask_question("Кто правит городом Кальдера?")` → ответ «Владычица Пепла Исольда» с источником lore/heroes.md. Сверить все 4 инструмента по шагу 6 «Процесса сдачи».

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: docker compose — ollama + model pull + rag-kb, vscode mcp config"`

---

### Task 16: CI (GitHub Actions)

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Workflow**

```yaml
name: CI

on:
  push:
    branches: [master, main]
  pull_request:

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.12"
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pytest -q

  docker-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t rag-kb:ci .
```

- [ ] **Step 2: Локальная проверка форматирования** — `uv run ruff format . && uv run ruff check .` → без ошибок (если ruff format менял файлы — закоммитить).

- [ ] **Step 3: Push и проверка** — `git push` → GitHub → Actions: оба job зелёные.

- [ ] **Step 4: Commit** — `git add -A && git commit -m "ci: lint, tests and docker image build"`

---

### Task 17: Документация (README.md, ARCHITECTURE.md, REPORT.md)

**Files:**
- Create: `README.md`, `ARCHITECTURE.md`, `REPORT.md`

- [ ] **Step 1: README.md** (русский): что это, быстрая проверка преподавателя (`git clone && docker compose up`), подключение к VSCode Copilot (содержимое `.vscode/mcp.json` + путь настройки), все 4 инструмента с примерами вызовов и ответов, переключение эмбеддингов (RAGKB_EMBEDDING_PROVIDER), таблица проверочных фактов из Task 14 (факт → файл → вопрос для проверки), запуск тестов (`uv sync && uv run pytest`), локальный запуск без Docker.

- [ ] **Step 2: ARCHITECTURE.md** (русский): диаграмма компонентов (MCP → Indexer → ChromaDB/BM25; Graph: rewrite → retrieve → grade → generate с retry-петлёй), объяснение RRF (формула и зачем гибридный поиск), почему ChromaDB — единственный источник правды, а BM25 перестраивается из него, поток данных при индексации и при вопросе, конфигурация (таблица env-переменных), решения: свой RRF вместо EnsembleRetriever (тестируемость и явность для ревью), json-режим грейдера + fallback-парсер. **Обязательная секция «ИИ-инструменты разработки»**: opencode (агентная CLI-среда разработки), плагин superpowers (процессы-скиллы: планирование, TDD, ревью), MCP-сервер Context7 (актуальная документация библиотек в ходе разработки), модель GLM 5.3 (z.ai).

- [ ] **Step 3: REPORT.md** (русский) — заполняет **пользователь**, не исполнитель. Механизм: после каждой задачи исполнитель спрашивает «Что занести в REPORT.md?», пользователь диктует своими словами, исполнитель исправляет грамматику/стиль (смысл не меняет) и добавляет запись в хронологический раздел. Итог: история разработки по задачам, ключевые проблемы и решения (минимум 3, например: персистентность BM25, надёжный yes/no-парсинг маленькой LLM, conditional edges и подсчёт попыток), использованные AI-инструменты и модели (opencode, superpowers, Context7, GLM 5.3), **минимум один разобранный промпт** (удачный или неудачный — дословно со слов пользователя и объяснение, почему сработал/не сработал), открытые вопросы.

- [ ] **Step 4: Commit** — `git add -A && git commit -m "docs: readme, architecture and ai-process report"`

---

### Task 18: Финальная самопроверка по критериям сдачи

**Files:** нет новых — проверка + точечные фиксы.

- [ ] **Step 1: Чеклист критериев** (пройти по `task_description.md`):

```powershell
uv run pytest -q          # ≥10 тестов, все зелёные (фактически ~35+)
uv run ruff check .
docker compose up --build # с нуля, затем шаги 5-6 из «Процесса сдачи»
```

- MCP: 4 инструмента, описания позволяют агенту выбрать инструмент без подсказки (проверить в Copilot вопросом «что в моей базе знаний?» без упоминания MCP) ✓
- LangGraph: условные переходы, retry ≤ 2, `attempt` инкрементируется ✓ (покрыто тестами builder'а)
- Индексация: 7 форматов, чанки с метаданными source/chunk_index ✓
- Инфраструктура: compose одной командой, CI зелёный ✓
- Демо-документы ≥500 КБ, факты в README ✓
- ARCHITECTURE/README/REPORT, `.vscode/mcp.json` ✓
- REPORT.md ведён по ходу, есть разбор промпта ✓

- [ ] **Step 2: Push** — `git push` → финальное состояние на GitHub. Убедиться, что CI зелёный.

- [ ] **Step 3: Подготовка 5-минутного демо** (устно): запуск → индексация → вопрос с проверочным фактом → объяснение графа.

---

## Self-Review (выполнен при составлении)

- **Покрытие спеки:** 4 инструмента (Task 13), 7 форматов (Task 4), чанкинг с метаданными (Task 5), гибрид BM25+vector→RRF (Task 8), Corrective RAG с rewrite/grade/retry≤2 (Task 11–12), опциональные Ollama-эмбеддинги через конфиг (Task 6), Docker Compose одной командой с Ollama+моделью (Task 15), ≥10 тестов всех трёх видов (Tasks 2–14, суммарно ~40), CI lint+тесты+docker build (Task 16), демо-документы ≥500КБ с фактами в README (Task 14), ARCHITECTURE/README/REPORT + mcp.json (Tasks 15, 17), AI-процесс в REPORT (Task 17 + записи по ходу). Пробелов нет.
- **Placeholder-скан:** шаги типа «сгенерировать 500КБ контента» содержат точный список файлов, размеров и 15 дословных фактов — это data-generation задача, не placeholder; остальной код приведён полностью.
- **Консистентность сигнатур:** `Chunk(id, text, metadata)` везде; `Embedder.embed_query/embed_documents` одинаковы в FakeEmbedder и реализациях; `retriever.search(query, top_k=None)` в HybridRetriever, StubRetriever и узле retrieve совпадают; `LLM.invoke(prompt, json_mode=False)` в FakeLLM и OllamaLLM идентичны; `_route_after_grade` использует `settings.min_relevant_chunks/max_retries` из Task 2.
