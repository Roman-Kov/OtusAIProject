# src/rag_kb/types.py
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoadedDoc:
    """Загруженный файл до разбивки на чанки."""

    path: Path
    text: str
    doc_type: str  # "markdown" | "text" | "python" | "js" | "ts" | "json" | "yaml"


@dataclass
class Chunk:
    """Кусок документа с метаданными о происхождении."""

    id: str
    text: str
    metadata: dict  # {"source": str, "chunk_index": int, "total_chunks": int, "doc_type": str}


@dataclass
class IndexReport:
    """Результат вызова index_folder."""

    files: int
    chunks: int
    seconds: float
    errors: list  # list[str]


@dataclass
class IndexStats:
    """Результат вызова index_status."""

    files: int
    chunks: int
    last_indexed_at: str | None
    indexing_in_progress: bool = False
    progress: str | None = None  # "3/17" во время фоновой индексации
    last_report: IndexReport | None = None
