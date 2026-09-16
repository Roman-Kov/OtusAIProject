# src/rag_kb/indexing/indexer.py
import threading
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
        self._indexing_lock = threading.Lock()
        self._indexing_in_progress = False
        self._progress: str | None = None
        self._last_report: IndexReport | None = None

    def start_indexing(self, folder: str | Path, pattern: str = "**/*") -> dict:
        """Запустить индексацию в фоновом потоке и ответить сразу.

        Повторный вызов, пока индексация идёт, безопасен: вернёт already_running.
        Прогресс и итог — в status() (indexing_in_progress, progress, last_report).
        """
        folder = Path(folder).resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")
        with self._indexing_lock:
            if self._indexing_in_progress:
                return {"status": "already_running", "progress": self._progress}
            self._indexing_in_progress = True
            self._progress = None
        threading.Thread(target=self._run_indexing, args=(folder, pattern), daemon=True).start()
        return {"status": "started", "path": str(folder), "pattern": pattern}

    def _run_indexing(self, folder: Path, pattern: str) -> None:
        try:
            self.index_folder(folder, pattern)
        except Exception as exc:  # noqa: BLE001 — фоновый поток не должен ронять сервер
            self._last_report = IndexReport(
                files=0, chunks=0, seconds=0.0, errors=[f"{type(exc).__name__}: {exc}"]
            )
        finally:
            self._indexing_in_progress = False

    def index_folder(self, folder: str | Path, pattern: str = "**/*") -> IndexReport:
        folder = Path(folder).resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")
        started = time.perf_counter()
        errors: list[str] = []
        total_chunks = 0
        files = scan_folder(folder, pattern)
        self._progress = f"0/{len(files)}"
        for done, path in enumerate(files, start=1):
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
            self._progress = f"{done}/{len(files)}"
        self._bm25.build(self.vector_store.all_chunks())
        self.vector_store.set_last_indexed_at(datetime.now(UTC).isoformat(timespec="seconds"))
        report = IndexReport(
            files=len(files) - len(errors),
            chunks=total_chunks,
            seconds=round(time.perf_counter() - started, 2),
            errors=errors,
        )
        self._last_report = report
        return report

    def status(self) -> IndexStats:
        return IndexStats(
            files=len(self.vector_store.unique_sources()),
            chunks=self.vector_store.count(),
            last_indexed_at=self.vector_store.get_last_indexed_at(),
            indexing_in_progress=self._indexing_in_progress,
            progress=self._progress,
            last_report=self._last_report,
        )
