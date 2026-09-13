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
    texts = " ".join(c.text for c in indexer.vector_store.all_chunks())
    assert "вторая" in texts
    assert not any("первая" in c.text for c in indexer.vector_store.all_chunks())


def test_index_folder_missing_path_raises(indexer):
    with pytest.raises(FileNotFoundError):
        indexer.index_folder("Z:/definitely/missing")


def test_index_folder_isolates_broken_file(indexer, tmp_path):
    (tmp_path / "good.md").write_text("нормальный файл", encoding="utf-8")
    (tmp_path / "bad.md").write_bytes(b"\xff\xfe\x00broken")
    report = indexer.index_folder(tmp_path)
    assert report.files == 1
    assert len(report.errors) == 1
    assert "bad.md" in report.errors[0]
    assert indexer.status().files == 1  # хороший файл проиндексирован
