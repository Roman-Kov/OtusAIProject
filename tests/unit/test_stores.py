# tests/unit/test_stores.py
from rag_kb.retrieval.stores import VectorStore
from rag_kb.types import Chunk


def make_chunks(n=3, source="a.md"):
    return [
        Chunk(id=f"id{i}", text=f"текст номер {i}",
              metadata={"source": source, "chunk_index": i, "total_chunks": n, "doc_type": "text"})
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
    store.add_chunks(make_chunks(2, source="a.md") + make_chunks(2, source="b.md"),
                     [[0.0] * 4] * 4)
    store.delete_by_source("a.md")
    assert store.count() == 2
    assert store.unique_sources() == ["b.md"]


def test_last_indexed_at(tmp_path):
    store = make_store(tmp_path)
    assert store.get_last_indexed_at() is None
    store.set_last_indexed_at("2026-09-13T10:00:00")
    assert store.get_last_indexed_at() == "2026-09-13T10:00:00"
