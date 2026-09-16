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


def test_bm25_punctuation_only_query_returns_empty():
    store = BM25Store()
    store.build([chunk(0, "обычный текст"), chunk(1, "ещё текст")])
    assert store.search("!!! ???", top_k=3) == []


def test_bm25_rebuild_old_terms_gone():
    store = BM25Store()
    store.build([chunk(0, "старый индекс")])
    store.build([chunk(0, "новое содержимое про docker")])
    assert store.search("docker", top_k=1)[0][0].text.startswith("новое")
    assert store.search("старый", top_k=1) == []  # замена, а не дополнение
