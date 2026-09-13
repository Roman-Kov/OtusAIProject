# tests/unit/test_rrf.py
from rag_kb.retrieval.hybrid import rrf_fuse
from rag_kb.types import Chunk


def chunk(i):
    return Chunk(id=str(i), text=f"t{i}",
                 metadata={"source": "a", "chunk_index": i, "total_chunks": 5, "doc_type": "text"})


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
