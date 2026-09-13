# src/rag_kb/retrieval/hybrid.py
from rag_kb.config import Settings
from rag_kb.indexing.embedder import Embedder
from rag_kb.retrieval.bm25_store import BM25Store
from rag_kb.retrieval.stores import VectorStore
from rag_kb.types import Chunk


def rrf_fuse(list_a: list[Chunk], list_b: list[Chunk], k: int = 60,
             top_k: int | None = None) -> list[Chunk]:
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

    def __init__(self, vector_store: VectorStore, embedder: Embedder, bm25: BM25Store,
                 settings: Settings) -> None:
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
