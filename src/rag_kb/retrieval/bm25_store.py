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
        # не > 0: в rank-bm25 0.2.2 при крошечных корпусах (терм во всех документах)
        # IDF становится отрицательным (eps = epsilon * average_idf < 0),
        # и совпадение получает отрицательный скор; документы без терма дают ровно 0.0
        return [(self._chunks[i], float(scores[i])) for i in order if scores[i] != 0]
