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
            metadata={"hnsw:space": "l2"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        # chromadb 1.5.9 запрещает дубликаты id внутри одного upsert-батча;
        # оставляем последнее вхождение (семантика upsert: последняя запись побеждает)
        dedup: dict[str, tuple[Chunk, list[float]]] = {}
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            dedup[chunk.id] = (chunk, embedding)
        pairs = list(dedup.values())
        self._collection.upsert(
            ids=[c.id for c, _ in pairs],
            documents=[c.text for c, _ in pairs],
            metadatas=[c.metadata for c, _ in pairs],
            embeddings=[e for _, e in pairs],
        )

    def query(self, embedding: list[float], top_k: int) -> list[Chunk]:
        if self.count() == 0:
            return []
        res = self._collection.query(
            query_embeddings=[embedding], n_results=min(top_k, self.count()),
            include=["documents", "metadatas"],
        )
        return [
            Chunk(id=i, text=d, metadata=m)
            for i, d, m in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], strict=True)
        ]

    def all_chunks(self) -> list[Chunk]:
        res = self._collection.get(include=["documents", "metadatas"])
        return [Chunk(id=i, text=d, metadata=m)
                for i, d, m in zip(res["ids"], res["documents"], res["metadatas"], strict=True)]

    def count(self) -> int:
        return self._collection.count()

    def unique_sources(self) -> list[str]:
        return sorted({c.metadata["source"] for c in self.all_chunks()})

    def delete_by_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def set_last_indexed_at(self, iso: str) -> None:
        merged = {**(self._collection.metadata or {}), "last_indexed_at": iso}
        # chromadb запрещает передавать hnsw:* в modify, даже без изменений;
        # настройка пространства сохраняется в индексе после создания коллекции
        merged = {k: v for k, v in merged.items() if not k.startswith("hnsw:")}
        self._collection.modify(metadata=merged)

    def get_last_indexed_at(self) -> str | None:
        return (self._collection.metadata or {}).get("last_indexed_at")
