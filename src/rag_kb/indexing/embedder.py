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

    def __init__(self, base_url: str, model: str, keep_alive: int) -> None:
        self._client = OllamaEmbeddings(base_url=base_url, model=model, keep_alive=keep_alive)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)


def create_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "chromadb":
        return ChromaDefaultEmbedder()
    if settings.embedding_provider == "ollama":
        return OllamaEmbedder(
            settings.ollama_base_url,
            settings.ollama_embedding_model,
            keep_alive=settings.ollama_keep_alive_sec,
        )
    raise ValueError(f"Unknown embedding_provider: {settings.embedding_provider}")
