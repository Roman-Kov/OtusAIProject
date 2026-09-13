# tests/unit/test_embedder.py
from rag_kb.config import Settings
from rag_kb.indexing.embedder import ChromaDefaultEmbedder, OllamaEmbedder, create_embedder


def test_factory_returns_chromadb_default():
    emb = create_embedder(Settings(embedding_provider="chromadb"))
    assert isinstance(emb, ChromaDefaultEmbedder)


def test_factory_returns_ollama():
    emb = create_embedder(Settings(embedding_provider="ollama", ollama_base_url="http://x:11434"))
    assert isinstance(emb, OllamaEmbedder)


def test_factory_rejects_unknown():
    import pytest
    with pytest.raises(ValueError, match="embedding_provider"):
        create_embedder(Settings(embedding_provider="openai"))
