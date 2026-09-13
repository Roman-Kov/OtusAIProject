# tests/unit/test_config.py
from rag_kb.config import Settings


def test_defaults():
    s = Settings()
    assert s.llm_model == "qwen2.5:3b"
    assert s.embedding_provider == "chromadb"
    assert s.top_k == 6
    assert s.min_relevant_chunks == 1
    assert s.max_retries == 2


def test_env_override(monkeypatch):
    monkeypatch.setenv("RAGKB_LLM_MODEL", "phi3:mini")
    monkeypatch.setenv("RAGKB_EMBEDDING_PROVIDER", "ollama")
    s = Settings()
    assert s.llm_model == "phi3:mini"
    assert s.embedding_provider == "ollama"
