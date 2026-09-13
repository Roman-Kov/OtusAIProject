# src/rag_kb/config.py
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAGKB_", extra="ignore")

    # MCP-сервер
    host: str = "0.0.0.0"
    port: int = 8000

    # Хранилище
    chroma_dir: Path = Path("data/chroma")
    collection_name: str = "documents"

    # LLM / эмбеддинги
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b"
    embedding_provider: str = "chromadb"  # "chromadb" | "ollama"
    ollama_embedding_model: str = "nomic-embed-text"

    # Чанкинг
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # Поиск и граф
    top_k: int = 6
    rrf_k: int = 60
    min_relevant_chunks: int = 1
    max_retries: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()
