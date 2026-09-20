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
    ollama_keep_alive_sec: int = 2_592_000  # удержание моделей в RAM Ollama, сек; 30 суток

    # Лимиты генерации (num_predict, токены)
    num_predict_batch_grade: int = 32  # пакетный грейдинг: список номеров релевантных
    num_predict_grade: int = 8  # поштучный грейдинг (фолбэк): yes/no
    num_predict_rewrite: int = 100  # переформулировка запроса
    num_predict_generate: int = 300  # генерация ответа (краткий ответ по сути)

    # Чанкинг
    chunk_size: int = 1200
    chunk_overlap: int = 200
    grade_chunk_chars: int = 1200  # символ чанка в промпте грейдера (чанки ~1200: без усечения)

    # Поиск и граф
    top_k: int = 10
    rrf_k: int = 60
    min_relevant_chunks: int = 1
    max_retries: int = 2
    grade_mode: str = "per_chunk"  # грейдинг: "per_chunk" (строже, надёжнее) | "batch" (быстрее)

    # ask_question
    # сколько ждать ответ графа до отдачи in_progress — меньше типовых клиентских таймаутов
    ask_wait_seconds: int = 25
    ask_cache_size: int = 10  # сколько последних вопросов держать в кеше ответов


@lru_cache
def get_settings() -> Settings:
    return Settings()
