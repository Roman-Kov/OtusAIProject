# src/rag_kb/graph/state.py
from typing import TypedDict

from rag_kb.types import Chunk


class GraphState(TypedDict):
    question: str          # исходный вопрос пользователя
    query: str             # текущий (пере)сформулированный поисковый запрос
    attempt: int           # номер попытки поиска (0 = первая)
    chunks: list[Chunk]    # чанки после retrieve
    relevant: list[Chunk]  # чанки, оценённые LLM как релевантные
    answer: str
    sources: list[str]     # уникальные source релевантных чанков
