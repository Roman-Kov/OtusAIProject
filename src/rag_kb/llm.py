# src/rag_kb/llm.py
from typing import Protocol

from langchain_ollama import ChatOllama


class LLM(Protocol):
    def invoke(self, prompt: str, json_mode: bool = False) -> str: ...


class OllamaLLM:
    """Обёртка над ChatOllama: plain-режим для текста, json-режим для грейдинга."""

    def __init__(self, base_url: str, model: str) -> None:
        self._plain = ChatOllama(base_url=base_url, model=model)
        self._json = ChatOllama(base_url=base_url, model=model, format="json")

    def invoke(self, prompt: str, json_mode: bool = False) -> str:
        client = self._json if json_mode else self._plain
        return client.invoke(prompt).content
