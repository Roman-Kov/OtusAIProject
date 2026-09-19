# src/rag_kb/llm.py
from typing import Protocol

from langchain_ollama import ChatOllama


class LLM(Protocol):
    def invoke(self, prompt: str, num_predict: int | None = None) -> str: ...


class OllamaLLM:
    """Обёртка над ChatOllama: клиенты кэшируются по num_predict.

    keep_alive удерживает модель в RAM Ollama между вызовами — без холодного
    старта после простоя.
    """

    def __init__(self, base_url: str, model: str, keep_alive: int) -> None:
        self._base_url = base_url
        self._model = model
        self._keep_alive = keep_alive
        self._clients: dict[int | None, ChatOllama] = {}

    def _client(self, num_predict: int | None) -> ChatOllama:
        if num_predict not in self._clients:
            self._clients[num_predict] = ChatOllama(
                base_url=self._base_url,
                model=self._model,
                keep_alive=self._keep_alive,
                num_predict=num_predict,
            )
        return self._clients[num_predict]

    def invoke(self, prompt: str, num_predict: int | None = None) -> str:
        return self._client(num_predict).invoke(prompt).content
