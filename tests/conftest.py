# tests/conftest.py
class FakeEmbedder:
    """Детерминированные эмбеддинги без сети: вектор из хэша текста."""

    def _vec(self, text: str) -> list[float]:
        import hashlib
        digest = hashlib.sha1(text.encode()).digest()
        return [b / 255.0 for b in digest[:16]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeLLM:
    """Программируемая LLM для тестов графа: pop ответов по очереди, запись промптов."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def invoke(self, prompt: str, json_mode: bool = False) -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0)
