# tests/unit/test_ask_runner.py
import time

from rag_kb.ask_runner import AskRunner


class StubGraph:
    def __init__(
        self, result: dict | None = None, delay: float = 0.0, error: Exception | None = None
    ):
        self.result = result or {"answer": "ответ", "sources": ["a.md"]}
        self.delay = delay
        self.error = error
        self.invocations = 0

    def invoke(self, _state) -> dict:
        self.invocations += 1
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result


def test_fast_graph_returns_answer_synchronously():
    graph = StubGraph({"answer": "Кальдерой правит Исольда.", "sources": ["calderra.md"]})
    runner = AskRunner(lambda q: graph, wait_seconds=5)
    out = runner.ask("кто правит?")
    assert out["status"] == "done"
    assert out["answer"] == "Кальдерой правит Исольда."
    assert out["sources"] == ["calderra.md"]


def test_wait_covers_short_delay():
    graph = StubGraph(delay=0.2)
    runner = AskRunner(lambda q: graph, wait_seconds=2)
    out = runner.ask("вопрос")
    assert out["status"] == "done"


def test_slow_graph_returns_in_progress_then_cached_result():
    graph = StubGraph(delay=0.5)
    runner = AskRunner(lambda q: graph, wait_seconds=0)
    first = runner.ask("медленный вопрос")
    assert first["status"] == "in_progress"
    assert first["sources"] == []
    time.sleep(0.7)
    second = runner.ask("медленный вопрос")
    assert second["status"] == "done"
    assert second["answer"] == "ответ"
    assert graph.invocations == 1  # граф не запускался повторно — ответ из кеша


def test_graph_error_becomes_answer_not_exception():
    graph = StubGraph(error=RuntimeError("ollama недоступна"))
    runner = AskRunner(lambda q: graph, wait_seconds=5)
    out = runner.ask("вопрос")
    assert out["status"] == "error"
    assert "ollama недоступна" in out["answer"]
    assert out["sources"] == []


def test_different_questions_run_separately():
    graph = StubGraph()
    runner = AskRunner(lambda q: graph, wait_seconds=5)
    runner.ask("первый")
    runner.ask("второй")
    assert graph.invocations == 2
