# src/rag_kb/ask_runner.py
"""Синхронный ask_question без клиентских таймаутов.

Граф исполняется в фоновом потоке, а вызов ждёт до wait_seconds: успел — клиент
получает обычный ответ с источниками. Не успел — вместо ошибки возвращается
status=in_progress, и повторный вызов с тем же вопросом мгновенно отдаёт
готовый результат из кеша. Ошибки графа не доходят до клиента MCP-ошибкой —
превращаются в текст ответа.
"""

import threading


class _Job:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.result: dict = {}


class AskRunner:
    """Запускает граф вопроса в фоне, кеширует последние ответы."""

    def __init__(self, graph_factory, wait_seconds: int, cache_size: int = 10) -> None:
        self._graph_factory = graph_factory
        self._wait_seconds = wait_seconds
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}

    def ask(self, question: str) -> dict:
        with self._lock:
            job = self._jobs.get(question)
            if job is None:
                job = _Job()
                self._jobs[question] = job
                self._evict()
                threading.Thread(target=self._run, args=(question, job), daemon=True).start()
        job.done.wait(self._wait_seconds)
        if job.done.is_set():
            return job.result
        return {
            "status": "in_progress",
            "retry_after_seconds": 30,
            "question": question,
            "answer": (
                "Ответ ещё готовится (локальная модель на CPU может работать минуты). "
                "Повторите вызов ask_question с этим же вопросом — готовый ответ "
                "вернётся мгновенно."
            ),
            "sources": [],
        }

    def _run(self, question: str, job: _Job) -> None:
        try:
            state = self._graph_factory(question).invoke({"question": question})
            job.result = {
                "status": "done",
                "answer": state.get("answer", ""),
                "sources": state.get("sources", []),
            }
        except Exception as exc:  # noqa: BLE001 — ошибка графа не должна ронять инструмент
            job.result = {
                "status": "error",
                "answer": f"Не удалось получить ответ: {exc}",
                "sources": [],
            }
        finally:
            job.done.set()

    def _evict(self) -> None:
        """Держим в кеше не больше cache_size последних вопросов."""
        while len(self._jobs) > self._cache_size:
            self._jobs.pop(next(iter(self._jobs)))
