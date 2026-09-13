# tests/unit/test_builder.py
from rag_kb.config import Settings
from rag_kb.graph.builder import build_graph
from rag_kb.types import Chunk
from tests.conftest import FakeLLM


class StubRetriever:
    def __init__(self):
        self.results = []
        self.queries = []

    def search(self, query, top_k=None):
        self.queries.append(query)
        return self.results


def chunk(i, text):
    return Chunk(id=str(i), text=text,
                 metadata={"source": f"f{i}.md", "chunk_index": 0, "total_chunks": 1,
                           "doc_type": "text"})


def build(retriever, answers):
    return build_graph(retriever, FakeLLM(answers), Settings())


def test_happy_path_generate_after_good_grade():
    r = StubRetriever()
    r.results = [chunk(0, "про кэш")]
    graph = build(r, ['{"relevant": "yes"}', "Кэш работает через Redis."])
    out = graph.invoke({"question": "как работает кэш?"})
    assert out["answer"] == "Кэш работает через Redis."
    assert out["sources"] == ["f0.md"]
    assert out["attempt"] == 1


def test_retry_loop_broadens_query():
    r = StubRetriever()
    r.results = [chunk(0, "про кэш")]
    graph = build(r, ['{"relevant": "no"}', "кэш redis TTL хранение",
                      '{"relevant": "yes"}', "Ответ."])
    out = graph.invoke({"question": "почему данные устаревают?"})
    assert out["answer"] == "Ответ."
    assert out["attempt"] == 2  # был повторный поиск
    assert r.queries == ["почему данные устаревают?", "кэш redis TTL хранение"]


def test_max_two_retries_then_generate_with_what_we_have():
    r = StubRetriever()
    r.results = [chunk(0, "нерелевантное")]
    graph = build(r, ['{"relevant": "no"}', "запрос2", '{"relevant": "no"}', "запрос3",
                      '{"relevant": "yes"}', "Спасательный ответ."])
    out = graph.invoke({"question": "q?"})
    assert out["attempt"] == 3  # 1 попытка + 2 retry
    assert out["answer"] == "Спасательный ответ."


def test_nothing_relevant_anywhere():
    r = StubRetriever()
    r.results = [chunk(0, "мимо")]
    graph = build(r, ['{"relevant": "no"}', "запрос2", '{"relevant": "no"}', "запрос3",
                      '{"relevant": "no"}'])
    out = graph.invoke({"question": "q?"})
    assert "ничего не найдено" in out["answer"].lower()
    assert out["sources"] == []
