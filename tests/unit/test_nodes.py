# tests/unit/test_nodes.py

from rag_kb.graph.nodes import (
    make_generator,
    make_grader,
    make_rewriter,
    retrieve_node,
)
from rag_kb.graph.state import GraphState
from tests.conftest import FakeEmbedder, FakeLLM  # noqa: F401  (FakeEmbedder для консистентности)


class StubRetriever:
    def __init__(self, results):
        self._results = results
        self.queries = []

    def search(self, query, top_k=None):
        self.queries.append(query)
        return self._results


def chunk(i, text):
    from rag_kb.types import Chunk
    return Chunk(id=str(i), text=text,
                 metadata={"source": f"f{i}.md", "chunk_index": 0, "total_chunks": 1,
                           "doc_type": "text"})


def base_state(**over) -> GraphState:
    state = GraphState(question="как работает кэш?", query="как работает кэш?", attempt=0,
                       chunks=[], relevant=[], answer="", sources=[])
    state.update(over)
    return state


def test_rewriter_first_attempt_normalizes():
    llm = FakeLLM(["не должно вызываться"])
    node = make_rewriter(llm)
    out = node(base_state())
    assert out["query"] == "как работает кэш?"  # первая попытка — LLM не зовём
    assert llm.prompts == []


def test_rewriter_broadens_on_retry():
    llm = FakeLLM(["кэш redis TTL инвалидация хранение"])
    node = make_rewriter(llm)
    out = node(base_state(attempt=1, query="стухание кэша"))
    assert out["query"] == "кэш redis TTL инвалидация хранение"
    assert "стухание кэша" in llm.prompts[0]


def test_retrieve_node_calls_retriever_and_counts_attempt():
    r = StubRetriever([chunk(0, "a"), chunk(1, "b")])
    out = retrieve_node(r)(base_state())
    assert [c.id for c in out["chunks"]] == ["0", "1"]
    assert out["attempt"] == 1


def test_grader_parses_yes_no():
    llm = FakeLLM(['{"relevant": "yes"}', '{"relevant": "no"}'])
    node = make_grader(llm)
    out = node(base_state(chunks=[chunk(0, "про кэш"), chunk(1, "про логи")]))
    assert [c.id for c in out["relevant"]] == ["0"]


def test_grader_tolerates_broken_json():
    llm = FakeLLM(["мусор без json", "relevant: yes точно"])
    node = make_grader(llm)
    out = node(base_state(chunks=[chunk(0, "a"), chunk(1, "b")]))
    assert [c.id for c in out["relevant"]] == ["1"]  # fallback-парсер нашёл yes


def test_grader_tolerates_bare_json_string():
    llm = FakeLLM(['"yes"'])
    node = make_grader(llm)
    out = node(base_state(chunks=[chunk(0, "a")]))
    assert [c.id for c in out["relevant"]] == []


def test_grader_accepts_boolean_json():
    llm = FakeLLM(['{"relevant": true}', '{"relevant": false}'])
    node = make_grader(llm)
    out = node(base_state(chunks=[chunk(0, "a"), chunk(1, "b")]))
    assert [c.id for c in out["relevant"]] == ["0"]


def test_grader_accepts_russian_da():
    llm = FakeLLM(['{"relevant": "да"}'])
    node = make_grader(llm)
    out = node(base_state(chunks=[chunk(0, "a")]))
    assert [c.id for c in out["relevant"]] == ["0"]


def test_generator_answer_and_sources():
    llm = FakeLLM(["Кэш работает так-то."])
    node = make_generator(llm)
    out = node(base_state(relevant=[chunk(0, "про кэш"), chunk(0, "про кэш ещё")]))
    assert out["answer"] == "Кэш работает так-то."
    assert out["sources"] == ["f0.md"]


def test_generator_nothing_relevant():
    llm = FakeLLM([])
    node = make_generator(llm)
    out = node(base_state(relevant=[]))
    assert "ничего не найдено" in out["answer"].lower()
