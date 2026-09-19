# tests/unit/test_nodes.py

from rag_kb.config import Settings
from rag_kb.graph.nodes import (
    make_generator,
    make_grader,
    make_rewriter,
    retrieve_node,
)
from rag_kb.graph.state import GraphState
from tests.conftest import FakeEmbedder, FakeLLM  # noqa: F401  (FakeEmbedder для консистентности)

SETTINGS = Settings()

# Ответ пакетного грейдера, который гарантированно не парсится как список номеров,
# чтобы переключить узел на поштучный фолбэк.
UNPARSEABLE = "не понимаю вопрос"


class StubRetriever:
    def __init__(self, results):
        self._results = results
        self.queries = []

    def search(self, query, top_k=None):
        self.queries.append(query)
        return self._results


def chunk(i, text):
    from rag_kb.types import Chunk

    return Chunk(
        id=str(i),
        text=text,
        metadata={"source": f"f{i}.md", "chunk_index": 0, "total_chunks": 1, "doc_type": "text"},
    )


def base_state(**over) -> GraphState:
    state = GraphState(
        question="как работает кэш?",
        query="как работает кэш?",
        attempt=0,
        chunks=[],
        relevant=[],
        answer="",
        sources=[],
    )
    state.update(over)
    return state


def test_rewriter_first_attempt_normalizes():
    llm = FakeLLM(["не должно вызываться"])
    node = make_rewriter(llm, SETTINGS)
    out = node(base_state())
    assert out["query"] == "как работает кэш?"  # первая попытка — LLM не зовём
    assert llm.prompts == []


def test_rewriter_broadens_on_retry():
    llm = FakeLLM(["кэш redis TTL инвалидация хранение"])
    node = make_rewriter(llm, SETTINGS)
    out = node(base_state(attempt=1, query="стухание кэша"))
    assert out["query"] == "кэш redis TTL инвалидация хранение"
    assert "стухание кэша" in llm.prompts[0]
    assert llm.calls[0].num_predict == SETTINGS.num_predict_rewrite


def test_retrieve_node_calls_retriever_and_counts_attempt():
    r = StubRetriever([chunk(0, "a"), chunk(1, "b")])
    out = retrieve_node(r)(base_state())
    assert [c.id for c in out["chunks"]] == ["0", "1"]
    assert out["attempt"] == 1


def test_grader_batch_single_call():
    llm = FakeLLM(["1, 3"])
    node = make_grader(llm, SETTINGS)
    out = node(
        base_state(chunks=[chunk(0, "про кэш"), chunk(1, "про логи"), chunk(2, "про TTL кэша")])
    )
    assert [c.id for c in out["relevant"]] == ["0", "2"]
    assert len(llm.prompts) == 1  # один вызов LLM на все чанки
    assert llm.calls[0].num_predict == SETTINGS.num_predict_batch_grade
    prompt = llm.prompts[0]
    assert "[1]" in prompt and "[3]" in prompt  # фрагменты пронумерованы
    # порядок «фрагменты раньше, вопрос ближе к концу» сохранён
    assert prompt.index("[1]") < prompt.index("как работает кэш?")


def test_grader_batch_parses_digits_in_text():
    llm = FakeLLM(["Релевантные фрагменты: 2 и 3."])
    node = make_grader(llm, SETTINGS)
    out = node(base_state(chunks=[chunk(0, "a"), chunk(1, "b"), chunk(2, "c")]))
    assert [c.id for c in out["relevant"]] == ["1", "2"]
    assert len(llm.prompts) == 1


def test_grader_batch_none_answer():
    llm = FakeLLM(["none"])
    node = make_grader(llm, SETTINGS)
    out = node(base_state(chunks=[chunk(0, "a"), chunk(1, "b")]))
    assert out["relevant"] == []
    assert len(llm.prompts) == 1


def test_grader_batch_russian_none():
    llm = FakeLLM(["нет"])
    node = make_grader(llm, SETTINGS)
    out = node(base_state(chunks=[chunk(0, "a")]))
    assert out["relevant"] == []


def test_grader_empty_chunks_skips_llm():
    llm = FakeLLM([])
    node = make_grader(llm, SETTINGS)
    out = node(base_state(chunks=[]))
    assert out["relevant"] == []
    assert llm.prompts == []


def test_grader_ignores_out_of_range_numbers():
    llm = FakeLLM(["9", '{"relevant": "yes"}'])
    node = make_grader(llm, SETTINGS)
    out = node(base_state(chunks=[chunk(0, "a")]))
    # «9» вне диапазона и без слов «нет» — пакет не распарсился, ушёл в фолбэк
    assert [c.id for c in out["relevant"]] == ["0"]
    assert len(llm.prompts) == 2


def test_grader_contradictory_none_with_numbers_falls_back():
    # «ни один из 6 не релевантен» — «нет» вместе с номером; трактовать наугад нельзя
    chunks = [chunk(i, f"текст {i}") for i in range(6)]
    llm = FakeLLM(["ни один из 6 фрагментов не релевантен"] + ['{"relevant": "no"}'] * 6)
    node = make_grader(llm, SETTINGS)
    out = node(base_state(chunks=chunks))
    assert out["relevant"] == []
    assert len(llm.prompts) == 7  # пакетный ответ забракован + 6 поштучных


def test_grader_negated_digit_does_not_select_chunk():
    # само по себе «нет» без номеров — валидный пустой ответ, фолбэка нет
    llm = FakeLLM(["нет релевантных фрагментов"])
    node = make_grader(llm, SETTINGS)
    out = node(base_state(chunks=[chunk(0, "a"), chunk(1, "b")]))
    assert out["relevant"] == []
    assert len(llm.prompts) == 1


def test_grader_fallback_per_chunk_on_unparseable():
    llm = FakeLLM([UNPARSEABLE, '{"relevant": "yes"}', '{"relevant": "no"}'])
    node = make_grader(llm, SETTINGS)
    out = node(base_state(chunks=[chunk(0, "про кэш"), chunk(1, "про логи")]))
    assert [c.id for c in out["relevant"]] == ["0"]
    assert len(llm.prompts) == 3  # 1 пакетный + 2 поштучных
    assert llm.calls[1].num_predict == SETTINGS.num_predict_grade


def test_grader_fallback_parses_yes_no():
    llm = FakeLLM([UNPARSEABLE, '{"relevant": "yes"}', '{"relevant": "no"}'])
    node = make_grader(llm, SETTINGS)
    out = node(base_state(chunks=[chunk(0, "про кэш"), chunk(1, "про логи")]))
    assert [c.id for c in out["relevant"]] == ["0"]


def test_grader_fallback_tolerates_broken_json():
    llm = FakeLLM([UNPARSEABLE, "relevant: yes точно", "no"])
    node = make_grader(llm, SETTINGS)
    out = node(base_state(chunks=[chunk(0, "a"), chunk(1, "b")]))
    assert [c.id for c in out["relevant"]] == ["0"]  # fallback-парсер нашёл yes


def test_grader_fallback_tolerates_bare_json_string():
    llm = FakeLLM([UNPARSEABLE, '"yes"'])
    node = make_grader(llm, SETTINGS)
    out = node(base_state(chunks=[chunk(0, "a")]))
    assert [c.id for c in out["relevant"]] == []


def test_grader_fallback_accepts_boolean_json():
    llm = FakeLLM([UNPARSEABLE, '{"relevant": true}', '{"relevant": false}'])
    node = make_grader(llm, SETTINGS)
    out = node(base_state(chunks=[chunk(0, "a"), chunk(1, "b")]))
    assert [c.id for c in out["relevant"]] == ["0"]


def test_grader_fallback_accepts_russian_da():
    llm = FakeLLM([UNPARSEABLE, '{"relevant": "да"}'])
    node = make_grader(llm, SETTINGS)
    out = node(base_state(chunks=[chunk(0, "a")]))
    assert [c.id for c in out["relevant"]] == ["0"]


def test_grader_truncates_chunks_by_setting():
    llm = FakeLLM(["1"])
    s = Settings(grade_chunk_chars=50)
    node = make_grader(llm, s)
    node(base_state(chunks=[chunk(0, "а" * 500)]))
    prompt = llm.prompts[0]
    assert "а" * 50 in prompt  # вошли первые 50 символов
    assert "а" * 51 not in prompt  # длиннее настройки — нет


def test_generator_answer_and_sources():
    llm = FakeLLM(["Кэш работает так-то."])
    node = make_generator(llm, SETTINGS)
    out = node(base_state(relevant=[chunk(0, "про кэш"), chunk(0, "про кэш ещё")]))
    assert out["answer"] == "Кэш работает так-то."
    assert out["sources"] == ["f0.md"]
    assert llm.calls[0].num_predict == SETTINGS.num_predict_generate


def test_generator_nothing_relevant():
    llm = FakeLLM([])
    node = make_generator(llm, SETTINGS)
    out = node(base_state(relevant=[]))
    assert "ничего не найдено" in out["answer"].lower()
