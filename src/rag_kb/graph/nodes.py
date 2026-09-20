# src/rag_kb/graph/nodes.py
import json
import re

from rag_kb.config import Settings
from rag_kb.graph.state import GraphState
from rag_kb.llm import LLM
from rag_kb.types import Chunk

REWRITE_PROMPT = (
    "Ты — помощник по поиску во внутренней базе знаний. Поисковый запрос не нашёл достаточно "
    "материала. Переформулируй и расширь его: добавь синонимы и связанные термины, убери жаргон. "
    "Верни только поисковый запрос одной строкой, без пояснений.\n\nВопрос: {question}\n"
    "Неудачный запрос: {query}"
)

GRADE_PROMPT = (
    "Фрагмент:\n{chunk}\n\nВопрос: {question}\n\n"
    "Содержит ли фрагмент информацию для ответа на вопрос? Ответь одним словом: yes или no."
)

BATCH_GRADE_PROMPT = (
    "Фрагменты:\n{fragments}\n\nВопрос: {question}\n\n"
    "Какие из фрагментов содержат информацию для ответа на вопрос? "
    "Перечисли номера релевантных фрагментов (например: 1, 3). "
    "Если релевантных нет — напиши: none."
)

GENERATE_PROMPT = (
    "Ты — помощник по внутренней базе знаний. Ответь на вопрос пользователя, опираясь ТОЛЬКО "
    "на приведённые фрагменты. Не выдумывай. Если фрагменты не содержат ответа — так и скажи. "
    "Отвечай кратко, в 2-4 предложения, но точные числа, даты, имена и названия переписывай "
    "из фрагментов дословно — не округляй и не заменяй их.\n\n"
    "Вопрос: {question}\n\nФрагменты:\n{context}"
)

NOT_FOUND_ANSWER = (
    "В проиндексированной базе знаний ничего не найдено "
    "(включая повторные запросы с расширенной формулировкой). "
    "Попробуйте переформулировать вопрос или проиндексировать дополнительные папки."
)

_BATCH_NONE = re.compile(r"\b(none|нет|никакие|ни один)\b", re.IGNORECASE)


def make_rewriter(llm: LLM, settings: Settings):
    def rewrite(state: GraphState) -> dict:
        if state.get("attempt", 0) == 0:
            return {"query": state["question"].strip()}
        query = llm.invoke(
            REWRITE_PROMPT.format(question=state["question"], query=state["query"]),
            num_predict=settings.num_predict_rewrite,
        )
        return {"query": query.strip()}

    return rewrite


def retrieve_node(retriever):
    def retrieve(state: GraphState) -> dict:
        chunks = retriever.search(state["query"])
        return {"chunks": chunks, "attempt": state.get("attempt", 0) + 1}

    return retrieve


def _parse_relevant(raw: str) -> bool:
    try:
        value = json.loads(raw)["relevant"]
    except json.JSONDecodeError:
        return bool(re.search(r"\byes\b", raw, re.IGNORECASE))
    except (KeyError, AttributeError, TypeError):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower().startswith(("y", "д"))


def _parse_relevant_numbers(raw: str, count: int) -> list[int] | None:
    """Номера (1-based) релевантных фрагментов из ответа LLM; None — ответ не распарсился.

    Противоречивый ответ («нет» вместе с номерами, «ни один из 6») считается
    нераспознанным и уходит в поштучный фолбэк, а не трактуется наугад.
    """
    none_hit = bool(_BATCH_NONE.search(raw))
    numbers = sorted({n for n in (int(d) for d in re.findall(r"\d+", raw)) if 1 <= n <= count})
    if numbers and none_hit:
        return None
    if none_hit:
        return []
    return numbers or None


def make_grader(llm: LLM, settings: Settings):
    def _grade_each(chunks: list[Chunk], question: str) -> list[Chunk]:
        """Поштучный грейдинг: строже пакетного, при пустом результате запускает retry-цикл."""
        relevant: list[Chunk] = []
        for chunk in chunks:
            raw = llm.invoke(
                GRADE_PROMPT.format(
                    question=question, chunk=chunk.text[: settings.grade_chunk_chars]
                ),
                num_predict=settings.num_predict_grade,
            )
            if _parse_relevant(raw):
                relevant.append(chunk)
        return relevant

    def _grade_batch(chunks: list[Chunk], question: str) -> list[Chunk]:
        """Пакетный грейдинг: один вызов LLM на все чанки (быстрее, но оптимистичнее)."""
        fragments = "\n\n".join(
            f"[{i}]\n{c.text[: settings.grade_chunk_chars]}" for i, c in enumerate(chunks, start=1)
        )
        raw = llm.invoke(
            BATCH_GRADE_PROMPT.format(question=question, fragments=fragments),
            num_predict=settings.num_predict_batch_grade,
        )
        numbers = _parse_relevant_numbers(raw, len(chunks))
        if numbers is None:
            return _grade_each(chunks, question)
        return [chunks[n - 1] for n in numbers]

    def grade(state: GraphState) -> dict:
        chunks: list[Chunk] = state["chunks"]
        if not chunks:
            return {"relevant": []}
        if settings.grade_mode == "batch":
            return {"relevant": _grade_batch(chunks, state["question"])}
        return {"relevant": _grade_each(chunks, state["question"])}

    return grade


def make_generator(llm: LLM, settings: Settings):
    def generate(state: GraphState) -> dict:
        if not state["relevant"]:
            return {"answer": NOT_FOUND_ANSWER, "sources": []}
        context = "\n\n---\n\n".join(
            f"[{c.metadata['source']}]\n{c.text}" for c in state["relevant"]
        )
        answer = llm.invoke(
            GENERATE_PROMPT.format(question=state["question"], context=context),
            num_predict=settings.num_predict_generate,
        )
        sources = sorted({c.metadata["source"] for c in state["relevant"]})
        return {"answer": answer.strip(), "sources": sources}

    return generate
