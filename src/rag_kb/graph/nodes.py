# src/rag_kb/graph/nodes.py
import json
import re

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
    "Определи, релевантен ли фрагмент документа вопросу для ответа на него.\n"
    'Ответь строго JSON: {{"relevant": "yes"}} или {{"relevant": "no"}}.\n\n'
    "Вопрос: {question}\n\nФрагмент:\n{chunk}"
)

GENERATE_PROMPT = (
    "Ты — помощник по внутренней базе знаний. Ответь на вопрос пользователя, опираясь ТОЛЬКО "
    "на приведённые фрагменты. Не выдумывай. Если фрагменты не содержат ответа — так и скажи.\n\n"
    "Вопрос: {question}\n\nФрагменты:\n{context}"
)

NOT_FOUND_ANSWER = (
    "В проиндексированной базе знаний ничего не найдено "
    "(включая повторные запросы с расширенной формулировкой). "
    "Попробуйте переформулировать вопрос или проиндексировать дополнительные папки."
)


def make_rewriter(llm: LLM):
    def rewrite(state: GraphState) -> dict:
        if state["attempt"] == 0:
            return {"query": state["question"].strip()}
        query = llm.invoke(REWRITE_PROMPT.format(question=state["question"], query=state["query"]))
        return {"query": query.strip()}

    return rewrite


def retrieve_node(retriever):
    def retrieve(state: GraphState) -> dict:
        chunks = retriever.search(state["query"])
        return {"chunks": chunks, "attempt": state["attempt"] + 1}

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


def make_grader(llm: LLM):
    def grade(state: GraphState) -> dict:
        relevant: list[Chunk] = []
        for chunk in state["chunks"]:
            raw = llm.invoke(
                GRADE_PROMPT.format(question=state["question"], chunk=chunk.text[:1500]),
                json_mode=True,
            )
            if _parse_relevant(raw):
                relevant.append(chunk)
        return {"relevant": relevant}

    return grade


def make_generator(llm: LLM):
    def generate(state: GraphState) -> dict:
        if not state["relevant"]:
            return {"answer": NOT_FOUND_ANSWER, "sources": []}
        context = "\n\n---\n\n".join(
            f"[{c.metadata['source']}]\n{c.text}" for c in state["relevant"]
        )
        answer = llm.invoke(GENERATE_PROMPT.format(question=state["question"], context=context))
        sources = sorted({c.metadata["source"] for c in state["relevant"]})
        return {"answer": answer.strip(), "sources": sources}

    return generate
