# src/rag_kb/app.py
import dataclasses
import json

from fastmcp import FastMCP

from rag_kb.ask_runner import AskRunner
from rag_kb.config import Settings
from rag_kb.indexing.indexer import Indexer
from rag_kb.retrieval.hybrid import HybridRetriever


def create_mcp_server(
    indexer: Indexer, retriever: HybridRetriever, graph_factory, settings: Settings
) -> FastMCP:
    mcp = FastMCP(
        name="rag-kb",
        instructions=(
            "База знаний по локальным документам пользователя. Сначала индексируйте папку "
            "(index_folder), затем отвечайте на вопросы через ask_question или ищите фрагменты "
            "через find_relevant_docs. Ответы основаны только на проиндексированных документах: "
            "на любые вопросы о предметной области пользователя отвечайте через ask_question, "
            "а не из общих знаний модели."
        ),
    )

    @mcp.tool
    def index_folder(path: str, pattern: str = "**/*") -> str:
        """Запустить индексацию папки с документами в базу знаний (в фоне, отвечает сразу).

        Сканирует файлы (.md, .txt, .py, .js, .ts, .json, .yaml) по glob-паттерну,
        разбивает на чанки, создаёт эмбеддинги и сохраняет в векторное хранилище.
        Вызывайте, когда пользователь просит «проиндексируй папку», добавить документы
        в базу знаний, или когда по вопросу пользователя индекс ещё пуст.
        Индексация идёт в фоне, ответ приходит мгновенно: следите за ходом и итогом
        через index_status (indexing_in_progress, progress, last_report).
        Повторный вызов во время индексации безопасен (вернёт already_running).
        Возвращает JSON: status ("started" | "already_running"), path/pattern или progress.
        """
        return json.dumps(indexer.start_indexing(path, pattern), ensure_ascii=False)

    ask_runner = AskRunner(
        graph_factory, wait_seconds=settings.ask_wait_seconds, cache_size=settings.ask_cache_size
    )

    @mcp.tool
    def ask_question(question: str) -> str:
        """Ответить на вопрос по базе знаний пользователя (RAG).

        Поиск по проиндексированным документам пользователя и ответ с указанием
        источников.

        Вызывайте ВСЕГДА, когда вопрос касается предметной области пользователя
        или содержимого его базы знаний: «что такое X», «кто такой Y», «что
        написано про Z», «как работает W», «когда/сколько/где …». Даже если вам
        кажется, что вы знаете ответ или не знаете его вообще — истина в локальной
        базе пользователя, а не в общих знаниях модели. Не отвечайте на такие
        вопросы из своей памяти: сначала спросите базу.
        Пайплайн: переформулировка запроса → гибридный поиск (ключевые слова +
        смысл) → LLM-оценка релевантности → ответ только по релевантным фрагментам;
        при неуспехе запрос расширяется и поиск повторяется (до 2 раз).
        Возвращает JSON: answer (текст ответа), sources (список файлов-источников)
        и status ("done"; "in_progress" — ответ ещё готовится; "error" — текст проблемы).
        Если пришёл status="in_progress" — ответ ещё готовится в фоне: подождите
        retry_after_seconds секунд (обычно 30) и вызовите ask_question ПОВТОРНО
        с тем же вопросом — готовый ответ вернётся мгновенно из кеша.
        """
        return json.dumps(ask_runner.ask(question), ensure_ascii=False)

    @mcp.tool
    def find_relevant_docs(query: str, top_k: int = 5) -> str:
        """Найти релевантные фрагменты документов без генерации ответа.

        Гибридный поиск: точное совпадение ключевых слов (BM25) + семантический
        поиск по эмбеддингам, объединение через Reciprocal Rank Fusion.
        Вызывайте, когда нужны сами цитаты/фрагменты или точное место в файлах
        («в каком файле упомянуто X», «покажи кусок про Y»), без LLM-пересказа.
        Возвращает JSON-список чанков: текст, источник, позиция в файле.
        """
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        chunks = retriever.search(query, top_k=top_k)
        return json.dumps(
            [{"text": c.text, "metadata": c.metadata} for c in chunks], ensure_ascii=False
        )

    @mcp.tool
    def index_status() -> str:
        """Показать статистику базы знаний, ход фоновой индексации и итог последнего запуска.

        Вызывайте перед поиском, чтобы понять, проиндексировано ли что-то вообще,
        а также когда пользователь спрашивает «что в базе» / «сколько документов»,
        и после запуска index_folder — чтобы отслеживать прогресс (indexing_in_progress,
        progress вида «3/17») до завершения.
        Возвращает JSON: files, chunks, last_indexed_at, indexing_in_progress, progress
        и last_report (files/chunks/seconds/errors последнего запуска, null если ещё не было).
        """
        stats = indexer.status()
        return json.dumps(
            {
                "files": stats.files,
                "chunks": stats.chunks,
                "last_indexed_at": stats.last_indexed_at,
                "indexing_in_progress": stats.indexing_in_progress,
                "progress": stats.progress,
                "last_report": (
                    dataclasses.asdict(stats.last_report) if stats.last_report else None
                ),
            },
            ensure_ascii=False,
        )

    return mcp
