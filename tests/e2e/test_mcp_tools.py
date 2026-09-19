# tests/e2e/test_mcp_tools.py
import asyncio
import json
import time

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from rag_kb.app import create_mcp_server
from rag_kb.config import Settings
from rag_kb.graph.builder import build_graph
from rag_kb.indexing.indexer import Indexer
from rag_kb.retrieval.bm25_store import BM25Store
from rag_kb.retrieval.hybrid import HybridRetriever
from rag_kb.retrieval.stores import VectorStore
from tests.conftest import FakeEmbedder, FakeLLM


async def _call(client, tool: str, args: dict) -> dict:
    return json.loads((await client.call_tool(tool, args)).content[0].text)


async def _wait_until_indexed(client, timeout: float = 60.0) -> dict:
    """Поллинг index_status, пока фоновая индексация не завершится."""
    status = await _call(client, "index_status", {})
    deadline = time.monotonic() + timeout
    while status["indexing_in_progress"] and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
        status = await _call(client, "index_status", {})
    assert not status["indexing_in_progress"]
    return status


@pytest.fixture
def mcp(tmp_path):
    settings = Settings(chroma_dir=tmp_path / "chroma")
    vector_store = VectorStore(settings)
    embedder = FakeEmbedder()
    bm25 = BM25Store()
    retriever = HybridRetriever(vector_store, embedder, bm25, settings)
    indexer = Indexer(vector_store=vector_store, embedder=embedder, bm25=bm25, settings=settings)
    graph = build_graph(retriever, FakeLLM(["1", "Лавовый дракон: урон 40-63."]), settings)
    return create_mcp_server(
        indexer=indexer, retriever=retriever, graph_factory=lambda q: graph, settings=settings
    )


async def test_all_four_tools_end_to_end(mcp, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text(
        "Лавовый Дракон — элита Кальдеры: урон 40-63, скорость 9, иммунитет к огню.",
        encoding="utf-8",
    )
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert names == {"index_folder", "ask_question", "find_relevant_docs", "index_status"}
        assert all(t.description and len(t.description) > 40 for t in tools)

        # fastmcp 4.x: call_tool возвращает CallToolResult, текст — в .content[0].text
        status0 = await _call(client, "index_status", {})
        assert status0["files"] == 0
        assert status0["indexing_in_progress"] is False

        started = await _call(client, "index_folder", {"path": str(docs)})
        assert started["status"] in {"started", "already_running"}

        status1 = await _wait_until_indexed(client)
        assert status1["files"] == 1 and status1["chunks"] >= 1
        assert status1["last_report"]["files"] == 1
        assert status1["last_report"]["errors"] == []

        found = json.loads(
            (
                await client.call_tool(
                    "find_relevant_docs", {"query": "статы лавового дракона", "top_k": 3}
                )
            )
            .content[0]
            .text
        )
        assert len(found) >= 1
        assert any("a.md" in c["metadata"]["source"] for c in found)

        answer = await client.call_tool(
            "ask_question", {"question": "какие статы у лавового дракона?"}
        )
        text = answer.content[0].text
        assert "40-63" in text
        assert "a.md" in text  # источники приложены


async def test_find_relevant_docs_rejects_bad_top_k(mcp):
    async with Client(mcp) as client:
        for bad in (0, 51):
            with pytest.raises(ToolError):
                await client.call_tool("find_relevant_docs", {"query": "x", "top_k": bad})


async def test_ask_question_nothing_found(mcp, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    async with Client(mcp) as client:
        started = await _call(client, "index_folder", {"path": str(empty)})
        assert started["status"] in {"started", "already_running"}
        status = await _wait_until_indexed(client)
        assert status["files"] == 0
        result = await client.call_tool("ask_question", {"question": "что-нибудь"})
        assert "ничего не найдено" in result.content[0].text.lower()
