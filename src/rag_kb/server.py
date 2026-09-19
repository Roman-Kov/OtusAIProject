# src/rag_kb/server.py
"""Входная точка: сборка реальных зависимостей и запуск MCP-сервера."""

from rag_kb.app import create_mcp_server
from rag_kb.config import get_settings
from rag_kb.graph.builder import build_graph
from rag_kb.indexing.embedder import create_embedder
from rag_kb.indexing.indexer import Indexer
from rag_kb.llm import OllamaLLM
from rag_kb.retrieval.bm25_store import BM25Store
from rag_kb.retrieval.hybrid import HybridRetriever
from rag_kb.retrieval.stores import VectorStore


def main() -> None:
    settings = get_settings()
    vector_store = VectorStore(settings)
    embedder = create_embedder(settings)
    bm25 = BM25Store()
    retriever = HybridRetriever(vector_store, embedder, bm25, settings)
    retriever.rebuild_bm25()  # восстановить BM25 из персистентного Chroma после рестарта
    indexer = Indexer(vector_store=vector_store, embedder=embedder, bm25=bm25, settings=settings)
    llm = OllamaLLM(
        settings.ollama_base_url, settings.llm_model, keep_alive=settings.ollama_keep_alive_sec
    )
    graph = build_graph(retriever, llm, settings)
    mcp = create_mcp_server(
        indexer=indexer,
        retriever=retriever,
        graph_factory=lambda _q: graph,
        settings=settings,
    )
    mcp.run(transport="streamable-http", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
