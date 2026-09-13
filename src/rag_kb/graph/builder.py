# src/rag_kb/graph/builder.py
from langgraph.graph import END, START, StateGraph

from rag_kb.config import Settings
from rag_kb.graph.nodes import make_generator, make_grader, make_rewriter, retrieve_node
from rag_kb.graph.state import GraphState


def _route_after_grade(state: GraphState, settings: Settings) -> str:
    if len(state["relevant"]) >= settings.min_relevant_chunks:
        return "generate"
    if state["attempt"] <= settings.max_retries:
        return "rewrite"
    return "generate"


def build_graph(retriever, llm, settings: Settings):
    graph = StateGraph(GraphState)
    graph.add_node("rewrite", make_rewriter(llm))
    graph.add_node("retrieve", retrieve_node(retriever))
    graph.add_node("grade", make_grader(llm))
    graph.add_node("generate", make_generator(llm))
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        lambda state: _route_after_grade(state, settings),
        {"rewrite": "rewrite", "generate": "generate"},
    )
    graph.add_edge("generate", END)
    return graph.compile()
