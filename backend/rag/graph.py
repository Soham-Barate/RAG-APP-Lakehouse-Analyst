"""
LangGraph StateGraph — the agentic RAG pipeline.

Wires all nodes into a cyclic, self-correcting graph with conditional
edges for retry logic and route-based branching.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from backend.rag.nodes.compressor import ContextualCompressor
from backend.rag.nodes.generator import ResponseGenerator
from backend.rag.nodes.grader import RelevanceGrader
from backend.rag.nodes.guardrails import GuardrailsNode
from backend.rag.nodes.hyde import HyDEGenerator
from backend.rag.nodes.query_rewriter import QueryRewriter
from backend.rag.nodes.reranker import CrossEncoderReranker
from backend.rag.nodes.retriever import HybridRetriever
from backend.rag.nodes.router import QueryRouter
from backend.rag.state import RAGState
from config.settings import Settings

logger = logging.getLogger(__name__)


def build_rag_graph(
    vector_store: Any,
    llm_service: Any,
    embedding_service: Any,
    settings: Settings,
) -> Any:
    """
    Build and compile the LangGraph RAG pipeline.

    Graph structure:

        START
          │
          ▼
        router ──→ query_rewriter ──→ hyde ──→ retriever
                                                  │
                                                  ▼
                                              reranker
                                                  │
                                                  ▼
                                              compressor
                                                  │
                                                  ▼
                                               grader
                                              ╱       ╲
                                    [relevant]         [irrelevant
                                        │               + iter < max]
                                        ▼                    │
                                    generator                │
                                        │                    ▼
                                        ▼              query_rewriter
                                    guardrails            (loop)
                                        │
                                        ▼
                                       END
    """

    # ----- Instantiate nodes -----
    router = QueryRouter(llm_service=llm_service)
    query_rewriter = QueryRewriter(llm_service=llm_service)
    hyde_generator = HyDEGenerator(llm_service=llm_service)
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_service=embedding_service,
        top_k=settings.rag.retrieval_top_k,
    )
    reranker = CrossEncoderReranker(
        cohere_api_key=settings.cohere.api_key.get_secret_value(),
        rerank_model=settings.cohere.rerank_model,
        top_n=settings.rag.rerank_top_n,
    )
    compressor = ContextualCompressor(
        llm_service=llm_service,
        max_context_tokens=4000,
    )
    grader = RelevanceGrader(
        llm_service=llm_service,
        relevance_threshold=settings.rag.relevance_threshold,
        max_iterations=settings.rag.max_retrieval_iterations,
    )
    generator = ResponseGenerator(
        llm_service=llm_service,
        confidence_floor=settings.rag.confidence_floor,
    )
    guardrails = GuardrailsNode(
        llm_service=llm_service,
        confidence_floor=settings.rag.confidence_floor,
    )

    # ----- Build the graph -----
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("router", router)
    graph.add_node("query_rewriter", query_rewriter)
    graph.add_node("hyde", hyde_generator)
    graph.add_node("retriever", retriever)
    graph.add_node("reranker", reranker)
    graph.add_node("compressor", compressor)
    graph.add_node("grader", grader)
    graph.add_node("generator", generator)
    graph.add_node("guardrails", guardrails)

    # ----- Define edges -----

    # Linear flow: START → router → query_rewriter → hyde → retriever
    graph.set_entry_point("router")
    graph.add_edge("router", "query_rewriter")
    graph.add_edge("query_rewriter", "hyde")
    graph.add_edge("hyde", "retriever")

    # Retriever → reranker → compressor → grader
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "compressor")
    graph.add_edge("compressor", "grader")

    # Conditional edge: grader decides to generate or retry
    graph.add_conditional_edges(
        "grader",
        _grader_decision,
        {
            "generate": "generator",
            "retry": "query_rewriter",
        },
    )

    # Generator → guardrails → END
    graph.add_edge("generator", "guardrails")
    graph.add_edge("guardrails", END)

    # ----- Compile -----
    compiled = graph.compile()
    logger.info("RAG pipeline compiled successfully")

    return compiled


def _grader_decision(state: RAGState) -> str:
    """
    Conditional edge function for the grader node.

    If documents are relevant enough → proceed to generation.
    If not and retries remain → loop back to query_rewriter.
    If max retries reached → force generation with what we have.
    """
    should_retry = state.get("should_retry", False)

    if should_retry:
        iteration = state.get("iteration_count", 1)
        logger.info(
            "Grader: retrieval insufficient, retrying (iteration %d)",
            iteration,
        )
        return "retry"

    return "generate"
