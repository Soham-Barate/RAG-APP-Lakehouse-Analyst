"""
LangGraph StateGraph — the agentic RAG pipeline.

Wires all nodes into a cyclic, self-correcting graph with conditional
edges for retry logic, route-based branching, and optional deep-reasoning
nodes (chain-of-thought, answer verification, hallucination detection).
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from backend.rag.nodes.compressor import ContextualCompressor
from backend.rag.nodes.generator import ResponseGenerator
from backend.rag.nodes.grader import RelevanceGrader
from backend.rag.nodes.guardrails import GuardrailsNode
from backend.rag.nodes.hallucination_detector import HallucinationDetector
from backend.rag.nodes.hyde import HyDEGenerator
from backend.rag.nodes.metadata_extractor import MetadataExtractor
from backend.rag.nodes.query_rewriter import QueryRewriter
from backend.rag.nodes.reranker import CrossEncoderReranker
from backend.rag.nodes.retriever import HybridRetriever
from backend.rag.nodes.router import QueryRouter
from backend.rag.nodes.thinker import ChainOfThoughtReasoner
from backend.rag.nodes.verifier import AnswerVerifier
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

    Graph structure (full mode):

        START
          │
          ▼
        router ──→ metadata_extractor ──→ query_rewriter ──→ hyde ──→ retriever
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
                                                          thinker                  │
                                                              │                    ▼
                                                              ▼              query_rewriter
                                                          generator            (loop)
                                                              │
                                                              ▼
                                                          verifier
                                                              │
                                                              ▼
                                                     hallucination_detector
                                                              │
                                                              ▼
                                                          guardrails
                                                              │
                                                              ▼
                                                             END

    In fast_mode, thinker + verifier + hallucination_detector are skipped.
    """

    # ----- Instantiate nodes -----
    router = QueryRouter(llm_service=llm_service)
    metadata_extractor = MetadataExtractor(llm_service=llm_service)
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
    thinker = ChainOfThoughtReasoner(llm_service=llm_service)
    generator = ResponseGenerator(
        llm_service=llm_service,
        confidence_floor=settings.rag.confidence_floor,
    )
    verifier = AnswerVerifier(llm_service=llm_service)
    hallucination_detector = HallucinationDetector(llm_service=llm_service)
    guardrails = GuardrailsNode(
        llm_service=llm_service,
        confidence_floor=settings.rag.confidence_floor,
    )
    from backend.rag.nodes.sql_node import SafeSQLNode
    sql_node = SafeSQLNode(llm_service=llm_service)

    # ----- Build the graph -----
    graph = StateGraph(RAGState)

    # Add all nodes
    graph.add_node("router", router)
    graph.add_node("sql_node", sql_node)
    graph.add_node("metadata_extractor", metadata_extractor)
    graph.add_node("query_rewriter", query_rewriter)
    graph.add_node("hyde", hyde_generator)
    graph.add_node("retriever", retriever)
    graph.add_node("reranker", reranker)
    graph.add_node("compressor", compressor)
    graph.add_node("grader", grader)
    graph.add_node("thinker", thinker)
    graph.add_node("generator", generator)
    graph.add_node("verifier", verifier)
    graph.add_node("hallucination_detector", hallucination_detector)
    graph.add_node("guardrails", guardrails)

    # ----- Define edges -----

    # Entry point is router, which conditionally routes to SQL, Hybrid, or Vector path
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _route_decision,
        {
            "sql": "sql_node",
            "hybrid": "sql_node",
            "vector": "metadata_extractor",
        },
    )

    # SQL node: for 'sql' route, proceeds to generator/thinker; for 'hybrid' route, continues to vector pipeline
    graph.add_conditional_edges(
        "sql_node",
        _sql_post_decision,
        {
            "think": "thinker",
            "generate": "generator",
            "continue_to_vector": "metadata_extractor",
        },
    )

    # Vector pipeline: metadata_extractor → query_rewriter → hyde → retriever
    graph.add_edge("metadata_extractor", "query_rewriter")
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
            "think": "thinker",
            "generate": "generator",
            "retry": "query_rewriter",
        },
    )

    # Thinker → generator (deep reasoning path)
    graph.add_edge("thinker", "generator")

    # Generator → conditional: fast_mode skips verification
    graph.add_conditional_edges(
        "generator",
        _post_generation_decision,
        {
            "verify": "verifier",
            "skip_to_guardrails": "guardrails",
        },
    )

    # Verifier → hallucination_detector → guardrails → END
    graph.add_edge("verifier", "hallucination_detector")
    graph.add_edge("hallucination_detector", "guardrails")
    graph.add_edge("guardrails", END)

    # ----- Compile -----
    compiled = graph.compile()
    logger.info("RAG pipeline compiled successfully (with Text-to-SQL + CoT + verification)")

    return compiled


def _route_decision(state: RAGState) -> str:
    """Conditional edge from router: branch to sql_node (for sql and hybrid) or vector retrieval."""
    route = state.get("route", "hybrid")
    if route == "sql":
        return "sql"
    elif route == "hybrid":
        return "hybrid"
    return "vector"


def _sql_post_decision(state: RAGState) -> str:
    """
    Conditional edge from sql_node.

    - For an explicit SQL route ('sql'), NEVER fall back to vector retrieval,
      regardless of whether results were found, zero rows returned, or an error occurred.
      Proceed directly to generation (think or generate) with transparent lakehouse context.
    - For 'hybrid' route, proceed to the vector pipeline ('continue_to_vector') so
      unstructured qualitative evidence is retrieved and merged with the structured results.
    """
    if state.get("route", "sql") == "sql":
        return "generate" if state.get("fast_mode", False) else "think"

    return "continue_to_vector"


def _grader_decision(state: RAGState) -> str:
    """
    Conditional edge function for the grader node.

    If documents are relevant enough:
      - fast_mode → go directly to generator
      - normal mode → think first (chain-of-thought)
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

    # Skip chain-of-thought in fast mode
    if state.get("fast_mode", False):
        return "generate"

    return "think"


def _post_generation_decision(state: RAGState) -> str:
    """
    Conditional edge after generation.

    In fast_mode, skip verification and go straight to guardrails.
    In normal mode, run the full verification pipeline.
    """
    if state.get("fast_mode", False):
        return "skip_to_guardrails"
    return "verify"
