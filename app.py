"""Streamlit UI for the local Northstar RAG assistant."""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from northstar.embeddings import NeuralEmbeddingProvider  # noqa: E402
from northstar.agents import (  # noqa: E402
    AgentRoute,
    calculate_expense_approval,
    classify_request,
    extract_amount,
)
from northstar.persistence import ChromaVectorStore  # noqa: E402
from northstar.rag import (  # noqa: E402
    ContextBuilder,
    KnowledgeResponseBuilder,
    filter_current_policy_results,
)
from northstar.rag.models import ContextSource  # noqa: E402
from northstar.retrieval import ChromaVectorSearch  # noqa: E402


EMPTY_DATABASE_MESSAGE = (
    "The Northstar Chroma database is empty. Run "
    "`python scripts\\ingest_chunks_to_chroma.py` before starting the chatbot."
)
FAST_RESPONSES = {
    "hi": "Hello. How can I help with Northstar policies and procedures?",
    "hello": "Hello. How can I help with Northstar policies and procedures?",
    "hey": "Hello. How can I help with Northstar policies and procedures?",
    "thanks": "You're welcome.",
    "thank you": "You're welcome.",
}


def format_source_metadata(source: ContextSource) -> str:
    """Format source metadata without exposing chunk text or embeddings."""
    title = source.document_title or source.source_filename or "Unknown document"
    section = source.section_title or "Unknown section"
    details = [f"{source.citation_id} {title}", f"Section: {section}"]
    if source.document_version:
        details.append(f"Version: {source.document_version}")
    if source.source_filename:
        details.append(f"File: {source.source_filename}")
    return " | ".join(details)


def fast_response(question: str) -> str | None:
    """Return an immediate response for simple conversational messages."""
    return FAST_RESPONSES.get(question.strip().lower())


def render_tool_result(question: str) -> tuple[str, str] | None:
    amount = extract_amount(question)
    if classify_request(question) != AgentRoute.TOOL or amount is None:
        return None
    result = calculate_expense_approval(amount)
    source = (
        f"[S1] {result.document_title} | Section: {result.section_title} | "
        f"Version: {result.document_version}"
    )
    return result.answer_text, source


def warm_embedding_provider(embedding_provider: Any) -> Any:
    """Load and warm the embedding model once during cached initialization."""
    started = perf_counter()
    embedding_provider.embed_text("Northstar embedding warmup")
    print(
        "[Northstar timing] MiniLM initialization/warmup: "
        f"{perf_counter() - started:.3f}s"
    )
    return embedding_provider


@st.cache_resource
def build_knowledge_components() -> tuple[
    NeuralEmbeddingProvider,
    ChromaVectorSearch,
    ContextBuilder,
    KnowledgeResponseBuilder,
]:
    """Build the fast local evidence pipeline once per Streamlit process."""
    store = ChromaVectorStore()
    if store.count() == 0:
        raise RuntimeError(EMPTY_DATABASE_MESSAGE)

    embedding_provider = NeuralEmbeddingProvider()
    warm_embedding_provider(embedding_provider)
    retriever = ChromaVectorSearch(store)
    return (
        embedding_provider,
        retriever,
        ContextBuilder(max_sources=3),
        KnowledgeResponseBuilder(),
    )


def build_knowledge_response(
    question: str,
    embedding_provider: Any,
    retriever: Any,
    context_builder: ContextBuilder,
    response_builder: KnowledgeResponseBuilder,
) -> Any:
    """Run retrieval and deterministic evidence rendering without an LLM."""
    embedding_started = perf_counter()
    query_embedding = embedding_provider.embed_text(question)
    print(
        "[Northstar timing] question embedding: "
        f"{perf_counter() - embedding_started:.3f}s"
    )

    retrieval_started = perf_counter()
    retrieval_results = filter_current_policy_results(
        retriever.search(query_embedding, top_k=3)
    )
    print(
        "[Northstar timing] Chroma retrieval: "
        f"{perf_counter() - retrieval_started:.3f}s"
    )
    context = context_builder.build(retrieval_results)
    return response_builder.build(question, context)


def initialize_chat_history() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            result = message.get("result")
            if result is not None and result.sources:
                st.markdown("**Retrieved sources**")
                for source in result.sources:
                    st.caption(format_source_metadata(source))


def main() -> None:
    st.set_page_config(page_title="Northstar AI Assistant", page_icon="N")
    st.title("Northstar AI Assistant")
    st.subheader("Enterprise Policy & Procedure Assistant")
    st.write(
        "Ask questions about Northstar policies, procedures, cybersecurity, "
        "expenses, remote work, AML, IT access, and employee guidance."
    )
    with st.sidebar:
        st.subheader("Northstar Agent V2")
        st.caption("Agent mode: Fast Local RAG")
        st.caption("Generation: Deterministic evidence synthesis")
        st.caption("Optional LLM: Qwen3-0.6B local")
        st.caption("Embeddings: MiniLM")
        st.caption("Vector DB: ChromaDB")
        st.caption("Cost: Local / $0")
    initialize_chat_history()

    for message in st.session_state.messages:
        render_message(message)

    question = st.chat_input("Ask a question about Northstar documents")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            quick_answer = fast_response(question)
            if quick_answer is not None:
                st.markdown(quick_answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": quick_answer}
                )
                return

            tool_result = render_tool_result(question)
            if tool_result is not None:
                answer_text, source = tool_result
                st.markdown("**Tool used: Expense Approval Checker**")
                st.markdown(answer_text)
                st.caption(source)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"**Tool used: Expense Approval Checker**\n\n{answer_text}",
                    }
                )
                return

            embedding_provider, retriever, context_builder, response_builder = (
                build_knowledge_components()
            )
            with st.spinner("Searching Northstar documents and generating answer..."):
                result = build_knowledge_response(
                    question,
                    embedding_provider,
                    retriever,
                    context_builder,
                    response_builder,
                )
            st.markdown(result.answer_text)
            if result.sources:
                st.markdown("**Retrieved sources**")
                for source in result.sources:
                    st.caption(format_source_metadata(source))
            st.session_state.messages.append(
                {"role": "assistant", "content": result.answer_text, "result": result}
            )
        except (RuntimeError, ValueError, OSError) as error:
            st.error(str(error))


if __name__ == "__main__":
    main()
