"""Streamlit UI for the local Northstar RAG assistant."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from northstar.embeddings import NeuralEmbeddingProvider  # noqa: E402
from northstar.llm import HuggingFaceLocalProvider  # noqa: E402
from northstar.persistence import ChromaVectorStore  # noqa: E402
from northstar.rag import ContextBuilder, PromptBuilder, RAGOrchestrator  # noqa: E402
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


@st.cache_resource
def build_orchestrator() -> RAGOrchestrator:
    """Build the local RAG pipeline once per Streamlit process."""
    store = ChromaVectorStore()
    if store.count() == 0:
        raise RuntimeError(EMPTY_DATABASE_MESSAGE)

    embedding_provider = NeuralEmbeddingProvider()
    retriever = ChromaVectorSearch(store)
    llm_provider = HuggingFaceLocalProvider()
    return RAGOrchestrator(
        embedding_provider,
        retriever,
        ContextBuilder(max_sources=3),
        PromptBuilder(),
        llm_provider,
    )


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

            orchestrator = build_orchestrator()
            with st.spinner("Searching Northstar documents and generating answer..."):
                result = orchestrator.answer(question)
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
