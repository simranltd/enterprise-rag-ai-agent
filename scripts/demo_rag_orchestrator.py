"""Demonstrate RAG orchestration with a deterministic fake LLM only."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from northstar.embeddings import NeuralEmbeddingProvider  # noqa: E402
from northstar.llm import FakeLLMProvider  # noqa: E402
from northstar.persistence import ChromaVectorStore  # noqa: E402
from northstar.rag import ContextBuilder, PromptBuilder, RAGOrchestrator  # noqa: E402
from northstar.retrieval import ChromaVectorSearch  # noqa: E402


QUESTIONS = [
    "What approval is required for an expense around $5,000?",
    "Am I allowed to work overseas?",
    "What extra checks apply to a risky customer?",
    "What is Northstar's maternity leave allowance?",
]


def main() -> None:
    provider = NeuralEmbeddingProvider()
    retriever = ChromaVectorSearch(ChromaVectorStore())
    fake_llm = FakeLLMProvider()
    orchestrator = RAGOrchestrator(
        provider,
        retriever,
        ContextBuilder(max_sources=3),
        PromptBuilder(),
        fake_llm,
    )

    for question in QUESTIONS:
        result = orchestrator.answer(question)
        print(f"\nQUESTION: {question}")
        print("RETRIEVED SOURCES (debug similarity scores)")
        for source in result.sources:
            print(
                f"  {source.citation_id} {source.document_title} / "
                f"{source.section_title} / score={source.similarity_score}"
            )
        print(f"FAKE/TEST ANSWER: {result.answer_text}")
        print(f"PROVIDER/MODEL: {result.provider}/{result.model}")

    empty = RAGOrchestrator(
        provider,
        _EmptyRetriever(),
        ContextBuilder(),
        PromptBuilder(),
        fake_llm,
    ).answer("Question with no retrieved evidence")
    print("\nEMPTY RETRIEVAL")
    print(f"ANSWER: {empty.answer_text}")
    print("LLM CALLED: no")


class _EmptyRetriever:
    def search(self, query_embedding: list[float], top_k: int = 3) -> list:
        return []


if __name__ == "__main__":
    main()
