"""Run the fully local Qwen RAG demonstration."""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from northstar.embeddings import NeuralEmbeddingProvider  # noqa: E402
from northstar.llm import HuggingFaceLocalProvider  # noqa: E402
from northstar.persistence import ChromaVectorStore  # noqa: E402
from northstar.rag import ContextBuilder, PromptBuilder, RAGOrchestrator  # noqa: E402
from northstar.retrieval import ChromaVectorSearch  # noqa: E402


QUESTIONS = [
    "What approval is required for an expense above $5,000?",
    "Am I allowed to work overseas?",
    "What extra checks apply to a high-risk customer?",
    "What is Northstar's maternity leave allowance?",
]


def main() -> None:
    orchestrator = RAGOrchestrator(
        NeuralEmbeddingProvider(),
        ChromaVectorSearch(ChromaVectorStore()),
        ContextBuilder(max_sources=3),
        PromptBuilder(),
        HuggingFaceLocalProvider(),
    )
    for question in QUESTIONS:
        started = perf_counter()
        result = orchestrator.answer(question)
        elapsed = perf_counter() - started
        print(f"\nQUESTION: {question}")
        print(f"ANSWER: {result.answer_text}")
        print(f"PROVIDER/MODEL: {result.provider}/{result.model}")
        print(f"GENERATION/PIPELINE TIME: {elapsed:.2f}s")
        print("RETRIEVED SOURCES (debug similarity scores)")
        for source in result.sources:
            print(
                f"  {source.citation_id} {source.document_title} / "
                f"{source.section_title} / score={source.similarity_score}"
            )


if __name__ == "__main__":
    main()
