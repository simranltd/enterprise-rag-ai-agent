"""Inspect deterministic RAG context and prompts without generating answers."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from northstar.embeddings import NeuralEmbeddingProvider  # noqa: E402
from northstar.ingestion import chunk_document, load_markdown_document  # noqa: E402
from northstar.persistence import ChromaVectorStore  # noqa: E402
from northstar.rag import ContextBuilder, PromptBuilder  # noqa: E402
from northstar.retrieval import ChromaVectorSearch  # noqa: E402


QUESTIONS = {
    "A": "What approval is required for an expense around $5,000?",
    "B": "Am I allowed to work overseas?",
    "C": "What extra checks apply to a risky customer?",
    "D": "What is Northstar's maternity leave allowance?",
}


def main() -> None:
    provider = NeuralEmbeddingProvider()
    store = ChromaVectorStore()
    search = ChromaVectorSearch(store)
    context_builder = ContextBuilder(max_sources=3)
    prompt_builder = PromptBuilder()

    for label, question in QUESTIONS.items():
        results = search.search(provider.embed_text(question), top_k=3)
        context = context_builder.build(results)
        prompt = prompt_builder.build(question, context)
        print(f"\n{'=' * 80}\n{label}. QUESTION\n{question}")
        print("\nTOP RETRIEVED SOURCES")
        for source in context.sources:
            print(
                f"{source.citation_id} {source.document_title} / "
                f"{source.section_title} / similarity={source.similarity_score}"
            )
        print("\nFORMATTED CONTEXT\n" + context.formatted_context)
        print("\nFINAL DETERMINISTIC PROMPT\n" + prompt)


if __name__ == "__main__":
    main()
