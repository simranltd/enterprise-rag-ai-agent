"""Evaluate retrieval signals without applying a sufficiency threshold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from northstar.embeddings import NeuralEmbeddingProvider  # noqa: E402
from northstar.persistence import ChromaVectorStore  # noqa: E402
from northstar.retrieval import ChromaVectorSearch  # noqa: E402
from evaluation.retrieval_sufficiency_cases import (  # noqa: E402
    ALL_CASES,
    RetrievalSufficiencyCase,
)


@dataclass(frozen=True)
class EvaluationRow:
    case: RetrievalSufficiencyCase
    top1_similarity: float
    top2_similarity: float
    gap: float
    expected_source_top1: bool
    expected_source_top3: bool


def similarity_at(results: list, position: int) -> float:
    """Return a ranked similarity, or zero when that rank is unavailable."""
    if position >= len(results):
        return 0.0
    return float(results[position].cosine_similarity)


def expected_source_matches(case: RetrievalSufficiencyCase, result: object) -> bool:
    """Match the expected document requested by the evaluation protocol."""
    if not case.expected_answerable:
        return False
    return getattr(result, "document_title", None) == case.expected_document


def evaluate_case(
    case: RetrievalSufficiencyCase,
    results: list,
) -> EvaluationRow:
    top1_similarity = similarity_at(results, 0)
    top2_similarity = similarity_at(results, 1)
    return EvaluationRow(
        case=case,
        top1_similarity=top1_similarity,
        top2_similarity=top2_similarity,
        gap=top1_similarity - top2_similarity,
        expected_source_top1=bool(results and expected_source_matches(case, results[0])),
        expected_source_top3=any(expected_source_matches(case, result) for result in results[:3]),
    )


def print_case(row: EvaluationRow, results: list) -> None:
    case = row.case
    print(f"\nQUESTION: {case.question}")
    print(f"EXPECTED ANSWERABLE: {case.expected_answerable}")
    if case.expected_answerable:
        print(f"EXPECTED SOURCE: {case.expected_document} / {case.expected_section}")
    print("TOP 3 RESULTS")
    for result in results[:3]:
        print(
            f"  {result.rank}. {result.document_title} / {result.section_title} / "
            f"similarity={result.cosine_similarity:.4f}"
        )
    print(f"TOP1 SIMILARITY: {row.top1_similarity:.4f}")
    print(f"TOP2 SIMILARITY: {row.top2_similarity:.4f}")
    print(f"TOP1-TOP2 GAP: {row.gap:.4f}")
    print(f"EXPECTED DOCUMENT/SECTION TOP1: {row.expected_source_top1}")
    print(f"EXPECTED DOCUMENT/SECTION TOP3: {row.expected_source_top3}")


def print_summary(rows: Iterable[EvaluationRow]) -> None:
    rows = list(rows)
    supported = [row for row in rows if row.case.expected_answerable]
    top1_accuracy = sum(row.expected_source_top1 for row in supported) / len(supported)
    top3_recall = sum(row.expected_source_top3 for row in supported) / len(supported)

    print("\nSUMMARY")
    print("question | answerable | top1_similarity | top2_similarity | gap | expected_source_top1 | expected_source_top3")
    for row in rows:
        print(
            f"{row.case.question} | {row.case.expected_answerable} | "
            f"{row.top1_similarity:.4f} | {row.top2_similarity:.4f} | "
            f"{row.gap:.4f} | {row.expected_source_top1} | {row.expected_source_top3}"
        )
    print(f"\nSUPPORTED TOP-1 RETRIEVAL ACCURACY: {top1_accuracy:.2%}")
    print(f"SUPPORTED TOP-3 RETRIEVAL RECALL: {top3_recall:.2%}")


def main() -> None:
    provider = NeuralEmbeddingProvider()
    retriever = ChromaVectorSearch(ChromaVectorStore())
    rows = []
    for case in ALL_CASES:
        results = retriever.search(provider.embed_text(case.question), top_k=3)
        row = evaluate_case(case, results)
        rows.append(row)
        print_case(row, results)
    print_summary(rows)


if __name__ == "__main__":
    main()
