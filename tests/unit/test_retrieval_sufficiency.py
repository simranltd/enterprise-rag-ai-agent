from types import SimpleNamespace

import pytest

from evaluation.retrieval_sufficiency_cases import SUPPORTED_CASES, UNSUPPORTED_CASES
from scripts.evaluate_retrieval_sufficiency import evaluate_case, similarity_at


def result(document: str, section: str, score: float) -> SimpleNamespace:
    return SimpleNamespace(
        document_title=document,
        section_title=section,
        cosine_similarity=score,
        rank=1,
    )


def test_similarity_at_returns_zero_for_missing_rank() -> None:
    assert similarity_at([], 0) == 0.0


def test_evaluate_case_preserves_scores_and_calculates_gap_and_source_hits() -> None:
    case = SUPPORTED_CASES[0]
    row = evaluate_case(
        case,
        [
            result(case.expected_document, case.expected_section, 0.82),
            result("Other Policy", "Other section", 0.61),
        ],
    )

    assert row.top1_similarity == 0.82
    assert row.top2_similarity == 0.61
    assert row.gap == pytest.approx(0.21)
    assert row.expected_source_top1 is True
    assert row.expected_source_top3 is True


def test_unsupported_cases_do_not_claim_expected_source_matches() -> None:
    row = evaluate_case(
        UNSUPPORTED_CASES[0],
        [result("Employee Handbook", "Leaving Northstar", 0.5)],
    )

    assert row.expected_source_top1 is False
    assert row.expected_source_top3 is False
