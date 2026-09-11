"""Structured retrieval-sufficiency evaluation cases."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalSufficiencyCase:
    question: str
    expected_answerable: bool
    expected_document: str | None = None
    expected_section: str | None = None


SUPPORTED_CASES = (
    RetrievalSufficiencyCase(
        "Can I work from another country?",
        True,
        "Northstar Financial Services Remote Work Policy",
        "Working outside Canada",
    ),
    RetrievalSufficiencyCase(
        "Am I allowed to work overseas?",
        True,
        "Northstar Financial Services Remote Work Policy",
        "Working outside Canada",
    ),
    RetrievalSufficiencyCase(
        "What approval is required for an expense above $5,000?",
        True,
        "Northstar Financial Services Expense Policy",
        "Approval thresholds",
    ),
    RetrievalSufficiencyCase(
        "Who authorizes a large purchase?",
        True,
        "Northstar Financial Services Expense Policy",
        "Approval thresholds",
    ),
    RetrievalSufficiencyCase(
        "Is MFA required?",
        True,
        "Northstar Financial Services Cybersecurity Policy",
        "Account protection",
    ),
    RetrievalSufficiencyCase(
        "Do employees need a second authentication factor?",
        True,
        "Northstar Financial Services Cybersecurity Policy",
        "Account protection",
    ),
    RetrievalSufficiencyCase(
        "What checks are required for a high-risk client?",
        True,
        "Northstar Financial Services Anti-Money Laundering Procedure",
        "High-risk client checks",
    ),
    RetrievalSufficiencyCase(
        "What extra checks apply to a risky customer?",
        True,
        "Northstar Financial Services Anti-Money Laundering Procedure",
        "High-risk client checks",
    ),
)

UNSUPPORTED_CASES = (
    RetrievalSufficiencyCase("What is Northstar's maternity leave allowance?", False),
    RetrievalSufficiencyCase("Does Northstar provide dental insurance?", False),
    RetrievalSufficiencyCase(
        "How many vacation days do employees receive after 10 years?", False
    ),
    RetrievalSufficiencyCase(
        "What is the company pension matching percentage?", False
    ),
    RetrievalSufficiencyCase("Does Northstar reimburse gym memberships?", False),
    RetrievalSufficiencyCase("What is the dress code for client meetings?", False),
)

ALL_CASES = SUPPORTED_CASES + UNSUPPORTED_CASES
