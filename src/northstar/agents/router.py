"""Small deterministic router for Agent V2."""

from __future__ import annotations

from enum import Enum
import re


class AgentRoute(str, Enum):
    GREETING = "GREETING"
    KNOWLEDGE = "KNOWLEDGE"
    TOOL = "TOOL"


_GREETING_REQUESTS = {"hi", "hello", "hey", "thanks", "thank you"}
_EXPENSE_TERMS = ("expense", "purchase", "spend", "approval")
_AMOUNT_RE = re.compile(r"\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)")


def extract_amount(question: str) -> float | None:
    match = _AMOUNT_RE.search(question)
    return float(match.group(1).replace(",", "")) if match else None


def classify_request(question: str) -> AgentRoute:
    normalized = question.strip().lower()
    if normalized in _GREETING_REQUESTS:
        return AgentRoute.GREETING
    if any(term in normalized for term in _EXPENSE_TERMS):
        if extract_amount(normalized) is not None:
            return AgentRoute.TOOL
    return AgentRoute.KNOWLEDGE
