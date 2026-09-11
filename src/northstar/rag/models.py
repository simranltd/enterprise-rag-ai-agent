"""Data models shared by retrieval and future RAG generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ContextSource:
    """Retrieved evidence with citation metadata.

    ``similarity_score`` is a retrieval score, not a probability or confidence
    estimate.
    """

    citation_id: str
    document_title: Optional[str] = None
    source_filename: Optional[str] = None
    document_version: Optional[str] = None
    document_type: Optional[str] = None
    department: Optional[str] = None
    jurisdiction: Optional[str] = None
    classification: Optional[str] = None
    section_title: Optional[str] = None
    chunk_index: Optional[int] = None
    chunk_text: str = ""
    similarity_score: Optional[float] = None


@dataclass(frozen=True)
class RAGResult:
    """Structured output from the deterministic RAG orchestration layer."""

    question: str
    answer_text: str
    sources: List[ContextSource]
    provider: str
    model: str
