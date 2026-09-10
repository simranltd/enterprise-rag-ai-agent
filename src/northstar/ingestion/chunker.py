"""Structure-aware Markdown chunking for Phase 1A."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List

from .markdown_loader import LoadedDocument, Section


@dataclass(frozen=True)
class Chunk:
    """A searchable-ready text unit, without embeddings or persistence."""

    text: str
    metadata: Dict[str, object]


def chunk_document(
    document: LoadedDocument,
    target_chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> List[Chunk]:
    """Create ordered chunks by section and paragraph boundaries."""

    if target_chunk_size <= 0:
        raise ValueError("target_chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= target_chunk_size:
        raise ValueError("chunk_overlap must be smaller than target_chunk_size")

    chunks: List[Chunk] = []
    for section in document.sections:
        for text in _split_section(section, target_chunk_size, chunk_overlap):
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "document_title": document.title,
                    "source_filename": document.source_filename,
                    "document_version": document.metadata.get("version", ""),
                    "document_type": document.metadata.get("document type", ""),
                    "department": document.metadata.get("department", ""),
                    "jurisdiction": document.metadata.get("jurisdiction", ""),
                    "classification": document.metadata.get("classification", ""),
                    "section_title": section.title,
                    "section_level": section.level,
                    "source_line_start": section.line_start,
                    "source_line_end": section.line_end,
                    "chunk_index": len(chunks),
                }
            )
            chunks.append(Chunk(text=text, metadata=metadata))
    return chunks


def _split_section(
    section: Section,
    target_size: int,
    overlap: int,
) -> List[str]:
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", section.text)
        if paragraph.strip()
    ]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > target_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_paragraph(paragraph, target_size, overlap))
            continue

        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if current and len(candidate) > target_size:
            chunks.append(current)
            prefix = current[-overlap:] if overlap else ""
            current = _fit_with_prefix(prefix, paragraph, target_size)
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_long_paragraph(text: str, target_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if current and len(candidate) > target_size:
            chunks.append(current)
            prefix = current[-overlap:] if overlap else ""
            current = _fit_with_prefix(prefix, word, target_size)
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _fit_with_prefix(prefix: str, text: str, target_size: int) -> str:
    available = target_size - len(text) - 1
    if available <= 0:
        return text[:target_size]
    return f"{prefix[-available:]} {text}".strip()
