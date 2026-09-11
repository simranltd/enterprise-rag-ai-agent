"""Deterministic conversion of retrieval results into grounded context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping

from .models import ContextSource


@dataclass(frozen=True)
class ContextBuildResult:
    sources: List[ContextSource]
    formatted_context: str


class ContextBuilder:
    """Builds clearly delimited evidence without interpreting its content."""

    def __init__(self, max_sources: int = 3) -> None:
        if max_sources <= 0:
            raise ValueError("max_sources must be greater than zero")
        self.max_sources = max_sources

    def build(self, retrieval_results: Iterable[Any]) -> ContextBuildResult:
        sources = [
            self._source_from_result(result, index)
            for index, result in enumerate(
                list(retrieval_results)[: self.max_sources], start=1
            )
        ]
        blocks = [
            self._format_source(source)
            for source in sources
        ]
        return ContextBuildResult(sources=sources, formatted_context="\n\n".join(blocks))

    @staticmethod
    def _source_from_result(result: Any, index: int) -> ContextSource:
        metadata = _value(result, "metadata", {})
        chunk = _value(result, "chunk", None)
        if chunk is not None:
            chunk_metadata = _value(chunk, "metadata", {})
            metadata = {**chunk_metadata, **metadata}
            chunk_text = _value(chunk, "text", "")
        else:
            chunk_text = _value(result, "chunk_text", _value(result, "text", ""))
        similarity = _value(result, "similarity_score", None)
        if similarity is None:
            similarity = _value(result, "cosine_similarity", None)
        if similarity is None:
            similarity = _value(result, "score", None)
        if similarity is None:
            distance = _value(result, "distance", None)
            similarity = 1.0 - float(distance) if distance is not None else None

        return ContextSource(
            citation_id=f"[S{index}]",
            document_title=_text(_value(result, "document_title", metadata.get("document_title"))),
            source_filename=_text(_value(result, "source_filename", metadata.get("source_filename"))),
            document_version=_text(metadata.get("document_version")),
            document_type=_text(metadata.get("document_type")),
            department=_text(metadata.get("department")),
            jurisdiction=_text(metadata.get("jurisdiction")),
            classification=_text(metadata.get("classification")),
            section_title=_text(_value(result, "section_title", metadata.get("section_title"))),
            chunk_index=_optional_int(_value(result, "chunk_index", metadata.get("chunk_index"))),
            chunk_text=str(chunk_text),
            similarity_score=float(similarity) if similarity is not None else None,
        )

    @staticmethod
    def _format_source(source: ContextSource) -> str:
        lines = [source.citation_id]
        fields = (
            ("Document", source.document_title),
            ("Section", source.section_title),
            ("Version", source.document_version),
            ("Type", source.document_type),
            ("Department", source.department),
            ("Jurisdiction", source.jurisdiction),
            ("Classification", source.classification),
            ("Source", source.source_filename),
        )
        lines.extend(f"{label}: {value}" for label, value in fields if value)
        if source.chunk_index is not None:
            lines.append(f"Chunk: {source.chunk_index}")
        lines.extend(("--- BEGIN RETRIEVED EVIDENCE ---", source.chunk_text, "--- END RETRIEVED EVIDENCE ---"))
        return "\n".join(lines)


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _text(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None and value != "" else None
