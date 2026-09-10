"""Load synthetic Northstar Markdown documents without external dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Dict, List


METADATA_FIELDS = (
    "title",
    "document type",
    "version",
    "effective date",
    "owner",
    "department",
    "jurisdiction",
    "classification",
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TABLE_ROW_RE = re.compile(r"^\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*$")


@dataclass(frozen=True)
class Section:
    """A Markdown section and its source location."""

    title: str
    level: int
    text: str
    line_start: int
    line_end: int


@dataclass(frozen=True)
class LoadedDocument:
    """A loaded document with normalized metadata and ordered sections."""

    title: str
    source_filename: str
    metadata: Dict[str, str] = field(default_factory=dict)
    sections: List[Section] = field(default_factory=list)


def load_markdown_document(path: str | Path) -> LoadedDocument:
    """Load a Markdown file, tolerating missing metadata and empty content."""

    document_path = Path(path)
    text = document_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    metadata = _extract_metadata(lines)
    headings = [
        (index + 1, match.group(1), match.group(2))
        for index, line in enumerate(lines)
        if (match := _HEADING_RE.match(line))
    ]

    title = metadata.get("title", "")
    if headings and headings[0][1] == "#":
        title = title or headings[0][2]
    title = title or document_path.stem.replace("-", " ").title()
    metadata.setdefault("title", title)

    sections: List[Section] = []
    section_headings = [heading for heading in headings if heading[1] != "#"]
    for position, (line_number, hashes, heading_title) in enumerate(section_headings):
        next_line = (
            section_headings[position + 1][0] - 1
            if position + 1 < len(section_headings)
            else len(lines)
        )
        body_lines = lines[line_number:next_line]
        body = _clean_section_body(body_lines)
        if body:
            sections.append(
                Section(
                    title=heading_title,
                    level=len(hashes),
                    text=body,
                    line_start=line_number,
                    line_end=next_line,
                )
            )

    return LoadedDocument(
        title=title,
        source_filename=document_path.name,
        metadata=metadata,
        sections=sections,
    )


def _extract_metadata(lines: List[str]) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for line in lines:
        match = _TABLE_ROW_RE.match(line)
        if not match:
            continue
        key = match.group(1).strip().lower()
        value = match.group(2).strip()
        if key in METADATA_FIELDS and value:
            metadata[key] = value
    return metadata


def _clean_section_body(lines: List[str]) -> str:
    cleaned: List[str] = []
    for line in lines:
        if _TABLE_ROW_RE.match(line) or re.match(r"^\|?\s*:?-{3,}", line):
            continue
        cleaned.append(line.rstrip())
    return "\n".join(cleaned).strip()
