from pathlib import Path

import pytest

from northstar.ingestion import chunk_document, load_markdown_document


def write_document(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "sample.md"
    path.write_text(content, encoding="utf-8")
    return path


def test_metadata_extraction_and_title(tmp_path: Path) -> None:
    document = load_markdown_document(
        write_document(
            tmp_path,
            "# Fallback title\n\n"
            "| Title | Example Policy |\n|---|---|\n"
            "| Version | 7.2 |\n|---|---|\n"
            "| Document type | Policy |\n|---|---|\n"
            "| Department | Finance |\n|---|---|\n"
            "## Approval\n\nSubmit the request.",
        )
    )

    assert document.title == "Example Policy"
    assert document.metadata["version"] == "7.2"
    assert document.metadata["department"] == "Finance"
    assert document.source_filename == "sample.md"


def test_section_preservation_and_location(tmp_path: Path) -> None:
    document = load_markdown_document(
        write_document(tmp_path, "# Title\n\n## First\n\nAlpha.\n\n## Second\n\nBeta.")
    )

    assert [section.title for section in document.sections] == ["First", "Second"]
    assert document.sections[0].line_start < document.sections[0].line_end
    assert document.sections[0].text == "Alpha."


def test_chunk_creation_metadata_and_ordering(tmp_path: Path) -> None:
    document = load_markdown_document(
        write_document(
            tmp_path,
            "# Policy\n\n| Version | 1.0 |\n|---|---|\n"
            "## One\n\nFirst paragraph.\n\n## Two\n\nSecond paragraph.",
        )
    )

    chunks = chunk_document(document, target_chunk_size=40, chunk_overlap=5)

    assert len(chunks) == 2
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1]
    assert [chunk.metadata["section_title"] for chunk in chunks] == ["One", "Two"]
    assert chunks[0].metadata["document_title"] == "Policy"
    assert chunks[0].metadata["source_filename"] == "sample.md"
    assert chunks[0].metadata["document_version"] == "1.0"


def test_configurable_chunk_size(tmp_path: Path) -> None:
    document = load_markdown_document(
        write_document(tmp_path, "# Policy\n\n## Details\n\n" + ("word " * 80))
    )

    small = chunk_document(document, target_chunk_size=50, chunk_overlap=0)
    large = chunk_document(document, target_chunk_size=200, chunk_overlap=0)

    assert len(small) > len(large)
    assert all(len(chunk.text) <= 50 for chunk in small)


def test_overlap_behavior(tmp_path: Path) -> None:
    document = load_markdown_document(
        write_document(
            tmp_path,
            "# Policy\n\n## Details\n\n"
            "Alpha paragraph with shared context.\n\n"
            "Beta paragraph starts the next chunk.",
        )
    )

    chunks = chunk_document(document, target_chunk_size=40, chunk_overlap=10)

    assert len(chunks) == 2
    assert chunks[0].text[-10:] in chunks[1].text


@pytest.mark.parametrize(
    ("content", "expected_title"),
    [("", "Sample"), ("# Only a title\n", "Only a title"), ("not markdown metadata", "Sample")],
)
def test_empty_or_malformed_documents_are_safe(
    tmp_path: Path, content: str, expected_title: str
) -> None:
    document = load_markdown_document(write_document(tmp_path, content))

    assert document.title == expected_title
    assert document.sections == []
    assert chunk_document(document) == []
