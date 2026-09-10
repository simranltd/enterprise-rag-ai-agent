"""Print Phase 1A chunks for manual inspection."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from northstar.ingestion import chunk_document, load_markdown_document  # noqa: E402


def main() -> None:
    documents_path = ROOT / "data" / "sample_documents"
    for path in sorted(documents_path.glob("*.md")):
        document = load_markdown_document(path)
        chunks = chunk_document(document)
        print(f"\n{document.source_filename}: {len(chunks)} chunks")
        for chunk in chunks:
            text = " ".join(chunk.text.split())
            approximation = len(text.split())
            print(
                f"  [{chunk.metadata['chunk_index']}] "
                f"section={chunk.metadata['section_title']!r} "
                f"chars={len(chunk.text)} tokens~{approximation}"
            )
            print(f"      {text[:150]}")


if __name__ == "__main__":
    main()
