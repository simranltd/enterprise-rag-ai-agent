from types import SimpleNamespace

from northstar.rag import ContextBuilder, KnowledgeResponseBuilder
from northstar.rag.models import ContextSource
from northstar.rag.prompt_builder import INSUFFICIENT_EVIDENCE_RESPONSE
from app import build_knowledge_response


class FakeEmbeddingProvider:
    def __init__(self):
        self.calls = []

    def embed_text(self, question):
        self.calls.append(question)
        return [0.1, 0.2]


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, embedding, top_k=3):
        self.calls.append((embedding, top_k))
        return self.results


def retrieval_result(section, text):
    return {
        "document_title": "Northstar Financial Services Remote Work Policy",
        "source_filename": "remote-work-policy.md",
        "document_version": "2.0",
        "section_title": section,
        "chunk_index": 0,
        "chunk_text": text,
        "similarity_score": 0.49,
    }


def test_knowledge_response_preserves_remote_work_evidence_and_citation() -> None:
    context = ContextBuilder().build(
        [
            retrieval_result(
                "2. Working outside Canada",
                "Employees may not work outside Canada without written approval.",
            )
        ]
    )

    result = KnowledgeResponseBuilder().build(
        "Can I work from another country?",
        context,
    )

    assert "Employees may not work outside Canada" in result.answer_text
    assert "[S1]" in result.answer_text
    assert result.sources[0].citation_id == "[S1]"
    assert result.provider == "deterministic-evidence"


def test_knowledge_route_does_not_invoke_qwen_or_any_llm() -> None:
    embedding = FakeEmbeddingProvider()
    retriever = FakeRetriever(
        [
            retrieval_result(
                "2. Working outside Canada",
                "Written approval is required before working outside Canada.",
            ),
            retrieval_result(
                "6. Superseded version",
                "This outdated rule must not be used.",
            ),
        ]
    )
    result = build_knowledge_response(
        "Can I work from another country?",
        embedding,
        retriever,
        ContextBuilder(),
        KnowledgeResponseBuilder(),
    )

    assert embedding.calls == ["Can I work from another country?"]
    assert retriever.calls == [([0.1, 0.2], 3)]
    assert result.sources[0].section_title == "2. Working outside Canada"
    assert "outdated rule" not in result.answer_text
    assert len(result.sources) == 1


def test_empty_or_textless_evidence_uses_standard_abstention() -> None:
    result = KnowledgeResponseBuilder().build(
        "Unknown question",
        SimpleNamespace(
            sources=[
                ContextSource(
                    citation_id="[S1]",
                    chunk_text="",
                )
            ]
        ),
    )

    assert result.answer_text == INSUFFICIENT_EVIDENCE_RESPONSE
    assert result.sources == []
