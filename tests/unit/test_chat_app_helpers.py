from northstar.rag.models import ContextSource

from app import (
    fast_response,
    format_source_metadata,
    render_tool_result,
    warm_embedding_provider,
)


def test_format_source_metadata_includes_citation_and_available_fields() -> None:
    source = ContextSource(
        citation_id="[S1]",
        document_title="Expense Policy",
        source_filename="expense-policy.md",
        document_version="3.0",
        section_title="Approval thresholds",
    )

    formatted = format_source_metadata(source)

    assert "[S1] Expense Policy" in formatted
    assert "Section: Approval thresholds" in formatted
    assert "Version: 3.0" in formatted
    assert "File: expense-policy.md" in formatted


def test_format_source_metadata_handles_missing_optional_values() -> None:
    formatted = format_source_metadata(ContextSource(citation_id="[S2]"))

    assert formatted == "[S2] Unknown document | Section: Unknown section"


def test_fast_response_handles_greetings_without_rag() -> None:
    assert fast_response("  Hello ") == (
        "Hello. How can I help with Northstar policies and procedures?"
    )
    assert fast_response("What is the expense policy?") is None


def test_expense_tool_fast_path_returns_policy_source() -> None:
    answer = render_tool_result("What approval do I need for a $7,000 expense?")
    assert answer is not None
    assert "department head and Finance Operations" in answer[0]
    assert "Expense Policy" in answer[1]


def test_warm_embedding_provider_warms_once_and_reuses_provider(capsys) -> None:
    class FakeEmbeddingProvider:
        def __init__(self):
            self.calls = []

        def embed_text(self, text):
            self.calls.append(text)
            return [0.1, 0.2]

    provider = FakeEmbeddingProvider()

    assert warm_embedding_provider(provider) is provider
    assert provider.calls == ["Northstar embedding warmup"]
    assert "MiniLM initialization/warmup" in capsys.readouterr().out
