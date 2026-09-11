from northstar.llm import FakeLLMProvider
from northstar.rag import ContextBuilder, PromptBuilder, RAGOrchestrator
from northstar.rag.prompt_builder import INSUFFICIENT_EVIDENCE_RESPONSE


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.questions = []

    def embed_text(self, text: str) -> list[float]:
        self.questions.append(text)
        return [0.1, 0.2]


class FakeRetriever:
    def __init__(self, results: list) -> None:
        self.results = results
        self.received_embedding = None
        self.received_top_k = None

    def search(self, query_embedding: list[float], top_k: int = 3) -> list:
        self.received_embedding = query_embedding
        self.received_top_k = top_k
        return self.results


def retrieval_result() -> dict:
    return {
        "document_title": "Expense Policy",
        "source_filename": "expense-policy.md",
        "section_title": "Approval thresholds",
        "chunk_index": 2,
        "chunk_text": "Department head and Finance Operations approval.",
        "similarity_score": 0.91,
    }


def test_orchestrator_embeds_retrieves_builds_prompt_and_returns_result() -> None:
    embedding = FakeEmbeddingProvider()
    retriever = FakeRetriever([retrieval_result()])
    llm = FakeLLMProvider("deterministic answer")
    result = RAGOrchestrator(
        embedding,
        retriever,
        ContextBuilder(),
        PromptBuilder(),
        llm,
        top_k=2,
    ).answer("What approval is needed?")

    assert embedding.questions == ["What approval is needed?"]
    assert retriever.received_embedding == [0.1, 0.2]
    assert retriever.received_top_k == 2
    assert result.answer_text == "deterministic answer"
    assert result.sources[0].citation_id == "[S1]"
    assert result.sources[0].similarity_score == 0.91
    assert result.provider == "fake"
    assert result.model == "fake-deterministic-v1"
    assert llm.last_prompt is not None
    expected_prompt = PromptBuilder().build(
        "What approval is needed?",
        ContextBuilder().build([retrieval_result()]),
    )
    assert llm.last_prompt == expected_prompt


def test_empty_retrieval_abstains_without_calling_llm() -> None:
    embedding = FakeEmbeddingProvider()
    llm = FakeLLMProvider()
    result = RAGOrchestrator(
        embedding,
        FakeRetriever([]),
        ContextBuilder(),
        PromptBuilder(),
        llm,
    ).answer("Unsupported question")

    assert result.answer_text == INSUFFICIENT_EVIDENCE_RESPONSE
    assert result.sources == []
    assert result.provider == "none"
    assert result.model == "none"
    assert llm.last_prompt is None


def test_retrieval_order_and_similarity_are_preserved() -> None:
    first = retrieval_result()
    second = {**retrieval_result(), "section_title": "Exceptions", "similarity_score": 0.4}
    llm = FakeLLMProvider()
    result = RAGOrchestrator(
        FakeEmbeddingProvider(),
        FakeRetriever([first, second]),
        ContextBuilder(),
        PromptBuilder(),
        llm,
    ).answer("Question")

    assert [source.citation_id for source in result.sources] == ["[S1]", "[S2]"]
    assert [source.section_title for source in result.sources] == [
        "Approval thresholds",
        "Exceptions",
    ]
    assert [source.similarity_score for source in result.sources] == [0.91, 0.4]
