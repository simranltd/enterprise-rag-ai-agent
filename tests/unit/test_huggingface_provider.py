import pytest
import sys
import types

from northstar.llm import HuggingFaceLocalProvider


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values[0]))
        self.devices = []

    def __getitem__(self, item):
        if isinstance(item, tuple):
            _, column_slice = item
            return FakeTensor([self.values[0][column_slice]])
        return self.values[item]

    def to(self, device):
        self.devices.append(device)
        return self


class FakeTokenizer:
    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return {"input_ids": FakeTensor([[10, 11, 12]])}

    def batch_decode(self, generated_ids, **kwargs):
        assert kwargs == {"skip_special_tokens": True}
        return ["Grounded local answer [S1]."]


class FakeModel:
    def __init__(self):
        self.generate_calls = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return FakeTensor([[10, 11, 12, 13, 14]])


class FakeTorch:
    float32 = "float32"
    float16 = "float16"

    @staticmethod
    def set_num_threads(count):
        return None

    class cuda:
        @staticmethod
        def is_available():
            return False


def test_cpu_model_loading_requests_float32(monkeypatch) -> None:
    calls = []
    tokenizer = FakeTokenizer()
    model = FakeModel()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name):
            return tokenizer

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls.append((model_name, kwargs))
            return model

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=FakeAutoTokenizer,
            AutoModelForCausalLM=FakeAutoModel,
        ),
    )

    provider = HuggingFaceLocalProvider(device="cpu")
    provider.loaded_model

    assert calls == [("Qwen/Qwen3-1.7B", {"torch_dtype": "float32"})]


def test_local_provider_uses_qwen_template_and_deterministic_generation() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    provider = HuggingFaceLocalProvider(
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=128,
        device="cpu",
    )

    response = provider.generate("Answer using the evidence.")

    messages, template_kwargs = tokenizer.template_calls[0]
    assert messages == [{"role": "user", "content": "Answer using the evidence."}]
    assert template_kwargs["enable_thinking"] is False
    assert template_kwargs["add_generation_prompt"] is True
    assert model.generate_calls[0]["max_new_tokens"] == 128
    assert model.generate_calls[0]["do_sample"] is False
    assert response.text == "Grounded local answer [S1]."
    assert response.provider == "huggingface-local"
    assert response.model == "Qwen/Qwen3-1.7B"


def test_model_and_tokenizer_are_reused() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    provider = HuggingFaceLocalProvider(tokenizer=tokenizer, model=model)

    assert provider.tokenizer is tokenizer
    assert provider.loaded_model is model
    assert provider.tokenizer is tokenizer
    assert provider.loaded_model is model


def test_provider_requires_both_injected_dependencies() -> None:
    with pytest.raises(ValueError, match="provided together"):
        HuggingFaceLocalProvider(tokenizer=FakeTokenizer())


def test_empty_prompt_is_rejected() -> None:
    provider = HuggingFaceLocalProvider(
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )
    with pytest.raises(ValueError, match="prompt cannot be empty"):
        provider.generate(" ")
