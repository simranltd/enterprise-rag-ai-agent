"""Local Hugging Face causal language model provider."""

from __future__ import annotations

import os
from typing import Any

from .base import LLMResponse

DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.7B"
DEFAULT_MAX_NEW_TOKENS = 128


class HuggingFaceLocalProvider:
    """Generate deterministic responses with a local Transformers model."""

    provider = "huggingface-local"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        device: str | None = None,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        if (tokenizer is None) != (model is None):
            raise ValueError("tokenizer and model must be provided together")
        self.model_name = model_name
        self.model = model_name
        self.max_new_tokens = max_new_tokens
        self._device_name = device
        self._resolved_device: str | None = None
        self._tokenizer = tokenizer
        self._model = model

    @property
    def tokenizer(self) -> Any:
        self._load()
        return self._tokenizer

    @property
    def loaded_model(self) -> Any:
        self._load()
        return self._model

    def generate(self, prompt: str) -> LLMResponse:
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        tokenizer, model = self._load()
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        inputs = self._move_inputs_to_device(inputs)
        import torch

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        input_ids = inputs["input_ids"]
        generated_ids = generated[:, input_ids.shape[-1] :]
        text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return LLMResponse(text=text, provider=self.provider, model=self.model_name)

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "torch and transformers are required for HuggingFaceLocalProvider"
            ) from error

        torch.set_num_threads(max(1, os.cpu_count() or 1))
        device = self._device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        self._resolved_device = device
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model_kwargs = {
            "torch_dtype": torch.float32 if device == "cpu" else torch.float16
        }
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        self._model.to(device)
        self._model.eval()
        return self._tokenizer, self._model

    def _move_inputs_to_device(self, inputs: Any) -> Any:
        device = self._resolved_device
        if device is None:
            try:
                import torch
            except ImportError as error:
                raise RuntimeError(
                    "torch is required for HuggingFaceLocalProvider generation"
                ) from error
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if hasattr(inputs, "to"):
            return inputs.to(device)
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
