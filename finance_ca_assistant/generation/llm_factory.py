"""LLM provider factory for generation."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

# Qwen3 chat templates enable thinking by default; the reasoning block must
# never reach JSON parsing or users.
_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", flags=re.DOTALL | re.IGNORECASE)


class LLMClient(ABC):
    """Interface for chat generation clients."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 900) -> str:
        """Generate a response string."""


class EchoLLM(LLMClient):
    """Deterministic local LLM used for tests and offline dry-runs."""

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 900) -> str:
        return user_prompt[:max_tokens]


class HuggingFacePipelineLLM(LLMClient):
    """Transformers chat-generation pipeline.

    Defaults to ``Qwen/Qwen3-4B`` for Kaggle/local GPU runs. The pipeline can be
    injected for tests or created lazily from transformers when real local
    generation is needed.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        pipeline: Optional[object] = None,
        device_map: str = "auto",
        dtype: str = "auto",
        torch_dtype: Optional[str] = None,
        temperature: float = 0.1,
        do_sample: bool = False,
    ) -> None:
        self.model_name = model_name
        self.dtype = torch_dtype or dtype
        self.pipeline = pipeline or self._load_pipeline(
            model_name=model_name,
            device_map=device_map,
            dtype=self.dtype,
        )
        self.temperature = temperature
        self.do_sample = do_sample

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 900) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        generation_kwargs: Dict[str, Any] = {
            "do_sample": self.do_sample,
        }
        prompt_token_count = self._prompt_token_count(messages)
        if prompt_token_count:
            generation_kwargs["max_length"] = prompt_token_count + max_tokens
        else:
            generation_kwargs["max_new_tokens"] = max_tokens
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature

        output = self.pipeline(messages, **generation_kwargs)
        return self._extract_text(output)

    def generate_batch(
        self,
        system_prompt: str,
        user_prompts: Sequence[str],
        max_tokens: int = 900,
        batch_size: int = 8,
    ) -> List[str]:
        """Generate one response per prompt, batching on the GPU."""

        if not user_prompts:
            return []
        conversations = [
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            for prompt in user_prompts
        ]
        tokenizer = getattr(self.pipeline, "tokenizer", None)
        if tokenizer is not None:
            # Decoder-only models must be left-padded for batched generation.
            tokenizer.padding_side = "left"
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token
        generation_kwargs: Dict[str, Any] = {"do_sample": self.do_sample, "max_new_tokens": max_tokens}
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature
        outputs = self.pipeline(conversations, batch_size=batch_size, **generation_kwargs)
        return [self._extract_text(output) for output in outputs]

    def _prompt_token_count(self, messages: List[Dict[str, str]]) -> Optional[int]:
        tokenizer = getattr(self.pipeline, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            return None

        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception:
            return None

        shape = getattr(encoded, "shape", None)
        if shape and len(shape) >= 2:
            return int(shape[-1])
        try:
            return len(encoded[0])
        except Exception:
            return None

    @staticmethod
    def _extract_text(output: object) -> str:
        return _THINK_BLOCK.sub("", HuggingFacePipelineLLM._raw_text(output)).strip()

    @staticmethod
    def _raw_text(output: object) -> str:
        if isinstance(output, str):
            return output
        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, dict):
                generated = first.get("generated_text")
                if isinstance(generated, str):
                    return generated
                if isinstance(generated, list):
                    return _last_assistant_content(generated)
        return str(output)

    @staticmethod
    def _load_pipeline(model_name: str, device_map: str, dtype: str) -> object:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Install transformers and torch to use Hugging Face generation: "
                "pip install transformers torch accelerate"
            ) from exc

        kwargs = {
            "model": model_name,
            "tokenizer": model_name,
            "device_map": device_map,
        }
        if dtype:
            kwargs["dtype"] = dtype
        try:
            return pipeline("text-generation", **kwargs)
        except TypeError:
            kwargs.pop("dtype")
            if dtype != "auto":
                kwargs["torch_dtype"] = dtype
            return pipeline("text-generation", **kwargs)


def _last_assistant_content(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    if messages and "content" in messages[-1]:
        return str(messages[-1].get("content") or "")
    return str(messages)


def create_llm(name: str = "echo", **kwargs: object) -> LLMClient:
    """Create an LLM client by name."""

    normalized = name.lower()
    if normalized in {"echo", "local"}:
        return EchoLLM()
    if normalized in {"huggingface", "transformers", "qwen3-4b", "qwen/qwen3-4b"}:
        kwargs.setdefault("model_name", "Qwen/Qwen3-4B")
        return HuggingFacePipelineLLM(**kwargs)
    if normalized in {"qwen3-8b", "qwen/qwen3-8b"}:
        kwargs.setdefault("model_name", "Qwen/Qwen3-8B")
        return HuggingFacePipelineLLM(**kwargs)
    raise ValueError(f"Unknown LLM provider: {name}")
