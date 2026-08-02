# -*- coding: utf-8 -*-
"""LLM provider: Google Gemini (cloud), with structural PII enforcement (Template Method).

PII enforcement (Template Method — Zero Trust architecture): BaseLLMProvider.chat() is the *only* public method (marked @final), and always performs PII enforcement (via app.pipeline.pii.redact_texts) *before* it passes the messages to the abstract _call_api() hook that each concrete provider implements. The hook has no access to the raw messages at all. The layer is preserved even when there is only one provider: so any future provider added will inherit the structural enforcement and cannot "forget" it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, final

from ..config import Settings


class LLMError(RuntimeError):
    pass


class BaseLLMProvider(ABC):
    """Template Method: chat() is the only public method, and cannot be overridden (@final).

    The algorithm is fixed: (1) local PII enforcement on all message contents, (2) passing to _call_api (the abstract hook that each concrete provider implements). No provider can skip step 1.
    """

    def __init__(self, settings: Settings):
        self._settings = settings

    @final
    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        redacted = self._redact(messages)
        return self._call_api(redacted, temperature)

    def _redact(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Required step: local PII enforcement (HebSafeHarbor) on all message contents — system,
        user and assistant together — before going to the API. If the PII environment is not enforced,
        fail (PIIError) and do not send raw data.
        """
        from .pii import redact_texts

        texts = [m.get("content", "") for m in messages]
        redacted_texts = redact_texts(self._settings, texts)
        return [
            {"role": m.get("role", "user"), "content": text}
            for m, text in zip(messages, redacted_texts)
        ]

    @abstractmethod
    def _call_api(self, messages: list[dict[str, str]], temperature: float) -> str:
        """Hook that each concrete provider implements: actual call to its API. The messages that come here have already gone through PII enforcement (in chat()) — it is forbidden for a provider to call any API with content that has not gone through _redact()."""
        raise NotImplementedError


# ---------- Cloud Provider (Google Gemini, google-genai SDK) ----------

class GeminiModelProvider(BaseLLMProvider):
    """Google Gemini — the only summarization engine of the system.

    Based on the official google-genai SDK (not google-generativeai, which Google stopped supporting). The message contents have already gone through local PII enforcement *before* they reach here — it is forbidden for a provider to call any API with content that has not gone through _redact() in BaseLLMProvider.chat() (Template Method), not here.
    """

    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise LLMError(
                "Missing Gemini API key. Set PUHA_GEMINI_API_KEY in .env, "
                "or enter a key in the settings (⚙) — it overrides the .env."
            )
        super().__init__(settings)
        self._genai_types = self._load_sdk_types()
        self._client = self._genai_types["genai"].Client(api_key=settings.gemini_api_key)

    @staticmethod
    def _load_sdk_types() -> dict[str, Any]:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMError(
                "The google-genai package is not installed. Run: pip install google-genai"
            ) from exc
        return {"genai": genai, "types": types}

    @staticmethod
    def _to_gemini_turns(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert messages in OpenAI format (system/user/assistant) to Gemini format.

        system -> system_instruction separated (Gemini does not use the system role inside contents). assistant -> "model" (the name Gemini expects); user remains user.
        """
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        turns = [
            {"role": "model" if m.get("role") == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages
            if m.get("role") != "system"
        ]
        return ("\n\n".join(system_parts) or None), turns

    def _build_config(self, types_mod: Any, system_instruction: str | None, temperature: float, with_thinking: bool):
        kwargs: dict[str, Any] = {
            "system_instruction": system_instruction,
            "temperature": temperature,
            "max_output_tokens": self._settings.llm_max_tokens,
        }
        if with_thinking:
            kwargs["thinking_config"] = types_mod.ThinkingConfig(
                thinking_budget=self._settings.gemini_thinking_budget,
                include_thoughts=False,
            )
        return types_mod.GenerateContentConfig(**kwargs)

    def _call_api(self, messages: list[dict[str, str]], temperature: float) -> str:
        settings = self._settings
        types_mod = self._genai_types["types"]

        system_instruction, turns = self._to_gemini_turns(messages)
        if not turns:
            raise LLMError("אין תוכן משתמש לשליחה ל-Gemini.")

        config = self._build_config(types_mod, system_instruction, temperature, with_thinking=True)
        try:
            response = self._client.models.generate_content(
                model=settings.gemini_model, contents=turns, config=config
            )
        except Exception as exc:  # noqa: BLE001
            # A specific model/version may reject thinking_config (e.g. budget not supported for that model generation) — try again without it.
            if "thinking" in str(exc).lower() or "INVALID_ARGUMENT" in str(exc):
                config = self._build_config(types_mod, system_instruction, temperature, with_thinking=False)
                try:
                    response = self._client.models.generate_content(
                        model=settings.gemini_model, contents=turns, config=config
                    )
                except Exception as exc2:  # noqa: BLE001
                    raise LLMError(f"Error calling Gemini ({settings.gemini_model}): {exc2}") from exc2
            else:
                raise LLMError(f"Error calling Gemini ({settings.gemini_model}): {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            raise LLMError(f"Empty or unexpected format response from Gemini: {response!r}")
        return text


# ---------- Factory ----------

def get_provider(settings: Settings) -> BaseLLMProvider:
    """Return the LLM provider. Currently only Gemini (the local provider was removed)."""
    return GeminiModelProvider(settings)
