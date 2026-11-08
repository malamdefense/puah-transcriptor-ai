# -*- coding: utf-8 -*-
"""LLM provider: Google Gemini (cloud), with structural PII enforcement (Template Method).

Architectural note: the system previously also ran against a local model (Ollama/gemma4)
behind a Strategy pattern. That provider was removed, and Gemini is now the only
summarization engine. The consequence is deliberate and worth stating: producing a
summary/feedback requires a valid API key and internet connectivity, and the call
content (after PII redaction) leaves to Google's servers — so the original "0% cloud
egress" criterion can no longer be met.

What remains local (On-Prem): transcription (faster-whisper), speaker diarization
(pyannote), PII redaction (HebSafeHarbor) and the RAG embeddings (bge-m3 via Ollama).

PII enforcement (Template Method — Zero Trust): BaseLLMProvider.chat() is the *only*
public method (marked @final) and always runs PII redaction (via
app.pipeline.pii.redact_texts) *before* handing the messages to the abstract
_call_api() hook that each concrete provider implements. The hook never sees the raw
messages. The layer is kept even though a single provider remains, so any provider
added later inherits the enforcement structurally and cannot "forget" it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re
from typing import Any, final
import urllib.error
import urllib.request

from ..config import Settings
from . import import_failure_reason


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

        Re-identification (reidentify.py): when a deliverable is being generated with a vault, the
        generic tags the engines just produced for names/organisations are numbered into it, by
        aligning each message before and after the engines. This only renames tags in text that is
        already redacted — nothing unredacted is added — and a tag whose value is ambiguous stays
        generic (unrestorable).
        """
        from .pii import redact_texts
        from .reidentify import current_vault

        texts = [m.get("content", "") for m in messages]
        redacted_texts = redact_texts(self._settings, texts)
        vault = current_vault()
        if vault is not None:
            redacted_texts = [vault.number(t, r) for t, r in zip(texts, redacted_texts)]
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
                import_failure_reason(
                    exc, "google.genai", "pip install google-genai", package="google-genai"
                )
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


# ---------- Local Ollama Provider (e.g. muse-glimmer) ----------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_GLIMMER_STOP_TOKENS = ("<|eot|>", "<|eom|>", "<|end_of_text|>")


def _ollama_root(base_url: str) -> str:
    root = (base_url or "http://localhost:11434").rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3].rstrip("/")
    return root


def ollama_reachable(settings: Settings, timeout: float = 0.5) -> bool:
    """Fast local check whether the configured Ollama server is listening."""
    if not settings.llm_base_url or not settings.llm_model:
        return False
    url = f"{_ollama_root(settings.llm_base_url)}/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:  # noqa: BLE001
        return False


def _format_glimmer_fast_prompt(messages: list[dict[str, str]]) -> str:
    """Format chat messages into Glimmer's native Fast-mode prompt (no thinking turn).

    In Ollama 0.40.14, calling `/api/chat` with `think: false` on `muse-glimmer` triggers a bug
    in `GlimmerParser`: `GlimmerRenderer` pre-fills `<|start|>assistant to=user<|message|>` at the
    end of the prompt so the model immediately generates the response body, but `GlimmerParser`
    initializes in `consumeHeader` waiting for `<|message|>` in the generated stream and swallows
    the entire response (`content: ""`). Using `/api/generate` with `raw: True` and Glimmer's
    native `Reasoning strength: none` + `<|start|>assistant to=user<|message|>` header forces
    strict Fast mode (zero thinking tokens) and bypasses the broken parser state machine.
    """
    parts: list[str] = []
    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system:
        parts.append("<|start|>system<|message|>Reasoning strength: none<|eot|>")
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if role == "system":
            if "reasoning strength:" not in content.lower():
                content = f"{content}\n\nReasoning strength: none" if content else "Reasoning strength: none"
            parts.append(f"<|start|>system<|message|>{content}<|eot|>")
        elif role == "assistant":
            parts.append(f"<|start|>assistant to=user<|message|>{content}<|eot|>")
        else:
            parts.append(f"<|start|>user<|message|>{content}<|eot|>")
    parts.append("<|start|>assistant to=user<|message|>")
    return "".join(parts)


def _glimmer_num_ctx(prompt_text: str, num_predict: int) -> int:
    """Choose a tight context window (multiple of 2048) so KV cache fits in 16 GB GPU VRAM."""
    est_prompt_tokens = int(len(prompt_text) * 0.65) + 256
    needed = est_prompt_tokens + num_predict
    if needed <= 4096:
        return 4096
    if needed <= 6144:
        return 6144
    if needed <= 8192:
        return 8192
    return 12288


class OllamaModelProvider(BaseLLMProvider):
    """Local Ollama LLM provider (e.g. muse-glimmer).

    Inherits BaseLLMProvider without overriding @final chat(), so all messages still pass
    through local PII redaction (and numbered-tag vault recording) before _call_api().
    """

    def __init__(self, settings: Settings):
        if not (settings.llm_model or "").strip():
            raise LLMError("Missing Ollama model name (PUAH_LLM_MODEL).")
        super().__init__(settings)

    def _call_api(self, messages: list[dict[str, str]], temperature: float) -> str:
        settings = self._settings
        root = _ollama_root(settings.llm_base_url)
        is_glimmer_fast = ("glimmer" in (settings.llm_model or "").lower()) and not settings.llm_think
        num_predict = min(settings.llm_max_tokens, 2048) if not settings.llm_think else settings.llm_max_tokens

        if is_glimmer_fast:
            url = f"{root}/api/generate"
            raw_prompt = _format_glimmer_fast_prompt(messages)
            payload: dict[str, Any] = {
                "model": settings.llm_model,
                "prompt": raw_prompt,
                "raw": True,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                    "num_ctx": _glimmer_num_ctx(raw_prompt, num_predict),
                    "stop": list(_GLIMMER_STOP_TOKENS),
                },
            }
        else:
            url = f"{root}/api/chat"
            payload = {
                "model": settings.llm_model,
                "messages": messages,
                "stream": False,
                "think": bool(settings.llm_think),
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                    "num_ctx": 8192,
                },
            }

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=settings.llm_timeout_sec) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400] if exc.fp else str(exc)
            raise LLMError(f"Error calling Ollama ({settings.llm_model}, HTTP {exc.code}): {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(f"Ollama server unreachable at {url} ({settings.llm_model}): {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Invalid JSON from Ollama ({settings.llm_model}): {exc}") from exc

        content = ""
        if isinstance(data, dict):
            if isinstance(data.get("response"), str) and data["response"]:
                content = data["response"]
            else:
                msg = data.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    content = msg["content"]
                elif isinstance(data.get("choices"), list) and data["choices"]:
                    c0 = data["choices"][0]
                    if isinstance(c0, dict) and isinstance(c0.get("message"), dict):
                        content = c0["message"].get("content") or ""
        if not settings.llm_think and content:
            content = _THINK_RE.sub("", content)
            for tok in _GLIMMER_STOP_TOKENS:
                content = content.replace(tok, "")
            content = content.strip()
        if not content or not content.strip():
            raise LLMError(f"Empty response from Ollama ({settings.llm_model}).")
        return content


# ---------- Factory ----------

def get_provider(settings: Settings) -> BaseLLMProvider:
    """Return the configured LLM provider (Gemini in cloud production, Ollama for local runs)."""
    if getattr(settings, "llm_provider", "gemini") == "ollama":
        return OllamaModelProvider(settings)
    return GeminiModelProvider(settings)
