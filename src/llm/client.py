"""High-level LLM client with retries, token guard, and graceful fallback.

`invoke_with_retry` is the function every node should call instead of
hitting the provider directly. It:
  * approximates token usage and trims older messages if over budget,
  * retries transient errors with exponential backoff (tenacity),
  * falls back to the smaller configured fallback model if the primary
    times out / rate-limits repeatedly,
  * returns a deterministic stub response if all retries fail, so the
    graph never crashes.
"""
from __future__ import annotations

import time
from typing import Any, List, Optional

import tiktoken
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_settings
from src.llm.providers import get_chat_model, get_fallback_model
from src.logging_setup import get_logger


log = get_logger(__name__)


# tiktoken doesn't ship a tokenizer for Qwen/Phi, but cl100k_base is a
# reasonable approximation and only used for budget guards.
_ENCODER = tiktoken.get_encoding("cl100k_base")

# Errors we treat as retryable (classic network errors).
RETRYABLE_EXCEPTIONS = (TimeoutError, ConnectionError, OSError)


def _is_retryable_http_error(exc: BaseException) -> bool:
    """Classify openai / httpx / langchain exceptions that warrant a retry.

    GitHub Models (langchain-openai), Cerebras, and Groq all surface
    rate-limit / transient server errors as openai.RateLimitError or
    APIStatusError. We classify by inspecting the exception name and
    optional `.status_code` attribute, which avoids hard-importing optional
    SDKs."""
    if isinstance(exc, RETRYABLE_EXCEPTIONS):
        return True
    name = type(exc).__name__.lower()
    if any(k in name for k in ("ratelimit", "timeout", "apiconnection",
                                  "internalserver", "serviceunavailable")):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if isinstance(status, int) and status in (408, 409, 429, 500, 502, 503, 504):
        return True
    return False


def estimate_tokens(text: str) -> int:
    return len(_ENCODER.encode(text or ""))


def estimate_message_tokens(msgs: List[BaseMessage]) -> int:
    return sum(estimate_tokens(getattr(m, "content", "") or "") for m in msgs)


def trim_to_budget(msgs: List[BaseMessage], budget: int) -> List[BaseMessage]:
    """Drop the oldest non-system messages until under the token budget.

    Keeps the first SystemMessage if present, and always keeps the most
    recent message.
    """
    if estimate_message_tokens(msgs) <= budget:
        return msgs

    system: List[BaseMessage] = [m for m in msgs[:1] if isinstance(m, SystemMessage)]
    rest = msgs[len(system):]
    while rest and estimate_message_tokens(system + rest) > budget and len(rest) > 1:
        rest = rest[1:]
    log.warning("llm.context_trimmed", final_msgs=len(system + rest), budget=budget)
    return system + rest


def _llm_invoke_inner(model, msgs: List[BaseMessage], tools: Optional[list]) -> AIMessage:
    if tools:
        model = model.bind_tools(tools)
    return model.invoke(msgs)


def invoke_with_retry(
    msgs: List[BaseMessage],
    tools: Optional[list] = None,
    temperature: Optional[float] = None,
    structured_output_schema: Any = None,
) -> AIMessage:
    """Invoke the configured LLM with retries and fallback.

    If `structured_output_schema` is provided, the schema is bound via
    `.with_structured_output` and the call returns an AIMessage whose
    `.content` is a Python object matching the schema (LangChain auto-parses).
    """
    settings = get_settings()
    msgs = trim_to_budget(msgs, settings.max_tokens_per_turn)

    @retry(
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception(_is_retryable_http_error),
        reraise=True,
    )
    def _call(model):
        t0 = time.time()
        if structured_output_schema is not None:
            bound = model.with_structured_output(structured_output_schema)
            result = bound.invoke(msgs)
            elapsed_ms = int((time.time() - t0) * 1000)
            log.info(
                "llm.invoke",
                latency_ms=elapsed_ms,
                input_tokens=estimate_message_tokens(msgs),
                mode="structured",
            )
            return AIMessage(content=str(result), additional_kwargs={"parsed": result})
        ai_msg: AIMessage = _llm_invoke_inner(model, msgs, tools)
        elapsed_ms = int((time.time() - t0) * 1000)
        log.info(
            "llm.invoke",
            latency_ms=elapsed_ms,
            input_tokens=estimate_message_tokens(msgs),
            output_tokens=estimate_tokens(ai_msg.content or ""),
            tool_calls=len(getattr(ai_msg, "tool_calls", []) or []),
            mode="chat",
        )
        return ai_msg

    # Try primary, then fallback, then stub.
    try:
        primary = get_chat_model(temperature=temperature)
        return _call(primary)
    except Exception as e:
        # Log full exception details so root-cause is easy to see in traces.
        log.warning(
            "llm.primary_failed",
            error_type=type(e).__name__,
            status_code=getattr(e, "status_code", None)
                        or getattr(e, "http_status", None),
            error=str(e)[:300],
        )
        try:
            fallback = get_fallback_model(temperature=temperature)
            return _call(fallback)
        except Exception as e2:
            log.error(
                "llm.fallback_failed",
                error_type=type(e2).__name__,
                status_code=getattr(e2, "status_code", None)
                            or getattr(e2, "http_status", None),
                error=str(e2)[:300],
            )
            return _fallback_stub(last_error=e2 or e)


def _fallback_stub(last_error: Optional[BaseException] = None) -> AIMessage:
    """Deterministic stub used when the LLM is fully unavailable.

    The message is provider-aware so users running on GitHub Models /
    Cerebras / Groq don't get told to "start Ollama".
    """
    settings = get_settings()
    provider = settings.llm_provider
    err_kind = type(last_error).__name__ if last_error else ""
    status = (
        getattr(last_error, "status_code", None)
        or getattr(last_error, "http_status", None)
        if last_error else None
    )

    hint = ""
    if status == 429 or "ratelimit" in err_kind.lower():
        hint = (
            " You've hit the provider's rate limit. Wait ~60 seconds and "
            "send the message again."
        )
    elif status in (401, 403):
        hint = (
            " Your API key was rejected. Open `.env`, refresh the token, "
            "and restart the app."
        )
    elif status in (500, 502, 503, 504):
        hint = " The provider's API is having issues. Try again in a minute."

    provider_specifics = {
        "github": (
            "I'm currently unable to reach GitHub Models."
            + hint
            + " Check https://github.com/marketplace/models for status, or "
            "switch to a different provider in `.env`."
        ),
        "cerebras": (
            "I'm currently unable to reach Cerebras."
            + hint
            + " Verify CEREBRAS_API_KEY in `.env`."
        ),
        "groq": (
            "I'm currently unable to reach Groq."
            + hint
            + " Verify GROQ_API_KEY in `.env`."
        ),
        "huggingface": (
            "I'm currently unable to reach the HuggingFace Inference API."
            + hint
            + " Verify HF_TOKEN in `.env` and that the model is loaded."
        ),
        "ollama": (
            "I'm currently unable to reach the language model. Please make "
            "sure Ollama is running (`ollama serve`) and the configured "
            "model is pulled (`ollama pull " + settings.ollama_model + "`)."
        ),
        "openai": (
            "I'm currently unable to reach OpenAI." + hint
            + " Verify OPENAI_API_KEY in `.env`."
        ),
        "anthropic": (
            "I'm currently unable to reach Anthropic." + hint
            + " Verify ANTHROPIC_API_KEY in `.env`."
        ),
    }
    msg = provider_specifics.get(
        provider,
        "I'm currently unable to reach the language model." + hint,
    )
    return AIMessage(content=msg)
