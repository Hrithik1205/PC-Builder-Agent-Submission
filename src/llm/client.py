"""High-level LLM client with retries, token guard, and graceful fallback.

`invoke_with_retry` is the function every node should call instead of
hitting the provider directly. It:
  * approximates token usage and trims older messages if over budget,
  * retries transient errors with exponential backoff (tenacity),
  * falls back to the smaller `OLLAMA_FALLBACK_MODEL` if the primary
    times out repeatedly,
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
    retry_if_exception_type,
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

# Errors we treat as retryable. Plain Exception subclasses; we deliberately
# don't import Ollama-specific exceptions to keep this provider-agnostic.
RETRYABLE_EXCEPTIONS = (TimeoutError, ConnectionError, OSError)


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
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
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
    except (RetryError, *RETRYABLE_EXCEPTIONS) as e:
        log.warning("llm.primary_failed", error=str(e)[:200])
        try:
            fallback = get_fallback_model(temperature=temperature)
            return _call(fallback)
        except Exception as e2:
            log.error("llm.fallback_failed", error=str(e2)[:200])
            return _fallback_stub()
    except Exception as e:
        # Non-retryable error from the LLM (e.g. invalid arguments) - log and stub.
        log.error("llm.unexpected_error", error=str(e)[:200])
        return _fallback_stub()


def _fallback_stub() -> AIMessage:
    """Deterministic stub used when the LLM is fully unavailable."""
    return AIMessage(content=(
        "I'm currently unable to reach the language model. Please make sure "
        "Ollama is running (`ollama serve`) and the configured model is pulled "
        "(`ollama pull qwen2.5:7b-instruct`), then try again."
    ))
