"""LLM provider abstraction.

Supports five providers:
- **HuggingFace** (default) - hosted Inference API, free, serves open-source
  models like Mistral, Qwen, Llama. Needs `HF_TOKEN`. Corporate-friendly.
- **Groq** - hosted, free tier, very fast Llama 3.3 / Llama 3.1. Needs `GROQ_API_KEY`.
- **Ollama** - fully local, no API key, needs Ollama installed.
- **OpenAI / Anthropic** - optional paid providers.

The goal of this module is that the rest of the codebase imports
`get_chat_model()` and never knows which provider it is talking to.

Corporate-network note: `DATA_SSL_VERIFY=false` (originally added for GitHub CSV
downloads behind PwC's TLS-inspecting proxy) is also honored here. When false,
the underlying httpx client used by Groq/OpenAI/HF skips cert verification so
the requests can pass through the corporate MITM proxy.
"""
from __future__ import annotations

from typing import Any

from src.config import get_settings
from src.logging_setup import get_logger


log = get_logger(__name__)


def _httpx_clients_for_corp_proxy() -> dict:
    """Return kwargs that disable httpx cert verification for SDK clients.

    Returns an empty dict if cert verification is enabled (the default).
    Used by Groq / OpenAI / Anthropic providers, all of which expose an
    `http_client` constructor arg.
    """
    settings = get_settings()
    if settings.data_ssl_verify:
        return {}
    try:
        import httpx
    except ImportError:
        return {}
    return {
        "http_client": httpx.Client(verify=False, timeout=settings.llm_timeout_s),
        "http_async_client": httpx.AsyncClient(verify=False, timeout=settings.llm_timeout_s),
    }


def get_chat_model(temperature: float | None = None, **kwargs: Any):
    """Return a LangChain chat model for the configured provider."""
    settings = get_settings()
    temp = temperature if temperature is not None else settings.llm_temperature

    if settings.llm_provider == "github":
        return _make_github(temp, settings.github_model, **kwargs)
    if settings.llm_provider == "cerebras":
        return _make_cerebras(temp, settings.cerebras_model, **kwargs)
    if settings.llm_provider == "huggingface":
        return _make_huggingface(temp, settings.hf_model, **kwargs)
    if settings.llm_provider == "groq":
        return _make_groq(temp, settings.groq_model, **kwargs)
    if settings.llm_provider == "ollama":
        return _make_ollama(temp, settings.ollama_model, **kwargs)
    if settings.llm_provider == "openai":
        return _make_openai(temp, **kwargs)
    if settings.llm_provider == "anthropic":
        return _make_anthropic(temp, **kwargs)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")


def get_fallback_model(temperature: float | None = None, **kwargs: Any):
    """Return the configured fallback model (smaller / faster)."""
    settings = get_settings()
    temp = temperature if temperature is not None else settings.llm_temperature

    if settings.llm_provider == "github":
        return _make_github(temp, settings.github_fallback_model, **kwargs)
    if settings.llm_provider == "cerebras":
        return _make_cerebras(temp, settings.cerebras_fallback_model, **kwargs)
    if settings.llm_provider == "huggingface":
        return _make_huggingface(temp, settings.hf_fallback_model, **kwargs)
    if settings.llm_provider == "groq":
        return _make_groq(temp, settings.groq_fallback_model, **kwargs)
    if settings.llm_provider == "ollama":
        return _make_ollama(temp, settings.ollama_fallback_model, **kwargs)
    return get_chat_model(temperature, **kwargs)


def _make_github(temperature: float, model: str, **kwargs: Any):
    """GitHub Models (Azure OpenAI-compatible API).

    Free for personal use with a GitHub Personal Access Token.
    PwC-compatible: endpoint passes through the corporate AI policy filter
    because it's categorized as a Microsoft/GitHub service.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise RuntimeError(
            "github provider selected but `langchain-openai` is not installed. "
            "Run: pip install langchain-openai"
        ) from e
    settings = get_settings()
    if not settings.github_token:
        raise RuntimeError(
            "LLM_PROVIDER=github but GITHUB_TOKEN is not set.\n"
            "Create a free Personal Access Token at https://github.com/settings/tokens\n"
            "(classic token with the 'read:user' scope is enough) and add to .env:\n"
            "  GITHUB_TOKEN=ghp_..."
        )
    log.debug(
        "llm.github.create",
        model=model,
        temperature=temperature,
        ssl_verify=settings.data_ssl_verify,
    )
    # IMPORTANT: pass max_tokens explicitly. Without it, langchain-openai
    # routes the request through code paths that, for non-standard models,
    # can end up sending a small default (we saw ~16 tokens). Setting it to
    # 2048 leaves plenty of room for full Markdown responses with tables.
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=settings.github_token,
        base_url=settings.github_models_base_url,
        timeout=settings.llm_timeout_s,
        max_tokens=2048,
        **_httpx_clients_for_corp_proxy(),
        **kwargs,
    )


def _make_cerebras(temperature: float, model: str, **kwargs: Any):
    """Cerebras Inference API (OpenAI-compatible).

    Uses `langchain-openai`'s ChatOpenAI pointed at the Cerebras base URL.
    Cerebras serves Llama 3.3 70B and Llama 3.1 8B with a generous free tier.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise RuntimeError(
            "cerebras provider selected but `langchain-openai` is not installed. "
            "Run: pip install langchain-openai"
        ) from e
    settings = get_settings()
    if not settings.cerebras_api_key:
        raise RuntimeError(
            "LLM_PROVIDER=cerebras but CEREBRAS_API_KEY is not set.\n"
            "Get a free API key at https://cloud.cerebras.ai (sign in with Google),\n"
            "then add to .env:  CEREBRAS_API_KEY=csk-..."
        )
    log.debug(
        "llm.cerebras.create",
        model=model,
        temperature=temperature,
        ssl_verify=settings.data_ssl_verify,
    )
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=settings.cerebras_api_key,
        base_url=settings.cerebras_base_url,
        timeout=settings.llm_timeout_s,
        max_tokens=2048,
        **_httpx_clients_for_corp_proxy(),
        **kwargs,
    )


def _make_huggingface(temperature: float, model: str, **kwargs: Any):
    """HuggingFace Inference API via langchain-huggingface.

    Uses the free serverless Inference API. The HF token is a personal access
    token from https://huggingface.co/settings/tokens (no credit card needed).
    """
    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    except ImportError as e:
        raise RuntimeError(
            "huggingface provider selected but `langchain-huggingface` is not installed. "
            "Run: pip install langchain-huggingface huggingface-hub"
        ) from e
    settings = get_settings()
    if not settings.hf_token:
        raise RuntimeError(
            "LLM_PROVIDER=huggingface but HF_TOKEN is not set.\n"
            "Get a free token at https://huggingface.co/settings/tokens\n"
            "(any role works; 'Read' is enough) and add to .env:\n"
            "  HF_TOKEN=hf_..."
        )
    log.debug("llm.hf.create", model=model, temperature=temperature)
    endpoint = HuggingFaceEndpoint(
        repo_id=model,
        task="text-generation",
        max_new_tokens=2048,
        temperature=max(temperature, 0.01),  # HF rejects exactly 0
        huggingfacehub_api_token=settings.hf_token,
        timeout=settings.llm_timeout_s,
    )
    return ChatHuggingFace(llm=endpoint, model_id=model)


def _make_groq(temperature: float, model: str, **kwargs: Any):
    try:
        from langchain_groq import ChatGroq
    except ImportError as e:
        raise RuntimeError(
            "groq provider selected but `langchain-groq` is not installed. "
            "Run: pip install langchain-groq"
        ) from e
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError(
            "LLM_PROVIDER=groq but GROQ_API_KEY is not set.\n"
            "Get a free API key at https://console.groq.com/keys and add it to .env:\n"
            "  GROQ_API_KEY=gsk_..."
        )
    log.debug(
        "llm.groq.create",
        model=model,
        temperature=temperature,
        ssl_verify=settings.data_ssl_verify,
    )
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=settings.groq_api_key,
        timeout=settings.llm_timeout_s,
        **_httpx_clients_for_corp_proxy(),
        **kwargs,
    )


def _make_ollama(temperature: float, model: str, **kwargs: Any):
    from langchain_ollama import ChatOllama  # local import keeps optional providers truly optional

    settings = get_settings()
    log.debug("llm.ollama.create", model=model, temperature=temperature)
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=settings.ollama_base_url,
        timeout=settings.llm_timeout_s,
        num_predict=2048,
        **kwargs,
    )


def _make_openai(temperature: float, **kwargs: Any):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise RuntimeError(
            "openai provider selected but `langchain-openai` is not installed. "
            "Run: pip install langchain-openai"
        ) from e
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set in .env")
    log.debug("llm.openai.create", model=settings.openai_model)
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=temperature,
        api_key=settings.openai_api_key,
        timeout=settings.llm_timeout_s,
        **_httpx_clients_for_corp_proxy(),
        **kwargs,
    )


def _make_anthropic(temperature: float, **kwargs: Any):
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        raise RuntimeError(
            "anthropic provider selected but `langchain-anthropic` is not installed. "
            "Run: pip install langchain-anthropic"
        ) from e
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set in .env")
    log.debug("llm.anthropic.create", model=settings.anthropic_model)
    return ChatAnthropic(
        model=settings.anthropic_model,
        temperature=temperature,
        api_key=settings.anthropic_api_key,
        timeout=settings.llm_timeout_s,
        **kwargs,
    )
