"""Environment-driven configuration via Pydantic Settings.

All knobs (model name, temperature, paths, limits) live here so nothing is
hard-coded deeper in the codebase. Reads from a `.env` file in the project
root if present; falls back to OS environment variables; finally falls back
to the defaults below.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- LLM provider ----------
    # "cerebras"   = hosted, free, very fast Llama 3.3 70B. Needs CEREBRAS_API_KEY.
    # "groq"       = hosted, free, Llama / Qwen. Needs GROQ_API_KEY.
    # "huggingface"= hosted, free, open-source models via HF Inference. Needs HF_TOKEN.
    # "ollama"     = fully local, no API key. Needs Ollama installed locally.
    # "openai" / "anthropic" = optional paid providers.
    llm_provider: Literal[
        "cerebras", "groq", "huggingface", "ollama", "openai", "anthropic"
    ] = Field(
        default="cerebras",
        description="Which LLM backend to use. Default is Cerebras (free, fast, PwC-compatible).",
    )

    # ---------- Cerebras (default - PwC-compatible, very fast Llama 3.3) ----------
    cerebras_api_key: Optional[str] = None
    cerebras_base_url: str = "https://api.cerebras.ai/v1"
    cerebras_model: str = "llama-3.3-70b"
    cerebras_fallback_model: str = "llama3.1-8b"

    # ---------- HuggingFace Inference (corporate-friendly hosted option) ----------
    hf_token: Optional[str] = None
    # Default model: Mistral 7B Instruct (Apache 2.0). Always-free on HF Inference,
    # no license-acceptance gate. Alternative: "meta-llama/Llama-3.1-8B-Instruct"
    # (requires one-click license accept on HF), "Qwen/Qwen2.5-7B-Instruct".
    hf_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    hf_fallback_model: str = "HuggingFaceH4/zephyr-7b-beta"

    # ---------- Groq (alternative - hosted, open-source models, free tier) ----------
    groq_api_key: Optional[str] = None
    # Llama 3.3 70B - open weights (Llama 3.3 Community License); great quality.
    # Alternative: "llama-3.1-8b-instant" (faster), "qwen-2.5-32b" (Apache 2.0).
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fallback_model: str = "llama-3.1-8b-instant"

    # ---------- Ollama (offline / fully-local option) ----------
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_fallback_model: str = "phi3:mini"

    # ---------- Generic LLM behavior ----------
    llm_temperature: float = 0.2
    llm_timeout_s: int = 120
    llm_max_retries: int = 3
    max_tokens_per_turn: int = 8000
    max_agent_steps: int = 25

    # ---------- Optional alternative providers ----------
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-sonnet-latest"

    # ---------- Paths ----------
    data_dir: Path = PROJECT_ROOT / "data" / "csv"
    trace_dir: Path = PROJECT_ROOT / "traces"
    memory_db_path: Path = PROJECT_ROOT / "data" / "memory.sqlite"

    # ---------- Data download ----------
    # Set to false on corporate networks that use SSL inspection (PwC VPN, etc.)
    data_ssl_verify: bool = True

    def ensure_dirs(self) -> None:
        """Create on-disk directories that the app expects to exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.memory_db_path.parent.mkdir(parents=True, exist_ok=True)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the cached singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings
