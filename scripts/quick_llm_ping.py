"""Tiny LLM ping: prove the configured model responds with > 16 tokens."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_settings
from src.llm.client import invoke_with_retry


def main() -> int:
    s = get_settings()
    print(f"Provider:        {s.llm_provider}")
    print(f"Primary model:   {s.github_model}")
    print(f"Fallback model:  {s.github_fallback_model}")
    print()
    msgs = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="In one short sentence, name a good budget CPU."),
    ]
    ai = invoke_with_retry(msgs, temperature=0.2)
    content = (ai.content or "").strip()
    print(f"Output ({len(content)} chars):")
    print(f"  {content!r}")
    if "unable to reach" in content.lower():
        print("\nFAIL: still rate-limited or unreachable.")
        return 1
    if len(content) < 30:
        print("\nFAIL: response too short (still throttled).")
        return 1
    print("\nOK: model responding normally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
