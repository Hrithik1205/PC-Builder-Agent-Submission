"""Stress-test the configured LLM with a long prompt to confirm it can
produce the multi-paragraph + table response the responder needs."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_settings
from src.llm.client import invoke_with_retry


PROMPT = """\
You are PCBuilderAgent. Given the build below, produce a Markdown response with:
1. A one-sentence opening summary.
2. A table of all parts with columns | Component | Part | Price |.
3. "Why these picks" with 3 bullet points.
4. A "Want me to swap anything?" closer.

Build:
- CPU: AMD Ryzen 9 9950X ($533.83)
- Motherboard: ASRock B650M-H/M.2+ ($99.99)
- Memory: G.Skill Flare X5 64 GB ($142.99)
- Video card: MSI GT 710 ($45.99)
- Storage: Silicon Power A60 ($95.97)
- PSU: MSI MAG A550BN ($54.99)
- Case: Cooler Master MasterBox Q300L ($36.99)
- Cooler: Iceberg Thermal IceFLOE T95 ($9.99)

Total: $1020.74, Budget: $1500.
"""


def main() -> int:
    s = get_settings()
    print(f"Provider: {s.llm_provider}")
    print(f"Model:    {s.github_model}")
    print()
    msgs = [
        SystemMessage(content="You are a helpful PC building assistant."),
        HumanMessage(content=PROMPT),
    ]
    ai = invoke_with_retry(msgs, temperature=0.3)
    content = (ai.content or "").strip()
    print(f"Output: {len(content)} chars\n")
    print(content[:1500])
    print()
    if "unable to reach" in content.lower():
        print("FAIL: LLM unreachable.")
        return 1
    if len(content) < 300:
        print("FAIL: response too short (likely throttled).")
        return 1
    if "|" not in content:
        print("FAIL: no markdown table in response.")
        return 1
    print("OK: model produces full markdown response.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
