"""End-to-end smoke test: build, then revise the budget, then verify the
responder includes a comparison section.

Run with:
    .venv\\Scripts\\python.exe scripts\\smoke_e2e_compare.py

Requires the LLM provider configured in .env to be reachable.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage

from src.agent.graph import build_graph, make_thread_config
from src.logging_setup import configure_logging


def main() -> int:
    configure_logging()
    graph = build_graph(with_memory=True)
    thread_id = f"smoke-{uuid.uuid4().hex[:8]}"
    cfg = make_thread_config(thread_id)

    print("\n=== Turn 1: initial gaming PC at $1500 ===")
    r1 = graph.invoke(
        {"messages": [HumanMessage(content="Gaming PC for $1500, 1440p")]},
        config=cfg,
    )
    build1 = r1.get("build") or {}
    print(f"  Parts: {len(build1)}")
    print(f"  Total: ${sum(float(c.get('price', 0) or 0) for c in build1.values() if c):.2f}")

    print("\n=== Turn 2: lower the budget to $900 (should trigger comparison) ===")
    r2 = graph.invoke(
        {"messages": [HumanMessage(content="Actually, make my budget $900 instead")]},
        config=cfg,
    )
    build2 = r2.get("build") or {}
    prev = r2.get("previous_build") or {}
    print(f"  Previous build present in state: {bool(prev)}")
    print(f"  New total: ${sum(float(c.get('price', 0) or 0) for c in build2.values() if c):.2f}")

    reply = r2.get("final_response") or ""
    has_section = "what changed" in reply.lower()
    print(f"  Response includes 'What changed' section: {has_section}")
    if not has_section:
        print("\n--- Response preview ---")
        print(reply[:1500])
        return 1

    print("\nSUCCESS: comparison section appears in the response.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
