"""Reproduce the user's reported bug: clarifying-question + answer should
merge into a complete Requirements object, not loop forever.
"""
from __future__ import annotations

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
    thread_id = f"smoke-clarify-{uuid.uuid4().hex[:8]}"
    cfg = make_thread_config(thread_id)

    print("\n=== Turn 1: vague request (expect clarifying questions) ===")
    r1 = graph.invoke(
        {"messages": [HumanMessage(
            content="i want a PC with 512 GB storage and 8 gb RAM"
        )]},
        config=cfg,
    )
    reqs1 = r1.get("requirements") or {}
    reply1 = r1.get("final_response") or ""
    print(f"  use_case={reqs1.get('use_case')}, budget={reqs1.get('budget_usd')}, "
          f"confidence={reqs1.get('confidence')}")
    print(f"  must_have={reqs1.get('must_have')}")
    print(f"  reply preview: {reply1[:120]!r}")
    has_clarify = "more info" in reply1.lower() or "?" in reply1
    print(f"  Asked clarifying question: {has_clarify}")
    assert has_clarify, "Expected clarifying questions on turn 1"

    print("\n=== Turn 2: user answers (expect build, not another question) ===")
    r2 = graph.invoke(
        {"messages": [HumanMessage(
            content="i need it for office use and budget is 600 to 700"
        )]},
        config=cfg,
    )
    reqs2 = r2.get("requirements") or {}
    build2 = r2.get("build") or {}
    reply2 = r2.get("final_response") or ""

    print(f"  use_case={reqs2.get('use_case')}, "
          f"budget_min={reqs2.get('budget_min_usd')}, "
          f"budget={reqs2.get('budget_usd')}, "
          f"confidence={reqs2.get('confidence')}")
    print(f"  must_have={reqs2.get('must_have')}")
    print(f"  build parts: {len(build2)}")
    total = sum(float(c.get("price", 0) or 0) for c in build2.values() if c)
    print(f"  total: ${total:.2f}")

    still_asking = "more info before" in reply2.lower()
    print(f"  Still asking clarifying questions: {still_asking}")
    if still_asking:
        print("\n--- Response preview ---")
        print(reply2[:800])
        return 1

    if reqs2.get("use_case") != "office":
        print(f"FAIL: use_case should be 'office', got {reqs2.get('use_case')!r}")
        return 1
    if reqs2.get("budget_usd") != 700 or reqs2.get("budget_min_usd") != 600:
        print(f"FAIL: budget should be 600-700, got "
              f"{reqs2.get('budget_min_usd')}-{reqs2.get('budget_usd')}")
        return 1
    if not build2:
        print("FAIL: expected a build to be produced")
        return 1

    print("\nSUCCESS: gatherer merged turn 1 info with turn 2 answer and built a PC.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
