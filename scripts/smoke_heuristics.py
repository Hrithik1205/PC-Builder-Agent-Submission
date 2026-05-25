"""Quick check for both heuristics: use_case and feedback intent."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.nodes import _heuristic_use_case, _heuristic_feedback


def main() -> int:
    print("=== use_case heuristic (single-word) ===")
    cases = [
        ("Home PC for $500", "office"),
        ("Personal", "office"),
        ("office", "office"),
        ("home", "office"),
        ("casual", "office"),
        ("everyday", "office"),
        ("personal use, $500", "office"),
        ("gaming", "gaming"),
        ("workstation", "workstation"),
        ("video editing rig", "content_creation"),
        ("home server", "home_server"),
        ("$1500 budget", None),
        ("I want a PC", None),
    ]
    failed = 0
    for text, expected in cases:
        got = _heuristic_use_case(text)
        ok = got == expected
        flag = "OK " if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{flag}] {text!r:40s} -> {got!r:18s} (expected {expected!r})")

    print("\n=== feedback heuristic ===")
    fb_cases = [
        ("increase budget to $600", "change_budget", 600),
        ("can you bump the budget to $1200?", "change_budget", 1200),
        ("budget is now $900", "change_budget", 900),
        ("between $1000 and $1500", "change_budget", 1500),
        ("make it cheaper", "swap_part", None),  # price_lower
        ("looks good, thanks", "approve", None),
        ("ship it", "approve", None),
        ("quieter please", "swap_part", None),
        ("more storage", "swap_part", None),
        ("more RAM", "swap_part", None),
        ("change the GPU", "swap_part", None),
        ("swap the cpu", "swap_part", None),
        ("what's the weather", None, None),
    ]
    for text, expected_intent, expected_budget in fb_cases:
        fb = _heuristic_feedback(text)
        got_intent = fb.get("intent") if fb else None
        got_budget = (fb or {}).get("delta_constraints", {}).get("budget_usd")
        ok = got_intent == expected_intent and (
            expected_budget is None or got_budget == expected_budget
        )
        flag = "OK " if ok else "FAIL"
        if not ok:
            failed += 1
        budget_str = f" budget={got_budget}" if got_budget else ""
        print(f"  [{flag}] {text!r:40s} -> intent={got_intent!r}{budget_str} "
              f"(expected {expected_intent!r}{f' budget={expected_budget}' if expected_budget else ''})")

    print()
    if failed:
        print(f"{failed} case(s) failed.")
        return 1
    print(f"All heuristic checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
