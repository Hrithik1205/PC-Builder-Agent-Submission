"""Quick check that vague-but-clear use-case phrases get mapped, not asked."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.nodes import _heuristic_use_case


def main() -> int:
    cases = [
        ("I want a PC for personal use and my budget is $500", "office"),
        ("home use, $700", "office"),
        ("just for browsing and Netflix", "office"),
        ("everyday tasks, $500", "office"),
        ("for school work and YouTube", "office"),
        ("1440p gaming, $1500", "gaming"),
        ("personal gaming PC, $1000", "gaming"),
        ("video editing, $2500", "content_creation"),
        ("CAD work, $3000", "workstation"),
        ("Plex home server $800", "home_server"),
        ("I want a PC", None),
        ("$1500 budget", None),
    ]
    failed = 0
    for text, expected in cases:
        got = _heuristic_use_case(text)
        ok = got == expected
        flag = "OK " if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{flag}] {text!r:48s} -> {got!r}  (expected {expected!r})")
    print()
    if failed:
        print(f"{failed} case(s) failed.")
        return 1
    print("All use-case heuristic checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
