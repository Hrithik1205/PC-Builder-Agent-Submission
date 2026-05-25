"""Quick smoke checks for the budget/off-topic/comparison features."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from src.agent.nodes import (
        _build_comparison_markdown,
        _extract_budget_range,
        _looks_off_topic,
    )

    print("=== Budget range parsing ===")
    cases = [
        "budget $1000-$1500 gaming PC",
        "between 800 and 1200 dollars",
        "I want a $1500 PC",
        "1440p gaming, budget $1500",
        "build for 1000 to 1500",
        "office PC for around 700",
        "gaming PC, 4K resolution, $2000 budget",
    ]
    for c in cases:
        lo, hi = _extract_budget_range(c)
        print(f"  {c!r:50s} -> min={lo}, max={hi}")

    print("\n=== Off-topic detection ===")
    cases_ot = [
        "what is the weather today",
        "tell me a joke please",
        "build me a gaming PC for $1500",
        "which RAM is best for AM5",
        "what's 2+2",
        "compare this with a $1000 budget",
        "make it cheaper",
        "translate this to French",
    ]
    for c in cases_ot:
        print(f"  off_topic={_looks_off_topic(c)!s:5s}  - {c!r}")

    print("\n=== Comparison markdown ===")
    prev = {
        "cpu": {"name": "Ryzen 5 7600", "price": 199.0},
        "video_card": {"name": "RTX 4060", "price": 299.0},
        "memory": {"name": "Corsair 16GB", "price": 79.0},
    }
    new = {
        "cpu": {"name": "Ryzen 7 7700X", "price": 299.0},
        "video_card": {"name": "RTX 4070", "price": 549.0},
        "memory": {"name": "Corsair 16GB", "price": 79.0},  # unchanged
    }
    md = _build_comparison_markdown(prev, new, 800, 1100)
    print(md)
    print()
    print("Smoke OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
