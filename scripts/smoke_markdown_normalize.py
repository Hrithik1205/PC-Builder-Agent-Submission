"""Unit-level test of _normalize_response_markdown."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.nodes import _normalize_response_markdown


def show(label: str, before: str) -> None:
    after = _normalize_response_markdown(before)
    print(f"=== {label} ===")
    print("--- BEFORE ---")
    print(before)
    print("--- AFTER ---")
    print(after)
    print()


def main() -> int:
    case1 = (
        "| Component | Part | Price |\n"
        "| --- | --- | --- |\n"
        "| CPU | i7-14700K | $400.00 |\n"
        "| Power Supply | Logisys PS480D2 | $30.99 |\n"
        "The total price of this build is $697.70, which is within your budget of $700.\n"
        "The Intel Core i7-14700K was chosen for its high core count.\n"
    )
    show("Missing blank line after table", case1)

    case2 = (
        "Some intro.\n"
        "### What changed vs your previous build\n"
        "- **CPU**: old -> new\n"
    )
    show("Missing blank line before heading", case2)

    case3 = (
        "The total price is *$697.70* which is fine.\n"
        "Italics around prices should be stripped.\n"
    )
    show("Italic-wrapped price", case3)

    # Idempotency: already-correct markdown should pass through unchanged.
    case4 = (
        "| A | B |\n| --- | --- |\n| 1 | 2 |\n\nGood prose here.\n"
    )
    after4 = _normalize_response_markdown(case4)
    assert after4 == case4, f"Idempotency broken: {after4!r}"
    print("Idempotency check OK.")

    # Verify the table now has a blank line:
    after1 = _normalize_response_markdown(case1)
    assert "Logisys PS480D2 | $30.99 |\n\nThe total" in after1, (
        f"Blank line not inserted: {after1!r}"
    )
    print("Table-blank-line insertion OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
