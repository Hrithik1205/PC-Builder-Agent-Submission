"""Regression test for GPU brand swap.

Cases covered:
 1. "swap Sparkle ROC Luna with NVIDIA"   - brand follows a card name, no
    explicit "gpu/graphics" word. Must still route to a GPU swap.
 2. "swap to NVIDIA video card"            - explicit phrasing.
 3. "give me an AMD graphics card"          - alt swap verb + brand + kw.
 4. "switch to a Radeon"                   - swap verb + brand only.
 5. The actual GPU picker must NOT pick a 2011 GTX 570 - it should pick a
    modern card (>= 4 GB VRAM).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.nodes import _heuristic_feedback, _pick_video_card

PASS, FAIL = 0, 0


def check(label: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label} -- {detail}")


# ---- Heuristic detection ----
print("== GPU brand swap detection ==")

heuristic_cases = [
    # (text, expected_brand)
    ("swap Sparkle ROC Luna with NVIDIA",  "nvidia"),
    ("swap to NVIDIA video card",          "nvidia"),
    ("change my GPU to nvidia",            "nvidia"),
    ("switch the graphics card to nvidia", "nvidia"),
    ("use a GeForce instead",              "nvidia"),
    ("give me an RTX card",                "nvidia"),
    ("swap GPU with AMD",                  "amd"),
    ("change to a Radeon",                 "amd"),
    ("switch to a Radeon",                 "amd"),
    ("give me an AMD graphics card",       "amd"),
]
for text, expected in heuristic_cases:
    fb = _heuristic_feedback(text)
    got = ((fb or {}).get("delta_constraints") or {}).get("gpu_brand_preference")
    check(
        f"detect {text!r} -> {got}",
        got == expected,
        f"expected {expected}",
    )

# ---- Picker quality - no 2011 cards in a modern build ----
print("\n== GPU picker (no ancient cards) ==")

plan = {
    "budget_allocation": {"video_card": 365},  # ~32% of $1137
    "platform_preference": "any",
}

for brand in ("nvidia", "amd"):
    reqs = {
        "use_case": "gaming",
        "budget_usd": 1137.0,
        "gpu_brand_preference": brand,
        "must_have": [], "nice_to_have": [],
    }
    pick = _pick_video_card(plan, reqs, build={})
    if not pick:
        check(f"picker returned a card for brand={brand}", False, "got None")
        continue

    name = pick.get("name", "")
    chipset = pick.get("chipset", "")
    vram = float(pick.get("memory") or 0)
    price = float(pick.get("price") or 0)
    print(f"  -> {brand}: {name!r}  chipset={chipset!r} vram={vram} price=${price}")

    check(
        f"{brand}: picked card has VRAM >= 4 GB ({vram})",
        vram >= 4,
        f"got {vram}",
    )
    expect = "GeForce" if brand == "nvidia" else "Radeon"
    check(
        f"{brand}: chipset contains {expect!r}",
        expect.lower() in chipset.lower(),
        f"got {chipset!r}",
    )
    # The previous TDP-DESC sort produced cards like the GTX 570 / 970.
    bad_substrings = ["GTX 570", "GTX 580", "GTX 670", "GTX 680", "GTX 770", "GTX 970"]
    for bad in bad_substrings:
        check(
            f"{brand}: did NOT pick ancient {bad}",
            bad.lower() not in chipset.lower(),
            f"picked {chipset!r}",
        )

print(f"\n{'='*60}")
print(f"GPU swap regression: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")
sys.exit(0 if FAIL == 0 else 1)
