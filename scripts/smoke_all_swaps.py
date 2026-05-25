"""Comprehensive swap-type audit.

Confirms every documented swap phrasing produces the right intent and
delta_constraints. Covers ~50 phrasings across all swap categories.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.nodes import _heuristic_feedback

PASS, FAIL = 0, 0


def check(label: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label} -- {detail}")


def assert_intent(text: str, intent: str, must_have_key: str | None = None,
                  must_have_value=None):
    fb = _heuristic_feedback(text)
    got_intent = (fb or {}).get("intent")
    deltas = (fb or {}).get("delta_constraints") or {}
    ok = got_intent == intent
    if must_have_key is not None:
        ok = ok and (must_have_key in deltas)
        if must_have_value is not None:
            ok = ok and deltas.get(must_have_key) == must_have_value
    detail = f"intent={got_intent}, deltas={deltas}"
    check(f"{text!r:50s} -> {intent} ({must_have_key})", ok, detail)


# ---------------------------------------------------------------------------
# 1. CPU swaps - AMD <-> Intel, various verbs
# ---------------------------------------------------------------------------
print("== CPU brand swaps ==")
for txt in [
    "swap Intel CPU with AMD",
    "change CPU to Ryzen",
    "use AMD CPU",
    "give me an Intel processor",
    "switch to Intel core i7",
    "I want AMD CPU not Intel",
    "upgrade the cpu to Ryzen",
    "go with Intel",
    "make it an AMD build",
    "use an i5 CPU",
]:
    assert_intent(txt, "swap_part", "cpu_brand_preference")

# ---------------------------------------------------------------------------
# 2. GPU swaps - NVIDIA <-> AMD, various verbs
# ---------------------------------------------------------------------------
print("\n== GPU brand swaps ==")
for txt in [
    "swap Sparkle ROC Luna with NVIDIA",
    "swap to NVIDIA video card",
    "change my GPU to nvidia",
    "switch the graphics card to nvidia",
    "use a GeForce instead",
    "give me an RTX card",
    "swap GPU with AMD",
    "change to a Radeon",
    "switch to a Radeon",
    "give me an AMD graphics card",
]:
    assert_intent(txt, "swap_part", "gpu_brand_preference")

# ---------------------------------------------------------------------------
# 3. "Cheaper" - whole build (no number)
# ---------------------------------------------------------------------------
print("\n== Cheaper (whole build, no number) ==")
for txt in [
    "cheaper",
    "make it cheaper",
    "less expensive",
    "lower price",
    "lower cost",
]:
    assert_intent(txt, "swap_part", "price_lower", True)

# ---------------------------------------------------------------------------
# 4. "Cheaper X" - category-specific
# ---------------------------------------------------------------------------
print("\n== Cheaper X (category-specific) ==")
for txt, cat in [
    ("less expensive cpu", "cpu"),
    ("cheaper gpu", "video_card"),
    ("less ram", "memory"),
    ("smaller storage", "storage"),
    ("less expensive case", "case"),
    ("cheaper power supply", "power_supply"),
]:
    assert_intent(txt, "swap_part", "price_lower_category", cat)

# ---------------------------------------------------------------------------
# 5. "Quieter" / acoustic swap
# ---------------------------------------------------------------------------
print("\n== Quieter ==")
for txt in [
    "quieter please",
    "silent build",
    "less noisy",
    "quiet PC",
    "noiseless cooler",
]:
    assert_intent(txt, "swap_part", "noise_preference")

# ---------------------------------------------------------------------------
# 6. "More storage" / capacity changes
# ---------------------------------------------------------------------------
print("\n== Storage capacity ==")
for txt in [
    "more storage",
    "bigger SSD",
    "larger storage",
    "more space",
]:
    assert_intent(txt, "swap_part", "storage_capacity_gte")

# ---------------------------------------------------------------------------
# 7. "More RAM"
# ---------------------------------------------------------------------------
print("\n== Memory capacity ==")
for txt in [
    "more ram",
    "more memory",
    "bigger ram",
    "larger ram",
]:
    assert_intent(txt, "swap_part", "memory_gte")

# ---------------------------------------------------------------------------
# 8. Budget changes (absolute)
# ---------------------------------------------------------------------------
print("\n== Budget changes (absolute) ==")
for txt, expected in [
    ("increase budget to $2000",         2000),
    ("reduce budget to $500",            500),
    ("lower the budget to $600",         600),
    ("raise to $1500",                   1500),
    ("set budget to $900",               900),
    ("budget $700",                      700),
]:
    fb = _heuristic_feedback(txt) or {}
    deltas = fb.get("delta_constraints") or {}
    got = deltas.get("budget_usd")
    check(
        f"{txt!r:40s} -> budget=${got}",
        fb.get("intent") == "change_budget" and got == expected,
        f"intent={fb.get('intent')}, deltas={deltas}",
    )

# ---------------------------------------------------------------------------
# 9. Budget changes (relative)
# ---------------------------------------------------------------------------
print("\n== Budget changes (relative) ==")
for txt, mult in [
    ("double my budget",          2.0),
    ("triple the budget",         3.0),
    ("halve the budget",          0.5),
    ("cut my budget in half",     0.5),
    ("1.5x the budget",           1.5),
]:
    fb = _heuristic_feedback(txt) or {}
    deltas = fb.get("delta_constraints") or {}
    got = deltas.get("budget_multiplier")
    check(
        f"{txt!r:35s} -> multiplier={got}",
        fb.get("intent") == "change_budget" and got == mult,
        f"intent={fb.get('intent')}, deltas={deltas}",
    )

# ---------------------------------------------------------------------------
# 10. Comparison
# ---------------------------------------------------------------------------
print("\n== Comparison ==")
for txt in [
    "compare with $900 budget",
    "compare to a $2000 build",
    "what if I had $2000",
    "what could I get for $1500",
    "show me a build for $800",
]:
    assert_intent(txt, "compare_builds")

# ---------------------------------------------------------------------------
# 11. Approval
# ---------------------------------------------------------------------------
print("\n== Approval ==")
for txt in [
    "approve",
    "approved",
    "looks good",
    "looks great",
    "perfect",
    "ship it",
    "yes",
    "thanks",
    "go with this",
]:
    assert_intent(txt, "approve")


print(f"\n{'='*70}")
print(f"All-swaps audit: {PASS} passed, {FAIL} failed")
print(f"{'='*70}")
sys.exit(0 if FAIL == 0 else 1)
