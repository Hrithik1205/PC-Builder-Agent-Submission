"""Verify the new budget-trim pass and 'less expensive X' heuristic.

Covers the bugs reported by the user:
1. $300 browsing build came in at $400 (over budget by 33%).
2. 'less CPU' did nothing.
3. 'less expensive cpu' actually RAISED the price (Intel $113 -> AMD $135).
4. Memory was 32 GB on a $300 budget; user wanted 16 GB.
5. Office build still shipped a discrete GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.nodes import (
    _heuristic_feedback,
    _fallback_plan,
    _PICKERS,
    _pick_relaxed,
    _budget_fill_pass,
    _budget_trim_pass,
    CATEGORY_ORDER,
)
from src.tools.search import search_components_impl  # noqa: F401


def total(build):
    return round(sum(float(c.get("price", 0) or 0)
                     for c in build.values() if c), 2)


def build_office_300():
    """Reproduce the user's $300 browsing build end-to-end (no LLM).

    Mirrors component_selector: strict picker, relaxed fallback, fill pass,
    then trim pass.
    """
    reqs = {
        "use_case": "office",
        "budget_usd": 300.0,
        "must_have": [],
        "nice_to_have": [],
    }
    plan = _fallback_plan(reqs)
    build = {}
    for cat in CATEGORY_ORDER:
        if build.get(cat):
            continue
        pick = _PICKERS[cat](plan, reqs, build)
        if not pick:
            pick = _pick_relaxed(cat, plan, reqs, build)
        if pick:
            build[cat] = pick
    build = _budget_fill_pass(build, plan, reqs)
    build = _budget_trim_pass(build, plan, reqs)
    return reqs, plan, build


print("== Test 1: $300 browsing build stays within budget ==")
reqs, plan, build = build_office_300()
tot = total(build)
print(f"  total = ${tot}  budget = ${reqs['budget_usd']}")
parts = {cat: (c.get("name"), c.get("price")) for cat, c in build.items()}
for cat, (name, price) in parts.items():
    print(f"    {cat:14s} {name:40s}  ${price}")
assert tot <= reqs["budget_usd"] * 1.05, \
    f"build is over budget: ${tot} > ${reqs['budget_usd']} (5% slack)"
assert "video_card" not in build, \
    "office build should NOT include a discrete GPU"
mem_gb = build.get("memory", {}).get("total_gb")
assert mem_gb is not None and mem_gb <= 16, \
    f"office build memory should be <= 16 GB, got {mem_gb} GB"
print("  PASS - within budget, no GPU, memory <= 16GB")


print("\n== Test 2: 'less CPU' is classified ==")
fb = _heuristic_feedback("less CPU")
print(f"  -> {fb}")
assert fb is not None and fb.get("intent") == "swap_part"
assert "cpu" in (fb.get("target_categories") or [])
print("  PASS")


print("\n== Test 3: 'less expensive cpu' targets only CPU and caps its price ==")
fb = _heuristic_feedback("select a less expensive cpu for me")
print(f"  -> {fb}")
assert fb is not None and fb.get("intent") == "swap_part"
assert fb.get("target_categories") == ["cpu"], \
    f"expected ['cpu'], got {fb.get('target_categories')}"
assert (fb.get("delta_constraints") or {}).get("price_lower_category") == "cpu"
print("  PASS - delta = price_lower_category=cpu")


print("\n== Test 4: 'make it cheaper' (no category) still shrinks budget ==")
fb = _heuristic_feedback("make it cheaper")
print(f"  -> {fb}")
assert fb is not None and fb.get("intent") == "swap_part"
assert (fb.get("delta_constraints") or {}).get("price_lower") is True
print("  PASS - delta = price_lower")


print("\n== Test 5: 'less ram' -> memory-scoped cheaper request ==")
fb = _heuristic_feedback("less ram please")
print(f"  -> {fb}")
assert fb is not None
assert fb.get("target_categories") == ["memory"]
assert (fb.get("delta_constraints") or {}).get("price_lower_category") == "memory"
print("  PASS")


print("\n== Test 6: 'lower price gpu' -> video_card-scoped cheaper request ==")
fb = _heuristic_feedback("lower price gpu please")
print(f"  -> {fb}")
assert fb is not None
assert fb.get("target_categories") == ["video_card"]
print("  PASS")


print("\n== Test 7: per-category ceiling actually picks a cheaper CPU ==")
# Build the office baseline.
reqs, plan, build = build_office_300()
old_cpu = build["cpu"]
old_price = float(old_cpu.get("price", 0))
print(f"  baseline CPU: {old_cpu.get('name')} @ ${old_price}")
# Apply 'less expensive cpu' delta manually and re-pick CPU.
reqs2 = dict(reqs)
reqs2["category_price_ceilings"] = {"cpu": round(old_price * 0.80, 2)}
build2 = dict(build)
build2.pop("cpu", None)
build2.pop("motherboard", None)  # socket may change
new_cpu = _PICKERS["cpu"](plan, reqs2, build2)
new_price = float(new_cpu.get("price", 0)) if new_cpu else None
print(f"  ceiling = ${reqs2['category_price_ceilings']['cpu']}")
print(f"  new CPU:  {new_cpu.get('name')} @ ${new_price}")
assert new_cpu is not None
assert new_price < old_price, \
    f"new CPU ({new_price}) should be cheaper than old ({old_price})"
print("  PASS - new CPU is cheaper")


print("\nAll budget-trim + 'less expensive X' assertions passed.")
