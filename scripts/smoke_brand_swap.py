"""Verify CPU brand swap works end-to-end (heuristic + picker)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.nodes import (
    _heuristic_feedback,
    _heuristic_requirements,
    _pick_cpu,
    _fallback_plan,
)


def assert_eq(actual, expected, label):
    if actual == expected:
        print(f"  OK  {label}: {actual!r}")
    else:
        print(f"  FAIL {label}: expected {expected!r}, got {actual!r}")
        raise SystemExit(1)


print("== Heuristic feedback - CPU brand swap ==")
for phrase in [
    "i want AMD cpu not intel",
    "swap Intel cpu with AMD",
    "use ryzen instead of intel",
    "change to AMD",
    "give me an intel processor",
    "switch to AMD please",
]:
    fb = _heuristic_feedback(phrase)
    print(f"\n  Input: {phrase!r}")
    assert fb is not None, f"heuristic should have classified: {phrase!r}"
    assert_eq(fb.get("intent"), "swap_part", "intent")
    cats = set(fb.get("target_categories") or [])
    expected_cats = {"cpu", "motherboard", "memory"}
    assert cats == expected_cats, f"expected {expected_cats}, got {cats}"
    print(f"  OK  target_categories: {sorted(cats)}")
    deltas = fb.get("delta_constraints") or {}
    brand = deltas.get("cpu_brand_preference")
    print(f"  brand preference: {brand}")
    assert brand in ("amd", "intel"), f"expected amd/intel, got {brand!r}"


print("\n== Initial requirements - brand detection ==")
for phrase, expected_cpu, expected_gpu in [
    ("I want an AMD gaming PC with $1500 budget", "amd", None),
    ("Intel build please, $1000", "intel", None),
    ("Gaming PC with Ryzen and RTX 4070, $2000", "amd", "nvidia"),
    ("budget $800 for office", None, None),
    ("$1500 build with Radeon", None, "amd"),
]:
    base = {"use_case": "general", "must_have": [], "nice_to_have": []}
    out = _heuristic_requirements(phrase, base)
    print(f"\n  Input: {phrase!r}")
    print(f"  -> cpu_brand={out.get('cpu_brand_preference')}, "
          f"gpu_brand={out.get('gpu_brand_preference')}")
    assert out.get("cpu_brand_preference") == expected_cpu, \
        f"expected cpu={expected_cpu}, got {out.get('cpu_brand_preference')}"
    assert out.get("gpu_brand_preference") == expected_gpu, \
        f"expected gpu={expected_gpu}, got {out.get('gpu_brand_preference')}"
    print("  OK")


print("\n== Picker - brand actually filters catalog ==")
reqs_amd = {
    "use_case": "office",
    "budget_usd": 800,
    "cpu_brand_preference": "amd",
}
plan = _fallback_plan(reqs_amd)
cpu = _pick_cpu(plan, reqs_amd, {})
print(f"\n  AMD request -> {cpu.get('name')}")
assert cpu is not None and cpu.get("name", "").lower().startswith("amd"), \
    f"expected AMD CPU, got {cpu}"
print("  OK")

reqs_intel = {
    "use_case": "office",
    "budget_usd": 800,
    "cpu_brand_preference": "intel",
}
plan = _fallback_plan(reqs_intel)
cpu = _pick_cpu(plan, reqs_intel, {})
print(f"  Intel request -> {cpu.get('name')}")
assert cpu is not None and cpu.get("name", "").lower().startswith("intel"), \
    f"expected Intel CPU, got {cpu}"
print("  OK")

print("\nAll brand-swap assertions passed.")
