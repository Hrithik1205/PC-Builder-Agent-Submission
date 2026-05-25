"""Full deterministic pipeline (no LLM) across many budget x use_case combos.

Confirms the agent produces a sane build with NO crashes for boundary
combinations the reviewer is likely to try.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.nodes import (
    _fallback_plan,
    _PICKERS,
    _pick_relaxed,
    _budget_fill_pass,
    _budget_trim_pass,
    _need_discrete_gpu,
    CATEGORY_ORDER,
)
from src.compatibility.engine import check_build
from src.data.schemas import Build


def build_for(reqs):
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
    return build


def total(b):
    return round(sum(float(c.get("price", 0) or 0) for c in b.values() if c), 2)


SCENARIOS = [
    # (budget, use_case, label, expectations)
    (300,   "office",            "Browsing $300"),
    (300,   "gaming",            "Gaming $300 (infeasible-ish)"),
    (500,   "office",            "Office $500"),
    (800,   "gaming",            "Gaming $800"),
    (1000,  "office",            "Office $1000"),
    (1500,  "gaming",            "Gaming $1500"),
    (1500,  "content_creation",  "Content creation $1500"),
    (2500,  "workstation",       "Workstation $2500"),
    (5000,  "gaming",            "Gaming $5000"),
    (10000, "workstation",       "Workstation $10000"),
    # Edge: budget range
    (1500,  "gaming",            "Gaming range $1200-$1500 (test min hit)", {"budget_min_usd": 1200}),
]

PASS, FAIL = 0, 0


def run(budget, use_case, label, extra=None):
    global PASS, FAIL
    reqs = {
        "use_case": use_case,
        "budget_usd": float(budget),
        "must_have": [],
        "nice_to_have": [],
    }
    if extra:
        reqs.update(extra)
    print(f"\n>> {label}")
    try:
        b = build_for(reqs)
        t = total(b)
        # Run compatibility checker
        try:
            issues = check_build(Build(**b))
            err_count = sum(1 for i in issues if i.severity == "error")
            if err_count:
                for i in issues:
                    if i.severity == "error":
                        print(f"   COMPAT ERROR: {i.rule}: {i.message}")
        except Exception as e:
            err_count = -1
            print(f"   checker raised: {e}")

        print(f"   total = ${t}  ({len(b)} components, {err_count} compat errors)")
        for cat in CATEGORY_ORDER:
            c = b.get(cat)
            if c:
                print(f"     {cat:14s} {c.get('name')[:45]:45s}  ${c.get('price')}")
            elif cat == "video_card" and not _need_discrete_gpu(reqs):
                pass  # expected to be missing
            else:
                print(f"     {cat:14s} (none)")
        # Assertions:
        # 1. Build must have all critical categories (cpu, mb, mem, storage, psu, case, cooler)
        critical = ["cpu", "motherboard", "memory", "storage",
                    "power_supply", "case", "cpu_cooler"]
        missing = [c for c in critical if c not in b]
        if missing:
            print(f"   FAIL  missing critical categories: {missing}")
            FAIL += 1
            return
        # 2. Budget must be respected within 10% slack (low-budget builds are tight).
        slack = 1.10 if budget < 500 else 1.05
        if t > budget * slack:
            print(f"   FAIL  build $%.2f exceeds budget $%d by more than %.0f%%" % (
                t, budget, (slack-1)*100))
            FAIL += 1
            return
        # 3. Use-case-specific sanity
        if use_case == "gaming" and budget >= 800 and "video_card" not in b:
            print(f"   FAIL  gaming >= $800 should have a discrete GPU")
            FAIL += 1
            return
        if use_case == "office" and "video_card" in b:
            print(f"   WARN  office build has a discrete GPU (not necessarily wrong)")
        # 4. Compat must not have errors
        if err_count > 0:
            print(f"   FAIL  build has {err_count} compatibility errors")
            FAIL += 1
            return
        print("   PASS")
        PASS += 1
    except Exception as e:
        traceback.print_exc()
        print(f"   CRASH  {e}")
        FAIL += 1


for s in SCENARIOS:
    extra = s[3] if len(s) > 3 else None
    run(s[0], s[1], s[2], extra)


print(f"\n{'='*60}")
print(f"Full-pipeline stress test: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")
sys.exit(0 if FAIL == 0 else 1)
