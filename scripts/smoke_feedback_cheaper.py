"""Regression test for: bare 'cheaper' must REDUCE (not raise) the budget.

User reported: budget was $400, then they typed "cheaper" and the agent
silently set the budget to $1000. Root cause: the LLM hallucinated a
budget number because the few-shot example included "cheaper, around $1000".

This test exercises the real feedback_handler with a stub state that has
an existing build at $400 budget, and confirms the new budget is LOWER
than the original (and the heuristic path is used).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage

from src.agent.nodes import feedback_handler


def make_state(user_msg: str, budget: float, build_total: float) -> dict:
    """Mock state with an existing build at the given budget/total."""
    # A minimal sketch of a build whose total = build_total.
    # Component names don't matter for the heuristic-first path.
    fake_cpu_price = build_total * 0.35
    fake_mem_price = build_total * 0.10
    rest = build_total - fake_cpu_price - fake_mem_price
    build = {
        "cpu":          {"name": "AMD Ryzen 5 5600G",         "price": round(fake_cpu_price, 2)},
        "motherboard":  {"name": "ASRock A520M-HDV",           "price": round(rest * 0.20, 2)},
        "memory":       {"name": "Kingston KVR21R15D4 16 GB",  "price": round(fake_mem_price, 2)},
        "storage":      {"name": "Intel 660p",                 "price": round(rest * 0.25, 2)},
        "power_supply": {"name": "Logisys PS480D2",            "price": round(rest * 0.10, 2)},
        "case":         {"name": "Cooler Master MasterBox Q300L", "price": round(rest * 0.12, 2)},
        "cpu_cooler":   {"name": "Iceberg Thermal IceFLOE T95", "price": round(rest * 0.03, 2)},
    }
    return {
        "messages": [HumanMessage(content=user_msg)],
        "requirements": {
            "use_case": "office",
            "budget_usd": float(budget),
            "must_have": [],
            "nice_to_have": [],
        },
        "build": build,
        "plan": None,
    }


PASS, FAIL = 0, 0


def check(label: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label} -- {detail}")


# Scenario: $400 budget, build at $380.93, user types "cheaper"
state = make_state("cheaper", budget=400, build_total=380.93)
result = feedback_handler(state)
reqs = result.get("requirements") or {}
new_budget = reqs.get("budget_usd")

print(f"Input:    budget=$400, build_total=$380.93, user='cheaper'")
print(f"Result:   new_budget=${new_budget}")
print(f"Expected: a value strictly LESS than $400 (likely ~$304)")
print()

check(
    "feedback_handler produced a new budget",
    new_budget is not None,
    f"got {new_budget}",
)
check(
    f"new budget (${new_budget}) is STRICTLY LESS than original $400",
    new_budget is not None and new_budget < 400,
    f"got {new_budget}, expected < 400",
)
check(
    f"new budget did NOT jump to $1000",
    new_budget != 1000,
    f"got {new_budget}",
)
check(
    "new budget is in plausible range ($100-$400)",
    new_budget is not None and 100 <= new_budget <= 400,
    f"got {new_budget}",
)

# Same regression: $700 budget, "make it cheaper" must drop budget.
state2 = make_state("make it cheaper", budget=700, build_total=680)
result2 = feedback_handler(state2)
nb2 = (result2.get("requirements") or {}).get("budget_usd")
print(f"\nInput:    budget=$700, build_total=$680, user='make it cheaper'")
print(f"Result:   new_budget=${nb2}")
check(
    f"'make it cheaper' on $700 budget reduces below $700",
    nb2 is not None and nb2 < 700,
    f"got {nb2}",
)

# Make sure "cheaper, around $1000" (explicit number) STILL works as a budget change
state3 = make_state("cheaper, around $1000", budget=1500, build_total=1480)
result3 = feedback_handler(state3)
nb3 = (result3.get("requirements") or {}).get("budget_usd")
print(f"\nInput:    budget=$1500, user='cheaper, around $1000'")
print(f"Result:   new_budget=${nb3}")
check(
    "'cheaper, around $1000' DOES set budget to $1000",
    nb3 is not None and abs(nb3 - 1000) < 50,
    f"got {nb3}",
)


print(f"\n{'='*60}")
print(f"feedback 'cheaper' regression test: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")
sys.exit(0 if FAIL == 0 else 1)
