"""Verify a wide vocabulary of swap/change/update verbs across every
component category is correctly classified as a swap_part intent.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.nodes import _heuristic_feedback


CASES = [
    # ---- CPU brand swap, many verbs ----
    ("swap Intel cpu with AMD",          "swap_part", {"cpu", "motherboard", "memory"}, "amd"),
    ("change to AMD please",             "swap_part", {"cpu", "motherboard", "memory"}, "amd"),
    ("update the cpu to ryzen",          "swap_part", {"cpu", "motherboard", "memory"}, "amd"),
    ("upgrade the processor to AMD",     "swap_part", {"cpu", "motherboard", "memory"}, "amd"),
    ("replace intel with amd",           "swap_part", {"cpu", "motherboard", "memory"}, "amd"),
    ("switch to intel processor",        "swap_part", {"cpu", "motherboard", "memory"}, "intel"),
    ("go with intel",                    "swap_part", {"cpu", "motherboard", "memory"}, "intel"),
    ("use ryzen instead",                "swap_part", {"cpu", "motherboard", "memory"}, "amd"),
    # ---- Generic component swaps ----
    ("change the GPU please",            "swap_part", {"video_card"}, None),
    ("update the graphics card",         "swap_part", {"video_card"}, None),
    ("upgrade the storage",              "swap_part", {"storage"}, None),
    ("replace the ssd",                  "swap_part", {"storage"}, None),
    ("swap the motherboard",             "swap_part", {"motherboard"}, None),
    ("update the mobo",                  "swap_part", {"motherboard"}, None),
    ("change the power supply",          "swap_part", {"power_supply"}, None),
    ("swap psu",                         "swap_part", {"power_supply"}, None),
    ("update the case",                  "swap_part", {"case"}, None),
    ("change the cpu cooler",            "swap_part", {"cpu_cooler"}, None),
    ("upgrade the heatsink",             "swap_part", {"cpu_cooler"}, None),
    ("change ram",                       "swap_part", {"memory"}, None),
    ("update memory",                    "swap_part", {"memory"}, None),
    ("please change the CPU cooler",     "swap_part", {"cpu_cooler"}, None),
    # ---- Special-cased intents that should still fire correctly ----
    ("more storage please",              "swap_part", {"storage"}, None),
    ("make it quieter",                  "swap_part", {"cpu_cooler", "case"}, None),
    ("increase budget to $1500",         "change_budget", set(), None),
    ("make it cheaper",                  "swap_part", set(), None),
]


def main() -> int:
    fails = 0
    for text, want_intent, want_cats, want_cpu_brand in CASES:
        fb = _heuristic_feedback(text)
        if fb is None:
            print(f"  FAIL  {text!r}: heuristic returned None")
            fails += 1
            continue
        got_intent = fb.get("intent")
        got_cats = set(fb.get("target_categories") or [])
        got_brand = (fb.get("delta_constraints") or {}).get("cpu_brand_preference")
        ok = (
            got_intent == want_intent
            and got_cats == want_cats
            and got_brand == want_cpu_brand
        )
        mark = "OK  " if ok else "FAIL"
        if not ok:
            fails += 1
        print(
            f"  {mark}  {text!r:40s} -> "
            f"intent={got_intent}, cats={sorted(got_cats)}, "
            f"cpu_brand={got_brand}"
        )
    if fails:
        print(f"\n{fails} failure(s).")
        return 1
    print(f"\nAll {len(CASES)} swap-verb cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
