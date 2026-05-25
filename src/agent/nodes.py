"""Agent node functions for the LangGraph.

Each node:
- reads the current `AgentState`,
- emits one structured log entry per invocation,
- returns a state delta (only the keys it wants to update).

We deliberately avoid mutating the input dict; LangGraph merges the returned
dict back into the canonical state.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.guards import filter_real_components, validate_user_message
from src.agent.prompts import (
    CRITIC_SYSTEM,
    FEEDBACK_SYSTEM,
    PLANNER_SYSTEM,
    REQUIREMENT_GATHERER_FEWSHOTS,
    REQUIREMENT_GATHERER_SYSTEM,
    RESPONDER_SYSTEM,
    SELECTOR_SYSTEM,
)
from src.agent.state import AgentState
from src.compatibility.engine import check_build, has_errors, summarize_issues
from src.config import get_settings
from src.data.schemas import Build, Issue
from src.llm.client import invoke_with_retry
from src.logging_setup import get_logger
from src.tools.registry import get_tools
from src.tools.search import search_components_impl


log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_user_text(state: AgentState) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


def _parse_json_safely(text: str) -> Dict[str, Any] | None:
    """Tolerant JSON extraction: tries the whole string, then a {...} block."""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the largest {...} substring
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None


def _few_shot_messages(fewshots: List[Dict[str, Any]]):
    msgs = []
    for ex in fewshots:
        msgs.append(HumanMessage(content=ex["user"]))
        msgs.append(AIMessage(content=json.dumps(ex["assistant"], indent=2)))
    return msgs


def _extract_budget_range(user_text: str) -> tuple[float | None, float | None]:
    """Parse `(min, max)` budget. Returns `(None, max)` if only a single value.

    Avoids confusing display resolutions (1440p, 1080p, 4K) with dollars.
    """
    text = user_text.replace(",", "")
    # Patterns that capture an explicit range.
    range_patterns = [
        r"\$?\s*(\d{2,5})\s*(?:-|to|and)\s*\$?\s*(\d{2,5})",
        r"between\s*\$?\s*(\d{2,5})\s*(?:and|to)\s*\$?\s*(\d{2,5})",
    ]
    for pat in range_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            if lo > hi:
                lo, hi = hi, lo
            if 150 <= lo <= 25000 and 150 <= hi <= 25000 and hi - lo < hi:
                return lo, hi
    # Single-value patterns (preferred over loose digit matches).
    single_patterns = [
        r"\$\s*(\d{2,5})\b",
        r"(?:for|budget)\s*\$?\s*(\d{2,5})\b",
        r"(\d{2,5})\s*(?:usd|dollars?)\b",
    ]
    for pat in single_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if 150 <= val <= 25000:
                return None, val
    # Loose fallback: any 3-5 digit number not followed by 'p' (skip 1440p, 4K).
    for m in re.finditer(r"(?<!\d)(\d{3,5})(?!p\b)", text, re.IGNORECASE):
        val = float(m.group(1))
        if 150 <= val <= 25000:
            return None, val
    return None, None


def _extract_budget_usd(user_text: str) -> float | None:
    """Back-compat wrapper: return only the upper bound."""
    _, hi = _extract_budget_range(user_text)
    return hi


# Keywords that strongly indicate a non-PC topic (used as a heuristic
# off-topic deflection when the LLM is unavailable or returned no flag).
_OFF_TOPIC_HINTS = re.compile(
    r"\b(weather|forecast|recipe|cook(?:ing|ery)?|joke|poem|story|"
    r"translat\w*|who won|movie|song|lyrics|capital of|"
    r"president|prime minister|stock price)\b|\b2\s*\+\s*2\b",
    re.IGNORECASE,
)
# Cheap positive signal that we're discussing PC hardware.
_ON_TOPIC_HINTS = re.compile(
    r"\b(pc|computer|build|gaming|workstation|cpu|gpu|ram|memory|"
    r"motherboard|psu|ssd|nvme|case|cooler|tower|amd|intel|nvidia|"
    r"radeon|geforce|ryzen|core\s?ultra|am[45]|lga\s?\d|ddr[45]|"
    r"1080p|1440p|4k|fps|render|stream|edit|davinci|premiere)\b",
    re.IGNORECASE,
)


def _looks_off_topic(text: str) -> bool:
    """Conservative heuristic - only flags clearly off-topic short messages."""
    if not text or len(text) > 300:
        return False
    if _ON_TOPIC_HINTS.search(text):
        return False
    return bool(_OFF_TOPIC_HINTS.search(text))


OFF_TOPIC_REPLY = (
    "I can't help with that - I can only assist you with building a PC. "
    "Tell me about the kind of PC you'd like and your budget, and I'll "
    "design a compatible build for you."
)


def _heuristic_requirements(user_text: str, base: Dict[str, Any]) -> Dict[str, Any]:
    """Fill gaps when the LLM is down and could not parse requirements."""
    t = user_text.lower()
    if base.get("budget_usd") is None:
        lo, hi = _extract_budget_range(user_text)
        if hi is not None:
            base["budget_usd"] = hi
        if lo is not None and base.get("budget_min_usd") is None:
            base["budget_min_usd"] = lo
    if base.get("use_case") == "general":
        if any(w in t for w in ("gaming", "game", "1440p", "1080p", "fps")):
            base["use_case"] = "gaming"
        elif any(w in t for w in ("office", "word", "excel", "zoom", "browse")):
            base["use_case"] = "office"
        elif any(w in t for w in ("video edit", "davinci", "premiere", "render")):
            base["use_case"] = "content_creation"
    if base.get("budget_usd") and base.get("use_case") != "general":
        base["confidence"] = "high"
        base["clarifying_questions"] = []
    return base


# ---------------------------------------------------------------------------
# 1. requirement_gatherer
# ---------------------------------------------------------------------------

def requirement_gatherer(state: AgentState) -> Dict[str, Any]:
    t0 = time.time()
    user_text = _last_user_text(state)

    ok, reason = validate_user_message(user_text)
    if not ok:
        log.warning("node.gather.blocked", reason=reason)
        return {
            "final_response": reason,
            "mode": "respond",
        }

    # Fast-path heuristic off-topic deflection. Saves an LLM call for the
    # obvious cases ("what's the weather?", "tell me a joke", etc.).
    if _looks_off_topic(user_text):
        log.info("node.gather.off_topic_heuristic")
        return {"final_response": OFF_TOPIC_REPLY, "mode": "respond"}

    msgs = [SystemMessage(content=REQUIREMENT_GATHERER_SYSTEM)]
    msgs.extend(_few_shot_messages(REQUIREMENT_GATHERER_FEWSHOTS))
    history = state.get("messages", [])
    msgs.extend(history)

    ai = invoke_with_retry(msgs, temperature=0.0)
    parsed = _parse_json_safely(ai.content) or {}

    # LLM-flagged off-topic - return the standard refusal.
    if parsed.get("is_on_topic") is False:
        log.info("node.gather.off_topic_llm")
        return {"final_response": OFF_TOPIC_REPLY, "mode": "respond"}

    requirements = {
        "is_on_topic": parsed.get("is_on_topic", True),
        "use_case": parsed.get("use_case", "general"),
        "budget_usd": parsed.get("budget_usd"),
        "budget_min_usd": parsed.get("budget_min_usd"),
        "budget_flexible": bool(parsed.get("budget_flexible", False)),
        "noise_preference": parsed.get("noise_preference"),
        "form_factor_preference": parsed.get("form_factor_preference", "any"),
        "os_needed": bool(parsed.get("os_needed", False)),
        "peripherals_needed": parsed.get("peripherals_needed", []) or [],
        "must_have": parsed.get("must_have", []) or [],
        "nice_to_have": parsed.get("nice_to_have", []) or [],
        "confidence": parsed.get("confidence", "low"),
        "clarifying_questions": parsed.get("clarifying_questions", []) or [],
    }
    # When the LLM is unavailable (or returned junk), parse budget/use-case from text.
    if not parsed or "unable to reach the language model" in (ai.content or "").lower():
        requirements = _heuristic_requirements(user_text, requirements)
    else:
        # Backfill budget from text if the LLM missed it / didn't see a range.
        lo, hi = _extract_budget_range(user_text)
        if requirements.get("budget_usd") is None and hi is not None:
            requirements["budget_usd"] = hi
        if requirements.get("budget_min_usd") is None and lo is not None:
            requirements["budget_min_usd"] = lo

    log.info(
        "node.gather.done",
        confidence=requirements["confidence"],
        budget=requirements["budget_usd"],
        budget_min=requirements.get("budget_min_usd"),
        use_case=requirements["use_case"],
        elapsed_ms=int((time.time() - t0) * 1000),
    )

    # If confidence is low and we have clarifying questions, ask the user.
    if requirements["confidence"] == "low" and requirements["clarifying_questions"]:
        q_text = "I need a little more info before I can suggest a build:\n\n"
        q_text += "\n".join(f"- {q}" for q in requirements["clarifying_questions"][:3])
        return {
            "requirements": requirements,
            "final_response": q_text,
            "mode": "respond",
        }

    return {"requirements": requirements, "mode": "plan"}


# ---------------------------------------------------------------------------
# 2. planner (chain-of-thought)
# ---------------------------------------------------------------------------

def planner(state: AgentState) -> Dict[str, Any]:
    t0 = time.time()
    reqs = state.get("requirements") or {}

    msgs = [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=(
            "Requirements (JSON):\n"
            f"{json.dumps(reqs, indent=2)}\n\n"
            "Now produce the plan JSON."
        )),
    ]
    ai = invoke_with_retry(msgs, temperature=0.2)
    plan = _parse_json_safely(ai.content)

    if not plan or "budget_allocation" not in plan:
        log.warning("node.plan.fallback_default", raw=str(ai.content)[:200])
        plan = _fallback_plan(reqs)

    log.info(
        "node.plan.done",
        tier=plan.get("performance_tier"),
        platform=plan.get("platform_preference"),
        warnings=plan.get("warnings", []),
        elapsed_ms=int((time.time() - t0) * 1000),
    )

    return {"plan": plan, "mode": "select", "selector_attempts": 0, "critique_attempts": 0}


def _fallback_plan(reqs: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic default if the LLM planner output is unparseable."""
    budget = float(reqs.get("budget_usd") or 1500)
    use_case = reqs.get("use_case", "general")
    is_gaming = use_case == "gaming"
    is_creator = use_case in ("content_creation", "workstation")
    allocation = {
        "cpu": budget * (0.18 if is_gaming else 0.25),
        "motherboard": budget * 0.10,
        "memory": budget * (0.08 if is_gaming else 0.12),
        "video_card": budget * (0.35 if is_gaming else 0.15),
        "storage": budget * 0.08,
        "power_supply": budget * 0.06,
        "case": budget * 0.06,
        "cpu_cooler": budget * 0.04,
    }
    if is_creator:
        allocation["memory"] = budget * 0.15
    return {
        "reasoning": "Fallback heuristic allocation (LLM plan was unparseable).",
        "performance_tier": "mainstream",
        "platform_preference": "AM5",
        "budget_allocation": allocation,
        "constraints": [],
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# 3. component_selector (deterministic, uses search tool directly)
# ---------------------------------------------------------------------------

CATEGORY_ORDER = [
    "cpu", "motherboard", "memory", "video_card",
    "storage", "power_supply", "case", "cpu_cooler",
]


def _need_discrete_gpu(reqs: Dict[str, Any]) -> bool:
    use_case = reqs.get("use_case")
    if use_case in ("gaming", "content_creation", "workstation"):
        return True
    if any("gpu" in (h.lower() if isinstance(h, str) else "") for h in reqs.get("must_have", [])):
        return True
    return False


def _pick_cpu(plan: Dict[str, Any], reqs: Dict[str, Any], build: Dict[str, Any]) -> Dict[str, Any] | None:
    budget = plan["budget_allocation"]["cpu"] * 1.15  # 15% per-category slack
    platform = (plan.get("platform_preference") or "any").upper()
    need_igpu = not _need_discrete_gpu(reqs)
    filters: Dict[str, Any] = {"price_lte": budget, "price_gte": 50}
    if platform != "ANY":
        filters["socket"] = platform
    if need_igpu:
        filters["has_integrated_graphics"] = True
    # Sort by core_count desc within budget - "the most cores you can afford"
    results = search_components_impl("cpu", filters=filters,
                                     sort_by="core_count", ascending=False, top_k=10)
    if not results and platform != "ANY":
        # Relax platform constraint
        filters.pop("socket", None)
        results = search_components_impl("cpu", filters=filters,
                                         sort_by="core_count", ascending=False, top_k=10)
    if not results:
        return None
    # Pick the highest-core, then highest-boost option that has a known socket
    for r in results:
        if r.get("socket"):
            return r
    return results[0]


def _pick_motherboard(plan, reqs, build) -> Dict[str, Any] | None:
    if not build.get("cpu"):
        return None
    socket = build["cpu"].get("socket")
    budget = plan["budget_allocation"]["motherboard"] * 1.2
    filters: Dict[str, Any] = {"price_lte": budget, "price_gte": 50, "socket": socket}
    form_pref = (reqs.get("form_factor_preference") or "any").lower()
    if form_pref == "mini_itx":
        filters["form_factor"] = "Mini ITX"
    elif form_pref == "micro_atx":
        filters["form_factor_in"] = ["Micro ATX", "Mini ITX"]
    elif form_pref == "atx":
        filters["form_factor_in"] = ["ATX", "Micro ATX"]
    results = search_components_impl("motherboard", filters=filters,
                                     sort_by="price", ascending=True, top_k=10)
    if not results:
        # Relax: drop form factor pref
        for k in list(filters):
            if k.startswith("form_factor"):
                filters.pop(k)
        results = search_components_impl("motherboard", filters=filters,
                                         sort_by="price", ascending=True, top_k=10)
    return results[0] if results else None


def _pick_memory(plan, reqs, build) -> Dict[str, Any] | None:
    if not build.get("motherboard"):
        return None
    ddr = build["motherboard"].get("ddr_gen")
    budget = plan["budget_allocation"]["memory"] * 1.2
    target_gb = 32 if reqs.get("use_case") in ("content_creation", "workstation") else 16
    # Allow >= target_gb up to max board capacity
    max_mem = build["motherboard"].get("max_memory") or 64
    filters: Dict[str, Any] = {
        "price_lte": budget,
        "ddr_gen": ddr,
        "total_gb_gte": target_gb,
        "total_gb_lte": max_mem,
    }
    # Slot constraint
    slots = build["motherboard"].get("memory_slots")
    if slots:
        filters["module_count_lte"] = slots
    results = search_components_impl("memory", filters=filters,
                                     sort_by="total_gb", ascending=False, top_k=10)
    if not results:
        filters.pop("total_gb_gte", None)
        results = search_components_impl("memory", filters=filters,
                                         sort_by="total_gb", ascending=False, top_k=10)
    if not results:
        return None
    # Prefer the kit with most total_gb within budget, then lower CL
    return sorted(
        results,
        key=lambda r: (-(r.get("total_gb") or 0), r.get("cas_latency") or 99),
    )[0]


def _pick_video_card(plan, reqs, build) -> Dict[str, Any] | None:
    if not _need_discrete_gpu(reqs):
        return None
    budget = plan["budget_allocation"]["video_card"] * 1.15
    filters = {"price_lte": budget, "price_gte": 80}
    results = search_components_impl("video_card", filters=filters,
                                     sort_by="estimated_tdp", ascending=False, top_k=10)
    if not results:
        return None
    # Most performant under budget that also actually exists in catalog
    return results[0]


def _pick_storage(plan, reqs, build) -> Dict[str, Any] | None:
    budget = plan["budget_allocation"]["storage"] * 1.2
    target = 1000 if reqs.get("use_case") in ("content_creation", "workstation") else 500
    filters = {
        "price_lte": budget,
        "type": "SSD",
        "capacity_gte": target,
    }
    results = search_components_impl("storage", filters=filters,
                                     sort_by="capacity", ascending=False, top_k=10)
    if not results:
        filters.pop("capacity_gte", None)
        results = search_components_impl("storage", filters=filters,
                                         sort_by="capacity", ascending=False, top_k=10)
    if not results:
        return None
    # Prefer NVMe (M.2) over SATA
    nvme = [r for r in results if "m.2" in (r.get("interface") or "").lower()]
    return nvme[0] if nvme else results[0]


def _pick_psu(plan, reqs, build) -> Dict[str, Any] | None:
    from src.compatibility.power_rules import estimate_load_watts
    build_obj = _build_obj(build)
    needed = max(450, estimate_load_watts(build_obj))
    budget = plan["budget_allocation"]["power_supply"] * 1.3
    filters = {"price_lte": budget, "wattage_gte": needed}
    results = search_components_impl("power_supply", filters=filters,
                                     sort_by="wattage", ascending=True, top_k=10)
    if not results:
        # Relax budget if no PSU is big enough
        filters.pop("price_lte", None)
        results = search_components_impl("power_supply", filters=filters,
                                         sort_by="price", ascending=True, top_k=10)
    return results[0] if results else None


def _pick_case(plan, reqs, build) -> Dict[str, Any] | None:
    if not build.get("motherboard"):
        return None
    budget = plan["budget_allocation"]["case"] * 1.3
    form = (build["motherboard"].get("form_factor") or "").strip()
    # Pick a case `type` that supports this form factor.
    if form == "Mini ITX":
        type_filter = {"type_contains": "Mini ITX"}
    elif form == "Micro ATX":
        type_filter = {"type_contains": "ATX"}  # ATX/microATX/etc.
    else:
        type_filter = {"type_contains": "ATX"}
    filters = {"price_lte": budget, **type_filter}
    results = search_components_impl("case", filters=filters,
                                     sort_by="price", ascending=True, top_k=10)
    if not results:
        results = search_components_impl("case", filters={"price_lte": budget},
                                         sort_by="price", ascending=True, top_k=10)
    return results[0] if results else None


def _pick_cooler(plan, reqs, build) -> Dict[str, Any] | None:
    budget = plan["budget_allocation"]["cpu_cooler"] * 1.5
    filters = {"price_lte": budget}
    results = search_components_impl("cpu_cooler", filters=filters,
                                     sort_by="price", ascending=True, top_k=10)
    if not results:
        return None
    if (reqs.get("noise_preference") or "") == "quiet":
        # Prefer quiet (lower noise) coolers; noise_level field is "min,max" string,
        # fall back to price if we cannot parse.
        def _quiet_score(r):
            nl = r.get("noise_level")
            try:
                vals = [float(x) for x in str(nl).split(",")]
                return max(vals) if vals else 99.0
            except ValueError:
                return 99.0
        results.sort(key=_quiet_score)
    return results[0]


_PICKERS = {
    "cpu": _pick_cpu,
    "motherboard": _pick_motherboard,
    "memory": _pick_memory,
    "video_card": _pick_video_card,
    "storage": _pick_storage,
    "power_supply": _pick_psu,
    "case": _pick_case,
    "cpu_cooler": _pick_cooler,
}


def _pick_relaxed(
    cat: str, plan: Dict[str, Any], reqs: Dict[str, Any], build: Dict[str, Any]
) -> Dict[str, Any] | None:
    """Second-chance picker with looser filters when the strict pass finds nothing."""
    budget_total = float(reqs.get("budget_usd") or 1500)
    spent = sum(float(c.get("price", 0) or 0) for c in build.values() if c)
    headroom = max(50, budget_total - spent)

    if cat == "motherboard" and build.get("cpu"):
        socket = build["cpu"].get("socket")
        if socket:
            results = search_components_impl(
                "motherboard",
                filters={"socket": socket, "price_lte": headroom},
                sort_by="price",
                ascending=True,
                top_k=5,
            )
            return results[0] if results else None

    if cat == "memory" and build.get("motherboard"):
        ddr = build["motherboard"].get("ddr_gen")
        filters: Dict[str, Any] = {"price_lte": headroom}
        if ddr:
            filters["ddr_gen"] = ddr
        results = search_components_impl(
            "memory", filters=filters, sort_by="price", ascending=True, top_k=10
        )
        return results[0] if results else None

    if cat == "video_card" and _need_discrete_gpu(reqs):
        results = search_components_impl(
            "video_card",
            filters={"price_lte": headroom, "price_gte": 50},
            sort_by="price",
            ascending=False,
            top_k=10,
        )
        return results[0] if results else None

    if cat == "case":
        results = search_components_impl(
            "case",
            filters={"price_lte": min(headroom, 120)},
            sort_by="price",
            ascending=True,
            top_k=10,
        )
        return results[0] if results else None

    # Generic fallback for cpu, storage, psu, cooler
    results = search_components_impl(
        cat,
        filters={"price_lte": headroom},
        sort_by="price",
        ascending=True,
        top_k=5,
    )
    return results[0] if results else None


def _build_obj(build_dict: Dict[str, Any]) -> Build:
    return Build(**(build_dict or {}))


# ---------------------------------------------------------------------------
# Budget-fill pass: upgrade weakest parts when the build is well under budget
# ---------------------------------------------------------------------------

def _try_upgrade(category: str, current: Dict[str, Any], ceiling: float,
                 plan: Dict[str, Any], reqs: Dict[str, Any],
                 build: Dict[str, Any]) -> Dict[str, Any] | None:
    """Find a more-premium component within `ceiling`. Returns None if no real
    upgrade is possible (i.e. ceiling barely above current price).
    """
    cur_price = float(current.get("price", 0) or 0)
    if ceiling <= cur_price * 1.10:  # need at least 10% headroom for a real upgrade
        return None

    if category == "video_card":
        results = search_components_impl(
            "video_card",
            filters={"price_lte": ceiling, "price_gte": cur_price * 1.15},
            sort_by="estimated_tdp", ascending=False,
            top_k=5,
        )
        return results[0] if results else None

    if category == "cpu":
        platform = (plan.get("platform_preference") or "any").upper()
        filters: Dict[str, Any] = {
            "price_lte": ceiling,
            "price_gte": cur_price * 1.15,
        }
        if platform != "ANY":
            filters["socket"] = platform
        if not _need_discrete_gpu(reqs):
            filters["has_integrated_graphics"] = True
        results = search_components_impl(
            "cpu", filters=filters, sort_by="core_count", ascending=False, top_k=5
        )
        return results[0] if results else None

    if category == "memory":
        mb = build.get("motherboard") or {}
        filters = {"price_lte": ceiling, "price_gte": cur_price * 1.15}
        ddr = mb.get("ddr_gen")
        if ddr:
            filters["ddr_gen"] = ddr
        results = search_components_impl(
            "memory", filters=filters, sort_by="total_gb", ascending=False, top_k=5
        )
        return results[0] if results else None

    if category == "storage":
        results = search_components_impl(
            "storage",
            filters={
                "price_lte": ceiling,
                "price_gte": cur_price * 1.15,
                "type_contains": "SSD",
            },
            sort_by="capacity", ascending=False,
            top_k=5,
        )
        return results[0] if results else None

    if category == "power_supply":
        results = search_components_impl(
            "power_supply",
            filters={"price_lte": ceiling, "price_gte": cur_price * 1.15},
            sort_by="wattage", ascending=False,
            top_k=5,
        )
        return results[0] if results else None

    return None


def _budget_fill_pass(build: Dict[str, Any], plan: Dict[str, Any],
                      reqs: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade high-impact components if the build is well under budget.

    Aims for ~90% budget utilization. Upgrade priority depends on use case
    (gaming -> GPU first; content creation -> CPU/memory first).
    """
    budget = float(reqs.get("budget_usd") or 0)
    if not budget:
        return build

    def total_now() -> float:
        return round(sum(float(c.get("price", 0) or 0) for c in build.values() if c), 2)

    target = budget * 0.90  # land near 90% of budget upper bound
    floor = budget * 0.85   # but at least 85%

    if total_now() >= floor:
        return build

    use_case = reqs.get("use_case", "general")
    if use_case == "gaming":
        order = ["video_card", "cpu", "memory", "storage", "power_supply"]
    elif use_case in ("content_creation", "workstation"):
        order = ["cpu", "memory", "video_card", "storage", "power_supply"]
    else:
        order = ["cpu", "memory", "storage", "video_card", "power_supply"]

    # Two passes: first try to reach target, then push as close as possible.
    for _pass in range(2):
        for cat in order:
            if total_now() >= target:
                return build
            current = build.get(cat)
            if not current:
                continue
            headroom = budget - total_now()
            if headroom <= 5:
                continue
            cur_price = float(current.get("price", 0) or 0)
            new_ceiling = cur_price + headroom  # we'll absorb the full delta if we upgrade
            upgrade = _try_upgrade(cat, current, new_ceiling, plan, reqs, build)
            if upgrade and upgrade.get("name") != current.get("name"):
                delta = float(upgrade.get("price", 0) or 0) - cur_price
                if delta > 0:
                    log.info(
                        "node.select.budget_fill",
                        category=cat,
                        old=current.get("name"),
                        new=upgrade.get("name"),
                        delta=round(delta, 2),
                    )
                    build[cat] = upgrade
    return build


def component_selector(state: AgentState) -> Dict[str, Any]:
    """Deterministic per-category picker that consults the search tool.

    We choose this design over an LLM tool-calling loop because:
    1. It is far more reliable on 7B-class local models.
    2. It still satisfies the brief's 'at least one tool/function call'
       requirement (we call `search_components` 8+ times per run).
    3. The LLM still has plenty to do: planning, critique, response writing.
    """
    t0 = time.time()
    plan = state.get("plan") or {}
    reqs = state.get("requirements") or {}
    build: Dict[str, Any] = dict(state.get("build") or {})
    attempts = state.get("selector_attempts", 0) + 1

    # If critique pointed to a specific weakest part, drop it so we re-pick.
    critique = state.get("critique") or {}
    if critique.get("verdict") == "revise":
        wp = critique.get("weakest_part")
        if wp in build:
            log.info("node.select.repick_after_critique", category=wp)
            build.pop(wp, None)

    # Pick (or re-pick) each empty category in order.
    for cat in CATEGORY_ORDER:
        if build.get(cat):
            continue
        picker = _PICKERS[cat]
        choice = picker(plan, reqs, build)
        if not choice:
            choice = _pick_relaxed(cat, plan, reqs, build)
        if choice:
            build[cat] = choice
            log.info("node.select.pick", category=cat, name=choice.get("name"),
                     price=choice.get("price"))
        else:
            log.warning("node.select.no_choice", category=cat)

    # If the build is significantly under budget, upgrade key parts.
    # Skip on the second attempt (post-critique) so we don't overwrite the
    # critique-driven re-pick.
    if attempts == 1:
        build = _budget_fill_pass(build, plan, reqs)

    log.info(
        "node.select.done",
        attempts=attempts,
        selected=list(build.keys()),
        elapsed_ms=int((time.time() - t0) * 1000),
    )
    return {"build": build, "selector_attempts": attempts, "mode": "check"}


# ---------------------------------------------------------------------------
# 4. compatibility_checker (deterministic - no LLM)
# ---------------------------------------------------------------------------

def compatibility_checker(state: AgentState) -> Dict[str, Any]:
    t0 = time.time()
    build_dict = state.get("build") or {}
    # Strip hallucinated components before checking
    build_dict, dropped = filter_real_components(build_dict)
    if dropped:
        log.warning("node.check.dropped_hallucinated", parts=dropped)

    try:
        build_obj = Build(**build_dict)
        issues = check_build(build_obj)
    except Exception as e:
        log.error("node.check.exception", error=str(e)[:200])
        issues = [Issue(severity="error", rule="schema_error",
                        message=f"Could not validate build: {e}")]

    issue_dicts = [i.model_dump() for i in issues]
    log.info(
        "node.check.done",
        total=len(issues),
        errors=sum(1 for i in issues if i.severity == "error"),
        warnings=sum(1 for i in issues if i.severity == "warn"),
        elapsed_ms=int((time.time() - t0) * 1000),
    )

    next_mode = "critique"
    if has_errors(issues) and state.get("selector_attempts", 0) < 3:
        # Drop the offending parts so the selector re-picks them.
        offenders = set()
        for i in issues:
            if i.severity == "error":
                for c in i.components:
                    for cat, comp in (build_dict or {}).items():
                        if comp and comp.get("name") == c:
                            offenders.add(cat)
        for cat in offenders:
            build_dict.pop(cat, None)
        log.info("node.check.repick_categories", categories=list(offenders))
        next_mode = "select"

    return {
        "build": build_dict,
        "compat_issues": issue_dicts,
        "mode": next_mode,
    }


# ---------------------------------------------------------------------------
# 5. self_critique
# ---------------------------------------------------------------------------

def self_critique(state: AgentState) -> Dict[str, Any]:
    t0 = time.time()
    attempts = state.get("critique_attempts", 0) + 1
    # Cap critique iterations to 1 to avoid flip-flop.
    if attempts > 1:
        log.info("node.critique.skipped_cap")
        return {"critique": {"verdict": "approve", "summary": "Critique cap reached."},
                "critique_attempts": attempts, "mode": "respond"}

    reqs = state.get("requirements") or {}
    build = state.get("build") or {}
    settings = get_settings()

    # Compact build summary for the LLM prompt
    summary = {
        cat: {"name": comp.get("name"), "price": comp.get("price")}
        for cat, comp in build.items() if comp
    }

    msgs = [
        SystemMessage(content=CRITIC_SYSTEM),
        HumanMessage(content=(
            "Requirements:\n"
            f"{json.dumps(reqs, indent=2)}\n\n"
            "Selected build (summary):\n"
            f"{json.dumps(summary, indent=2)}\n\n"
            f"Total price: ${Build(**build).total_price()}\n\n"
            "Now produce the critique JSON."
        )),
    ]
    ai = invoke_with_retry(msgs, temperature=0.1)
    critique = _parse_json_safely(ai.content) or {"verdict": "approve"}

    log.info(
        "node.critique.done",
        verdict=critique.get("verdict"),
        weakest=critique.get("weakest_part"),
        elapsed_ms=int((time.time() - t0) * 1000),
    )

    next_mode = "respond" if critique.get("verdict") == "approve" else "select"
    return {"critique": critique, "critique_attempts": attempts, "mode": next_mode}


def _format_build_response(
    reqs: Dict[str, Any],
    build: Dict[str, Any],
    issues: List[Dict[str, Any]],
    total: float,
) -> str:
    """Markdown summary when Ollama is offline but parts were selected."""
    lines = [
        "Here is a build assembled from the parts catalog "
        "(Ollama is offline, so this is a structured summary rather than an LLM-written reply).",
        "",
        "| Component | Part | Price |",
        "|---|---|---|",
    ]
    for cat in CATEGORY_ORDER:
        comp = build.get(cat)
        if comp and isinstance(comp, dict):
            lines.append(
                f"| {cat} | {comp.get('name', '?')} | "
                f"${float(comp.get('price', 0) or 0):.2f} |"
            )
    lines.append("")
    budget = reqs.get("budget_usd")
    lines.append(f"**Total: ${total:.2f}**" + (f" (your budget: ${budget})" if budget else ""))
    missing = [c for c in CATEGORY_ORDER if not build.get(c)]
    if missing:
        lines.append("")
        lines.append(
            f"*Could not find suitable catalog parts for: {', '.join(missing)} "
            f"at this budget. Try raising the budget or relaxing requirements.*"
        )
    if issues:
        lines.append("")
        lines.append("**Compatibility notes:**")
        lines.append(summarize_issues([Issue(**i) for i in issues]))
    lines.append("")
    lines.append(
        "To get a full natural-language explanation, start Ollama and pull the model, "
        "then send another message:\n"
        "```\nollama serve\nollama pull qwen2.5:7b-instruct\n```"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. responder
# ---------------------------------------------------------------------------

def _build_comparison_markdown(
    prev: Dict[str, Any],
    new: Dict[str, Any],
    prev_budget: float | None,
    new_budget: float | None,
) -> str:
    """Deterministic markdown diff. Always correct - used as a fallback when
    the LLM omits the comparison section."""
    if not prev:
        return ""

    def total_of(b: Dict[str, Any]) -> float:
        return round(sum(float(c.get("price", 0) or 0) for c in (b or {}).values() if c), 2)

    lines = ["### What changed vs your previous build"]
    changed = 0
    unchanged = 0
    for cat in CATEGORY_ORDER:
        old = prev.get(cat)
        nxt = new.get(cat)
        old_name = old.get("name") if old else None
        nxt_name = nxt.get("name") if nxt else None
        if old_name == nxt_name and old_name is not None:
            unchanged += 1
            continue
        old_price = float(old.get("price", 0) or 0) if old else 0.0
        nxt_price = float(nxt.get("price", 0) or 0) if nxt else 0.0
        delta = nxt_price - old_price
        sign = "+" if delta >= 0 else ""
        if old_name and nxt_name:
            lines.append(
                f"- **{cat}**: {old_name} (${old_price:.2f}) -> "
                f"{nxt_name} (${nxt_price:.2f}) [{sign}${delta:.2f}]"
            )
            changed += 1
        elif nxt_name and not old_name:
            lines.append(f"- **{cat}**: added {nxt_name} (${nxt_price:.2f})")
            changed += 1
        elif old_name and not nxt_name:
            lines.append(f"- **{cat}**: removed {old_name} (was ${old_price:.2f})")
            changed += 1

    if changed == 0:
        return ""  # no meaningful diff to show

    if unchanged:
        lines.append(f"- *Unchanged: {unchanged} component(s)*")

    old_total = total_of(prev)
    new_total = total_of(new)
    total_delta = new_total - old_total
    total_sign = "+" if total_delta >= 0 else ""
    lines.append(
        f"\n**Total:** ${old_total:.2f} -> ${new_total:.2f} "
        f"({total_sign}${total_delta:.2f})"
    )
    if prev_budget and new_budget:
        lines.append(
            f"**Budget:** ${float(prev_budget):.0f} -> ${float(new_budget):.0f}"
        )
    return "\n".join(lines)


def responder(state: AgentState) -> Dict[str, Any]:
    t0 = time.time()
    # Already-set short-circuit response (e.g. from gatherer asking questions)
    pre = state.get("final_response")
    if pre:
        log.info("node.respond.short_circuit")
        msg = AIMessage(content=pre)
        return {"messages": [msg], "mode": "await_feedback"}

    reqs = state.get("requirements") or {}
    build = state.get("build") or {}
    issues = state.get("compat_issues", [])
    prev_build = state.get("previous_build") or {}
    prev_budget = state.get("previous_budget_usd")

    try:
        build_obj = Build(**{k: v for k, v in build.items() if v})
    except Exception:
        build_obj = Build()
    total = build_obj.total_price() if build else 0.0

    # Compose the comparison section deterministically first, then ask the
    # LLM to weave it into a friendly response. We append our deterministic
    # version if the LLM happens to skip it.
    comparison_md = _build_comparison_markdown(
        prev_build, build, prev_budget, reqs.get("budget_usd")
    )

    human_parts = [
        "Requirements:",
        json.dumps(reqs, indent=2),
        "",
        "Selected build (full rows):",
        json.dumps(build, indent=2, default=str),
        "",
        "Compatibility findings:",
        summarize_issues([Issue(**i) for i in issues]),
        "",
        f"Total price: ${total}",
        f"User budget: ${reqs.get('budget_usd', 'N/A')}",
    ]
    if comparison_md:
        human_parts.extend([
            "",
            "Previous build (use this for the 'What changed' section):",
            json.dumps(prev_build, indent=2, default=str),
            f"Previous budget: ${prev_budget if prev_budget else 'N/A'}",
            "",
            "Pre-computed comparison (you MAY use this verbatim or rephrase):",
            comparison_md,
        ])
    human_parts.append("\nNow write the user-facing response in Markdown.")

    msgs = [
        SystemMessage(content=RESPONDER_SYSTEM),
        HumanMessage(content="\n".join(human_parts)),
    ]
    ai = invoke_with_retry(msgs, temperature=0.3)
    content = ai.content or ""
    if "unable to reach the language model" in content.lower() and build:
        content = _format_build_response(reqs, build, issues, total)

    # Safety net: if the LLM forgot the comparison section, append our
    # deterministic one.
    if comparison_md and "what changed" not in content.lower():
        content = content.rstrip() + "\n\n" + comparison_md

    log.info(
        "node.respond.done",
        total_price=total,
        n_issues=len(issues),
        has_previous=bool(prev_build),
        elapsed_ms=int((time.time() - t0) * 1000),
    )
    return {
        "messages": [AIMessage(content=content)],
        "final_response": content,
        "mode": "await_feedback",
    }


# ---------------------------------------------------------------------------
# 7. feedback_handler
# ---------------------------------------------------------------------------

def feedback_handler(state: AgentState) -> Dict[str, Any]:
    """Parse the latest user message as feedback on the existing build."""
    t0 = time.time()
    user_text = _last_user_text(state)

    ok, reason = validate_user_message(user_text)
    if not ok:
        return {"final_response": reason, "mode": "respond"}

    # Fast-path heuristic off-topic deflection on follow-up turns.
    if _looks_off_topic(user_text):
        log.info("node.feedback.off_topic_heuristic")
        return {"final_response": OFF_TOPIC_REPLY, "mode": "respond"}

    msgs = [
        SystemMessage(content=FEEDBACK_SYSTEM),
        HumanMessage(content=(
            "Current build (summary):\n"
            f"{json.dumps({k: v.get('name') for k, v in (state.get('build') or {}).items() if v}, indent=2)}\n\n"
            "User feedback:\n"
            f"{user_text}\n\n"
            "Now produce the feedback JSON."
        )),
    ]
    ai = invoke_with_retry(msgs, temperature=0.1)
    fb = _parse_json_safely(ai.content) or {"intent": "unclear"}

    log.info(
        "node.feedback.parsed",
        intent=fb.get("intent"),
        targets=fb.get("target_categories"),
        elapsed_ms=int((time.time() - t0) * 1000),
    )

    intent = fb.get("intent")
    if intent == "approve":
        return {
            "feedback": fb,
            "final_response": "Glad it works for you. Happy building!",
            "mode": "respond",
        }
    if intent == "off_topic":
        return {
            "feedback": fb,
            "final_response": OFF_TOPIC_REPLY,
            "mode": "respond",
        }
    if intent == "unclear":
        return {
            "feedback": fb,
            "final_response": "I am not sure what you would like to change. "
                              "Try something like: 'make it cheaper', 'more "
                              "storage', 'quieter', or 'compare with a $900 budget'.",
            "mode": "respond",
        }

    # Snapshot the existing build so the responder can produce a diff.
    prev_build = dict(state.get("build") or {})
    prev_budget = (state.get("requirements") or {}).get("budget_usd")

    # Apply deltas to requirements + plan, then re-plan from scratch.
    reqs = dict(state.get("requirements") or {})
    deltas = fb.get("delta_constraints") or {}
    if "budget_usd" in deltas:
        reqs["budget_usd"] = deltas["budget_usd"]
    if "budget_min_usd" in deltas:
        reqs["budget_min_usd"] = deltas["budget_min_usd"]
    if "noise_preference" in deltas:
        reqs["noise_preference"] = deltas["noise_preference"]
    # Backfill range from raw text if the LLM missed it.
    lo, hi = _extract_budget_range(user_text)
    if hi is not None and "budget_usd" not in deltas:
        reqs["budget_usd"] = hi
    if lo is not None and "budget_min_usd" not in deltas:
        reqs["budget_min_usd"] = lo

    # Add explicit must-haves so the planner picks differently
    extra = []
    for k, v in deltas.items():
        if k not in ("budget_usd", "budget_min_usd", "noise_preference"):
            extra.append(f"{k}={v}")
    if extra:
        reqs["must_have"] = list(reqs.get("must_have") or []) + extra

    # For budget changes or explicit compare requests, rebuild from scratch
    # so the comparison reflects the budget swing across every category.
    rebuild_full = intent in ("change_budget", "compare_builds") or (
        "budget_usd" in deltas or "budget_min_usd" in deltas
    )
    if rebuild_full:
        build: Dict[str, Any] = {}
    else:
        build = dict(state.get("build") or {})
        for cat in fb.get("target_categories") or []:
            build.pop(cat, None)

    return {
        "requirements": reqs,
        "build": build,
        "previous_build": prev_build,
        "previous_budget_usd": prev_budget,
        "feedback": fb,
        "selector_attempts": 0,
        "critique_attempts": 0,
        "compat_issues": [],
        "final_response": None,
        "mode": "plan",
    }
