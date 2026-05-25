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
    """Build a flat list of HumanMessage / AIMessage from few-shot examples.

    Each example may be a simple `{user, assistant}` pair OR a multi-turn
    `{turns: [{role, content}, ...]}` sequence (where assistant content can be
    a dict, which gets serialised to JSON).
    """
    msgs = []
    for ex in fewshots:
        if "turns" in ex:
            for turn in ex["turns"]:
                content = turn["content"]
                if isinstance(content, (dict, list)):
                    content = json.dumps(content, indent=2)
                if turn["role"] == "user":
                    msgs.append(HumanMessage(content=content))
                else:
                    msgs.append(AIMessage(content=content))
            continue
        msgs.append(HumanMessage(content=ex["user"]))
        msgs.append(AIMessage(content=json.dumps(ex["assistant"], indent=2)))
    return msgs


def _extract_budget_range(user_text: str) -> tuple[float | None, float | None]:
    """Parse `(min, max)` budget. Returns `(None, max)` if only a single value.

    Handles common phrasings:
      - "$1500" / "1500$" / "1,500" / "$1,500"
      - "1500 dollars" / "1500 USD"
      - "$1.5k" / "2k budget" / "$2K"
      - "$1000-$1500" / "between $1000 and $1500" / "1000 to 1500"

    Suppresses false positives:
      - Display resolutions (1440p, 1080p, 4K)
      - GPU model numbers (RTX 4090, GTX 1660, RX 7800, Arc B580)
      - Storage / RAM / wattage / clock numbers (256 GB, 16 GB, 750 W, 6 GHz)
    """
    if not user_text:
        return None, None
    text = user_text.replace(",", "")

    # ---- "1.5k" / "2k" notation ----
    # Replace e.g. "$1.5k" -> "$1500", "2k" -> "2000". To avoid colliding
    # with display resolutions like "4K", "8K", we only expand when there's
    # a budget signal nearby (a $ sign or a budget-related word within ~30
    # chars on either side of the match).
    BUDGET_K_CONTEXT = re.compile(
        r"\$|\b(budget|spend|spending|cost|price|under|max|for|around|about|"
        r"have|up to|up-to|range)\b",
        re.IGNORECASE,
    )

    def _expand_k(m: "re.Match[str]") -> str:
        full_text = m.string
        start, end = m.start(), m.end()
        window_start = max(0, start - 30)
        window_end = min(len(full_text), end + 30)
        window = full_text[window_start:window_end]
        if not BUDGET_K_CONTEXT.search(window):
            return m.group(0)  # leave it alone (likely "4K", "8K" resolution)
        num = float(m.group(1))
        return f"{int(round(num * 1000))}"

    text = re.sub(r"\$?\s*(\d+(?:\.\d+)?)\s*[kK]\b", _expand_k, text)

    # Mask GPU model numbers so they cannot be mistaken for budgets.
    # We replace the digit run with spaces of equal length so other regex
    # offsets aren't disturbed.
    def _mask(pattern: str, s: str) -> str:
        return re.sub(
            pattern,
            lambda m: m.group(0)[: m.start(1) - m.start()] + " " * (m.end(1) - m.start(1)) + m.group(0)[m.end(1) - m.start():],
            s,
            flags=re.IGNORECASE,
        )

    # GPU families: RTX/GTX/RX/Arc XXXX(XX)
    text = _mask(r"\b(?:rtx|gtx|rx|arc|titan)\s*(\d{3,5})\b", text)
    # CPU model numbers like "i5-14600K", "Ryzen 7 7700X"
    text = _mask(r"\b(?:i[3579]|core\s*i[3579]|ryzen\s*\d?|core\s*ultra)\s*[- ]?\s*(\d{4,5})", text)

    # Patterns that capture an explicit range.
    range_patterns = [
        r"between\s*\$?\s*(\d{2,5})\s*(?:and|to|-)\s*\$?\s*(\d{2,5})",
        r"from\s*\$?\s*(\d{2,5})\s*(?:to|-)\s*\$?\s*(\d{2,5})",
        r"\$?\s*(\d{2,5})\s*(?:-|to|and)\s*\$?\s*(\d{2,5})",
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
        r"\$\s*(\d{2,5})\b",                          # $1500
        r"(\d{2,5})\s*\$",                            # 1500$
        r"(?:for|budget|around|about)\s*\$?\s*(\d{2,5})\b",
        r"(\d{2,5})\s*(?:usd|dollars?)\b",
    ]
    for pat in single_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if 150 <= val <= 25000:
                return None, val
    # Loose fallback: any 3-5 digit number that is NOT immediately followed by
    # a unit suffix that would make it something other than dollars:
    #   p     -> 1080p / 1440p resolutions
    #   gb/tb/mb/kb -> storage or memory capacity (e.g. "512 GB storage")
    #   mhz/ghz/hz  -> clock speeds
    #   w           -> wattage
    #   k           -> 4K, etc. (also covered by 'p' for 720p/1080p)
    for m in re.finditer(
        r"(?<!\d)(\d{3,5})(?!\s?(?:p|gb|tb|mb|kb|mhz|ghz|hz|w)\b)",
        text,
        re.IGNORECASE,
    ):
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


def _heuristic_use_case(user_text: str) -> str | None:
    """Detect use case from raw text. Returns None if truly unclear.

    Order matters: we check the most specific buckets first so a phrase like
    "personal gaming PC" maps to gaming, not office. We use both whole-phrase
    `substring` matches AND word-boundary regex matches so bare single-word
    answers (e.g. just "personal", "office", "home") still get classified.
    """
    t = (user_text or "").lower().strip()
    if not t:
        return None

    def has_word(*words: str) -> bool:
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", t):
                return True
        return False

    if has_word(
        "gaming", "game", "games", "gamer", "esports", "valorant", "fortnite",
        "cs2", "warzone", "minecraft", "roblox",
    ) or any(
        s in t for s in (
            "1440p", "1080p", "fps", "league of legends", "play games",
            "game on", "for games", "for game",
        )
    ):
        return "gaming"
    # Workstation: heavy compute (ML, scientific, CAD, engineering, 3D render).
    if has_word(
        "workstation", "cad", "solidworks", "matlab", "autocad", "revit",
        "fea", "simulation",
    ) or any(
        s in t for s in (
            "data science", "machine learning", "deep learning",
            "ml work", "ml training", "model training", "engineering",
            "3d render", "scientific computing", "neural network",
            "ai training", "llm training",
        )
    ):
        return "workstation"
    if has_word(
        "davinci", "premiere", "blender", "render", "rendering",
        "streaming", "twitch", "lightroom",
    ) or any(
        s in t for s in (
            "video edit", "content creation", "youtube creator", "youtuber",
            "photo edit", "image edit", "podcast", "music production",
            "audio production",
        )
    ):
        return "content_creation"
    if has_word("plex", "nas") or any(
        s in t for s in ("home server", "media server", "file server")
    ):
        return "home_server"
    # Office / general home use. Bare single words like "personal", "home",
    # "office", "casual" should all classify here.
    if has_word(
        "office", "work", "working", "word", "excel", "zoom", "browse",
        "browsing", "spreadsheet", "spreadsheets", "email", "emails",
        "documents", "personal", "home", "general", "everyday", "casual",
        "basic", "school", "study", "homework", "internet", "netflix",
        "youtube", "web", "browser", "teams",
    ) or any(s in t for s in (
        "social media", "day to day", "day-to-day", "wfh", "work from home",
        "remote work",
    )):
        return "office"
    return None


def _heuristic_feedback(user_text: str) -> Dict[str, Any] | None:
    """Best-effort intent detection from raw text when the LLM fails.

    Returns the same shape as the LLM's feedback JSON, or None if the text
    is truly unclear.
    """
    if not user_text:
        return None
    t = user_text.lower().strip()

    def has_word(*words: str) -> bool:
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", t):
                return True
        return False

    # ---- Approval ----
    approve_phrases = (
        "looks good", "looks great", "perfect", "ship it",
        "go with this", "go with that", "i'll take it", "ill take it",
    )
    if len(t) <= 60 and (
        any(p in t for p in approve_phrases)
        or has_word("approve", "approved", "yes", "thanks")
    ):
        return {"intent": "approve", "delta_constraints": {}, "target_categories": []}

    # ---- Comparison requests (must come BEFORE change_budget so phrases
    # like "compare with $900 budget" are not eaten by the budget branch).
    lo, hi = _extract_budget_range(user_text)
    compare_signals = (
        "compare", "comparison", "vs.", " vs ", "versus",
        "side by side", "side-by-side",
    )
    hypothetical_signals = (
        "what if i had", "what if i spent", "what would i get",
        "what could i get", "if i had", "if i spent",
        "if my budget were", "if my budget was",
        "show me a build for",
    )
    if any(s in t for s in compare_signals) or any(
        s in t for s in hypothetical_signals
    ):
        deltas: Dict[str, Any] = {}
        if hi is not None:
            deltas["budget_usd"] = hi
        if lo is not None:
            deltas["budget_min_usd"] = lo
        return {
            "intent": "compare_builds",
            "delta_constraints": deltas,
            "target_categories": [],
        }

    # ---- Relative budget moves (no explicit dollar amount needed) ----
    # "double my budget", "halve it", "cut my budget in half", "1.5x budget".
    rel_budget_factor: float | None = None
    if re.search(r"\bdouble (the|my)? ?budget\b", t) or re.search(r"\b2x (the |my )?budget\b", t):
        rel_budget_factor = 2.0
    elif re.search(r"\btriple (the|my)? ?budget\b", t):
        rel_budget_factor = 3.0
    elif re.search(r"\b(half|halve)\b.*\bbudget\b", t) or re.search(r"\bcut.*budget.*half\b", t):
        rel_budget_factor = 0.5
    elif re.search(r"\b1\.5x (the |my )?budget\b", t):
        rel_budget_factor = 1.5
    if rel_budget_factor is not None:
        # Caller computes the actual new budget from current state.
        return {
            "intent": "change_budget",
            "delta_constraints": {"budget_multiplier": rel_budget_factor},
            "target_categories": [],
        }

    # ---- Budget changes ----
    budget_signals = (
        "budget", "increase", "decrease", "raise", "lower", "bump",
        "drop", "max", "maximum", "around", "spend", "between",
    )
    budget_signal_substrings = ("up to",)
    if hi is not None and (
        has_word(*budget_signals)
        or any(s in t for s in budget_signal_substrings)
        # A budget range like "$1000-$1500" or "between 800 and 1200"
        # captured by _extract_budget_range (lo is set) implies budget intent.
        or lo is not None
    ):
        deltas = {"budget_usd": hi}
        if lo is not None:
            deltas["budget_min_usd"] = lo
        return {
            "intent": "change_budget",
            "delta_constraints": deltas,
            "target_categories": [],
        }

    # "make it cheaper" / "more expensive" - relative budget.
    # We also detect category-scoped versions like "less expensive cpu",
    # "cheaper gpu", "less ram" - these should drop just that category and
    # re-pick a cheaper option instead of shrinking the whole budget (which
    # would be wasteful if everything else is already fine).
    CATEGORY_KEYWORDS = {
        "cpu": ("cpu", "processor"),
        "video_card": ("gpu", "video card", "graphics card", "graphics"),
        "memory": ("ram", "memory"),
        "storage": ("ssd", "storage", "drive", "hdd", "nvme"),
        "motherboard": ("motherboard", "mobo", "mainboard"),
        "power_supply": ("psu", "power supply"),
        "case": ("case", "chassis", "tower"),
        "cpu_cooler": ("cooler", "heatsink"),
    }
    cheaper_signal = (
        has_word("cheaper") or any(
            s in t for s in ("less expensive", "lower price", "lower cost", "less cost")
        )
    )
    less_only_signal = (
        cheaper_signal
        or has_word("less")  # "less cpu", "less ram"
        or "lower " in t
        or "smaller " in t  # "smaller storage" -> cheaper storage
    )
    if less_only_signal:
        # Find a category mentioned alongside the cheap-signal word.
        for cat, kws in CATEGORY_KEYWORDS.items():
            if any(re.search(rf"\b{re.escape(kw)}\b", t) for kw in kws):
                return {
                    "intent": "swap_part",
                    "delta_constraints": {"price_lower_category": cat},
                    "target_categories": [cat],
                }
    # No category mentioned - apply to the whole build.
    if cheaper_signal:
        return {
            "intent": "swap_part",
            "delta_constraints": {"price_lower": True},
            "target_categories": [],
        }

    # ---- Component-specific tweaks ----
    if has_word("quieter", "silent", "noiseless") or any(
        s in t for s in ("less noisy", "less noise", "quiet")
    ):
        return {
            "intent": "swap_part",
            "delta_constraints": {"noise_preference": "quiet"},
            "target_categories": ["cpu_cooler", "case"],
        }
    if any(s in t for s in ("more storage", "bigger ssd", "larger ssd",
                              "more disk", "bigger drive")):
        return {
            "intent": "swap_part",
            "delta_constraints": {"storage_capacity_gte": 2000},
            "target_categories": ["storage"],
        }
    if any(s in t for s in ("more ram", "more memory", "bigger ram", "larger ram")):
        return {
            "intent": "swap_part",
            "delta_constraints": {"memory_gte": 32},
            "target_categories": ["memory"],
        }
    # ---- GPU brand swaps ----
    # Triggers (any of):
    #  - "nvidia"/"geforce"/"rtx"/"gtx" + (gpu keyword OR swap context)
    #  - "radeon" anywhere (it's specifically a GPU brand, never a CPU)
    #  - bare "amd" near a gpu keyword OR in a swap context that ALSO mentions
    #    a gpu keyword - so "swap GPU with AMD" classifies as GPU, not CPU.
    has_nvidia = re.search(r"\b(nvidia|geforce|rtx|gtx)\b", t) is not None
    has_radeon = "radeon" in t
    has_amd_word = re.search(r"\bamd\b", t) is not None
    has_gpu_kw = any(s in t for s in ("gpu", "graphics", "video card", " card"))
    swap_context = any(
        s in t for s in ("swap", "change", "switch", "replace", "use ", "go with", "instead")
    )

    if has_nvidia and (has_gpu_kw or swap_context):
        return {
            "intent": "swap_part",
            "delta_constraints": {
                "chipset_contains": "GeForce",
                "gpu_brand_preference": "nvidia",
            },
            "target_categories": ["video_card"],
        }
    if has_radeon:
        return {
            "intent": "swap_part",
            "delta_constraints": {
                "chipset_contains": "Radeon",
                "gpu_brand_preference": "amd",
            },
            "target_categories": ["video_card"],
        }
    # Bare "AMD" + gpu keyword => GPU swap (NOT a CPU swap). This MUST come
    # before the CPU brand block below or "swap GPU with AMD" would be
    # misclassified as a CPU brand change.
    if has_amd_word and has_gpu_kw:
        return {
            "intent": "swap_part",
            "delta_constraints": {
                "chipset_contains": "Radeon",
                "gpu_brand_preference": "amd",
            },
            "target_categories": ["video_card"],
        }

    # ---- CPU brand swaps ----
    # Trigger when the user mentions AMD/Intel near a CPU-related keyword
    # OR uses Ryzen / Core (which are unambiguously CPU brand markers).
    # Examples: "i want AMD cpu not intel", "swap Intel cpu with AMD",
    # "give me ryzen instead", "use intel processor", "update to AMD",
    # "upgrade the cpu to ryzen", "go with intel".
    cpu_kw = re.search(r"\b(cpu|processor|ryzen|core\s?i\d|core\s?ultra)\b", t)
    has_amd = re.search(r"\b(amd|ryzen)\b", t) is not None
    has_intel = re.search(r"\b(intel|core\s?i\d|core\s?ultra)\b", t) is not None
    SWAP_VERBS = (
        "swap", "replace", "change", "switch", "update", "upgrade",
        "use", "give", "make", "go", "pick", "want", "prefer", "rather",
    )
    has_swap_verb = any(re.search(rf"\b{re.escape(v)}\b", t) for v in SWAP_VERBS)
    if cpu_kw or has_amd or has_intel:
        # Prefer AMD if the user wrote "amd ... not intel" or "amd instead",
        # or if only AMD markers are present.
        wants_amd = (
            has_amd and (
                re.search(r"\bnot\s+intel\b", t) or
                "instead" in t or
                has_swap_verb or
                not has_intel
            )
        )
        wants_intel = (
            has_intel and not wants_amd and (
                re.search(r"\bnot\s+amd\b", t) or
                "instead" in t or
                has_swap_verb or
                not has_amd
            )
        )
        if wants_amd:
            return {
                "intent": "swap_part",
                "delta_constraints": {"cpu_brand_preference": "amd"},
                # Motherboard socket and memory DDR generation change with
                # the CPU brand, so they must be re-picked too.
                "target_categories": ["cpu", "motherboard", "memory"],
            }
        if wants_intel:
            return {
                "intent": "swap_part",
                "delta_constraints": {"cpu_brand_preference": "intel"},
                "target_categories": ["cpu", "motherboard", "memory"],
            }

    # ---- Generic "change X" requests ----
    # Look for "<change-verb> ... <component-noun>" in either order.
    # Aliases are intentionally broad so user wording like "graphics card",
    # "power supply", "heatsink", "drive", "mobo" all work.
    component_synonyms = {
        "video_card": (
            "gpu", "video card", "graphics card", "graphics", "vga",
        ),  # keep before "cpu" so "graphics card" isn't shadowed
        "cpu_cooler": (
            "cooler", "cpu cooler", "heatsink", "aio", "liquid cooler",
        ),
        "cpu": ("cpu", "processor", "chip"),
        "memory": ("ram", "memory", "ddr"),
        "storage": (
            "storage", "ssd", "hdd", "disk", "drive", "nvme", "m.2",
        ),
        "motherboard": ("motherboard", "mobo", "mainboard", "board"),
        "power_supply": ("psu", "power supply", "power-supply", "power_supply"),
        "case": ("case", "chassis", "tower", "enclosure"),
    }
    change_verbs = (
        "change", "swap", "replace", "different", "another", "new",
        "update", "upgrade", "switch", "use", "pick",
    )
    has_change_verb = any(
        re.search(rf"\b{re.escape(v)}\b", t) for v in change_verbs
    )
    if has_change_verb:
        for cat, kws in component_synonyms.items():
            if any(re.search(rf"\b{re.escape(kw)}\b", t) for kw in kws):
                return {
                    "intent": "swap_part",
                    "delta_constraints": {},
                    "target_categories": [cat],
                }

    return None


def _heuristic_requirements(user_text: str, base: Dict[str, Any]) -> Dict[str, Any]:
    """Fill gaps when the LLM is down and could not parse requirements."""
    if base.get("budget_usd") is None:
        lo, hi = _extract_budget_range(user_text)
        if hi is not None:
            base["budget_usd"] = hi
        if lo is not None and base.get("budget_min_usd") is None:
            base["budget_min_usd"] = lo
    if base.get("use_case") in (None, "general"):
        uc = _heuristic_use_case(user_text)
        if uc:
            base["use_case"] = uc
    # Detect brand preferences expressed in the initial message itself
    # (e.g. "I want an AMD gaming PC", "Intel build please").
    if not base.get("cpu_brand_preference"):
        tt = user_text.lower()
        # Trigger on a bare brand mention - the user almost never types AMD
        # or Intel without meaning the CPU brand. (GPU brand is detected
        # separately below.)
        if re.search(r"\b(amd|ryzen)\b", tt) and "amd gpu" not in tt and "radeon" not in tt:
            base["cpu_brand_preference"] = "amd"
        elif re.search(r"\b(intel|core\s?i\d|core\s?ultra)\b", tt):
            base["cpu_brand_preference"] = "intel"
    if not base.get("gpu_brand_preference"):
        tt = user_text.lower()
        if re.search(r"\b(nvidia|geforce|rtx|gtx)\b", tt):
            base["gpu_brand_preference"] = "nvidia"
        elif "radeon" in tt or "amd gpu" in tt:
            base["gpu_brand_preference"] = "amd"
    if base.get("budget_usd") and base.get("use_case") not in (None, "general"):
        base["confidence"] = "high"
        base["clarifying_questions"] = []
    return base


def _merge_requirements(prev: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a freshly-extracted Requirements dict over the previous turn's.

    Rules:
    - Scalar fields: prefer the new value if non-empty/non-default; else keep
      the previous value (so info already given in turn 1 isn't lost).
    - List fields (must_have, nice_to_have, peripherals_needed): union.
    - Confidence: recomputed at the end based on whether we now have both
      budget_usd and a concrete use_case.
    """
    out: Dict[str, Any] = dict(prev or {})
    new = new or {}

    SCALAR_KEYS = (
        "is_on_topic", "use_case", "budget_usd", "budget_min_usd",
        "budget_flexible", "noise_preference", "form_factor_preference",
        "cpu_brand_preference", "gpu_brand_preference",
        "os_needed",
    )
    for k in SCALAR_KEYS:
        v = new.get(k)
        if v is None:
            continue
        if k == "use_case" and v == "general" and out.get("use_case") not in (
            None, "", "general"
        ):
            continue
        if k == "form_factor_preference" and v == "any" and out.get(
            "form_factor_preference"
        ) not in (None, "", "any"):
            continue
        out[k] = v

    for k in ("must_have", "nice_to_have", "peripherals_needed"):
        merged = list(prev.get(k) or []) if prev else []
        for item in new.get(k) or []:
            if item not in merged:
                merged.append(item)
        out[k] = merged

    # Recompute confidence after merge. If we now have budget + use_case,
    # we're ready to plan - clear leftover clarifying questions.
    have_budget = out.get("budget_usd") is not None
    have_use_case = out.get("use_case") not in (None, "", "general")
    if have_budget and have_use_case:
        out["confidence"] = "high"
        out["clarifying_questions"] = []
    else:
        # Inherit clarifying_questions from new extraction; fall back to prev.
        out["clarifying_questions"] = (
            new.get("clarifying_questions")
            or (prev.get("clarifying_questions") if prev else [])
            or []
        )
        out["confidence"] = new.get("confidence") or (
            prev.get("confidence") if prev else "low"
        )
    return out


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

    # If a prior turn already extracted partial requirements (e.g. the user
    # answered clarifying questions), give the LLM that context so it can
    # MERGE rather than start from scratch and lose info.
    prev_reqs = state.get("requirements") or {}
    if prev_reqs:
        msgs.append(HumanMessage(content=(
            "Context from earlier in this conversation - the previously "
            "extracted requirements snapshot. MERGE these with anything new "
            "in my latest message above (do not lose information that was "
            "given earlier):\n\n"
            + json.dumps(prev_reqs, indent=2)
        )))

    ai = invoke_with_retry(msgs, temperature=0.0)
    parsed = _parse_json_safely(ai.content) or {}

    # LLM-flagged off-topic - but only trust it if our positive PC-signal
    # regex ALSO agrees the text has no PC content. The LLM has been seen
    # to misfire on inputs like "Personal PC with 512 GB storage" where
    # "Personal" pattern-matches off-topic to the model. The heuristic is
    # conservative and high-precision, so it wins ties.
    if parsed.get("is_on_topic") is False and not _ON_TOPIC_HINTS.search(user_text):
        log.info("node.gather.off_topic_llm")
        return {"final_response": OFF_TOPIC_REPLY, "mode": "respond"}
    if parsed.get("is_on_topic") is False and _ON_TOPIC_HINTS.search(user_text):
        # Override the LLM - keep parsing requirements as if on-topic.
        log.info("node.gather.off_topic_llm_override", reason="on_topic_hint_matched")
        parsed["is_on_topic"] = True

    requirements = {
        "is_on_topic": parsed.get("is_on_topic", True),
        "use_case": parsed.get("use_case", "general"),
        "budget_usd": parsed.get("budget_usd"),
        "budget_min_usd": parsed.get("budget_min_usd"),
        "budget_flexible": bool(parsed.get("budget_flexible", False)),
        "noise_preference": parsed.get("noise_preference"),
        "form_factor_preference": parsed.get("form_factor_preference", "any"),
        "cpu_brand_preference": parsed.get("cpu_brand_preference"),
        "gpu_brand_preference": parsed.get("gpu_brand_preference"),
        "os_needed": bool(parsed.get("os_needed", False)),
        "peripherals_needed": parsed.get("peripherals_needed", []) or [],
        "must_have": parsed.get("must_have", []) or [],
        "nice_to_have": parsed.get("nice_to_have", []) or [],
        "confidence": parsed.get("confidence", "low"),
        "clarifying_questions": parsed.get("clarifying_questions", []) or [],
    }

    # Always run heuristic backfill on the latest message. This catches cases
    # where the user supplied new info in a short follow-up ("office, $600-$700")
    # that the LLM under-extracted from.
    llm_unreachable = (
        not parsed
        or "unable to reach the language model" in (ai.content or "").lower()
    )
    requirements = _heuristic_requirements(user_text, requirements)

    # Merge with previously-extracted requirements so info from turn 1
    # ("512 GB storage, 8 GB RAM") isn't lost when turn 2 only provides
    # the missing pieces ("office, $600-700").
    if prev_reqs:
        requirements = _merge_requirements(prev_reqs, requirements)

    # Deterministic gating: we MUST have both budget and use_case before
    # planning. Don't trust the LLM's `confidence` field alone - if it set
    # confidence=high without enough info, override and ask anyway.
    have_budget = requirements.get("budget_usd") is not None
    have_use_case = requirements.get("use_case", "general") not in (None, "", "general")
    needs_clarification = not (have_budget and have_use_case)

    log.info(
        "node.gather.done",
        confidence=requirements["confidence"],
        budget=requirements["budget_usd"],
        budget_min=requirements.get("budget_min_usd"),
        use_case=requirements["use_case"],
        merged=bool(prev_reqs),
        llm_unreachable=llm_unreachable,
        needs_clarification=needs_clarification,
        elapsed_ms=int((time.time() - t0) * 1000),
    )

    if needs_clarification:
        # Use the LLM's clarifying questions if it gave any; otherwise
        # synthesize the missing ones deterministically.
        qs = list(requirements.get("clarifying_questions") or [])
        if not qs:
            if not have_use_case:
                qs.append(
                    "What is your primary use case "
                    "(gaming, office, content creation, browsing, workstation)?"
                )
            if not have_budget:
                qs.append("What is your total budget in USD (single number or a range)?")
        requirements["clarifying_questions"] = qs
        requirements["confidence"] = "low"
        q_text = "I need a little more info before I can suggest a build:\n\n"
        q_text += "\n".join(f"- {q}" for q in qs[:3])
        return {
            "requirements": requirements,
            "final_response": q_text,
            "mode": "respond",
        }

    # IMPORTANT: clear any stale `final_response` from a prior turn's
    # short-circuit reply (e.g. clarifying questions). Otherwise the
    # responder will short-circuit on it and never call the LLM.
    return {"requirements": requirements, "final_response": None, "mode": "plan"}


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
    # AMD -> AM5, Intel -> LGA1700, otherwise "ANY" so the picker can roam.
    cpu_brand = (reqs.get("cpu_brand_preference") or "").lower()
    if cpu_brand == "amd":
        platform = "AM5"
    elif cpu_brand == "intel":
        platform = "LGA1700"
    else:
        platform = "ANY"
    return {
        "reasoning": "Fallback heuristic allocation (LLM plan was unparseable).",
        "performance_tier": "mainstream",
        "platform_preference": platform,
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


def _category_price_ceiling(reqs: Dict[str, Any], category: str,
                            default_budget: float) -> float:
    """Honor a user-requested per-category price cap ('less expensive cpu')
    by clamping the effective per-category budget.
    """
    ceilings = reqs.get("category_price_ceilings") or {}
    cap = ceilings.get(category)
    if cap is not None:
        return min(float(cap), default_budget)
    return default_budget


# Mainstream sockets, in order of motherboard availability / common-ness.
# When no explicit platform is set we restrict the CPU picker to this list
# so we never end up with an LGA1851/sTRX4 CPU paired with a board that
# doesn't exist or doesn't fit the budget.
_MAINSTREAM_SOCKETS = ["AM5", "AM4", "LGA1700", "LGA1200"]


def _pick_cpu(plan: Dict[str, Any], reqs: Dict[str, Any], build: Dict[str, Any]) -> Dict[str, Any] | None:
    budget = plan["budget_allocation"]["cpu"] * 1.15  # 15% per-category slack
    budget = _category_price_ceiling(reqs, "cpu", budget)
    platform = (plan.get("platform_preference") or "any").upper()
    need_igpu = not _need_discrete_gpu(reqs)
    base_filters: Dict[str, Any] = {"price_lte": budget, "price_gte": 50}
    if need_igpu:
        base_filters["has_integrated_graphics"] = True
    # Honor an explicit CPU brand preference (e.g. "I want AMD, not Intel").
    # The CSV's `name` column starts with the brand ("AMD Ryzen ..." /
    # "Intel Core ..."), so a substring match on it is a reliable filter.
    brand = (reqs.get("cpu_brand_preference") or "").lower()
    if brand in ("amd", "intel"):
        base_filters["name_contains"] = brand.upper() if brand == "amd" else "Intel"

    # ---- Build the candidate-socket list ----
    # 1. Explicit platform_preference (planner / brand) wins.
    # 2. Otherwise restrict to mainstream sockets so the motherboard picker
    #    always has plenty of in-budget boards to pair the CPU with.
    candidate_sockets: list[str] = []
    if platform != "ANY":
        # Drop conflicting socket vs brand (e.g. AM5 + Intel).
        if brand == "amd" and platform.startswith("LGA"):
            candidate_sockets = ["AM5", "AM4"]
        elif brand == "intel" and platform.startswith("AM"):
            candidate_sockets = ["LGA1700", "LGA1200"]
        else:
            candidate_sockets = [platform]
    else:
        # No explicit platform - constrain to mainstream sockets, optionally
        # filtered by brand.
        if brand == "amd":
            candidate_sockets = ["AM5", "AM4"]
        elif brand == "intel":
            candidate_sockets = ["LGA1700", "LGA1200"]
        else:
            candidate_sockets = list(_MAINSTREAM_SOCKETS)

    # Try sockets in order, keeping the best-cored result.
    results: list = []
    for sock in candidate_sockets:
        f = {**base_filters, "socket": sock}
        r = search_components_impl(
            "cpu", filters=f, sort_by="core_count", ascending=False, top_k=10
        )
        if r:
            results = r
            break

    # Fallbacks: drop platform, then drop brand.
    if not results:
        results = search_components_impl(
            "cpu", filters=base_filters,
            sort_by="core_count", ascending=False, top_k=10,
        )
    if not results and brand:
        base_filters.pop("name_contains", None)
        log.info("node.select.brand_relaxed", brand=brand, category="cpu")
        results = search_components_impl(
            "cpu", filters=base_filters,
            sort_by="core_count", ascending=False, top_k=10,
        )
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
    budget = _category_price_ceiling(reqs, "motherboard", budget)
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
    budget = _category_price_ceiling(reqs, "memory", budget)
    # Per-use-case target: do not overshoot. Picking the largest kit that fits
    # is wasteful for low-budget office/browsing builds (32 GB for $300
    # browsing PC pushes the total well over budget).
    use_case = reqs.get("use_case", "general")
    if use_case in ("content_creation", "workstation"):
        target_gb, hard_cap_gb = 32, 64
    elif use_case == "gaming":
        target_gb, hard_cap_gb = 16, 32
    else:  # office / general / browsing
        target_gb, hard_cap_gb = 16, 16
    # Allow >= target_gb up to the smaller of the board's max and our hard cap.
    max_mem = min(build["motherboard"].get("max_memory") or 64, hard_cap_gb)
    filters: Dict[str, Any] = {
        "price_lte": budget,
        "ddr_gen": ddr,
        "total_gb_gte": target_gb,
        "total_gb_lte": max_mem,
    }
    slots = build["motherboard"].get("memory_slots")
    if slots:
        filters["module_count_lte"] = slots
    # Sort by price ASC and pick the cheapest kit that meets target_gb. This
    # respects the user's needs without over-spec'ing memory.
    results = search_components_impl("memory", filters=filters,
                                     sort_by="price", ascending=True, top_k=10)
    if not results:
        # Fallback: relax the upper cap (some boards report no max_memory)
        filters.pop("total_gb_lte", None)
        results = search_components_impl("memory", filters=filters,
                                         sort_by="price", ascending=True, top_k=10)
    if not results:
        filters.pop("total_gb_gte", None)
        results = search_components_impl("memory", filters=filters,
                                         sort_by="price", ascending=True, top_k=10)
    if not results:
        return None
    # Among the cheapest 5, prefer the one closest to target_gb (not over it)
    # then lower CAS latency.
    top = results[:5]
    return sorted(
        top,
        key=lambda r: (
            abs((r.get("total_gb") or 0) - target_gb),
            r.get("cas_latency") or 99,
        ),
    )[0]


def _pick_video_card(plan, reqs, build) -> Dict[str, Any] | None:
    if not _need_discrete_gpu(reqs):
        return None
    budget = plan["budget_allocation"]["video_card"] * 1.15
    budget = _category_price_ceiling(reqs, "video_card", budget)
    # Filter out ancient / extremely-low-end cards. A 2011 GTX 570 has 1.28 GB
    # VRAM and high TDP; we never want it in a modern build. Modern entry-
    # level discrete GPUs have >= 4 GB VRAM.
    filters: Dict[str, Any] = {
        "price_lte": budget,
        "price_gte": 80,
        "memory_gte": 4,
    }
    # Honor an explicit GPU brand preference (chipset column carries the
    # marketing name, e.g. "GeForce RTX 4070" or "Radeon RX 7800 XT").
    gbrand = (reqs.get("gpu_brand_preference") or "").lower()
    if gbrand == "nvidia":
        filters["chipset_contains"] = "GeForce"
    elif gbrand == "amd":
        filters["chipset_contains"] = "Radeon"

    # Sort by price DESC - within budget, the most expensive card is almost
    # always the newest/most performant generation. Using estimated_tdp DESC
    # (the previous strategy) accidentally favored ancient inefficient cards
    # like the GTX 570 (219W) over modern efficient ones (RTX 4060 at 115W).
    results = search_components_impl(
        "video_card", filters=filters,
        sort_by="price", ascending=False, top_k=10,
    )
    if not results:
        # Relax the VRAM floor first (very tight budgets).
        filters.pop("memory_gte", None)
        results = search_components_impl(
            "video_card", filters=filters,
            sort_by="price", ascending=False, top_k=10,
        )
    if not results and gbrand:
        # Drop brand filter as a last resort - better a wrong-brand modern
        # card than no card.
        filters.pop("chipset_contains", None)
        log.info("node.select.brand_relaxed", brand=gbrand, category="video_card")
        results = search_components_impl(
            "video_card", filters=filters,
            sort_by="price", ascending=False, top_k=10,
        )
    if not results:
        return None
    return results[0]


def _pick_storage(plan, reqs, build) -> Dict[str, Any] | None:
    budget = plan["budget_allocation"]["storage"] * 1.2
    budget = _category_price_ceiling(reqs, "storage", budget)
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
    load = estimate_load_watts(build_obj)
    # Compat check requires PSU >= load * 1.10. Target an even higher
    # wattage (load * 1.15) so the chosen PSU clears the bar with margin.
    needed = max(450, int(round(load * 1.15)))
    budget = plan["budget_allocation"]["power_supply"] * 1.3
    budget = _category_price_ceiling(reqs, "power_supply", budget)
    filters = {"price_lte": budget, "wattage_gte": needed}
    results = search_components_impl("power_supply", filters=filters,
                                     sort_by="wattage", ascending=True, top_k=10)
    if not results:
        # Relax budget if no PSU is big enough - we'd rather slightly bust
        # the per-category allocation than ship an undersized PSU.
        filters.pop("price_lte", None)
        results = search_components_impl("power_supply", filters=filters,
                                         sort_by="price", ascending=True, top_k=10)
    return results[0] if results else None


def _pick_case(plan, reqs, build) -> Dict[str, Any] | None:
    if not build.get("motherboard"):
        return None
    budget = plan["budget_allocation"]["case"] * 1.3
    budget = _category_price_ceiling(reqs, "case", budget)
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
    budget = _category_price_ceiling(reqs, "cpu_cooler", budget)
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

    if cat == "motherboard":
        # NEVER fall through to a socket-blind fallback here. A mismatched
        # motherboard creates a non-functional build. If the CPU has no
        # socket info or no compatible board fits, return None and let
        # component_selector log the gap.
        cpu = build.get("cpu") or {}
        socket = cpu.get("socket")
        if not socket:
            return None
        results = search_components_impl(
            "motherboard",
            filters={"socket": socket, "price_lte": headroom},
            sort_by="price",
            ascending=True,
            top_k=5,
        )
        if results:
            return results[0]
        # Last resort: relax price ceiling (a real $300 LGA1851 board is
        # better than no board) so the build is at least functional.
        results = search_components_impl(
            "motherboard",
            filters={"socket": socket},
            sort_by="price",
            ascending=True,
            top_k=5,
        )
        return results[0] if results else None

    if cat == "memory":
        # Memory MUST match the motherboard's DDR generation. If no
        # motherboard exists yet, we cannot safely pick memory.
        mb = build.get("motherboard") or {}
        if not mb:
            return None
        filters: Dict[str, Any] = {"price_lte": headroom}
        ddr = mb.get("ddr_gen")
        if ddr:
            filters["ddr_gen"] = ddr
        results = search_components_impl(
            "memory", filters=filters, sort_by="price", ascending=True, top_k=10
        )
        if results:
            return results[0]
        # Relax: drop price ceiling but keep DDR generation constraint.
        if ddr:
            results = search_components_impl(
                "memory", filters={"ddr_gen": ddr},
                sort_by="price", ascending=True, top_k=5,
            )
        return results[0] if results else None

    if cat == "video_card":
        # Office / browsing / general builds use the CPU's iGPU - never
        # silently slip a discrete GPU into them just because the strict
        # picker returned None.
        if not _need_discrete_gpu(reqs):
            return None
        gbrand = (reqs.get("gpu_brand_preference") or "").lower()
        filters: Dict[str, Any] = {
            "price_lte": headroom,
            "price_gte": 50,
            "memory_gte": 4,  # exclude ancient sub-4GB cards
        }
        if gbrand == "nvidia":
            filters["chipset_contains"] = "GeForce"
        elif gbrand == "amd":
            filters["chipset_contains"] = "Radeon"
        results = search_components_impl(
            "video_card", filters=filters,
            sort_by="price", ascending=False, top_k=10,
        )
        if not results:
            filters.pop("memory_gte", None)
            results = search_components_impl(
                "video_card", filters=filters,
                sort_by="price", ascending=False, top_k=10,
            )
        if not results and gbrand:
            filters.pop("chipset_contains", None)
            log.info("node.select.brand_relaxed", brand=gbrand, category="video_card")
            results = search_components_impl(
                "video_card", filters=filters,
                sort_by="price", ascending=False, top_k=10,
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
        gbrand = (reqs.get("gpu_brand_preference") or "").lower()
        filters: Dict[str, Any] = {
            "price_lte": ceiling,
            "price_gte": cur_price * 1.15,
            "memory_gte": 4,
        }
        if gbrand == "nvidia":
            filters["chipset_contains"] = "GeForce"
        elif gbrand == "amd":
            filters["chipset_contains"] = "Radeon"
        # Price DESC - the priciest in-budget card is usually the best
        # (modern, more VRAM, more cores) - far more reliable than TDP DESC
        # which would favor ancient inefficient cards.
        results = search_components_impl(
            "video_card",
            filters=filters,
            sort_by="price", ascending=False,
            top_k=5,
        )
        return results[0] if results else None

    if category == "cpu":
        # CRITICAL: a CPU upgrade must stay on the SAME socket as the existing
        # motherboard, otherwise we silently create a build with a CPU/mobo
        # mismatch (e.g. AM4 board + LGA1851 Core Ultra). If no motherboard
        # exists yet, fall back to platform_preference; if that's "ANY",
        # restrict to mainstream sockets only.
        mb = build.get("motherboard") or {}
        mb_socket = (mb.get("socket") or "").upper()
        platform = (plan.get("platform_preference") or "any").upper()
        filters: Dict[str, Any] = {
            "price_lte": ceiling,
            "price_gte": cur_price * 1.15,
        }
        if not _need_discrete_gpu(reqs):
            filters["has_integrated_graphics"] = True
        brand = (reqs.get("cpu_brand_preference") or "").lower()
        if brand in ("amd", "intel"):
            filters["name_contains"] = brand.upper() if brand == "amd" else "Intel"
        if mb_socket:
            filters["socket"] = mb_socket
            results = search_components_impl(
                "cpu", filters=filters, sort_by="core_count", ascending=False, top_k=5,
            )
            return results[0] if results else None
        # No motherboard yet (uncommon during upgrade pass) - try each mainstream
        # socket in order so we don't accidentally pick an edge-case CPU.
        sockets = ([platform] if platform != "ANY" else list(_MAINSTREAM_SOCKETS))
        for s in sockets:
            f = {**filters, "socket": s}
            r = search_components_impl(
                "cpu", filters=f, sort_by="core_count", ascending=False, top_k=5,
            )
            if r:
                return r[0]
        return None

    if category == "memory":
        # CRITICAL: keep the DDR generation in sync with the motherboard AND
        # respect the board's max_memory and memory_slots so we don't
        # silently violate physical limits.
        mb = build.get("motherboard") or {}
        filters = {"price_lte": ceiling, "price_gte": cur_price * 1.15}
        ddr = mb.get("ddr_gen")
        if ddr:
            filters["ddr_gen"] = ddr
        # Don't blow past use-case appropriate memory caps even when filling.
        use_case = reqs.get("use_case", "general")
        if use_case in ("content_creation", "workstation"):
            use_cap = 128
        elif use_case == "gaming":
            use_cap = 64
        else:
            use_cap = 32
        # Hard cap = MIN of (use-case soft cap, motherboard max_memory).
        mb_max = mb.get("max_memory")
        if mb_max and float(mb_max) > 0:
            filters["total_gb_lte"] = min(use_cap, int(mb_max))
        else:
            filters["total_gb_lte"] = use_cap
        # Slot constraint: kit's module_count must fit in the board's slots.
        slots = mb.get("memory_slots")
        if slots:
            filters["module_count_lte"] = int(slots)
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
    (gaming -> GPU first; content creation -> CPU/memory first). When the
    user gave a range (budget_min_usd), the lower bound is a hard floor we
    push above.
    """
    budget = float(reqs.get("budget_usd") or 0)
    if not budget:
        return build

    def total_now() -> float:
        return round(sum(float(c.get("price", 0) or 0) for c in build.values() if c), 2)

    budget_min = reqs.get("budget_min_usd")
    # Soft target = 90% of upper bound; hard floor = max(85% of upper, user's lower bound).
    target = budget * 0.90
    floor = budget * 0.85
    if budget_min is not None:
        floor = max(floor, float(budget_min))
        target = max(target, float(budget_min))

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
                break
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

    # After upgrades, the CPU/GPU TDP may have grown, leaving the PSU
    # undersized. Re-pick the PSU against the latest build to keep
    # compatibility checks clean.
    _ensure_psu_sized(build, plan, reqs)
    return build


def _ensure_psu_sized(build: Dict[str, Any], plan: Dict[str, Any],
                      reqs: Dict[str, Any]) -> None:
    """If the current PSU can't handle the upgraded build, re-pick it."""
    from src.compatibility.power_rules import estimate_load_watts
    psu = build.get("power_supply")
    if not psu:
        return
    try:
        load = estimate_load_watts(_build_obj(build))
    except Exception:
        return
    cur_w = float(psu.get("wattage") or 0)
    if cur_w >= load * 1.10:  # 10% safety margin -> we're good
        return
    new_psu = _pick_psu(plan, reqs, build)
    if new_psu and float(new_psu.get("wattage") or 0) >= load * 1.10:
        log.info(
            "node.select.psu_resized",
            old=psu.get("name"),
            old_w=cur_w,
            new=new_psu.get("name"),
            new_w=new_psu.get("wattage"),
            load=round(load, 1),
        )
        build["power_supply"] = new_psu


# ---------------------------------------------------------------------------
# Budget-trim pass: downgrade components when the build overshoots the budget
# ---------------------------------------------------------------------------

def _try_downgrade(category: str, current: Dict[str, Any], max_price: float,
                   plan: Dict[str, Any], reqs: Dict[str, Any],
                   build: Dict[str, Any]) -> Dict[str, Any] | None:
    """Find a cheaper component for this category that stays compatible.

    `max_price` is the highest price we'll accept. We pick the closest item
    below that ceiling so we don't downgrade more than necessary.
    """
    cur_price = float(current.get("price", 0) or 0)
    if max_price >= cur_price - 1:
        return None  # nothing meaningful to gain

    if category == "cpu":
        # Downgrade must stay on the existing motherboard's socket.
        mb = build.get("motherboard") or {}
        mb_socket = (mb.get("socket") or "").upper()
        platform = (plan.get("platform_preference") or "any").upper()
        need_igpu = not _need_discrete_gpu(reqs)
        filters: Dict[str, Any] = {"price_lte": max_price, "price_gte": 40}
        if mb_socket:
            filters["socket"] = mb_socket
        elif platform != "ANY":
            filters["socket"] = platform
        if need_igpu:
            filters["has_integrated_graphics"] = True
        brand = (reqs.get("cpu_brand_preference") or "").lower()
        if brand in ("amd", "intel"):
            filters["name_contains"] = brand.upper() if brand == "amd" else "Intel"
        results = search_components_impl(
            "cpu", filters=filters, sort_by="core_count", ascending=False, top_k=5
        )
        if not results and (mb_socket or platform != "ANY"):
            filters.pop("socket", None)
            results = search_components_impl(
                "cpu", filters=filters, sort_by="core_count", ascending=False, top_k=5
            )
        return results[0] if results else None

    if category == "memory" and build.get("motherboard"):
        ddr = build["motherboard"].get("ddr_gen")
        # Keep at least the use-case-appropriate floor (16GB office, 8GB browsing).
        target_gb = 32 if reqs.get("use_case") in ("content_creation", "workstation") else 16
        filters = {"price_lte": max_price, "total_gb_gte": target_gb}
        if ddr:
            filters["ddr_gen"] = ddr
        results = search_components_impl(
            "memory", filters=filters, sort_by="total_gb", ascending=True, top_k=5
        )
        if not results and target_gb > 8:
            # Relax target if too aggressive (e.g. low-budget builds)
            filters["total_gb_gte"] = 8
            results = search_components_impl(
                "memory", filters=filters, sort_by="total_gb", ascending=True, top_k=5
            )
        return results[0] if results else None

    if category == "storage":
        # Floor: 256 GB - anything below that isn't usable in 2025.
        results = search_components_impl(
            "storage",
            filters={"price_lte": max_price, "capacity_gte": 256},
            sort_by="price", ascending=False, top_k=5,
        )
        return results[0] if results else None

    if category == "video_card":
        # If iGPU is fine, the better downgrade is to remove the GPU entirely
        # - signal that with None and let the caller delete it.
        if not _need_discrete_gpu(reqs):
            return None  # caller handles removal
        gbrand = (reqs.get("gpu_brand_preference") or "").lower()
        filters: Dict[str, Any] = {
            "price_lte": max_price,
            "price_gte": 50,
            "memory_gte": 4,
        }
        if gbrand == "nvidia":
            filters["chipset_contains"] = "GeForce"
        elif gbrand == "amd":
            filters["chipset_contains"] = "Radeon"
        # Price DESC - even when trimming we want the most modern card we
        # can still afford.
        results = search_components_impl(
            "video_card",
            filters=filters,
            sort_by="price", ascending=False, top_k=5,
        )
        if not results:
            filters.pop("memory_gte", None)
            results = search_components_impl(
                "video_card",
                filters=filters,
                sort_by="price", ascending=False, top_k=5,
            )
        return results[0] if results else None

    # Generic: pick the most expensive thing that still fits the new ceiling
    # (so we downgrade minimally).
    results = search_components_impl(
        category,
        filters={"price_lte": max_price},
        sort_by="price", ascending=False, top_k=5,
    )
    return results[0] if results else None


def _budget_trim_pass(build: Dict[str, Any], plan: Dict[str, Any],
                      reqs: Dict[str, Any]) -> Dict[str, Any]:
    """Downgrade components iteratively until total <= budget.

    Strategy:
    1. If the build has a discrete GPU but iGPU is sufficient, drop the GPU
       (single biggest win - usually saves $50+).
    2. Otherwise, find the most expensive non-critical component, replace it
       with the next-cheapest compatible alternative. Repeat until in budget
       or no more cuts possible.
    """
    budget = float(reqs.get("budget_usd") or 0)
    if not budget:
        return build

    def total_now() -> float:
        return round(sum(float(c.get("price", 0) or 0) for c in build.values() if c), 2)

    if total_now() <= budget:
        return build

    # Step 1: drop discrete GPU if iGPU works for this use case.
    if build.get("video_card") and not _need_discrete_gpu(reqs):
        gpu_price = float(build["video_card"].get("price", 0) or 0)
        log.info(
            "node.select.trim_drop_gpu",
            name=build["video_card"].get("name"),
            saved=round(gpu_price, 2),
        )
        build.pop("video_card", None)
        if total_now() <= budget:
            return build

    # Step 2: iteratively downgrade the highest-priced category.
    # Order: try the most flexible categories first - cpu_cooler, case,
    # power_supply, memory, storage, cpu, motherboard. We avoid touching
    # the motherboard unless we have to (changing it cascades to cpu).
    PREFER_ORDER = (
        "cpu_cooler", "case", "power_supply", "video_card",
        "memory", "storage", "cpu",
    )
    uncuttable: set[str] = set()  # categories that have no cheaper alternative
    max_iters = 14
    for _i in range(max_iters):
        over = total_now() - budget
        if over <= 0:
            return build
        # Pick the highest-priced flexible category still considered cuttable.
        target_cat: str | None = None
        target_price = 0.0
        for cat in PREFER_ORDER:
            if cat in uncuttable:
                continue
            comp = build.get(cat)
            if not comp:
                continue
            p = float(comp.get("price", 0) or 0)
            if p > target_price and p > 15:  # nothing below $15 is worth trimming
                target_cat = cat
                target_price = p
        if not target_cat:
            break  # every category is either tiny or already at its floor

        # Accept progressively more aggressive cuts: first try to absorb the
        # whole overage, then settle for ANY cheaper alternative.
        new_pick = _try_downgrade(
            target_cat, build[target_cat],
            max(15.0, target_price - over - 1),
            plan, reqs, build,
        )
        if not new_pick or new_pick.get("name") == build[target_cat].get("name"):
            # Fall back: try for any cheaper alternative at all (10% off).
            new_pick = _try_downgrade(
                target_cat, build[target_cat],
                max(15.0, target_price * 0.90),
                plan, reqs, build,
            )
        if not new_pick or new_pick.get("name") == build[target_cat].get("name"):
            log.info("node.select.trim_no_cheaper", category=target_cat)
            uncuttable.add(target_cat)
            continue
        log.info(
            "node.select.budget_trim",
            category=target_cat,
            old=build[target_cat].get("name"),
            new=new_pick.get("name"),
            delta=round(float(new_pick.get("price", 0) or 0) - target_price, 2),
        )
        build[target_cat] = new_pick

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
    # If the build overshoots the user's budget, the agent (not the user)
    # should bring it back in line. This runs on every attempt because a
    # critique-driven re-pick can also bust the budget.
    build = _budget_trim_pass(build, plan, reqs)
    # Final guard: make sure the PSU can still handle the (possibly trimmed)
    # build.
    _ensure_psu_sized(build, plan, reqs)

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
    intro: str | None = None,
    note_kind: str = "unreachable",
) -> str:
    """Markdown summary used when the LLM is unreachable / rate-limited but
    the deterministic agent layer already picked all the parts.

    `intro` lets the caller prepend any partial text the LLM did manage to
    return (so we don't throw away the LLM's character summary).
    `note_kind` controls the explanatory note at the top:
       - "unreachable": the LLM call failed outright
       - "truncated":   the LLM returned a tiny response (likely rate-limited)
    """
    settings = get_settings()
    if note_kind == "truncated":
        note = (
            f"Note: `{settings.llm_provider}` is heavily rate-limited right "
            f"now (most likely you've hit a daily token quota), so the model "
            f"only returned a partial response. I've filled in the rest "
            f"deterministically from the parts catalog. Wait ~60 seconds "
            f"before sending the next message, or switch provider in `.env`."
        )
    else:
        note = (
            f"Note: `{settings.llm_provider}` was briefly unreachable, so "
            f"this is a structured summary instead of the usual prose "
            f"response. Send your next message in ~60 seconds and the full "
            f"explanation should come back."
        )

    lines: List[str] = []
    if intro:
        lines.append(intro.strip())
        lines.append("")
    lines.append(note)
    lines.append("")
    lines.append("| Component | Part | Price |")
    lines.append("|---|---|---|")
    for cat in CATEGORY_ORDER:
        comp = build.get(cat)
        if comp and isinstance(comp, dict):
            lines.append(
                f"| {cat} | {comp.get('name', '?')} | "
                f"${float(comp.get('price', 0) or 0):.2f} |"
            )
    lines.append("")
    budget = reqs.get("budget_usd")
    budget_min = reqs.get("budget_min_usd")
    if budget_min and budget:
        budget_str = f" (your budget range: ${budget_min:.0f}-${budget:.0f})"
    elif budget:
        budget_str = f" (your budget: ${budget})"
    else:
        budget_str = ""
    lines.append(f"**Total: ${total:.2f}**{budget_str}")

    # Quick deterministic rationale highlighting the headline picks.
    headline_picks = []
    cpu = build.get("cpu") or {}
    gpu = build.get("video_card") or {}
    mem = build.get("memory") or {}
    sto = build.get("storage") or {}
    if cpu.get("name"):
        headline_picks.append(
            f"- **CPU** `{cpu['name']}` ({cpu.get('core_count', '?')} cores) - "
            f"the strongest in-budget option for your use case."
        )
    if gpu.get("name") and gpu.get("price", 0) > 80:
        headline_picks.append(
            f"- **Video card** `{gpu['name']}` ({gpu.get('chipset') or '?'}) - "
            f"selected for the performance/$ trade-off."
        )
    if mem.get("name"):
        total_gb = mem.get("total_gb") or "?"
        headline_picks.append(
            f"- **Memory** `{mem['name']}` ({total_gb} GB total) - matches the "
            f"motherboard's DDR generation."
        )
    if sto.get("name"):
        cap = sto.get("capacity") or "?"
        headline_picks.append(
            f"- **Storage** `{sto['name']}` ({cap} GB) - fast SSD within budget."
        )
    if headline_picks:
        lines.append("")
        lines.append("**Why these picks:**")
        lines.extend(headline_picks)

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
        "Want me to swap anything? Just say what you would like to change "
        "(cheaper, quieter, smaller, more storage, etc.)."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. responder
# ---------------------------------------------------------------------------

def _normalize_response_markdown(text: str) -> str:
    """Repair common GFM rendering pitfalls in the LLM's response.

    The Streamlit / GFM markdown parser greedily extends a table until it
    sees a blank line. LLMs often forget that rule and start writing prose
    immediately after the last `|`-row, which then gets rendered as more
    table rows. This normaliser:
      1. Inserts a blank line after the LAST consecutive `|...|` line in
         every table block.
      2. Inserts a blank line before any `### ` heading that follows a
         non-blank line.
      3. Strips `*...*` wrappers around plain dollar amounts (e.g.
         `*$697.70*`) which can render as LaTeX-ish italics.
    """
    if not text:
        return text

    lines = text.splitlines()
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        is_table_row = line.lstrip().startswith("|")
        # End of a table block? Next line exists, isn't blank, isn't a |-row.
        next_line = lines[i + 1] if i + 1 < len(lines) else None
        if (
            is_table_row
            and next_line is not None
            and next_line.strip() != ""
            and not next_line.lstrip().startswith("|")
        ):
            out.append("")  # force the table block to close
        # Blank line before a heading if previous line had content.
        if (
            next_line is not None
            and next_line.lstrip().startswith("### ")
            and line.strip() != ""
            and not is_table_row  # tables already handled above
        ):
            out.append("")
        i += 1

    cleaned = "\n".join(out)
    if text.endswith("\n") and not cleaned.endswith("\n"):
        cleaned += "\n"
    # Strip *...* italics around plain dollar amounts.
    cleaned = re.sub(r"\*\s*(\$\s*\d[\d,]*(?:\.\d+)?)\s*\*", r"\1", cleaned)
    return cleaned


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
    content = (ai.content or "").strip()

    # Case 1: the LLM was completely unreachable - fall back to fully
    # deterministic markdown using the picked parts.
    if "unable to reach" in content.lower() and build:
        content = _format_build_response(
            reqs, build, issues, total, intro=None, note_kind="unreachable"
        )

    # Case 2: the LLM responded but the reply is suspiciously short and
    # doesn't even contain the parts table. Most common cause is GitHub
    # Models throttling output tokens when the daily quota is exhausted.
    # Treat the short reply as an "intro" and append a deterministic
    # build summary so the user still sees the full information.
    elif build and len(content) < 400 and "|" not in content:
        log.warning(
            "node.respond.short_llm_output",
            content_len=len(content),
            content_preview=content[:160],
        )
        content = _format_build_response(
            reqs, build, issues, total, intro=content, note_kind="truncated"
        )

    # Safety net: if the LLM forgot the comparison section, append our
    # deterministic one.
    if comparison_md and "what changed" not in content.lower():
        content = content.rstrip() + "\n\n" + comparison_md

    # Normalise markdown so the build table doesn't swallow subsequent
    # paragraphs as extra rows (GFM rule: tables end at the first blank line).
    content = _normalize_response_markdown(content)

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

    # ---- Deterministic-first intent classification ----
    # For short, canonical feedback phrases ("cheaper", "approve", "more
    # storage", "increase budget to $X", "swap CPU with AMD", etc.) we trust
    # our heuristic over the LLM. LLMs frequently hallucinate budget numbers
    # for bare phrases like "cheaper" because they pattern-match against
    # few-shot examples like "cheaper, around $1000" - we'd rather miss a
    # nuanced phrasing than silently change the user's budget.
    fb: Dict[str, Any] | None = None
    if user_text and len(user_text.strip()) <= 80:
        det = _heuristic_feedback(user_text)
        if det:
            fb = det
            log.info(
                "node.feedback.heuristic_first",
                intent=fb.get("intent"),
                deltas=list((fb.get("delta_constraints") or {}).keys()),
            )

    if fb is None:
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

        # If the LLM bailed out as "unclear" (often because the small model
        # produced non-JSON / truncated output), try to recover with a
        # deterministic intent classifier on the raw text.
        if fb.get("intent") in (None, "unclear"):
            recovered = _heuristic_feedback(user_text)
            if recovered:
                log.info(
                    "node.feedback.heuristic_recovery",
                    original=fb.get("intent"),
                    recovered=recovered.get("intent"),
                )
                fb = recovered

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
        # Same safety net as the gatherer: only trust the LLM's off_topic
        # classification if our positive PC-signal regex doesn't fire.
        if not _ON_TOPIC_HINTS.search(user_text):
            return {
                "feedback": fb,
                "final_response": OFF_TOPIC_REPLY,
                "mode": "respond",
            }
        log.info("node.feedback.off_topic_llm_override", reason="on_topic_hint_matched")
        # Fall through to deterministic handling - treat as a budget/swap
        # request and let the rest of the function classify it.
        fb["intent"] = "unclear"
        intent = "unclear"
    if intent == "unclear":
        return {
            "feedback": fb,
            "final_response": (
                "I am not sure what you would like to change. Try something "
                "like: 'increase budget to $700', 'make it cheaper', "
                "'more storage', 'quieter', or 'compare with a $900 budget'."
            ),
            "mode": "respond",
        }

    # Snapshot the existing build so the responder can produce a diff.
    prev_build = dict(state.get("build") or {})
    prev_budget = (state.get("requirements") or {}).get("budget_usd")

    # Apply deltas to requirements + plan, then re-plan from scratch.
    reqs = dict(state.get("requirements") or {})
    # Per-category price ceilings are one-shot - clear any left over from a
    # previous turn so they do not silently keep capping picks forever.
    reqs.pop("category_price_ceilings", None)
    deltas = fb.get("delta_constraints") or {}

    # ---- Relative budget moves: "double my budget" / "halve it" ----
    mult = deltas.pop("budget_multiplier", None)
    if mult is not None:
        cur = float(reqs.get("budget_usd") or 0)
        if cur > 0:
            new_budget = round(cur * float(mult), 2)
            deltas["budget_usd"] = new_budget
            log.info("node.feedback.budget_multiplier",
                     multiplier=mult, old=cur, new=new_budget)
        else:
            log.warning("node.feedback.budget_multiplier_no_current_budget",
                        multiplier=mult)

    # Translate the relative "make it cheaper" signal into a concrete
    # budget reduction. Target ~80% of the LOWER of (current build total,
    # current budget) so we never accidentally raise the budget when the
    # previous build overshot it.
    if deltas.get("price_lower"):
        current_total = sum(
            float(c.get("price", 0) or 0)
            for c in (state.get("build") or {}).values() if c
        )
        cur_budget = float(reqs.get("budget_usd") or 0)
        # Use the smaller of the two so a previous over-budget build does not
        # become the new ceiling. If only one is set, use that.
        candidates = [v for v in (current_total, cur_budget) if v > 0]
        anchor = min(candidates) if candidates else 0
        if anchor > 0:
            reqs["budget_usd"] = round(anchor * 0.80, 2)
            reqs.pop("budget_min_usd", None)
            log.info("node.feedback.price_lower_anchor",
                     current_total=round(current_total, 2),
                     current_budget=cur_budget,
                     anchor=anchor,
                     new_budget=reqs["budget_usd"])
        deltas.pop("price_lower", None)

    # "less expensive <category>" - cap that one category's price at 80% of
    # its current pick (so the new pick is meaningfully cheaper without
    # shrinking the whole budget). Stored on reqs so the per-category picker
    # can honor it.
    if deltas.get("price_lower_category"):
        cat = deltas.pop("price_lower_category")
        cur = (state.get("build") or {}).get(cat) or {}
        cur_price = float(cur.get("price", 0) or 0)
        if cur_price > 0:
            ceilings = dict(reqs.get("category_price_ceilings") or {})
            ceilings[cat] = round(cur_price * 0.80, 2)
            reqs["category_price_ceilings"] = ceilings
            log.info("node.feedback.category_price_ceiling",
                     category=cat, old_price=cur_price,
                     new_ceiling=ceilings[cat])

    # Structured deltas - apply directly to Requirements fields.
    KNOWN_FIELD_DELTAS = {
        "budget_usd", "budget_min_usd", "noise_preference", "use_case",
        "form_factor_preference", "os_needed",
        "cpu_brand_preference", "gpu_brand_preference",
    }
    for k in KNOWN_FIELD_DELTAS:
        if k in deltas and deltas[k] is not None:
            reqs[k] = deltas[k]
    # Backfill range from raw text if the LLM missed it.
    lo, hi = _extract_budget_range(user_text)
    if hi is not None and "budget_usd" not in deltas:
        reqs["budget_usd"] = hi
    if lo is not None and "budget_min_usd" not in deltas:
        reqs["budget_min_usd"] = lo
    # Backfill use_case from heuristic on the latest message if still vague.
    if reqs.get("use_case", "general") in (None, "", "general"):
        uc = _heuristic_use_case(user_text)
        if uc:
            reqs["use_case"] = uc

    # Any remaining delta keys (e.g. storage_capacity_gte, brand=AMD) are
    # passed to the planner as explicit must-haves.
    extra = []
    for k, v in deltas.items():
        if k in KNOWN_FIELD_DELTAS or v is None:
            continue
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
