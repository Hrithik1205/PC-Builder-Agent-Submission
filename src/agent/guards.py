"""Input validation, prompt-injection sniffing, hallucination guards."""
from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Tuple

from src.data.loader import get_catalog


# Strings strongly indicative of jailbreak / prompt-extraction attempts.
# Match conservatively - false positives are worse than letting an obvious
# attempt through (we always have the LLM-side TOPIC GATE as second line).
_INJECTION_PATTERNS = [
    r"\bignore (all|any|every|previous|prior|above) (instructions?|prompts?|rules?)\b",
    r"\bforget (everything|all|previous|prior|your) (instructions?|prompts?|rules?|context)?\b",
    r"\byou are now\b",
    r"\bdisregard (the|all|previous|any) (instructions?|rules?)\b",
    r"\bact as (a|an) [a-z]+\b",
    r"\bpretend (to be|you are)\b",
    r"\b(roleplay|role-play) as\b",
    r"\bdeveloper mode\b",
    r"\bjailbreak\b",
    r"\bsystem prompt\b",
    r"\bshow (me )?your (prompt|instructions?|rules?|guidelines?)\b",
    r"\b(what|which) (are|is) your (prompt|instructions?|rules?|guidelines?)\b",
    r"\breveal your (prompt|instructions?|system message)\b",
    r"\bprint your (system )?(prompt|instructions?)\b",
]


MAX_MESSAGE_CHARS = 4000


def validate_user_message(text: str) -> Tuple[bool, Optional[str]]:
    """Return (ok, reason_if_blocked)."""
    if not text or not text.strip():
        return False, "Empty message - please describe the PC you want to build."

    if len(text) > MAX_MESSAGE_CHARS:
        return False, (
            f"Message too long ({len(text)} chars). Please keep it under "
            f"{MAX_MESSAGE_CHARS} characters."
        )

    lower = text.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lower):
            return False, (
                "I can only help with PC build recommendations. "
                "Please describe the kind of PC you want to build (use case, budget)."
            )
    return True, None


def validate_component_exists(category: str, name: str) -> bool:
    """Check whether the LLM-named component is actually in the catalog."""
    catalog = get_catalog()
    if category not in catalog.categories:
        return False
    df = catalog[category]
    return bool((df["name"].astype(str).str.lower() == name.lower()).any())


def filter_real_components(build: Dict[str, dict]) -> Tuple[Dict[str, dict], list[str]]:
    """Strip any component the LLM 'made up'. Returns (clean_build, dropped)."""
    clean: Dict[str, dict] = {}
    dropped: list[str] = []
    for cat, comp in (build or {}).items():
        if comp is None:
            continue
        name = comp.get("name") if isinstance(comp, dict) else None
        if name and validate_component_exists(cat, name):
            clean[cat] = comp
        else:
            dropped.append(f"{cat}:{name}")
    return clean, dropped
