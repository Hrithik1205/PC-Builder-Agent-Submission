"""Case <-> motherboard / GPU compatibility heuristics.

The dataset's `case.csv` has only `type` (e.g. "ATX Mid Tower") and an
`external_volume`. There is no explicit max GPU length or supported form
factor list, so we encode reasonable industry defaults here. These are
heuristics, not facts; warnings rather than errors where appropriate.
"""
from __future__ import annotations

from typing import Iterable, List

from src.data.schemas import Build, Issue


# Each case type supports its own form factor AND any smaller ones.
# Form-factor size order from largest to smallest:
SIZE_ORDER = ["EATX", "ATX", "Micro ATX", "Mini ITX"]


CASE_TYPE_MAX_FORM_FACTOR: dict[str, str] = {
    # Full towers / super towers
    "ATX Full Tower": "EATX",
    "EATX Full Tower": "EATX",
    "Full Tower": "EATX",
    # Mid towers
    "ATX Mid Tower": "ATX",
    "Mid Tower": "ATX",
    # Mini towers / desktops
    "ATX Mini Tower": "Micro ATX",
    "MicroATX Mid Tower": "Micro ATX",
    "MicroATX Mini Tower": "Micro ATX",
    "MicroATX Slim": "Micro ATX",
    "MicroATX Desktop": "Micro ATX",
    "HTPC": "Micro ATX",
    # Mini-ITX
    "Mini ITX Tower": "Mini ITX",
    "Mini ITX Desktop": "Mini ITX",
}


# Rough max GPU length per case category (mm). Conservative.
CASE_TYPE_MAX_GPU_MM: dict[str, int] = {
    "ATX Full Tower": 420,
    "EATX Full Tower": 450,
    "Full Tower": 420,
    "ATX Mid Tower": 360,
    "Mid Tower": 360,
    "ATX Mini Tower": 320,
    "MicroATX Mid Tower": 320,
    "MicroATX Mini Tower": 280,
    "MicroATX Slim": 220,
    "MicroATX Desktop": 240,
    "HTPC": 220,
    "Mini ITX Tower": 280,
    "Mini ITX Desktop": 220,
}


def _supported_form_factors(case_type: str) -> Iterable[str]:
    """Return the set of motherboard form factors a case can fit."""
    max_ff = CASE_TYPE_MAX_FORM_FACTOR.get(case_type)
    if not max_ff:
        return SIZE_ORDER  # unknown case type -> assume universal, warn instead
    idx = SIZE_ORDER.index(max_ff)
    return SIZE_ORDER[idx:]


def check_case_motherboard(build: Build) -> List[Issue]:
    issues: List[Issue] = []
    if build.case is None or build.motherboard is None:
        return issues
    case_type = (build.case.type or "").strip()
    mobo_ff = (build.motherboard.form_factor or "").strip()
    if not case_type or not mobo_ff:
        return issues
    supported = list(_supported_form_factors(case_type))
    if mobo_ff not in supported and case_type in CASE_TYPE_MAX_FORM_FACTOR:
        issues.append(Issue(
            severity="error",
            rule="case_form_factor_mismatch",
            message=(
                f"Case '{build.case.name}' ({case_type}) does not support "
                f"motherboard form factor '{mobo_ff}'. Supported: {supported}."
            ),
            components=[build.case.name, build.motherboard.name],
        ))
    return issues


def check_case_gpu_length(build: Build) -> List[Issue]:
    issues: List[Issue] = []
    if build.case is None or build.video_card is None:
        return issues
    gpu_len = build.video_card.length
    if not gpu_len:
        return issues
    case_type = (build.case.type or "").strip()
    max_len = CASE_TYPE_MAX_GPU_MM.get(case_type)
    if max_len and gpu_len > max_len:
        issues.append(Issue(
            severity="warn",
            rule="gpu_length_may_not_fit",
            message=(
                f"GPU '{build.video_card.name}' is {gpu_len}mm long; case "
                f"'{build.case.name}' ({case_type}) typically supports up to "
                f"{max_len}mm. Verify against the case spec sheet."
            ),
            components=[build.video_card.name, build.case.name],
        ))
    return issues
