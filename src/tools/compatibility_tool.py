"""LangChain tool wrapper around the compatibility engine."""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.tools import tool

from src.compatibility.engine import check_build
from src.data.schemas import Build


def _build_from_dict(build_dict: Dict[str, Any]) -> Build:
    """Coerce a dict of `{category: component_dict}` into a Build model.

    Tolerates partial builds (only some categories filled in) so the LLM
    can validate work-in-progress configurations.
    """
    # Pydantic models can absorb extra fields as long as required ones are present
    return Build(**build_dict)


def check_build_impl(build_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    build = _build_from_dict(build_dict)
    issues = check_build(build)
    return [i.model_dump() for i in issues]


@tool("check_compatibility")
def check_compatibility(build: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run deterministic compatibility checks on a (partial or complete) build.

    Args:
        build: Dict mapping component category to its row dict. Keys:
            cpu, motherboard, memory, video_card, storage, power_supply,
            case, cpu_cooler. Any subset is allowed.

    Returns:
        List of issues. Each has severity (error|warn|info), rule, message,
        and the affected component names. Empty list means no problems.
    """
    return check_build_impl(build)
