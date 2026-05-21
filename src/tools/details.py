"""Look up the full details of a specific named component."""
from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.tools import tool

from src.data.loader import get_catalog
from src.tools.search import _serialize_row


def get_component_details_impl(category: str, name: str) -> Optional[Dict[str, Any]]:
    catalog = get_catalog()
    df = catalog[category]
    matches = df[df["name"].astype(str).str.lower() == name.lower()]
    if matches.empty:
        # try a contains-match fallback
        matches = df[df["name"].astype(str).str.contains(name, case=False, na=False)]
    if matches.empty:
        return None
    return _serialize_row(matches.iloc[0])


@tool("get_component_details")
def get_component_details(category: str, name: str) -> Optional[Dict[str, Any]]:
    """Look up the full row for a component by name.

    Useful when the selector wants to confirm a part exists in the catalog
    before committing to it.

    Args:
        category: Component category (cpu, motherboard, memory, ...).
        name: Exact or partial component name.

    Returns:
        The full component row as a dict, or None if no match found.
    """
    return get_component_details_impl(category, name)
