"""Component search - the agent's primary 'act' tool.

Exposed to the LLM as a LangChain tool. The LLM passes a category + filter
dict (e.g. `{"socket": "AM5", "price_lte": 250}`); we run a pandas query
and return the top-k rows as plain dicts so the LLM can pick one.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.tools import tool

from src.data.loader import get_catalog


# Default sort columns per category (cheaper-is-better unless overridden)
DEFAULT_SORT: Dict[str, tuple[str, bool]] = {
    "cpu": ("price", True),
    "motherboard": ("price", True),
    "memory": ("total_gb", False),
    "video_card": ("estimated_tdp", False),
    "power_supply": ("wattage", False),
    "case": ("price", True),
    "storage": ("capacity", False),
    "cpu_cooler": ("price", True),
}


def _apply_filter(df: pd.DataFrame, key: str, value: Any) -> pd.DataFrame:
    """Apply a single filter clause. Supports suffixed operators."""
    if value is None:
        return df

    # ---- operator suffixes ----
    if key.endswith("_lte"):
        col = key[:-4]
        if col in df.columns:
            return df[df[col] <= float(value)]
        return df
    if key.endswith("_gte"):
        col = key[:-4]
        if col in df.columns:
            return df[df[col] >= float(value)]
        return df
    if key.endswith("_lt"):
        col = key[:-3]
        if col in df.columns:
            return df[df[col] < float(value)]
        return df
    if key.endswith("_gt"):
        col = key[:-3]
        if col in df.columns:
            return df[df[col] > float(value)]
        return df
    if key.endswith("_contains"):
        col = key[:-9]
        if col in df.columns:
            return df[df[col].astype(str).str.contains(str(value), case=False, na=False)]
        return df
    if key.endswith("_in"):
        col = key[: -len("_in")]
        if col in df.columns and isinstance(value, (list, tuple)):
            return df[df[col].isin(list(value))]
        return df

    # ---- exact match ----
    if key in df.columns:
        if isinstance(value, str):
            return df[df[key].astype(str).str.lower() == value.lower()]
        return df[df[key] == value]
    return df


def _serialize_row(row: pd.Series) -> Dict[str, Any]:
    """Convert a pandas row to a JSON-safe dict (no NaN, no numpy types)."""
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, float) and math.isnan(v):
            out[k] = None
        elif hasattr(v, "item"):  # numpy scalar
            out[k] = v.item()
        else:
            out[k] = v
    return out


def search_components_impl(
    category: str,
    filters: Optional[Dict[str, Any]] = None,
    sort_by: Optional[str] = None,
    ascending: Optional[bool] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Plain-Python search used by tests and by the tool wrapper.

    Returns up to `top_k` rows (as dicts) for the given category, after
    applying the filter dict and the sort. Sensible defaults if not provided.
    """
    catalog = get_catalog()
    df = catalog[category]
    if filters:
        for key, value in filters.items():
            df = _apply_filter(df, key, value)

    sort_col, sort_asc = DEFAULT_SORT.get(category, ("price", True))
    if sort_by:
        sort_col = sort_by
    if ascending is not None:
        sort_asc = ascending
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=sort_asc, na_position="last")

    top_k = max(1, min(top_k, 50))
    rows = df.head(top_k).to_dict(orient="records")
    return [_serialize_row(pd.Series(r)) for r in rows]


@tool("search_components")
def search_components(
    category: str,
    filters: Optional[Dict[str, Any]] = None,
    sort_by: Optional[str] = None,
    ascending: Optional[bool] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Search the components catalog and return matching rows.

    Args:
        category: One of cpu, motherboard, memory, video_card, power_supply,
            case, storage, cpu_cooler.
        filters: Dict of column filters. Supports suffix operators _lte, _gte,
            _lt, _gt, _contains, _in. Bare keys are exact match. For instance
            socket AM5 with price_lte 250.
        sort_by: Optional column name to sort by.
        ascending: Optional bool to override the default sort direction.
        top_k: Max number of rows to return (1-50, default 10).

    Returns:
        List of dicts, one per matching component, with all available columns.
    """
    return search_components_impl(category, filters, sort_by, ascending, top_k)
