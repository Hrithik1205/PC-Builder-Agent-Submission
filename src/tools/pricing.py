"""Build pricing + power-estimation tools."""
from __future__ import annotations

from typing import Any, Dict

from langchain_core.tools import tool

from src.compatibility.power_rules import estimate_load_watts
from src.data.schemas import Build


def total_price_impl(build_dict: Dict[str, Any]) -> float:
    build = Build(**build_dict)
    return build.total_price()


def estimate_total_power_impl(build_dict: Dict[str, Any]) -> int:
    build = Build(**build_dict)
    return estimate_load_watts(build)


@tool("total_price")
def total_price(build: Dict[str, Any]) -> float:
    """Sum the prices of all parts in a (partial) build.

    Args:
        build: Dict mapping category -> component dict (any subset).

    Returns:
        Total cost in USD, rounded to two decimals.
    """
    return total_price_impl(build)


@tool("estimate_total_power")
def estimate_total_power(build: Dict[str, Any]) -> int:
    """Estimate the peak system power draw of a build in watts.

    Includes CPU TDP, an estimated GPU TDP, a fixed 50W overhead for
    storage/fans/RGB, and a 100W safety headroom for PSU sizing.

    Args:
        build: Dict mapping category -> component dict (any subset).

    Returns:
        Estimated peak draw in watts.
    """
    return estimate_total_power_impl(build)
