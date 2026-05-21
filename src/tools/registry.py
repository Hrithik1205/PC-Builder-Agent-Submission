"""Convenience registry of all agent-facing tools."""
from src.tools.search import search_components
from src.tools.details import get_component_details
from src.tools.compatibility_tool import check_compatibility
from src.tools.pricing import total_price, estimate_total_power


ALL_TOOLS = [
    search_components,
    get_component_details,
    check_compatibility,
    total_price,
    estimate_total_power,
]


def get_tools():
    """Return the canonical list of tools to bind to the LLM."""
    return ALL_TOOLS
