"""Tests for the search tool (uses mocked tiny catalog, no network)."""
from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture
def tiny_catalog():
    """Build a minimal Catalog with two CPUs and two motherboards."""
    from src.data.loader import Catalog

    cpus = pd.DataFrame([
        {"name": "AMD Ryzen 5 7600", "price": 170.0, "core_count": 6,
         "microarchitecture": "Zen 4", "tdp": 105, "socket": "AM5"},
        {"name": "Intel Core i5-12400F", "price": 109.0, "core_count": 6,
         "microarchitecture": "Alder Lake", "tdp": 65, "socket": "LGA1700"},
    ])
    mobos = pd.DataFrame([
        {"name": "MSI B650 GAMING PLUS", "price": 170.0, "socket": "AM5",
         "form_factor": "ATX", "max_memory": 192, "memory_slots": 4, "ddr_gen": 5},
        {"name": "ASUS Z790-P", "price": 220.0, "socket": "LGA1700",
         "form_factor": "ATX", "max_memory": 192, "memory_slots": 4, "ddr_gen": 5},
    ])
    return Catalog({"cpu": cpus, "motherboard": mobos})


def test_search_components_filters_by_socket(tiny_catalog):
    with patch("src.tools.search.get_catalog", return_value=tiny_catalog):
        from src.tools.search import search_components_impl
        results = search_components_impl("cpu", filters={"socket": "AM5"})
        assert len(results) == 1
        assert results[0]["name"] == "AMD Ryzen 5 7600"


def test_search_components_filters_price_lte(tiny_catalog):
    with patch("src.tools.search.get_catalog", return_value=tiny_catalog):
        from src.tools.search import search_components_impl
        results = search_components_impl("cpu", filters={"price_lte": 150})
        assert len(results) == 1
        assert results[0]["name"] == "Intel Core i5-12400F"


def test_search_components_top_k(tiny_catalog):
    with patch("src.tools.search.get_catalog", return_value=tiny_catalog):
        from src.tools.search import search_components_impl
        results = search_components_impl("cpu", top_k=1)
        assert len(results) == 1


def test_search_components_no_match_returns_empty(tiny_catalog):
    with patch("src.tools.search.get_catalog", return_value=tiny_catalog):
        from src.tools.search import search_components_impl
        results = search_components_impl("cpu", filters={"socket": "sTR5"})
        assert results == []
