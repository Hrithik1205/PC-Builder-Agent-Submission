"""Tests for socket inference and DDR mapping."""
from src.compatibility.socket_map import ddr_gen_for_socket, infer_socket


def test_infer_socket_amd_zen_variants():
    assert infer_socket("Zen 5") == "AM5"
    assert infer_socket("Zen 4") == "AM5"
    assert infer_socket("Zen 3") == "AM4"


def test_infer_socket_intel():
    assert infer_socket("Raptor Lake Refresh") == "LGA1700"
    assert infer_socket("Alder Lake") == "LGA1700"
    assert infer_socket("Arrow Lake") == "LGA1851"


def test_infer_socket_case_insensitive_and_trimmed():
    assert infer_socket("  zen 4  ") == "AM5"
    assert infer_socket("ZEN 3") == "AM4"


def test_infer_socket_unknown_returns_none():
    assert infer_socket("Mystery Lake") is None
    assert infer_socket(None) is None
    assert infer_socket("") is None


def test_ddr_gen_for_socket():
    assert ddr_gen_for_socket("AM5") == 5
    assert ddr_gen_for_socket("AM4") == 4
    assert ddr_gen_for_socket("LGA1700") == 5  # defaults to DDR5 era
    assert ddr_gen_for_socket("LGA1200") == 4
    assert ddr_gen_for_socket(None) is None
    assert ddr_gen_for_socket("UNKNOWN") is None
