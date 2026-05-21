"""Tests for the deterministic compatibility engine."""
from src.compatibility.engine import check_build, has_errors
from src.compatibility.power_rules import estimate_load_watts
from src.data.schemas import (
    CPU, CPUCooler, Build, Case, Memory, Motherboard,
    PowerSupply, Storage, VideoCard,
)


def make_build(**overrides) -> Build:
    """Helper: a known-good AM5 gaming build that all rules should pass."""
    defaults = dict(
        cpu=CPU(name="AMD Ryzen 7 7800X3D", price=340.0, core_count=8,
                microarchitecture="Zen 4", tdp=120, socket="AM5"),
        motherboard=Motherboard(name="MSI B650 GAMING PLUS", price=170.0,
                                socket="AM5", form_factor="ATX",
                                max_memory=192, memory_slots=4, ddr_gen=5),
        memory=Memory(name="G.Skill Flare X5 32GB", price=90.0, ddr_gen=5,
                      mt_s=6000, module_count=2, module_gb=16, total_gb=32),
        video_card=VideoCard(name="RTX 4070", price=550.0,
                             chipset="GeForce RTX 4070", memory=12,
                             length=285, estimated_tdp=200),
        storage=Storage(name="Samsung 990 Pro 1TB", price=120.0, capacity=1000,
                        type="SSD", form_factor="M.2-2280",
                        interface="M.2 PCIe 4.0 X4"),
        power_supply=PowerSupply(name="Corsair RM750e", price=120.0,
                                 wattage=750, efficiency="gold"),
        case=Case(name="Phanteks XT PRO", price=68.0, type="ATX Mid Tower"),
        cpu_cooler=CPUCooler(name="Thermalright Peerless Assassin", price=35.0),
    )
    defaults.update(overrides)
    return Build(**defaults)


def test_good_build_passes():
    build = make_build()
    issues = check_build(build)
    assert not has_errors(issues), f"Unexpected errors: {issues}"


def test_cpu_socket_mismatch_detected():
    build = make_build(
        cpu=CPU(name="AMD Ryzen 5 5600", price=125.0, core_count=6,
                microarchitecture="Zen 3", tdp=65, socket="AM4"),
    )
    issues = check_build(build)
    rules = {i.rule for i in issues}
    assert "cpu_socket_mismatch" in rules
    assert has_errors(issues)


def test_memory_ddr_mismatch_detected():
    build = make_build(
        memory=Memory(name="Corsair LPX 16GB DDR4", price=40.0, ddr_gen=4,
                      mt_s=3200, module_count=2, module_gb=8, total_gb=16),
    )
    issues = check_build(build)
    assert any(i.rule == "memory_ddr_mismatch" for i in issues)
    assert has_errors(issues)


def test_memory_capacity_exceeded():
    build = make_build(
        motherboard=Motherboard(name="Cheap A520", price=70.0, socket="AM5",
                                form_factor="Micro ATX", max_memory=64,
                                memory_slots=2, ddr_gen=5),
        memory=Memory(name="128GB Kit", price=400.0, ddr_gen=5, mt_s=6000,
                      module_count=2, module_gb=64, total_gb=128),
    )
    issues = check_build(build)
    assert any(i.rule == "memory_capacity_exceeded" for i in issues)


def test_memory_slot_count_exceeded():
    build = make_build(
        motherboard=Motherboard(name="2-slot board", price=80.0, socket="AM5",
                                form_factor="Mini ITX", max_memory=64,
                                memory_slots=2, ddr_gen=5),
        memory=Memory(name="4x8GB Kit", price=100.0, ddr_gen=5, mt_s=6000,
                      module_count=4, module_gb=8, total_gb=32),
    )
    issues = check_build(build)
    assert any(i.rule == "memory_slot_count_exceeded" for i in issues)


def test_case_form_factor_mismatch():
    build = make_build(
        motherboard=Motherboard(name="ATX board", price=200.0, socket="AM5",
                                form_factor="ATX", max_memory=192,
                                memory_slots=4, ddr_gen=5),
        case=Case(name="Tiny ITX case", price=80.0, type="Mini ITX Tower"),
    )
    issues = check_build(build)
    assert any(i.rule == "case_form_factor_mismatch" for i in issues)


def test_psu_undersized_for_high_tdp_gpu():
    build = make_build(
        video_card=VideoCard(name="RTX 4090", price=1800.0,
                             chipset="GeForce RTX 4090", memory=24,
                             length=336, estimated_tdp=450),
        power_supply=PowerSupply(name="500W bronze", price=50.0, wattage=500,
                                 efficiency="bronze"),
    )
    issues = check_build(build)
    assert any(i.rule == "psu_undersized" for i in issues)


def test_estimate_load_includes_cpu_and_gpu_and_headroom():
    build = make_build()
    load = estimate_load_watts(build)
    # 120 (CPU) + 200 (GPU) + 50 (other) + 100 (headroom) = 470
    assert load == 470


def test_partial_build_does_not_crash():
    build = Build(cpu=CPU(name="AMD Ryzen 5 7600", price=170.0, core_count=6,
                          microarchitecture="Zen 4", tdp=105, socket="AM5"))
    issues = check_build(build)
    assert not has_errors(issues)  # nothing else picked yet => no conflicts
