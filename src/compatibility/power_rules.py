"""PSU sizing rules."""
from __future__ import annotations

from typing import List

from src.data.loader import estimate_gpu_tdp
from src.data.schemas import Build, Issue


HEADROOM_W = 100  # safety margin on top of estimated load
OTHER_COMPONENTS_W = 50  # storage, fans, RGB, etc.


def estimate_load_watts(build: Build) -> int:
    """Estimate the system's peak load in watts."""
    cpu_tdp = (build.cpu.tdp or 0) if build.cpu else 0
    gpu_tdp = 0
    if build.video_card and build.video_card.chipset:
        gpu_tdp = build.video_card.estimated_tdp or estimate_gpu_tdp(
            build.video_card.chipset
        )
    return cpu_tdp + gpu_tdp + OTHER_COMPONENTS_W + HEADROOM_W


def check_psu_wattage(build: Build) -> List[Issue]:
    issues: List[Issue] = []
    if build.power_supply is None:
        return issues
    needed = estimate_load_watts(build)
    have = build.power_supply.wattage or 0
    if needed and have and have < needed:
        issues.append(Issue(
            severity="error",
            rule="psu_undersized",
            message=(
                f"PSU '{build.power_supply.name}' is {have}W but estimated peak "
                f"system load (CPU+GPU+overhead+headroom) is ~{needed}W."
            ),
            components=[build.power_supply.name],
        ))
    elif needed and have and have > needed * 2.5:
        issues.append(Issue(
            severity="info",
            rule="psu_oversized",
            message=(
                f"PSU '{build.power_supply.name}' ({have}W) is significantly "
                f"larger than needed (~{needed}W). Not a problem, but a smaller "
                f"unit would save money."
            ),
            components=[build.power_supply.name],
        ))
    return issues
