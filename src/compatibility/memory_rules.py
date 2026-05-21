"""Memory <-> motherboard compatibility rules."""
from __future__ import annotations

from typing import List

from src.data.schemas import Build, Issue


def check_memory_ddr(build: Build) -> List[Issue]:
    issues: List[Issue] = []
    if build.memory is None or build.motherboard is None:
        return issues
    mobo_ddr = build.motherboard.ddr_gen
    mem_ddr = build.memory.ddr_gen
    if mobo_ddr and mem_ddr and mobo_ddr != mem_ddr:
        issues.append(Issue(
            severity="error",
            rule="memory_ddr_mismatch",
            message=(
                f"Memory '{build.memory.name}' is DDR{mem_ddr} but motherboard "
                f"'{build.motherboard.name}' expects DDR{mobo_ddr}."
            ),
            components=[build.memory.name, build.motherboard.name],
        ))
    return issues


def check_memory_capacity(build: Build) -> List[Issue]:
    issues: List[Issue] = []
    if build.memory is None or build.motherboard is None:
        return issues
    total = build.memory.total_gb or 0
    max_mem = build.motherboard.max_memory or 0
    if total and max_mem and total > max_mem:
        issues.append(Issue(
            severity="error",
            rule="memory_capacity_exceeded",
            message=(
                f"Memory total {total} GB exceeds motherboard max of {max_mem} GB "
                f"({build.motherboard.name})."
            ),
            components=[build.memory.name, build.motherboard.name],
        ))
    return issues


def check_memory_slots(build: Build) -> List[Issue]:
    issues: List[Issue] = []
    if build.memory is None or build.motherboard is None:
        return issues
    mods = build.memory.module_count or 0
    slots = build.motherboard.memory_slots or 0
    if mods and slots and mods > slots:
        issues.append(Issue(
            severity="error",
            rule="memory_slot_count_exceeded",
            message=(
                f"Memory kit has {mods} modules but motherboard "
                f"'{build.motherboard.name}' only has {slots} slots."
            ),
            components=[build.memory.name, build.motherboard.name],
        ))
    return issues
