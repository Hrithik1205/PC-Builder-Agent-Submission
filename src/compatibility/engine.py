"""Aggregate compatibility engine.

Composes individual rules into a single `check_build` entry point that
returns a list of `Issue`s. Pure Python, no LLM, deterministic.
"""
from __future__ import annotations

from typing import List

from src.data.schemas import Build, Issue
from src.compatibility.socket_map import infer_socket
from src.compatibility.memory_rules import (
    check_memory_capacity,
    check_memory_ddr,
    check_memory_slots,
)
from src.compatibility.case_rules import check_case_gpu_length, check_case_motherboard
from src.compatibility.power_rules import check_psu_wattage


def check_cpu_socket(build: Build) -> List[Issue]:
    issues: List[Issue] = []
    if build.cpu is None or build.motherboard is None:
        return issues
    cpu_socket = build.cpu.socket or infer_socket(build.cpu.microarchitecture)
    mobo_socket = build.motherboard.socket
    if cpu_socket and mobo_socket and cpu_socket != mobo_socket:
        issues.append(Issue(
            severity="error",
            rule="cpu_socket_mismatch",
            message=(
                f"CPU '{build.cpu.name}' uses socket {cpu_socket} but motherboard "
                f"'{build.motherboard.name}' is socket {mobo_socket}."
            ),
            components=[build.cpu.name, build.motherboard.name],
        ))
    elif build.cpu is not None and cpu_socket is None:
        issues.append(Issue(
            severity="warn",
            rule="cpu_socket_unknown",
            message=(
                f"Could not infer socket for CPU '{build.cpu.name}' "
                f"(microarchitecture='{build.cpu.microarchitecture}')."
            ),
            components=[build.cpu.name],
        ))
    return issues


def check_storage_present(build: Build) -> List[Issue]:
    """Warn if no SSD is selected (still functional, just slow)."""
    issues: List[Issue] = []
    if build.storage is None:
        return issues
    stype = (build.storage.type or "").upper()
    if "SSD" not in stype:
        issues.append(Issue(
            severity="warn",
            rule="no_ssd_selected",
            message=(
                f"Selected storage '{build.storage.name}' is {stype or 'unknown'}. "
                f"Booting from an HDD is significantly slower; an SSD is recommended."
            ),
            components=[build.storage.name],
        ))
    return issues


def check_build(build: Build) -> List[Issue]:
    """Run every rule and return the consolidated issue list."""
    issues: List[Issue] = []
    issues.extend(check_cpu_socket(build))
    issues.extend(check_memory_ddr(build))
    issues.extend(check_memory_capacity(build))
    issues.extend(check_memory_slots(build))
    issues.extend(check_case_motherboard(build))
    issues.extend(check_case_gpu_length(build))
    issues.extend(check_psu_wattage(build))
    issues.extend(check_storage_present(build))
    return issues


def has_errors(issues: List[Issue]) -> bool:
    return any(i.severity == "error" for i in issues)


def summarize_issues(issues: List[Issue]) -> str:
    if not issues:
        return "No compatibility issues detected."
    lines = []
    for i in issues:
        lines.append(f"- [{i.severity.upper()}] {i.rule}: {i.message}")
    return "\n".join(lines)
