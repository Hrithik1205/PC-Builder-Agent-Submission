"""Map a CPU microarchitecture to its motherboard socket.

The dataset's `cpu.csv` does NOT contain a `socket` column, so we infer it
from `microarchitecture`. Keys are normalized to lower-case. Unknown
microarchitectures are mapped to `None` and those CPUs are excluded from
selection rather than risking a wrong socket assignment.
"""
from __future__ import annotations

from typing import Optional


SOCKET_MAP: dict[str, str] = {
    # ---- AMD ----
    "zen 5": "AM5",
    "zen 4": "AM5",
    "zen 3": "AM4",
    "zen 2": "AM4",
    "zen+": "AM4",
    "zen": "AM4",
    # Threadripper (out of scope for typical builds but mapped for safety)
    "zen 4 (threadripper)": "sTR5",
    "zen 3 (threadripper)": "sWRX8",
    # ---- Intel ----
    "arrow lake": "LGA1851",
    "raptor lake refresh": "LGA1700",
    "raptor lake": "LGA1700",
    "alder lake": "LGA1700",
    "rocket lake": "LGA1200",
    "comet lake": "LGA1200",
    "coffee lake refresh": "LGA1151",
    "coffee lake": "LGA1151",
}


def infer_socket(microarchitecture: Optional[str]) -> Optional[str]:
    """Return the socket string for a microarchitecture, or None if unknown."""
    if not microarchitecture:
        return None
    return SOCKET_MAP.get(microarchitecture.strip().lower())


def ddr_gen_for_socket(socket: Optional[str]) -> Optional[int]:
    """Return DDR generation expected by a given socket.

    LGA1700 actually supports both DDR4 and DDR5 depending on the board, but
    for catalog-level recommendations we default to DDR5 since that is the
    current mainstream pairing.
    """
    if not socket:
        return None
    socket = socket.upper()
    if socket in {"AM5", "LGA1851", "LGA1700"}:
        return 5
    if socket in {"AM4", "LGA1200", "LGA1151"}:
        return 4
    return None
