"""Pydantic component schemas.

These mirror the columns we actually use from the CSVs. Fields not in the
upstream CSVs (`socket` on CPU, `ddr_gen` on memory) are derived during
loading and persisted onto the row dicts so the schemas can validate them
uniformly.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class CPU(BaseModel):
    name: str
    price: float
    core_count: int
    core_clock: Optional[float] = None
    boost_clock: Optional[float] = None
    microarchitecture: Optional[str] = None
    tdp: Optional[int] = None
    graphics: Optional[str] = None  # integrated GPU description or empty
    # Derived
    socket: Optional[str] = None  # filled by loader from SOCKET_MAP


class Motherboard(BaseModel):
    name: str
    price: float
    socket: str
    form_factor: str
    max_memory: Optional[int] = None
    memory_slots: Optional[int] = None
    color: Optional[str] = None
    # Derived
    ddr_gen: Optional[int] = None  # 4 or 5, inferred from socket era


class Memory(BaseModel):
    name: str
    price: float
    speed_raw: Optional[str] = None
    modules_raw: Optional[str] = None
    color: Optional[str] = None
    cas_latency: Optional[int] = None
    # Derived
    ddr_gen: Optional[int] = None
    mt_s: Optional[int] = None  # transfer speed in MT/s
    module_count: Optional[int] = None
    module_gb: Optional[int] = None
    total_gb: Optional[int] = None


class VideoCard(BaseModel):
    name: str
    price: float
    chipset: str
    memory: Optional[int] = None  # GB
    core_clock: Optional[int] = None
    boost_clock: Optional[int] = None
    color: Optional[str] = None
    length: Optional[int] = None  # mm
    # Derived
    estimated_tdp: Optional[int] = None


class PowerSupply(BaseModel):
    name: str
    price: float
    type: Optional[str] = None
    efficiency: Optional[str] = None
    wattage: int
    modular: Optional[str] = None
    color: Optional[str] = None


class Case(BaseModel):
    name: str
    price: float
    type: str
    color: Optional[str] = None
    psu: Optional[str] = None
    side_panel: Optional[str] = None
    external_volume: Optional[float] = None
    internal_35_bays: Optional[int] = None


class Storage(BaseModel):
    name: str
    price: float
    capacity: int  # GB
    price_per_gb: Optional[float] = None
    type: Optional[str] = None  # SSD / HDD / Hybrid
    cache: Optional[int] = None
    form_factor: Optional[str] = None
    interface: Optional[str] = None


class CPUCooler(BaseModel):
    name: str
    price: float
    rpm: Optional[str] = None
    noise_level: Optional[str] = None
    color: Optional[str] = None
    size: Optional[float] = None  # mm radiator size for AIO; null for air


class Build(BaseModel):
    """A complete (or partial) PC configuration."""
    cpu: Optional[CPU] = None
    motherboard: Optional[Motherboard] = None
    memory: Optional[Memory] = None
    video_card: Optional[VideoCard] = None
    storage: Optional[Storage] = None
    power_supply: Optional[PowerSupply] = None
    case: Optional[Case] = None
    cpu_cooler: Optional[CPUCooler] = None

    def total_price(self) -> float:
        parts = [
            self.cpu, self.motherboard, self.memory, self.video_card,
            self.storage, self.power_supply, self.case, self.cpu_cooler,
        ]
        return round(sum(p.price for p in parts if p is not None), 2)

    def selected_categories(self) -> List[str]:
        result = []
        for cat, val in [
            ("cpu", self.cpu),
            ("motherboard", self.motherboard),
            ("memory", self.memory),
            ("video_card", self.video_card),
            ("storage", self.storage),
            ("power_supply", self.power_supply),
            ("case", self.case),
            ("cpu_cooler", self.cpu_cooler),
        ]:
            if val is not None:
                result.append(cat)
        return result


class Requirements(BaseModel):
    """Structured user requirements extracted by the gatherer node."""
    use_case: str = Field(description="Primary use case e.g. gaming, office, content_creation")
    budget_usd: Optional[float] = Field(default=None, description="Total budget in USD (upper bound if a range was given)")
    budget_min_usd: Optional[float] = Field(default=None, description="Lower bound of a budget range, if user gave one")
    budget_flexible: bool = Field(default=False, description="Whether user is open to slight overspend")
    noise_preference: Optional[Literal["quiet", "balanced", "performance"]] = None
    form_factor_preference: Optional[Literal["mini_itx", "micro_atx", "atx", "any"]] = "any"
    cpu_brand_preference: Optional[Literal["amd", "intel"]] = Field(
        default=None,
        description="User-preferred CPU brand: 'amd', 'intel', or None for either"
    )
    gpu_brand_preference: Optional[Literal["nvidia", "amd"]] = Field(
        default=None,
        description="User-preferred GPU brand: 'nvidia', 'amd' (Radeon), or None"
    )
    os_needed: bool = False
    peripherals_needed: List[str] = Field(default_factory=list)
    must_have: List[str] = Field(default_factory=list)
    nice_to_have: List[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"
    clarifying_questions: List[str] = Field(default_factory=list)
    is_on_topic: bool = Field(default=True, description="False when user's message is not about PCs")


class Issue(BaseModel):
    """A compatibility check finding."""
    severity: Literal["error", "warn", "info"]
    rule: str
    message: str
    components: List[str] = Field(default_factory=list)
