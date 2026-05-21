"""Catalog loader.

Responsibilities:
- Download the components CSVs from GitHub on first run (cached locally).
- Load each CSV into a pandas DataFrame.
- Normalize quirky columns:
    * `memory.speed` like "5,6000" -> ddr_gen=5, mt_s=6000
    * `memory.modules` like "2,16" -> module_count=2, module_gb=16
    * `cpu.microarchitecture` -> derived `socket` via SOCKET_MAP
    * `motherboard.socket` -> derived `ddr_gen` for memory matching
- Expose a `Catalog` singleton with one DataFrame per component category.
"""
from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

from src.config import get_settings
from src.compatibility.socket_map import ddr_gen_for_socket, infer_socket
from src.logging_setup import get_logger


log = get_logger(__name__)


GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/vinayak-ensemble/"
    "Computer_Components_Dataset/main/data/csv"
)


# category -> source filename
CATEGORY_FILES: Dict[str, str] = {
    "cpu": "cpu.csv",
    "motherboard": "motherboard.csv",
    "memory": "memory.csv",
    "video_card": "video-card.csv",
    "power_supply": "power-supply.csv",
    "case": "case.csv",
    "storage": "internal-hard-drive.csv",
    "cpu_cooler": "cpu-cooler.csv",
}


def _download_csv(filename: str, dest: Path, timeout: int = 60) -> None:
    """Download one CSV from GitHub. Retries without SSL verify on cert errors."""
    settings = get_settings()
    url = f"{GITHUB_RAW_BASE}/{filename}"
    log.info("catalog.download", url=url, dest=str(dest), ssl_verify=settings.data_ssl_verify)

    verify = settings.data_ssl_verify
    try:
        resp = requests.get(url, timeout=timeout, verify=verify)
        resp.raise_for_status()
    except requests.exceptions.SSLError as exc:
        if verify:
            log.warning(
                "catalog.download.ssl_fallback",
                message="SSL verify failed; retrying without verification. "
                        "Set DATA_SSL_VERIFY=false in .env on corporate networks.",
                error=str(exc)[:200],
            )
            resp = requests.get(url, timeout=timeout, verify=False)
            resp.raise_for_status()
        else:
            raise RuntimeError(
                f"Could not download {filename} from GitHub (SSL error). "
                f"On a corporate VPN, add DATA_SSL_VERIFY=false to your .env file, "
                f"or run: powershell -File scripts/download_data.ps1"
            ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not download {filename} from GitHub. "
            f"Check your internet connection or run: powershell -File scripts/download_data.ps1"
        ) from exc

    dest.write_bytes(resp.content)


def _ensure_csv(category: str, data_dir: Path) -> Path:
    fname = CATEGORY_FILES[category]
    path = data_dir / fname
    if not path.exists() or path.stat().st_size == 0:
        _download_csv(fname, path)
    return path


# ---------- Normalization helpers ----------

def _parse_int(val) -> Optional[int]:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return None


def _parse_memory_speed(raw) -> tuple[Optional[int], Optional[int]]:
    """`"5,6000"` -> (5, 6000). Returns (ddr_gen, mt_s)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None, None
    s = str(raw).strip().strip('"').replace(" ", "")
    parts = s.split(",")
    if len(parts) != 2:
        return None, None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None, None


def _parse_memory_modules(raw) -> tuple[Optional[int], Optional[int]]:
    """`"2,16"` -> (module_count=2, module_gb=16)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None, None
    s = str(raw).strip().strip('"').replace(" ", "")
    parts = s.split(",")
    if len(parts) != 2:
        return None, None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None, None


# ---------- Per-category normalizers ----------

def _normalize_cpu(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["microarchitecture"] = df["microarchitecture"].fillna("").astype(str)
    df["socket"] = df["microarchitecture"].apply(infer_socket)
    df["tdp"] = df["tdp"].apply(_parse_int)
    df["core_count"] = df["core_count"].apply(_parse_int)
    df["has_integrated_graphics"] = (
        df["graphics"].fillna("").astype(str).str.strip().str.len() > 0
    )
    return df


def _normalize_motherboard(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["max_memory"] = df["max_memory"].apply(_parse_int)
    df["memory_slots"] = df["memory_slots"].apply(_parse_int)
    df["ddr_gen"] = df["socket"].apply(ddr_gen_for_socket)
    return df


def _normalize_memory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["speed_raw"] = df["speed"].astype(str)
    df["modules_raw"] = df["modules"].astype(str)
    parsed_speed = df["speed"].apply(_parse_memory_speed)
    df["ddr_gen"] = parsed_speed.apply(lambda t: t[0])
    df["mt_s"] = parsed_speed.apply(lambda t: t[1])
    parsed_mods = df["modules"].apply(_parse_memory_modules)
    df["module_count"] = parsed_mods.apply(lambda t: t[0])
    df["module_gb"] = parsed_mods.apply(lambda t: t[1])
    df["total_gb"] = df["module_count"] * df["module_gb"]
    return df


# Rough TDP table for GPU power estimation, keyed by substrings of `chipset`.
# Used by PSU sizing; intentionally conservative.
_GPU_TDP_LOOKUP: list[tuple[str, int]] = [
    ("rtx 5090", 575), ("rtx 5080", 360), ("rtx 5070 ti", 300), ("rtx 5070", 250),
    ("rtx 5060 ti", 180), ("rtx 5060", 150),
    ("rtx 4090", 450), ("rtx 4080 super", 320), ("rtx 4080", 320),
    ("rtx 4070 ti super", 285), ("rtx 4070 ti", 285), ("rtx 4070 super", 220), ("rtx 4070", 200),
    ("rtx 4060 ti", 165), ("rtx 4060", 115),
    ("rtx 3090 ti", 450), ("rtx 3090", 350),
    ("rtx 3080 ti", 350), ("rtx 3080", 320),
    ("rtx 3070 ti", 290), ("rtx 3070", 220),
    ("rtx 3060 ti", 200), ("rtx 3060", 170),
    ("rtx 3050", 130),
    ("gtx 1660", 125), ("gtx 1650", 75),
    ("rx 9070 xt", 304), ("rx 9070", 220),
    ("rx 9060 xt", 180), ("rx 9060", 150),
    ("rx 7900 xtx", 355), ("rx 7900 xt", 315), ("rx 7900", 260),
    ("rx 7800 xt", 263), ("rx 7700 xt", 245),
    ("rx 7600 xt", 190), ("rx 7600", 165),
    ("rx 6900 xt", 300), ("rx 6800 xt", 300), ("rx 6800", 250),
    ("rx 6700 xt", 230), ("rx 6700", 175),
    ("rx 6600 xt", 160), ("rx 6600", 132),
    ("rx 6500 xt", 107), ("rx 6400", 53),
    ("arc a770", 225), ("arc a750", 225), ("arc a580", 175), ("arc a380", 75),
]


def estimate_gpu_tdp(chipset: str) -> int:
    """Best-effort TDP estimate for a GPU chipset string. Defaults to 200W."""
    if not chipset:
        return 0
    s = chipset.lower()
    for key, tdp in _GPU_TDP_LOOKUP:
        if key in s:
            return tdp
    return 200  # conservative default for unknown discrete GPUs


def _normalize_video_card(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["memory"] = df["memory"].apply(_parse_int)
    df["length"] = df["length"].apply(_parse_int)
    df["chipset"] = df["chipset"].fillna("").astype(str)
    df["estimated_tdp"] = df["chipset"].apply(estimate_gpu_tdp)
    return df


def _normalize_power_supply(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["wattage"] = df["wattage"].apply(_parse_int)
    return df


def _normalize_case(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["type"] = df["type"].fillna("").astype(str)
    df["external_volume"] = pd.to_numeric(df["external_volume"], errors="coerce")
    df["internal_35_bays"] = df["internal_35_bays"].apply(_parse_int)
    return df


def _normalize_storage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["capacity"] = df["capacity"].apply(_parse_int)
    df["type"] = df["type"].fillna("").astype(str)
    df["interface"] = df["interface"].fillna("").astype(str)
    df["form_factor"] = df["form_factor"].fillna("").astype(str)
    return df


def _normalize_cpu_cooler(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df["is_aio"] = df["size"].notna()
    return df


NORMALIZERS = {
    "cpu": _normalize_cpu,
    "motherboard": _normalize_motherboard,
    "memory": _normalize_memory,
    "video_card": _normalize_video_card,
    "power_supply": _normalize_power_supply,
    "case": _normalize_case,
    "storage": _normalize_storage,
    "cpu_cooler": _normalize_cpu_cooler,
}


class Catalog:
    """In-memory component catalog. Singleton accessed via `get_catalog()`."""

    def __init__(self, tables: Dict[str, pd.DataFrame]):
        self._tables = tables

    def __getitem__(self, category: str) -> pd.DataFrame:
        if category not in self._tables:
            raise KeyError(
                f"Unknown category '{category}'. Available: "
                f"{sorted(self._tables.keys())}"
            )
        return self._tables[category]

    @property
    def categories(self) -> list[str]:
        return sorted(self._tables.keys())

    def stats(self) -> Dict[str, int]:
        return {cat: len(df) for cat, df in self._tables.items()}


_catalog: Optional[Catalog] = None


def get_catalog() -> Catalog:
    """Return the singleton catalog, loading it on first call."""
    global _catalog
    if _catalog is None:
        _catalog = _load_catalog()
    return _catalog


def _load_catalog() -> Catalog:
    settings = get_settings()
    data_dir = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    tables: Dict[str, pd.DataFrame] = {}
    t0 = time.time()
    for category in CATEGORY_FILES:
        path = _ensure_csv(category, data_dir)
        df = pd.read_csv(path)
        df = NORMALIZERS[category](df)
        tables[category] = df
        log.info(
            "catalog.loaded",
            category=category,
            rows=len(df),
            columns=list(df.columns),
        )
    log.info(
        "catalog.ready",
        total_rows=sum(len(df) for df in tables.values()),
        elapsed_s=round(time.time() - t0, 2),
    )
    return Catalog(tables)
