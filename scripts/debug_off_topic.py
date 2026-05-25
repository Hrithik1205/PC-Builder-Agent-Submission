"""Reproduce the off-topic misclassification for 'Personal PC with 512 GB storage'."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.nodes import _looks_off_topic, _ON_TOPIC_HINTS, _OFF_TOPIC_HINTS

TESTS = [
    "Personal PC with 512 GB storage",
    "I want a PC for personal use and my budget is $500 maximum",
    "Personal",
    "personal use",
    "Tell me a joke",
    "what's the weather",
]

for t in TESTS:
    on = bool(_ON_TOPIC_HINTS.search(t))
    off = bool(_OFF_TOPIC_HINTS.search(t))
    flagged = _looks_off_topic(t)
    print(f"  flagged={flagged}  on={on}  off={off}  -- {t!r}")
