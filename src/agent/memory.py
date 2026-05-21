"""LangGraph SQLite checkpointer for conversation persistence."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import get_settings


def get_checkpointer() -> SqliteSaver:
    """Return a SqliteSaver bound to the configured memory DB path.

    `check_same_thread=False` is required so the saver can be used from
    Streamlit / FastAPI threads.
    """
    settings = get_settings()
    db_path: Path = settings.memory_db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)
