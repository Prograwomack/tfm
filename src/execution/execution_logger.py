"""Structured JSONL logging utilities for simulated and testnet execution."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


class JsonlExecutionLogger:
    """Append-only JSONL logger intended for dashboard-friendly execution traces."""

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def log_event(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        event = {
            "logged_at": self._now_iso(),
            "event_type": event_type,
            **payload,
        }
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        return event


def read_jsonl_logs(log_path: str | Path) -> pd.DataFrame:
    """Read an execution JSONL log file as a pandas DataFrame."""

    path = Path(log_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)
