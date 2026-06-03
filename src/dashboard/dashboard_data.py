"""Data-loading and metric helpers for the Streamlit execution dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


TRADE_SIDES = {"BUY", "SELL"}
EXECUTION_EVENT_TYPES = {
    "paper_execution",
    "forced_paper_buy",
    "forced_paper_sell",
}
SIGNAL_EVENT_TYPE = "signal_generated"
RUN_ID_FALLBACK = "NO_RUN_ID"


def read_jsonl_logs(log_path: str | Path) -> pd.DataFrame:
    """Read an append-only JSONL execution log as a dataframe."""

    path = Path(log_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append(
                    {
                        "logged_at": None,
                        "event_type": "malformed_jsonl_line",
                        "line_number": line_number,
                        "raw_line": line,
                    }
                )

    logs_df = pd.DataFrame(records)
    return normalize_logs(logs_df)


def read_summary(summary_path: str | Path) -> pd.DataFrame:
    """Read the paper trading summary CSV if it exists."""

    path = Path(summary_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize_logs(logs_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize datatypes and helper columns used by the dashboard."""

    if logs_df.empty:
        return logs_df

    normalized_df = logs_df.copy()

    if "run_id" not in normalized_df.columns:
        normalized_df["run_id"] = RUN_ID_FALLBACK
    normalized_df["run_id"] = normalized_df["run_id"].fillna(RUN_ID_FALLBACK).astype(str)

    for column in ["logged_at", "timestamp"]:
        if column in normalized_df.columns:
            normalized_df[column] = pd.to_datetime(normalized_df[column], errors="coerce", utc=True)

    if "logged_at" in normalized_df.columns and "timestamp" in normalized_df.columns:
        normalized_df["event_time"] = normalized_df["timestamp"].fillna(normalized_df["logged_at"])
    elif "timestamp" in normalized_df.columns:
        normalized_df["event_time"] = normalized_df["timestamp"]
    elif "logged_at" in normalized_df.columns:
        normalized_df["event_time"] = normalized_df["logged_at"]
    else:
        normalized_df["event_time"] = pd.NaT

    numeric_columns = [
        "close",
        "ema_10",
        "ema_20",
        "mark_price",
        "execution_price",
        "quantity_base",
        "gross_quote",
        "fee_quote",
        "cash_quote_after",
        "position_base_after",
        "equity_quote_after",
        "event_sequence",
    ]
    for column in numeric_columns:
        if column in normalized_df.columns:
            normalized_df[column] = pd.to_numeric(normalized_df[column], errors="coerce")

    sort_columns = [column for column in ["event_time", "logged_at", "event_sequence"] if column in normalized_df.columns]
    if sort_columns:
        normalized_df = normalized_df.sort_values(sort_columns, kind="stable")

    return normalized_df.reset_index(drop=True)


def get_available_run_ids(logs_df: pd.DataFrame) -> list[str]:
    """Return run ids ordered by most recent event first."""

    if logs_df.empty or "run_id" not in logs_df.columns:
        return []

    ordered_runs_df = (
        logs_df.groupby("run_id", dropna=False)["event_time"]
        .max()
        .reset_index()
        .sort_values("event_time", ascending=False, na_position="last")
    )
    return ordered_runs_df["run_id"].astype(str).tolist()


def filter_by_run(logs_df: pd.DataFrame, selected_run_id: str | None) -> pd.DataFrame:
    """Filter logs by run id, leaving all rows when selected_run_id is None or ALL_RUNS."""

    if logs_df.empty or selected_run_id in {None, "ALL_RUNS"}:
        return logs_df.copy()
    return logs_df[logs_df["run_id"].astype(str) == str(selected_run_id)].copy()


def build_equity_curve(logs_df: pd.DataFrame) -> pd.DataFrame:
    """Build an equity curve from paper execution events."""

    if logs_df.empty or "equity_quote_after" not in logs_df.columns:
        return pd.DataFrame()

    equity_df = logs_df[logs_df["equity_quote_after"].notna()].copy()
    if equity_df.empty:
        return pd.DataFrame()

    keep_columns = [
        "event_time",
        "logged_at",
        "run_id",
        "event_id",
        "event_type",
        "side",
        "symbol",
        "mark_price",
        "cash_quote_after",
        "position_base_after",
        "equity_quote_after",
    ]
    keep_columns = [column for column in keep_columns if column in equity_df.columns]
    equity_df = equity_df[keep_columns].copy()
    equity_df = equity_df.rename(columns={"event_time": "time"})
    return equity_df.sort_values("time", na_position="last").reset_index(drop=True)


def build_operations(logs_df: pd.DataFrame, include_holds: bool = True) -> pd.DataFrame:
    """Extract execution-like events for the operations table."""

    if logs_df.empty:
        return pd.DataFrame()

    operation_mask = pd.Series(False, index=logs_df.index)
    if "event_type" in logs_df.columns:
        operation_mask = operation_mask | logs_df["event_type"].isin(EXECUTION_EVENT_TYPES)
    if "side" in logs_df.columns:
        if include_holds:
            operation_mask = operation_mask | logs_df["side"].isin(["BUY", "SELL", "HOLD"])
        else:
            operation_mask = operation_mask | logs_df["side"].isin(list(TRADE_SIDES))

    operations_df = logs_df[operation_mask].copy()
    if not include_holds and "side" in operations_df.columns:
        operations_df = operations_df[operations_df["side"].isin(list(TRADE_SIDES))].copy()

    display_columns = [
        "event_time",
        "event_type",
        "side",
        "symbol",
        "mark_price",
        "execution_price",
        "quantity_base",
        "gross_quote",
        "fee_quote",
        "cash_quote_after",
        "position_base_after",
        "equity_quote_after",
        "reason",
        "run_id",
        "event_id",
    ]
    display_columns = [column for column in display_columns if column in operations_df.columns]
    return operations_df[display_columns].sort_values("event_time", na_position="last").reset_index(drop=True)


def build_signals(logs_df: pd.DataFrame) -> pd.DataFrame:
    """Extract generated trading signals."""

    if logs_df.empty or "event_type" not in logs_df.columns:
        return pd.DataFrame()

    signals_df = logs_df[logs_df["event_type"] == SIGNAL_EVENT_TYPE].copy()
    display_columns = [
        "event_time",
        "symbol",
        "signal",
        "close",
        "ema_10",
        "ema_20",
        "reason",
        "run_id",
        "event_id",
    ]
    display_columns = [column for column in display_columns if column in signals_df.columns]
    return signals_df[display_columns].sort_values("event_time", na_position="last").reset_index(drop=True)


def build_event_counts(logs_df: pd.DataFrame) -> pd.DataFrame:
    """Count events by type for a compact operational summary."""

    if logs_df.empty or "event_type" not in logs_df.columns:
        return pd.DataFrame(columns=["event_type", "count"])

    return (
        logs_df.groupby("event_type", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def compute_dashboard_metrics(
    logs_df: pd.DataFrame,
    summary_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute headline metrics displayed at the top of the dashboard."""

    summary_df = summary_df if summary_df is not None else pd.DataFrame()
    equity_df = build_equity_curve(logs_df)
    trades_df = build_operations(logs_df, include_holds=False)
    signals_df = build_signals(logs_df)

    initial_balance = None
    final_equity = None

    if not summary_df.empty:
        if "initial_balance" in summary_df.columns:
            initial_balance = pd.to_numeric(summary_df["initial_balance"], errors="coerce").dropna().iloc[-1]
        if "final_equity" in summary_df.columns:
            final_equity = pd.to_numeric(summary_df["final_equity"], errors="coerce").dropna().iloc[-1]

    if final_equity is None and not equity_df.empty:
        final_equity = equity_df["equity_quote_after"].dropna().iloc[-1]

    if initial_balance is None:
        if not summary_df.empty and "initial_balance" in summary_df.columns:
            values = pd.to_numeric(summary_df["initial_balance"], errors="coerce").dropna()
            initial_balance = values.iloc[-1] if not values.empty else None
        elif not equity_df.empty:
            initial_balance = equity_df["equity_quote_after"].dropna().iloc[0]

    return_pct = None
    if initial_balance not in [None, 0] and final_equity is not None:
        return_pct = (float(final_equity) / float(initial_balance) - 1) * 100

    latest_equity_row = equity_df.dropna(subset=["equity_quote_after"]).tail(1)
    latest_cash = latest_position = latest_price = None
    if not latest_equity_row.empty:
        row = latest_equity_row.iloc[0]
        latest_cash = row.get("cash_quote_after")
        latest_position = row.get("position_base_after")
        latest_price = row.get("mark_price")

    last_signal = None
    if not signals_df.empty and "signal" in signals_df.columns:
        last_signal = signals_df["signal"].dropna().iloc[-1]

    total_fees = 0.0
    if not trades_df.empty and "fee_quote" in trades_df.columns:
        total_fees = float(pd.to_numeric(trades_df["fee_quote"], errors="coerce").fillna(0).sum())

    buys = sells = holds = 0
    if "side" in logs_df.columns:
        side_counts = logs_df["side"].value_counts(dropna=True).to_dict()
        buys = int(side_counts.get("BUY", 0))
        sells = int(side_counts.get("SELL", 0))
        holds = int(side_counts.get("HOLD", 0))

    last_event_time = None
    if not logs_df.empty and "event_time" in logs_df.columns:
        non_null_times = logs_df["event_time"].dropna()
        last_event_time = non_null_times.iloc[-1] if not non_null_times.empty else None

    return {
        "initial_balance": initial_balance,
        "final_equity": final_equity,
        "return_pct": return_pct,
        "latest_cash": latest_cash,
        "latest_position": latest_position,
        "latest_price": latest_price,
        "trade_count": int(len(trades_df)),
        "buy_count": buys,
        "sell_count": sells,
        "hold_count": holds,
        "signal_count": int(len(signals_df)),
        "last_signal": last_signal,
        "total_fees": total_fees,
        "event_count": int(len(logs_df)),
        "last_event_time": last_event_time,
    }


def format_number(value: Any, decimals: int = 2, suffix: str = "") -> str:
    """Format nullable numeric values for Streamlit metrics."""

    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{decimals}f}{suffix}"
