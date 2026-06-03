"""Create a tiny sample execution log for dashboard testing when no notebook logs exist.

Run from the project root:

    python scripts/create_sample_dashboard_logs.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "results" / "execution_logs" / "paper_trading.jsonl"
SUMMARY_PATH = PROJECT_ROOT / "results" / "11_testnet_paper_trading_summary.csv"
RUN_ID = f"sample_dashboard_run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"


def write_event(event_type: str, event_sequence: int, payload: dict) -> None:
    event = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "run_id": RUN_ID,
        "event_id": f"{RUN_ID}_{event_sequence:04d}",
        "event_sequence": event_sequence,
        "notebook": "sample_dashboard_logs",
        **payload,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


write_event("connectivity_check", 1, {"symbol": "DOGEUSDT", "ticker_price": 0.1})
write_event("signal_generated", 2, {"timestamp": datetime.now(timezone.utc).isoformat(), "symbol": "DOGEUSDT", "close": 0.1, "ema_10": 0.101, "ema_20": 0.099, "signal": "BUY", "reason": "Sample signal"})
write_event("forced_paper_buy", 3, {"timestamp": datetime.now(timezone.utc).isoformat(), "symbol": "DOGEUSDT", "side": "BUY", "mark_price": 0.1, "execution_price": 0.10005, "quantity_base": 99.85, "gross_quote": 10.0, "fee_quote": 0.01, "cash_quote_after": 990.0, "position_base_after": 99.85, "equity_quote_after": 999.985, "reason": "Sample BUY", "mode": "paper"})
write_event("forced_paper_sell", 4, {"timestamp": datetime.now(timezone.utc).isoformat(), "symbol": "DOGEUSDT", "side": "SELL", "mark_price": 0.1, "execution_price": 0.09995, "quantity_base": 99.85, "gross_quote": 9.98, "fee_quote": 0.01, "cash_quote_after": 999.97, "position_base_after": 0.0, "equity_quote_after": 999.97, "reason": "Sample SELL", "mode": "paper"})

SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH.write_text(
    "run_id,notebook,strategy,symbol,initial_balance,final_equity,return_pct,number_of_trades,fees_bps,slippage_bps,limitations\n"
    f"{RUN_ID},sample_dashboard_logs,Sample dashboard smoke test,DOGEUSDT,1000,999.97,-0.003,2,10,5,Sample data only\n",
    encoding="utf-8",
)

print(f"Sample run_id: {RUN_ID}")
print(f"Log written to: {LOG_PATH}")
print(f"Summary written to: {SUMMARY_PATH}")
