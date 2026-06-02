"""Run one safe paper-trading iteration using Binance Spot Testnet market data."""

from __future__ import annotations

import os
import sys
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.execution import BinanceSpotTestnetClient, JsonlExecutionLogger, PaperBroker, PaperBrokerConfig


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    symbol = os.getenv("TRADING_SYMBOL", "DOGEUSDT")
    log_path = PROJECT_ROOT / "results" / "execution_logs" / "paper_trading.jsonl"

    client = BinanceSpotTestnetClient.from_env()
    logger = JsonlExecutionLogger(log_path)
    broker = PaperBroker(
        PaperBrokerConfig(
            symbol=symbol,
            initial_quote_balance=Decimal(os.getenv("PAPER_INITIAL_QUOTE_BALANCE", "1000")),
            fee_bps=Decimal(os.getenv("PAPER_FEE_BPS", "10")),
            slippage_bps=Decimal(os.getenv("PAPER_SLIPPAGE_BPS", "5")),
        )
    )

    client.ping()
    ticker = client.ticker_price(symbol)
    price = Decimal(ticker["price"])

    # Placeholder policy: no model is called here yet. This runner verifies connectivity, logging and broker state.
    signal = "HOLD"
    result = broker.execute_signal(signal=signal, price=price, reason="Connectivity smoke test")
    logger.log_event("paper_decision", result)

    print(f"Logged {signal} decision for {symbol} at {price} into {log_path}")


if __name__ == "__main__":
    main()
