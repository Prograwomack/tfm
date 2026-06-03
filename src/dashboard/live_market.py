"""Live Binance market data helpers for the Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests


DEFAULT_SPOT_BASE_URL = "https://api.binance.com"
DEFAULT_TESTNET_BASE_URL = "https://testnet.binance.vision"


@dataclass(frozen=True)
class MarketDataConfig:
    """Configuration for public Binance market data requests."""

    base_url: str = DEFAULT_SPOT_BASE_URL
    timeout_seconds: int = 10


class BinanceMarketDataClient:
    """Small public REST client used by the dashboard for live candles and symbol metadata."""

    def __init__(self, config: MarketDataConfig | None = None) -> None:
        self.config = config or MarketDataConfig()
        self.base_url = self.config.base_url.rstrip("/")

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        response = requests.get(
            f"{self.base_url}{path}",
            params=params,
            timeout=self.config.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Binance market data error {response.status_code}: {response.text}")
        return response.json()

    def exchange_info(self, symbol: str | None = None) -> dict[str, Any]:
        params = {"symbol": symbol.upper()} if symbol else None
        return self._get("/api/v3/exchangeInfo", params=params)

    def ticker_price(self, symbol: str) -> dict[str, Any]:
        return self._get("/api/v3/ticker/price", params={"symbol": symbol.upper()})

    def klines(self, symbol: str, interval: str = "5m", limit: int = 200) -> list[list[Any]]:
        return self._get(
            "/api/v3/klines",
            params={"symbol": symbol.upper(), "interval": interval, "limit": int(limit)},
        )


def klines_to_dataframe(raw_klines: list[list[Any]]) -> pd.DataFrame:
    """Convert Binance kline payload into a typed dataframe."""

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    candles_df = pd.DataFrame(raw_klines, columns=columns)
    if candles_df.empty:
        return candles_df

    candles_df["open_time"] = pd.to_datetime(candles_df["open_time"], unit="ms", utc=True)
    candles_df["close_time"] = pd.to_datetime(candles_df["close_time"], unit="ms", utc=True)

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    for column in numeric_columns:
        candles_df[column] = pd.to_numeric(candles_df[column], errors="coerce")

    return candles_df.drop(columns=["ignore"]).sort_values("open_time").reset_index(drop=True)


def get_symbol_status(client: BinanceMarketDataClient, symbol: str) -> dict[str, Any]:
    """Return compact metadata for a single spot symbol."""

    info = client.exchange_info(symbol=symbol)
    symbols = info.get("symbols", [])
    if not symbols:
        return {"symbol": symbol.upper(), "exists": False, "status": "NOT_FOUND"}

    symbol_info = symbols[0]
    return {
        "symbol": symbol_info.get("symbol"),
        "exists": True,
        "status": symbol_info.get("status"),
        "base_asset": symbol_info.get("baseAsset"),
        "quote_asset": symbol_info.get("quoteAsset"),
        "order_types": ", ".join(symbol_info.get("orderTypes", [])),
        "is_spot_trading_allowed": symbol_info.get("isSpotTradingAllowed"),
    }
