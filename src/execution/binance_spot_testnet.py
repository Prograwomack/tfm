"""Minimal Binance Spot Testnet REST client.

This module intentionally avoids placing live orders by default. The client can validate signed
orders through /api/v3/order/test and only sends testnet orders when the caller explicitly enables it.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping
from urllib.parse import urlencode

import requests


@dataclass(frozen=True)
class BinanceRequestConfig:
    """HTTP configuration for Binance Spot Testnet requests."""

    base_url: str = "https://testnet.binance.vision"
    timeout: int = 15
    recv_window: int = 5_000


class BinanceSpotTestnetClient:
    """Small REST client for Binance Spot Testnet public, account and order-validation endpoints."""

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        config: BinanceRequestConfig | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("BINANCE_TESTNET_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_TESTNET_API_SECRET")
        self.config = config or BinanceRequestConfig(
            base_url=os.getenv("BINANCE_SPOT_TESTNET_BASE_URL", "https://testnet.binance.vision")
        )
        self.session = requests.Session()
        self.time_offset_ms = 0

    @classmethod
    def from_env(cls) -> "BinanceSpotTestnetClient":
        """Create a client using environment variables loaded from .env or the shell."""

        return cls()

    @staticmethod
    def _format_decimal(value: Decimal | float | int | str) -> str:
        decimal_value = Decimal(str(value))
        return format(decimal_value.normalize(), "f")

    def _timestamp_ms(self) -> int:
        return int(time.time() * 1_000) + self.time_offset_ms

    def _sign(self, params: Mapping[str, Any]) -> str:
        if not self.api_secret:
            raise RuntimeError("BINANCE_TESTNET_API_SECRET is required for signed endpoints.")

        query_string = urlencode(params, doseq=True)
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
    ) -> Any:
        request_params = dict(params or {})
        headers = {"User-Agent": "cryptobot-tfm/0.1"}

        if signed:
            if not self.api_key:
                raise RuntimeError("BINANCE_TESTNET_API_KEY is required for signed endpoints.")
            request_params.setdefault("recvWindow", self.config.recv_window)
            request_params["timestamp"] = self._timestamp_ms()
            request_params["signature"] = self._sign(request_params)
            headers["X-MBX-APIKEY"] = self.api_key
        elif self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key

        url = f"{self.config.base_url.rstrip('/')}{path}"
        response = self.session.request(
            method=method.upper(),
            url=url,
            params=request_params,
            headers=headers,
            timeout=self.config.timeout,
        )

        if response.status_code >= 400:
            try:
                error_payload = response.json()
            except ValueError:
                error_payload = response.text
            raise RuntimeError(f"Binance API error {response.status_code}: {error_payload}")

        if not response.content:
            return {}

        try:
            return response.json()
        except ValueError:
            return response.text

    def ping(self) -> dict[str, Any]:
        """Test REST connectivity."""

        return self._request("GET", "/api/v3/ping")

    def server_time(self) -> dict[str, Any]:
        """Return Binance server time."""

        return self._request("GET", "/api/v3/time")

    def sync_time_offset(self) -> int:
        """Estimate server/client clock offset in milliseconds for signed requests."""

        server_time = int(self.server_time()["serverTime"])
        local_time = int(time.time() * 1_000)
        self.time_offset_ms = server_time - local_time
        return self.time_offset_ms

    def exchange_info(self, symbol: str | None = None) -> dict[str, Any]:
        """Return current exchange rules and optional symbol filters."""

        params = {"symbol": symbol.upper()} if symbol else None
        return self._request("GET", "/api/v3/exchangeInfo", params=params)

    def ticker_price(self, symbol: str) -> dict[str, Any]:
        """Return latest ticker price for a symbol."""

        return self._request("GET", "/api/v3/ticker/price", params={"symbol": symbol.upper()})

    def klines(self, symbol: str, interval: str = "5m", limit: int = 100) -> list[list[Any]]:
        """Return recent OHLCV candles."""

        return self._request(
            "GET",
            "/api/v3/klines",
            params={"symbol": symbol.upper(), "interval": interval, "limit": limit},
        )

    def account(self) -> dict[str, Any]:
        """Return signed account information from Spot Testnet."""

        self.sync_time_offset()
        return self._request("GET", "/api/v3/account", signed=True)

    def test_order(
        self,
        symbol: str,
        side: str,
        order_type: str = "MARKET",
        quantity: Decimal | float | int | str | None = None,
        quote_order_qty: Decimal | float | int | str | None = None,
        price: Decimal | float | int | str | None = None,
        time_in_force: str | None = None,
    ) -> dict[str, Any]:
        """Validate a signed order without sending it to the matching engine."""

        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
        }
        if quantity is not None:
            params["quantity"] = self._format_decimal(quantity)
        if quote_order_qty is not None:
            params["quoteOrderQty"] = self._format_decimal(quote_order_qty)
        if price is not None:
            params["price"] = self._format_decimal(price)
        if time_in_force is not None:
            params["timeInForce"] = time_in_force

        self.sync_time_offset()
        return self._request("POST", "/api/v3/order/test", params=params, signed=True)

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str = "MARKET",
        quantity: Decimal | float | int | str | None = None,
        quote_order_qty: Decimal | float | int | str | None = None,
        allow_live_testnet_order: bool = False,
    ) -> dict[str, Any]:
        """Send a real order to Spot Testnet only when explicitly enabled by the caller."""

        if not allow_live_testnet_order:
            raise RuntimeError("Real testnet orders are disabled. Use test_order() or set allow_live_testnet_order=True.")

        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
        }
        if quantity is not None:
            params["quantity"] = self._format_decimal(quantity)
        if quote_order_qty is not None:
            params["quoteOrderQty"] = self._format_decimal(quote_order_qty)

        self.sync_time_offset()
        return self._request("POST", "/api/v3/order", params=params, signed=True)
