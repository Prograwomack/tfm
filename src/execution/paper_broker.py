"""Local paper broker used to simulate spot execution without sending orders to Binance."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class PaperBrokerConfig:
    """Configuration for deterministic paper execution."""

    symbol: str = "DOGEUSDT"
    base_asset: str = "DOGE"
    quote_asset: str = "USDT"
    initial_quote_balance: Decimal = Decimal("1000")
    initial_base_balance: Decimal = Decimal("0")
    fee_bps: Decimal = Decimal("10")
    slippage_bps: Decimal = Decimal("5")
    max_position_quote_pct: Decimal = Decimal("0.25")
    min_quote_order_qty: Decimal = Decimal("5")


@dataclass
class PaperBroker:
    """Simple long-only spot broker with quote cash, base inventory, fees and slippage."""

    config: PaperBrokerConfig = field(default_factory=PaperBrokerConfig)
    cash_quote: Decimal = field(init=False)
    position_base: Decimal = field(init=False)
    trades: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash_quote = Decimal(self.config.initial_quote_balance)
        self.position_base = Decimal(self.config.initial_base_balance)

    @staticmethod
    def _to_decimal(value: Decimal | float | int | str) -> Decimal:
        return Decimal(str(value))

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @property
    def fee_rate(self) -> Decimal:
        return self.config.fee_bps / Decimal("10000")

    @property
    def slippage_rate(self) -> Decimal:
        return self.config.slippage_bps / Decimal("10000")

    def mark_to_market(self, price: Decimal | float | int | str, timestamp: str | None = None) -> dict[str, Any]:
        mark_price = self._to_decimal(price)
        equity = self.cash_quote + self.position_base * mark_price
        snapshot = {
            "timestamp": timestamp or self._now_iso(),
            "symbol": self.config.symbol,
            "cash_quote": float(self.cash_quote),
            "position_base": float(self.position_base),
            "mark_price": float(mark_price),
            "equity_quote": float(equity),
        }
        self.equity_curve.append(snapshot)
        return snapshot

    def market_buy(
        self,
        price: Decimal | float | int | str,
        quote_amount: Decimal | float | int | str | None = None,
        timestamp: str | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        mark_price = self._to_decimal(price)
        equity = self.cash_quote + self.position_base * mark_price
        default_quote_amount = equity * self.config.max_position_quote_pct
        requested_quote_amount = self._to_decimal(quote_amount) if quote_amount is not None else default_quote_amount
        gross_quote = min(requested_quote_amount, self.cash_quote)

        if gross_quote < self.config.min_quote_order_qty:
            return self.hold(mark_price, timestamp=timestamp, reason="Insufficient quote balance for minimum order size")

        execution_price = mark_price * (Decimal("1") + self.slippage_rate)
        fee_quote = gross_quote * self.fee_rate
        net_quote = gross_quote - fee_quote
        quantity_base = net_quote / execution_price

        self.cash_quote -= gross_quote
        self.position_base += quantity_base
        trade = self._trade_payload(
            side="BUY",
            mark_price=mark_price,
            execution_price=execution_price,
            quantity_base=quantity_base,
            gross_quote=gross_quote,
            fee_quote=fee_quote,
            timestamp=timestamp,
            reason=reason,
        )
        self.trades.append(trade)
        self.mark_to_market(mark_price, timestamp=trade["timestamp"])
        return trade

    def market_sell(
        self,
        price: Decimal | float | int | str,
        quantity_base: Decimal | float | int | str | None = None,
        timestamp: str | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        mark_price = self._to_decimal(price)
        requested_quantity = self._to_decimal(quantity_base) if quantity_base is not None else self.position_base
        executed_quantity = min(requested_quantity, self.position_base)

        if executed_quantity <= Decimal("0"):
            return self.hold(mark_price, timestamp=timestamp, reason="No base position available to sell")

        execution_price = mark_price * (Decimal("1") - self.slippage_rate)
        gross_quote = executed_quantity * execution_price
        fee_quote = gross_quote * self.fee_rate
        net_quote = gross_quote - fee_quote

        self.position_base -= executed_quantity
        self.cash_quote += net_quote
        trade = self._trade_payload(
            side="SELL",
            mark_price=mark_price,
            execution_price=execution_price,
            quantity_base=executed_quantity,
            gross_quote=gross_quote,
            fee_quote=fee_quote,
            timestamp=timestamp,
            reason=reason,
        )
        self.trades.append(trade)
        self.mark_to_market(mark_price, timestamp=trade["timestamp"])
        return trade

    def hold(
        self,
        price: Decimal | float | int | str,
        timestamp: str | None = None,
        reason: str = "") -> dict[str, Any]:
        mark_price = self._to_decimal(price)
        snapshot = self.mark_to_market(mark_price, timestamp=timestamp)
        return {
            "timestamp": snapshot["timestamp"],
            "symbol": self.config.symbol,
            "side": "HOLD",
            "mark_price": float(mark_price),
            "execution_price": None,
            "quantity_base": 0.0,
            "gross_quote": 0.0,
            "fee_quote": 0.0,
            "cash_quote_after": snapshot["cash_quote"],
            "position_base_after": snapshot["position_base"],
            "equity_quote_after": snapshot["equity_quote"],
            "reason": reason,
            "mode": "paper",
        }

    def execute_signal(
        self,
        signal: str,
        price: Decimal | float | int | str,
        quote_amount: Decimal | float | int | str | None = None,
        quantity_base: Decimal | float | int | str | None = None,
        timestamp: str | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        action = signal.upper().strip()
        if action == "BUY":
            return self.market_buy(price, quote_amount=quote_amount, timestamp=timestamp, reason=reason)
        if action == "SELL":
            return self.market_sell(price, quantity_base=quantity_base, timestamp=timestamp, reason=reason)
        return self.hold(price, timestamp=timestamp, reason=reason or "No executable signal")

    def _trade_payload(
        self,
        side: str,
        mark_price: Decimal,
        execution_price: Decimal,
        quantity_base: Decimal,
        gross_quote: Decimal,
        fee_quote: Decimal,
        timestamp: str | None,
        reason: str,
    ) -> dict[str, Any]:
        equity_after = self.cash_quote + self.position_base * mark_price
        return {
            "timestamp": timestamp or self._now_iso(),
            "symbol": self.config.symbol,
            "side": side,
            "mark_price": float(mark_price),
            "execution_price": float(execution_price),
            "quantity_base": float(quantity_base),
            "gross_quote": float(gross_quote),
            "fee_quote": float(fee_quote),
            "cash_quote_after": float(self.cash_quote),
            "position_base_after": float(self.position_base),
            "equity_quote_after": float(equity_after),
            "reason": reason,
            "mode": "paper",
        }

    def trades_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.trades)

    def equity_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.equity_curve)
