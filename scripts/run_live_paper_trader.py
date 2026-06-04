"""Run a live paper-trading loop from Binance candles and a selected local model.

This script is paper-only. It fetches live market data, builds live features, converts model
inference into an execution signal through a simple configurable policy, executes that signal
against PaperBroker and writes JSONL events consumed by the Streamlit dashboard.

The execution policy can be edited while the process is running by changing:

    results/execution_logs/live_policy_config.json

Stop with Ctrl+C.

By default, the bot only infers and logs once per newly closed candle. The refresh loop
still polls more frequently so the process can react shortly after a candle closes without
writing duplicated signals for the same 5m interval.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dashboard.live_features import (  # noqa: E402
    LIVE_MODEL_FEATURE_COLUMNS,
    MIN_RECOMMENDED_CANDLES,
    build_live_features,
    feature_coverage,
    latest_complete_feature_row,
)
from src.dashboard.live_market import BinanceMarketDataClient, MarketDataConfig, klines_to_dataframe  # noqa: E402
from src.dashboard.model_registry import (  # noqa: E402
    build_model_input,
    discover_model_files,
    infer_feature_names,
    load_model,
    predict_signal,
    read_model_metadata,
)
from src.execution.execution_logger import JsonlExecutionLogger  # noqa: E402
from src.execution.paper_broker import PaperBroker, PaperBrokerConfig  # noqa: E402

SPOT_BASE_URL = "https://api.binance.com"
TESTNET_BASE_URL = "https://testnet.binance.vision"
DEFAULT_LOG_PATH = PROJECT_ROOT / "results" / "execution_logs" / "paper_trading.jsonl"
DEFAULT_STATE_PATH = PROJECT_ROOT / "results" / "execution_logs" / "live_paper_state.json"
DEFAULT_POLICY_CONFIG_PATH = PROJECT_ROOT / "results" / "execution_logs" / "live_policy_config.json"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


DEFAULT_POLICY_CONFIG: dict[str, object] = {
    "buy_threshold": 0.60,
    "sell_threshold": 0.40,
    "position_pct": 0.25,
    "sell_pct": 1.00,
    "allow_repeated_buy": False,
    "one_trade_per_candle": True,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def infer_assets(symbol: str) -> tuple[str, str]:
    symbol = symbol.upper()
    for quote in ["USDT", "USDC", "BUSD", "FDUSD", "BTC", "ETH"]:
        if symbol.endswith(quote):
            return symbol[: -len(quote)], quote
    return symbol[:-4], symbol[-4:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live paper trading runner for Cryptobot TFM.")
    parser.add_argument("--symbol", default="DOGEUSDT", help="Trading symbol. Default: DOGEUSDT")
    parser.add_argument("--interval", default="5m", help="Binance kline interval. Default: 5m")
    parser.add_argument("--candle-limit", type=int, default=500, help="Recent candles used for features. Default: 500")
    parser.add_argument("--refresh", type=int, default=10, help="Loop sleep time in seconds. Default: 10")
    parser.add_argument("--source", choices=["spot", "testnet"], default="spot", help="Market-data source. Default: spot")
    parser.add_argument("--model", default="none", help="Path to .joblib/.pkl model, or 'none' for EMA heuristic smoke test.")
    parser.add_argument("--list-models", action="store_true", help="List model files found under /models and exit.")
    parser.add_argument("--bankroll", type=str, default="1000", help="Initial quote bankroll for paper broker. Default: 1000")
    parser.add_argument("--position-pct", type=str, default="0.25", help="Fallback max quote exposure per BUY. Default: 0.25")
    parser.add_argument("--fee-bps", type=str, default="10", help="Paper fee in bps. Default: 10")
    parser.add_argument("--slippage-bps", type=str, default="5", help="Paper slippage in bps. Default: 5")
    parser.add_argument("--min-order", type=str, default="5", help="Minimum quote order size. Default: 5")
    parser.add_argument("--buy-threshold", type=float, default=0.60, help="Fallback probability threshold for BUY. Default: 0.60")
    parser.add_argument("--sell-threshold", type=float, default=0.40, help="Fallback probability threshold for SELL. Default: 0.40")
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH), help="JSONL log path consumed by dashboard.")
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH), help="Small JSON checkpoint path for paper balances.")
    parser.add_argument("--policy-config", default=str(DEFAULT_POLICY_CONFIG_PATH), help="JSON policy file optionally edited by Streamlit while running.")
    parser.add_argument("--reset-state", action="store_true", help="Start a fresh paper state even if a checkpoint exists.")
    parser.add_argument("--resume-state", action="store_true", help="Resume previous paper state from checkpoint. By default each run starts fresh.")
    parser.add_argument("--dry-run", action="store_true", help="Run one inference/execution cycle and exit.")
    parser.add_argument(
        "--process-open-candle",
        action="store_true",
        help="Use the currently open candle for inference. Disabled by default to avoid repeated signals inside the same 5m candle.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=60,
        help="Minimum seconds between console waiting messages when no new candle is available. Default: 60",
    )
    parser.add_argument(
        "--max-backfill-candles",
        type=int,
        default=100,
        help="Maximum number of missed closed candles to process sequentially after downtime. Default: 100",
    )
    return parser.parse_args()


def list_models() -> None:
    models = discover_model_files(DEFAULT_MODELS_DIR)
    if not models:
        print("No model files found under models/.")
        return
    print("Available models:")
    for model_path in models:
        print(f"- {model_path.relative_to(PROJECT_ROOT)}")


def resolve_model_path(label: str) -> Path | None:
    if label.lower() in {"none", "heuristic", "ema"}:
        return None
    path = Path(label)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return path


def load_checkpoint(path: Path) -> dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)


def clamp_float(value: object, minimum: float, maximum: float, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = fallback
    return min(max(parsed, minimum), maximum)


def load_policy_config(path: Path, args: argparse.Namespace) -> dict[str, object]:
    """Read execution policy from JSON, falling back to CLI/default values when missing."""

    policy = {
        **DEFAULT_POLICY_CONFIG,
        "buy_threshold": float(args.buy_threshold),
        "sell_threshold": float(args.sell_threshold),
        "position_pct": float(Decimal(str(args.position_pct))),
    }

    if path.exists() and path.stat().st_size > 0:
        try:
            with path.open("r", encoding="utf-8") as file:
                file_policy = json.load(file)
            if isinstance(file_policy, dict):
                policy.update(file_policy)
        except Exception as exc:  # keep daemon alive if the dashboard writes a broken partial file
            policy["policy_config_error"] = repr(exc)

    policy["buy_threshold"] = clamp_float(policy.get("buy_threshold"), 0.50, 0.95, float(args.buy_threshold))
    policy["sell_threshold"] = clamp_float(policy.get("sell_threshold"), 0.05, 0.50, float(args.sell_threshold))
    policy["position_pct"] = clamp_float(policy.get("position_pct"), 0.01, 1.00, float(Decimal(str(args.position_pct))))
    policy["sell_pct"] = clamp_float(policy.get("sell_pct"), 0.01, 1.00, 1.00)
    policy["allow_repeated_buy"] = bool(policy.get("allow_repeated_buy", False))
    policy["one_trade_per_candle"] = bool(policy.get("one_trade_per_candle", True))
    policy["updated_at_read_by_bot"] = utc_now_iso()
    return policy


def build_broker(args: argparse.Namespace, checkpoint: dict[str, object]) -> PaperBroker:
    base_asset, quote_asset = infer_assets(args.symbol)
    config = PaperBrokerConfig(
        symbol=args.symbol.upper(),
        base_asset=base_asset,
        quote_asset=quote_asset,
        initial_quote_balance=Decimal(str(args.bankroll)),
        initial_base_balance=Decimal("0"),
        fee_bps=Decimal(str(args.fee_bps)),
        slippage_bps=Decimal(str(args.slippage_bps)),
        max_position_quote_pct=Decimal(str(args.position_pct)),
        min_quote_order_qty=Decimal(str(args.min_order)),
    )
    broker = PaperBroker(config=config)

    if checkpoint and not args.reset_state:
        if str(checkpoint.get("symbol", "")).upper() == args.symbol.upper():
            broker.cash_quote = Decimal(str(checkpoint.get("cash_quote", broker.cash_quote)))
            broker.position_base = Decimal(str(checkpoint.get("position_base", broker.position_base)))

    return broker


def log_event(logger: JsonlExecutionLogger, event_type: str, run_id: str, sequence: int, payload: dict[str, object]) -> dict[str, object]:
    return logger.log_event(
        event_type,
        {
            "event_id": str(uuid4()),
            "run_id": run_id,
            "event_sequence": sequence,
            "event_time": utc_now_iso(),
            **payload,
        },
    )


def heuristic_signal(features_df: pd.DataFrame) -> dict[str, object]:
    latest = features_df.tail(1)
    if latest.empty:
        return {"signal": "HOLD", "confidence": None, "prediction_raw": None, "reason": "empty_features"}
    row = latest.iloc[0]
    ema_10 = row.get("ema_10")
    ema_20 = row.get("ema_20")
    rsi = row.get("rsi_14")
    if pd.isna(ema_10) or pd.isna(ema_20):
        return {"signal": "HOLD", "confidence": None, "prediction_raw": None, "reason": "insufficient_ema_features"}
    if ema_10 > ema_20 and (pd.isna(rsi) or rsi < 70):
        return {"signal": "BUY", "confidence": 0.51, "prediction_raw": "ema_10_gt_ema_20", "reason": "ema_heuristic_buy"}
    if ema_10 < ema_20 and (pd.isna(rsi) or rsi > 30):
        return {"signal": "SELL", "confidence": 0.49, "prediction_raw": "ema_10_lt_ema_20", "reason": "ema_heuristic_sell"}
    return {"signal": "HOLD", "confidence": 0.50, "prediction_raw": "neutral", "reason": "ema_heuristic_hold"}


def model_signal(model, model_path: Path, features_df: pd.DataFrame, buy_threshold: float, sell_threshold: float) -> tuple[dict[str, object], dict[str, object]]:
    metadata = read_model_metadata(model_path)
    feature_names = infer_feature_names(model, metadata)
    required_columns = feature_names or LIVE_MODEL_FEATURE_COLUMNS
    coverage = feature_coverage(features_df, required_columns=required_columns)

    latest_row = latest_complete_feature_row(features_df, required_columns=required_columns)
    diagnostics = {
        "model_path": str(model_path),
        "metadata_found": bool(metadata),
        "feature_count": len(feature_names) if feature_names else getattr(model, "n_features_in_", "unknown"),
        "expected_feature_columns": list(required_columns),
        "feature_coverage": coverage,
        "input_columns": [],
    }

    if latest_row.empty:
        reason = "no_complete_feature_row"
        null_columns = coverage.get("null_columns", [])
        missing_columns = coverage.get("missing_columns", [])
        if missing_columns:
            reason = f"missing_columns={missing_columns[:8]}"
        elif null_columns:
            reason = f"null_live_features={null_columns[:8]}"
        return {"signal": "HOLD", "confidence": None, "prediction_raw": None, "reason": reason}, diagnostics

    model_input, missing_columns = build_model_input(latest_row, feature_names, model)
    diagnostics["input_columns"] = list(model_input.columns) if not model_input.empty else []

    if missing_columns:
        return {"signal": "HOLD", "confidence": None, "prediction_raw": None, "reason": f"missing_columns={missing_columns[:8]}"}, diagnostics

    return predict_signal(model, model_input, buy_threshold=buy_threshold, sell_threshold=sell_threshold), diagnostics


def apply_execution_policy(
    *,
    raw_signal_payload: dict[str, object],
    broker: PaperBroker,
    price: float,
    candle_time: str,
    policy_config: dict[str, object],
    runtime_state: dict[str, object],
) -> dict[str, object]:
    """Convert a raw model signal into an executable paper order with explicit guardrails."""

    raw_signal = str(raw_signal_payload.get("signal", "HOLD")).upper().strip()
    mark_price = Decimal(str(price))
    cash_quote = Decimal(str(broker.cash_quote))
    position_base = Decimal(str(broker.position_base))
    equity_quote = cash_quote + position_base * mark_price
    min_order = Decimal(str(broker.config.min_quote_order_qty))
    position_pct = Decimal(str(policy_config.get("position_pct", 0.25)))
    sell_pct = Decimal(str(policy_config.get("sell_pct", 1.0)))
    allow_repeated_buy = bool(policy_config.get("allow_repeated_buy", False))
    one_trade_per_candle = bool(policy_config.get("one_trade_per_candle", True))

    decision = {
        "raw_model_signal": raw_signal,
        "executed_signal": "HOLD",
        "quote_amount": None,
        "quantity_base": None,
        "policy_reason": "policy_hold_default",
    }

    last_trade_candle_time = runtime_state.get("last_trade_candle_time")
    if raw_signal in {"BUY", "SELL"} and one_trade_per_candle and last_trade_candle_time == candle_time:
        decision["policy_reason"] = "policy_block_one_trade_per_candle"
        return decision

    if raw_signal == "BUY":
        if position_base > 0 and not allow_repeated_buy:
            decision["policy_reason"] = "policy_block_repeated_buy_position_open"
            return decision
        if cash_quote < min_order:
            decision["policy_reason"] = "policy_block_insufficient_cash"
            return decision
        requested_quote = min(equity_quote * position_pct, cash_quote)
        if requested_quote < min_order:
            decision["policy_reason"] = "policy_block_below_min_order"
            return decision
        decision.update(
            {
                "executed_signal": "BUY",
                "quote_amount": str(requested_quote),
                "policy_reason": "policy_execute_buy_capped_position_pct",
            }
        )
        return decision

    if raw_signal == "SELL":
        if position_base <= 0:
            decision["policy_reason"] = "policy_block_no_position_to_sell"
            return decision
        quantity_base = position_base * sell_pct
        if quantity_base <= 0:
            decision["policy_reason"] = "policy_block_zero_sell_quantity"
            return decision
        decision.update(
            {
                "executed_signal": "SELL",
                "quantity_base": str(quantity_base),
                "policy_reason": "policy_execute_sell_pct_position",
            }
        )
        return decision

    decision["policy_reason"] = "policy_hold_model_neutral"
    return decision




def get_signal_candles(candles_df: pd.DataFrame, process_open_candle: bool) -> tuple[pd.DataFrame, str]:
    """Return the candle subset allowed for inference and a label describing its status."""

    if candles_df.empty:
        return candles_df, "empty"

    if process_open_candle:
        return candles_df.copy(), "open_or_latest"

    now_utc = pd.Timestamp.now(tz="UTC")
    closed_df = candles_df[candles_df["close_time"] <= now_utc].copy()
    return closed_df, "closed"


def maybe_print_waiting(runtime_state: dict[str, object], args: argparse.Namespace, latest_candle_time: str | None) -> None:
    """Print a throttled waiting message without writing duplicate log events."""

    heartbeat_seconds = max(0, int(getattr(args, "heartbeat_seconds", 60)))
    if heartbeat_seconds <= 0:
        return

    now_ts = time.time()
    last_wait_print_ts = float(runtime_state.get("last_wait_print_ts", 0.0) or 0.0)
    if now_ts - last_wait_print_ts < heartbeat_seconds:
        return

    runtime_state["last_wait_print_ts"] = now_ts
    candle_msg = latest_candle_time if latest_candle_time else "unknown"
    print(
        f"[{utc_now_iso()}] waiting for new closed {args.interval} candle "
        f"latest_processed={candle_msg} poll_refresh={args.refresh}s"
    )


def select_candles_to_process(
    signal_candles_df: pd.DataFrame,
    *,
    process_open_candle: bool,
    runtime_state: dict[str, object],
    max_backfill_candles: int,
) -> tuple[pd.DataFrame, int]:
    """Select candles that still need inference, preserving closed-candle order.

    Fresh runs intentionally process only the latest available closed candle instead of replaying
    hundreds of historical candles. Resumed or long-running runs process every closed candle newer
    than the last processed one, capped by max_backfill_candles to avoid runaway catch-up after very
    long downtime.
    """

    if signal_candles_df.empty:
        return signal_candles_df.copy(), 0

    if process_open_candle:
        return signal_candles_df.tail(1).copy(), 0

    last_processed_candle_time = runtime_state.get("last_processed_candle_time")
    if not last_processed_candle_time:
        return signal_candles_df.tail(1).copy(), 0

    last_processed_ts = pd.Timestamp(last_processed_candle_time)
    if last_processed_ts.tzinfo is None:
        last_processed_ts = last_processed_ts.tz_localize("UTC")

    candidates_df = signal_candles_df[signal_candles_df["open_time"] > last_processed_ts].copy()
    if candidates_df.empty:
        return candidates_df, 0

    max_backfill_candles = int(max_backfill_candles)
    if max_backfill_candles <= 0:
        skipped = max(0, len(candidates_df) - 1)
        return candidates_df.tail(1).copy(), skipped

    if len(candidates_df) > max_backfill_candles:
        skipped = len(candidates_df) - max_backfill_candles
        return candidates_df.tail(max_backfill_candles).copy(), skipped

    return candidates_df, 0


def process_single_candle(
    *,
    args: argparse.Namespace,
    broker: PaperBroker,
    logger: JsonlExecutionLogger,
    run_id: str,
    sequence: int,
    model,
    model_path: Path | None,
    state_path: Path,
    policy_config: dict[str, object],
    runtime_state: dict[str, object],
    signal_candles_df: pd.DataFrame,
    target_candle: pd.Series,
    candle_status: str,
    backfill_position: int,
    backfill_total: int,
) -> int:
    """Run model inference and paper execution for exactly one candle.

    For backfilled candles, live features are calculated only with candles up to the target candle.
    This avoids accidentally using later candles as future information while catching up after sleep,
    network drops or temporary DNS failures.
    """

    price = float(target_candle["close"])
    candle_time = pd.Timestamp(target_candle["open_time"]).isoformat()
    candle_close_time = pd.Timestamp(target_candle["close_time"]).isoformat()

    history_until_target = signal_candles_df[signal_candles_df["open_time"] <= pd.Timestamp(target_candle["open_time"])].copy()
    features_df = build_live_features(history_until_target)

    if model is None:
        raw_signal_payload = heuristic_signal(features_df)
        model_diagnostics = {"model_path": None, "mode": "ema_heuristic"}
    else:
        raw_signal_payload, model_diagnostics = model_signal(
            model=model,
            model_path=model_path,
            features_df=features_df,
            buy_threshold=float(policy_config["buy_threshold"]),
            sell_threshold=float(policy_config["sell_threshold"]),
        )

    execution_decision = apply_execution_policy(
        raw_signal_payload=raw_signal_payload,
        broker=broker,
        price=price,
        candle_time=candle_time,
        policy_config=policy_config,
        runtime_state=runtime_state,
    )

    sequence += 1
    log_event(
        logger,
        "live_signal",
        run_id,
        sequence,
        {
            "symbol": args.symbol.upper(),
            "interval": args.interval,
            "source": args.source,
            "model_path": str(model_path) if model_path else "ema_heuristic",
            "candle_time": candle_time,
            "candle_close_time": candle_close_time,
            "candle_status": candle_status,
            "backfill_position": backfill_position,
            "backfill_total": backfill_total,
            "signal": execution_decision["executed_signal"],
            "model_signal": raw_signal_payload.get("signal"),
            "executed_signal": execution_decision["executed_signal"],
            "confidence": raw_signal_payload.get("confidence"),
            "prediction_raw": raw_signal_payload.get("prediction_raw"),
            "reason": execution_decision["policy_reason"],
            "model_reason": raw_signal_payload.get("reason"),
            "policy_reason": execution_decision["policy_reason"],
            "price": price,
            "mode": "paper_live",
            "model_diagnostics": model_diagnostics,
            "policy_config": policy_config,
        },
    )

    trade = broker.execute_signal(
        str(execution_decision["executed_signal"]),
        price=price,
        quote_amount=execution_decision.get("quote_amount"),
        quantity_base=execution_decision.get("quantity_base"),
        timestamp=candle_time,
        reason=str(execution_decision["policy_reason"]),
    )

    executed_side = str(trade.get("side", "HOLD")).upper()
    if executed_side in {"BUY", "SELL"}:
        runtime_state["last_trade_candle_time"] = candle_time
        runtime_state["last_trade_event_time"] = utc_now_iso()

    sequence += 1
    log_event(
        logger,
        "paper_trade",
        run_id,
        sequence,
        {
            **trade,
            "symbol": args.symbol.upper(),
            "interval": args.interval,
            "source": args.source,
            "model_path": str(model_path) if model_path else "ema_heuristic",
            "candle_time": candle_time,
            "candle_close_time": candle_close_time,
            "candle_status": candle_status,
            "backfill_position": backfill_position,
            "backfill_total": backfill_total,
            "signal": execution_decision["executed_signal"],
            "model_signal": raw_signal_payload.get("signal"),
            "executed_signal": execution_decision["executed_signal"],
            "confidence": raw_signal_payload.get("confidence"),
            "policy_reason": execution_decision["policy_reason"],
            "model_reason": raw_signal_payload.get("reason"),
            "policy_config": policy_config,
            "mode": "paper_live",
        },
    )

    save_checkpoint(
        state_path,
        {
            "run_id": run_id,
            "symbol": args.symbol.upper(),
            "updated_at": utc_now_iso(),
            "cash_quote": trade.get("cash_quote_after"),
            "position_base": trade.get("position_base_after"),
            "equity_quote": trade.get("equity_quote_after"),
            "last_price": price,
            "last_processed_candle_time": candle_time,
            "last_processed_candle_close_time": candle_close_time,
            "last_signal": execution_decision["executed_signal"],
            "last_model_signal": raw_signal_payload.get("signal"),
            "last_reason": execution_decision["policy_reason"],
            "policy_config": policy_config,
        },
    )

    runtime_state["last_processed_candle_time"] = candle_time
    runtime_state["last_processed_candle_close_time"] = candle_close_time

    backfill_label = ""
    if backfill_total > 1:
        backfill_label = f" backfill={backfill_position}/{backfill_total}"

    print(
        f"[{utc_now_iso()}] {args.symbol.upper()} {args.interval} "
        f"closed_candle={candle_time}{backfill_label} price={price:.8f} model={raw_signal_payload.get('signal')} exec={execution_decision['executed_signal']} "
        f"equity={float(trade.get('equity_quote_after', 0)):.4f} "
        f"cash={float(trade.get('cash_quote_after', 0)):.4f} "
        f"pos={float(trade.get('position_base_after', 0)):.8f} "
        f"reason={execution_decision['policy_reason']}"
    )
    return sequence


def run_once(
    *,
    args: argparse.Namespace,
    client: BinanceMarketDataClient,
    broker: PaperBroker,
    logger: JsonlExecutionLogger,
    run_id: str,
    sequence: int,
    model,
    model_path: Path | None,
    state_path: Path,
    policy_config_path: Path,
    runtime_state: dict[str, object],
) -> int:
    policy_config = load_policy_config(policy_config_path, args)

    raw_klines = client.klines(symbol=args.symbol, interval=args.interval, limit=args.candle_limit)
    candles_df = klines_to_dataframe(raw_klines)
    signal_candles_df, candle_status = get_signal_candles(candles_df, process_open_candle=bool(args.process_open_candle))

    if signal_candles_df.empty:
        maybe_print_waiting(runtime_state, args, latest_candle_time=None)
        return sequence

    candles_to_process_df, skipped_backfill = select_candles_to_process(
        signal_candles_df,
        process_open_candle=bool(args.process_open_candle),
        runtime_state=runtime_state,
        max_backfill_candles=int(args.max_backfill_candles),
    )

    if candles_to_process_df.empty:
        latest_candle_time = pd.Timestamp(signal_candles_df.iloc[-1]["open_time"]).isoformat()
        maybe_print_waiting(runtime_state, args, latest_candle_time=latest_candle_time)
        return sequence

    if skipped_backfill > 0:
        sequence += 1
        log_event(
            logger,
            "live_backfill_truncated",
            run_id,
            sequence,
            {
                "symbol": args.symbol.upper(),
                "interval": args.interval,
                "source": args.source,
                "skipped_backfill_candles": skipped_backfill,
                "max_backfill_candles": int(args.max_backfill_candles),
                "mode": "paper_live",
            },
        )
        print(
            f"[{utc_now_iso()}] WARNING: skipped {skipped_backfill} older missed candles "
            f"because max_backfill_candles={int(args.max_backfill_candles)}"
        )

    backfill_total = len(candles_to_process_df)
    for backfill_position, (_, target_candle) in enumerate(candles_to_process_df.iterrows(), start=1):
        sequence = process_single_candle(
            args=args,
            broker=broker,
            logger=logger,
            run_id=run_id,
            sequence=sequence,
            model=model,
            model_path=model_path,
            state_path=state_path,
            policy_config=policy_config,
            runtime_state=runtime_state,
            signal_candles_df=signal_candles_df,
            target_candle=target_candle,
            candle_status=candle_status,
            backfill_position=backfill_position,
            backfill_total=backfill_total,
        )

    return sequence

def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()

    if args.candle_limit < MIN_RECOMMENDED_CANDLES:
        print(
            f"WARNING: candle-limit={args.candle_limit} is below the recommended "
            f"{MIN_RECOMMENDED_CANDLES} candles needed for 24h support/resistance live features."
        )

    if args.list_models:
        list_models()
        return

    log_path = Path(args.log_path)
    state_path = Path(args.state_path)
    policy_config_path = Path(args.policy_config)
    # Default behaviour is intentionally fresh: a new live paper run should start from the requested bankroll.
    # Previous versions restored live_paper_state.json automatically, which made new runs inherit old DOGE positions.
    if args.reset_state and state_path.exists():
        state_path.unlink()

    checkpoint = load_checkpoint(state_path) if args.resume_state and not args.reset_state else {}
    broker = build_broker(args, checkpoint)
    logger = JsonlExecutionLogger(log_path)
    run_id = f"live-paper-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    sequence = 0
    runtime_state: dict[str, object] = {}
    if checkpoint and args.resume_state and not args.reset_state:
        if checkpoint.get("last_processed_candle_time"):
            runtime_state["last_processed_candle_time"] = checkpoint.get("last_processed_candle_time")
        if checkpoint.get("last_processed_candle_close_time"):
            runtime_state["last_processed_candle_close_time"] = checkpoint.get("last_processed_candle_close_time")

    base_url = TESTNET_BASE_URL if args.source == "testnet" else SPOT_BASE_URL
    client = BinanceMarketDataClient(MarketDataConfig(base_url=base_url, timeout_seconds=15))

    model_path = resolve_model_path(args.model)
    model = load_model(model_path) if model_path else None
    initial_policy_config = load_policy_config(policy_config_path, args)

    sequence += 1
    log_event(
        logger,
        "live_bot_started",
        run_id,
        sequence,
        {
            "symbol": args.symbol.upper(),
            "interval": args.interval,
            "source": args.source,
            "base_url": base_url,
            "model_path": str(model_path) if model_path else "ema_heuristic",
            "bankroll": float(Decimal(str(args.bankroll))),
            "position_pct": float(initial_policy_config.get("position_pct", Decimal(str(args.position_pct)))),
            "fee_bps": float(Decimal(str(args.fee_bps))),
            "slippage_bps": float(Decimal(str(args.slippage_bps))),
            "policy_config_path": str(policy_config_path),
            "policy_config": initial_policy_config,
            "mode": "paper_live",
            "process_open_candle": bool(args.process_open_candle),
            "inference_frequency": "every_poll" if args.process_open_candle else "new_closed_candle_only",
            "heartbeat_seconds": int(args.heartbeat_seconds),
            "max_backfill_candles": int(args.max_backfill_candles),
            "state_restored": bool(checkpoint and args.resume_state and not args.reset_state),
            "fresh_state": not bool(checkpoint and args.resume_state and not args.reset_state),
        },
    )

    print("Live paper trader started. Stop with Ctrl+C.")
    print(f"Run ID: {run_id}")
    print(f"Log: {log_path}")
    print(f"State: {state_path}")
    print(f"Policy: {policy_config_path}")
    if checkpoint and args.resume_state and not args.reset_state:
        print("State mode: RESUMED previous checkpoint")
    else:
        print("State mode: FRESH bankroll")
    print(f"Initial cash: {float(broker.cash_quote):.2f}")
    print(f"Initial position: {float(broker.position_base):.8f}")
    if args.process_open_candle:
        print("Inference mode: EVERY POLL using latest open/current candle")
    else:
        print("Inference mode: NEW CLOSED CANDLE ONLY")

    try:
        while True:
            try:
                sequence = run_once(
                    args=args,
                    client=client,
                    broker=broker,
                    logger=logger,
                    run_id=run_id,
                    sequence=sequence,
                    model=model,
                    model_path=model_path,
                    state_path=state_path,
                    policy_config_path=policy_config_path,
                    runtime_state=runtime_state,
                )
            except Exception as exc:
                sequence += 1
                log_event(
                    logger,
                    "live_bot_error",
                    run_id,
                    sequence,
                    {
                        "symbol": args.symbol.upper(),
                        "interval": args.interval,
                        "source": args.source,
                        "error": repr(exc),
                        "mode": "paper_live",
                    },
                )
                print(f"[{utc_now_iso()}] ERROR: {exc!r}")

            if args.dry_run:
                break
            time.sleep(max(1, int(args.refresh)))
    except KeyboardInterrupt:
        sequence += 1
        log_event(logger, "live_bot_stopped", run_id, sequence, {"symbol": args.symbol.upper(), "mode": "paper_live"})
        print("Live paper trader stopped.")


if __name__ == "__main__":
    main()
