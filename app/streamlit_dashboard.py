"""Streamlit dashboard for the Cryptobot TFM execution and live market layer.

Run from the project root with:

    python -m streamlit run app/streamlit_dashboard.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dashboard.dashboard_data import (  # noqa: E402
    build_equity_curve,
    build_event_counts,
    build_operations,
    build_signals,
    compute_dashboard_metrics,
    filter_by_run,
    format_number,
    get_available_run_ids,
    read_jsonl_logs,
    read_summary,
)
from src.dashboard.live_features import build_live_features  # noqa: E402
from src.dashboard.live_market import (  # noqa: E402
    BinanceMarketDataClient,
    MarketDataConfig,
    get_symbol_status,
    klines_to_dataframe,
)
from src.dashboard.model_registry import (  # noqa: E402
    build_model_input,
    discover_model_files,
    infer_feature_names,
    load_model,
    predict_signal,
    read_model_metadata,
)

DEFAULT_LOG_PATH = PROJECT_ROOT / "results" / "execution_logs" / "paper_trading.jsonl"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "results" / "11_testnet_paper_trading_summary.csv"
DEFAULT_POLICY_CONFIG_PATH = PROJECT_ROOT / "results" / "execution_logs" / "live_policy_config.json"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
SPOT_BASE_URL = "https://api.binance.com"
TESTNET_BASE_URL = "https://testnet.binance.vision"
LOCAL_TIMEZONE = ZoneInfo("Europe/Madrid")

DEFAULT_EXECUTION_POLICY = {
    "buy_threshold": 0.60,
    "sell_threshold": 0.40,
    "position_pct": 0.25,
    "sell_pct": 1.00,
    "allow_repeated_buy": False,
    "one_trade_per_candle": True,
}


st.set_page_config(
    page_title="Cryptobot TFM Dashboard",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(ttl=5)
def load_dashboard_data(log_path: str, summary_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load logs and summary with a short cache to keep refresh responsive."""

    logs_df = read_jsonl_logs(log_path)
    summary_df = read_summary(summary_path)
    return logs_df, summary_df


@st.cache_data(ttl=5)
def load_live_candles(base_url: str, symbol: str, interval: str, limit: int) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load live candles and compact symbol metadata from Binance public REST endpoints."""

    client = BinanceMarketDataClient(MarketDataConfig(base_url=base_url))
    symbol_status = get_symbol_status(client, symbol)
    raw_klines = client.klines(symbol=symbol, interval=interval, limit=limit)
    candles_df = klines_to_dataframe(raw_klines)
    return candles_df, symbol_status


@st.cache_data(ttl=5)
def load_server_time(base_url: str) -> pd.Timestamp | None:
    """Load Binance server time from the selected market data endpoint."""

    client = BinanceMarketDataClient(MarketDataConfig(base_url=base_url))
    payload = client._get("/api/v3/time")
    server_time_ms = payload.get("serverTime")
    if server_time_ms is None:
        return None
    return pd.to_datetime(server_time_ms, unit="ms", utc=True)


@st.cache_resource(show_spinner=False)
def load_cached_model(model_path: str):
    """Load a serialized model once per path."""

    return load_model(model_path)


def render_metric(label: str, value: str, delta: str | None = None, help_text: str | None = None) -> None:
    """Render a metric with a small wrapper for consistent missing values."""

    st.metric(label=label, value=value, delta=delta, help=help_text)


def format_clock(value: datetime | pd.Timestamp | None, target_timezone: ZoneInfo | timezone = timezone.utc) -> str:
    """Format a timestamp as a compact dashboard clock."""

    if value is None or pd.isna(value):
        return "N/A"
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert(target_timezone).strftime("%H:%M:%S")


def format_clock_with_zone(value: datetime | pd.Timestamp | None, target_timezone: ZoneInfo | timezone, zone_label: str) -> str:
    """Format a timestamp with an explicit timezone suffix."""

    clock = format_clock(value, target_timezone)
    return f"{clock} {zone_label}" if clock != "N/A" else "N/A"


def human_market_status(raw_status: object) -> str:
    """Convert Binance symbol status into a presentation-friendly label."""

    status = str(raw_status or "N/A").upper()
    if status == "TRADING":
        return "Mercado activo"
    if status == "BREAK":
        return "Pausado"
    if status == "HALT":
        return "Suspendido"
    return status


def coerce_event_time(logs_df: pd.DataFrame) -> pd.DataFrame:
    """Return logs with a parsed timestamp column for robust sorting."""

    if logs_df.empty or "event_time" not in logs_df.columns:
        return logs_df.copy()
    output_df = logs_df.copy()
    output_df["event_time_parsed"] = pd.to_datetime(output_df["event_time"], errors="coerce", utc=True)
    return output_df


def get_latest_live_run(logs_df: pd.DataFrame) -> str | None:
    """Detect the latest run produced by the live paper-trading daemon."""

    if logs_df.empty or "run_id" not in logs_df.columns:
        return None
    work_df = coerce_event_time(logs_df)
    if "mode" in work_df.columns:
        live_df = work_df[work_df["mode"].astype(str).eq("paper_live")].copy()
    else:
        live_df = pd.DataFrame()
    if live_df.empty and "event_type" in work_df.columns:
        live_df = work_df[work_df["event_type"].astype(str).str.startswith("live_", na=False)].copy()
    if live_df.empty:
        return None
    live_df = live_df.dropna(subset=["run_id"])
    if live_df.empty:
        return None
    live_df = live_df.sort_values("event_time_parsed", na_position="first")
    return str(live_df["run_id"].iloc[-1])


def summarize_live_deployment(logs_df: pd.DataFrame, run_id: str | None = None) -> dict[str, object]:
    """Build a compact status payload for the currently running or latest live paper deployment."""

    if logs_df.empty:
        return {"active": False, "reason": "empty_logs"}
    work_df = coerce_event_time(logs_df)
    if run_id is None:
        run_id = get_latest_live_run(work_df)
    if run_id is None:
        return {"active": False, "reason": "no_live_run_found"}

    run_df = work_df[work_df["run_id"].astype(str).eq(str(run_id))].copy()
    if run_df.empty:
        return {"active": False, "run_id": run_id, "reason": "run_not_found"}

    run_df = run_df.sort_values("event_time_parsed", na_position="first")
    event_types = run_df.get("event_type", pd.Series(dtype=str)).astype(str)
    started_df = run_df[event_types.eq("live_bot_started")]
    stopped_df = run_df[event_types.eq("live_bot_stopped")]
    errors_df = run_df[event_types.eq("live_bot_error")]
    signals_df = run_df[event_types.eq("live_signal")]
    trades_df = run_df[event_types.eq("paper_trade")]

    started_payload = started_df.iloc[-1].to_dict() if not started_df.empty else {}
    last_signal = signals_df.iloc[-1].to_dict() if not signals_df.empty else {}
    last_trade = trades_df.iloc[-1].to_dict() if not trades_df.empty else {}
    last_event = run_df.iloc[-1].to_dict()

    last_time = last_event.get("event_time_parsed")
    seconds_since_last = None
    if pd.notna(last_time):
        seconds_since_last = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(last_time)).total_seconds()

    is_running = stopped_df.empty and (seconds_since_last is None or seconds_since_last < 120)

    return {
        "active": True,
        "run_id": run_id,
        "is_running": is_running,
        "seconds_since_last": seconds_since_last,
        "event_count": int(len(run_df)),
        "signal_count": int(len(signals_df)),
        "trade_count": int(len(trades_df)),
        "error_count": int(len(errors_df)),
        "symbol": last_signal.get("symbol") or last_trade.get("symbol") or started_payload.get("symbol"),
        "interval": last_signal.get("interval") or last_trade.get("interval") or started_payload.get("interval"),
        "source": last_signal.get("source") or last_trade.get("source") or started_payload.get("source"),
        "model_path": last_signal.get("model_path") or last_trade.get("model_path") or started_payload.get("model_path"),
        "bankroll": started_payload.get("bankroll"),
        "position_pct": started_payload.get("position_pct"),
        "last_signal": last_signal.get("signal"),
        "last_reason": last_signal.get("reason"),
        "last_price": last_signal.get("price") or last_trade.get("price"),
        "cash_quote": last_trade.get("cash_quote_after"),
        "position_base": last_trade.get("position_base_after"),
        "equity_quote": last_trade.get("equity_quote_after"),
        "last_event_time": last_event.get("event_time"),
    }


def render_live_deployment_status(log_path: str) -> None:
    """Render a status panel that links the external live paper trader process with the dashboard."""

    logs_df = read_jsonl_logs(log_path)
    deployment = summarize_live_deployment(logs_df)

    if not deployment.get("active"):
        st.info("No se ha detectado ningún proceso `run_live_paper_trader.py` en los logs todavía.")
        st.code("python scripts/run_live_paper_trader.py --model none --bankroll 1000 --symbol DOGEUSDT --refresh 10")
        return

    status_label = "RUNNING" if deployment.get("is_running") else "SIN LATIDO RECIENTE"
    st.subheader("Live paper deployment")
    st.caption(
        "Este bloque lee el proceso externo `run_live_paper_trader.py`. El selector de modelo del sidebar solo controla la inferencia visual del dashboard."
    )

    raw_model_path = deployment.get("model_path")
    if raw_model_path in [None, "", "none"]:
        deploy_model_label = "EMA heuristic"
    else:
        deploy_model_label = Path(str(raw_model_path)).stem

    deploy_cols = st.columns(6)
    with deploy_cols[0]:
        render_metric("Bot", status_label, help_text="RUNNING indica que hay eventos recientes del proceso live paper. No implica órdenes reales.")
    with deploy_cols[1]:
        render_metric("Modelo deploy", deploy_model_label)
    with deploy_cols[2]:
        render_metric("Última señal bot", str(deployment.get("last_signal") or "N/A"))
    with deploy_cols[3]:
        render_metric("Equity bot", format_number(deployment.get("equity_quote"), 2, " USDT"))
    with deploy_cols[4]:
        render_metric("Cash bot", format_number(deployment.get("cash_quote"), 2, " USDT"))
    with deploy_cols[5]:
        render_metric("Posición", format_number(deployment.get("position_base"), 2, " DOGE"))

    st.caption(
        f"Run ID bot: `{deployment.get('run_id')}` | Símbolo: `{deployment.get('symbol')}` | Intervalo: `{deployment.get('interval')}` | "
        f"Fuente: `{deployment.get('source')}` | Último evento: `{deployment.get('last_event_time')}` | Razón: `{deployment.get('last_reason')}`"
    )


def load_execution_policy(policy_path: str | Path) -> dict[str, object]:
    """Load the live paper execution policy edited from Streamlit."""

    path = Path(policy_path)
    policy = dict(DEFAULT_EXECUTION_POLICY)
    if path.exists() and path.stat().st_size > 0:
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            if isinstance(payload, dict):
                policy.update(payload)
        except Exception as exc:
            policy["policy_config_error"] = repr(exc)
    return policy


def save_execution_policy(policy_path: str | Path, policy: dict[str, object]) -> None:
    """Persist the live paper execution policy so the external bot can read it."""

    path = Path(policy_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(policy)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def get_live_run_ids(logs_df: pd.DataFrame) -> list[str]:
    """Return run ids produced by the live paper trader, ordered by most recent event first."""

    if logs_df.empty or "run_id" not in logs_df.columns:
        return []

    work_df = coerce_event_time(logs_df)
    live_mask = pd.Series(False, index=work_df.index)
    if "mode" in work_df.columns:
        live_mask = live_mask | work_df["mode"].astype(str).eq("paper_live")
    if "event_type" in work_df.columns:
        event_type = work_df["event_type"].astype(str)
        live_mask = live_mask | event_type.str.startswith("live_", na=False) | event_type.eq("paper_trade")

    live_df = work_df[live_mask].dropna(subset=["run_id"]).copy()
    if live_df.empty:
        return []

    ordered_runs_df = (
        live_df.groupby("run_id", dropna=False)["event_time_parsed"]
        .max()
        .reset_index()
        .sort_values("event_time_parsed", ascending=False, na_position="last")
    )
    return ordered_runs_df["run_id"].astype(str).tolist()


def get_live_run_df(logs_df: pd.DataFrame, run_id: str | None) -> pd.DataFrame:
    """Return the logs belonging to one deployed bot run."""

    if logs_df.empty or run_id is None or "run_id" not in logs_df.columns:
        return pd.DataFrame()
    run_df = logs_df[logs_df["run_id"].astype(str).eq(str(run_id))].copy()
    return coerce_event_time(run_df).sort_values("event_time_parsed", na_position="first")


def build_live_trade_tables(run_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build signal, action and equity tables for the Deployed Bot tab."""

    if run_df.empty or "event_type" not in run_df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    signals_df = run_df[run_df["event_type"].astype(str).eq("live_signal")].copy()
    actions_df = run_df[run_df["event_type"].astype(str).eq("paper_trade")].copy()
    equity_df = build_equity_curve(run_df)

    signal_columns = [
        "event_time",
        "candle_time",
        "symbol",
        "model_signal",
        "executed_signal",
        "signal",
        "confidence",
        "price",
        "model_reason",
        "policy_reason",
        "reason",
    ]
    action_columns = [
        "event_time",
        "timestamp",
        "symbol",
        "side",
        "mark_price",
        "execution_price",
        "quantity_base",
        "gross_quote",
        "fee_quote",
        "cash_quote_after",
        "position_base_after",
        "equity_quote_after",
        "model_signal",
        "executed_signal",
        "policy_reason",
    ]
    signal_columns = [column for column in signal_columns if column in signals_df.columns]
    action_columns = [column for column in action_columns if column in actions_df.columns]

    return signals_df[signal_columns].copy() if signal_columns else signals_df, actions_df[action_columns].copy() if action_columns else actions_df, equity_df


def extract_latest_policy_from_run(run_df: pd.DataFrame) -> dict[str, object]:
    """Extract the most recent policy_config dict embedded in live logs."""

    if run_df.empty or "policy_config" not in run_df.columns:
        return {}
    values = run_df["policy_config"].dropna()
    if values.empty:
        return {}
    latest = values.iloc[-1]
    return latest if isinstance(latest, dict) else {}


def render_execution_policy_controls(policy_path: str | Path, run_df: pd.DataFrame) -> dict[str, object]:
    """Render an editable execution-policy panel and persist changes for the daemon."""

    policy_path = Path(policy_path)
    saved_policy = load_execution_policy(policy_path)
    observed_policy = extract_latest_policy_from_run(run_df)

    st.markdown("#### Execution policy")
    st.caption("Estos parámetros se guardan en JSON. El proceso `run_live_paper_trader.py` los lee en cada ciclo sin reiniciar.")

    with st.form("live_execution_policy_form"):
        buy_threshold = st.slider(
            "BUY threshold",
            min_value=0.50,
            max_value=0.95,
            value=float(saved_policy.get("buy_threshold", DEFAULT_EXECUTION_POLICY["buy_threshold"])),
            step=0.01,
            help="Probabilidad mínima de subida para permitir BUY.",
        )
        sell_threshold = st.slider(
            "SELL threshold",
            min_value=0.05,
            max_value=0.50,
            value=float(saved_policy.get("sell_threshold", DEFAULT_EXECUTION_POLICY["sell_threshold"])),
            step=0.01,
            help="Probabilidad máxima de subida para permitir SELL.",
        )
        position_pct = st.slider(
            "Position size por BUY",
            min_value=0.01,
            max_value=1.00,
            value=float(saved_policy.get("position_pct", DEFAULT_EXECUTION_POLICY["position_pct"])),
            step=0.01,
            help="Fracción máxima de equity/cash que puede usar cada BUY.",
        )
        sell_pct = st.slider(
            "Sell size por SELL",
            min_value=0.01,
            max_value=1.00,
            value=float(saved_policy.get("sell_pct", DEFAULT_EXECUTION_POLICY["sell_pct"])),
            step=0.01,
            help="Fracción de la posición DOGE que se vende cuando llega SELL.",
        )
        allow_repeated_buy = st.checkbox(
            "Permitir BUY repetidos con posición abierta",
            value=bool(saved_policy.get("allow_repeated_buy", DEFAULT_EXECUTION_POLICY["allow_repeated_buy"])),
        )
        one_trade_per_candle = st.checkbox(
            "Máximo una operación por vela",
            value=bool(saved_policy.get("one_trade_per_candle", DEFAULT_EXECUTION_POLICY["one_trade_per_candle"])),
        )
        submitted = st.form_submit_button("Guardar policy")

    updated_policy = {
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "position_pct": position_pct,
        "sell_pct": sell_pct,
        "allow_repeated_buy": allow_repeated_buy,
        "one_trade_per_candle": one_trade_per_candle,
    }

    if submitted:
        save_execution_policy(policy_path, updated_policy)
        st.success("Policy guardada. El bot externo la aplicará en el siguiente ciclo.")
        st.cache_data.clear()

    st.caption(f"Policy file: `{policy_path}`")
    if observed_policy:
        with st.expander("Última policy observada en logs"):
            st.json(observed_policy)

    return updated_policy

def build_candlestick_chart(candles_df: pd.DataFrame, features_df: pd.DataFrame, signal_payload: dict[str, object] | None) -> go.Figure:
    """Build the live DOGE candlestick chart used in the dashboard."""

    chart_df = candles_df.copy()
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=chart_df["open_time"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="OHLC",
        )
    )

    for column, label in [("ema_10", "EMA 10"), ("ema_20", "EMA 20")]:
        if column in features_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=features_df["open_time"],
                    y=features_df[column],
                    mode="lines",
                    name=label,
                )
            )

    if signal_payload and not chart_df.empty:
        latest = chart_df.iloc[-1]
        signal = str(signal_payload.get("signal", "HOLD"))
        fig.add_trace(
            go.Scatter(
                x=[latest["open_time"]],
                y=[latest["close"]],
                mode="markers+text",
                text=[signal],
                textposition="top center",
                marker={"size": 12},
                name="Last signal",
            )
        )

    fig.update_layout(
        height=520,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_xanchor="right",
        legend_x=1,
    )
    return fig



def build_deployed_bot_chart(candles_df: pd.DataFrame, features_df: pd.DataFrame, run_df: pd.DataFrame) -> go.Figure:
    """Build a broker-style live candlestick chart with executed bot entry/exit markers."""

    fig = build_candlestick_chart(candles_df, features_df, signal_payload=None)
    fig.update_layout(title="Deployed bot actions over live DOGE candles")

    if run_df.empty or "event_type" not in run_df.columns:
        return fig

    # Only actual paper executions are plotted as broker-style entry/exit markers.
    trades_df = run_df[run_df["event_type"].astype(str).eq("paper_trade")].copy()
    if trades_df.empty:
        return fig

    # Normalise side, time and price across logs generated by different notebook/script versions.
    if "side" in trades_df.columns:
        trades_df["action_side"] = trades_df["side"].astype(str).str.upper()
    elif "executed_signal" in trades_df.columns:
        trades_df["action_side"] = trades_df["executed_signal"].astype(str).str.upper()
    else:
        return fig

    # Select the first time column that actually contains valid datetimes.
    # Some paper-trade rows include a legacy `timestamp` column filled with None; using it blindly hides markers.
    time_source = None
    action_time_series = None
    for candidate in ["timestamp", "event_time", "candle_time"]:
        if candidate not in trades_df.columns:
            continue
        parsed_times = pd.to_datetime(trades_df[candidate], errors="coerce", utc=True)
        if parsed_times.notna().any():
            time_source = candidate
            action_time_series = parsed_times
            break
    if time_source is None or action_time_series is None:
        return fig
    trades_df["action_time"] = action_time_series

    # Prefer the simulated execution price. Fall back to mark_price/price for older log versions.
    price_source = None
    for candidate in ["execution_price", "mark_price", "price"]:
        if candidate in trades_df.columns:
            price_source = candidate
            break
    if price_source is None:
        return fig
    trades_df["action_price"] = pd.to_numeric(trades_df[price_source], errors="coerce")

    trades_df = trades_df[trades_df["action_side"].isin(["BUY", "SELL"])].copy()
    trades_df = trades_df.dropna(subset=["action_time", "action_price"])
    if trades_df.empty:
        return fig

    # Keep only markers visible in the current chart range, but use a small tolerance around the last candle.
    if not candles_df.empty and "open_time" in candles_df.columns:
        candle_times = pd.to_datetime(candles_df["open_time"], errors="coerce", utc=True)
        min_time = candle_times.min()
        max_time = candle_times.max()
        if pd.notna(min_time) and pd.notna(max_time):
            max_time = max_time + pd.Timedelta(minutes=5)
            trades_df = trades_df[(trades_df["action_time"] >= min_time) & (trades_df["action_time"] <= max_time)].copy()
    if trades_df.empty:
        return fig

    def _fmt_float(value: object, decimals: int = 4) -> str:
        try:
            if pd.isna(value):
                return "N/A"
            return f"{float(value):,.{decimals}f}"
        except Exception:
            return "N/A"

    for column in ["quantity_base", "gross_quote", "fee_quote", "equity_quote_after"]:
        if column not in trades_df.columns:
            trades_df[column] = pd.NA
    if "policy_reason" not in trades_df.columns:
        trades_df["policy_reason"] = trades_df.get("reason", "")

    trades_df["hover_quantity"] = trades_df["quantity_base"].apply(lambda value: _fmt_float(value, 2))
    trades_df["hover_gross"] = trades_df["gross_quote"].apply(lambda value: _fmt_float(value, 2))
    trades_df["hover_fee"] = trades_df["fee_quote"].apply(lambda value: _fmt_float(value, 4))
    trades_df["hover_equity"] = trades_df["equity_quote_after"].apply(lambda value: _fmt_float(value, 2))
    trades_df["hover_reason"] = trades_df["policy_reason"].fillna("").astype(str)

    marker_config = {
        "BUY": {
            "symbol": "triangle-up",
            "label": "Entry BUY",
            "text": "BUY",
            "textposition": "bottom center",
            "color": "#00CC96",
            "line_color": "#006B4E",
        },
        "SELL": {
            "symbol": "triangle-down",
            "label": "Exit SELL",
            "text": "SELL",
            "textposition": "top center",
            "color": "#EF553B",
            "line_color": "#8B1A10",
        },
    }

    for side, config in marker_config.items():
        side_df = trades_df[trades_df["action_side"].eq(side)].copy()
        if side_df.empty:
            continue

        customdata = side_df[["hover_quantity", "hover_gross", "hover_fee", "hover_equity", "hover_reason"]].to_numpy()
        fig.add_trace(
            go.Scatter(
                x=side_df["action_time"],
                y=side_df["action_price"],
                mode="markers+text",
                text=[config["text"]] * len(side_df),
                textposition=config["textposition"],
                marker={
                    "size": 17,
                    "symbol": config["symbol"],
                    "color": config["color"],
                    "line": {"width": 1.5, "color": config["line_color"]},
                },
                name=config["label"],
                customdata=customdata,
                hovertemplate=(
                    f"<b>{config['label']}</b><br>"
                    "Time: %{x}<br>"
                    "Execution price: %{y:.8f}<br>"
                    "Quantity DOGE: %{customdata[0]}<br>"
                    "Gross quote: %{customdata[1]} USDT<br>"
                    "Fee: %{customdata[2]} USDT<br>"
                    "Equity after: %{customdata[3]} USDT<br>"
                    "Reason: %{customdata[4]}"
                    "<extra></extra>"
                ),
            )
        )

        # Add faint vertical guides so executions remain visible when the marker overlaps a candle body.
        for action_time in side_df["action_time"]:
            fig.add_vline(
                x=action_time,
                line_width=1,
                line_dash="dot",
                line_color=config["color"],
                opacity=0.35,
            )

    # Connect executed trades with a thin line to make round trips easier to read visually.
    ordered_trades_df = trades_df.sort_values("action_time")
    if len(ordered_trades_df) >= 2:
        fig.add_trace(
            go.Scatter(
                x=ordered_trades_df["action_time"],
                y=ordered_trades_df["action_price"],
                mode="lines",
                line={"dash": "dot", "width": 1},
                name="Trade path",
                hoverinfo="skip",
            )
        )

    return fig



def render_deployed_bot_tab(
    *,
    log_path: str,
    policy_path: str | Path,
    candles_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> None:
    """Render the visual deployed-bot tab with policy controls and action markers."""

    logs_df = read_jsonl_logs(log_path)
    if logs_df.empty:
        st.info("Todavía no hay logs. Lanza `run_live_paper_trader.py` para ver el bot desplegado.")
        st.code("python scripts/run_live_paper_trader.py --model models/xgboost_doge.joblib --bankroll 1000 --symbol DOGEUSDT --refresh 10")
        return

    live_run_ids = get_live_run_ids(logs_df)
    if not live_run_ids:
        st.info("Hay logs, pero no se detectan runs del bot live paper.")
        return

    selected_run_id = st.selectbox(
        "Run deployed",
        options=live_run_ids,
        index=0,
        help="Runs generados por `run_live_paper_trader.py`, ordenados por último evento.",
    )
    run_df = get_live_run_df(logs_df, selected_run_id)
    deployment = summarize_live_deployment(logs_df, selected_run_id)
    signals_df, actions_df, equity_df = build_live_trade_tables(run_df)

    left_col, right_col = st.columns([0.30, 0.70])
    with left_col:
        render_execution_policy_controls(policy_path, run_df)

        raw_model_path = deployment.get("model_path")
        if raw_model_path in [None, "", "none", "ema_heuristic"]:
            deploy_model_label = "EMA heuristic"
        else:
            deploy_model_label = Path(str(raw_model_path)).stem

        st.markdown("#### Bot state")
        st.metric("Estado", "RUNNING" if deployment.get("is_running") else "SIN LATIDO")
        st.metric("Modelo", deploy_model_label)
        st.metric("Última señal", str(deployment.get("last_signal") or "N/A"))
        st.metric("Equity", format_number(deployment.get("equity_quote"), 2, " USDT"))
        st.metric("Cash", format_number(deployment.get("cash_quote"), 2, " USDT"))
        st.metric("Posición", format_number(deployment.get("position_base"), 2, " DOGE"))
        st.caption(f"Run ID: `{selected_run_id}`")

    with right_col:
        fig = build_deployed_bot_chart(candles_df, features_df, run_df)
        st.plotly_chart(fig, width="stretch")

        if actions_df.empty or "side" not in actions_df.columns:
            st.caption("No hay entradas/salidas ejecutadas todavía para marcar sobre el precio.")
        else:
            visible_actions_df = actions_df[actions_df["side"].astype(str).str.upper().isin(["BUY", "SELL"])].copy()
            buy_count = int(visible_actions_df["side"].astype(str).str.upper().eq("BUY").sum())
            sell_count = int(visible_actions_df["side"].astype(str).str.upper().eq("SELL").sum())
            st.caption(f"Marcadores del run seleccionado: {buy_count} BUY / {sell_count} SELL. Si no aparecen en el gráfico, aumenta el número de velas visibles.")

    metric_cols = st.columns(6)
    with metric_cols[0]:
        st.metric("Eventos", f"{len(run_df):,}")
    with metric_cols[1]:
        st.metric("Señales", f"{len(signals_df):,}")
    with metric_cols[2]:
        executed_trades = actions_df[actions_df.get("side", pd.Series(dtype=str)).astype(str).isin(["BUY", "SELL"])] if not actions_df.empty else pd.DataFrame()
        st.metric("Trades reales", f"{len(executed_trades):,}")
    with metric_cols[3]:
        st.metric("Fees", format_number(actions_df["fee_quote"].sum() if "fee_quote" in actions_df.columns and not actions_df.empty else 0, 4, " USDT"))
    with metric_cols[4]:
        st.metric("Retorno", format_number(compute_dashboard_metrics(run_df, pd.DataFrame()).get("return_pct"), 4, "%"))
    with metric_cols[5]:
        st.metric("Último precio", format_number(deployment.get("last_price"), 6, " USDT"))

    tab_equity, tab_actions, tab_signals, tab_raw = st.tabs(["Equity bot", "Acciones bot", "Señales bot", "Logs bot raw"])

    with tab_equity:
        if equity_df.empty:
            st.info("Todavía no hay equity curve para este run.")
        else:
            chart_df = equity_df.dropna(subset=["time", "equity_quote_after"]).copy()
            if chart_df.empty:
                st.info("Hay eventos, pero todavía no hay valores de equity.")
            else:
                st.line_chart(chart_df.set_index("time")[["equity_quote_after"]])
            st.dataframe(equity_df.tail(50).sort_values("time", ascending=False), width="stretch", hide_index=True)

    with tab_actions:
        if actions_df.empty:
            st.info("Todavía no hay acciones paper registradas.")
        else:
            st.dataframe(actions_df.tail(100).sort_values("event_time", ascending=False), width="stretch", hide_index=True)

    with tab_signals:
        if signals_df.empty:
            st.info("Todavía no hay señales registradas.")
        else:
            st.dataframe(signals_df.tail(100).sort_values("event_time", ascending=False), width="stretch", hide_index=True)

    with tab_raw:
        st.dataframe(run_df.tail(250).sort_values("event_time_parsed", ascending=False), width="stretch", hide_index=True)

def render_logs_dashboard(log_path: str, summary_path: str, include_holds: bool) -> None:
    """Render the original logs-based dashboard section."""

    logs_df, summary_df = load_dashboard_data(log_path, summary_path)

    if logs_df.empty:
        st.warning("No se han encontrado logs de ejecución. Ejecuta primero el notebook 11 para generar `paper_trading.jsonl`.")
        st.code(f"Ruta esperada: {log_path}")
        return

    available_run_ids = get_available_run_ids(logs_df)
    run_options = ["ALL_RUNS"] + available_run_ids
    selected_run_id = st.selectbox(
        "Run seleccionado",
        options=run_options,
        index=1 if len(run_options) > 1 else 0,
        help="Usa el último run por defecto. ALL_RUNS muestra el histórico completo append-only.",
    )

    selected_logs_df = filter_by_run(logs_df, selected_run_id)
    if selected_logs_df.empty:
        st.warning("El run seleccionado no contiene eventos.")
        return

    selected_summary_df = summary_df.copy()
    if not summary_df.empty and selected_run_id != "ALL_RUNS" and "run_id" in summary_df.columns:
        selected_summary_df = summary_df[summary_df["run_id"].astype(str) == str(selected_run_id)].copy()
        if selected_summary_df.empty:
            selected_summary_df = summary_df.tail(1).copy()

    metrics = compute_dashboard_metrics(selected_logs_df, selected_summary_df)

    metric_cols = st.columns(5)
    with metric_cols[0]:
        render_metric("Equity final", format_number(metrics["final_equity"], 2, " USDT"))
    with metric_cols[1]:
        render_metric("Retorno", format_number(metrics["return_pct"], 4, "%"))
    with metric_cols[2]:
        render_metric("Trades", f"{metrics['trade_count']:,}")
    with metric_cols[3]:
        render_metric("Fees", format_number(metrics["total_fees"], 6, " USDT"))
    with metric_cols[4]:
        render_metric("Última señal", metrics["last_signal"] or "N/A")

    st.caption(f"Run mostrado: `{selected_run_id}` | Último evento: `{metrics['last_event_time']}`")

    equity_df = build_equity_curve(selected_logs_df)
    operations_df = build_operations(selected_logs_df, include_holds=include_holds)
    signals_df = build_signals(selected_logs_df)
    event_counts_df = build_event_counts(selected_logs_df)

    tab_equity, tab_operations, tab_signals, tab_metrics, tab_raw = st.tabs(
        ["Equity", "Operaciones", "Señales", "Métricas", "Logs raw"]
    )

    with tab_equity:
        st.subheader("Equity curve")
        if equity_df.empty:
            st.info("No hay eventos con `equity_quote_after`. Ejecuta una operación paper para construir la curva.")
        else:
            chart_df = equity_df.dropna(subset=["time", "equity_quote_after"]).copy()
            chart_df = chart_df.set_index("time")[["equity_quote_after"]]
            st.line_chart(chart_df)
            st.dataframe(equity_df.tail(20), width='stretch', hide_index=True)

    with tab_operations:
        st.subheader("Últimas operaciones")
        if operations_df.empty:
            st.info("No hay operaciones paper registradas para el run seleccionado.")
        else:
            st.dataframe(
                operations_df.tail(50).sort_values("event_time", ascending=False),
                width='stretch',
                hide_index=True,
            )

    with tab_signals:
        st.subheader("Últimas señales")
        if signals_df.empty:
            st.info("No hay señales registradas para el run seleccionado.")
        else:
            st.dataframe(
                signals_df.tail(50).sort_values("event_time", ascending=False),
                width='stretch',
                hide_index=True,
            )

    with tab_metrics:
        st.subheader("Eventos por tipo")
        if event_counts_df.empty:
            st.info("No hay eventos agregables.")
        else:
            st.bar_chart(event_counts_df.set_index("event_type")[["count"]])
            st.dataframe(event_counts_df, width='stretch', hide_index=True)

        st.subheader("Summary CSV")
        if selected_summary_df.empty:
            st.info("No se ha encontrado summary CSV para el run seleccionado.")
        else:
            st.dataframe(selected_summary_df, width='stretch', hide_index=True)

    with tab_raw:
        st.subheader("Eventos raw")
        st.dataframe(
            selected_logs_df.tail(200).sort_values("event_time", ascending=False),
            width='stretch',
            hide_index=True,
        )


def main() -> None:
    st.title("Cryptobot TFM - Live Dashboard")
    st.caption("Monitorización local con mercado en vivo, inferencia visual de modelos y logs paper/testnet.")

    with st.sidebar:
        st.header("Live market")
        market_source = st.radio("Fuente", options=["Spot", "Spot Testnet"], index=0, horizontal=True)
        base_url = SPOT_BASE_URL if market_source == "Spot" else TESTNET_BASE_URL
        symbol = st.selectbox("Símbolo", options=["DOGEUSDT", "DOGEUSDC"], index=0)
        interval = st.selectbox("Intervalo", options=["1m", "5m", "15m", "1h"], index=1)
        candle_limit = st.slider("Velas", min_value=288, max_value=1000, value=500, step=50, help="Mínimo 288 para soportes/resistencias de 24h; 500 recomendado para features live estables.")
        refresh_seconds = st.selectbox("Refresh", options=[5, 10, 30, 60], index=1)
        auto_refresh = st.checkbox("Auto refresh", value=False)

        st.header("Modelo")
        models_dir = st.text_input("Models dir", value=str(DEFAULT_MODELS_DIR))
        model_files = discover_model_files(models_dir)
        model_options = ["None"] + [str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path) for path in model_files]
        selected_model_label = st.selectbox("Modelo activo", options=model_options, index=0)
        buy_threshold = st.slider("BUY threshold", min_value=0.50, max_value=0.90, value=0.55, step=0.01)
        sell_threshold = st.slider("SELL threshold", min_value=0.10, max_value=0.50, value=0.45, step=0.01)

        st.header("Logs")
        log_path = st.text_input("Execution log JSONL", value=str(DEFAULT_LOG_PATH))
        summary_path = st.text_input("Summary CSV", value=str(DEFAULT_SUMMARY_PATH))
        include_holds = st.checkbox("Incluir HOLD en operaciones", value=True)
        if st.button("Refrescar ahora"):
            st.cache_data.clear()
            st.rerun()

    effective_market_source = market_source
    effective_base_url = base_url
    market_warning: str | None = None

    try:
        candles_df, symbol_status = load_live_candles(base_url, symbol, interval, candle_limit)
    except Exception as exc:
        if market_source == "Spot Testnet":
            market_warning = (
                "Spot Testnet no devolvió datos públicos de mercado en esta ejecución. "
                "Se usa Spot como fallback visual para mantener el dashboard operativo."
            )
            try:
                candles_df, symbol_status = load_live_candles(SPOT_BASE_URL, symbol, interval, candle_limit)
                effective_market_source = "Spot fallback"
                effective_base_url = SPOT_BASE_URL
                symbol_status = dict(symbol_status)
                symbol_status["source_warning"] = str(exc)
            except Exception as fallback_exc:
                st.error(f"No se pudieron cargar velas live para `{symbol}` ni desde Testnet ni desde Spot.")
                with st.expander("Detalle técnico"):
                    st.write("Error Testnet:")
                    st.exception(exc)
                    st.write("Error fallback Spot:")
                    st.exception(fallback_exc)
                return
        else:
            st.error(f"No se pudieron cargar velas live para `{symbol}` desde `{base_url}`.")
            with st.expander("Detalle técnico"):
                st.exception(exc)
            return

    if market_warning:
        st.warning(market_warning)
        with st.expander("Detalle técnico del fallback"):
            st.write(symbol_status.get("source_warning", "No diagnostic detail available."))

    try:
        server_time = load_server_time(effective_base_url)
    except Exception:
        server_time = None

    local_time = datetime.now(LOCAL_TIMEZONE)

    features_df = build_live_features(candles_df)
    latest_row = features_df.tail(1).copy()

    signal_payload: dict[str, object] = {
        "signal": "HOLD",
        "confidence": None,
        "prediction_raw": None,
        "reason": "no_model_selected",
    }
    model_diagnostics: dict[str, object] = {}

    if selected_model_label != "None":
        selected_model_path = PROJECT_ROOT / selected_model_label if not Path(selected_model_label).is_absolute() else Path(selected_model_label)
        try:
            model = load_cached_model(str(selected_model_path))
            metadata = read_model_metadata(selected_model_path)
            feature_names = infer_feature_names(model, metadata)
            model_input, missing_columns = build_model_input(latest_row, feature_names, model)
            if missing_columns:
                signal_payload = {
                    "signal": "HOLD",
                    "confidence": None,
                    "prediction_raw": None,
                    "reason": f"missing_columns={missing_columns[:8]}",
                }
            else:
                signal_payload = predict_signal(model, model_input, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
            model_diagnostics = {
                "model_path": str(selected_model_path),
                "metadata_found": bool(metadata),
                "feature_count": len(feature_names) if feature_names else getattr(model, "n_features_in_", "unknown"),
                "input_columns": list(model_input.columns) if not model_input.empty else [],
            }
        except Exception as exc:
            signal_payload = {"signal": "HOLD", "confidence": None, "prediction_raw": None, "reason": f"model_error={exc}"}

    latest_price = None
    latest_time = None
    if not candles_df.empty:
        latest_price = float(candles_df["close"].iloc[-1])
        latest_time = candles_df["open_time"].iloc[-1]

    clock_cols = st.columns([3, 1, 1])
    with clock_cols[1]:
        render_metric("Local time", format_clock_with_zone(local_time, LOCAL_TIMEZONE, "Madrid"))
    with clock_cols[2]:
        render_metric("Server time", format_clock_with_zone(server_time, timezone.utc, "UTC"))

    status_help = (
        "Estado de disponibilidad del par según Binance. Este dashboard ejecuta inferencia visual sobre datos live y lectura de logs locales; "
        "no envía órdenes reales desde esta pantalla. La ejecución paper/testnet se mantiene separada para preservar trazabilidad."
    )

    status_cols = st.columns(6)
    with status_cols[0]:
        render_metric("Símbolo", str(symbol_status.get("symbol", symbol)))
    with status_cols[1]:
        render_metric("Estado mercado", human_market_status(symbol_status.get("status", "N/A")), help_text=status_help)
    with status_cols[2]:
        render_metric("Precio", format_number(latest_price, 6, f" {symbol[-4:]}"))
    with status_cols[3]:
        render_metric("Última vela", format_clock_with_zone(latest_time, timezone.utc, "UTC"))
    with status_cols[4]:
        confidence = signal_payload.get("confidence")
        render_metric("Confianza", format_number(confidence, 4) if confidence is not None else "N/A")
    with status_cols[5]:
        render_metric(
            "Señal live",
            str(signal_payload.get("signal", "HOLD")),
            help_text="Señal calculada para visualización/inferencia. No implica ejecución automática de una orden real."
        )

    st.caption(
        f"Fuente solicitada: `{market_source}` | Fuente efectiva: `{effective_market_source}` | Base URL: `{effective_base_url}` | "
        f"Modelo visual dashboard: `{selected_model_label}` | Razón señal visual: `{signal_payload.get('reason')}`"
    )

    render_live_deployment_status(log_path)

    tab_live, tab_deploy, tab_logs, tab_features, tab_model = st.tabs(["Live chart", "Deployed Bot", "Logs paper/testnet", "Features live", "Modelo visual"])

    with tab_live:
        fig = build_candlestick_chart(candles_df, features_df, signal_payload)
        st.plotly_chart(fig, width='stretch')
        st.dataframe(
            candles_df.tail(20).sort_values("open_time", ascending=False),
            width='stretch',
            hide_index=True,
        )

    with tab_logs:
        render_logs_dashboard(log_path, summary_path, include_holds)

    with tab_features:
        st.subheader("Última fila de features live")
        if latest_row.empty:
            st.info("No hay suficientes velas para calcular features.")
        else:
            display_columns = [
                "open_time",
                "close",
                "return_prev_1",
                "sma_20",
                "ema_10",
                "ema_20",
                "ema_50",
                "ema_200",
                "rsi_14",
                "macd",
                "macd_signal",
                "bb_percent",
                "atr_14",
                "price_position_in_recent_range",
                "dist_to_nearest_support",
                "dist_to_nearest_resistance",
            ]
            display_columns = [column for column in display_columns if column in latest_row.columns]
            st.dataframe(latest_row[display_columns], width='stretch', hide_index=True)

    with tab_model:
        st.subheader("Modelo visual del dashboard")
        st.json(
            {
                "selected_model": selected_model_label,
                "signal_payload": signal_payload,
                "diagnostics": model_diagnostics,
                "note": "La inferencia de esta pestaña pertenece a Streamlit. No controla el proceso externo `run_live_paper_trader.py`.",
            }
        )
        if selected_model_label == "None":
            st.info("Selecciona un modelo serializado en `/models` para activar inferencia visual en el dashboard.")

    with tab_deploy:
        st.subheader("Deployed Bot")
        st.caption("Vista dedicada al proceso externo de paper trading: chart con entradas/salidas, policy editable y métricas del run desplegado.")
        render_deployed_bot_tab(
            log_path=log_path,
            policy_path=DEFAULT_POLICY_CONFIG_PATH,
            candles_df=candles_df,
            features_df=features_df,
        )

    if auto_refresh:
        time.sleep(int(refresh_seconds))
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
