"""Streamlit dashboard for the Cryptobot TFM execution and live market layer.

Run from the project root with:

    python -m streamlit run app/streamlit_dashboard.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

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
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
SPOT_BASE_URL = "https://api.binance.com"
TESTNET_BASE_URL = "https://testnet.binance.vision"


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


@st.cache_resource(show_spinner=False)
def load_cached_model(model_path: str):
    """Load a serialized model once per path."""

    return load_model(model_path)


def render_metric(label: str, value: str, delta: str | None = None) -> None:
    """Render a metric with a small wrapper for consistent missing values."""

    st.metric(label=label, value=value, delta=delta)


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
            st.dataframe(equity_df.tail(20), use_container_width=True, hide_index=True)

    with tab_operations:
        st.subheader("Últimas operaciones")
        if operations_df.empty:
            st.info("No hay operaciones paper registradas para el run seleccionado.")
        else:
            st.dataframe(
                operations_df.tail(50).sort_values("event_time", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    with tab_signals:
        st.subheader("Últimas señales")
        if signals_df.empty:
            st.info("No hay señales registradas para el run seleccionado.")
        else:
            st.dataframe(
                signals_df.tail(50).sort_values("event_time", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    with tab_metrics:
        st.subheader("Eventos por tipo")
        if event_counts_df.empty:
            st.info("No hay eventos agregables.")
        else:
            st.bar_chart(event_counts_df.set_index("event_type")[["count"]])
            st.dataframe(event_counts_df, use_container_width=True, hide_index=True)

        st.subheader("Summary CSV")
        if selected_summary_df.empty:
            st.info("No se ha encontrado summary CSV para el run seleccionado.")
        else:
            st.dataframe(selected_summary_df, use_container_width=True, hide_index=True)

    with tab_raw:
        st.subheader("Eventos raw")
        st.dataframe(
            selected_logs_df.tail(200).sort_values("event_time", ascending=False),
            use_container_width=True,
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
        candle_limit = st.slider("Velas", min_value=50, max_value=500, value=200, step=50)
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

    status_cols = st.columns(6)
    with status_cols[0]:
        render_metric("Símbolo", str(symbol_status.get("symbol", symbol)))
    with status_cols[1]:
        render_metric("Estado", str(symbol_status.get("status", "N/A")))
    with status_cols[2]:
        render_metric("Precio", format_number(latest_price, 6, f" {symbol[-4:]}"))
    with status_cols[3]:
        render_metric("Última vela", str(latest_time) if latest_time is not None else "N/A")
    with status_cols[4]:
        confidence = signal_payload.get("confidence")
        render_metric("Confianza", format_number(confidence, 4) if confidence is not None else "N/A")
    with status_cols[5]:
        render_metric("Señal live", str(signal_payload.get("signal", "HOLD")))

    st.caption(
        f"Fuente solicitada: `{market_source}` | Fuente efectiva: `{effective_market_source}` | Base URL: `{effective_base_url}` | Modelo: `{selected_model_label}` | Razón señal: `{signal_payload.get('reason')}`"
    )

    tab_live, tab_logs, tab_features, tab_model = st.tabs(["Live chart", "Logs paper/testnet", "Features live", "Modelo"])

    with tab_live:
        fig = build_candlestick_chart(candles_df, features_df, signal_payload)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            candles_df.tail(20).sort_values("open_time", ascending=False),
            use_container_width=True,
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
            st.dataframe(latest_row[display_columns], use_container_width=True, hide_index=True)

    with tab_model:
        st.subheader("Modelo activo")
        st.json(
            {
                "selected_model": selected_model_label,
                "signal_payload": signal_payload,
                "diagnostics": model_diagnostics,
                "note": "La inferencia live es visual. La ejecución paper automática queda como siguiente capa para mantener trazabilidad.",
            }
        )
        if selected_model_label == "None":
            st.info("Selecciona un modelo serializado en `/models` para activar inferencia visual.")

    if auto_refresh:
        time.sleep(int(refresh_seconds))
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
