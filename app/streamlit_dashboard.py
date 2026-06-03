"""Streamlit dashboard for the Cryptobot TFM execution layer.

Run from the project root with:

    streamlit run app/streamlit_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
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

DEFAULT_LOG_PATH = PROJECT_ROOT / "results" / "execution_logs" / "paper_trading.jsonl"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "results" / "11_testnet_paper_trading_summary.csv"


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


def render_metric(label: str, value: str, delta: str | None = None) -> None:
    """Render a metric with a small wrapper for consistent missing values."""

    st.metric(label=label, value=value, delta=delta)


def main() -> None:
    st.title("Cryptobot TFM - Execution Dashboard")
    st.caption("Dashboard local para visualizar balance, operaciones, equity, señales y métricas de la capa paper/testnet.")

    with st.sidebar:
        st.header("Datos")
        log_path = st.text_input("Execution log JSONL", value=str(DEFAULT_LOG_PATH))
        summary_path = st.text_input("Summary CSV", value=str(DEFAULT_SUMMARY_PATH))
        include_holds = st.checkbox("Incluir HOLD en tabla de operaciones", value=True)

        if st.button("Refrescar datos"):
            st.cache_data.clear()

    logs_df, summary_df = load_dashboard_data(log_path, summary_path)

    if logs_df.empty:
        st.warning("No se han encontrado logs de ejecución. Ejecuta primero el notebook 11 para generar `paper_trading.jsonl`.")
        st.code(f"Ruta esperada: {log_path}")
        return

    available_run_ids = get_available_run_ids(logs_df)
    run_options = ["ALL_RUNS"] + available_run_ids

    with st.sidebar:
        selected_run_id = st.selectbox(
            "Run seleccionado",
            options=run_options,
            index=1 if len(run_options) > 1 else 0,
            help="Usa el último run por defecto. ALL_RUNS muestra el histórico completo append-only.",
        )
        st.caption(f"Eventos cargados: {len(logs_df):,}")
        st.caption(f"Runs detectados: {len(available_run_ids):,}")

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

    st.subheader("Estado general")
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

    balance_cols = st.columns(4)
    with balance_cols[0]:
        render_metric("Cash", format_number(metrics["latest_cash"], 2, " USDT"))
    with balance_cols[1]:
        render_metric("Posición DOGE", format_number(metrics["latest_position"], 6, " DOGE"))
    with balance_cols[2]:
        render_metric("Último precio", format_number(metrics["latest_price"], 6, " USDT"))
    with balance_cols[3]:
        render_metric("Eventos", f"{metrics['event_count']:,}")

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
            st.dataframe(
                equity_df.tail(20),
                use_container_width=True,
                hide_index=True,
            )

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
            operation_counts_df = pd.DataFrame(
                {
                    "side": ["BUY", "SELL", "HOLD"],
                    "count": [metrics["buy_count"], metrics["sell_count"], metrics["hold_count"]],
                }
            ).set_index("side")
            st.bar_chart(operation_counts_df)

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


if __name__ == "__main__":
    main()
