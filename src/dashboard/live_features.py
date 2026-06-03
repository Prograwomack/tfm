"""Feature engineering helpers for live dashboard inference.

The goal is not to replace the offline feature engineering notebook. This module builds a compact and reproducible live feature set from recent OHLCV candles so saved models can be tested visually in Streamlit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=window, min_periods=window).mean()


def build_live_features(candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the live feature dataframe used by the dashboard and model inference."""

    df = candles_df.copy().sort_values("open_time").reset_index(drop=True)
    if df.empty:
        return df

    df["return_prev_1"] = df["close"].pct_change()
    df["log_return_prev_1"] = np.log(df["close"] / df["close"].shift(1))

    df["sma_20"] = df["close"].rolling(window=20, min_periods=20).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    df["ema10_ema50_ratio"] = df["ema_10"] / df["ema_50"]
    df["ema50_ema200_ratio"] = df["ema_50"] / df["ema_200"]
    df["sma20_ema50_ratio"] = df["sma_20"] / df["ema_50"]

    df["volatility_1h"] = df["return_prev_1"].rolling(window=12, min_periods=12).std()
    rolling_mean_1h = df["close"].rolling(window=12, min_periods=12).mean()
    rolling_std_1h = df["close"].rolling(window=12, min_periods=12).std()
    df["zscore_close_1h"] = (df["close"] - rolling_mean_1h) / rolling_std_1h.replace(0, np.nan)

    df["rsi_14"] = _rsi(df["close"], window=14)

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["bb_mid"] = df["close"].rolling(window=20, min_periods=20).mean()
    bb_std = df["close"].rolling(window=20, min_periods=20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_percent"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    df["atr_14"] = _atr(df, window=14)

    recent_low = df["low"].rolling(window=96, min_periods=20).min()
    recent_high = df["high"].rolling(window=96, min_periods=20).max()
    range_width = (recent_high - recent_low).replace(0, np.nan)
    df["price_position_in_recent_range"] = (df["close"] - recent_low) / range_width
    df["recent_support"] = recent_low
    df["recent_resistance"] = recent_high
    df["dist_to_nearest_support"] = (df["close"] - df["recent_support"]) / df["close"]
    df["dist_to_nearest_resistance"] = (df["recent_resistance"] - df["close"]) / df["close"]
    df["near_support"] = (df["dist_to_nearest_support"] <= 0.01).astype(int)
    df["near_resistance"] = (df["dist_to_nearest_resistance"] <= 0.01).astype(int)
    df["support_strength"] = df["near_support"].rolling(window=96, min_periods=20).sum()
    df["resistance_strength"] = df["near_resistance"].rolling(window=96, min_periods=20).sum()
    df["touch_count_near_level"] = df["support_strength"].fillna(0) + df["resistance_strength"].fillna(0)

    return df


def latest_complete_feature_row(features_df: pd.DataFrame) -> pd.DataFrame:
    """Return the last row with enough non-null engineered features for inference."""

    if features_df.empty:
        return pd.DataFrame()
    numeric_df = features_df.select_dtypes(include=["number"]).copy()
    if numeric_df.empty:
        return pd.DataFrame()
    valid_rows = numeric_df.dropna(how="all")
    if valid_rows.empty:
        return pd.DataFrame()
    return features_df.loc[[valid_rows.index[-1]]].copy()
