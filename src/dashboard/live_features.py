"""Live feature engineering aligned with the offline TFM feature notebook.

This module intentionally mirrors `notebooks/02_feature_engineering.ipynb` for features that can be
computed from recent OHLCV candles without future information. Target columns such as
`future_return_*` and `up_*` are excluded because they are labels, not live inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COLUMNS = [
    "future_close_1",
    "future_return_1",
    "up_1",
    "future_close_3",
    "future_return_3",
    "up_3",
    "future_close_6",
    "future_return_6",
    "up_6",
    "future_close_12",
    "future_return_12",
    "up_12",
]

RAW_MARKET_COLUMNS = [
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

OFFLINE_ENGINEERED_FEATURE_COLUMNS = [
    "return_prev_1",
    "log_return_prev_1",
    "sma_20",
    "ema_10",
    "ema_50",
    "ema_200",
    "ema10_ema50_ratio",
    "ema50_ema200_ratio",
    "sma20_ema50_ratio",
    "volatility_1h",
    "zscore_close_1h",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_mid",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "bb_percent",
    "atr_14",
    "price_position_in_recent_range",
    "recent_support",
    "recent_resistance",
    "dist_to_nearest_support",
    "dist_to_nearest_resistance",
    "near_support",
    "near_resistance",
    "support_strength",
    "resistance_strength",
    "touch_count_near_level",
]

# Full non-target numeric feature universe available live. This preserves the raw OHLCV-style
# columns from Binance plus the engineered columns from the offline notebook.
LIVE_MODEL_FEATURE_COLUMNS = RAW_MARKET_COLUMNS + OFFLINE_ENGINEERED_FEATURE_COLUMNS

# Kept only for the optional EMA heuristic baseline used when no model is selected.
HEURISTIC_ONLY_COLUMNS = ["ema_20"]

MIN_RECOMMENDED_CANDLES = 500


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while converting zero denominators into missing values."""

    return numerator / denominator.replace(0, np.nan)


def _build_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Replicate the simple rolling RSI used in the feature engineering notebook."""

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _build_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Replicate ATR 14 from the offline notebook."""

    high_low = df["high"] - df["low"]
    high_close_prev = (df["high"] - df["close"].shift(1)).abs()
    low_close_prev = (df["low"] - df["close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()


def build_live_features(candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build live features using the same formulas as the offline feature engineering notebook."""

    df = candles_df.copy().sort_values("open_time").reset_index(drop=True)
    if df.empty:
        return df

    numeric_columns = [column for column in RAW_MARKET_COLUMNS if column in df.columns]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Historical returns (past information only)
    df["return_prev_1"] = df["close"].pct_change()
    df["log_return_prev_1"] = np.log(df["close"] / df["close"].shift(1))

    # Moving averages
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    # EMA ratios
    df["ema10_ema50_ratio"] = df["ema_10"] / df["ema_50"]
    df["ema50_ema200_ratio"] = df["ema_50"] / df["ema_200"]
    df["sma20_ema50_ratio"] = df["sma_20"] / df["ema_50"]

    # Rolling volatility: 1 hour = 12 candles of 5 minutes
    df["volatility_1h"] = df["log_return_prev_1"].rolling(window=12).std()

    # Z-score over 1 hour
    rolling_mean_1h = df["close"].rolling(window=12).mean()
    rolling_std_1h = df["close"].rolling(window=12).std()
    df["zscore_close_1h"] = _safe_divide(df["close"] - rolling_mean_1h, rolling_std_1h)

    # RSI 14
    df["rsi_14"] = _build_rsi(df["close"], window=14)

    # MACD: 12, 26, 9
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()

    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands: 20 periods, 2 std
    bb_window = 20
    bb_std = 2

    bb_mid = df["close"].rolling(window=bb_window).mean()
    bb_std_dev = df["close"].rolling(window=bb_window).std()

    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + bb_std * bb_std_dev
    df["bb_lower"] = bb_mid - bb_std * bb_std_dev
    df["bb_width"] = _safe_divide(df["bb_upper"] - df["bb_lower"], df["bb_mid"])
    df["bb_percent"] = _safe_divide(df["close"] - df["bb_lower"], df["bb_upper"] - df["bb_lower"])

    # ATR 14
    df["atr_14"] = _build_atr(df, window=14)

    # Support and resistance features
    range_window = 288  # 24 hours of 5m candles
    rolling_low = df["low"].rolling(window=range_window).min()
    rolling_high = df["high"].rolling(window=range_window).max()

    df["price_position_in_recent_range"] = _safe_divide(df["close"] - rolling_low, rolling_high - rolling_low)

    support_window = 288
    df["recent_support"] = df["low"].rolling(window=support_window).min()
    df["recent_resistance"] = df["high"].rolling(window=support_window).max()

    df["dist_to_nearest_support"] = (df["close"] - df["recent_support"]) / df["close"]
    df["dist_to_nearest_resistance"] = (df["recent_resistance"] - df["close"]) / df["close"]

    tolerance = 0.003  # 0.3%

    df["near_support"] = (((df["low"] - df["recent_support"]).abs() / df["close"] < tolerance)).astype(int)
    df["near_resistance"] = (((df["high"] - df["recent_resistance"]).abs() / df["close"] < tolerance)).astype(int)

    strength_window = 288

    df["support_strength"] = df["near_support"].rolling(window=strength_window).sum()
    df["resistance_strength"] = df["near_resistance"].rolling(window=strength_window).sum()

    df["touch_count_near_level"] = df["support_strength"] + df["resistance_strength"]

    return df


def feature_coverage(features_df: pd.DataFrame, required_columns: list[str] | None = None) -> dict[str, object]:
    """Return a compact diagnostic about feature availability in the latest live row."""

    if features_df.empty:
        return {
            "available": False,
            "row_count": 0,
            "required_count": 0,
            "missing_columns": [],
            "null_columns": [],
        }

    required = required_columns or LIVE_MODEL_FEATURE_COLUMNS
    latest = features_df.tail(1).copy()
    missing_columns = [column for column in required if column not in latest.columns]
    present_columns = [column for column in required if column in latest.columns]
    null_columns = [column for column in present_columns if pd.isna(latest.iloc[0][column])]

    return {
        "available": len(missing_columns) == 0 and len(null_columns) == 0,
        "row_count": int(len(features_df)),
        "required_count": int(len(required)),
        "missing_columns": missing_columns,
        "null_columns": null_columns,
    }


def latest_complete_feature_row(features_df: pd.DataFrame, required_columns: list[str] | None = None) -> pd.DataFrame:
    """Return the latest row where the required live model features are available."""

    if features_df.empty:
        return pd.DataFrame()

    required = required_columns or LIVE_MODEL_FEATURE_COLUMNS
    required_present = [column for column in required if column in features_df.columns]
    if not required_present:
        return pd.DataFrame()

    valid_rows = features_df.dropna(subset=required_present)
    if valid_rows.empty:
        return pd.DataFrame()

    return valid_rows.tail(1).copy()
