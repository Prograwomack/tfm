"""Model discovery and metadata helpers for the live dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


MODEL_EXTENSIONS = {".joblib", ".pkl", ".pickle"}
TARGET_COLUMNS = {
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
}
NON_FEATURE_COLUMNS = {
    "open_time",
    "close_time",
    "timestamp",
    "datetime",
    "date",
    "symbol",
    "ignore",
} | TARGET_COLUMNS


def discover_model_files(models_dir: str | Path) -> list[Path]:
    """Return available serialized model files ordered by name."""

    path = Path(models_dir)
    if not path.exists():
        return []
    return sorted(file for file in path.rglob("*") if file.suffix.lower() in MODEL_EXTENSIONS)


def read_model_metadata(model_path: str | Path) -> dict[str, Any]:
    """Read optional sidecar metadata next to a model file."""

    path = Path(model_path)
    candidates = [
        path.with_suffix(".json"),
        path.parent / f"{path.stem}_metadata.json",
        path.parent / f"{path.stem}_features.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as file:
                return json.load(file)
    return {}


def load_model(model_path: str | Path) -> Any:
    """Load a serialized model with joblib."""

    return joblib.load(model_path)


def infer_feature_names(model: Any, metadata: dict[str, Any] | None = None) -> list[str]:
    """Infer feature names from metadata or sklearn attributes when available."""

    metadata = metadata or {}
    for key in ["feature_names", "features", "feature_columns", "selected_features"]:
        value = metadata.get(key)
        if isinstance(value, list) and value:
            return [str(feature) for feature in value]

    if hasattr(model, "feature_names_in_"):
        return [str(feature) for feature in model.feature_names_in_]

    if hasattr(model, "get_booster"):
        try:
            booster_feature_names = model.get_booster().feature_names
            if booster_feature_names:
                return [str(feature) for feature in booster_feature_names]
        except Exception:
            pass

    return []


def fallback_feature_candidates(features_df: pd.DataFrame) -> list[str]:
    """Build a conservative feature list when a model does not expose feature names."""

    numeric_columns = features_df.select_dtypes(include=["number"]).columns.tolist()
    return [column for column in numeric_columns if column not in NON_FEATURE_COLUMNS]


def build_model_input(features_df: pd.DataFrame, feature_names: list[str], model: Any) -> tuple[pd.DataFrame, list[str]]:
    """Build a single-row model input, returning missing columns for diagnostics."""

    if features_df.empty:
        return pd.DataFrame(), feature_names

    latest_row = features_df.tail(1).copy()

    if not feature_names:
        feature_names = fallback_feature_candidates(features_df)
        expected_n_features = getattr(model, "n_features_in_", None)
        if expected_n_features is not None:
            feature_names = feature_names[: int(expected_n_features)]

    missing_columns = [feature for feature in feature_names if feature not in latest_row.columns]
    if missing_columns:
        return pd.DataFrame(), missing_columns

    input_df = latest_row[feature_names].copy()
    input_df = input_df.replace([float("inf"), float("-inf")], pd.NA)
    input_df = input_df.ffill(axis=1).fillna(0)
    return input_df, []


def predict_signal(model: Any, model_input: pd.DataFrame, buy_threshold: float = 0.55, sell_threshold: float = 0.45) -> dict[str, Any]:
    """Convert a model prediction into BUY, SELL or HOLD for dashboard display."""

    if model_input.empty:
        return {"signal": "HOLD", "confidence": None, "prediction_raw": None, "reason": "empty_model_input"}

    probability_up = None
    prediction_raw = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)
        if probabilities.shape[1] >= 2:
            probability_up = float(probabilities[0, 1])
            prediction_raw = probability_up
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(model_input)[0])
        probability_up = 1 / (1 + pow(2.718281828, -score))
        prediction_raw = score
    else:
        prediction = model.predict(model_input)[0]
        prediction_raw = float(prediction) if isinstance(prediction, (int, float)) else str(prediction)
        if str(prediction) in {"1", "BUY", "UP", "True", "true"}:
            probability_up = 1.0
        elif str(prediction) in {"0", "SELL", "DOWN", "False", "false"}:
            probability_up = 0.0

    if probability_up is None:
        return {"signal": "HOLD", "confidence": None, "prediction_raw": prediction_raw, "reason": "unsupported_prediction_output"}

    if probability_up >= buy_threshold:
        signal = "BUY"
    elif probability_up <= sell_threshold:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "signal": signal,
        "confidence": probability_up,
        "prediction_raw": prediction_raw,
        "reason": f"probability_up={probability_up:.4f}",
    }
