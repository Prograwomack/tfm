"""Model discovery, loading and inference helpers for the live dashboard and paper trader.

This module is deliberately tolerant with serialized model formats. Some notebooks save the
estimator directly with joblib, while others save a dictionary containing the estimator,
feature list, metrics and metadata. The live dashboard/paper trader needs the estimator object,
but it also benefits from the embedded feature metadata when available.
"""

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

PREDICTOR_KEYS = [
    "model",
    "estimator",
    "pipeline",
    "clf",
    "classifier",
    "best_model",
    "best_estimator",
    "trained_model",
    "xgb_model",
    "random_forest_model",
    "logistic_regression_model",
]
FEATURE_KEYS = [
    "feature_names",
    "features",
    "feature_columns",
    "selected_features",
    "X_columns",
    "x_columns",
    "columns",
]


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


def _extract_metadata_from_payload(payload: Any) -> dict[str, Any]:
    """Extract lightweight metadata from a serialized payload when it is a dictionary."""

    metadata: dict[str, Any] = {}

    if not isinstance(payload, dict):
        return metadata

    metadata["serialized_payload_keys"] = list(payload.keys())

    for key in FEATURE_KEYS:
        value = payload.get(key)
        if isinstance(value, list) and value:
            metadata["feature_names"] = [str(feature) for feature in value]
            break

    for key in ["target", "target_column", "horizon", "threshold", "model_name", "metrics", "classification_report"]:
        if key in payload:
            metadata[key] = payload[key]

    nested_metadata = payload.get("metadata")
    if isinstance(nested_metadata, dict):
        metadata.update(nested_metadata)

    return metadata


def _find_predictor(payload: Any) -> Any | None:
    """Find the first object with a predict method inside common joblib payload formats."""

    if hasattr(payload, "predict"):
        return payload

    if isinstance(payload, dict):
        for key in PREDICTOR_KEYS:
            if key in payload:
                candidate = _find_predictor(payload[key])
                if candidate is not None:
                    return candidate

        for value in payload.values():
            candidate = _find_predictor(value)
            if candidate is not None:
                return candidate

    if isinstance(payload, (list, tuple)):
        for value in payload:
            candidate = _find_predictor(value)
            if candidate is not None:
                return candidate

    return None


def _attach_embedded_metadata(model: Any, metadata: dict[str, Any], model_path: str | Path) -> None:
    """Attach embedded metadata to the loaded estimator when the object allows dynamic attrs."""

    try:
        setattr(model, "_cryptobot_embedded_metadata", metadata)
        setattr(model, "_cryptobot_model_source_path", str(model_path))
    except Exception:
        pass


def load_model(model_path: str | Path) -> Any:
    """Load a serialized model with joblib and unwrap dictionary payloads when needed."""

    payload = joblib.load(model_path)
    model = _find_predictor(payload)

    if model is None:
        if isinstance(payload, dict):
            payload_description = f"dict keys={list(payload.keys())}"
        else:
            payload_description = type(payload).__name__
        raise TypeError(f"No object with a predict method was found in {model_path}. Loaded payload: {payload_description}")

    embedded_metadata = _extract_metadata_from_payload(payload)
    _attach_embedded_metadata(model, embedded_metadata, model_path)
    return model


def infer_feature_names(model: Any, metadata: dict[str, Any] | None = None) -> list[str]:
    """Infer feature names from sidecar metadata, embedded payload metadata or model attributes."""

    merged_metadata: dict[str, Any] = {}
    embedded_metadata = getattr(model, "_cryptobot_embedded_metadata", {})
    if isinstance(embedded_metadata, dict):
        merged_metadata.update(embedded_metadata)
    if metadata:
        merged_metadata.update(metadata)

    for key in FEATURE_KEYS:
        value = merged_metadata.get(key)
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
    elif hasattr(model, "predict"):
        prediction = model.predict(model_input)[0]
        prediction_raw = float(prediction) if isinstance(prediction, (int, float, bool)) else str(prediction)
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
