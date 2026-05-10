from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import FeatureConfig
from .features import extract_video_features


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if std is None or np.isnan(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def _ensure_inference_features(features: pd.DataFrame) -> pd.DataFrame:
    enriched = features.copy()
    base_columns = [
        "stationary_ratio",
        "mean_speed",
        "total_distance",
        "direction_change_rate",
        "x_range",
        "y_range",
        "active_ratio",
        "path_efficiency",
    ]
    for column in base_columns:
        z_column = f"{column}_z"
        if column in enriched.columns and z_column not in enriched.columns:
            enriched[z_column] = _zscore(enriched[column])
    return enriched


def infer_video(video_path: Path, model_path: Path, output_dir: Path, config: FeatureConfig) -> Path:
    features = extract_video_features(video_path, config)
    if features.empty:
        raise RuntimeError(f"No usable windows generated from {video_path}")

    features = _ensure_inference_features(features)
    model = joblib.load(model_path)
    x = features.drop(columns=["video_name", "window_index", "recorded_at"], errors="ignore")
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)

    prediction_df = features.copy()
    prediction_df["prediction"] = predictions

    classes = list(model.named_steps["model"].classes_)
    for idx, class_name in enumerate(classes):
        prediction_df[f"score_{class_name}"] = probabilities[:, idx]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}_warning_predictions.csv"
    prediction_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path
