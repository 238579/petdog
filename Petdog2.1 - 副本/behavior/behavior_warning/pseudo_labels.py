from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import PseudoLabelConfig


LABEL_TO_ID = {
    "normal": 0,
    "long_static": 1,
    "activity_drop": 2,
    "frequent_walking": 3,
}


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if std is None or np.isnan(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def _groupwise_zscore(
    df: pd.DataFrame,
    source_column: str,
    use_per_video: bool,
    min_windows_for_video_zscore: int,
) -> pd.Series:
    if not use_per_video or "video_name" not in df.columns:
        return _zscore(df[source_column])

    global_z = _zscore(df[source_column])
    group_sizes = df.groupby("video_name", sort=False)["video_name"].transform("size")
    local_z = (
        df.groupby("video_name", sort=False)[source_column]
        .transform(_zscore)
        .fillna(0.0)
    )
    use_local = group_sizes >= max(min_windows_for_video_zscore, 2)
    return local_z.where(use_local, global_z)


def _stabilize_labels(labels: pd.Series, minimum_run: int) -> pd.Series:
    stabilized = labels.copy()
    if minimum_run <= 1:
        return stabilized

    current = stabilized.iloc[0]
    run_start = 0
    for idx in range(1, len(stabilized) + 1):
        boundary = idx == len(stabilized) or stabilized.iloc[idx] != current
        if boundary:
            run_length = idx - run_start
            if current != "normal" and run_length < minimum_run:
                stabilized.iloc[run_start:idx] = "normal"
            if idx < len(stabilized):
                current = stabilized.iloc[idx]
                run_start = idx
    return stabilized


def assign_pseudo_labels(feature_csv: Path, config: PseudoLabelConfig, output_dir: Path) -> Path:
    df = pd.read_csv(feature_csv)
    if df.empty:
        raise RuntimeError("Feature CSV is empty.")

    metric_z: Dict[str, pd.Series] = {
        "stationary_ratio_z": _groupwise_zscore(
            df,
            "stationary_ratio",
            config.use_per_video_zscore,
            config.min_windows_for_video_zscore,
        ),
        "mean_speed_z": _groupwise_zscore(
            df,
            "mean_speed",
            config.use_per_video_zscore,
            config.min_windows_for_video_zscore,
        ),
        "total_distance_z": _groupwise_zscore(
            df,
            "total_distance",
            config.use_per_video_zscore,
            config.min_windows_for_video_zscore,
        ),
        "direction_change_rate_z": _groupwise_zscore(
            df,
            "direction_change_rate",
            config.use_per_video_zscore,
            config.min_windows_for_video_zscore,
        ),
        "x_range_z": _groupwise_zscore(
            df,
            "x_range",
            config.use_per_video_zscore,
            config.min_windows_for_video_zscore,
        ),
        "y_range_z": _groupwise_zscore(
            df,
            "y_range",
            config.use_per_video_zscore,
            config.min_windows_for_video_zscore,
        ),
        "active_ratio_z": _groupwise_zscore(
            df,
            "active_ratio",
            config.use_per_video_zscore,
            config.min_windows_for_video_zscore,
        ),
        "path_efficiency_z": _groupwise_zscore(
            df,
            "path_efficiency",
            config.use_per_video_zscore,
            config.min_windows_for_video_zscore,
        ),
    }

    for column, values in metric_z.items():
        df[column] = values

    labels = np.full(len(df), "normal", dtype=object)
    rules = np.full(len(df), "", dtype=object)

    long_static_mask = (
        (df["stationary_ratio_z"] >= config.long_static_z)
        & (df["active_ratio_z"] <= -0.8)
        & (df["total_distance_z"] <= -0.8)
    )
    labels[long_static_mask] = "long_static"
    rules[long_static_mask] = "high_stationary_low_distance"

    activity_drop_mask = (
        (labels == "normal")
        & (df["active_ratio_z"] <= -(config.activity_drop_z * 0.75))
        & (df["total_distance_z"] <= -(config.activity_drop_z * 0.25))
        & (df["stationary_ratio_z"] >= -0.2)
        & (df["stationary_ratio_z"] < (config.long_static_z + 0.3))
    )
    labels[activity_drop_mask] = "activity_drop"
    rules[activity_drop_mask] = "low_active_ratio_low_distance"

    frequent_walking_mask = (
        (labels == "normal")
        & (df["mean_speed_z"] >= config.frequent_walking_speed_z)
        & (df["total_distance_z"] >= config.frequent_walking_distance_z)
        & (df["active_ratio_z"] >= 0.5)
    )
    labels[frequent_walking_mask] = "frequent_walking"
    rules[frequent_walking_mask] = "high_speed_high_distance"

    df["label"] = labels
    df["pseudo_rule"] = rules

    if "video_name" in df.columns:
        stabilized_parts = []
        for _, group in df.groupby("video_name", sort=False):
            group = group.copy()
            group["label"] = _stabilize_labels(group["label"], config.min_positive_duration_windows)
            group.loc[group["label"] == "normal", "pseudo_rule"] = ""
            group.loc[
                (group["label"] != "normal") & (group["pseudo_rule"] == ""),
                "pseudo_rule",
            ] = "stabilized_positive_run"
            stabilized_parts.append(group)
        df = pd.concat(stabilized_parts, ignore_index=True)

    df["label_id"] = df["label"].map(LABEL_TO_ID)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "behavior_labeled_windows.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def summarize_labels(label_csv: Path) -> Dict[str, Dict[str, int]]:
    df = pd.read_csv(label_csv)
    if df.empty:
        raise RuntimeError("Label CSV is empty.")

    label_counts = df["label"].value_counts().to_dict() if "label" in df.columns else {}
    rule_series = df["pseudo_rule"] if "pseudo_rule" in df.columns else pd.Series(dtype=object)
    rule_counts = rule_series.replace("", "normal").value_counts().to_dict()
    return {
        "rows": {"total": int(len(df))},
        "labels": {str(key): int(value) for key, value in label_counts.items()},
        "rules": {str(key): int(value) for key, value in rule_counts.items()},
    }
