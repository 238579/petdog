from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import cv2
import numpy as np
import pandas as pd

from .config import FeatureConfig


def discover_videos(dataset_dir: Path) -> List[Path]:
    return sorted(
        [path for path in dataset_dir.rglob("*") if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
    )


def _safe_float(value: float) -> float:
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return float(value)


def _video_timestamp(video_path: Path) -> Optional[pd.Timestamp]:
    digits = "".join(ch for ch in video_path.stem if ch.isdigit())
    if len(digits) >= 13:
        try:
            return pd.to_datetime(int(digits[:13]), unit="ms")
        except Exception:
            return None
    return None


def _extract_frame_stats(
    frame: np.ndarray,
    subtractor: cv2.BackgroundSubtractor,
    config: FeatureConfig,
) -> Dict[str, float]:
    resized = cv2.resize(frame, (config.resize_width, config.resize_height))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, config.gaussian_blur, 0)
    mask = subtractor.apply(blurred)

    kernel = np.ones(config.morph_kernel, dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    total_pixels = mask.shape[0] * mask.shape[1]
    min_area = total_pixels * config.min_area_ratio
    max_area = total_pixels * config.max_area_ratio

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]
    motion_pixels = cv2.countNonZero(mask)
    motion_ratio = motion_pixels / max(total_pixels, 1)

    if not valid:
        return {
            "motion_ratio": motion_ratio,
            "cx": np.nan,
            "cy": np.nan,
            "bbox_area_ratio": 0.0,
            "bbox_w_ratio": 0.0,
            "bbox_h_ratio": 0.0,
            "contour_count": 0.0,
        }

    contour = max(valid, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx = x + w / 2.0
        cy = y + h / 2.0

    return {
        "motion_ratio": motion_ratio,
        "cx": cx / mask.shape[1],
        "cy": cy / mask.shape[0],
        "bbox_area_ratio": (w * h) / max(total_pixels, 1),
        "bbox_w_ratio": w / mask.shape[1],
        "bbox_h_ratio": h / mask.shape[0],
        "contour_count": float(len(valid)),
    }


def _window_features(frame_df: pd.DataFrame, config: FeatureConfig, video_name: str) -> List[Dict[str, float]]:
    if frame_df.empty:
        return []

    frames_per_window = max(int(config.window_seconds * config.sample_fps), 1)
    stride_frames = max(int(config.stride_seconds * config.sample_fps), 1)
    rows: List[Dict[str, float]] = []

    for window_index, start_idx in enumerate(range(0, max(len(frame_df) - frames_per_window + 1, 1), stride_frames)):
        window = frame_df.iloc[start_idx : start_idx + frames_per_window].copy()
        if len(window) < max(frames_per_window // 2, 1):
            continue

        centered = window[["cx", "cy"]].ffill().bfill()
        dx = centered["cx"].diff().fillna(0.0)
        dy = centered["cy"].diff().fillna(0.0)
        speed = np.sqrt(dx**2 + dy**2)
        heading = np.arctan2(dy, dx)
        turn_rate = np.abs(np.diff(np.unwrap(heading.to_numpy()))) if len(window) > 2 else np.array([0.0])
        moving_mask = speed > max(speed.median(), 0.002)

        feature_row = {
            "video_name": video_name,
            "window_index": window_index,
            "window_start_sec": _safe_float(window["timestamp_sec"].iloc[0]),
            "window_end_sec": _safe_float(window["timestamp_sec"].iloc[-1]),
            "mean_motion_ratio": _safe_float(window["motion_ratio"].mean()),
            "std_motion_ratio": _safe_float(window["motion_ratio"].std()),
            "mean_speed": _safe_float(speed.mean()),
            "max_speed": _safe_float(speed.max()),
            "total_distance": _safe_float(speed.sum()),
            "stationary_ratio": _safe_float((~moving_mask).mean()),
            "active_ratio": _safe_float(moving_mask.mean()),
            "mean_bbox_area_ratio": _safe_float(window["bbox_area_ratio"].mean()),
            "bbox_area_std": _safe_float(window["bbox_area_ratio"].std()),
            "x_range": _safe_float(centered["cx"].max() - centered["cx"].min()),
            "y_range": _safe_float(centered["cy"].max() - centered["cy"].min()),
            "path_efficiency": _safe_float(
                np.sqrt(
                    (centered["cx"].iloc[-1] - centered["cx"].iloc[0]) ** 2
                    + (centered["cy"].iloc[-1] - centered["cy"].iloc[0]) ** 2
                )
                / max(speed.sum(), 1e-6)
            ),
            "direction_change_rate": _safe_float(turn_rate.mean()),
            "high_turn_ratio": _safe_float((turn_rate > 0.8).mean()) if len(turn_rate) else 0.0,
            "motion_peaks": float(((window["motion_ratio"] > window["motion_ratio"].quantile(0.75))).sum()),
            "missing_track_ratio": _safe_float(window["cx"].isna().mean()),
            "contour_count_mean": _safe_float(window["contour_count"].mean()),
        }
        rows.append(feature_row)

    return rows


def extract_video_features(video_path: Path, config: FeatureConfig) -> pd.DataFrame:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(fps / config.sample_fps)), 1)
    subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

    frame_rows: List[Dict[str, float]] = []
    frame_index = 0
    sampled_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % step == 0:
            stats = _extract_frame_stats(frame, subtractor, config)
            stats["frame_index"] = sampled_index
            stats["timestamp_sec"] = frame_index / max(fps, 1e-6)
            frame_rows.append(stats)
            sampled_index += 1
        frame_index += 1

    capture.release()

    frame_df = pd.DataFrame(frame_rows)
    if frame_df.empty:
        return pd.DataFrame()

    features_df = pd.DataFrame(_window_features(frame_df, config, video_path.name))
    if features_df.empty:
        return features_df

    timestamp = _video_timestamp(video_path)
    if timestamp is not None:
        features_df["recorded_at"] = timestamp
        features_df["hour_of_day"] = timestamp.hour
        features_df["is_daytime"] = int(config.daytime_start_hour <= timestamp.hour < config.daytime_end_hour)
    else:
        features_df["recorded_at"] = pd.NaT
        features_df["hour_of_day"] = -1
        features_df["is_daytime"] = -1

    return features_df


def build_feature_dataset(config: FeatureConfig) -> Path:
    dataset_dir = Path(config.dataset_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for video_path in discover_videos(dataset_dir):
        video_df = extract_video_features(video_path, config)
        if not video_df.empty:
            rows.append(video_df)

    if not rows:
        raise RuntimeError(f"No usable videos found under {dataset_dir}")

    features_df = pd.concat(rows, ignore_index=True)
    output_path = output_dir / "behavior_features.csv"
    features_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    config_path = output_dir / "feature_config.json"
    config_payload = {}
    for key, value in asdict(config).items():
        if isinstance(value, Path):
            config_payload[key] = str(value)
        elif isinstance(value, tuple):
            config_payload[key] = list(value)
        else:
            config_payload[key] = value
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
