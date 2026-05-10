from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class FeatureConfig:
    dataset_dir: Path = Path("Activity Analysis")
    output_dir: Path = Path("outputs_tuned")
    resize_width: int = 640
    resize_height: int = 360
    sample_fps: float = 2.0
    window_seconds: int = 30
    stride_seconds: int = 10
    min_area_ratio: float = 0.002
    max_area_ratio: float = 0.45
    gaussian_blur: Tuple[int, int] = (5, 5)
    morph_kernel: Tuple[int, int] = (5, 5)
    daytime_start_hour: int = 7
    daytime_end_hour: int = 21


@dataclass
class PseudoLabelConfig:
    use_per_video_zscore: bool = True
    min_windows_for_video_zscore: int = 3
    long_static_z: float = 1.2
    activity_drop_z: float = 1.0
    frequent_walking_speed_z: float = 1.0
    frequent_walking_distance_z: float = 1.0
    night_active_z: float = 1.0
    min_positive_duration_windows: int = 2


@dataclass
class TrainConfig:
    group_column: str = "video_name"
    label_column: str = "label"
    feature_drop_columns: List[str] = field(
        default_factory=lambda: [
            "video_name",
            "window_index",
            "window_start_sec",
            "window_end_sec",
            "pseudo_rule",
            "label",
        ]
    )
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    n_estimators: int = 300
    max_depth: Optional[int] = 10
    min_samples_leaf: int = 2
