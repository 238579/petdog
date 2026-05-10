from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import FeatureConfig, PseudoLabelConfig, TrainConfig
from .features import build_feature_dataset
from .infer import infer_video
from .pseudo_labels import assign_pseudo_labels, summarize_labels
from .train import train_behavior_model


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dog behavior warning pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    feature_parser = subparsers.add_parser("extract", help="Extract windowed activity features from videos")
    feature_parser.add_argument("--dataset-dir", default="Activity Analysis")
    feature_parser.add_argument("--output-dir", default="outputs_tuned")
    feature_parser.add_argument("--sample-fps", type=float, default=2.0)
    feature_parser.add_argument("--window-seconds", type=int, default=30)
    feature_parser.add_argument("--stride-seconds", type=int, default=10)

    label_parser = subparsers.add_parser("label", help="Generate pseudo labels")
    label_parser.add_argument("--feature-csv", default="outputs_tuned/behavior_features.csv")
    label_parser.add_argument("--output-dir", default="outputs_tuned")
    label_parser.add_argument("--global-zscore", action="store_true", help="Use global z-score instead of per-video z-score")
    label_parser.add_argument("--min-video-zscore-windows", type=int, default=3)
    label_parser.add_argument("--long-static-z", type=float, default=1.2)
    label_parser.add_argument("--activity-drop-z", type=float, default=1.0)
    label_parser.add_argument("--frequent-walking-speed-z", type=float, default=1.0)
    label_parser.add_argument("--frequent-walking-distance-z", type=float, default=1.0)
    label_parser.add_argument("--min-positive-windows", type=int, default=2)

    train_parser = subparsers.add_parser("train", help="Train warning model from pseudo labels")
    train_parser.add_argument("--label-csv", default="outputs_tuned/behavior_labeled_windows.csv")
    train_parser.add_argument("--output-dir", default="outputs_tuned")
    train_parser.add_argument("--cv-folds", type=int, default=5)

    full_parser = subparsers.add_parser("pipeline", help="Run extract + label + train")
    full_parser.add_argument("--dataset-dir", default="Activity Analysis")
    full_parser.add_argument("--output-dir", default="outputs_tuned")
    full_parser.add_argument("--sample-fps", type=float, default=2.0)
    full_parser.add_argument("--window-seconds", type=int, default=30)
    full_parser.add_argument("--stride-seconds", type=int, default=10)
    full_parser.add_argument("--global-zscore", action="store_true", help="Use global z-score instead of per-video z-score")
    full_parser.add_argument("--min-video-zscore-windows", type=int, default=3)
    full_parser.add_argument("--long-static-z", type=float, default=1.2)
    full_parser.add_argument("--activity-drop-z", type=float, default=1.0)
    full_parser.add_argument("--frequent-walking-speed-z", type=float, default=1.0)
    full_parser.add_argument("--frequent-walking-distance-z", type=float, default=1.0)
    full_parser.add_argument("--min-positive-windows", type=int, default=2)
    full_parser.add_argument("--cv-folds", type=int, default=5)

    infer_parser = subparsers.add_parser("infer", help="Predict warning labels for one video")
    infer_parser.add_argument("--video-path", required=True)
    infer_parser.add_argument("--model-path", default="outputs_tuned/behavior_warning_model.joblib")
    infer_parser.add_argument("--output-dir", default="outputs_tuned/inference")
    infer_parser.add_argument("--sample-fps", type=float, default=2.0)
    infer_parser.add_argument("--window-seconds", type=int, default=30)
    infer_parser.add_argument("--stride-seconds", type=int, default=10)

    report_parser = subparsers.add_parser("report", help="Summarize pseudo-label distribution")
    report_parser.add_argument("--label-csv", default="outputs_tuned/behavior_labeled_windows.csv")

    return parser


def main() -> None:
    parser = _base_parser()
    args = parser.parse_args()

    feature_config = FeatureConfig(
        dataset_dir=Path(getattr(args, "dataset_dir", "Activity Analysis")),
        output_dir=Path(getattr(args, "output_dir", "outputs_tuned")),
        sample_fps=getattr(args, "sample_fps", 2.0),
        window_seconds=getattr(args, "window_seconds", 30),
        stride_seconds=getattr(args, "stride_seconds", 10),
    )
    pseudo_label_config = PseudoLabelConfig(
        use_per_video_zscore=not getattr(args, "global_zscore", False),
        min_windows_for_video_zscore=getattr(args, "min_video_zscore_windows", 3),
        long_static_z=getattr(args, "long_static_z", 1.2),
        activity_drop_z=getattr(args, "activity_drop_z", 1.0),
        frequent_walking_speed_z=getattr(args, "frequent_walking_speed_z", 1.0),
        frequent_walking_distance_z=getattr(args, "frequent_walking_distance_z", 1.0),
        min_positive_duration_windows=getattr(args, "min_positive_windows", 2),
    )

    if args.command == "extract":
        output = build_feature_dataset(feature_config)
        print(output)
        return

    if args.command == "label":
        output = assign_pseudo_labels(Path(args.feature_csv), pseudo_label_config, Path(args.output_dir))
        print(output)
        print(json.dumps(summarize_labels(output), ensure_ascii=False, indent=2))
        return

    if args.command == "train":
        outputs = train_behavior_model(
            Path(args.label_csv),
            Path(args.output_dir),
            TrainConfig(cv_folds=getattr(args, "cv_folds", 5)),
        )
        for key, value in outputs.items():
            print(f"{key}={value}")
        return

    if args.command == "pipeline":
        feature_csv = build_feature_dataset(feature_config)
        label_csv = assign_pseudo_labels(feature_csv, pseudo_label_config, Path(args.output_dir))
        outputs = train_behavior_model(
            label_csv,
            Path(args.output_dir),
            TrainConfig(cv_folds=getattr(args, "cv_folds", 5)),
        )
        print(f"feature_csv={feature_csv}")
        print(f"label_csv={label_csv}")
        print(json.dumps(summarize_labels(label_csv), ensure_ascii=False, indent=2))
        for key, value in outputs.items():
            print(f"{key}={value}")
        return

    if args.command == "infer":
        infer_config = FeatureConfig(
            output_dir=Path(args.output_dir),
            sample_fps=args.sample_fps,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
        )
        output = infer_video(Path(args.video_path), Path(args.model_path), Path(args.output_dir), infer_config)
        print(output)
        return

    if args.command == "report":
        print(json.dumps(summarize_labels(Path(args.label_csv)), ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
