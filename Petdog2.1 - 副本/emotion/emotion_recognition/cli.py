from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cnn_infer import infer_emotion_cnn
from .cnn_train import export_experiment_table, train_emotion_cnn
from .config import EmotionCnnTrainConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dog emotion recognition pipeline based on ResNet50")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the ResNet50 dog emotion recognition model")
    train_parser.add_argument("--dataset-dir", default="Dog Emotions - 5 Classes\\train_images_5_class")
    train_parser.add_argument("--output-dir", default="outputs_emotion")
    train_parser.add_argument("--epochs", type=int, default=24)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--image-size", type=int, default=256)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--val-split", type=float, default=0.2)
    train_parser.add_argument("--dropout", type=float, default=0.3)
    train_parser.add_argument("--label-smoothing", type=float, default=0.1)
    train_parser.add_argument("--early-stopping-patience", type=int, default=6)
    train_parser.add_argument("--freeze-backbone-epochs", type=int, default=3)
    train_parser.add_argument("--backbone-learning-rate-scale", type=float, default=0.2)
    train_parser.add_argument("--loss-name", choices=["cross_entropy", "focal"], default="cross_entropy")
    train_parser.add_argument("--focal-gamma", type=float, default=2.0)
    train_parser.add_argument("--num-workers", type=int, default=0)

    infer_parser = subparsers.add_parser("infer", help="Predict emotion for one image with the ResNet50 model")
    infer_parser.add_argument("--image-path", required=True)
    infer_parser.add_argument("--model-path", default="outputs_emotion/emotion_resnet50.pt")

    export_parser = subparsers.add_parser("export-experiments", help="Export experiment log to CSV")
    export_parser.add_argument("--output-dir", default="outputs_emotion")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        outputs = train_emotion_cnn(
            EmotionCnnTrainConfig(
                dataset_dir=Path(args.dataset_dir),
                output_dir=Path(args.output_dir),
                epochs=args.epochs,
                batch_size=args.batch_size,
                image_size=args.image_size,
                learning_rate=args.learning_rate,
                min_learning_rate=args.min_learning_rate,
                weight_decay=args.weight_decay,
                val_split=args.val_split,
                dropout=args.dropout,
                label_smoothing=args.label_smoothing,
                early_stopping_patience=args.early_stopping_patience,
                freeze_backbone_epochs=args.freeze_backbone_epochs,
                backbone_learning_rate_scale=args.backbone_learning_rate_scale,
                loss_name=args.loss_name,
                focal_gamma=args.focal_gamma,
                num_workers=args.num_workers,
            )
        )
        for key, value in outputs.items():
            print(f"{key}={value}")
        return

    if args.command == "infer":
        result = infer_emotion_cnn(Path(args.image_path), Path(args.model_path))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "export-experiments":
        output_path = export_experiment_table(Path(args.output_dir))
        print(output_path)
        return


if __name__ == "__main__":
    main()
