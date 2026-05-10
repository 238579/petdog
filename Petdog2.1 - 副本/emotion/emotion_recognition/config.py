from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class EmotionCnnTrainConfig:
    dataset_dir: Path = Path("Dog Emotions - 5 Classes") / "train_images_5_class"
    output_dir: Path = Path("outputs_emotion")
    image_size: int = 256
    batch_size: int = 16
    epochs: int = 24
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    val_split: float = 0.2
    random_state: int = 42
    num_workers: int = 0
    dropout: float = 0.3
    label_smoothing: float = 0.1
    early_stopping_patience: int = 6
    freeze_backbone_epochs: int = 3
    backbone_learning_rate_scale: float = 0.2
    loss_name: Literal["cross_entropy", "focal"] = "cross_entropy"
    focal_gamma: float = 2.0
