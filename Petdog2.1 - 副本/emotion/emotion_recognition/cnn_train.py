from __future__ import annotations

import json
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

from .config import EmotionCnnTrainConfig
from .features import label_name_map


class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_weight = (1.0 - pt).pow(self.gamma)
        return (focal_weight * ce_loss).mean()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def _set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        if not name.startswith("fc."):
            parameter.requires_grad = trainable


def _build_optimizer(model: nn.Module, config: EmotionCnnTrainConfig) -> torch.optim.Optimizer:
    head_params = []
    backbone_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    param_groups = []
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": config.learning_rate * config.backbone_learning_rate_scale,
            }
        )
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": config.learning_rate,
            }
        )
    return torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)


def _plot_training_curves(history: List[Dict[str, float]], output_path: Path) -> None:
    epochs = [item["epoch"] for item in history]
    train_acc = [item["train_accuracy"] for item in history]
    val_acc = [item["val_accuracy"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_acc, marker="o", label="Train Accuracy")
    axes[0].plot(epochs, val_acc, marker="o", label="Val Accuracy")
    axes[0].set_title("Accuracy Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[1].plot(epochs, val_loss, marker="o", label="Val Loss")
    axes[1].set_title("Loss Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _build_autocast_context(device: torch.device, enabled: bool):
    if enabled:
        return torch.amp.autocast(device_type=device.type)
    return nullcontext()


def _build_grad_scaler(device: torch.device, enabled: bool):
    if enabled:
        return torch.amp.GradScaler(device.type)
    return torch.amp.GradScaler(device.type, enabled=False)


def _build_criterion(config: EmotionCnnTrainConfig) -> nn.Module:
    if config.loss_name == "focal":
        return FocalLoss(gamma=config.focal_gamma, label_smoothing=config.label_smoothing)
    return nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)


def _append_experiment_record(output_dir: Path, record: Dict[str, object]) -> Path:
    record_path = output_dir / "emotion_experiment_log.jsonl"
    with record_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record_path


def export_experiment_table(output_dir: Path) -> Path:
    log_path = output_dir / "emotion_experiment_log.jsonl"
    if not log_path.exists():
        raise RuntimeError(f"Experiment log not found: {log_path}")

    records: List[Dict[str, object]] = []
    with log_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        raise RuntimeError(f"No experiment records found in: {log_path}")

    dataframe = pd.DataFrame(records)
    sort_columns = [column for column in ["best_val_macro_f1", "best_val_accuracy", "timestamp"] if column in dataframe.columns]
    ascending = [False, False, False][: len(sort_columns)]
    if sort_columns:
        dataframe = dataframe.sort_values(by=sort_columns, ascending=ascending)

    export_path = output_dir / "emotion_experiment_summary.csv"
    dataframe.to_csv(export_path, index=False, encoding="utf-8-sig")
    return export_path


def train_emotion_cnn(config: EmotionCnnTrainConfig) -> Dict[str, Path]:
    _set_seed(config.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    base_dataset = datasets.ImageFolder(root=str(config.dataset_dir))
    class_names: List[str] = list(base_dataset.classes)
    if not class_names:
        raise RuntimeError(f"No classes found under {config.dataset_dir}")

    targets = np.array(base_dataset.targets)
    indices = np.arange(len(base_dataset))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=config.val_split,
        random_state=config.random_state,
        stratify=targets,
    )
    train_subset = Subset(base_dataset, train_indices.tolist())
    val_subset = Subset(base_dataset, val_indices.tolist())

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(config.image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(int(config.image_size * 1.14)),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = TransformSubset(train_subset, train_transform)
    val_dataset = TransformSubset(val_subset, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = _build_model(len(class_names)).to(device)
    if isinstance(model.fc, nn.Sequential):
        model.fc[0].p = config.dropout

    _set_backbone_trainable(model, trainable=False)

    criterion = _build_criterion(config)
    optimizer = _build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.min_learning_rate,
    )
    scaler = _build_grad_scaler(device, use_amp)

    best_state = None
    best_accuracy = 0.0
    best_f1 = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        if epoch == config.freeze_backbone_epochs + 1:
            _set_backbone_trainable(model, trainable=True)
            optimizer = _build_optimizer(model, config)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(config.epochs - config.freeze_backbone_epochs, 1),
                eta_min=config.min_learning_rate,
            )
            print(
                "[Fine Tune] backbone unfrozen at epoch={epoch} backbone_lr={backbone_lr:.7f} head_lr={head_lr:.7f}".format(
                    epoch=epoch,
                    backbone_lr=config.learning_rate * config.backbone_learning_rate_scale,
                    head_lr=config.learning_rate,
                )
            )

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with _build_autocast_context(device, use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        y_true: List[int] = []
        y_pred: List[int] = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                with _build_autocast_context(device, use_amp):
                    logits = model(images)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                predictions = logits.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(predictions.cpu().tolist())

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(train_total, 1),
                "train_accuracy": train_acc,
                "val_loss": val_loss / max(val_total, 1),
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "learning_rate": max(group["lr"] for group in optimizer.param_groups),
                "phase": "head" if epoch <= config.freeze_backbone_epochs else "finetune",
            }
        )

        scheduler.step()

        print(
            "[Epoch {epoch:02d}/{total:02d}] "
            "train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            "val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            "val_macro_f1={val_f1:.4f} phase={phase} lr={lr:.7f}".format(
                epoch=epoch,
                total=config.epochs,
                train_loss=history[-1]["train_loss"],
                train_acc=train_acc,
                val_loss=history[-1]["val_loss"],
                val_acc=val_acc,
                val_f1=val_f1,
                phase=history[-1]["phase"],
                lr=history[-1]["learning_rate"],
            )
        )

        improved = (val_f1 > best_f1) or (val_f1 == best_f1 and val_acc >= best_accuracy)
        if improved:
            best_accuracy = val_acc
            best_f1 = val_f1
            best_epoch = epoch
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.early_stopping_patience:
            print(
                f"[Early Stop] epoch={epoch} patience={config.early_stopping_patience} "
                f"best_epoch={best_epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    model.eval()

    final_true: List[int] = []
    final_pred: List[int] = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            final_true.extend(labels.cpu().tolist())
            final_pred.extend(predictions.cpu().tolist())

    target_names = class_names
    report = classification_report(final_true, final_pred, target_names=target_names, output_dict=True, zero_division=0)
    matrix = confusion_matrix(final_true, final_pred).tolist()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "emotion_resnet50.pt"
    report_path = output_dir / "emotion_cnn_report.json"
    label_map_path = output_dir / "emotion_label_map.json"
    history_path = output_dir / "emotion_cnn_history.json"
    confusion_matrix_path = output_dir / "emotion_cnn_confusion_matrix.json"
    curve_path = output_dir / "emotion_cnn_training_curves.png"
    summary_path = output_dir / "emotion_cnn_summary.json"
    experiment_log_path = output_dir / "emotion_experiment_log.jsonl"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": config.image_size,
            "architecture": "resnet50",
            "dropout": config.dropout,
        },
        model_path,
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    label_map_path.write_text(json.dumps(label_name_map(class_names), ensure_ascii=False, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    confusion_matrix_path.write_text(
        json.dumps({"labels": class_names, "matrix": matrix}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_val_accuracy": best_accuracy,
                "best_val_macro_f1": best_f1,
                "epochs_completed": len(history),
                "freeze_backbone_epochs": config.freeze_backbone_epochs,
                "image_size": config.image_size,
                "loss_name": config.loss_name,
                "focal_gamma": config.focal_gamma,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_training_curves(history, curve_path)
    experiment_record_path = _append_experiment_record(
        output_dir,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset_dir": str(config.dataset_dir),
            "image_size": config.image_size,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "min_learning_rate": config.min_learning_rate,
            "weight_decay": config.weight_decay,
            "val_split": config.val_split,
            "dropout": config.dropout,
            "label_smoothing": config.label_smoothing,
            "early_stopping_patience": config.early_stopping_patience,
            "freeze_backbone_epochs": config.freeze_backbone_epochs,
            "backbone_learning_rate_scale": config.backbone_learning_rate_scale,
            "loss_name": config.loss_name,
            "focal_gamma": config.focal_gamma,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_accuracy,
            "best_val_macro_f1": best_f1,
            "epochs_completed": len(history),
            "model_path": str(model_path),
            "report_path": str(report_path),
            "history_path": str(history_path),
            "summary_path": str(summary_path),
        },
    )

    print(
        "[Best Model] best_epoch={best_epoch} best_val_acc={best_acc:.4f} best_val_macro_f1={best_f1:.4f}".format(
            best_epoch=best_epoch,
            best_acc=best_accuracy,
            best_f1=best_f1,
        )
    )
    print(f"[Experiment Log] {experiment_record_path}")

    return {
        "model_path": model_path,
        "report_path": report_path,
        "label_map_path": label_map_path,
        "history_path": history_path,
        "confusion_matrix_path": confusion_matrix_path,
        "curve_path": curve_path,
        "summary_path": summary_path,
        "experiment_log_path": experiment_log_path,
    }
