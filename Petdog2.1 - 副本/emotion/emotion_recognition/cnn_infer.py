from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from .features import label_name_map


def _build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def infer_emotion_cnn(image_path: Path, model_path: Path) -> Dict[str, object]:
    checkpoint = torch.load(model_path, map_location="cpu")
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", 224)
    architecture = checkpoint.get("architecture", "resnet50")
    dropout = checkpoint.get("dropout", 0.3)
    if architecture != "resnet50":
        raise RuntimeError(f"Unsupported model architecture: {architecture}")

    model = _build_model(len(class_names))
    if isinstance(model.fc, nn.Sequential):
        model.fc[0].p = dropout
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).tolist()

    score_map = {label: float(score) for label, score in zip(class_names, probabilities)}
    prediction = max(score_map, key=score_map.get)
    names_cn = label_name_map(class_names)
    return {
        "prediction": prediction,
        "prediction_cn": names_cn.get(prediction, prediction),
        "scores": score_map,
        "scores_cn": {names_cn.get(label, label): float(score) for label, score in score_map.items()},
    }
