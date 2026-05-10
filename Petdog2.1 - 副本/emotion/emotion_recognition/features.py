from __future__ import annotations

from typing import Dict, Iterable


def label_name_map(labels: Iterable[str]) -> Dict[str, str]:
    mapping = {
        "alert": "alert",
        "angry": "angry",
        "frown": "frown",
        "happy": "happy",
        "relax": "relax",
    }
    return {label: mapping.get(label, label) for label in labels}