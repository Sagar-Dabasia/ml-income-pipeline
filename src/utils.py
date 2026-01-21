from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


@dataclass
class Paths:
    root: Path
    artifacts: Path
    model_path: Path
    metrics_path: Path

    @staticmethod
    def make(root: Path) -> "Paths":
        artifacts = root / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        return Paths(
            root=root,
            artifacts=artifacts,
            model_path=artifacts / "best_model.joblib",
            metrics_path=artifacts / "metrics.json",
        )


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
