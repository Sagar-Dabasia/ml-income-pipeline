from __future__ import annotations

import json
from pathlib import Path
import joblib
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_path = root / "artifacts" / "best_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run: python src/train.py")

    model = joblib.load(model_path)

    # Example single-row inference (edit these values)
    sample = {
        "age": 37,
        "workclass": "Private",
        "fnlwgt": 284582,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States",
    }

    X = pd.DataFrame([sample])
    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)

    print(json.dumps({"predicted_income_gt_50k": pred, "probability": proba}, indent=2))


if __name__ == "__main__":
    main()
