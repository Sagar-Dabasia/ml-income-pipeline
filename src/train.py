from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)

from utils import Paths, save_json


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = Paths.make(root)

    data_dir = root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "adult.csv"

    # 1) Load dataset from OpenML (real public dataset)
    data = fetch_openml(name="adult", version=2, as_frame=True)
    df = data.frame.copy()

    # Save a local copy into the repo for reproducibility
    if not csv_path.exists():
        df.to_csv(csv_path, index=False)
        print(f"Saved dataset snapshot to: {csv_path}")
    else:
        print(f"Dataset snapshot already exists: {csv_path}")

    # 2) Basic cleaning
    df = df.replace("?", np.nan)

    target_col = "class"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str)
    y = y.apply(lambda v: 1 if ">50K" in v else 0)

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Preprocessing
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transform = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_transform = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transform, numeric_cols),
            ("cat", cat_transform, cat_cols),
        ],
        remainder="drop",
    )

    # 5) Models
    lr = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=2000)),
    ])

    rf = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )),
    ])

    models = {"logreg": lr, "random_forest": rf}

    results = {}
    best_name, best_auc, best_model = None, -1.0, None

    for name, model in models.items():
        print(f"\nTraining: {name}")
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="binary", zero_division=0
        )
        auc = roc_auc_score(y_test, proba)
        cm = confusion_matrix(y_test, preds).tolist()

        results[name] = {
            "accuracy": float(acc),
            "precision": float(pr),
            "recall": float(rc),
            "f1": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": cm,
        }
        print(results[name])

        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = model

    assert best_model is not None and best_name is not None

    joblib.dump(best_model, paths.model_path)

    report = {
        "dataset_source": "OpenML: adult (Census Income)",
        "dataset_snapshot": str(csv_path),
        "best_model": best_name,
        "metrics": results,
        "notes": [
            "Replaced '?' with NaN and applied imputation.",
            "Used ColumnTransformer for numeric/categorical preprocessing.",
            "Selected best model by ROC-AUC on held-out test set.",
        ],
    }
    save_json(paths.metrics_path, report)

    RocCurveDisplay.from_predictions(y_test, best_model.predict_proba(X_test)[:, 1])
    plt.title(f"ROC Curve (Best: {best_name})")
    fig_path = paths.artifacts / "roc_curve.png"
    plt.savefig(fig_path, bbox_inches="tight", dpi=200)
    plt.close()

    print(f"\nSaved best model to: {paths.model_path}")
    print(f"Saved metrics to: {paths.metrics_path}")
    print(f"Saved ROC curve to: {fig_path}")


if __name__ == "__main__":
    main()
