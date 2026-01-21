# Adult Income Prediction — End-to-End ML Pipeline (Python)

A reproducible tabular ML pipeline that predicts whether income is `>50K` using the Adult Census Income dataset.

## What this project demonstrates
- Real dataset usage (OpenML Adult)
- Data cleaning (handling missing values)
- Proper preprocessing with `ColumnTransformer`
- Model training + comparison (Logistic Regression vs Random Forest)
- Evaluation with ROC-AUC, F1, confusion matrix
- Saved artifacts for reproducible inference

## Dataset
- Source: OpenML — Adult Census Income dataset (`adult`, version 2)
- Target: `class` (income `>50K` vs `<=50K`)
- A local snapshot is saved to: `data/raw/adult.csv`

## Project structure
ml-income-pipeline/

data/raw/adult.csv

src/

train.py

predict.py

utils.py

artifacts/

metrics.json

roc_curve.png

## Setup (Windows / PowerShell)

python -m venv .venv

.venv\Scripts\Activate.ps1

pip install -r requirements.txt

## Train + Evaluate
python src/train.py


## This:

- downloads the dataset from OpenML
- cleans missing values (? -> NaN)
- preprocesses numeric + categorical features
- trains two models:
- Logistic Regression (baseline)
- Random Forest (nonlinear)
- selects best by ROC-AUC

## Writes:
artifacts/metrics.json

artifacts/roc_curve.png

artifacts/best_model.joblib (ignored in git)

## Inference
python src/predict.py

## Results
Best model: Random Forest

ROC-AUC: 0.9055

Accuracy: 0.8558

F1 (positive class): 0.6745

Confusion matrix:

TN=6900, FP=531

FN=878, TP=1460
