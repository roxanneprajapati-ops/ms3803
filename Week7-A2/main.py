"""
Fraud Detection Project - Week 7 Activity 2
Model: Linear Support Vector Machine (SVM)
Dataset: Kaggle Credit Card Fraud Detection dataset

Run:
    pip install -r requirements.txt
    python src/train_fraud_svm.py --input data/processed/fraud_detection_staging_sample.csv --output outputs

For best results, place the full Kaggle CSV in data/raw/ and run with that file path.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, precision_recall_curve
)


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Original shape: {df.shape}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def train_svm(df: pd.DataFrame, output_dir: Path):
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, random_state=42, stratify=y_trainval
    )

    model = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('svm', LinearSVC(class_weight='balanced', random_state=42, max_iter=5000, C=0.01))
    ])
    model.fit(X_train, y_train)

    # Tune the SVM decision threshold on validation data to improve F1-score.
    val_scores = model.decision_function(X_val)
    precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1_scores[:-1]))
    threshold = float(thresholds[best_idx])

    test_scores = model.decision_function(X_test)
    y_pred = (test_scores >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, test_scores),
        'average_precision_pr_auc': average_precision_score(y_test, test_scores),
        'confusion_matrix': cm.tolist(),
        'threshold': threshold,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred, digits=4, zero_division=0))
    joblib.dump({'model': model, 'threshold': threshold}, output_dir / 'svm_fraud_model.joblib')

    plt.figure(figsize=(6, 5), dpi=150)
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix - SVM Fraud Detection')
    plt.xticks([0, 1], ['Predicted Genuine', 'Predicted Fraud'], rotation=20, ha='right')
    plt.yticks([0, 1], ['Actual Genuine', 'Actual Fraud'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', fontsize=13)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', bbox_inches='tight')
    plt.close()

    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/processed/fraud_detection_staging_sample.csv')
    parser.add_argument('--output', default='outputs')
    args = parser.parse_args()

    df = load_and_clean(args.input)
    train_svm(df, Path(args.output))


if __name__ == '__main__':
    main()
