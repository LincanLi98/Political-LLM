"""
evaluation.py
-----------------------------------
Unified evaluation module for Political-LLM experiments
(Applies to base, gen, and NP pipelines).

Computes standardized performance metrics:
- Accuracy
- F1-score (macro and weighted)
- Precision, Recall
- Calibration Error (ECE)
- Correlation (for continuous ideology scores)
- Confusion Matrix summary

Usage:
    $ python evaluation.py --data FPP_ANES_2016_base/results.csv --out FPP_ANES_2016_base/eval_summary.csv
"""

import argparse
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# ------------------------------------------------------------
# 1. Helper metrics
# ------------------------------------------------------------

def compute_accuracy(y_true, y_pred) -> float:
    return accuracy_score(y_true, y_pred)


def compute_f1_scores(y_true, y_pred) -> Tuple[float, float]:
    """Return (macro F1, weighted F1)."""
    return (
        f1_score(y_true, y_pred, average="macro"),
        f1_score(y_true, y_pred, average="weighted")
    )


def compute_precision_recall(y_true, y_pred) -> Tuple[float, float]:
    """Return (precision, recall)."""
    return (
        precision_score(y_true, y_pred, average="macro"),
        recall_score(y_true, y_pred, average="macro")
    )


def compute_confusion_matrix_summary(y_true, y_pred, labels=None) -> pd.DataFrame:
    """Return confusion matrix as a DataFrame."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    return cm_df


def compute_expected_calibration_error(df: pd.DataFrame, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Requires 'confidence' column.
    If missing, assumes uniform confidence of 0.5.
    """
    if "confidence" not in df.columns:
        df["confidence"] = 0.5

    df["correct"] = (df["predicted_vote"] == df["true_vote"]).astype(int)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (df["confidence"] > bins[i]) & (df["confidence"] <= bins[i + 1])
        bin_size = bin_mask.mean()
        if bin_size > 0:
            acc_bin = df.loc[bin_mask, "correct"].mean()
            conf_bin = df.loc[bin_mask, "confidence"].mean()
            ece += bin_size * abs(acc_bin - conf_bin)
    return ece


def compute_correlation(df: pd.DataFrame) -> float:
    """
    Compute correlation between predicted and true ideology if columns exist.
    Returns np.nan if unavailable.
    """
    if {"predicted_ideology", "true_ideology"}.issubset(df.columns):
        return df["predicted_ideology"].corr(df["true_ideology"], method="pearson")
    else:
        return np.nan

# ------------------------------------------------------------
# 2. Evaluation pipeline
# ------------------------------------------------------------

def evaluate_political_llm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run standardized evaluation for Political-LLM experiments.
    Expected columns:
    - predicted_vote
    - true_vote
    - optional: confidence, predicted_ideology, true_ideology
    """
    y_true = df["true_vote"].astype(str)
    y_pred = df["predicted_vote"].astype(str)

    accuracy = compute_accuracy(y_true, y_pred)
    f1_macro, f1_weighted = compute_f1_scores(y_true, y_pred)
    precision, recall = compute_precision_recall(y_true, y_pred)
    ece = compute_expected_calibration_error(df)
    corr = compute_correlation(df)

    summary = {
        "Accuracy": round(accuracy, 4),
        "F1_macro": round(f1_macro, 4),
        "F1_weighted": round(f1_weighted, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "ECE": round(ece, 4),
        "Ideology_Correlation": round(corr, 4) if not np.isnan(corr) else "N/A",
        "Samples": len(df)
    }

    return pd.DataFrame([summary])


# ------------------------------------------------------------
# 3. Command-line Interface
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Political-LLM predictions with standardized metrics.")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file with model outputs.")
    parser.add_argument("--out", type=str, default="eval_summary.csv", help="Path to output CSV summary.")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    required_cols = {"predicted_vote", "true_vote"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input file must contain columns: {required_cols}")

    eval_df = evaluate_political_llm(df)
    eval_df.to_csv(args.out, index=False)

    print("\n=== Political-LLM Evaluation Summary ===")
    print(eval_df.to_string(index=False))
    print(f"\nSaved evaluation summary to: {args.out}")

    # Optionally print confusion matrix for debugging
    labels = sorted(df["true_vote"].dropna().unique())
    cm = compute_confusion_matrix_summary(df["true_vote"], df["predicted_vote"], labels=labels)
    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
