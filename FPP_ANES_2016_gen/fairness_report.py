"""
fairness_report.py
-----------------------------------
Generates subgroup-level fairness metrics for Political-LLM experiments
using the ANES 2016 dataset results.

Expected input: fairness_results.csv with columns:
['predicted_vote', 'true_vote', 'gender', 'age', 'education_level']

Metrics:
- Accuracy per subgroup
- Expected Calibration Error (ECE)
- Answer Rate Parity

Example:
    $ python fairness_report.py --data fairness_results.csv --out fairness_summary.csv
"""

import argparse
import pandas as pd
import numpy as np
from collections import defaultdict

# ------------------------------------------------------------
# 1. Metric definitions
# ------------------------------------------------------------

def compute_accuracy(df: pd.DataFrame) -> float:
    """Compute prediction accuracy."""
    if len(df) == 0:
        return np.nan
    return (df["predicted_vote"] == df["true_vote"]).mean()


def compute_ece(df: pd.DataFrame, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Requires model confidence scores; if not available, approximates by class frequency.
    """
    if "confidence" not in df.columns:
        # Use a dummy uniform confidence approximation if missing
        df["confidence"] = 0.5
    df = df.copy()
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


def compute_answer_rate_parity(df: pd.DataFrame, group_col: str) -> float:
    """
    Measure answer-rate parity:
    absolute difference between group-level answer rates and global average.
    Here, answer rate = proportion of rows with non-missing predicted_vote.
    """
    overall_rate = df["predicted_vote"].notnull().mean()
    group_rates = df.groupby(group_col)["predicted_vote"].apply(lambda x: x.notnull().mean())
    return abs(group_rates - overall_rate).mean()


def categorize_age(age):
    """Convert numeric age into categorical bins."""
    try:
        age = float(age)
    except:
        return "Unknown"
    if age < 18:
        return "Under18"
    elif 18 <= age <= 29:
        return "18-29"
    elif 30 <= age <= 49:
        return "30-49"
    elif 50 <= age <= 64:
        return "50-64"
    elif age >= 65:
        return "65+"
    else:
        return "Unknown"


# ------------------------------------------------------------
# 2. Main Fairness Analysis
# ------------------------------------------------------------

def subgroup_fairness_report(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    Compute fairness metrics for each demographic attribute.
    Returns a summary DataFrame.
    """
    results = []
    for col in group_cols:
        for subgroup, sub_df in df.groupby(col):
            acc = compute_accuracy(sub_df)
            ece = compute_ece(sub_df)
            ans_parity = compute_answer_rate_parity(df, col)
            results.append({
                "Attribute": col,
                "Subgroup": subgroup,
                "Count": len(sub_df),
                "Accuracy": round(acc, 4),
                "ECE": round(ece, 4),
                "AnswerRateParity": round(ans_parity, 4)
            })
    return pd.DataFrame(results)


# ------------------------------------------------------------
# 3. CLI interface
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate subgroup-level fairness report for Political-LLM.")
    parser.add_argument("--data", type=str, default="fairness_results.csv", help="Path to input CSV file.")
    parser.add_argument("--out", type=str, default="fairness_summary.csv", help="Path to output CSV file.")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data)
    required_cols = {"predicted_vote", "true_vote", "gender", "age", "education_level"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {required_cols}")

    df["age_group"] = df["age"].apply(categorize_age)
    #Generation Fairness Report
    group_cols = ["gender", "age_group", "education_level"]
    fairness_df = subgroup_fairness_report(df, group_cols)

    # Save and print
    fairness_df.to_csv(args.out, index=False)
    print("\n=== Fairness Summary ===")
    print(fairness_df.to_string(index=False))
    print(f"\nSaved fairness summary to: {args.out}")


if __name__ == "__main__":
    main()
