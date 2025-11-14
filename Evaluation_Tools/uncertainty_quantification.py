"""
uncertainty_quantification.py
-----------------------------------
Quantifies uncertainty in Political-LLM simulation metrics using bootstrap resampling.
Implements 95% confidence interval estimation for any aggregate metric function.

Example:
    $ python uncertainty_quantification.py --data data/predictions.csv --metric vote_ratio
"""

import argparse
import numpy as np
import pandas as pd
from typing import Callable, Tuple


# 1. Metric definitions (you can customize these)
def compute_vote_ratio(df: pd.DataFrame) -> float:
    """
    Example metric: fraction of predicted votes for Candidate A.
    Assumes df['predicted_vote'] ∈ {'A', 'B'}.
    """
    if 'predicted_vote' not in df.columns:
        raise ValueError("Column 'predicted_vote' not found in dataset.")
    total = len(df)
    if total == 0:
        return np.nan
    return (df['predicted_vote'] == 'A').mean()


def compute_ideology_alignment(df: pd.DataFrame) -> float:
    """
    Example metric: Pearson correlation between predicted and ground-truth ideology scores.
    Assumes df contains 'predicted_ideology' and 'true_ideology' columns.
    """
    if not {'predicted_ideology', 'true_ideology'}.issubset(df.columns):
        raise ValueError("Columns 'predicted_ideology' and 'true_ideology' must exist.")
    return df['predicted_ideology'].corr(df['true_ideology'], method='pearson')


# 2. Bootstrap resampling procedure
def bootstrap_confidence_interval(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    n_bootstrap: int = 100,
    ci: float = 0.95,
    random_state: int = 42
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute mean and confidence interval for a metric using bootstrap resampling.

    Args:
        df: Pandas DataFrame containing model predictions and labels.
        metric_fn: Function to compute the target metric.
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level (default 0.95).
        random_state: Random seed for reproducibility.

    Returns:
        mean_metric: Mean of bootstrap metric values.
        (lower, upper): Confidence interval bounds.
    """
    rng = np.random.default_rng(random_state)
    metrics = []

    for i in range(n_bootstrap):
        sample_idx = rng.choice(df.index, size=len(df), replace=True)
        sample_df = df.loc[sample_idx]
        metric_value = metric_fn(sample_df)
        metrics.append(metric_value)

    metrics = np.array(metrics)
    mean_metric = np.nanmean(metrics)
    lower = np.nanpercentile(metrics, (1 - ci) / 2 * 100)
    upper = np.nanpercentile(metrics, (1 + ci) / 2 * 100)

    return mean_metric, (lower, upper)


# 3. CLI interface for convenience

def main():
    parser = argparse.ArgumentParser(description="Bootstrap uncertainty quantification for Political-LLM metrics.")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file containing model outputs.")
    parser.add_argument("--metric", type=str, choices=["vote_ratio", "ideology_alignment"], required=True,
                        help="Metric to evaluate.")
    parser.add_argument("--n_boot", type=int, default=100, help="Number of bootstrap samples (default: 100).")
    parser.add_argument("--ci", type=float, default=0.95, help="Confidence level (default: 0.95).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    args = parser.parse_args()
    df = pd.read_csv(args.data)

    metric_map = {
        "vote_ratio": compute_vote_ratio,
        "ideology_alignment": compute_ideology_alignment
    }

    metric_fn = metric_map[args.metric]
    mean_val, (low, high) = bootstrap_confidence_interval(
        df, metric_fn, n_bootstrap=args.n_boot, ci=args.ci, random_state=args.seed
    )

    print(f"Metric: {args.metric}")
    print(f"Mean: {mean_val:.4f}")
    print(f"{int(args.ci * 100)}% Confidence Interval: [{low:.4f}, {high:.4f}]")

if __name__ == "__main__":
    main()
