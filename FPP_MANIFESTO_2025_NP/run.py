# ==========================================================
# FPP_MANIFESTO_2025_NP/run.py
# Cross-national, No Political Ideology Experiment
# ==========================================================
import os
import pandas as pd
from manifesto_loader import load_manifesto_data
from identity import IdentityProcessor
from config import load_model
import argparse

def main(args):
    print("=== FPP_MANIFESTO_2025_NP: Cross-national experiment (No Ideology) ===")
    print(f"Using model: {args.model}")

    # Load Manifesto dataset (without ideology)
    df = load_manifesto_data(include_ideology=False)
    print(f"Loaded {len(df)} manifesto samples.")

    # Initialize model client
    model_client = load_model(args.model)

    # Initialize identity processor
    processor = IdentityProcessor(model_client=model_client)

    # Run predictions
    outputs = []
    for i, row in df.iterrows():
        identity = processor.create_identity_profile(
            country=row["country"],
            year=row["year"],
            party=row["party"],
            text=row["text"],
            ideology=None  # explicitly None
        )
        result = processor.query_vote_preference(identity)
        outputs.append({
            "party": row["party"],
            "country": row["country"],
            "year": row["year"],
            "predicted_vote": result,
            "true_vote": row["true_vote"] if "true_vote" in df.columns else None
        })

    # Save results
    out_path = "results.csv"
    pd.DataFrame(outputs).to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()
    main(args)
