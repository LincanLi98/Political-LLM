# ==========================================================
# FPP_MANIFESTO_2025_gen/run.py
# Cross-national, LLM-generated ideology experiment
# ==========================================================
import pandas as pd
from manifesto_loader import generate_policy_topics
from ideology_generator import generate_ideology_embedding
from identity import IdentityProcessor
from config import load_model
import argparse

def main(args):
    print("=== FPP_MANIFESTO_2025_gen: Cross-national experiment (LLM-generated Ideology) ===")
    print(f"Using model: {args.model}")

    # Step 1: ManifestoBERTa inference — extract 56-topic policy distributions
    print("Generating policy-topic distributions using ManifestoBERTa...")
    df = generate_policy_topics(args.manifesto_csv)
    print(f"Extracted topic features for {len(df)} manifestos.")

    # Step 2: LLM generates ideological embeddings
    print("Generating ideology embeddings using LLM...")
    df["generated_ideology"] = df.apply(lambda r: generate_ideology_embedding(r["topic_distribution"], args.model), axis=1)

    # Step 3: Query LLM for downstream task (e.g., vote or stance prediction)
    model_client = load_model(args.model)
    processor = IdentityProcessor(model_client=model_client)

    outputs = []
    for _, row in df.iterrows():
        identity = processor.create_identity_profile(
            country=row["country"],
            year=row["year"],
            party=row["party"],
            ideology=row["generated_ideology"],
            text=None  # ideology embedding replaces raw text
        )
        result = processor.query_vote_preference(identity)
        outputs.append({
            "party": row["party"],
            "country": row["country"],
            "year": row["year"],
            "generated_ideology": row["generated_ideology"],
            "predicted_vote": result,
            "true_vote": row.get("true_vote", None)
        })

    # Step 4: Save results
    out_path = "results.csv"
    pd.DataFrame(outputs).to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--manifesto_csv", type=str, default="../../data/manifesto_sentences.csv")
    args = parser.parse_args()
    main(args)
