# run.py
import argparse
import json
import time
import pandas as pd
import tqdm

from manifesto import load_manifesto_data
from Identity import ManifestoIdentity
from bedrock_client import query_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model ID (e.g. gpt-4o-mini or meta.llama3-1-70b-instruct-v1:0)")
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    print(f"Loading Manifesto dataset (sample_size={args.sample}) ...")
    df = load_manifesto_data(sample_size=args.sample)

    results = []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        identity = ManifestoIdentity(row)
        prompt = identity.build_prompt()

        try:
            response = query_model(prompt, model_id=args.model)
            parsed = json.loads(response)
        except Exception as e:
            parsed = {"predicted_rile": None, "rationale": str(e)}

        results.append({
            "country": identity.country,
            "party": identity.party,
            "year": identity.date,
            "true_rile": identity.true_rile,
            "predicted_rile": parsed.get("predicted_rile"),
            "rationale": parsed.get("rationale")
        })

        time.sleep(args.delay)

    out_path = f"manifesto_results_base_{args.model.replace('/', '_')}.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()