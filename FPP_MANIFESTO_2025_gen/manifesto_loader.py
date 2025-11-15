# ==========================================================
# FPP_MANIFESTO_2025_gen/manifesto_loader.py
# ==========================================================
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

MODEL_NAME = "manifesto-project/manifestoberta-56topics-sentence"

def generate_policy_topics(input_csv):
    """
    Load manifesto sentences and run ManifestoBERTa to get topic distribution.
    Returns a DataFrame grouped by party, country, year, with average topic probs.
    """
    print(f"Loading ManifestoBERTa model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    df = pd.read_csv(input_csv)
    grouped = []

    for (party, country, year), subset in tqdm(df.groupby(["party", "country", "year"])):
        inputs = tokenizer(list(subset["text"]), truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            mean_probs = probs.mean(dim=0).numpy()
        grouped.append({
            "party": party,
            "country": country,
            "year": year,
            "topic_distribution": mean_probs.tolist()
        })

    return pd.DataFrame(grouped)

