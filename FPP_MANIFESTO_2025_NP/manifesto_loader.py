# ==========================================================
# FPP_MANIFESTO_2025_NP/manifesto_loader.py
# ==========================================================
import pandas as pd

def load_manifesto_data(include_ideology: bool = False):
    """
    Loads Manifesto Project data for the 2025 cross-national experiment.
    include_ideology=False means 'rile' and related ideology columns are excluded.
    """
    df = pd.read_csv("../../data/manifesto_data.csv")

    # Keep relevant columns
    cols = ["party", "country", "year", "text"]
    if include_ideology and "rile" in df.columns:
        cols.append("rile")
    df = df[cols]

    # Drop rows with missing text
    df = df.dropna(subset=["text"])
    return df

