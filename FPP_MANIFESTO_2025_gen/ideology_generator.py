# ==========================================================
# FPP_MANIFESTO_2025_gen/ideology_generator.py
# ==========================================================
import numpy as np
from openai import OpenAI
import json

client = OpenAI()

def generate_ideology_embedding(topic_distribution, model_name):
    """
    Given a list of 56 topic probabilities, ask the LLM to infer overall ideology.
    Returns: one of ['left', 'center', 'right'] or a descriptive embedding string.
    """
    prompt = f"""
    The following is a political manifesto represented as a 56-topic probability distribution (ManifestoBERTa output):
    {json.dumps(topic_distribution[:10])} ... [truncated]
    Based on this distribution, infer the general ideological orientation of this party as one of:
    - "left" (progressive, welfare-oriented, social equality)
    - "center" (moderate, balanced policy mix)
    - "right" (market-oriented, traditional, national sovereignty)
    Respond concisely with one of these three labels.
    """

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.3
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"[Warning] LLM ideology generation failed: {e}")
        return "unknown"

