# Identity.py
class ManifestoIdentity:
    def __init__(self, row):
        self.country = row["country"]
        self.party = row["partyname"]
        self.date = row["date"]
        self.text = row["text"]
        self.true_rile = row["rile"]

    def build_prompt(self):
        """
        构造LLM的提示词。模型知道党派、国家、年份和宣言文本。
        要求模型预测 RILE (Left-Right) 分数。
        """
        return f"""
You are a comparative political scientist specializing in party manifestos.

Task:
Estimate the ideological leaning (Left–Right) of the following political manifesto
on a numeric scale from -100 (far left) to +100 (far right).

Provide your reasoning briefly.

Country: {self.country}
Party: {self.party}
Year: {self.date}

Manifesto excerpt:
{self.text}

Output your result in JSON format only:
{{
  "predicted_rile": <numeric_value>,
  "rationale": "<short explanation>"
}}
"""