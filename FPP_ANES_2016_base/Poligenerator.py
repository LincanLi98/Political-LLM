import re
import time
from bedrock_client import invoke_model

class PoliticalBiasProcessor:
    def __init__(self, model_id='gpt-4o-mini', max_retries=5):
        self.model_id = model_id
        self.max_retries = max_retries

    def create_prompt(self, identity):
        question = (
            "When it comes to politics, would you describe yourself as:\n"
            "   - No answer\n"
            "   - Very liberal\n"
            "   - Somewhat liberal\n"
            "   - Closer to liberal\n"
            "   - Moderate\n"
            "   - Closer to conservative\n"
            "   - Somewhat conservative\n"
            "   - Very conservative"
        )
        return f"{identity}\n\nTask:\nPlease answer the following question as if you were the resident:\n1. {question}"

    def generate_polibias(self, identity):
        prompt = self.create_prompt(identity)
        response = self.call_api(prompt)
        
        if response:
            ideology_text = self.extract_ideology_text(response)
            return self.insert_ideology_into_description(identity, ideology_text)
        return identity

    def call_api(self, prompt):
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                result = invoke_model(
                    model_id=self.model_id,
                    prompt=prompt,
                    max_tokens=200,
                    temperature=0.7
                )
                return result.get("generation", "")
            
            except Exception as e:
                retry_count += 1
                wait_time = 2 ** retry_count
                print(f"API error. Retrying in {wait_time} seconds... (Attempt {retry_count}/{self.max_retries})")
                print(f"Error: {e}")
                time.sleep(wait_time)
        
        print("Failed to get a valid response after maximum retries.")
        return None

    def extract_ideology_text(self, response):
        match = re.search(
            r'Closer to conservative|Closer to liberal|Very liberal|Somewhat liberal|Moderate|Somewhat conservative|Very conservative|No answer',
            response,
            re.IGNORECASE
        )
        return match.group(0).capitalize() if match else "No answer"

    def insert_ideology_into_description(self, identity, ideology_text):
        first_period_index = identity.find('.')
        if first_period_index != -1:
            return f"{identity[:first_period_index + 1]} When it comes to politics, you would describe yourself as {ideology_text}.{identity[first_period_index + 1:]}"
        return f"{identity} When it comes to politics, you would describe yourself as {ideology_text}."


def generate_polibias(identity, model_id='gpt-4o-mini'):
    processor = PoliticalBiasProcessor(model_id=model_id)
    return processor.generate_polibias(identity)