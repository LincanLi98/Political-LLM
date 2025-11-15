# Identity.py
import os
import re
import threading
import time
from bedrock_client import invoke_model

class PoliticalBias:
    def __init__(self, model_id='gpt-4o-mini', output_file='responses/results.txt', 
                 log_file='responses/prompt_history.txt', max_retries=3):
        self.model_id = model_id
        self.max_retries = max_retries
        self.first_question = """
        As of today, will you vote for the Democratic Party (Hillary Clinton), the Republican Party (Donald Trump), or do you have no preference?\n   - Democratic\n   - Republican\n   - No Preference
        """
        self.output_file = output_file
        self.log_file = log_file
        self.republican_votes = 0
        self.democratic_votes = 0
        self.no_preference_votes = 0
        self.lock = threading.Lock()

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def create_prompt(self, identity, questions):
        questions_copy = [self.first_question.strip()] + questions
        questions_prompt = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions_copy)])
        prompt = f"{identity}\n\nTask:\nPlease answer the following questions as if you are the resident:\n{questions_prompt}"
        self.log_prompt(prompt)
        return prompt

    def log_prompt(self, prompt):
        with self.lock:
            with open(self.log_file, 'a') as f:
                f.write(f"{prompt}\n{'='*50}\n")

    def get_response(self, identity, questions):
        prompt = self.create_prompt(identity, questions)
        
        response = ""
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                result = invoke_model(
                    model_id=self.model_id,
                    prompt=prompt,
                    max_tokens=500,
                    temperature=0.7
                )
                response = result.get("generation", "")
                break
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                
                if "ThrottlingException" in error_msg or "Too many requests" in error_msg:
                    if retry_count < self.max_retries:
                        wait_time = 2 ** retry_count
                        print(f"\n⚠️  Rate limit hit. Waiting {wait_time}s... (Retry {retry_count}/{self.max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"\n❌ Max retries reached. Skipping this identity.")
                        return 0
                else:
                    print(f"\n❌ API Error: {e}")
                    return 0

        self.save_response(identity, response)
        first_answer = self.get_first_answer(response)
        score = self.extract_score(first_answer)
        self.update_votes(score, identity)
        return score

    def save_response(self, identity, response):
        with self.lock:
            with open(self.output_file, 'a') as f:
                f.write(f"Response for: {identity}\n\n")
                f.write(response)
                f.write("\n" + "="*50 + "\n")

    def get_first_answer(self, response):
        match = re.search(r'1\.\s*(.*?)\s*(?=\d\.)', response, re.DOTALL)
        return match.group(1).strip() if match else response

    def extract_score(self, sentence):
        sentence_lower = sentence.lower()
        
        if 'republican' in sentence_lower or 'trump' in sentence_lower:
            return 1
        elif 'democratic' in sentence_lower or 'democrat' in sentence_lower or 'clinton' in sentence_lower:
            return -1
        elif 'no preference' in sentence_lower or 'neither' in sentence_lower or 'undecided' in sentence_lower:
            return 0
        
        return 0

    def update_votes(self, score, identity):
        dirs = {
            1: 'republican_supporter',
            -1: 'democratic_supporter',
            0: 'nopreference_supporter'
        }
        dir_path = dirs.get(score)
        os.makedirs(dir_path, exist_ok=True)

        with self.lock:
            if score == 1:
                self.republican_votes += 1
            elif score == -1:
                self.democratic_votes += 1
            elif score == 0:
                self.no_preference_votes += 1

            with open(os.path.join(dir_path, 'identities.txt'), 'a') as f:
                f.write(identity + '\n')

    def get_results(self):
        return {
            "Republican": self.republican_votes,
            "Democratic": self.democratic_votes,
            "No Preference": self.no_preference_votes
        }
    