import os 
import requests


class GeminiAPI:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={self.api_key}"

    def ask(self, context_chunks, user_question):
        context = "\n\n".join(context_chunks)
        prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {user_question}
Answer:"""

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        response = requests.post(self.url, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            raise Exception(f"Gemini API Error: {response.status_code} - {response.text}")