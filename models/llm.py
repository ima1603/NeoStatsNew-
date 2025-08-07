import requests
from config.config import GEMINI_API_KEY

from langchain_groq import ChatGroq

def get_gemini_flash_model():
    """Return a callable Gemini Flash model using HTTP API"""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing. Please set it in your environment or config.py.")

    def chat(prompt: str) -> str:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": GEMINI_API_KEY
        }
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ]
        }
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Gemini API request failed: {str(e)}")
        except (KeyError, IndexError):
            raise RuntimeError(f"Unexpected Gemini response format: {data}")

    return chat

