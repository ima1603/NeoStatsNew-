import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
if EMBEDDING_MODEL_NAME is None:
    raise ValueError("EMBEDDING_MODEL_NAME is not set. Check your .env file.")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"  # or "gemini-1.5-pro"

if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY. Set it as an environment variable.")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "mixtral-8x7b"  # or any other model you use
if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY is not set. Check your .env file.")
 
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")  # For live web search


