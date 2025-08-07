import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings

#  Load environment variables from .env file
load_dotenv()

def get_embedding_model():
    try:
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception as e:
        raise RuntimeError(f"Embedding model init failed: {str(e)}")