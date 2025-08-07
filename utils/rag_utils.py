import os
import re
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from models.embeddings import get_embedding_model

def load_scheme_data(file_path="data/schemes.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    scheme_blocks = re.split(r"\n\d+\.\n", raw_text)
    documents = []

    for block in scheme_blocks:
        if not block.strip():
            continue

        scheme = {}
        for field in [
            "scheme_name", "benefit", "eligibility", "category",
            "start_date", "application_details", "region", "source_url"
        ]:
            match = re.search(rf"{field}:\s*(.*?)(?=\n\w+:|\Z)", block, re.DOTALL)
            scheme[field] = match.group(1).strip() if match else "Not specified"

        content = f"""
📌 **Scheme Name**: {scheme['scheme_name']}
💰 **Benefit**: {scheme['benefit']}
🧾 **Eligibility**: {scheme['eligibility']}
🏷️ **Category**: {scheme['category']}
📅 **Start Date**: {scheme['start_date']}
📝 **Application Details**: {scheme['application_details']}
📍 **Region**: {scheme['region']}
🔗 **Source**: {scheme['source_url']}
"""
        documents.append(Document(page_content=content))

    return documents

def create_vectorstore_from_documents(documents):
    embeddings = get_embedding_model()
    return FAISS.from_documents(documents, embeddings)

def create_vectorstore_from_text(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    documents = [Document(page_content=text) for text in texts]
    split_docs = splitter.split_documents(documents)
    return create_vectorstore_from_documents(split_docs)

def retrieve_relevant_chunks(vectorstore, query, k=3):
    return vectorstore.similarity_search(query, k=k)
