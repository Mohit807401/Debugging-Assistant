import os
import json
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

def ask_debug_agent(query):
    try:
        retriever = load_vectorstore().as_retriever()
        docs = retriever.get_relevant_documents(query)
        context = "\n---\n".join([doc.page_content for doc in docs[:2]])

        prompt = (
            "You are a Microbit debugging expert.\n\n"
            "Here are known cases and issues:\n"
            f"{context}\n\n"
            f"User's Question: {query}\n\n"
            "Answer in a helpful and clear way:"
        )

        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }

        json_data = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.7
        }

        response = requests.post("https://api.together.xyz/inference", headers=headers, json=json_data)
        return response.json()["choices"][0]["text"].strip()

    except Exception as e:
        import traceback
        return f"❌ Error:\n{traceback.format_exc()}"
