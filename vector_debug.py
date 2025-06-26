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

        prompt = f"""You are a Microbit debugging assistant. 

Only extract SOLUTIONS from the given context and format them as bullet points.
Do NOT include greetings, explanations, or anything else.

✅ Each point should start with a UPPERCASE action heading, followed by a short description.
✅ Be crisp and technical.

Context:
{context}

Issue:
{query}

Answer:"""


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
