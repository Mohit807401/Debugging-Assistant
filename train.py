from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import json

with open("debug_cases.json") as f:
    raw = json.load(f)

docs = []
for case in raw["microbit"]["cases"]:
    content = f"Title: {case['title']}\\nSymptoms: {', '.join(case['symptoms'])}\\nCauses: {', '.join(case['causes'])}\\nSolutions: {', '.join(case['solutions'])}"
    docs.append(Document(page_content=content, metadata={"id": case["id"]}))

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)
db.save_local("vectorstore")
