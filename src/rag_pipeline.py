import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv

# === Chargement de la clé API depuis le fichier .env ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ Aucune clé API détectée. Vérifie ton fichier .env.")

client = OpenAI(api_key=api_key)
TOP_K = 3

# === Chargement de l'index FAISS et des noms de fichiers ===
index = faiss.read_index("data/index/faiss_index.index")
with open("data/index/doc_names.pkl", "rb") as f:
    doc_names = pickle.load(f)

# === Chargement des textes originaux ===
DOCS_DIR = Path("data/docs")
texts = {
    name: (DOCS_DIR / f"{name}.txt").read_text(encoding="utf-8")
    for name in doc_names
}

# === Modèle d'embedding ===
model = SentenceTransformer("all-MiniLM-L6-v2")

def ask_question(question):
    print(f"\n💬 Question : {question}")
    query_embedding = model.encode([question])
    D, I = index.search(np.array(query_embedding), TOP_K)

    context_docs = [texts[doc_names[i]] for i in I[0]]
    context = "\n---\n".join(context_docs)

    prompt = f"""Tu es un assistant intelligent.
Voici des extraits de documents :
{context}

Réponds à cette question de manière claire et concise : "{question}"
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Tu es un assistant intelligent."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    print("🧠 Réponse :")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    while True:
        q = input("\n❓ Pose une question (ou tape 'exit') : ")
        if q.lower() in {"exit", "quit"}:
            break
        ask_question(q)
