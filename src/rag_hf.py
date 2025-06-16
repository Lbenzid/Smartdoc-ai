
import faiss
import pickle
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv
import os

# === Chargement de la clé HuggingFace
load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if token is None:
    raise ValueError("Le token Hugging Face n'a pas été trouvé dans le fichier .env")

# === Paramètres ===
TOP_K = 3
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# === Chargement index FAISS et documents ===
index = faiss.read_index("data/index/faiss_index.index")
with open("data/index/doc_names.pkl", "rb") as f:
    doc_names = pickle.load(f)

DOCS_DIR = Path("data/docs")
texts = {
    name: (DOCS_DIR / f"{name}.txt").read_text(encoding="utf-8")
    for name in doc_names
}

# === Embedding model
model_embed = SentenceTransformer("all-MiniLM-L6-v2")

# === Chargement du modèle LLM avec token
print(f"🔄 Chargement du modèle {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    token=token
).to("cpu")  # CPU-friendly

print("✅ Modèle chargé.")

# === Fonction de réponse
def ask_question(question):
    print(f"\n❓ Question : {question}")
    query_embedding = model_embed.encode([question])
    D, I = index.search(np.array(query_embedding), TOP_K)

    context_docs = [texts[doc_names[i]][:500] for i in I[0]]
    context = "\n---\n".join(context_docs)

    prompt = f"""Tu es un assistant intelligent.
Voici des extraits de documents :
{context}

Réponds à la question suivante de manière claire et synthétique : {question}
"""

    MAX_INPUT_CHARS = 2500
    if len(prompt) > MAX_INPUT_CHARS:
        prompt = prompt[:MAX_INPUT_CHARS]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    try:
        print("⏳ Génération en cours...")

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("🧠 Réponse :\n")
        print(response)

    except Exception as e:
        print("Erreur pendant la génération :", e)

# === Boucle interactive
if __name__ == "__main__":
    while True:
        q = input("\n💬 Pose une question (ou tape 'exit') : ")
        if q.lower() in {"exit", "quit"}:
            break
        ask_question(q)

