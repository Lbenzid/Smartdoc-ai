from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

DOCS_DIR = Path("data/docs")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Modèle d'embedding (petit et rapide)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Charger les textes
docs = []
names = []

for file in DOCS_DIR.glob("*.txt"):
    text = file.read_text(encoding="utf-8")
    docs.append(text)
    names.append(file.stem)

print(f"🔍 {len(docs)} documents à indexer.")

# Calculer les embeddings
embeddings = model.encode(docs)

# Création de l'index FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Sauvegarder l'index
faiss.write_index(index, str(INDEX_DIR / "faiss_index.index"))

# Sauvegarder les noms associés
with open(INDEX_DIR / "doc_names.pkl", "wb") as f:
    pickle.dump(names, f)

print("✅ Index FAISS créé et sauvegardé.")
