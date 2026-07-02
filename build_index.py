import json
from pathlib import Path

import faiss
import numpy as np

from qdrant_store import upload_chunks_to_qdrant

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# LOAD DATA
# ==========================================

with open(BASE_DIR / "bbc_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

embeddings = np.load(BASE_DIR / "bbc_embeddings.npy")

print("Embeddings shape:", embeddings.shape)

# ==========================================
# BUILD FAISS INDEX
# ==========================================

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

faiss.write_index(
    index,
    str(BASE_DIR / "bbc_faiss.index")
)

print("FAISS index created successfully")

# ==========================================
# BUILD QDRANT INDEX
# ==========================================

try:

    upload_chunks_to_qdrant(
        chunks,
        embeddings
    )

    print("Qdrant collection created successfully")

except Exception as e:

    print("Qdrant index build failed")
    print(e)

print("Total vectors:", index.ntotal)