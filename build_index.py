import json
import numpy as np
import faiss

# load chunks
with open("bbc_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# load embeddings
embeddings = np.load("bbc_embeddings.npy")

print("Embeddings shape:", embeddings.shape)

# normalize for cosine similarity
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

faiss.write_index(index, "bbc_faiss.index")

print("Index created successfully")
print("Total vectors:", index.ntotal)