import json
import numpy as np
from sentence_transformers import SentenceTransformer

with open("bbc_chunks.json","r",encoding="utf-8") as f:
    chunks = json.load(f)

texts = [
    chunk["chunk_text"]
    for chunk in chunks
]

print("Chunks:", len(texts))

model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True
)

print("Embedding Shape:")
print(embeddings.shape)

np.save(
    "bbc_embeddings.npy",
    embeddings
)