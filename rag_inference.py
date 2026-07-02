import json
import os
from pathlib import Path

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from qdrant_store import search_qdrant

BASE_DIR = Path(__file__).resolve().parent

# =====================================
# LOAD DATA
# =====================================

with open(
    BASE_DIR / "bbc_chunks.json",
    "r",
    encoding="utf-8"
) as f:
    chunks = json.load(f)

'''index = faiss.read_index(
    str(BASE_DIR / "bbc_faiss.index")
)

model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)
USE_QDRANT = True
##USE_QDRANT = os.getenv("USE_QDRANT", "0").lower() in {"1", "true", "yes", "on"}
'''

model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)

USE_QDRANT = True
# or
# USE_QDRANT = os.getenv("USE_QDRANT", "0").lower() in {"1","true","yes","on"}

if not USE_QDRANT:
    index = faiss.read_index(
        str(BASE_DIR / "bbc_faiss.index")
    )
else:
    index = None
# =====================================
# BUILD MAPPINGS
# =====================================

row_to_chunkid = {}
chunkid_to_row = {}

for row_idx, chunk in enumerate(chunks):

    uid = (
        f"{chunk['article_id']}_"
        f"{chunk['chunk_id']}"
    )

    row_to_chunkid[row_idx] = uid
    chunkid_to_row[uid] = row_idx

# =====================================
# TF-IDF SETUP
# =====================================

chunk_texts = [
    chunk["chunk_text"]
    for chunk in chunks
]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    max_features=50000
)

tfidf_matrix = vectorizer.fit_transform(
    chunk_texts
)

# =====================================
# SEMANTIC RETRIEVAL
# =====================================

def semantic_retrieve(
    question,
    top_k=20
):

    if USE_QDRANT:
        print("Using Qdrant")
        try:
            results = search_qdrant(
                question,
                top_k=top_k,
                model=model
            )
            return [
                (uid, float(score))
                for uid, score in results
            ]
        except Exception as exc:  # pragma: no cover - defensive path
            print(f"Qdrant search failed, falling back to FAISS: {exc}")

    q_emb = model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(q_emb)

    scores, indices = index.search(
        q_emb,
        top_k
    )

    return [
        (row_to_chunkid[i], float(scores[0][j]))
        for j, i in enumerate(indices[0])
    ]

# =====================================
# LEXICAL RETRIEVAL
# =====================================

def lexical_retrieve(
    question,
    top_k=20
):

    q_tfidf = vectorizer.transform(
        [question]
    )

    scores = (
        tfidf_matrix @ q_tfidf.T
    ).toarray().ravel()

    top_indices = np.argsort(
        -scores
    )[:top_k]

    return [
        (
            row_to_chunkid[i],
            float(scores[i])
        )
        for i in top_indices
        if scores[i] > 0
    ]

# =====================================
# RRF RETRIEVAL
# =====================================

def retrieve(
    question,
    top_k=5,
    sem_k=20,
    lex_k=20,
    rrf_k=60
):

    sem_results = semantic_retrieve(
        question,
        sem_k
    )

    lex_results = lexical_retrieve(
        question,
        lex_k
    )

    rrf_scores = {}

    for rank, (uid, _) in enumerate(
        sem_results,
        start=1
    ):

        rrf_scores[uid] = (
            rrf_scores.get(uid, 0)
            + 1 / (rrf_k + rank)
        )

    for rank, (uid, _) in enumerate(
        lex_results,
        start=1
    ):

        rrf_scores[uid] = (
            rrf_scores.get(uid, 0)
            + 1 / (rrf_k + rank)
        )

    ranking = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        uid
        for uid, _
        in ranking[:top_k]
    ]
# =====================================
# GET CHUNK TEXTS
# =====================================

def get_chunk_texts(
    retrieved_chunk_ids
):

    retrieved_chunks = []

    for uid in retrieved_chunk_ids:

        row_idx = chunkid_to_row[uid]

        retrieved_chunks.append(
            chunks[row_idx]["chunk_text"]
        )

    return retrieved_chunks

# =====================================
# BUILD CONTEXT
# =====================================

def build_context(
    retrieved_texts
):

    context = "\n\n".join(
        retrieved_texts
    )

    return context

# =====================================
# TEST
# =====================================

if __name__ == "__main__":

    question = (
        "Who backed Chelsea's decision to sack Adrian Mutu?"
    )

    retrieved_ids = retrieve(
        question,
        top_k=5
    )

    print("\nRetrieved IDs:\n")
    print(retrieved_ids)

    retrieved_texts = get_chunk_texts(
        retrieved_ids
    )

    print("\nRetrieved Chunks:\n")

    for i, chunk in enumerate(
        retrieved_texts,
        start=1
    ):
        print(f"\nChunk {i}\n")
        print(chunk)

    context = build_context(
        retrieved_texts
    )

    print("\n")
    print("=" * 80)
    print("FINAL CONTEXT")
    print("=" * 80)

    print(context)