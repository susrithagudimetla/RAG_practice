import os
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

# ==========================================
# CONFIGURATION
# ==========================================

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "bbc_chunks"
)

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

# ==========================================
# CLIENT
# ==========================================

def get_client():

    return QdrantClient(url=QDRANT_URL)

# ==========================================
# BUILD POINTS
# ==========================================

def build_point_payloads(
    chunks,
    embeddings
):

    points = []

    for idx, chunk in enumerate(chunks):

        unique_id = (
            f"{chunk['article_id']}_"
            f"{chunk['chunk_id']}"
        )

        point = qmodels.PointStruct(

            # Integer ID required by Qdrant
            id=idx,

            vector=embeddings[idx].tolist(),

            payload={

                "unique_id": unique_id,

                "article_id": chunk["article_id"],

                "chunk_id": chunk["chunk_id"],

                "chunk_text": chunk["chunk_text"],

                "tags": chunk.get("tags", "")
            }
        )

        points.append(point)

    return points

# ==========================================
# UPLOAD
# ==========================================

def upload_chunks_to_qdrant(
    chunks,
    embeddings,
    collection_name=COLLECTION_NAME
):

    client = get_client()

    client.recreate_collection(

        collection_name=collection_name,

        vectors_config=qmodels.VectorParams(

            size=embeddings.shape[1],

            distance=qmodels.Distance.COSINE

        )
    )

    points = build_point_payloads(
        chunks,
        embeddings
    )

    batch_size = 500

    for start in range(0, len(points), batch_size):

        end = start + batch_size

        client.upsert(

            collection_name=collection_name,

            points=points[start:end]

        )

    return collection_name

# ==========================================
# SEARCH
# ==========================================
def search_qdrant(
    question,
    top_k=5,
    model=None,
    collection_name=COLLECTION_NAME
):

    if model is None:
        model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2"
        )

    client = get_client()

    query_vector = model.encode(
        question,
        convert_to_numpy=True
    ).astype(np.float32).tolist()

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )

    results = []

    for point in response.points:

        uid = f"{point.payload['article_id']}_{point.payload['chunk_id']}"

        results.append(
            (
                uid,
                float(point.score)
            )
        )

    return results