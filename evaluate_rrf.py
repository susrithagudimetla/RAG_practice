import json
import math
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# ======================================
# LOAD DATA
# ======================================

with open("bbc_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open("qa_dataset_clean.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

index = faiss.read_index("bbc_faiss.index")


model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)

# ======================================
# BUILD MAPPINGS
# ======================================

row_to_chunkid = {}
chunkid_to_row = {}

for row_idx, chunk in enumerate(chunks):

    unique_id = (
        f"{chunk['article_id']}_"
        f"{chunk['chunk_id']}"
    )

    row_to_chunkid[row_idx] = unique_id
    chunkid_to_row[unique_id] = row_idx

chunk_texts = [chunk["chunk_text"] for chunk in chunks]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    max_features=50000
)

tfidf_matrix = vectorizer.fit_transform(chunk_texts)

# ======================================
# RETRIEVAL
# ======================================

def semantic_retrieve(question, top_k=20):

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


def lexical_retrieve(question, top_k=20):

    q_tfidf = vectorizer.transform([question])
    lex_scores = (tfidf_matrix @ q_tfidf.T).toarray().ravel()
    top_indices = np.argsort(-lex_scores)[:top_k]

    return [
        (row_to_chunkid[i], float(lex_scores[i]))
        for i in top_indices
        if lex_scores[i] > 0
    ]

def retrieve(
    question,
    top_k=5,
    sem_k=20,
    lex_k=20,
    rrf_k=60
):

    sem_results = semantic_retrieve(
        question,
        top_k=sem_k
    )

    lex_results = lexical_retrieve(
        question,
        top_k=lex_k
    )

    rrf_scores = {}

    # Dense retrieval ranks
    for rank, (uid, _) in enumerate(
        sem_results,
        start=1
    ):

        rrf_scores[uid] = (
            rrf_scores.get(uid, 0)
            + 1 / (rrf_k + rank)
        )

    # Lexical retrieval ranks
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
# ======================================
# METRICS
# ======================================

def recall_at_k(retrieved, relevant):

    hits = len(
        set(retrieved) &
        set(relevant)
    )

    return hits / len(relevant)


def reciprocal_rank(retrieved, relevant):

    for rank, doc in enumerate(
        retrieved,
        start=1
    ):

        if doc in relevant:
            return 1 / rank

    return 0


def average_precision(
    retrieved,
    relevant
):

    hits = 0
    score = 0

    for rank, doc in enumerate(
        retrieved,
        start=1
    ):

        if doc in relevant:

            hits += 1

            score += hits / rank

    if len(relevant) == 0:
        return 0

    return score / len(relevant)


def ndcg(
    retrieved,
    relevant
):

    dcg = 0

    for i, doc in enumerate(retrieved):

        if doc in relevant:

            dcg += (
                1 /
                math.log2(i + 2)
            )

    ideal_dcg = 0

    for i in range(
        min(
            len(relevant),
            len(retrieved)
        )
    ):

        ideal_dcg += (
            1 /
            math.log2(i + 2)
        )

    if ideal_dcg == 0:
        return 0

    return dcg / ideal_dcg
# ======================================
# EVALUATION
# ======================================

recall_scores = []
mrr_scores = []
ap_scores = []
ndcg_scores = []

results = []

for sample in qa_pairs:

    question = sample["question"]

    gt = sample["ground_truth"]

    gt_chunk = (
        f"{gt['article_id']}_"
        f"{gt['chunk_id']}"
    )

    relevant = [gt_chunk]

    retrieved = retrieve(
        question,
        top_k=5
    )

    recall_scores.append(
        recall_at_k(
            retrieved,
            relevant
        )
    )

    mrr_scores.append(
        reciprocal_rank(
            retrieved,
            relevant
        )
    )

    ap_scores.append(
        average_precision(
            retrieved,
            relevant
        )
    )

    ndcg_scores.append(
        ndcg(
            retrieved,
            relevant
        )
    )

    results.append({

        "question": question,

        "ground_truth": gt_chunk,

        "retrieved_chunks": retrieved,

        "hit": gt_chunk in retrieved
    })
# ======================================
# FINAL RESULTS
# ======================================

print("\n" + "=" * 50)
print("RETRIEVAL EVALUATION")
print("=" * 50)

print(
    f"Questions Evaluated: "
    f"{len(qa_pairs)}"
)

print(
    f"Recall@5 : "
    f"{np.mean(recall_scores):.4f}"
)

print(
    f"MRR       : "
    f"{np.mean(mrr_scores):.4f}"
)

print(
    f"MAP       : "
    f"{np.mean(ap_scores):.4f}"
)

print(
    f"nDCG      : "
    f"{np.mean(ndcg_scores):.4f}"
)

# ======================================
# SAVE RESULTS
# ======================================

with open(
    "retrieval_results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )

print(
    "\nSaved detailed results to "
    "retrieval_results.json"
)

