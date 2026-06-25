import json
import re
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from rouge_score import rouge_scorer

from rag_inference import retrieve
from rag_inference import get_chunk_texts
from rag_inference import build_context

from prompt_builder import build_prompt
from generate_answers import generate_answer
from resume_utils import resolve_resume_start_idx

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# LOAD DATASET
# ==========================================

with open(
    BASE_DIR / "qa_dataset_clean.json",
    "r",
    encoding="utf-8"
) as f:

    qa_pairs = json.load(f)

# ==========================================
# RESUME SUPPORT
# ==========================================

RESULTS_FILE = BASE_DIR / "generation_results.json"

if RESULTS_FILE.exists():

    print(
        f"\nFound existing {RESULTS_FILE}"
    )

    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        results = json.load(f)

    start_idx = resolve_resume_start_idx(
        results,
        qa_pairs
    )

    print(
        f"Resuming from question "
        f"{start_idx + 1}"
    )

else:

    results = []
    start_idx = 0

    print(
        "\nStarting fresh evaluation..."
    )

# ==========================================
# NORMALIZATION
# ==========================================

def normalize(text):

    if text is None:
        return ""

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    text = " ".join(text.split())

    return text

# ==========================================
# EXACT MATCH
# ==========================================

def exact_match(
    prediction,
    ground_truth
):

    return int(
        normalize(prediction)
        ==
        normalize(ground_truth)
    )

# ==========================================
# TOKEN F1
# ==========================================

def token_f1(
    prediction,
    ground_truth
):

    pred_tokens = normalize(
        prediction
    ).split()

    gt_tokens = normalize(
        ground_truth
    ).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0

    common = (
        set(pred_tokens)
        &
        set(gt_tokens)
    )

    if len(common) == 0:
        return 0

    precision = (
        len(common)
        /
        len(pred_tokens)
    )

    recall = (
        len(common)
        /
        len(gt_tokens)
    )

    return (
        2 * precision * recall
        /
        (precision + recall)
    )

# ==========================================
# ROUGE-L
# ==========================================

scorer = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True
)

def rouge_l(
    prediction,
    ground_truth
):

    score = scorer.score(
        ground_truth,
        prediction
    )

    return score["rougeL"].fmeasure

# ==========================================
# SEMANTIC SIMILARITY
# ==========================================

print(
    "\nLoading evaluation model..."
)

sim_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)

def semantic_similarity(
    prediction,
    ground_truth
):

    emb1 = sim_model.encode(
        [prediction]
    )

    emb2 = sim_model.encode(
        [ground_truth]
    )

    score = cosine_similarity(
        emb1,
        emb2
    )[0][0]

    return float(score)

# ==========================================
# MAIN LOOP
# ==========================================

for idx in tqdm(
    range(start_idx, len(qa_pairs))
):

    sample = qa_pairs[idx]

    question = sample["question"]

    gt_answer = sample["answer"]

    try:

        # -------------------------
        # RETRIEVE
        # -------------------------

        retrieved_ids = retrieve(
            question,
            top_k=5
        )

        retrieved_texts = (
            get_chunk_texts(
                retrieved_ids
            )
        )

        context = build_context(
            retrieved_texts
        )

        # -------------------------
        # PROMPT
        # -------------------------

        prompt = build_prompt(
            question,
            context
        )

        # -------------------------
        # GENERATE
        # -------------------------
        print(
            f"\nGenerating answer for "
            f"Question {idx + 1}/{len(qa_pairs)}"
        )
        prediction = generate_answer(
            prompt
        )

        # -------------------------
        # EVALUATION
        # -------------------------

        em = exact_match(
            prediction,
            gt_answer
        )

        f1 = token_f1(
            prediction,
            gt_answer
        )

        rouge = rouge_l(
            prediction,
            gt_answer
        )

        sem = semantic_similarity(
            prediction,
            gt_answer
        )

        results.append({

            "question": question,
            "question_index": idx,

            "ground_truth": gt_answer,

            "prediction": prediction,

            "exact_match": em,

            "f1": f1,

            "rougeL": rouge,

            "semantic_similarity": sem
        })

        print(
            f"\nCompleted Question "
            f"{idx + 1}/{len(qa_pairs)}"
        )

        # -------------------------
        # SAVE EVERY 5 QUESTIONS
        # -------------------------

        if (idx + 1) % 5 == 0:

            with open(
                RESULTS_FILE,
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
                "Checkpoint saved."
            )

      

    except Exception as e:

        print(
            f"\nError at question "
            f"{idx + 1}"
        )

        print(e)

        continue

# ==========================================
# FINAL SAVE
# ==========================================

with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )

# ==========================================
# FINAL METRICS
# ==========================================

em_scores = [
    r["exact_match"]
    for r in results
]

f1_scores = [
    r["f1"]
    for r in results
]

rouge_scores = [
    r["rougeL"]
    for r in results
]

semantic_scores = [
    r["semantic_similarity"]
    for r in results
]

print("\n")
print("=" * 60)
print("RAG GENERATION EVALUATION")
print("=" * 60)

print(
    f"Questions: {len(results)}"
)

print(
    f"Exact Match: "
    f"{np.mean(em_scores):.4f}"
)

print(
    f"F1 Score: "
    f"{np.mean(f1_scores):.4f}"
)

print(
    f"ROUGE-L: "
    f"{np.mean(rouge_scores):.4f}"
)

print(
    f"Semantic Similarity: "
    f"{np.mean(semantic_scores):.4f}"
)

# ==========================================
# SAVE CSV
# ==========================================

pd.DataFrame(
    results
).to_csv(
    "generation_results.csv",
    index=False
)

print(
    "\nSaved:"
)

print(
    "- generation_results.json"
)

print(
    "- generation_results.csv"
)