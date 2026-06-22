import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

df = pd.read_csv("BBCNews.csv")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

all_chunks = []

for idx, row in df.iterrows():

    article = row["descr"]

    chunks = splitter.split_text(article)

    for chunk_id, chunk in enumerate(chunks):

        all_chunks.append({
            "article_id": int(idx),
            "chunk_id": int(chunk_id),
            "chunk_text": chunk,
            "tags": row["tags"]
        })

with open("bbc_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=4, ensure_ascii=False)

print(f"Saved {len(all_chunks)} chunks")

print("Articles:", len(df))
print("Chunks:", len(all_chunks))
print(
    "Average chunks/article:",
    len(all_chunks) / len(df)
)

print(json.dumps(all_chunks[0], indent=4))