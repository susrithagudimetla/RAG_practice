# embedding_demo.py

import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

df = pd.read_csv("BBCNews.csv")

text = df.iloc[0]["descr"]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embedding = model.encode(chunks[0])

print(type(embedding))
print(embedding.shape)

print("\nFirst 20 values:\n")
print(embedding[:20])

embeddings = model.encode(chunks)
print(type(embeddings))
print(embeddings.shape)
