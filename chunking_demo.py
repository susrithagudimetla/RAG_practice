import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

df = pd.read_csv("BBCNews.csv")

text = df.iloc[0]["descr"]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

print("Number of chunks:", len(chunks))

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

for i, chunk in enumerate(chunks):

    token_count = len(
        tokenizer.tokenize(chunk)
    )

    print(
        f"Chunk {i}: "
        f"{len(chunk)} chars, "
        f"{token_count} tokens"
    )