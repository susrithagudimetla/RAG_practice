from transformers import AutoTokenizer
import pandas as pd

df = pd.read_csv("BBCNews.csv")

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

token_counts = []

for article in df["descr"]:
    token_counts.append(
        len(tokenizer.tokenize(article))
    )

df["token_count"] = token_counts

print(df["token_count"].describe())