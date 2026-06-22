from transformers import AutoTokenizer
import pandas as pd

df = pd.read_csv("BBCNews.csv")

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

sample_text = df.iloc[0]["descr"]

tokens = tokenizer.tokenize(sample_text)

print("Character Count:", len(sample_text))
print("Token Count:", len(tokens))

print("\nFirst 100 Tokens:\n")
print(tokens[:100])