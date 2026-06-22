from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "What is biology?"
embedding = model.encode(text)

print("Embedding shape:", embedding.shape)
print("First 10 values:", embedding[:10])