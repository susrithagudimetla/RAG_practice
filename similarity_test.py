from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

s1 = "What is biology?"
s2 = "Study of living organisms"
s3 = "How to bake a cake"

e1 = model.encode(s1)
e2 = model.encode(s2)
e3 = model.encode(s3)

print("Biology similarity:",
      cosine_similarity([e1], [e2])[0][0])

print("Cake similarity:",
      cosine_similarity([e1], [e3])[0][0])