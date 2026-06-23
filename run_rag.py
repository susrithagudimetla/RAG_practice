from rag_inference import retrieve
from rag_inference import get_chunk_texts
from rag_inference import build_context

from prompt_builder import build_prompt
from generate_answers import generate_answer

question = (
    "Who backed Chelsea's decision "
    "to sack Adrian Mutu?"
)

# Retrieve chunk ids

retrieved_ids = retrieve(
    question,
    top_k=5
)

# Convert ids to text

retrieved_texts = get_chunk_texts(
    retrieved_ids
)

# Build context

context = build_context(
    retrieved_texts
)

# Build prompt

prompt = build_prompt(
    question,
    context
)

# Generate answer

answer = generate_answer(
    prompt
)

print("\nQUESTION:")
print(question)

print("\nANSWER:")
print(answer)