from rag_inference import (
    retrieve,
    get_chunk_texts,
    build_context
)

from prompt_builder import build_prompt
from generate_answers import generate_answer


def ask_question(question):

    # Retrieve chunks
    retrieved_ids = retrieve(
        question,
        top_k=5
    )

    # Convert ids to text
    retrieved_chunks = get_chunk_texts(
        retrieved_ids
    )

    # Build context
    context = build_context(
        retrieved_chunks
    )

    # Create prompt
    prompt = build_prompt(
        question,
        context
    )

    # Generate answer
    answer = generate_answer(
        prompt
    )

    return answer, retrieved_ids


def main():

    print("=" * 60)
    print("BBC News RAG Chat")
    print("Type 'exit' to quit")
    print("=" * 60)

    while True:

        question = input("\nYou : ")

        if question.lower() in ["exit", "quit"]:
            print("\nGoodbye!")
            break

        answer, retrieved = ask_question(question)

        print("\nRetrieved Chunks:")
        for chunk in retrieved:
            print("-", chunk)

        print("\nAssistant:")
        print(answer)


if __name__ == "__main__":
    main()