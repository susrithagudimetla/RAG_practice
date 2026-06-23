def build_prompt(
    question,
    context
):

    return f"""
You are a question-answering assistant.

Use ONLY the information contained
in the context below.

Do not use outside knowledge.

If the answer cannot be found in the context,
reply exactly:

I could not find the answer in the provided context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""