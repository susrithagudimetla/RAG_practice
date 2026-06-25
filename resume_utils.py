def resolve_resume_start_idx(results, qa_pairs):
    if not results:
        return 0

    last_index = None
    for item in reversed(results):
        question_index = item.get("question_index")
        if isinstance(question_index, int):
            last_index = question_index
            break

    if last_index is None:
        return len(results)

    if last_index + 1 < len(qa_pairs):
        return last_index + 1

    return len(qa_pairs)
