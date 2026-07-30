def format_answer(answer: str, task_type: str) -> str:
    if task_type == "coding" and "```" not in answer:
        return f"```\n{answer}\n```"
    if task_type == "translation" and "\n" in answer.strip():
        # Escaped outside the f-string on purpose: a backslash inside an f-string's {expression}
        # is a SyntaxError before Python 3.12, and CI runs 3.11 (see .github/workflows/checks.yml).
        lines = [line.replace("|", "\\|") for line in answer.splitlines() if line.strip()]
        rows = "\n".join(f"| {line} |" for line in lines)
        return f"| Translation |\n|---|\n{rows}"
    return answer
