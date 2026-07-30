def format_answer(answer: str, task_type: str) -> str:
    if task_type == "coding" and "```" not in answer:
        return f"```\n{answer}\n```"
    if task_type == "translation" and "\n" in answer.strip():
        lines = [line for line in answer.splitlines() if line.strip()]
        rows = "\n".join(f"| {line.replace('|', '\\|')} |" for line in lines)
        return f"| Translation |\n|---|\n{rows}"
    return answer
