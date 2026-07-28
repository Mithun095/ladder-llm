from src.classifier import classify

samples = [
    ("What is a closure in Python?", "qa"),
    ("Write a function to reverse a linked list in O(1) space", "coding"),
    ("Prove that the square root of 2 is irrational", "reasoning"),
    ("Summarize: The quick brown fox jumps over the lazy dog repeatedly.", "summarization"),
    ("Translate 'good morning' to French", "translation"),
]

for query, expected_type in samples:
    result = classify(query)
    print(f"{query!r} -> difficulty={result.difficulty}, type={result.type}")
    assert result.type == expected_type, f"expected type={expected_type}, got {result.type}"
    assert result.difficulty in ("easy", "medium", "hard", "expert")
    assert result.optimized_prompt

print("classifier check passed.")
