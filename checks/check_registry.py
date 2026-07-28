from src.registry import MAX_TIER, TASK_TYPES, get_model

for tier in range(1, MAX_TIER + 1):
    for task_type in TASK_TYPES:
        config = get_model(tier, task_type)
        assert config.provider in ("groq", "openrouter")
        assert config.model_id
        assert config.active_params_b > 0

print(f"registry check passed: {MAX_TIER * len(TASK_TYPES)} pairs resolved.")
