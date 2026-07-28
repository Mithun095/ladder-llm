from dataclasses import dataclass

TASK_TYPES = ("qa", "coding", "reasoning", "summarization", "translation")
MAX_TIER = 4


@dataclass(frozen=True)
class ModelConfig:
    provider: str  # "groq" | "openrouter"
    model_id: str
    active_params_b: float  # active params per token, not total (matters for MoE models)


# active_params_b sources:
# - dense models (llama, qwen, gemma-non-MoE): active == total.
# - nvidia nemotron "*-aNb" suffix names the active param count directly (e.g. a55b = 55B active).
# - gpt-oss-120b is a real MoE: ~117B total but only ~5.1B active/token — smaller active
#   footprint than the dense 70B tier-2 model below it. Deliberately left as-is (not swapped
#   for a "monotonically bigger" tier-3 model): it's an honest example of why this system
#   tracks active params instead of total params, not a bug to paper over.
# - poolside/cohere "mini"/"xs"/"m" sizes have no published param count; estimated from naming
#   (mini/xs ~7B, m ~32B) as a rough size-class approximation, not a cited figure.
REGISTRY: dict[tuple[int, str], ModelConfig] = {
    (1, "qa"): ModelConfig("groq", "llama-3.1-8b-instant", 8),
    (1, "coding"): ModelConfig("openrouter", "cohere/north-mini-code:free", 7),
    (1, "reasoning"): ModelConfig("groq", "llama-3.1-8b-instant", 8),
    (1, "summarization"): ModelConfig("groq", "llama-3.1-8b-instant", 8),
    (1, "translation"): ModelConfig("groq", "llama-3.1-8b-instant", 8),

    (2, "qa"): ModelConfig("groq", "qwen/qwen3.6-27b", 27),
    (2, "coding"): ModelConfig("openrouter", "poolside/laguna-xs-2.1:free", 7),
    (2, "reasoning"): ModelConfig("groq", "qwen/qwen3.6-27b", 27),
    # Was llama-3.3-70b-versatile (70B dense). Dropped: 70B active is *more* than the 55B-active
    # tier-4 baseline every saving is measured against, so escalating to it reported negative
    # savings — a tier-2 slot that costs more than the ceiling is a routing bug, not a tradeoff.
    (2, "summarization"): ModelConfig("groq", "qwen/qwen3.6-27b", 27),
    (2, "translation"): ModelConfig("openrouter", "google/gemma-4-26b-a4b-it:free", 4),

    (3, "qa"): ModelConfig("groq", "openai/gpt-oss-120b", 5.1),
    # Was poolside/laguna-m.1:free — delisted from OpenRouter's catalog mid-project and started
    # returning 404 "No endpoints found". Caught by checks/check_model_ids.py, which is why
    # that check validates the registry instead of just printing the catalog.
    (3, "coding"): ModelConfig("openrouter", "poolside/laguna-s-2.1:free", 14),
    (3, "reasoning"): ModelConfig("openrouter", "nvidia/nemotron-3-super-120b-a12b:free", 12),
    (3, "summarization"): ModelConfig("groq", "openai/gpt-oss-120b", 5.1),
    (3, "translation"): ModelConfig("groq", "openai/gpt-oss-120b", 5.1),

    (4, "qa"): ModelConfig("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free", 55),
    (4, "coding"): ModelConfig("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free", 55),
    (4, "reasoning"): ModelConfig("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free", 55),
    (4, "summarization"): ModelConfig("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free", 55),
    (4, "translation"): ModelConfig("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free", 55),
}


def get_model(tier: int, task_type: str) -> ModelConfig:
    config = REGISTRY.get((tier, task_type))
    if config is None:
        raise KeyError(f"No model registered for tier={tier}, type={task_type}")
    return config
