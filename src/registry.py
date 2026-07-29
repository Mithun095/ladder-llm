from dataclasses import dataclass

TASK_TYPES = ("qa", "coding", "reasoning", "summarization", "translation")
MAX_TIER = 4


@dataclass(frozen=True)
class ModelConfig:
    provider: str  # "groq" | "openrouter"
    model_id: str
    active_params_b: float  # active params per token — what a token actually costs to compute
    total_params_b: float   # all weights — a rough proxy for how much the model *knows*

    # Published $/1M tokens for the *paid* listing of the same open-weight model, taken from
    # OpenRouter's catalog (see checks/check_model_ids.py). Every model actually called here is
    # a free endpoint, so this is a provider-neutral market proxy for what the compute is worth
    # — not a bill anyone pays. None for models with no paid listing anywhere (the coding tiers),
    # which fall back to the active-param estimate in metrics.py.
    usd_per_m_in: float | None = None
    usd_per_m_out: float | None = None


# --- Why the tiers are ordered the way they are -------------------------------------------
#
# active_params_b sources:
# - dense models (llama, qwen, gemma-non-MoE): active == total.
# - nvidia nemotron "*-aNb" suffix names the active param count directly (e.g. a55b = 55B active).
# - gpt-oss-120b is a real MoE: ~117B total, ~5.1B active/token.
# - poolside/cohere "mini"/"xs"/"s" sizes have no published param count; estimated from naming
#   (mini/xs ~7B, s ~14B) as a rough size-class approximation, not a cited figure.
#
# Tiers 2 and 3 used to be the other way round for qa/reasoning/summarization/translation, on the
# assumption that a bigger *total* param count means a more expensive tier. Checking published
# per-token rates showed that is badly wrong for sparse models — the ladder was inverted:
#
#   qwen3.6-27b   (27B active, dense)   $0.300 in / $2.000 out per 1M
#   gpt-oss-120b  (5.1B active, MoE)    $0.037 in / $0.170 out per 1M   <- 8x/12x CHEAPER
#
# So the old tier 3 was ~12x cheaper per output token than the old tier 2 it escalated *from*,
# while also being the larger, generally stronger model. Combined with CEILING_TIER, which stops
# easy/medium queries at tier 2, the cascade was structurally incapable of reaching a model that
# is simultaneously better AND cheaper. Swapping them makes the ladder monotonic in real price
# and fixes the ceiling without having to raise it. See BUILD-LOG.md #20.
#
# Note the two cost metrics disagree on tier 1 vs tier 2: llama-3.1-8b has more active params
# (8B vs 5.1B) but is cheaper per blended token than gpt-oss-120b. Active params are a proxy for
# compute; price is a proxy for what the market charges for it, and they are not the same
# ordering. The ladder is ordered by price, and metrics.py reports both.
REGISTRY: dict[tuple[int, str], ModelConfig] = {
    (1, "qa"): ModelConfig("groq", "llama-3.1-8b-instant", 8, 8, 0.050, 0.080),
    (1, "coding"): ModelConfig("openrouter", "cohere/north-mini-code:free", 7, 7),
    (1, "reasoning"): ModelConfig("groq", "llama-3.1-8b-instant", 8, 8, 0.050, 0.080),
    (1, "summarization"): ModelConfig("groq", "llama-3.1-8b-instant", 8, 8, 0.050, 0.080),
    (1, "translation"): ModelConfig("groq", "llama-3.1-8b-instant", 8, 8, 0.050, 0.080),

    # Tier 2 is the sparse-MoE slot: large total parameter count, small active footprint, and
    # the cheapest published rate in the ladder. This is where most escalations now land.
    (2, "qa"): ModelConfig("groq", "openai/gpt-oss-120b", 5.1, 117, 0.037, 0.170),
    (2, "coding"): ModelConfig("openrouter", "poolside/laguna-xs-2.1:free", 7, 7),
    (2, "reasoning"): ModelConfig("groq", "openai/gpt-oss-120b", 5.1, 117, 0.037, 0.170),
    (2, "summarization"): ModelConfig("groq", "openai/gpt-oss-120b", 5.1, 117, 0.037, 0.170),
    (2, "translation"): ModelConfig("groq", "openai/gpt-oss-120b", 5.1, 117, 0.037, 0.170),

    # Tier 3 is the dense-model slot: fewer total params than tier 2 but every one of them active
    # on every token, which is why it costs several times more per token despite being "smaller".
    (3, "qa"): ModelConfig("groq", "qwen/qwen3.6-27b", 27, 27, 0.300, 2.000),
    # Was poolside/laguna-m.1:free — delisted from OpenRouter's catalog mid-project and started
    # returning 404 "No endpoints found". Caught by checks/check_model_ids.py, which is why
    # that check validates the registry instead of just printing the catalog.
    (3, "coding"): ModelConfig("openrouter", "poolside/laguna-s-2.1:free", 14, 14),
    (3, "reasoning"): ModelConfig("openrouter", "nvidia/nemotron-3-super-120b-a12b:free", 12, 120, 0.085, 0.400),
    (3, "summarization"): ModelConfig("groq", "qwen/qwen3.6-27b", 27, 27, 0.300, 2.000),
    (3, "translation"): ModelConfig("openrouter", "google/gemma-4-26b-a4b-it:free", 4, 26, 0.070, 0.340),

    (4, "qa"): ModelConfig("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free", 55, 550, 0.500, 2.200),
    (4, "coding"): ModelConfig("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free", 55, 550, 0.500, 2.200),
    (4, "reasoning"): ModelConfig("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free", 55, 550, 0.500, 2.200),
    (4, "summarization"): ModelConfig("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free", 55, 550, 0.500, 2.200),
    (4, "translation"): ModelConfig("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free", 55, 550, 0.500, 2.200),
}


def get_model(tier: int, task_type: str) -> ModelConfig:
    config = REGISTRY.get((tier, task_type))
    if config is None:
        raise KeyError(f"No model registered for tier={tier}, type={task_type}")
    return config
