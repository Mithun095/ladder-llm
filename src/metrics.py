from src.cascade import TraceStep
from src.registry import MAX_TIER, get_model

# ponytail: illustrative-only $/1M-tokens-per-active-billion-param rate, loosely anchored to
# published small-hosted-model pricing (~$0.15/1M tokens for an 8B-class model). These are
# free models with no real per-token bill — this exists to translate the abstract active-param
# savings into a unit a non-ML interviewer immediately understands. Upgrade to real per-model
# published rates if this number needs to hold up under scrutiny.
ILLUSTRATIVE_USD_PER_MILLION_TOKENS_PER_ACTIVE_B = 0.02
AVG_TOKENS_PER_ANSWER = 300  # rough blended input+output estimate for a typical response


def total_active_params_burned(trace: list[TraceStep]) -> float:
    return sum(step.active_params_b for step in trace if step.status != "unavailable")


def compute_saved_pct(trace: list[TraceStep], task_type: str) -> float:
    baseline = get_model(MAX_TIER, task_type).active_params_b
    used = total_active_params_burned(trace)
    return max(0.0, (baseline - used) / baseline * 100)


def estimate_dollar_cost(active_params_b: float) -> float:
    return active_params_b * ILLUSTRATIVE_USD_PER_MILLION_TOKENS_PER_ACTIVE_B * (AVG_TOKENS_PER_ANSWER / 1_000_000)


def estimate_dollar_saved(trace: list[TraceStep], task_type: str) -> float:
    baseline_b = get_model(MAX_TIER, task_type).active_params_b
    used_b = total_active_params_burned(trace)
    return estimate_dollar_cost(max(0.0, baseline_b - used_b))
