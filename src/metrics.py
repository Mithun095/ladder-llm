from src.cascade import TraceStep
from src.registry import MAX_TIER, ModelConfig, get_model

# A typical exchange here: a short query plus system prompt in, a paragraph-ish answer out.
# Split explicitly rather than as one blended number, because input and output are priced very
# differently (often 5-10x apart) and a single blended figure quietly picks a ratio for you.
AVG_TOKENS_IN = 100
AVG_TOKENS_OUT = 200

# Fallback only, for the coding-tier models that have no paid listing on any provider and so no
# published rate to cite. Loosely anchored to small-hosted-model pricing. Every model that *does*
# have a published rate carries it on its ModelConfig instead — this constant used to be applied
# to all of them, which ranked a 5.1B-active MoE as 5x more expensive than a 27B dense model when
# the real rates say the opposite by 12x. See BUILD-LOG.md #20.
FALLBACK_USD_PER_M_PER_ACTIVE_B = 0.02


def total_active_params_burned(trace: list[TraceStep]) -> float:
    return sum(step.active_params_b for step in trace if step.status != "unavailable")


def compute_saved_pct(trace: list[TraceStep], task_type: str) -> float:
    """Deliberately NOT clamped at 0. A cascade that escalates through several tiers can burn
    more active params than one direct max-tier call would have (e.g. expert-difficulty coding:
    14B at tier 3 + 55B at tier 4 = 69B vs. a 55B baseline). Clamping that to "0% saved" would
    hide the routing's genuine worst case; showing -25% is the honest number."""
    baseline = get_model(MAX_TIER, task_type).active_params_b
    used = total_active_params_burned(trace)
    return (baseline - used) / baseline * 100


def model_cost_usd(config: ModelConfig) -> float:
    """Cost of one average exchange with this model, at published $/1M-token rates.

    These are free endpoints, so nothing is actually billed — this converts the routing decision
    into the unit an interviewer (or a finance team) actually reasons in.
    """
    if config.usd_per_m_in is None or config.usd_per_m_out is None:
        rate = config.active_params_b * FALLBACK_USD_PER_M_PER_ACTIVE_B
        return rate * (AVG_TOKENS_IN + AVG_TOKENS_OUT) / 1_000_000
    return (config.usd_per_m_in * AVG_TOKENS_IN
            + config.usd_per_m_out * AVG_TOKENS_OUT) / 1_000_000


def trace_cost_usd(trace: list[TraceStep], task_type: str) -> float:
    """What this cascade run cost, summed over the tiers that actually ran.

    Keyed off the trace's own tier numbers rather than its model IDs so a registry swap can't
    silently mis-price a historical trace.
    """
    return sum(
        model_cost_usd(get_model(step.tier, task_type))
        for step in trace
        if step.status != "unavailable"
    )


def estimate_dollar_saved(trace: list[TraceStep], task_type: str) -> float:
    """Also unclamped, for the same reason as compute_saved_pct — a negative return means the
    cascade cost more than the always-max-tier baseline on this query."""
    baseline = model_cost_usd(get_model(MAX_TIER, task_type))
    return baseline - trace_cost_usd(trace, task_type)


def dollar_saved_pct(trace: list[TraceStep], task_type: str) -> float:
    """Savings in money rather than in active params. Reported alongside compute_saved_pct
    because the two metrics genuinely disagree — a sparse MoE can be cheaper per token than a
    model with fewer active params, so "compute saved" and "cost saved" are not the same claim."""
    baseline = model_cost_usd(get_model(MAX_TIER, task_type))
    if baseline == 0:
        return 0.0
    return (baseline - trace_cost_usd(trace, task_type)) / baseline * 100
