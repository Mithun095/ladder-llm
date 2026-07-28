from src.cascade import TraceStep
from src.registry import MAX_TIER, get_model


def total_active_params_burned(trace: list[TraceStep]) -> float:
    return sum(step.active_params_b for step in trace if step.status != "unavailable")


def compute_saved_pct(trace: list[TraceStep], task_type: str) -> float:
    baseline = get_model(MAX_TIER, task_type).active_params_b
    used = total_active_params_burned(trace)
    return max(0.0, (baseline - used) / baseline * 100)
