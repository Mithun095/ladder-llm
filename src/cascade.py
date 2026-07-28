from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from src.classifier import classify
from src.judge import judge
from src.llm_client import ModelUnavailable, call_model
from src.registry import get_model

ANSWER_SYSTEM_PROMPT = """Answer the user's question. Respond with ONLY this JSON shape, no \
other text:
{"answer": "<your answer>", "confidence": <integer 1-10, your honest confidence this answer is correct>}"""

STARTING_TIER = {"easy": 1, "medium": 1, "hard": 2, "expert": 3}
CEILING_TIER = {"easy": 2, "medium": 2, "hard": 3, "expert": 4}


class AnswerResult(BaseModel):
    answer: str
    confidence: int = Field(ge=1, le=10)


@dataclass
class TraceStep:
    tier: int
    model_id: str
    status: Literal["accepted", "escalated", "judged_fail", "unavailable", "malformed_response"]
    confidence: int | None = None
    judge_reason: str | None = None
    active_params_b: float = 0


@dataclass
class CascadeResult:
    answer: str
    trace: list[TraceStep] = field(default_factory=list)
    tier_used: int = 0
    type: str = "qa"


def run_cascade(query: str) -> CascadeResult:
    classification = classify(query)
    trace: list[TraceStep] = []
    tier = STARTING_TIER[classification.difficulty]
    ceiling = CEILING_TIER[classification.difficulty]
    last_answer = None

    while tier <= ceiling:
        model_config = get_model(tier, classification.type)
        try:
            result = call_model(model_config, ANSWER_SYSTEM_PROMPT, classification.optimized_prompt, AnswerResult)
        except ModelUnavailable:
            trace.append(TraceStep(tier, model_config.model_id, "unavailable"))
            tier += 1
            continue

        if result is None:
            trace.append(TraceStep(tier, model_config.model_id, "malformed_response"))
            tier += 1
            continue

        last_answer = result.answer

        if result.confidence >= 8:
            trace.append(TraceStep(tier, model_config.model_id, "accepted", result.confidence, active_params_b=model_config.active_params_b))
            break
        if result.confidence <= 4:
            trace.append(TraceStep(tier, model_config.model_id, "escalated", result.confidence, active_params_b=model_config.active_params_b))
            tier += 1
            continue

        verdict = judge(query, result.answer)
        if verdict is not None and verdict.verdict == "pass":
            trace.append(TraceStep(tier, model_config.model_id, "accepted", result.confidence, verdict.reason, model_config.active_params_b))
            break
        reason = verdict.reason if verdict is not None else "judge call failed"
        trace.append(TraceStep(tier, model_config.model_id, "judged_fail", result.confidence, reason, model_config.active_params_b))
        tier += 1

    final_tier = trace[-1].tier if trace else tier
    return CascadeResult(
        answer=last_answer or "No model produced a usable answer.",
        trace=trace,
        tier_used=final_tier,
        type=classification.type,
    )
