import time
from dataclasses import dataclass, field, replace
from typing import Literal

from pydantic import BaseModel, Field

from src.classifier import classify
from src.judge import judge
from src.llm_client import ModelUnavailable, call_model
from src.registry import get_model

ANSWER_SYSTEM_PROMPT = """Complete the user's request. Respond with ONLY this JSON shape, no \
other text:
{"answer": "<your answer>", "confidence": <integer 1-10, your honest confidence this answer is correct>}"""

# Prompt rewriting is only safe when the query is *purely an instruction*. Summarization and
# translation queries carry a payload — the text to be summarized or translated — and the
# classifier reliably paraphrases that payload away ("Summarize: <200 words>" became
# "Summarize the main points from the paragraph about the stock market", with no paragraph
# left to summarize). The downstream model then answers from world knowledge instead of the
# supplied text. For these types the raw query is sent through untouched.
PRESERVE_QUERY_TYPES = {"summarization", "translation"}

# Where a query enters the ladder. Starting above tier 1 is a bet that the lower tiers are too
# weak to be worth a call — it is NOT a cost decision, because the price ladder is monotonic, so
# every skipped tier is strictly cheaper than the one you land on.
#
# `expert` used to start at tier 3, which was correct when tier 3 held the largest model. After
# the tier 2/3 swap in registry.py that stopped being true: tier 3 is now the *dense, expensive*
# slot and tier 2 holds the sparse 120B model. So expert queries were skipping the strongest
# cheap model on the ladder to open at one costing ~11x more per query, and paying that on every
# expert query whether or not it needed it. The "too weak to bother with" argument does not
# apply to a 120B MoE.
#
# Expert now enters at tier 2 alongside hard. The two are still distinct: they differ in how far
# they may climb (CEILING_TIER 4 vs 3), which is the part difficulty should actually control.
STARTING_TIER = {"easy": 1, "medium": 1, "hard": 2, "expert": 2}

# The ceiling bounds worst-case spend: it's what stops a query the judge keeps rejecting from
# climbing to the 550B model and burning the entire saving. The catch is that it's set from the
# classifier's difficulty guess, made before any answer exists, and no amount of contrary
# evidence lifts it — so an easy/medium query that fails twice stops at tier 2 with tiers 3 and 4
# unused. That looked like the bug behind "no tier produced an answer" reports, and the obvious
# fix was to raise it.
#
# It wasn't, and raising it would have been the expensive wrong fix. The real problem was that
# tier 2 held a 27B dense model costing 12x more per output token than the 120B sparse model
# sitting unreachable at tier 3 (see registry.py). Reordering those two means the ceiling now
# stops at the strongest cheap model in the ladder instead of below it — a better answer for
# less money, without touching this dict. Kept as-is deliberately.
#
# ponytail: static ceiling from a pre-answer guess. Upgrade path if the tail ever matters:
# re-classify difficulty using the judge's rejection reasons and lift the ceiling on evidence.
# Not done, because on this benchmark the effect is smaller than the run-to-run noise floor
# (two identical sweeps scored 72% and 84%), so it cannot currently be measured. See BUILD-LOG #21.
CEILING_TIER = {"easy": 2, "medium": 2, "hard": 3, "expert": 4}

# Live testing showed self-reported confidence landing at 9-10 on every call regardless of
# answer correctness — measured later as an ECE of 0.23-0.52, with the top confidence bucket
# claiming ~0.98 and delivering ~0.75. The ambiguous 5-7 band the fast paths depend on is
# never reached in practice, so both shortcuts are disabled and the judge fires on every
# answer. Kept behind a flag rather than deleted: see BUILD-LOG.md #7.
JUDGE_ALWAYS = True

# ponytail: exact-match in-process cache — normalized whitespace/case, nothing smarter.
# A repeat query costs zero LLM calls, which is the single largest saving available here
# (the classifier alone is a call the cascade otherwise always pays). Upgrade path if
# repeat-rate ever matters: embed the query and match on cosine similarity, backed by a
# persistent store, so paraphrases hit too. Cleared per process; not shared across users.
_CACHE: dict[str, "CascadeResult"] = {}


class AnswerResult(BaseModel):
    answer: str
    confidence: int = Field(ge=1, le=10)


@dataclass
class TraceStep:
    tier: int
    model_id: str
    # "accepted_unverified" is deliberately distinct from "accepted": the model answered and the
    # judge never got to look at it (rate-limited or unparseable). Showing that answer to a user
    # is reasonable; scoring it as a benchmark pass is not, because then a sweep that loses judge
    # calls to a rate limit reports a HIGHER pass rate than one that measures everything.
    status: Literal["accepted", "accepted_unverified", "escalated", "judged_fail",
                    "unavailable", "malformed_response"]
    confidence: int | None = None
    judge_reason: str | None = None
    active_params_b: float = 0
    elapsed_ms: int = 0


@dataclass
class CascadeResult:
    answer: str
    trace: list[TraceStep] = field(default_factory=list)
    tier_used: int = 0
    type: str = "qa"
    difficulty: str = "medium"
    optimized_prompt: str = ""
    elapsed_ms: int = 0
    cached: bool = False

    @property
    def accepted(self) -> bool:
        """Did this run deliver an answer the cascade stands behind enough to show?

        True for a verified pass AND for one the judge never got to see. Use this for anything
        user-facing: an unverified answer is still the best thing available, and withholding it
        because the judge was rate-limited helps nobody.

        Lives here because every caller needs it and two of them previously each rolled their
        own version — the UI and the eval harness both tested `answer != "<sentinel>"`, which
        asks "is there a string?" not "did this work". A judge-rejected answer is still a
        string, so both reported compute savings on runs that delivered nothing usable.
        """
        return bool(self.trace) and self.trace[-1].status in ("accepted", "accepted_unverified")

    @property
    def verified(self) -> bool:
        """Did the judge actually look at this answer and pass it?

        Use this, not `accepted`, for any MEASUREMENT. Counting an unjudged answer as a pass
        means a sweep that loses judge calls to a rate limit scores higher than one that checks
        every answer — a metric that improves the less data it collects (BUILD-LOG #24).
        """
        return bool(self.trace) and self.trace[-1].status == "accepted"


def _cache_key(query: str) -> str:
    return " ".join(query.lower().split())


def run_cascade(query: str, use_cache: bool = True) -> CascadeResult:
    started = time.perf_counter()
    key = _cache_key(query)
    if use_cache and key in _CACHE:
        # The trace is kept as-is so the UI can still show how this answer was originally
        # reached; `cached` is what tells the caller no models ran *this* time.
        return replace(_CACHE[key], cached=True, elapsed_ms=0)

    classification = classify(query)
    prompt = query if classification.type in PRESERVE_QUERY_TYPES else classification.optimized_prompt
    trace: list[TraceStep] = []
    tier = STARTING_TIER[classification.difficulty]
    ceiling = CEILING_TIER[classification.difficulty]
    last_answer = None

    while tier <= ceiling:
        model_config = get_model(tier, classification.type)
        step_started = time.perf_counter()
        try:
            result = call_model(model_config, ANSWER_SYSTEM_PROMPT, prompt, AnswerResult)
        except ModelUnavailable:
            trace.append(TraceStep(tier, model_config.model_id, "unavailable",
                                   elapsed_ms=_ms_since(step_started)))
            tier += 1
            continue

        if result is None:
            # The model still ran and generated tokens (compute was spent) even though the
            # output didn't parse as valid JSON — unlike "unavailable", this isn't free.
            trace.append(TraceStep(tier, model_config.model_id, "malformed_response",
                                   active_params_b=model_config.active_params_b,
                                   elapsed_ms=_ms_since(step_started)))
            tier += 1
            continue

        last_answer = result.answer

        if not JUDGE_ALWAYS:
            if result.confidence >= 8:
                trace.append(TraceStep(tier, model_config.model_id, "accepted", result.confidence,
                                       active_params_b=model_config.active_params_b,
                                       elapsed_ms=_ms_since(step_started)))
                break
            if result.confidence <= 4:
                trace.append(TraceStep(tier, model_config.model_id, "escalated", result.confidence,
                                       active_params_b=model_config.active_params_b,
                                       elapsed_ms=_ms_since(step_started)))
                tier += 1
                continue

        verdict = judge(query, result.answer, classification.type)
        if verdict is None:
            # The judge itself is down or unparseable. Escalating would be pure waste — the
            # next tier's answer would hit the same broken judge and end up here again, one
            # tier more expensive. Accept, but say plainly in the trace that it's unverified.
            status, reason = "accepted_unverified", "accepted unverified — judge unavailable"
        else:
            status = "accepted" if verdict.verdict == "pass" else "judged_fail"
            reason = verdict.reason
        trace.append(TraceStep(tier, model_config.model_id, status, result.confidence, reason,
                               model_config.active_params_b, _ms_since(step_started)))
        if status in ("accepted", "accepted_unverified"):
            break
        tier += 1

    final_tier = trace[-1].tier if trace else tier
    result = CascadeResult(
        answer=last_answer or "No model produced a usable answer.",
        trace=trace,
        tier_used=final_tier,
        type=classification.type,
        difficulty=classification.difficulty,
        optimized_prompt=prompt,
        elapsed_ms=_ms_since(started),
    )
    # Cache accepted runs only. This used to be `if last_answer:`, which is the same
    # "is there a string?" test that shipped as a bug three times already (BUILD-LOG #11, #16,
    # #19) — a judge-rejected answer is still a string, so failures were being cached too.
    #
    # It matters more here than elsewhere. The judge is non-deterministic enough to flip roughly
    # one verdict in five between identical runs (#21), so a query that failed once would often
    # pass on a retry — except the cache served it the stored failure forever, and no retry could
    # ever happen. Caching a failure converts a transient wrong verdict into a permanent one.
    # `verified`, not `accepted`: an answer the judge never saw hasn't been checked, and caching
    # it would freeze an unchecked result in place for the life of the process.
    if use_cache and result.verified:
        _CACHE[key] = result
    return result


def _ms_since(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
