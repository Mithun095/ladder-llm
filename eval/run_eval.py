import json

from eval.benchmark_set import BENCHMARK_QUERIES
from eval.calibration import compute_ece, reliability_table
from src.cascade import ANSWER_SYSTEM_PROMPT, AnswerResult, run_cascade
from src.judge import judge
from src.llm_client import ModelUnavailable, call_model
from src.metrics import compute_saved_pct, dollar_saved_pct, estimate_dollar_saved
from src.registry import MAX_TIER, get_model


def run_baseline(query: str, task_type: str):
    """Always-max-tier baseline: what you'd get from just calling the biggest model
    with the raw query, no classification, no prompt optimization, no escalation."""
    model_config = get_model(MAX_TIER, task_type)
    try:
        result = call_model(model_config, ANSWER_SYSTEM_PROMPT, query, AnswerResult)
    except ModelUnavailable:
        return None
    if result is None:
        return None
    verdict = judge(query, result.answer, task_type)
    return {"passed": verdict is not None and verdict.verdict == "pass"}


def main():
    cascade_pass = 0
    baseline_pass = 0
    baseline_ran = 0
    answered_count = 0
    unverified = 0
    total_saved_pct = 0.0
    total_cost_saved_pct = 0.0
    total_dollar_saved = 0.0
    confidence_verdict_pairs: list[tuple[int, bool]] = []
    per_query_rows = []

    skipped = []
    for i, query in enumerate(BENCHMARK_QUERIES, 1):
        try:
            # Cache off: an eval number that silently benefits from a cache hit isn't
            # measuring the router, and every benchmark query is distinct anyway.
            result = run_cascade(query, use_cache=False)

            # `verified`, not `accepted`. When the judge is rate-limited the cascade still
            # returns its best answer and marks it accepted-but-unverified, which is right for a
            # user and wrong for a benchmark: counting it as a pass means a sweep that loses
            # judge calls scores HIGHER than one that checks everything.
            cascade_ok = result.verified
            cascade_pass += cascade_ok
            unverified += result.accepted and not result.verified

            for step in result.trace:
                # `accepted_unverified` is EXCLUDED on purpose, not by oversight. Calibration
                # pairs a stated confidence with whether the answer was actually right; for an
                # answer the judge never saw, that second term is unknown. Including it would
                # mean guessing the outcome, and guessing "correct" would make the model look
                # better calibrated the more often the judge was rate-limited.
                if step.confidence is not None and step.status in ("accepted", "judged_fail"):
                    confidence_verdict_pairs.append((step.confidence, step.status == "accepted"))

            # Savings are averaged over accepted runs only, not merely over runs where some
            # model emitted text. A query that burns two tiers and gets both rejected delivered
            # nothing, so crediting it with "36% saved vs. max tier" inflates the headline with
            # the cost of failures. `answered` is still tracked separately, because "no model
            # was reachable" and "models ran and were wrong" are different results and
            # collapsing them would hide provider outages inside the pass rate.
            saved_pct = compute_saved_pct(result.trace, result.type)
            cost_saved_pct = dollar_saved_pct(result.trace, result.type)
            dollar_saved = estimate_dollar_saved(result.trace, result.type)
            answered = result.answer != "No model produced a usable answer."
            answered_count += answered
            if cascade_ok:
                total_saved_pct += saved_pct
                total_cost_saved_pct += cost_saved_pct
                total_dollar_saved += dollar_saved

            # `None` means the baseline model was unavailable, not that it answered wrongly.
            # Scoring those as failures would credit the cascade for a provider outage.
            baseline = run_baseline(query, result.type)
            baseline_ok = bool(baseline and baseline["passed"])
            if baseline is not None:
                baseline_ran += 1
                baseline_pass += baseline_ok

            per_query_rows.append({
                "query": query,
                "type": result.type,
                "difficulty": result.difficulty,
                "tier_used": result.tier_used,
                "cascade_passed": cascade_ok,
                # Recorded separately because `baseline_passed: false` on its own conflates "the
                # tier-4 model ran and got it wrong" with "the tier-4 model was never reachable",
                # which makes a paired cascade-vs-baseline comparison impossible to reconstruct
                # from the JSON afterwards. Same conflation as BUILD-LOG #19, one layer out.
                "baseline_ran": baseline is not None,
                "baseline_passed": baseline_ok if baseline is not None else None,
                "answered": answered,
                "verified": result.verified,
                "compute_saved_pct": round(saved_pct, 1) if cascade_ok else None,
                "cost_saved_pct": round(cost_saved_pct, 1) if cascade_ok else None,
            })
            print(f"[{i}/{len(BENCHMARK_QUERIES)}] {result.type:14s} {result.difficulty:6s} "
                  f"tier={result.tier_used} "
                  f"cascade={'pass' if cascade_ok else 'fail'} "
                  f"baseline={'pass' if baseline_ok else ('fail' if baseline is not None else 'n/a')} "
                  f"saved={f'{saved_pct:.0f}%' if cascade_ok else 'n/a (not accepted)'}")
        except Exception as e:
            # One unexpected failure (a new provider quirk we haven't hardened against yet)
            # shouldn't throw away every query already computed in this sweep.
            skipped.append({"query": query, "error": str(e)})
            print(f"[{i}/{len(BENCHMARK_QUERIES)}] SKIPPED — {type(e).__name__}: {e}")

    n = len(BENCHMARK_QUERIES) - len(skipped)
    if n == 0:
        print("Every query failed — nothing to report.")
        return
    ece = compute_ece(confidence_verdict_pairs)

    summary = {
        "num_queries": n,
        "num_answered": answered_count,
        # Answers the judge never saw (rate-limited). Excluded from the pass rate on purpose;
        # reported so a sweep degraded by rate limits is visibly degraded rather than flattered.
        "num_unverified": unverified,
        "cascade_pass_rate": cascade_pass / n,
        "baseline_ran": baseline_ran,
        "baseline_pass_rate": baseline_pass / baseline_ran if baseline_ran else None,
        "avg_compute_saved_pct": total_saved_pct / cascade_pass if cascade_pass else 0.0,
        # Reported alongside, not instead: active params measure compute, published rates measure
        # what that compute costs, and for sparse MoE models the two disagree — see registry.py.
        "avg_cost_saved_pct": total_cost_saved_pct / cascade_pass if cascade_pass else 0.0,
        "total_dollar_saved_illustrative": round(total_dollar_saved, 4),
        "confidence_calibration_ece": round(ece, 3),
        "reliability_table": reliability_table(confidence_verdict_pairs),
        "per_query": per_query_rows,
        "skipped": skipped,
    }

    print("\n=== Summary ===")
    print(f"Cascade pass rate:              {summary['cascade_pass_rate'] * 100:.1f}% "
          f"(judge-verified only)")
    if unverified:
        print(f"  ...plus {unverified} answer(s) the judge never saw (rate-limited). NOT counted "
              f"as passes — an unchecked answer is not a verified one.")
    if baseline_ran:
        print(f"Baseline (tier {MAX_TIER}) pass rate:      {summary['baseline_pass_rate'] * 100:.1f}% "
              f"(over the {baseline_ran} queries where the tier-{MAX_TIER} model was reachable)")
    else:
        print(f"Baseline (tier {MAX_TIER}) pass rate:      n/a — the tier-{MAX_TIER} model was "
              f"unavailable for every query, so there is no comparison to make this run.")
    print(f"Avg compute saved (active params): {summary['avg_compute_saved_pct']:.1f}% "
          f"(over the {cascade_pass} queries the judge accepted; {answered_count} produced text at all)")
    print(f"Avg cost saved ($/token rates):    {summary['avg_cost_saved_pct']:.1f}%")
    # `cascade_pass`, not `n`: this total accumulates only under `if cascade_ok`, so labelling it
    # "across 25 queries" attributes savings to three runs that contributed nothing to it.
    print(f"Illustrative $ saved:           ${summary['total_dollar_saved_illustrative']:.4f} "
          f"across the {cascade_pass} accepted queries")
    print(f"Confidence ECE:                 {summary['confidence_calibration_ece']:.3f} "
          f"(0=perfectly calibrated, higher=more overconfident)")
    if skipped:
        print(f"Skipped {len(skipped)}/{len(BENCHMARK_QUERIES)} queries due to unexpected errors — see 'skipped' in the report.")

    with open("eval/results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nFull report written to eval/results.json")


if __name__ == "__main__":
    main()
