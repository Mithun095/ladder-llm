def compute_ece(confidence_verdict_pairs: list[tuple[int, bool]], num_bins: int = 5) -> float:
    """Expected Calibration Error: how far self-reported confidence diverges from actual
    accuracy. 0 = perfectly calibrated, higher = more overconfident/underconfident."""
    if not confidence_verdict_pairs:
        return 0.0

    bins: list[list[tuple[float, bool]]] = [[] for _ in range(num_bins)]
    for confidence, passed in confidence_verdict_pairs:
        normalized = (confidence - 1) / 9  # confidence is 1-10 -> normalize to 0-1
        bin_index = min(int(normalized * num_bins), num_bins - 1)
        bins[bin_index].append((normalized, passed))

    total = len(confidence_verdict_pairs)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_confidence = sum(c for c, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, passed in bucket if passed) / len(bucket)
        ece += (len(bucket) / total) * abs(accuracy - avg_confidence)
    return ece


def reliability_table(confidence_verdict_pairs: list[tuple[int, bool]], num_bins: int = 5) -> list[dict]:
    """Per-bin breakdown for printing: how confident the model claimed to be vs. how often
    the judge actually passed it, in that confidence range."""
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(num_bins)]
    for confidence, passed in confidence_verdict_pairs:
        normalized = (confidence - 1) / 9
        bin_index = min(int(normalized * num_bins), num_bins - 1)
        bins[bin_index].append((normalized, passed))

    rows = []
    for i, bucket in enumerate(bins):
        if not bucket:
            continue
        avg_confidence = sum(c for c, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, passed in bucket if passed) / len(bucket)
        rows.append({
            "bin": f"{i / num_bins:.1f}-{(i + 1) / num_bins:.1f}",
            "count": len(bucket),
            "avg_confidence": avg_confidence,
            "accuracy": accuracy,
        })
    return rows
