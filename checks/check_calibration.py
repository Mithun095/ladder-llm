from eval.calibration import compute_ece

# Perfectly calibrated: confidence 10 always passes, confidence 1 always fails.
perfect = [(10, True)] * 5 + [(1, False)] * 5
ece_perfect = compute_ece(perfect)
assert ece_perfect < 0.05, f"expected near-zero ECE for perfect calibration, got {ece_perfect}"

# Badly overconfident: confidence always 10, but only half actually pass.
overconfident = [(10, True)] * 5 + [(10, False)] * 5
ece_over = compute_ece(overconfident)
assert ece_over > 0.4, f"expected high ECE for overconfident case, got {ece_over}"

assert compute_ece([]) == 0.0

print(f"perfect calibration ECE={ece_perfect:.3f}, overconfident ECE={ece_over:.3f}")
print("calibration check passed.")
