#!/usr/bin/env bash
# One-shot: run the coding-model comparison once the OpenRouter free-tier daily quota resets.
#
# The free-models-per-day cap is account-wide and resets at 00:00 UTC. Until it does, every
# OpenRouter candidate reports `unavailable` and the comparison only covers the Groq models,
# which is half the question.
#
# Installed as a cron entry that DELETES ITSELF after running, so this benchmark does not quietly
# burn the daily quota every morning forever. Remove it early with:
#     crontab -l | grep -v run_after_quota_reset | crontab -
set -uo pipefail

REPO="/home/mithun/Desktop/ladder-llm"
OUT="$REPO/eval/coding_comparison_$(date -u +%Y%m%d).log"

cd "$REPO" || exit 1

{
  echo "=== coding model comparison — $(date -u '+%Y-%m-%d %H:%M UTC') ==="
  "$REPO/.venv/bin/python" -m eval.compare_coding_models
  echo "=== exit code: $? ==="
} >"$OUT" 2>&1

# Remove this job so it runs exactly once.
crontab -l 2>/dev/null | grep -v 'run_after_quota_reset' | crontab - 2>/dev/null

echo "results in $OUT"
