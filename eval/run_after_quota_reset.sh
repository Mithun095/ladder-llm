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

# Remove this job so it runs exactly once. Absolute path because cron's PATH is minimal — if
# `crontab` isn't found here the entry survives and this benchmark quietly burns the daily
# OpenRouter quota every morning forever, which is a much worse failure than not running at all.
if [ -x /usr/bin/crontab ]; then
  /usr/bin/crontab -l 2>/dev/null | grep -v 'run_after_quota_reset' | /usr/bin/crontab - 2>/dev/null \
    && echo "removed the cron entry; this was a one-shot" >>"$OUT"
else
  echo "WARNING: /usr/bin/crontab not found — remove the entry by hand:" >>"$OUT"
  echo "  crontab -l | grep -v run_after_quota_reset | crontab -" >>"$OUT"
fi

echo "results in $OUT"
