#!/usr/bin/env bash
# One-shot: re-measure everything that needs the OpenRouter quota, once it resets at 00:00 UTC.
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
OUT="$REPO/eval/postreset_$(date -u +%Y%m%d).log"

cd "$REPO" || exit 1

{
  echo "=== run started $(date -u '+%Y-%m-%d %H:%M UTC') ==="
  # Stamp the commit. Routing and scoring both change often here, so a benchmark number is only
  # interpretable if you know which code produced it — otherwise this log becomes the same
  # unattributed-numbers problem the README results table had to be labelled for.
  echo "commit:  $(cd "$REPO" && /usr/bin/git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "branch:  $(cd "$REPO" && /usr/bin/git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "dirty:   $(cd "$REPO" && [ -n "$(/usr/bin/git status --porcelain 2>/dev/null)" ] && echo yes || echo no)"
  echo
  # Full benchmark FIRST. Both jobs draw on the same account-wide daily quota, and the eval is
  # the one whose numbers the README quotes — if only one can finish, it should be this one.
  # It is also stale by three routing changes (expert entry tier, classifier swap, coding tiers),
  # so this is the first clean measurement of the current configuration.
  echo "--- full benchmark: python -m eval.run_eval ---"
  "$REPO/.venv/bin/python" -m eval.run_eval
  echo "=== run_eval exit: $? ==="
  echo

  echo "--- coding model comparison: python -m eval.compare_coding_models ---"
  "$REPO/.venv/bin/python" -m eval.compare_coding_models
  echo "=== compare_coding_models exit: $? ==="
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
