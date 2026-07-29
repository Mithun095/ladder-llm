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
  # ORDER MATTERS, and getting it wrong made this script unable to do its own job.
  #
  # Both commands draw on ONE account-wide budget of 50 OpenRouter requests/day:
  #   compare_coding_models  3 OpenRouter candidates x 2 repeats x 6 tasks   = 36 requests
  #   run_eval               1 tier-4 baseline call per benchmark query      = 25 requests
  #                          (+ tier 3/4 escalations, + a retry on each 429)
  # 36 + 25 = 61 > 50. Whichever runs second loses its OpenRouter arm entirely.
  #
  # The first version ran run_eval first "because the README quotes its numbers", which
  # guaranteed the coding comparison came back all-unavailable — precisely the result this
  # script was written to escape — and then deleted its own cron entry, so there was no retry.
  #
  # Coding comparison goes first because OpenRouter is the ONLY way to get it: those three
  # candidates exist nowhere else, and without them the comparison is the same half-answer it
  # already has. run_eval degrades gracefully instead — its cascade arm is almost entirely Groq
  # now, so it still produces pass rate and both savings figures; only the always-tier-4 baseline
  # arm is OpenRouter, and run_eval already reports that arm over however many queries it
  # actually reached rather than scoring an outage as a failure.
  echo "--- coding model comparison: python -m eval.compare_coding_models ---"
  echo "    (runs first: its OpenRouter candidates are unobtainable any other way)"
  "$REPO/.venv/bin/python" -m eval.compare_coding_models
  echo "=== compare_coding_models exit: $? ==="
  echo

  echo "--- full benchmark: python -m eval.run_eval ---"
  echo "    (the always-tier-4 baseline arm may be truncated by the remaining quota; the"
  echo "     cascade arm is Groq-backed and unaffected)"
  "$REPO/.venv/bin/python" -m eval.run_eval
  echo "=== run_eval exit: $? ==="
} >"$OUT" 2>&1

# Remove this job so it runs exactly once. Absolute path because cron's PATH is minimal — if
# `crontab` isn't found here the entry survives and this benchmark quietly burns the daily
# OpenRouter quota every morning forever, which is a much worse failure than not running at all.
if [ -x /usr/bin/crontab ]; then
  # `|| true` on the grep: when this is the ONLY crontab entry, grep -v matches nothing and exits
  # 1, and with `set -o pipefail` that made the whole pipeline report failure even though the
  # (now empty) crontab installed fine. The success message never printed, and because stderr is
  # discarded a genuine failure printed nothing either — so the good and bad outcomes looked
  # identical, and the bad one is the case that leaves this burning the quota every morning.
  if /usr/bin/crontab -l 2>/dev/null | { grep -v 'run_after_quota_reset' || true; } \
       | /usr/bin/crontab - 2>>"$OUT"; then
    echo "removed the cron entry; this was a one-shot" >>"$OUT"
  else
    echo "ERROR: could not remove the cron entry — it will run again tomorrow and spend the" >>"$OUT"
    echo "       daily OpenRouter quota. Remove it by hand:" >>"$OUT"
    echo "  crontab -l | grep -v run_after_quota_reset | crontab -" >>"$OUT"
  fi
else
  echo "WARNING: /usr/bin/crontab not found — remove the entry by hand:" >>"$OUT"
  echo "  crontab -l | grep -v run_after_quota_reset | crontab -" >>"$OUT"
fi

echo "results in $OUT"
