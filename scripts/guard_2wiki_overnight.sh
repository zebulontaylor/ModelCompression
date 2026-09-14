#!/bin/bash
# Independent overnight guard for the 2Wiki chain. Runs in its own screen session
# so it survives the terminal, the Claude session, and logout.
#
#   Phase A: make sure the baseline filter produces a manifest, re-running it if
#            the original process dies without one.
#   Phase B: keep the sweep alive. run_2wiki_overnight.sh skips completed stages
#            and the sweep resumes from state.json, so a restart is safe.
#
# Writes artifacts/2wiki_morning_summary.txt when it stops.
set -uo pipefail
cd /home/zeb/Desktop/ModelCompression

GUARD_LOG=artifacts/2wiki_guard.log
CHAIN_LOG=artifacts/2wiki_overnight.log
FILTER_LOG=artifacts/2wiki_baseline_filter.log
KNOWN=data/calibration/2wiki_baseline_known
SWEEP=artifacts/iterative_deletion/2wiki_sweep_1
SUMMARY=artifacts/2wiki_morning_summary.txt

MAX_FILTER_ATTEMPTS=2
MAX_SWEEP_RESTARTS=5

say() { echo "[$(date -Is)] $*" | tee -a "$GUARD_LOG"; }

run_filter() {
  say "starting baseline filter attempt $1"
  uv run python scripts/filter_mquake_baseline_known.py --offline \
    --input data/calibration/2wiki \
    --output "$KNOWN" \
    --batch-size 32 >> "$FILTER_LOG" 2>&1
  local rc=$?
  say "filter attempt $1 exited rc=$rc"
  return $rc
}

summarize() {
  {
    echo "2Wiki overnight summary — written $(date -Is)"
    echo
    echo "== guard =="
    tail -20 "$GUARD_LOG" 2>/dev/null
    echo
    echo "== baseline filter =="
    if [ -f "$KNOWN/manifest.json" ]; then
      python3 -c "
import json
m=json.load(open('$KNOWN/manifest.json'))
for s,v in m['splits'].items():
    print(f\"  {s}: {v['accepted_groups']}/{v['candidate_groups']} groups accepted, {v['examples']} examples\")
" 2>/dev/null
    else
      echo "  NO MANIFEST — filter did not complete"
    fi
    echo
    echo "== sweep state =="
    if [ -f "$SWEEP/state.json" ]; then
      python3 -c "
import json
s=json.load(open('$SWEEP/state.json'))
print('  status:', s.get('status'), '| stage:', s.get('stage'), '| round:', s.get('round'))
for key in ('cumulative_count','cumulative_fraction','final_evaluation'):
    if key in s: print(f'  {key}: {s[key]}')
d=s.get('decisions') or []
print(f'  attempted rounds: {len(d)}')
for x in d[-8:]:
    print(f\"    round {x.get('round')}: {x.get('decision')} batch={x.get('batch_count')} cumulative={x.get('cumulative_count')} ({x.get('cumulative_fraction',0)*100:.3f}%)\")
" 2>/dev/null || echo "  (state.json unreadable)"
    else
      echo "  sweep has not written state yet"
    fi
    echo
    echo "== last chain log lines =="
    tail -25 "$CHAIN_LOG" 2>/dev/null
  } > "$SUMMARY" 2>&1
  say "wrote $SUMMARY"
}

trap 'summarize' EXIT

say "guard started"

# ---- Phase A: baseline filter ----
attempt=0
while [ ! -f "$KNOWN/manifest.json" ]; do
  if pgrep -f "filter_mquake_baseline_known.py" > /dev/null; then
    sleep 30
    continue
  fi
  # No manifest and no running filter: the original died. Re-run it ourselves.
  attempt=$((attempt + 1))
  if [ "$attempt" -gt "$MAX_FILTER_ATTEMPTS" ]; then
    say "FATAL: filter failed after $MAX_FILTER_ATTEMPTS re-run attempts"
    exit 1
  fi
  say "filter not running and no manifest present; re-running"
  run_filter "$attempt"
  sleep 5
done
say "baseline filter manifest present"

# ---- Phase B: keep the sweep alive ----
restarts=0
while true; do
  if grep -q "sweep finished" "$CHAIN_LOG" 2>/dev/null; then
    say "sweep finished normally"
    exit 0
  fi
  if grep -q "FATAL" "$CHAIN_LOG" 2>/dev/null; then
    say "chain reported FATAL; not restarting"
    exit 1
  fi
  if ! screen -ls 2>/dev/null | grep -q "2wiki_sweep"; then
    restarts=$((restarts + 1))
    if [ "$restarts" -gt "$MAX_SWEEP_RESTARTS" ]; then
      say "FATAL: chain died $MAX_SWEEP_RESTARTS times; giving up"
      exit 1
    fi
    say "chain session gone without finishing; restart $restarts of $MAX_SWEEP_RESTARTS"
    screen -dmS 2wiki_sweep bash scripts/run_2wiki_overnight.sh
    sleep 60
  fi
  sleep 120
done
