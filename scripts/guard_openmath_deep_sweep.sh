#!/bin/bash
# Independent guard for the deep OpenMathInstruct sweep. Restarts the chain if
# it dies (the sweep resumes from its own state.json) and writes a summary.
set -uo pipefail
cd /home/zeb/Desktop/ModelCompression

GUARD_LOG=artifacts/openmath_guard.log
CHAIN_LOG=artifacts/openmath_deep_sweep.log
SUMMARY=artifacts/openmath_summary.txt
NAME=openmath_deep_50pct
MAX_RESTARTS=5

say() { echo "[$(date -Is)] $*" | tee -a "$GUARD_LOG"; }

summarize() {
  {
    echo "Deep sweep summary — written $(date -Is)"
    echo
    echo "== $NAME =="
    python3 -c "
import json
try:
    s=json.load(open('artifacts/iterative_deletion/$NAME/state.json'))
except Exception:
    print('   not started'); raise SystemExit
print('   status:', s.get('status'), '| stage:', s.get('stage'), '| round:', s.get('round'))
for k in ('current_count','current_fraction','target_count','final_evaluation','rejection'):
    if k in s: print(f'   {k}: {s[k]}')
print('   accepted rounds:', s.get('accepted_rounds'))
" 2>/dev/null || echo "   (state unreadable)"
    echo
    echo "== accepted trajectory =="
    uv run python scripts/sweep_trend.py "artifacts/iterative_deletion/$NAME" 2>/dev/null || echo "   (no trend yet)"
    echo
    echo "== guard =="
    tail -15 "$GUARD_LOG" 2>/dev/null
    echo
    echo "== chain log tail =="
    tail -40 "$CHAIN_LOG" 2>/dev/null
  } > "$SUMMARY" 2>&1
  say "wrote $SUMMARY"
}

trap 'summarize' EXIT
say "guard started"

restarts=0
while true; do
  if grep -q "sweep chain finished" "$CHAIN_LOG" 2>/dev/null; then
    say "chain finished normally"
    summarize
    exit 0
  fi
  if ! screen -ls 2>/dev/null | grep -q "openmath_sweep"; then
    restarts=$((restarts + 1))
    if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
      say "FATAL: chain died $MAX_RESTARTS times; giving up"
      exit 1
    fi
    say "chain session gone without finishing; restart $restarts of $MAX_RESTARTS"
    screen -dmS openmath_sweep bash scripts/run_openmath_deep_sweep.sh
    sleep 90
  fi
  sleep 120
  summarize
done
