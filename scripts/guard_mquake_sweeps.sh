#!/bin/bash
# Independent guard for the MQuAKE sweep chain. Restarts the chain if it dies
# (each sweep resumes from its own state.json) and writes a morning summary.
set -uo pipefail
cd /home/zeb/Desktop/ModelCompression

GUARD_LOG=artifacts/mquake_guard.log
CHAIN_LOG=artifacts/mquake_sweeps.log
SUMMARY=artifacts/morning_summary.txt
MAX_RESTARTS=5

say() { echo "[$(date -Is)] $*" | tee -a "$GUARD_LOG"; }

summarize() {
  {
    echo "Overnight summary — written $(date -Is)"
    echo
    echo "== 2WikiMultiHopQA outcome (abandoned) =="
    echo "  Baseline-known filter accepted 559/31886 recall candidates (1.75%)."
    echo "  Cause: obscure tail entities the 1.7B baseline does not know, not answer formatting"
    echo "  (containment rescues only 3.41%). PopQA probe was 10.9%, so this is a general"
    echo "  property of tail-entity corpora, not specific to 2Wiki. Artifacts kept under"
    echo "  data/calibration/2wiki* and artifacts/2wiki_*."
    echo
    echo "== MQuAKE CF9k sweeps =="
    for s in cf9k_sweep_3 cf9k_sweep_4 cf9k_sweep_5; do
      echo "-- $s"
      python3 -c "
import json
try:
    s=json.load(open('artifacts/iterative_deletion/$s/state.json'))
except Exception:
    print('   not started'); raise SystemExit
print('   status:', s.get('status'), '| stage:', s.get('stage'), '| round:', s.get('round'))
for k in ('cumulative_count','cumulative_fraction','final_evaluation'):
    if k in s: print(f'   {k}: {s[k]}')
d=s.get('decisions') or []
print(f'   attempted rounds: {len(d)}')
for x in d[-10:]:
    print(f\"     round {x.get('round')}: {x.get('decision')} batch={x.get('batch_count')} \"
          f\"cumulative={x.get('cumulative_count')} ({x.get('cumulative_fraction',0)*100:.3f}%)\")
" 2>/dev/null || echo "   (state unreadable)"
    done
    echo
    echo "== guard =="
    tail -15 "$GUARD_LOG" 2>/dev/null
    echo
    echo "== chain log tail =="
    tail -30 "$CHAIN_LOG" 2>/dev/null
  } > "$SUMMARY" 2>&1
  say "wrote $SUMMARY"
}

trap 'summarize' EXIT
say "guard started"

restarts=0
while true; do
  if grep -q "sweep chain finished" "$CHAIN_LOG" 2>/dev/null; then
    say "chain finished normally"
    exit 0
  fi
  if ! screen -ls 2>/dev/null | grep -q "mquake_sweeps"; then
    restarts=$((restarts + 1))
    if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
      say "FATAL: chain died $MAX_RESTARTS times; giving up"
      exit 1
    fi
    say "chain session gone without finishing; restart $restarts of $MAX_RESTARTS"
    screen -dmS mquake_sweeps bash scripts/run_mquake_sweeps.sh
    sleep 90
  fi
  sleep 120
done
