#!/bin/bash
# Overnight chain for the 2WikiMultiHopQA corpus:
#   wait for the baseline filter -> v2 enrichment -> initial channel scores -> iterative deletion sweep.
# Each stage is skipped if its output already exists, so the script can be re-run to resume.
set -euo pipefail
cd /home/zeb/Desktop/ModelCompression

LOG=artifacts/2wiki_overnight.log
FILTER_LOG=artifacts/2wiki_baseline_filter.log
KNOWN=data/calibration/2wiki_baseline_known
V2=data/calibration/2wiki_v2
SCORES=artifacts/channel_scores/2wiki_qwen3_1.7b_v2
SWEEP=artifacts/iterative_deletion/2wiki_sweep_1

CAL_LIMIT=3000
VAL_LIMIT=1000

say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "chain started"

# 1. Wait for the baseline filter to write its manifest. Give up rather than spin
# forever if the filter dies without writing one.
WAITED=0
while [ ! -f "$KNOWN/manifest.json" ]; do
  if grep -q "filter finished" "$FILTER_LOG" 2>/dev/null; then break; fi
  if [ "$WAITED" -ge 7200 ]; then break; fi
  sleep 30
  WAITED=$((WAITED + 30))
done
if [ ! -f "$KNOWN/manifest.json" ]; then
  say "FATAL: baseline filter produced no manifest after ${WAITED}s; see $FILTER_LOG"
  exit 1
fi
say "baseline filter complete"
python3 -c "
import json
m=json.load(open('$KNOWN/manifest.json'))
for s,v in m['splits'].items():
    print(f\"  {s}: {v['accepted_groups']}/{v['candidate_groups']} accepted, {v['examples']} examples\")
" | tee -a "$LOG"

# 2. v2 enrichment: relation-matched alternatives and prompt variants.
if [ ! -f "$V2/manifest.json" ]; then
  say "stage: v2 enrichment"
  uv run python scripts/prepare_v2_data.py \
    --use-record-metadata \
    --input "$KNOWN" \
    --output "$V2" >> "$LOG" 2>&1
  say "v2 enrichment complete"
else
  say "skipping v2 enrichment (exists)"
fi

# 3. Initial channel scores on the unpruned model.
if [ ! -f "$SCORES/scores.pt" ]; then
  say "stage: initial channel scores (limit $CAL_LIMIT groups)"
  uv run python scripts/score_channels_v2.py --offline \
    --input "$V2/calibration.jsonl" \
    --output "$SCORES" \
    --limit "$CAL_LIMIT" >> "$LOG" 2>&1
  say "initial scores complete"
else
  say "skipping initial scores (exists)"
fi

# 4. Iterative deletion sweep. Resumable: re-running picks up the recorded stage.
say "stage: iterative deletion sweep -> $SWEEP"
uv run python scripts/run_iterative_deletion.py --offline \
  --target-fraction 0.8 \
  --fraction 0.04 \
  --max-extraction-exact-drop 0.02 \
  --max-extraction-containment-drop 0.02 \
  --max-reasoning-exact-drop 0.02 \
  --max-reasoning-containment-drop 0.02 \
  --protected-quantile 0.9 \
  --max-layer-fraction 1.0 \
  --output "$SWEEP" \
  --initial-scores "$SCORES/scores.pt" \
  --initial-mask "$SWEEP/initial_mask.json" \
  --calibration-data "$V2/calibration.jsonl" \
  --validation-data "$V2/validation.jsonl" \
  --calibration-limit "$CAL_LIMIT" \
  --validation-limit "$VAL_LIMIT" >> "$LOG" 2>&1

say "sweep finished"
