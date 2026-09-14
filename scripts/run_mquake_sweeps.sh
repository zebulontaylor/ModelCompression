#!/bin/bash
# Sequential overnight MQuAKE CF9k sweeps. Each run is resumable (state.json) and
# is skipped once its state reports a terminal status, so re-running resumes.
#
#   sweep_3  the parameters as given
#   sweep_4  same, but a stronger recall-removal preference (--recall-weight 0.30)
#   sweep_5  same as sweep_3, but conservative eligibility and a layer-spread cap
#
# All three start from the same point (cf9k v2 scores, empty mask) so they are
# comparable to each other.
set -uo pipefail
cd /home/zeb/Desktop/ModelCompression

LOG=artifacts/mquake_sweeps.log
SCORES=artifacts/channel_scores/mquake_cf9k_qwen3_1.7b_v2/scores.pt
CAL=data/calibration/mquake_remastered_cf9k_v2/calibration.jsonl
VAL=data/calibration/mquake_remastered_cf9k_v2/validation.jsonl

say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

finished() {
  # Treat a sweep as done when its state reports a terminal status.
  python3 -c "
import json,sys
try:
    s=json.load(open('artifacts/iterative_deletion/$1/state.json'))
except Exception:
    sys.exit(1)
sys.exit(0 if s.get('status') in ('complete','finished','stopped','done') else 1)
" 2>/dev/null
}

sweep() {
  local name=$1; shift
  if finished "$name"; then
    say "skipping $name (already complete)"
    return 0
  fi
  say "starting $name"
  uv run python scripts/run_iterative_deletion.py --offline \
    --output "artifacts/iterative_deletion/$name" \
    --initial-scores "$SCORES" \
    --initial-mask artifacts/initial_masks/empty.json \
    --calibration-data "$CAL" \
    --validation-data "$VAL" \
    "$@" >> "$LOG" 2>&1
  local rc=$?
  say "$name exited rc=$rc"
  return $rc
}

COMMON=(
  --target-fraction 0.8
  --fraction 0.04
  --max-extraction-exact-drop 0.02
  --max-extraction-containment-drop 0.02
  --max-reasoning-exact-drop 0.02
  --max-reasoning-containment-drop 0.02
)

say "sweep chain started"

# 1. Exactly the parameters given.
sweep cf9k_sweep_3 "${COMMON[@]}" --protected-quantile 0.9 --max-layer-fraction 1.0

# 2. Same budget, stronger preference for removing recall-supporting channels.
sweep cf9k_sweep_4 "${COMMON[@]}" --protected-quantile 0.9 --max-layer-fraction 1.0 \
  --recall-weight 0.30

# 3. Conservative protected eligibility plus a per-layer cap, so the cheapest late
#    layers cannot absorb the whole mask.
sweep cf9k_sweep_5 "${COMMON[@]}" --protected-quantile 0.6 --max-layer-fraction 0.25

say "sweep chain finished"
