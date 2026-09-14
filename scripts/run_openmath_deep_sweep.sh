#!/bin/bash
# Deep channel-deletion sweep with a protected OpenMathInstruct-2 math-reasoning
# family alongside the matched MQuAKE CF9k conditions.
#
# Predeclared acceptance budgets (deliberately looser than the 2-point budgets
# that stopped cf9k_sweep_3/4 near 22-25%, because the question here is how far
# pure deletion goes before protected behaviour actually collapses):
#
#   extraction / reasoning exact match   10 points
#   extraction / reasoning containment    5 points
#   worst protected family exact match   20 points   (extraction, reasoning, openmath)
#
# Containment is the trustworthy content signal; the exact-match budgets are
# wide because deep masks reformat answers long before they lose content, and
# the per-family gate is what keeps free-running math derivations honest.
#
# Resumable: re-running continues from state.json.
set -uo pipefail
cd /home/zeb/Desktop/ModelCompression

LOG=artifacts/openmath_deep_sweep.log
NAME=openmath_deep_50pct
SCORES=artifacts/channel_scores/mquake_cf9k_openmath_v2/scores.pt
DATA=data/calibration/mquake_cf9k_openmath_v2

say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "sweep chain started"

uv run python scripts/run_iterative_deletion.py --offline \
  --output "artifacts/iterative_deletion/$NAME" \
  --initial-scores "$SCORES" \
  --initial-mask artifacts/initial_masks/empty.json \
  --calibration-data "$DATA/calibration.jsonl" \
  --validation-data "$DATA/validation.jsonl" \
  --target-fraction 0.55 \
  --fraction 0.04 \
  --min-backoff-fraction 0.01 \
  --protected-quantile 0.9 \
  --max-layer-fraction 1.0 \
  --recall-weight 0.30 \
  --max-extraction-exact-drop 0.10 \
  --max-extraction-containment-drop 0.05 \
  --max-reasoning-exact-drop 0.10 \
  --max-reasoning-containment-drop 0.05 \
  --max-protected-family-exact-drop 0.20 \
  >> "$LOG" 2>&1
rc=$?
say "$NAME exited rc=$rc"

say "sweep chain finished"
