# Overnight run — September 13–14, 2026

## 1. 2WikiMultiHopQA was abandoned

Built a full matched corpus from `voidful/2WikiMultihopQA` (pinned revision
`16852fde9d85cba158cf7e6517e7a3f9415a28c0`): 167,454 source records → 78,807
connected two-hop chains → 31,886 unique final triples, with counterfactual
final hops mirroring the MQuAKE construction.

The baseline-known filter accepted **559/31,886 (1.75%)**, against MQuAKE CF9k's
58%. Relaxing to containment rescues only 3.41%, so this is not an answer-format
problem: the failures are confidently wrong facts (`cholera` → `heart attack`,
`6 February 1292` → `1126`). 2WikiMultiHopQA uses obscure tail entities by
design, and Qwen3-1.7B does not hold those facts parametrically.

Acceptance by relation (calibration split):

| relation | accepted | rate |
|---|---|---|
| date of birth | 1 / 5,191 | 0.0% |
| date of death | 0 / 4,249 | 0.0% |
| country of citizenship | 274 / 3,717 | 7.4% |
| place of birth | 29 / 2,570 | 1.1% |
| father | 3 / 1,138 | 0.3% |

A PopQA probe (800 sampled questions, natural question form, alias-aware exact
match) returned **10.9%**, so this is a general property of tail-entity corpora
rather than a 2Wiki quirk. MQuAKE's 58% reflects that its cases are built on
popular entities.

**Lesson for future corpus selection:** schema fit and raw size are not the
binding constraint; baseline-known rate is. Probe a 500-case sample with
`scripts/probe_recall_acceptance.py` before building a full corpus.

CounterFact (21,919 cases) remains the untested candidate most likely to clear
the bar, since it is constructed so base models know `target_true`.

Artifacts kept: `data/calibration/2wiki*`, `artifacts/2wiki_*`,
`artifacts/probe_popqa.json`.

## 2. MQuAKE CF9k sweeps

Three sequential sweeps, all starting from
`artifacts/channel_scores/mquake_cf9k_qwen3_1.7b_v2/scores.pt` and an **empty**
mask (`artifacts/initial_masks/empty.json`), so they are comparable to each
other but not to `cf9k_sweep_2`, which inherited a cf3k-era 64-channel mask.

Shared: `--target-fraction 0.8 --fraction 0.04`, all four acceptance budgets
0.02, calibration/validation from `mquake_remastered_cf9k_v2`.

### Terminal comparison

| | sweep 3 (`recall-weight 0.05`) | sweep 4 (`recall-weight 0.30`) |
|---|---|---|
| protected-quantile / max-layer-fraction | 0.9 / 1.0 | 0.9 / 1.0 |
| channels deleted | 42,152 (24.50%) | **38,454 (22.35%)** |
| recall exact | 0.6955 → 0.6722 (−2.3 pts) | 0.6955 → **0.5108 (−18.5 pts)** |
| recall containment | 0.9835 → 0.9125 (−7.1) | 0.9835 → **0.8403 (−14.3)** |
| extraction containment | 0.9659 → 0.9608 (−0.5) | 0.9659 → 0.9523 (−1.4) |
| reasoning containment | 0.9864 → 0.9784 (−0.8) | 0.9864 → 0.9784 (−0.8) |
| recall margin Δ | +0.43 nats | **−2.64 nats** |
| extraction margin Δ | +2.08 nats | +1.49 nats |
| reasoning margin Δ | +1.87 nats | +1.39 nats |
| recall content correct→missing (net) | −125 | **−252** |
| stop reason | protected budget | protected budget |

Both stopped on the reasoning exact-match budget. Neither approached
`--target-fraction 0.8`.

### Main finding

Raising `--recall-weight` from the default 0.05 to 0.30 removed **8× more recall
by exact match and 2× more by containment, using 2.15 percentage points fewer
channels, at identical reasoning containment cost** and 0.9 pt more extraction
cost. The contrastive recall-vs-reasoning screen supports substantially more
selective forgetting than the default weighting extracted.

For sweep 4, all three metric families agree the damage is recall-specific:
exact match, containment, and teacher-forced margin (recall margin −2.64 nats
while both protected margins rose).

### Caveats

- **Sweep 3's margins do not corroborate its containment result.** Recall margin
  *rose* (+0.43 nats) while recall generation degraded. Quote sweep 3 on
  containment only. Sweep 4 does not have this problem.
- **The stopping fractions are soft.** In both sweeps the final five or six
  backoff rejections sat at 102–117% of the reasoning budget regardless of batch
  size — roughly two validation examples out of ~2,800. In sweep 4's round 9 a
  430-channel batch measured *worse* than an 860-channel one. Read the boundary
  as "near 22–25% under a 2-point reasoning budget," not as a precise threshold.
- **Recall exact is format-noisy.** It bounced non-monotonically in both sweeps
  (sweep 3: 0.628–0.685). Containment is the trustworthy line and fell
  monotonically across accepted rounds.
- Single-round recall readings carry several points of noise; judge at terminal
  states, not per round.

### Efficiency note for next time

Both sweeps spent their last ~1h45m per round exhausting the backoff chain
(4% → 2% → 1% → 0.5% → 0.25% → 173) to gain ~0.1%. Sweep 3's round 9 and sweep
4's rounds 8–9 were almost pure cost. Raising `--min-backoff-fraction` well
above the default 0.001 would cut these endgames short and fit a third
configuration in the same wall-clock.

## 3. Sweep 5 — completed; read its exact-match numbers with care

`--protected-quantile 0.6 --max-layer-fraction 0.25`, default `--recall-weight
0.05`. Ran 03:56–06:06, terminated `stopped_protected_budget` at 27,958
channels (16.25%).

Accepted trajectory (recall containment): 4% 0.9631, 8% 0.9580, 12% 0.9574,
16% 0.9250, 16.25% 0.9398.

### Three-way terminal comparison

| | sweep 3 | sweep 4 | sweep 5 |
|---|---|---|---|
| config | `rw 0.05`, q 0.9, layer 1.0 | **`rw 0.30`**, q 0.9, layer 1.0 | `rw 0.05`, **q 0.6, layer 0.25** |
| channels deleted | 42,152 (24.50%) | 38,454 (22.35%) | 27,958 (16.25%) |
| recall containment | −7.1 pts | **−14.3 pts** | −4.4 pts |
| recall exact | −2.3 pts | −18.5 pts | −33.9 pts (mostly format) |
| extraction containment | −0.5 | −1.4 | +0.2 |
| reasoning containment | −0.8 | −0.8 | 0.0 |

Sweep 4 wins on the research question. Sweep 5 deleted the fewest channels and
removed the least content despite the largest exact-match drop.

### Round 5 cliff

At 16% accepted, the next 4% batch produced a qualitative collapse rather than a
budget overrun:

| at 20% (rejected) | drop | % of 0.02 budget |
|---|---|---|
| extraction exact | +0.4489 | 2244% |
| reasoning exact | +0.1969 | 984% |
| extraction containment | +0.0972 | 486% |

Every other rejection in all three sweeps topped out near 170%. Extraction exact
match fell ~45 points in one round from a mask healthy at 16%. Plausible
mechanism: `--protected-quantile 0.6` excludes 40% of channels and
`--max-layer-fraction 0.25` caps per-layer take, so by round 5 the selector may
have exhausted cheap candidates and been forced into jointly-essential ones.

### The failure is formatting, not content

Throughout the round-5 backoff chain, exact match collapsed while containment
stayed healthy — the model stops emitting terse correctly-formatted answers
across *every* condition:

| backoff | recall exact | recall containment | extraction exact drop | extraction containment drop |
|---|---|---|---|---|
| 3441 | 0.1352 | 0.9369 | 668% | 151% |
| 1720 | 0.2409 | 0.9511 | 278% | 8.5% |
| 860 | 0.3068 | 0.9392 | 153% | 5.7% |
| 430 (accepted) | 0.3568 | 0.9398 | 65% | −11% |

The accepted 16.25% state shows recall exact −33.9 pts but recall containment
only −4.4 pts, with reasoning containment *exactly unchanged* and extraction
containment slightly improved. **This is mostly reformatting, not forgetting.**

Ranking on actual content removal is therefore unchanged:
**sweep 4 (−14.3 pts recall containment) > sweep 3 (−7.1) > sweep 5 (−4.4).**
Reading sweep 5's exact-match column alone would invert this.

Run `scripts/audit_evaluation.py` against sweep 5's mask before quoting any of
its numbers.

### Design implication

Acceptance requires all four budgets to pass, so a formatting regression can
halt a sweep that is losing almost no content — at backoff 860, reasoning
containment was 1.4% of budget while extraction exact was 153% over. Whether
exact-match budgets should gate acceptance this way is a design call; the README
already notes that strict exact match counts answer-format changes as failures.

## Tools added

- `scripts/prepare_2wiki_calibration.py` — 2Wiki matched-corpus builder
- `scripts/probe_recall_acceptance.py` — baseline-known rate probe (PopQA / CounterFact)
- `scripts/sweep_trend.py <dir>` — accepted trajectory for one sweep
- `scripts/sweep_compare.py <dir…>` — matched-fraction comparison across sweeps
- `scripts/run_mquake_sweeps.sh`, `scripts/guard_mquake_sweeps.sh` — sequential runner + watchdog
- `scripts/prepare_v2_data.py` gained `--use-record-metadata` (opt-in; MQuAKE path unchanged)
