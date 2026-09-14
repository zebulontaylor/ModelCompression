# External generalization check — CF9k sweep 4

Mask: `artifacts/iterative_deletion/cf9k_sweep_4/round_008_backoff_06/candidate_cumulative_mask.json`
Deleted channels: **38,454 (22.35%)**

All comparisons use the same deterministic sample and greedy decoding. Recall retention is
conditioned on baseline containment-correct examples. McNemar p-values are exact, two-sided,
and descriptive (no multiple-comparison correction).

| benchmark | n | baseline | masked | delta | correct→wrong | wrong→correct | paired p |
|---|---:|---:|---:|---:|---:|---:|---:|
| arc_challenge | 300 | 71.0% | 71.7% | +0.7% | 4 | 6 | 0.7539 |
| gsm8k | 64 | 71.9% | 4.7% | -67.2% | 45 | 2 | 1.604e-11 |
| popqa | 800 | 14.4% | 10.1% | -4.2% | 45 | 11 | 5.378e-06 |
| counterfact | 800 | 35.8% | 33.6% | -2.1% | 54 | 37 | 0.09295 |

## Recall retention

| benchmark | baseline-known | retained |
|---|---:|---:|
| popqa | 115 | 60.9% |
| counterfact | 286 | 81.1% |

## Token-limit diagnostic

| benchmark | baseline | masked |
|---|---:|---:|
| arc_challenge | 76.0% | 77.3% |
| gsm8k | 0.0% | 34.4% |
| popqa | 0.6% | 2.5% |
| counterfact | 10.9% | 6.2% |

ARC's eight-token cap only checks the leading answer letter; reaching it is not a
failure. On GSM8K, the cap is diagnostic: prompts explicitly request a short calculation
and baseline responses all terminate, while masked responses that hit the cap usually loop.

## Reproducibility

Seed: `314159`. Samples: `{"arc_challenge": 300, "counterfact": 800, "gsm8k": 64, "popqa": 800}`.
Model: `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
ARC revision: `210d026faf9955653af8916fad021475a3f00453`. GSM8K revision: `740312add88f781978c0658806c59bc2815b9866`.
