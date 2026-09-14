# Channel selection mechanism pilot

All results are exploratory validation results, not sealed-test estimates.

## Design

Same pinned Qwen3-1.7B checkpoint and unmasked parent for every method; these are one-shot masks, not full iterative sweeps. Existing target-balanced CF9k gradients provide recall/extraction/reasoning scores. Broader protection starts from 16 synthetic calibration examples for each of five families, retaining only baseline-generated solutions with solver-verified final answers. The protected gradient objective is mean token CE over the complete generated solution; intermediate steps are not independently verified. Evaluation uses 256 hash-selected MQuAKE groups (original prompts only) and 32 independently seeded synthetic examples per family. Numerical tasks use disjoint first operands across splits; other tasks use split-specific entity labels. These are in-template generalization tests with greedy generation: 48 tokens for MQuAKE and 384 for synthetic working plus a marked final answer. The model's thinking flag remains disabled; explicit working is requested in the prompt. Synthetic accuracy grades only the final-answer line. These remain small procedural problems, not a broad benchmark of long-chain reasoning.

Task formatting was finalized using baseline-only prechecks. An empty-heading parsing bug was corrected uniformly across all saved raw evaluation responses; the accepted calibration IDs used to compute family scores remain frozen. The final grading implementation never accepts an answer merely because it appears inside the working.

Protected-only uses recall weight 0; contrastive uses 0.05. Broader uses 0.05 and the maximum sensitivity percentile over all seven protected families. Every method retains the 25th-percentile eligibility thresholds and 15% per-layer cap. A failed selection is reported, not replaced with a smaller mask.

Bias-compensated arms use the identical deletion masks plus constant W_down[:, deleted] @ mean_activation[deleted]. Means use protected calibration only: 80 synthetic examples and 16 examples each for MQuAKE extraction and reasoning. No surviving weights change and there is no optimization or recovery training. The means average tokens within each teacher-forced example, then examples equally. These arms are not pure deletion and their added biases must be preserved in any compact model.

Pilot acceptance thresholds, written before evaluation: at most 2 percentage points of exact-accuracy loss, 0.10 nats of mean answer-margin loss, and 0.10 nats/token of answer-CE increase in **every** protected family. These are diagnostic point-estimate thresholds, not evidence of statistical noninferiority; this small sample cannot establish tight retention budgets. Margin/answer-CE diagnostics teacher-force the short correct answer directly, without supplying a solved trace. On synthetic prompts requesting working, these are auxiliary likelihood probes, not confidence estimates of the final generated answer.

## Results

| Method | Recall containment | MQuAKE reasoning exact | Worst protected exact drop (pp) | Worst protected margin drop (nats) | Worst protected CE increase | All family budgets pass? |
|---|---:|---:|---:|---:|---:|---|
| baseline | 99.22% | 99.61% | 0.00 | 0.0000 | 0.0000 | yes |
| protected_only_1pct | 99.22% | 99.22% | 6.25 | 0.0549 | 0.0589 | no |
| contrastive_1pct | 99.22% | 99.22% | 9.38 | 0.0532 | 0.0633 | no |
| broader_1pct | 99.22% | 99.61% | 3.12 | 0.0598 | 0.0813 | no |
| protected_only_4pct | 96.48% | 99.61% | 25.00 | -0.1031 | 0.7873 | no |
| contrastive_4pct | 96.48% | 99.61% | 28.12 | -0.0843 | 0.6591 | no |
| broader_4pct | 95.70% | 99.61% | 12.50 | 0.1449 | 0.2915 | no |
| protected_only_5pct | 97.27% | 99.61% | 37.50 | 0.9901 | 1.2946 | no |
| contrastive_5pct | 95.70% | 99.61% | 34.38 | 0.8293 | 0.9796 | no |
| protected_only_4pct_compensated | 96.48% | 99.61% | 25.00 | -0.0585 | 0.4222 | no |
| contrastive_4pct_compensated | 96.88% | 99.61% | 28.12 | -0.0446 | 0.3395 | no |
| broader_4pct_compensated | 95.70% | 99.61% | 12.50 | 0.1266 | 0.0485 | no |

Successful calibration solutions by family: arithmetic 16/16, program 16/16, rules 13/16, temporal 11/16, planning 8/16.
Final-answer grading strips markdown and optional literal answer tags. For route planning, an explicit correct first edge or complete optimal route is accepted as an alias for the requested next node. Only the final-answer line is scored; a correct intermediate value cannot rescue an incorrect or missing final answer.

## Per-family exact accuracy

| Method | arithmetic | mquake_extraction | mquake_reasoning | mquake_recall | planning | program | rules | temporal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 100.00% | 96.88% | 99.61% | 98.83% | 34.38% | 100.00% | 90.62% | 65.62% |
| protected_only_1pct | 100.00% | 96.88% | 99.22% | 99.22% | 40.62% | 100.00% | 87.50% | 59.38% |
| contrastive_1pct | 100.00% | 96.88% | 99.22% | 98.83% | 40.62% | 100.00% | 100.00% | 56.25% |
| broader_1pct | 100.00% | 96.88% | 99.61% | 98.44% | 37.50% | 100.00% | 87.50% | 65.62% |
| protected_only_4pct | 96.88% | 96.88% | 99.61% | 96.48% | 50.00% | 87.50% | 100.00% | 40.62% |
| contrastive_4pct | 96.88% | 96.88% | 99.61% | 96.09% | 53.12% | 87.50% | 93.75% | 37.50% |
| broader_4pct | 100.00% | 96.88% | 99.61% | 95.31% | 34.38% | 100.00% | 84.38% | 53.12% |
| protected_only_5pct | 62.50% | 96.88% | 99.61% | 97.27% | 59.38% | 75.00% | 100.00% | 37.50% |
| contrastive_5pct | 78.12% | 96.88% | 99.61% | 95.70% | 31.25% | 65.62% | 96.88% | 37.50% |
| protected_only_4pct_compensated | 96.88% | 96.88% | 99.61% | 96.48% | 50.00% | 87.50% | 96.88% | 40.62% |
| contrastive_4pct_compensated | 96.88% | 96.88% | 99.61% | 96.48% | 31.25% | 87.50% | 96.88% | 37.50% |
| broader_4pct_compensated | 100.00% | 96.88% | 99.61% | 95.70% | 40.62% | 100.00% | 78.12% | 56.25% |

## Matched comparisons

Paired percentile-bootstrap 95% intervals, 2,000 resamples per family. Positive deltas mean more accuracy/containment or higher margin. Intervals are exploratory and unadjusted for multiple comparisons. Percentile bootstrap intervals can under-cover sparse binary changes; the JSON also reports exact paired two-sided binomial p-values for accuracy/containment differences.

| Comparison | Recall containment delta (pp), 95% CI | Recall margin delta (nats), 95% CI |
|---|---:|---:|
| contrastive_1pct_minus_protected_only_1pct | +0.00 [+0.00, +0.00] | -0.0727 [-0.0932, -0.0530] |
| broader_1pct_minus_contrastive_1pct | +0.00 [+0.00, +0.00] | +0.0354 [+0.0077, +0.0616] |
| contrastive_4pct_minus_protected_only_4pct | +0.00 [-1.17, +1.17] | -0.0780 [-0.1028, -0.0542] |
| broader_4pct_minus_contrastive_4pct | -0.78 [-2.73, +1.17] | +0.0850 [+0.0179, +0.1516] |
| contrastive_5pct_minus_protected_only_5pct | -1.56 [-3.12, -0.39] | -0.0933 [-0.1220, -0.0643] |
| protected_only_4pct_compensated_minus_protected_only_4pct | +0.00 [-1.17, +1.17] | +0.0846 [+0.0666, +0.1015] |
| contrastive_4pct_compensated_minus_contrastive_4pct | +0.39 [+0.00, +1.17] | +0.0900 [+0.0736, +0.1061] |
| broader_4pct_compensated_minus_broader_4pct | +0.00 [-1.17, +1.17] | +0.0366 [+0.0209, +0.0521] |

## Selection limits

These limits apply to a one-shot candidate from the saved scores; they are not ceilings on an iterative, rescored pruning run.

- broader_5pct: only 8128 channels meet the protected thresholds and per-layer cap; need 8602
- protected_only_10pct: only 9437 channels meet the protected thresholds and per-layer cap; need 17204
- contrastive_10pct: only 9437 channels meet the protected thresholds and per-layer cap; need 17204
- broader_10pct: only 8128 channels meet the protected thresholds and per-layer cap; need 17204

## Causal fidelity

Four recall-selective, four protected-important, and four random channels, plus each four-channel group. Both 25% attenuation and full deletion are compared with local gate gradients on the same 16 held-out groups. These diagnostic gradients never select production masks. BF16 can obscure very small intervention effects.

| Condition | Attenuation | Pearson | Spearman | Mean absolute CE prediction error |
|---|---:|---:|---:|---:|
| recall | 0.25 | 0.595 | 0.483 | 0.004335 |
| recall | 1.0 | 0.936 | 0.427 | 0.007821 |
| extraction | 0.25 | 0.757 | 0.259 | 0.001597 |
| extraction | 1.0 | 0.944 | 0.552 | 0.002506 |
| reasoning | 0.25 | 0.443 | 0.098 | 0.000006 |
| reasoning | 1.0 | -0.188 | -0.070 | 0.000022 |

FP32 precision control, with TF32 disabled, on identical channels and examples:

| Condition | Attenuation | Pearson | Spearman | Mean absolute CE prediction error |
|---|---:|---:|---:|---:|
| recall | 0.25 | 0.993 | 1.000 | 0.00013673 |
| recall | 1.0 | 0.946 | 0.853 | 0.00130674 |
| extraction | 0.25 | 0.999 | 1.000 | 0.00005596 |
| extraction | 1.0 | 0.972 | 0.993 | 0.00095074 |
| reasoning | 0.25 | 0.649 | 0.671 | 0.00000094 |
| reasoning | 1.0 | -0.194 | 0.203 | 0.00000917 |

## Complete-solution likelihood diagnostic

Post-hoc check on fixed baseline-generated solutions with correct final answers. This uses the same type of complete-solution CE objective as broader calibration, rather than the short-answer probes above. These measurements did not select masks or change the predeclared acceptance rules. Positive values are mean CE increases in nats/token; teacher-forced trace likelihood does not establish free-running reasoning retention.

| Method | arithmetic | planning | program | rules | temporal |
|---|---:|---:|---:|---:|---:|
| baseline | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 |
| protected_only_1pct | +0.02335 | +0.04073 | +0.00587 | +0.02384 | +0.01492 |
| contrastive_1pct | +0.02100 | +0.04321 | +0.00058 | +0.02294 | +0.04015 |
| broader_1pct | -0.00011 | +0.00060 | +0.00002 | -0.00009 | +0.00047 |
| protected_only_4pct | +0.18419 | +0.21714 | +0.26074 | +0.06264 | +0.07620 |
| contrastive_4pct | +0.19203 | +0.21463 | +0.15240 | +0.06255 | +0.08154 |
| broader_4pct | +0.00162 | +0.00218 | -0.00007 | +0.00026 | +0.00110 |
| protected_only_5pct | +0.41535 | +0.26766 | +0.45245 | +0.08258 | +0.17784 |
| contrastive_5pct | +0.37388 | +0.26045 | +0.31239 | +0.08326 | +0.16145 |
| protected_only_4pct_compensated | +0.18666 | +0.21590 | +0.25427 | +0.06183 | +0.07555 |
| contrastive_4pct_compensated | +0.19294 | +0.21108 | +0.15359 | +0.06209 | +0.08281 |
| broader_4pct_compensated | +0.00283 | +0.00234 | -0.00051 | -0.00051 | +0.00052 |

Reference counts: arithmetic: 32, planning: 11, program: 32, rules: 29, temporal: 21.

Broader minus current selection at 4%: paired bootstrap 95% intervals for complete-solution CE. Negative is better preservation.

| Family | Mean difference | 95% interval |
|---|---:|---:|
| arithmetic | -0.19041 | [-0.22211, -0.15917] |
| planning | -0.21245 | [-0.25578, -0.16684] |
| program | -0.15247 | [-0.20176, -0.10700] |
| rules | -0.06230 | [-0.07136, -0.05363] |
| temporal | -0.08044 | [-0.10147, -0.06012] |

## Artifacts and reproducibility

Run:

```bash
uv run python -m scripts.test_selection_mechanism --offline
uv run python -m scripts.evaluate_reference_traces
uv run python -m scripts.check_causal_precision
uv run python -m scripts.summarize_selection_mechanism
```

`configuration.json` records predeclared settings. `metrics.json` contains family accuracy, CE/margin changes, and paired wins/losses. `paired_comparisons.json` includes all family confidence intervals. `causal.json` and `causal_summary.json` contain finite-ablation measurements. `masks/`, `family_scores.pt`, `activation_means.pt`, and `evaluations/` retain the interventions, calibration estimates, and raw predictions.
