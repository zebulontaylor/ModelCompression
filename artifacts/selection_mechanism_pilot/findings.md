# Findings from the channel-selection pilot

**Keep the gate-gradient screen, but broaden the protected calibration and evaluate each task family separately. Bias compensation was not a consistent improvement.**

The experiment compared protected-only selection, the current recall-weighted selector, and protection using successful generated solutions from five additional families. Every mask starts from the same unmasked checkpoint. There were eleven masked evaluations plus a baseline, using 256 MQuAKE validation groups and 32 examples each of arithmetic, program execution, rule deduction, temporal ordering, and route planning. All original weights remained frozen; these are temporary masks, not physically compact checkpoints.

## What changed at 4% channel deletion

| Method | Arithmetic accuracy | Program accuracy | Temporal accuracy | Rule accuracy | Recall containment |
|---|---:|---:|---:|---:|---:|
| Baseline | 100.0% | 100.0% | 65.6% | 90.6% | 99.2% |
| Protected-only | 96.9% | 87.5% | 40.6% | 100.0% | 96.5% |
| Current recall-weighted | 96.9% | 87.5% | 37.5% | 93.8% | 96.5% |
| Broader protection | 100.0% | 100.0% | 53.1% | 84.4% | 95.7% |
| Broader protection + bias compensation | 100.0% | 100.0% | 56.3% | 78.1% | 95.7% |

MQuAKE extraction and reasoning accuracy were unchanged across these 4% masks. The original matched suite would therefore have missed large losses in program execution and temporal ordering. Aggregate accuracy would also obscure tradeoffs: some masks improved rule or planning accuracy while damaging other families.

Broader protection recovered all four lost program answers relative to the current selector and improved temporal performance, but it did not preserve every family. None of the tested masks met all predeclared pilot thresholds. With only 32 examples per synthetic family, generated-accuracy differences remain preliminary: four paired program improvements and no regressions give an exact two-sided paired p-value of 0.125. The report includes paired counts, exploratory bootstrap intervals, and exact paired tests.

## Stronger evidence from complete solutions

A separate, post-hoc check scored 125 fixed baseline-generated solutions whose final answers were correct. At 4% deletion, the current selector increased mean solution-token loss by 0.063–0.215 nats/token across the five families. Broader protection changed it by approximately -0.00007 to +0.00218 nats/token. Paired bootstrap intervals favor broader protection in every family for this likelihood objective.

This supports using generated solutions for calibration. It does not guarantee preserved reasoning: nearly unchanged average trace likelihood coexisted with final-answer regressions. Free generation and per-family acceptance remain necessary. Intermediate reasoning steps were not independently verified.

## The gradient calculation itself

Twelve individual channels and three four-channel groups were tested at 25% attenuation and full deletion on sixteen matched validation groups. FP32 controls were necessary because BF16 rounding obscured tiny effects.

In FP32, predicted versus measured single-channel loss changes had Pearson correlations of 0.993 for recall and 0.999 for extraction at 25% attenuation. At full deletion they fell to 0.946 and 0.972. Reasoning-loss effects were tiny and much less predictable. Group effects also departed from sums of individual effects: for one protected-important group, the summed recall loss change was -0.0412 nats, while the actual group change was -0.0225.

The local screen is useful. Its additive predictions and absolute-gradient sums are not guarantees about finite, multi-channel deletion.

## Recall weighting and bias compensation

At 4%, recall weighting did not reduce recall containment beyond protected-only selection. At 5% it reduced containment further, but both methods substantially damaged protected synthetic tasks. This pilot does not demonstrate more compression at matched reasoning retention from the recall term.

Bias compensation replaced deleted channels with their mean contribution, estimated only from protected calibration inputs. It barely changed complete-solution losses and did not improve the worst family accuracy drop for any of the three masks. Its generated effects were mixed: with broader protection it gained one temporal answer but lost two rule answers. I would keep it optional rather than enable it by default.

## Implementation and next experiment

The repository now supports additional family sensitivities in selection, optional per-family accuracy/margin/CE acceptance budgets, explicit final-answer grading, calibration-mean compensation, reproducible pilot evaluation, and BF16/FP32 causal checks. Existing default pruning settings remain unchanged.

The next justified experiment is a smaller-batch iterative sweep with broader, checked-solution calibration and per-family generation budgets, followed by substantially larger and more diverse held-out evaluation. The sealed test split was not used here. One-shot eligibility failures at 5% or 10% are limits of this score snapshot and its thresholds, not ceilings on iterative compression.

[Detailed results and reproduction commands](report.md) · [Per-family metrics](metrics.json) · [Raw loss audit](loss_audit.json) · [FP32 causal results](causal_fp32/summary.json)
