# ModelCompression

Frozen-weight MLP channel-scoring and deletion experiments for compatible
Hugging Face causal language models. The original results use pinned
Qwen3-1.7B weights; the quick-start interface also supports smaller Colab
models and custom model IDs.

## Run in Google Colab

1. Publish this repository to GitHub (instructions below).
2. Open [`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb)
   from GitHub with Google Colab.
3. Select a GPU runtime, set `REPO_URL`, then use the model and dataset dropdowns.
4. Run all cells. The default smoke test uses Qwen3-0.6B and three examples.

The notebook installs the package, downloads model weights from Hugging Face,
runs generation, and writes channel scores to `outputs/channel_scores.pt`.
Increase `LIMIT` only after the smoke test works; full gradient scoring is much
slower and more memory-intensive than generation.

The equivalent terminal flow is:

```bash
git clone https://github.com/YOUR_USERNAME/model-compression.git
cd model-compression
pip install -e .
model-compression list
model-compression generate --model qwen3-0.6b
model-compression score --model qwen3-0.6b --dataset mquake-cf3k --limit 3
```

`--model` accepts a listed preset or any Hugging Face repository ID.
`--dataset` accepts a listed preset or a JSONL path. Custom JSONL rows need at
least `prompt` and `target` fields; `id` and `score_answer` are optional. The
model must expose the common `model.layers[*].mlp.down_proj` structure used by
Qwen, Llama, and related decoder-only architectures.

To put this local Git repository on GitHub:

```bash
git remote add origin https://github.com/YOUR_USERNAME/model-compression.git
git push -u origin main
```

Raw source datasets, virtual environments, logs, predictions, and large tensor
checkpoints are intentionally ignored. The derived calibration JSONL files
needed by the dropdown are versioned, so a clone can run without rebuilding
the datasets.

## Channel-selection mechanism pilot

Read the [pilot findings](artifacts/selection_mechanism_pilot/findings.md) and
[detailed results](artifacts/selection_mechanism_pilot/report.md).
Reproduce it with:

```bash
uv run python -m scripts.test_selection_mechanism --offline
uv run python -m scripts.evaluate_reference_traces
uv run python -m scripts.check_causal_precision
uv run python -m scripts.summarize_selection_mechanism
```

This compares protected-only selection (`recall_weight=0`), the current
contrastive selector (`0.05`), and additional protection from baseline-generated
synthetic solutions whose final answers pass procedural checks. It uses the
same unmasked parent and matched deletion counts. The additional families are
arithmetic, short Python execution, rule deduction, temporal ordering, and
minimum-cost route choice. Their intermediate reasoning is not independently
verified. The synthetic validation split is instance-disjoint and uses the same
task templates, so this remains a pilot rather than a broad reasoning benchmark.

The pilot also measures individual and small-group causal ablations, and tests
constant calibration-mean bias compensation on identical 4% masks. Compensation
adds the deleted channels' mean down-projected contribution; it does not update
surviving weights, but is a separate experimental arm from pure deletion.
The existing pure-deletion constraints continue to describe that original arm.

Additional family tensors can be passed to `select_iterative_batch.py` with
`--family-scores PATH`. Each family contributes to the maximum protected
percentile cost and must pass the protected quantile threshold. Family scores
must be measured under the same parent mask; automatic iterative rescoring of
the synthetic families is not implemented by the existing sweep runner.

The sweep runner now accepts optional per-family acceptance budgets:

```text
--max-protected-family-exact-drop 0.02
--max-protected-family-margin-drop 0.10
--max-protected-family-ce-increase 0.10
```

Margin and CE limits are in nats per answer token and enable likelihood
evaluation before **every** acceptance decision. Family metrics are grouped by
the input `family` field, falling back to condition. Existing aggregate budgets
still apply, and the defaults are unchanged. These flags validate the families
actually present in the supplied evaluation data; they do not add tasks on their
own. The pilot's broader likelihood probes directly teacher-force the short
answer, while generated final-answer accuracy evaluates the complete response.

The selector's `first_order_abs_bound` diagnostics bound the linearized loss
change only, not the true finite-deletion effect.

## Protected math-reasoning family (OpenMathInstruct-2)

The matched MQuAKE conditions cover recall, extraction, and compositional graph
reasoning, but not arithmetic or multi-step numeric derivation. A protected
`openmath` family is built from one pinned shard of
[`nvidia/OpenMathInstruct-2`](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)
(revision `469216e3f46f4dacf476b382e192485ea51a143e`, `data/train-00000-of-00032.parquet`,
CC BY 4.0). The shard is filtered to original — not augmented — GSM8K and MATH
problems with integer answers, deduplicated by normalized problem text, then
reduced by the project's baseline-known rule: keep only problems the frozen
baseline already solves free-running, in the pilot's explicit-working format.

```bash
uv run --extra data python scripts/prepare_openmath_family.py --offline
uv run python scripts/merge_calibration_corpora.py \
  --corpus data/calibration/mquake_remastered_cf9k_v2 \
  --corpus data/calibration/openmath_v2 \
  --output data/calibration/mquake_cf9k_openmath_v2
uv run python scripts/score_channels_v2.py --offline \
  --input data/calibration/mquake_cf9k_openmath_v2/calibration.jsonl \
  --output artifacts/channel_scores/mquake_cf9k_openmath_v2
```

The records are `condition: reasoning`, `family: openmath`, so they enter the
protected side of the contrastive screen automatically; no `--family-scores`
tensor is needed. Two harness changes support them:

- **Per-record generation budgets.** `evaluate_robustness.py` now reads an
  optional `max_new_tokens` field per record and never mixes budgets inside one
  generation batch. Math records carry 384 tokens; MQuAKE records keep the
  sweep default. Without this the model's derivations are cut off mid-working
  and every math row scores zero.
- **Scoring the derivation, not the last token.** `score_channels_v2.py` reads
  an optional `score_answer` field, set to the baseline's own accepted
  derivation including its final-answer line. Scoring the bare numeral while
  supplying no working would miss the machinery that produces the working,
  which is the trap described in *Loss construction and common traps* above.

### Corpus result — September 14, 2026

Of 1,200 sampled candidates the baseline solved 611 (50.0% of the calibration
candidates, 52.6% of the validation candidates), far above the 1.75% that killed
2WikiMultiHopQA, because solving a supplied problem does not depend on tail
entity knowledge. 320 calibration and 128 validation problems were kept
(263/104 GSM8K, 57/24 MATH). About 30% of attempted derivations run past the
384-token budget and are rejected as truncated, which biases the family toward
shorter derivations.

The direct-answer format was tried first and abandoned: Qwen3-1.7B works
through a problem regardless of an instruction to emit only a number, so
short-budget scoring accepted only 1.5% of candidates and measured formatting
compliance rather than arithmetic.

**Limitation.** The family is a protected-behaviour guard, not a math benchmark.
It measures whether a masked model still reproduces derivations the unmasked
model already produced, on problems selected for the baseline getting them
right.

## Deep deletion sweep with the math family

```bash
screen -dmS openmath_sweep bash scripts/run_openmath_deep_sweep.sh
screen -dmS openmath_guard bash scripts/guard_openmath_deep_sweep.sh
```

The sweep targets 55% cumulative deletion from an empty mask, 4% per round,
`--recall-weight 0.30` (the configuration that removed the most recall content
overnight), `--protected-quantile 0.9`, `--max-layer-fraction 1.0`, and
`--min-backoff-fraction 0.01` so the endgame cannot spend hours halving batches
for a tenth of a percent.

Predeclared acceptance budgets, deliberately wider than the two-point budgets
that stopped `cf9k_sweep_3` and `cf9k_sweep_4` near 22–25%:

| budget | limit |
|---|---|
| extraction / reasoning exact match | 10 points |
| extraction / reasoning containment | 5 points |
| worst protected family exact match | 20 points |

Containment is the trustworthy content signal — the overnight sweeps showed
exact match collapsing on answer formatting while containment held — so the
exact-match budgets are wide and the per-family gate carries the weight. The
family gate applies to the worst of `extraction`, `reasoning`, and `openmath`;
because every kept math problem is correct at baseline by construction, a
20-point drop there is a genuine loss of 26 of 128 derivations.

Reaching 55% is not an expectation. Both overnight sweeps stopped on the
protected budget well short of their 80% target, and the honest output of this
run is the fraction at which free-running protected behaviour actually breaks.

## Qwen3 baseline setup

The baseline is [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B),
pinned to checkpoint revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
The original, unquantized weights are frozen. CUDA uses BF16; CPU uses FP32.
The loader uses PyTorch SDPA attention; channel instrumentation occurs in the
MLPs and does not require eager attention.

From this directory, with Python 3.10–3.13 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run qwen --download-only
uv run qwen --offline "What is 2 + 2? Answer briefly."
uv run python scripts/check_baseline.py
uv run qwen --offline --thinking --max-new-tokens 512 "If A implies B and B implies C, what follows from A?"
```

`uv.lock` pins dependencies, including a CUDA 12.8 PyTorch build for the local
NVIDIA GPU. Downloads use the standard Hugging Face cache (`~/.cache/huggingface`,
unless overridden by `HF_HOME`); weights are not stored in the source tree.
Allow about 3.5 GB for the checkpoint plus several GB for PyTorch/CUDA dependencies.
`--offline` requires the download to have completed. `--device cpu` forces CPU
execution. Non-thinking generation is the default; generation uses the sampling
settings recommended in the model card and seed 42 (overridable with `--seed`).
Thinking output can be truncated by the token budget; increase it for harder tasks.
An 8 GB GPU supports short baseline runs, but long contexts and gradient scoring
need separate memory budgeting.

For experiments:

```python
from model_compression.qwen import load_baseline

model, tokenizer = load_baseline(local_files_only=True)
mlp = model.model.layers[0].mlp
# Channel j corresponds to gate_proj/up_proj row j and down_proj column j.
```

The loader leaves autograd available, with every original parameter frozen and
the model in evaluation mode. The scorer introduces differentiable auxiliary
gates before computing gate gradients and does not use `inference_mode()` around
that measurement. The generation CLIs use inference mode.
This setup does not yet implement iterative deletion, physical compaction, or the
broader protected-task suites beyond matched MQuAKE evaluation.

## First channel-screening experiment

The filtered matched corpus can now be used to compute answer-token gate-gradient
scores for every MLP channel. Original model parameters remain frozen. The scorer
uses one matched example per microbatch, keeps recall gradients signed, takes
absolute gradients for the protected extraction/reasoning conditions, records
stable calibration-half statistics, and writes resumable checkpoints.

```bash
uv run python scripts/score_channels.py --offline
```

The default output is `artifacts/channel_scores/mquake_qwen3_1.7b/`. To validate
the screen, select small contrastive, low-importance, protected-important, and
layer-matched random cohorts, then evaluate them as temporary zero masks:

```bash
uv run python scripts/select_channel_masks.py \
  artifacts/channel_scores/mquake_qwen3_1.7b/scores.pt

uv run python scripts/evaluate_masks.py --offline \
  --mask artifacts/channel_masks/initial_validation/contrastive.json \
  --mask artifacts/channel_masks/initial_validation/low_importance.json \
  --mask artifacts/channel_masks/initial_validation/protected_important.json \
  --mask artifacts/channel_masks/initial_validation/random_layer_matched.json
```

Evaluation defaults to the validation split and always records an unmasked
baseline in the same run. The test split is sealed unless `--allow-test` is
explicitly supplied. Use `--limit N` for smoke tests; it limits matched groups,
not individual condition rows. These masks are diagnostic interventions, not a
physically compact checkpoint.

Because strict exact match can count answer-format changes as failures, audit any
observed accuracy change separately (containment remains diagnostic, not a
replacement metric):

```bash
uv run python scripts/audit_evaluation.py
```

### Initial screen-validation result — September 13, 2026

The complete 669-group calibration run produced scores for all 172,032 MLP
channels. Calibration-half Spearman correlations were 0.899 for recall, 0.975
for extraction, and 0.816 for reasoning. The reasoning top-1% overlap was only
0.235, so extreme-tail reasoning rankings require more caution than the aggregate
correlation suggests.

On all 350 validation groups, temporary deletion of the 32 contrastive channels
changed strict recall accuracy from 99.7% to 83.1%, while extraction remained
97.4% and reasoning remained 99.4%. Neither the 32 low-importance channels nor
the layer-matched random controls changed accuracy. The protected-important
control reduced both recall and extraction by 4.6 percentage points.

This is evidence that the gradient screen finds functionally consequential
channels, but it is not yet evidence of broad factual forgetting: 54 of the 58
contrastive strict-match losses still contained an accepted answer, predominantly
format variants such as `Europe` becoming `Continent: Europe` or
`Continental Europe`. Only four losses omitted the accepted answer. The next
experiment must therefore test paraphrases, relations, and answer-margin changes
and must avoid letting one frequent target dominate selection.

## Target-balanced robustness screen (v2)

The v2 experiment equalizes the total calibration weight of each normalized
answer within each condition, scores recall with both answer-token
cross-entropy and a correct-versus-relation-matched-alternatives margin, and
keeps those rankings separate as well as producing a consensus cohort. It does
not read or copy the sealed test split.

```bash
uv run --extra data python scripts/prepare_v2_data.py
uv run python scripts/score_channels_v2.py --offline
uv run python scripts/select_channel_masks_v2.py \
  artifacts/channel_scores/mquake_qwen3_1.7b_v2/scores.pt
uv run python scripts/evaluate_robustness.py --offline \
  --mask artifacts/channel_masks/robust_validation/contrastive_ce_balanced.json \
  --mask artifacts/channel_masks/robust_validation/contrastive_margin_balanced.json \
  --mask artifacts/channel_masks/robust_validation/contrastive_consensus.json \
  --mask artifacts/channel_masks/robust_validation/random_layer_matched_seed_42.json \
  --mask artifacts/channel_masks/robust_validation/random_layer_matched_seed_43.json \
  --mask artifacts/channel_masks/robust_validation/random_layer_matched_seed_44.json
```

Robust evaluation includes exact match, answer containment, target-macro and
relation-macro results, answer margins, an instruction-first prompt format, and
the two additional source-written paraphrases available for multi-hop reasoning.
Do not begin iterative deletion unless content loss and margin degradation span
multiple targets/relations while extraction and reasoning stay inside a
predeclared performance budget.

To benchmark the optimized evaluation and channel-scoring hot paths on a small
sample:

```bash
uv run python scripts/benchmark_evaluation.py --offline
```

Increase `--groups` to measure padding overhead on more of the validation set.
The report covers length-bucketed padding overhead, selected-position margin
scoring against full-vocabulary logits, per-candidate evaluation cost before and
after baseline caching plus deferred margins, and sequential against batched
recall-distractor gate gradients. Run it with `--device cpu` to check the
batched gradient in FP32, where the batched and sequential paths must agree to
rounding.

### Measured optimization results — September 13, 2026

On an RTX 4060 Laptop GPU (BF16, SDPA, batch size 8, 32 new tokens):

| Optimization | Before | After | Change |
|---|---|---|---|
| Per-candidate evaluation, 7,040 variant rows | 1,191.8 s | 348.3 s | 3.42x faster |
| Recall-distractor gate gradients, per example | 0.168 s | 0.106 s | 1.60x faster |
| Calibration forward/backward passes, 2,007 examples | 4,002 | 2,676 | 33% fewer |
| Margin batch scoring, 8 answers | 0.0619 s | 0.0558 s | 1.11x faster |

The evaluation figure combines both evaluation optimizations: a fingerprinted
baseline cache removes the repeated unmasked run (about half the work), and
`--no-margins` screening removes the four-answer margin pass from both modes.
Deferring margins is the larger share, because margins score four answers per
prompt while generation runs once.

Batching the three recall distractors into one backward pass is a numerical
no-op: the gradient of the mean loss equals the mean of the per-answer
gradients. In FP32 on CPU the two paths agree to 3.3e-07 relative error. The
GPU comparison shows 3.2% relative difference and 0.998 cosine similarity,
which is BF16 rounding rather than a change of objective. Per-example absolute
CE gradients are untouched; only the recall margin term is batched.

### V2 result — September 13, 2026

The complete target-balanced run scored all 669 calibration groups and selected
three 32-channel cohorts. The CE and margin top cohorts overlapped on 8 channels;
the consensus shared 19 channels with CE and 12 with margin.
Calibration-half Spearman correlations were 0.524 for balanced CE recall,
0.684 for recall margin, 0.951 for extraction sensitivity, and 0.831 for
reasoning sensitivity. Corresponding top-1% overlaps were 0.475, 0.616, 0.727,
and 0.278, so protected reasoning extremes still warrant conservative batches.

Across 350 validation groups and 2,800 prompt variants per mode, all three
selected masks reduced recall while leaving extraction and reasoning exact-match
accuracy unchanged. The consensus mask reduced recall answer containment from
99.0% to 97.3% and recall answer margin by 0.347 nats on average; three
layer-matched random controls changed recall margin by +0.023, +0.009, and
+0.048 nats. It caused 13 paired content losses across the variants versus
1, 2, and 1 for the random controls. These losses included capital, continent,
country-of-citizenship, and headquarters-location relations rather than one
answer alone.

On original prompts only, the consensus changed recall exact match from 350/350
to 334/350 and answer containment from 350/350 to 343/350, with extraction fixed
at 341/350 and reasoning fixed at 348/350. The instruction-first format itself
caused substantial baseline exact-format variation, so containment and margin
remain necessary alongside strict match.

Validated locally on September 13, 2026: offline BF16 GPU generation returned
`4.` for the arithmetic example. The structure/gradient check passed for 28 layers,
hidden width 2,048, and MLP width 6,144, with all 1,720,574,976 original parameters
frozen. That short gradient check peaked at 3.79 GiB of allocated CUDA memory;
this is not an estimate for a full calibration run.

Research plan — September 13, 2026

## Objective

Shrink an existing language model by physically deleting MLP intermediate channels that are dispensable for reasoning, preferentially removing channels that support encyclopedic recall. Preserve the surviving weights and their correspondence to the original model so interpretability findings can be tested in the complete model.

The central research question is: **Does distinguishing factual recall from supplied-context reasoning allow more channel removal at a fixed reasoning-performance budget than ordinary task-focused pruning?**

Removing most geography, history, biographies, and other encyclopedic knowledge is an aspiration, not an assumed outcome. The experiment should measure compression, reasoning retention, factual recall, and mechanistic transfer separately.

## Fixed constraints

- No retraining, fine-tuning, distillation, or recovery training.
- Freeze every original model weight. Backpropagation is allowed for measurement; there is no optimizer step.
- Focus on MLP intermediate channels, preserving residual-stream width and model depth.
- Initially use pure deletion: no learned gates, bias compensation, weight reconstruction, or changes to surviving weights.
- Maintain an exact index map from compact-model channels to the original checkpoint.
- Use calibration data to choose deletions and separate held-out data to evaluate them.

## 1. Define what should survive

Retain language understanding, extraction from supplied context, and reasoning over explicit premises. Losing a particular person's birthplace should not prevent the model from reading a supplied biography or inferring a birth country from a supplied map.

Construct three matched conditions:

| Condition | Example structure | Purpose |
|---|---|---|
| Recall | Ask a known birthplace without supplying it | Measure use of parametric facts |
| Extraction | Supply the birthplace and ask the same question | Protect reading, entity processing, and output generation |
| Reasoning | Supply a birthplace plus containment relations; ask for the country | Protect composition over explicit premises |

Add fictional worlds with randomized names and underlying relations, not merely a stable renaming of real facts. Include temporal reasoning, graph traversal, arithmetic, small-program execution, rule application, and planning where feasible. Include insufficient-evidence cases.

Match prompt style, answer format, vocabulary, and difficulty where practical. A contrast between Wikipedia prose and mathematical notation alone could identify superficial domain differences instead of lookup-specific computation.

Use facts the baseline demonstrably knows. Keep evaluation splits separate by entity, template, and task family where possible. Broad forgetting should be tested on untargeted entities and relations as well as the deletion-calibration distribution.

## 2. Cheap channel screening with activation × gradient

In a gated MLP, let a[t,j] be intermediate channel j after the elementwise gated nonlinearity and before the down projection. Insert an auxiliary scalar gate m[j], shared across token positions:

    a_masked[t,j] = m[j] * a[t,j]
    m[j] = 1

For a single example with loss L:

    g[j] = dL/dm[j] = sum_t a[t,j] * dL/da[t,j]
    predicted deletion loss change ≈ -g[j]

The second line is a first-order Taylor approximation to changing m[j] from 1 to 0. A positive predicted change means deletion is expected to hurt the measured task. One forward/backward pass scores all instrumented channels; there is no separate pass per channel.

For each channel, accumulate:

    K[j] = mean over recall examples of -g[j]
    R[j] = mean over reasoning examples of abs(g[j])
    E[j] = mean over extraction examples of abs(g[j])

K estimates directional damage to recall. R and E conservatively measure sensitivity of protected behavior. Take absolute values per example before averaging, so effects on different examples do not cancel. Microbatch size 1 is a simple starting implementation; larger batches need per-example statistics or activation-gradient hooks that retain the example dimension.

Keep reasoning-family scores separate and inspect upper-tail sensitivity. An average can hide a channel crucial for a rare skill. Also record signed scores, uncertainty across samples, and ranking stability across independent calibration subsets.

Select channels with low R and E, then prioritize high positive K. Prefer thresholding protected-task scores over dividing K by R, which can create unstable rankings near zero. Low importance on all conditions is also useful for compression; keep those channels as a separate candidate category.

This score is a proposed contrastive screening heuristic. It estimates local task sensitivity, not an exclusive semantic role or proof that a channel stores a fact.

## 3. Loss construction and common traps

For recall and extraction, compute loss on answer tokens, normalized consistently per example. Exclude prompt tokens from the loss, but include all causal token positions in the gate-gradient sum: prompt-position activations can be important for the answer.

Cross-entropy gradients can be small on confident correct answers. As an additional screening objective, use a correct-versus-alternative log-probability margin:

    s = log p(correct answer | prompt) - log p(alternative answer | prompt)

For this score, positive ds/dm[j] predicts a loss of correct-answer margin when the channel is deleted. Do not reuse the negative-loss sign convention blindly. Use plausible alternatives of matched form and several alternatives to check robustness. Keep margin and cross-entropy rankings separate initially; their scales differ.

For reasoning models, obtain successful, checked traces from the original model and score prediction of reasoning tokens as well as the final answer. Scoring only the answer while providing an entire solved trace can miss the machinery needed to produce the trace. Normalize answer and trace objectives separately so length does not dominate by accident.

Teacher-forced gradients do not differentiate through discrete generated token choices. Consequently, screening must be followed by fresh free-running generation tests after masking. Good trace likelihood alone does not establish preserved reasoning.

## 4. Validate the screen before extensive pruning

Start with approximately 256–1,024 calibration examples per condition as a provisional budget, spread across domains. Increase coverage if rankings are unstable; this is not a known sufficient sample size.

1. Measure baseline recall, extraction, and generated reasoning performance.
2. Compute frozen-weight gate-gradient scores.
3. Compare rankings across two independent halves of calibration data.
4. Select a small sample of predicted lookup-selective channels, shared-important channels, low-importance channels, and layer-matched random controls.
5. Actually zero each sampled channel and compare measured effects with predictions.
6. Test small groups, because redundant channels may look individually dispensable while being jointly essential.

Proceed only if the screen predicts useful deletion effects better than simple controls. If it fails, investigate saturation, data mismatch, score aggregation, or nonlinear deletion effects before scaling the experiment. Partial gate reductions can help distinguish a poor local gradient estimate from failure of the extrapolation to full deletion.

## 5. Iterative deletion with no weight updates

Use percentage-based deletion batches. The iterative selector defaults to 1% of
the original MLP channels per round and stops at 5% cumulative deletion, which
is appropriate for the current small-data experiment. For Qwen3-1.7B this is
about 1,721 channels per full round instead of the old fixed batch of 32.

```bash
uv run python scripts/select_iterative_batch.py \
  artifacts/channel_scores/next_round/scores.pt \
  artifacts/iterative_deletion/previous/cumulative_mask.json \
  --output artifacts/iterative_deletion/next_round
```

For a better-supported full sweep, raise the cumulative target explicitly:

```bash
uv run python scripts/select_iterative_batch.py SCORES MASK --output OUTPUT \
  --target-fraction 0.80
```

Raise `--fraction` too if validation and the protected-channel eligibility pool
support still larger rounds. `--count` remains available for exact-size
diagnostic batches. These rates are experimental rather than validated safe
settings. Measure actual parameter savings as well as channel fractions.

To run selection, validation, automatic acceptance, score recomputation, and
subsequent rounds as one resumable 5% sweep from the latest accepted mask:

```bash
uv run python scripts/run_iterative_deletion.py --offline
```

Selection now treats the maximum of extraction and reasoning sensitivity
percentile ranks as a continuous protected cost. Recall-removal utility is a
small secondary term (`--recall-weight`, default 0.05), and
`--max-layer-fraction` (default 0.15) prevents the globally cheapest late layers
from absorbing an unbounded fraction of the mask. The protected quantile remains
a hard eligibility guard.

The sweep writes `state.json` plus one directory per attempted round under
`artifacts/iterative_deletion/continuous_cost_sweep/`. Re-running the same command
resumes the recorded stage.

Round evaluations screen candidates with `--no-margins`, because acceptance
reads only extraction and reasoning exact match and containment. Every round
also shares one fingerprinted `baseline_cache.json`, so the deterministic
unmasked run happens once per sweep instead of once per candidate; backoff
retries reuse it too. When the sweep finishes, either at the target or at a
protected-budget stop, it runs one full margin evaluation of the final mask into
`final_evaluation/` and records the path as `final_evaluation` in the state.
That report, not the screening rounds, is the one to cite for answer margins. By default, a candidate is rejected if any aggregate
extraction or reasoning exact-match or containment metric drops by more than
0.5 percentage points. Alternative budgets can be supplied with
`--max-extraction-exact-drop`, `--max-extraction-containment-drop`,
`--max-reasoning-exact-drop`, and `--max-reasoning-containment-drop`.

After a rejection, the runner halves the proposed batch down to
`--min-backoff-fraction` (default 0.1% of original channels). Each retry is a
safer prefix of the same continuously ranked candidate set, not automatic
acceptance of the rejected set in smaller pieces; every prefix is independently
validated against the cumulative baseline budget. Use `--no-backoff` to retain
strict all-or-nothing rounds.

On the current 350-group validation case, a single continuously ranked candidate
of 8,602/172,032 channels (5.0002%) passed without backoff. Extraction exact
match dropped 0.286 percentage points, extraction containment was unchanged,
reasoning exact match dropped 0.071 points, and reasoning containment dropped
0.143 points, all within the predeclared 0.5-point limits. The result is under
`artifacts/iterative_deletion/continuous_cost_5pct/` and does not use the sealed
test split.

After each batch:

- Evaluate cumulative masks on a selection-validation split, including free-running reasoning generation.
- Roll back batches that exceed the predeclared protected-task budget.
- Recompute screening scores on the surviving network; importance changes after deletion.
- Save the mask, channel index map, scores, configuration, and metrics.

Use a final untouched test set to report results, avoiding repeated selection against the final evaluation. Specify acceptable reasoning and extraction loss before the sweep; inspect each task family as well as aggregates. Stop at the measured tradeoff boundary rather than forcing a target compression ratio.

## 6. Physically compact the MLP

For the column-vector convention:

    MLP(x) = W_down [SiLU(W_gate x) ⊙ (W_up x)]

Deleting intermediate channel j removes row j from W_gate, row j from W_up, and column j from W_down. Remove corresponding intermediate biases if the architecture has them. Framework storage conventions must be checked before slicing.

The resulting smaller matrices preserve surviving weights. A single channel in this bias-free formulation saves 3 × residual_width parameters. Actual latency gains depend on matrix shapes and kernels; memory reduction does not imply proportional speedup.

Keep original layer IDs and channel IDs in a machine-readable mapping. Support per-layer intermediate widths in the model implementation if pruning ratios differ across layers. Verify that compact-model outputs match the complete model with the same zero masks applied, within numerical tolerance, and verify surviving tensor values against the checkpoint.

## 7. Baselines and success criteria

Compare all methods at matched parameter counts and calibration budgets:

| Method | What it tests |
|---|---|
| Random channel deletion, matched by layer | Whether selection adds value |
| Simple activation-based channel pruning | Whether gradients justify their cost |
| Protected-task gradient pruning only | Whether knowledge contrast adds value |
| Contrastive recall-versus-protected-task screening | Main proposal |

Report parameter count, checkpoint bytes, peak inference memory, prefill/decode latency under fixed settings, factual recall, extraction, and generated reasoning. Plot tradeoffs across pruning levels, with uncertainty over examples and random-control seeds.

Audit factual recall through paraphrases, aliases, candidate likelihoods, indirect clues, and languages the baseline supports. Reduced recall does not prove information erasure. Retraining-based recovery attacks are outside this project's constraints; report that limitation rather than silently adding them.

The key positive result would be more compression at matched reasoning retention, plus lower factual recall, than protected-task-only pruning. If contrastive screening adds no benefit, ordinary reasoning-focused pruning may still produce a useful smaller research model.

## 8. Transfer interpretability findings to the complete model

Preserving weights and indices enables transfer tests but does not guarantee unchanged function: upstream deletions alter surviving activations.

For each proposed reasoning mechanism, replay the same mapped intervention in:

1. The physically compact model.
2. The complete model with the pruning mask applied.
3. The unmodified complete model.

Agreement between 1 and 2 verifies implementation. Similar causal effects in 3 support transfer of the finding. Measure task-specific intervention effects and, where useful, activation patching results; output accuracy or representational similarity alone is insufficient.

## First implementation milestone

Choose one accessible open-weight checkpoint and a modest calibration corpus. Implement gate instrumentation, per-example score accumulation, temporary channel masking, and a held-out evaluation harness. Produce a report comparing predicted and actual effects for a small channel sample before writing an extensive pruning pipeline.

The initial corpus is prepared from the corrected MQuAKE-Remastered CF3k split. See
[`data/README.md`](data/README.md) for its matched-condition construction,
reproducible download command, split policy, and required baseline-known filter.

Qwen3-1.7B is selected as the initial baseline, with a local 8 GB RTX 4060 GPU.
The exact protected-task suite and acceptable performance-loss thresholds remain
to be selected. The research plan below has not yet been evaluated experimentally.

## Relevant research

- [Importance Estimation for Neural Network Pruning](https://arxiv.org/abs/1906.10771), Molchanov et al., 2019. First- and second-order Taylor importance estimates; foundational screening method, not evidence of knowledge/reasoning separation in LLMs.
- [Knowledge Neurons in Pretrained Transformers](https://aclanthology.org/2022.acl-long.581/), 2022. Factual-neuron attribution and intervention in pretrained transformers; narrow factual localization does not establish broad removable memory modules.
- [Large Scale Knowledge Washing](https://arxiv.org/abs/2405.16720), 2024. Broad factual unlearning while evaluating reasoning retention. Relevant background, but its weight edits violate this project's deletion-only constraint.
- [Mechanistic Unlearning](https://arxiv.org/abs/2410.12949), 2024. Motivation for causal localization of factual recall; its editing procedure is not the proposed frozen-weight method.
- [Fluctuation-based Adaptive Structured Pruning for Large Language Models](https://arxiv.org/abs/2312.11983), 2023. Retraining-free structured pruning. Its bias compensation should be excluded from a pure-deletion comparison or clearly separated.
- [MINI-LLM](https://arxiv.org/abs/2407.11681), 2024. Memory-conscious gradient estimation for structured pruning; a possible implementation reference if backward activation memory is prohibitive.
- [Think Before You Prune](https://arxiv.org/abs/2512.02185), 2025. Reasoning-specific structured pruning using self-generated calibration and gradient-based importance; relevant to protecting decode-time behavior.
- [On the Limits of Layer Pruning for Generative Reasoning in LLMs](https://arxiv.org/abs/2602.01997), 2026. Evidence that classification-style pruning tolerance should not be equated with preserved generated reasoning.

These papers motivate components of the plan. None establishes that pure channel deletion can remove most encyclopedic knowledge while preserving the original model's broad reasoning mechanisms.
