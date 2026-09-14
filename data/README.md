# Calibration data

The initial calibration source is the corrected
[`MQuAKE-Remastered CF3k`](https://huggingface.co/datasets/henryzhongsc/MQuAKE-Remastered)
split. Both its repository revision and file SHA-256 are pinned in
`scripts/prepare_mquake_calibration.py`. The dataset is CC BY 4.0; preserve its
attribution when redistributing derived records.

Run:

```bash
uv run --extra data python scripts/prepare_mquake_calibration.py
```

This downloads the 3,000-case source file and writes three JSONL files under
`data/calibration/mquake_remastered_cf3k/`. Every source case yields a matched
group:

- `recall`: the final original single-hop question, with no facts supplied;
- `extraction`: the final counterfactual fact and its direct question;
- `reasoning`: the complete 2–4-hop counterfactual chain and a composed question.

Counterfactual facts are useful here because the protected conditions cannot be
solved reliably by merely repeating the model's stored real-world knowledge.
The split keeps repeated first-hop subjects together. Shared high-frequency
intermediate entities (such as countries) can still cross splits, so the final
report should describe this as root-entity-disjoint rather than fully
entity-disjoint.

## Required baseline filter

The generated records are candidates, not the final calibration set. Before
computing channel scores, run the unchanged pinned Qwen baseline on each recall
record and retain only matched groups whose recall answer is correct under the
project's alias-aware exact-match rule. Freeze that accepted-ID list before any
pruning experiment. Do not use validation or test outcomes to choose channels.

Run the filter with deterministic, non-thinking generation:

```bash
uv run python scripts/filter_mquake_baseline_known.py --offline
```

This writes the filtered matched groups, the frozen accepted recall-ID list, all
baseline predictions, and a reproducibility manifest under
`data/calibration/mquake_remastered_cf3k_baseline_known/`. Exact match applies
Unicode normalization and case folding, removes punctuation and English
articles, collapses whitespace, and accepts either the target or any listed
alias. It does not use substring or fuzzy matching.

MQuAKE-Remastered covers factual recall, contextual extraction, compositional graph
reasoning, and 2–4-hop depth. It does not cover arithmetic, program execution,
planning, or truly fictional entity names. Add those as separate protected-task
families later; do not pool them into the matched MQuAKE contrast without
reporting per-family scores.

## Protected math-reasoning family

`data/calibration/openmath_v2/` holds a protected `openmath` family built from a
pinned shard of `nvidia/OpenMathInstruct-2` (CC BY 4.0; preserve its attribution
when redistributing derived records). See the README section *Protected
math-reasoning family (OpenMathInstruct-2)* for construction, the baseline-known
rule, and its limitations.

`data/calibration/mquake_cf9k_openmath_v2/` is the concatenation of the MQuAKE
CF9k v2 corpus and that family, produced by
`scripts/merge_calibration_corpora.py`. The merge refuses duplicate record ids,
group ids spanning corpora, and any split that loses one of the three matched
conditions. The sealed test split is neither read nor written.
