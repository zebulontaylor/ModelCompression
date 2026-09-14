# Findings

The sweep-4 effect is not limited to MQuAKE, but it is also not cleanly
recall-selective outside MQuAKE.

- External factual recall declined on both datasets. PopQA containment fell
  4.25 points (14.38% to 10.12%; 45 losses versus 11 gains; exact paired
  p=5.38e-6). Of the 115 PopQA questions known by the baseline, only 60.9%
  remained known. CounterFact containment fell 2.13 points (35.75% to 33.62%);
  this net change was not significant at the 0.05 level (54 losses versus 37
  gains; p=0.093), although only 81.1% of baseline-known cases were retained.
- Reasoning was task-dependent. ARC-Challenge was unchanged within noise
  (71.0% to 71.7%; p=0.754), so the mask did not cause a uniform loss of all
  reasoning. GSM8K collapsed from 71.9% to 4.7% (45 losses versus 2 gains;
  p=1.60e-11).
- The GSM8K result is not an answer-extraction artifact. All 64 baseline
  responses terminated within the 256-token allowance, while 22/64 masked
  responses hit the limit. Inspection shows repetitive text and elementary
  arithmetic corruption in the masked outputs, including those that terminate.

Consequently, the factual-forgetting signal generalizes beyond MQuAKE, most
clearly to PopQA. The stronger claim that the sweep preserves unrelated
reasoning does not generalize: it preserves short multiple-choice science
reasoning but severely damages free-generation arithmetic reasoning. This mask
should not be described as broadly selective without adding external protected
families during selection and acceptance.

These are deterministic, single-sample evaluations. The paired tests quantify
within-sample changes, but the 64-example GSM8K result should still be replicated
on a larger sample before estimating the exact size of the degradation.
