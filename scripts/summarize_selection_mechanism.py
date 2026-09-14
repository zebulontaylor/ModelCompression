#!/usr/bin/env python3
"""Summarize the channel-selection pilot, including paired uncertainty."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from model_compression.channels import read_jsonl, is_correct, contains_accepted_answer
from scripts.evaluate_robustness import scored_answer
from scripts.test_selection_mechanism import metrics


def bootstrap_mean(values, seed=42):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist()}


def binary_comparison(values):
    result = bootstrap_mean(values)
    wins, losses = sum(x > 0 for x in values), sum(x < 0 for x in values)
    discordant = wins + losses
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant) if discordant else 1.
    return {**result, "wins": wins, "losses": losses, "paired_exact_two_sided_p": p}


def ranks(values):
    values = np.asarray(values)
    return np.array([(np.sum(values < x) + (np.sum(values == x) - 1) / 2) for x in values])


def correlation(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def causal_statistics(causal):
    summary = {}
    for condition in ("recall", "extraction", "reasoning"):
        summary[condition] = {}
        for strength in (.25, 1.):
            cases = [r for r in causal["interventions"] if r["strength"] == strength and r["kind"] == "single"]
            x = [r["effects"][condition]["predicted_ce_increase"] for r in cases]
            y = [r["effects"][condition]["actual_ce_increase"] for r in cases]
            summary[condition][str(strength)] = {
                "pearson": correlation(x, y), "spearman": correlation(ranks(x), ranks(y)),
                "mean_abs_prediction_error": float(np.mean(np.abs(np.array(x) - np.array(y)))),
                "actual_abs_above_linear_abs_bound": sum(abs(r["effects"][condition]["actual_ce_increase"]) > r["effects"][condition]["first_order_abs_bound"] for r in cases),
                "single_channel_count": len(cases),
            }
        interactions = {}
        for cohort in ("recall_selective", "protected_important", "random"):
            cases = [r for r in causal["interventions"] if r["cohort"] == cohort and r["strength"] == 1.]
            group = next(r for r in cases if r["kind"] == "group")
            single_sum = sum(r["effects"][condition]["actual_ce_increase"] for r in cases if r["kind"] == "single")
            actual = group["effects"][condition]["actual_ce_increase"]
            interactions[cohort] = {"sum_single_effects": single_sum, "group_effect": actual,
                                    "interaction_residual": actual - single_sum}
        summary[condition]["group_interactions"] = interactions
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path("artifacts/selection_mechanism_pilot"))
    args = parser.parse_args()
    root = args.output
    results = json.loads((root / "metrics.json").read_text())
    config = json.loads((root / "configuration.json").read_text())
    rows = read_jsonl(root / "validation.jsonl")
    families = sorted({r["family"] for r in rows})
    calibration_filter = json.loads((root / "calibration_filter.json").read_text())
    evaluations = {p.stem: json.loads(p.read_text()) for p in (root / "evaluations").glob("*.json")}
    source_rows = {r["id"]: r for r in rows}
    # Regrade every arm consistently from saved raw outputs. In particular, an
    # empty markdown Final answer heading must not consume the next answer line.
    for mode, evaluation in evaluations.items():
        for prediction in evaluation["predictions"]:
            row = source_rows[prediction["id"]]
            value = scored_answer(prediction["prediction"], row)
            prediction["scored_prediction"] = value
            prediction["exact_correct"] = is_correct(value, row["target"], row["aliases"])
            prediction["contains_answer"] = contains_accepted_answer(value, row["target"], row["aliases"])
        (root / "evaluations" / f"{mode}.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    results = {mode: metrics(evaluations[mode], evaluations["baseline"], rows) for mode in results}
    (root / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    def index(mode, key):
        return {x["id"]: x for x in evaluations[mode][key]}
    baseline_predictions = index("baseline", "predictions")
    audit = {}
    for mode in results:
        if mode == "baseline":
            continue
        changed = []
        for row in evaluations[mode]["predictions"]:
            before = baseline_predictions[row["id"]]
            if before["exact_correct"] and not row["exact_correct"]:
                if row["contains_answer"]:
                    category = "accepted_answer_in_final_but_noncanonical_format"
                elif not row.get("scored_prediction", row["prediction"]):
                    category = "missing_final_answer"
                else:
                    category = "final_answer_omits_accepted_answer"
                changed.append({"id": row["id"], "family": row["family"], "target": row["target"],
                                "category": category, "baseline": before["prediction"], "masked": row["prediction"]})
        audit[mode] = changed
    (root / "loss_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    comparisons = {}
    pairs = [(mode, "baseline") for mode in results if mode != "baseline"]
    for fraction in ("1", "4", "5"):
        pairs.extend([(f"contrastive_{fraction}pct", f"protected_only_{fraction}pct"),
                      (f"broader_{fraction}pct", f"contrastive_{fraction}pct")])
    pairs.extend([(mode, mode.removesuffix("_compensated")) for mode in results if mode.endswith("_compensated")])
    for left, right in pairs:
        if left not in evaluations or right not in evaluations:
            continue
        lp, rp = index(left, "predictions"), index(right, "predictions")
        lm, rm = index(left, "margins"), index(right, "margins")
        comparison = {}
        for family in families:
            # One original prompt per family per group; resampling rows is thus
            # resampling independent group units for each family comparison.
            ids = [r["id"] for r in rows if r["family"] == family]
            comparison[family] = {
                "exact_delta": binary_comparison([lp[i]["exact_correct"] - rp[i]["exact_correct"] for i in ids]),
                "containment_delta": binary_comparison([lp[i]["contains_answer"] - rp[i]["contains_answer"] for i in ids]),
                "margin_delta": bootstrap_mean([lm[i]["margin"] - rm[i]["margin"] for i in ids]),
            }
        comparisons[f"{left}_minus_{right}"] = comparison
    (root / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")

    causal_path = root / "causal.json"
    causal_summary = {}
    if causal_path.exists():
        causal_summary = causal_statistics(json.loads(causal_path.read_text()))
        (root / "causal_summary.json").write_text(json.dumps(causal_summary, indent=2) + "\n")
    fp32_path = root / "causal_fp32" / "causal.json"
    fp32_summary = causal_statistics(json.loads(fp32_path.read_text())) if fp32_path.exists() else {}
    if fp32_summary:
        (root / "causal_fp32" / "summary.json").write_text(json.dumps(fp32_summary, indent=2) + "\n")

    lines = ["# Channel selection mechanism pilot", "", "All results are exploratory validation results, not sealed-test estimates.", "",
             "## Design", "", "Same pinned Qwen3-1.7B checkpoint and unmasked parent for every method; these are one-shot masks, not full iterative sweeps. "
             "Existing target-balanced CF9k gradients provide recall/extraction/reasoning scores. "
             f"Broader protection starts from {config['calibration_per_family']} synthetic calibration examples for each of five families, retaining only baseline-generated solutions with solver-verified final answers. "
             "The protected gradient objective is mean token CE over the complete generated solution; intermediate steps are not independently verified. "
             f"Evaluation uses {config['groups']} hash-selected MQuAKE groups (original prompts only) and {config['validation_per_family']} independently seeded synthetic examples per family. "
             "Numerical tasks use disjoint first operands across splits; other tasks use split-specific entity labels. "
             "These are in-template generalization tests with greedy generation: 48 tokens for MQuAKE and 384 for synthetic working plus a marked final answer. "
             "The model's thinking flag remains disabled; explicit working is requested in the prompt. Synthetic accuracy grades only the final-answer line. "
             "These remain small procedural problems, not a broad benchmark of long-chain reasoning.", "",
             "Task formatting was finalized using baseline-only prechecks. An empty-heading parsing bug was corrected uniformly across all saved raw evaluation responses; "
             "the accepted calibration IDs used to compute family scores remain frozen. The final grading implementation never accepts an answer merely because it appears inside the working.", "",
             "Protected-only uses recall weight 0; contrastive uses 0.05. Broader uses 0.05 and the maximum sensitivity percentile over all seven protected families. "
             "Every method retains the 25th-percentile eligibility thresholds and 15% per-layer cap. "
             "A failed selection is reported, not replaced with a smaller mask.", "",
             "Bias-compensated arms use the identical deletion masks plus constant W_down[:, deleted] @ mean_activation[deleted]. "
             f"Means use protected calibration only: {5*config['calibration_per_family']} synthetic examples and {config['calibration_per_family']} examples each for MQuAKE extraction and reasoning. "
             "No surviving weights change and there is no optimization or recovery training. The means average tokens within each teacher-forced example, then examples equally. "
             "These arms are not pure deletion and their added biases must be preserved in any compact model.", "",
             "Pilot acceptance thresholds, written before evaluation: at most 2 percentage points of exact-accuracy loss, 0.10 nats of mean answer-margin loss, "
             "and 0.10 nats/token of answer-CE increase in **every** protected family. "
             "These are diagnostic point-estimate thresholds, not evidence of statistical noninferiority; this small sample cannot establish tight retention budgets. "
             "Margin/answer-CE diagnostics teacher-force the short correct answer directly, without supplying a solved trace. On synthetic prompts requesting working, "
             "these are auxiliary likelihood probes, not confidence estimates of the final generated answer.", "",
             "## Results", "",
             "| Method | Recall containment | MQuAKE reasoning exact | Worst protected exact drop (pp) | Worst protected margin drop (nats) | Worst protected CE increase | All family budgets pass? |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for mode, metric in results.items():
        protected = [v for k, v in metric.items() if k != "mquake_recall"]
        lines.append(f"| {mode} | {metric['mquake_recall']['containment']*100:.2f}% | {metric['mquake_reasoning']['exact']*100:.2f}% | "
                     f"{max(x['exact_drop'] for x in protected)*100:.2f} | {max(x['margin_drop'] for x in protected):.4f} | "
                     f"{max(x['ce_increase'] for x in protected):.4f} | {'yes' if all(x['passes_pilot_budget'] for x in protected) else 'no'} |")
    lines += ["", "Successful calibration solutions by family: " + ", ".join(
        f"{family} {count}/{config['calibration_per_family']}" for family, count in calibration_filter["counts"].items()) + ".",
        "Final-answer grading strips markdown and optional literal answer tags. For route planning, an explicit correct first edge or complete optimal route is accepted as an alias for the requested next node. "
        "Only the final-answer line is scored; a correct intermediate value cannot rescue an incorrect or missing final answer.",
        "", "## Per-family exact accuracy", "", "| Method | " + " | ".join(families) + " |",
              "|---|" + "---:|" * len(families)]
    for mode, metric in results.items():
        lines.append("| " + mode + " | " + " | ".join(f"{metric[f]['exact']*100:.2f}%" for f in families) + " |")
    lines += ["", "## Matched comparisons", "", "Paired percentile-bootstrap 95% intervals, 2,000 resamples per family. "
              "Positive deltas mean more accuracy/containment or higher margin. Intervals are exploratory and unadjusted for multiple comparisons. "
              "Percentile bootstrap intervals can under-cover sparse binary changes; the JSON also reports exact paired two-sided binomial p-values for accuracy/containment differences.", "",
              "| Comparison | Recall containment delta (pp), 95% CI | Recall margin delta (nats), 95% CI |",
              "|---|---:|---:|"]
    for name, comparison in comparisons.items():
        if name.endswith("_minus_baseline"):
            continue
        c = comparison["mquake_recall"]["containment_delta"]
        m = comparison["mquake_recall"]["margin_delta"]
        lines.append(f"| {name} | {c['mean']*100:+.2f} [{c['ci95'][0]*100:+.2f}, {c['ci95'][1]*100:+.2f}] | "
                     f"{m['mean']:+.4f} [{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}] |")
    failures = json.loads((root / "selection_failures.json").read_text())
    lines += ["", "## Selection limits", "", "These limits apply to a one-shot candidate from the saved scores; they are not ceilings on an iterative, rescored pruning run.", ""] + [f"- {name}: {reason}" for name, reason in failures.items()]
    lines += ["", "## Causal fidelity", "", "Four recall-selective, four protected-important, and four random channels, plus each four-channel group. "
              "Both 25% attenuation and full deletion are compared with local gate gradients on the same 16 held-out groups. "
              "These diagnostic gradients never select production masks. BF16 can obscure very small intervention effects.", ""]
    if causal_summary:
        lines += ["| Condition | Attenuation | Pearson | Spearman | Mean absolute CE prediction error |", "|---|---:|---:|---:|---:|"]
        for condition, summary in causal_summary.items():
            for strength in ("0.25", "1.0"):
                s = summary[strength]
                lines.append(f"| {condition} | {strength} | {s['pearson']:.3f} | {s['spearman']:.3f} | {s['mean_abs_prediction_error']:.6f} |")
    else:
        lines.append("Causal run still pending.")
    if fp32_summary:
        lines += ["", "FP32 precision control, with TF32 disabled, on identical channels and examples:", "",
                  "| Condition | Attenuation | Pearson | Spearman | Mean absolute CE prediction error |", "|---|---:|---:|---:|---:|"]
        for condition, summary in fp32_summary.items():
            for strength in ("0.25", "1.0"):
                s = summary[strength]
                lines.append(f"| {condition} | {strength} | {s['pearson']:.3f} | {s['spearman']:.3f} | {s['mean_abs_prediction_error']:.8f} |")
    reference_path = root / "reference_trace_metrics.json"
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        trace_families = sorted(reference["baseline"])
        lines += ["", "## Complete-solution likelihood diagnostic", "",
                  "Post-hoc check on fixed baseline-generated solutions with correct final answers. This uses the same type of complete-solution CE objective as broader calibration, "
                  "rather than the short-answer probes above. These measurements did not select masks or change the predeclared acceptance rules. "
                  "Positive values are mean CE increases in nats/token; teacher-forced trace likelihood does not establish free-running reasoning retention.", "",
                  "| Method | " + " | ".join(trace_families) + " |", "|---|" + "---:|" * len(trace_families)]
        for mode in results:
            if mode in reference:
                lines.append("| " + mode + " | " + " | ".join(f"{reference[mode][f]['ce_increase']:+.5f}" for f in trace_families) + " |")
        lines += ["", "Reference counts: " + ", ".join(f"{f}: {reference['baseline'][f]['count']}" for f in trace_families) + "."]
        left_path = root / "reference_traces" / "broader_4pct.json"
        right_path = root / "reference_traces" / "contrastive_4pct.json"
        if left_path.exists() and right_path.exists():
            left, right = json.loads(left_path.read_text()), json.loads(right_path.read_text())
            comparison = {family: bootstrap_mean([
                left[i]["ce"] - right[i]["ce"] for i in left if left[i]["family"] == family
            ]) for family in trace_families}
            (root / "reference_trace_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
            lines += ["", "Broader minus current selection at 4%: paired bootstrap 95% intervals for complete-solution CE. Negative is better preservation.", "",
                      "| Family | Mean difference | 95% interval |", "|---|---:|---:|"]
            for family, value in comparison.items():
                lines.append(f"| {family} | {value['mean']:+.5f} | [{value['ci95'][0]:+.5f}, {value['ci95'][1]:+.5f}] |")
    lines += ["", "## Artifacts and reproducibility", "", "Run:", "", "```bash",
              "uv run python -m scripts.test_selection_mechanism --offline",
              "uv run python -m scripts.evaluate_reference_traces",
              "uv run python -m scripts.check_causal_precision",
              "uv run python -m scripts.summarize_selection_mechanism", "```", "",
              "`configuration.json` records predeclared settings. `metrics.json` contains family accuracy, CE/margin changes, and paired wins/losses. "
              "`paired_comparisons.json` includes all family confidence intervals. `causal.json` and `causal_summary.json` contain finite-ablation measurements. "
              "`masks/`, `family_scores.pt`, `activation_means.pt`, and `evaluations/` retain the interventions, calibration estimates, and raw predictions.", ""]
    (root / "report.md").write_text("\n".join(lines))
    source_files = [Path("scripts/test_selection_mechanism.py"), Path("scripts/select_iterative_batch.py"),
                    Path("scripts/evaluate_robustness.py"), Path("src/model_compression/compensation.py"),
                    Path("scripts/evaluate_reference_traces.py"), Path("scripts/check_causal_precision.py"),
                    Path("scripts/score_channels_v2.py"), Path("src/model_compression/channels.py"),
                    Path(config["scores"]),
                    root / "calibration.jsonl", root / "validation.jsonl", root / "configuration.json"]
    (root / "sha256.json").write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}, indent=2) + "\n")
    print(root / "report.md")


if __name__ == "__main__":
    main()
