#!/usr/bin/env python3
"""Reproducible pilot of recall utility, broader protection, and causal fidelity.

Run with python -m scripts.test_selection_mechanism --offline. No test split or
weight updates are used. All decisions are recorded before model evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch

from model_compression.channels import ChannelGates, read_jsonl, write_jsonl, is_correct, contains_accepted_answer
from model_compression.channels import answer_token_batch
from model_compression.compensation import MeanBiasCompensation
from model_compression.qwen import MODEL_ID, REVISION, load_baseline
from scripts.evaluate_robustness import (
    generate, margins, prepare_generation, prepare_margins, scored_answer,
)
from scripts.score_channels_v2 import loss_gradient
from scripts.select_iterative_batch import select_channels


FAMILIES = ("arithmetic", "program", "rules", "temporal", "planning")


def synthetic_records(split: str, count: int) -> list[dict]:
    """Procedural tasks with computed targets and independently seeded splits.

    These are in-template held-out instances, not a general reasoning benchmark.
    """
    rng = random.Random(1907 if split == "calibration" else 7319)
    rows = []
    for family in FAMILIES:
        for i in range(count):
            names = ("Mira Luno Tavi Neri Sora Kelu Pavo" if split == "calibration"
                     else "Rina Daro Vela Kori Mavi Zeno Fira").split()
            rng.shuffle(names)
            # Disjoint first operands prevent accidental cross-split duplicates.
            a = rng.choice(list(range(4, 14, 2) if split == "calibration" else range(3, 14, 2)))
            b, c = rng.randint(2, 5), rng.randint(2, 9)
            aliases = []
            if family == "arithmetic":
                target = str((a + b) * c)
                prompt = f"Compute ({a} + {b}) * {c}."
                alternatives = [str(a + b * c), str(a * c), str((a + b) * c + 1)]
            elif family == "program":
                value = (a + b) * c
                target = str(value)
                prompt = f"What number does this Python code print?\nx = {a}\nx = x + {b}\nx = x * {c}\nprint(x)"
                alternatives = [str(a), str(a + b), str(value + 1)]
            elif family == "rules":
                # Only the last reachable property is queried; disconnected rules are distractors.
                chain = rng.sample(["striped", "quiet", "round", "warm", "green", "soft"], 4)
                rules = [f"If something is {x}, then it is {y}." for x, y in zip(chain, chain[1:])]
                rules.append("If something is purple, then it is metallic.")
                rng.shuffle(rules)
                target = chain[-1]
                alternatives = ["metallic", "purple", "unknown"]
                options = [target, *alternatives]
                rng.shuffle(options)
                prompt = f"Use only these facts and rules. {names[0]} is {chain[0]}. " + " ".join(rules)
                prompt += f" Which one of {', '.join(options)} must describe {names[0]}?"
            elif family == "temporal":
                order = names[:3]
                statements = [f"{x} happened before {y}." for x, y in zip(order, order[1:])]
                rng.shuffle(statements)
                target = order[1]
                alternatives = [order[0], order[2], names[3]]
                prompt = " ".join(statements) + " Which event happened second?"
            else:
                # Two disjoint routes: costs make the optimal first step unique.
                start, left, right, end = names[:4]
                costs = rng.sample(range(1, 15), 4)
                while costs[0] + costs[1] == costs[2] + costs[3]:
                    costs = rng.sample(range(1, 15), 4)
                edges = [(start, left, costs[0]), (left, end, costs[1]),
                         (start, right, costs[2]), (right, end, costs[3])]
                rng.shuffle(edges)
                prompt = "A robot can use only these directed moves: " + "; ".join(
                    f"{x} to {y} costs {cost}" for x, y, cost in edges)
                prompt += f". Starting at {start}, which node should it move to first to reach {end} at minimum total cost?"
                target = left if costs[0] + costs[1] < costs[2] + costs[3] else right
                alternatives = [x for x in (start, left, right, end) if x != target]
                aliases = [f"{start} {arrow} {target}{suffix}" for arrow in ("→", "->")
                           for suffix in ("", f" {arrow} {end}")]
            prompt += "\nWork through the problem briefly. End with exactly 'Final answer: <answer>' on its own line. The final answer must contain only the requested number, property, or node name."
            row_id = f"synthetic:{split}:{family}:{i:04d}"
            rows.append(dict(id=row_id, source_id=row_id, group_id=row_id, split=split,
                             condition="reasoning", relation=family, family=family,
                             variant="original", prompt=prompt, target=target, aliases=aliases,
                             answer_format="final_answer",
                             alternatives=list(dict.fromkeys(x for x in alternatives if x != target))))
    return rows


def sample_mquake(path: Path, count: int) -> list[dict]:
    records = read_jsonl(path)
    if any(row.get("split") != "validation" for row in records):
        raise ValueError("pilot requires validation data only")
    groups = sorted({x["group_id"] for x in records}, key=lambda x: hashlib.sha256(x.encode()).digest())[:count]
    chosen = set(groups)
    return [{**row, "source_id": row["id"], "variant": "original",
             "family": "mquake_" + row["condition"]}
            for row in records if row["group_id"] in chosen]


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def calibration_means(model, tokenizer, cal, mquake_path, output):
    path = output / "activation_means.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    records = read_jsonl(mquake_path)
    if any(r.get("split") != "calibration" for r in records):
        raise ValueError("activation means require calibration data")
    sample = []
    count = len(cal) // len(FAMILIES)
    for condition in ("extraction", "reasoning"):
        pool = sorted([r for r in records if r["condition"] == condition],
                      key=lambda r: hashlib.sha256(r["id"].encode()).digest())
        sample.extend(pool[:count])
    rows = cal + sample
    totals = [torch.zeros(layer.mlp.down_proj.weight.shape[1], device=model.device) for layer in model.model.layers]
    handles = []
    for index, layer in enumerate(model.model.layers):
        def collect(module, inputs, i=index):
            totals[i].add_(inputs[0].detach().float().mean(dim=(0, 1)), alpha=1 / len(rows))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(collect))
    try:
        with torch.inference_mode():
            for row in rows:
                batch = answer_token_batch(tokenizer, row["prompt"], row["target"], model.device, row.get("completion_prefix", ""))
                batch.pop("labels")
                model.model(**batch, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    means = torch.stack(totals).cpu()
    torch.save(means, path)
    save(output / "activation_means_manifest.json", {
        "model": MODEL_ID, "revision": REVISION, "masked_channels": [],
        "records": [r["id"] for r in rows], "aggregation": "mean over tokens per example, then equal example mean",
        "data": str(mquake_path), "synthetic_data": str(output / "calibration.jsonl"),
    })
    return means


def evaluate(model, tokenizer, gates, channels, rows, path, batch_size):
    if path.exists():
        return json.loads(path.read_text())
    gates.mask(channels)
    predictions = []
    for synthetic, token_budget in ((False, 48), (True, 384)):
        subset = [r for r in rows if r["id"].startswith("synthetic:") == synthetic]
        predictions.extend(generate(model, tokenizer, prepare_generation(tokenizer, subset), batch_size, token_budget, path.stem))
    margin_rows = margins(model, tokenizer, rows, prepare_margins(tokenizer, rows), batch_size)
    result = {"predictions": predictions, "margins": margin_rows}
    save(path, result)
    return result


def checked_calibration(model, tokenizer, rows, output, batch_size):
    """Use baseline-correct generated derivations as the protected CE target."""
    path = output / "calibration_predictions.json"
    if path.exists():
        predictions = json.loads(path.read_text())
    else:
        predictions = generate(model, tokenizer, prepare_generation(tokenizer, rows), batch_size, 384, "checked calibration")
        save(path, predictions)
    filter_path = output / "calibration_filter.json"
    if (output / "family_scores.pt").exists() and filter_path.exists():
        # Once scores exist, the accepted calibration IDs are frozen. A grading
        # fix must not silently change their provenance during an evaluation resume.
        frozen = json.loads(filter_path.read_text())
        ids = set(frozen["accepted_ids"])
        by_id = {r["id"]: r for r in predictions}
        accepted = [{**r, "answer_target": r["target"], "target": by_id[r["id"]]["prediction"]}
                    for r in rows if r["id"] in ids]
        if len(accepted) != len(ids):
            raise ValueError("frozen calibration IDs differ from the supplied data")
        return accepted, frozen["counts"]
    source = {r["id"]: r for r in rows}
    for prediction in predictions:
        row = source[prediction["id"]]
        prediction["scored_prediction"] = scored_answer(prediction["prediction"], row)
        prediction["exact_correct"] = is_correct(prediction["scored_prediction"], row["target"], row["aliases"])
        prediction["contains_answer"] = contains_accepted_answer(prediction["scored_prediction"], row["target"], row["aliases"])
        prediction["aliases"] = row["aliases"]
    save(path, predictions)
    by_id = {r["id"]: r for r in predictions}
    accepted = [{**r, "answer_target": r["target"], "target": by_id[r["id"]]["prediction"]}
                for r in rows if by_id[r["id"]]["exact_correct"]]
    counts = {f: sum(r["family"] == f for r in accepted) for f in FAMILIES}
    save(output / "calibration_filter.json", {
        "counts": counts, "accepted_ids": [r["id"] for r in accepted],
        "objective": "mean token CE of complete baseline-generated derivation with verified final answer",
        "verification": "final answers checked against procedural solver; intermediate reasoning not independently verified",
    })
    if min(counts.values()) < 2:
        raise ValueError(f"insufficient demonstrated baseline ability for a protected family: {counts}")
    return accepted, counts


def metrics(result, baseline, rows):
    pred = {x["id"]: x for x in result["predictions"]}
    bp = {x["id"]: x for x in baseline["predictions"]}
    mar = {x["id"]: x for x in result["margins"]}
    bm = {x["id"]: x for x in baseline["margins"]}
    summary = {}
    for family in sorted({x["family"] for x in rows}):
        ids = [x["id"] for x in rows if x["family"] == family]
        n = len(ids)
        exact_drop = sum(bp[i]["exact_correct"] - pred[i]["exact_correct"] for i in ids) / n
        margin_drop = sum(bm[i]["margin"] - mar[i]["margin"] for i in ids) / n
        ce_increase = sum(bm[i]["correct_logp"] - mar[i]["correct_logp"] for i in ids) / n
        summary[family] = {
            "count": n,
            "baseline_exact": sum(bp[i]["exact_correct"] for i in ids) / n,
            "exact": sum(pred[i]["exact_correct"] for i in ids) / n,
            "baseline_containment": sum(bp[i]["contains_answer"] for i in ids) / n,
            "containment": sum(pred[i]["contains_answer"] for i in ids) / n,
            "exact_drop": exact_drop, "margin_drop": margin_drop, "ce_increase": ce_increase,
            "correct_to_wrong": sum(bp[i]["exact_correct"] and not pred[i]["exact_correct"] for i in ids),
            "wrong_to_correct": sum(not bp[i]["exact_correct"] and pred[i]["exact_correct"] for i in ids),
            "passes_pilot_budget": exact_drop <= .02 + 1e-12 and margin_drop <= .10 + 1e-12 and ce_increase <= .10 + 1e-12,
        }
    return summary


def causal_check(model, tokenizer, gates, scores, validation, output, batch_size):
    """Compare local gradients and finite interventions on the SAME fixed examples.

    The examples come from validation and are diagnostic only, never selection.
    One gradient per example scores the entire candidate shortlist.
    """
    path = output / "causal.json"
    if path.exists():
        return json.loads(path.read_text())
    shape = scores["recall_importance_ce"].shape
    ce, margin, extraction, reasoning = [scores[k] for k in (
        "recall_importance_ce", "recall_importance_margin", "extraction_sensitivity", "reasoning_sensitivity")]
    eligible = (extraction <= torch.quantile(extraction, .25)) & (reasoning <= torch.quantile(reasoning, .25))
    selected = torch.topk(ce.masked_fill(~eligible, -torch.inf).flatten(), 4).indices.tolist()
    protected = torch.topk(reasoning.flatten(), 4).indices.tolist()
    rng = random.Random(982)
    random_ids = rng.sample([x for x in range(ce.numel()) if x not in selected + protected], 4)
    cohorts = {"recall_selective": selected, "protected_important": protected, "random": random_ids}
    groups = sorted({r["group_id"] for r in validation}, key=lambda x: hashlib.sha256(("causal" + x).encode()).digest())[:16]
    rows = [r for r in validation if r["group_id"] in groups]
    grad_means = {}
    with ChannelGates(model) as differentiable:
        for condition in ("recall", "extraction", "reasoning"):
            subset = [r for r in rows if r["condition"] == condition]
            signed = torch.zeros(shape)
            absolute = torch.zeros(shape)
            for row in subset:
                g = loss_gradient(model, tokenizer, differentiable, row["prompt"], row["target"]).cpu()
                signed += g / len(subset)
                absolute += g.abs() / len(subset)
            grad_means[condition] = (signed, absolute)
    items = prepare_margins(tokenizer, rows)
    gates.reset()
    base = margins(model, tokenizer, rows, items, batch_size)
    base = {r["id"]: r for r in base}
    interventions = []
    for cohort, flat_ids in cohorts.items():
        sets = [("single", [i]) for i in flat_ids] + [("group", flat_ids)]
        for kind, ids in sets:
            channels = [divmod(i, shape[1]) for i in ids]
            loc = tuple(zip(*channels))
            for strength in (.25, 1.0):
                gates.reset()
                with torch.no_grad():
                    gates.values[loc] = 1 - strength
                measured = margins(model, tokenizer, rows, items, batch_size)
                effects = {}
                for condition, (signed, absolute) in grad_means.items():
                    subset = [r for r in measured if r["condition"] == condition]
                    effects[condition] = {
                        "predicted_ce_increase": float(-strength * signed[loc].sum()),
                        "first_order_abs_bound": float(strength * absolute[loc].sum()),
                        "actual_ce_increase": sum(base[r["id"]]["correct_logp"] - r["correct_logp"] for r in subset) / len(subset),
                        "actual_margin_drop": sum(base[r["id"]]["margin"] - r["margin"] for r in subset) / len(subset),
                    }
                interventions.append(dict(cohort=cohort, kind=kind, channels=channels, strength=strength, effects=effects))
                print(f"causal {cohort} {kind} {strength}: {effects['recall']}", flush=True)
    result = {"groups": groups, "examples_per_condition": 16,
              "selection_source": "calibration scores", "gradient_source": "same validation examples as finite ablations",
              "interventions": interventions, "dtype": str(model.dtype),
              "precision_note": "BF16 small effects may be rounding noise; compare the FP32 control where available"}
    save(path, result)
    gates.reset()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/selection_mechanism_pilot"))
    parser.add_argument("--scores", type=Path, default=Path("artifacts/channel_scores/mquake_cf9k_qwen3_1.7b_v2/scores.pt"))
    parser.add_argument("--validation", type=Path, default=Path("data/calibration/mquake_remastered_cf9k_v2/validation.jsonl"))
    parser.add_argument("--mquake-calibration", type=Path, default=Path("data/calibration/mquake_remastered_cf9k_v2/calibration.jsonl"))
    parser.add_argument("--groups", type=int, default=256)
    parser.add_argument("--calibration-per-family", type=int, default=16)
    parser.add_argument("--validation-per-family", type=int, default=32)
    parser.add_argument("--fractions", nargs="+", type=float, default=[.01, .04, .05, .10])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(model=MODEL_ID, revision=REVISION, seed=1907, generator_schema=5,
                  protected_quantile=.25, max_layer_fraction=.15,
                  pilot_budgets={"per_family_exact_drop": .02, "per_family_margin_drop": .10, "per_family_ce_increase": .10},
                  test_split_used=False, parent_mask=[], generation_tokens={"mquake": 48, "synthetic": 384},
                  scope="in-template procedural generalization; explicit working, checked final answer; full successful-derivation CE for broad protection")
    config["bias_compensation"] = "all feasible 4% masks; means from protected calibration only"
    if (args.output / "configuration.json").exists():
        if json.loads((args.output / "configuration.json").read_text()) != config:
            raise ValueError("configuration differs; use a fresh output directory")
    save(args.output / "configuration.json", config)
    scores = torch.load(args.scores, map_location="cpu", weights_only=False)
    if scores.get("masked_channels") or (scores["model"], scores["revision"]) != (MODEL_ID, REVISION):
        raise ValueError("pilot requires unmasked scores for the pinned model")
    cal = synthetic_records("calibration", args.calibration_per_family)
    validation = sample_mquake(args.validation, args.groups)
    rows = validation + synthetic_records("validation", args.validation_per_family)
    write_jsonl(args.output / "calibration.jsonl", cal)
    write_jsonl(args.output / "validation.jsonl", rows)
    model, tokenizer = load_baseline(local_files_only=args.offline)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    means = calibration_means(model, tokenizer, cal, args.mquake_calibration, args.output)
    checked, checked_counts = checked_calibration(model, tokenizer, cal, args.output, args.batch_size)
    family_path = args.output / "family_scores.pt"
    if family_path.exists():
        family_scores = torch.load(family_path, map_location="cpu", weights_only=False)
    else:
        shape = scores["recall_importance_ce"].shape
        sensitivities = {f: torch.zeros(shape) for f in FAMILIES}
        with ChannelGates(model) as gates:
            for i, row in enumerate(checked):
                gradient = loss_gradient(model, tokenizer, gates, row["prompt"], row["target"], row.get("completion_prefix", ""))
                sensitivities[row["family"]] += gradient.cpu().abs() / checked_counts[row["family"]]
                if (i + 1) % 16 == 0:
                    print(f"broader calibration {i+1}/{len(checked)}", flush=True)
        family_scores = dict(model=MODEL_ID, revision=REVISION, masked_channels=[],
                             family_sensitivities=sensitivities, calibration=config)
        torch.save(family_scores, family_path)
    tensors = [scores[k].float() for k in ("recall_importance_ce", "recall_importance_margin", "extraction_sensitivity", "reasoning_sensitivity")]
    surviving = torch.ones_like(tensors[0], dtype=torch.bool)
    masks, failures = {}, {}
    for fraction in args.fractions:
        count = math.ceil(fraction * tensors[0].numel())
        for method, recall_weight, families in (
            ("protected_only", 0., None), ("contrastive", .05, None),
            ("broader", .05, family_scores["family_sensitivities"]),
        ):
            name = f"{method}_{fraction*100:g}pct"
            try:
                channels, _ = select_channels(*tensors, surviving, [], count, .25, recall_weight, .15, families)
            except ValueError as exc:
                failures[name] = str(exc)
                continue
            masks[name] = channels
            save(args.output / "masks" / f"{name}.json", dict(
                model=MODEL_ID, revision=REVISION, method=method, count=count,
                channels=[dict(layer=l, channel=c) for l, c in channels],
                source_scores=str(args.scores), parent_mask=[],
                family_scores=str(family_path) if families is not None else None))
    save(args.output / "selection_failures.json", failures)
    with ChannelGates(model, requires_grad=False) as gates:
        baseline = evaluate(model, tokenizer, gates, [], rows, args.output / "evaluations" / "baseline.json", args.batch_size)
        results = {"baseline": metrics(baseline, baseline, rows)}
        for name, channels in masks.items():
            result = evaluate(model, tokenizer, gates, channels, rows, args.output / "evaluations" / f"{name}.json", args.batch_size)
            results[name] = metrics(result, baseline, rows)
            save(args.output / "metrics.json", results)
            print(json.dumps({name: results[name]}), flush=True)
        for name, channels in masks.items():
            if name.endswith("_4pct"):
                mode = name + "_compensated"
                with MeanBiasCompensation(model, channels, means):
                    result = evaluate(model, tokenizer, gates, channels, rows, args.output / "evaluations" / f"{mode}.json", args.batch_size)
                results[mode] = metrics(result, baseline, rows)
                save(args.output / "metrics.json", results)
                print(json.dumps({mode: results[mode]}), flush=True)
        gates.reset()
        causal_check(model, tokenizer, gates, scores, validation, args.output, args.batch_size)
    print(f"Pilot complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
