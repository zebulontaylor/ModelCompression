#!/usr/bin/env python3
"""Evaluate masks across prompt variants, targets, relations, and answer margins."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm

from model_compression.channels import (
    CONDITIONS, ChannelGates, contains_accepted_answer, is_correct, load_mask,
    normalize_answer, read_jsonl, write_jsonl,
)
from model_compression.qwen import MODEL_ID, REVISION, load_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=Path("data/calibration/mquake_remastered_cf3k_v2/validation.jsonl"),
    )
    parser.add_argument("--mask", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("artifacts/mask_evaluations/robust_validation"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, help="evaluate at most this many matched groups")
    parser.add_argument("--no-margins", action="store_true")
    parser.add_argument(
        "--baseline-cache", type=Path,
        help="reuse a fingerprinted unmasked baseline across evaluation runs",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("batch size and max new tokens must be positive")
    return args


def expand_variants(records: list[dict]) -> list[dict]:
    expanded = []
    for row in records:
        for variant in row.get("prompt_variants", [{"label": "original", "prompt": row["prompt"]}]):
            expanded.append({
                **{key: value for key, value in row.items() if key not in {"prompt", "prompt_variants"}},
                "source_id": row["id"],
                "id": f"{row['id']}::{variant['label']}",
                "variant": variant["label"],
                "prompt": variant["prompt"],
            })
    return expanded


def chat_prompts(tokenizer, rows: list[dict]) -> list[str]:
    return [tokenizer.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    ) + row.get("completion_prefix", "") for row in rows]


@dataclass(frozen=True)
class GenerationItem:
    index: int
    row: dict
    prompt: str
    token_length: int
    max_new_tokens: int


@dataclass(frozen=True)
class MarginItem:
    index: int
    owner: int
    prompt: str
    text: str
    token_length: int


def _token_lengths(tokenizer, texts: list[str]) -> list[int]:
    return tokenizer(texts, add_special_tokens=False, return_length=True)["length"]


def prepare_generation(
    tokenizer, rows: list[dict], max_new_tokens: int,
) -> list[GenerationItem]:
    """Families that must show their working carry their own token budget."""
    prompts = chat_prompts(tokenizer, rows)
    return [
        GenerationItem(index, row, prompt, length, int(row.get("max_new_tokens", max_new_tokens)))
        for index, (row, prompt, length) in enumerate(zip(
            rows, prompts, _token_lengths(tokenizer, prompts), strict=True,
        ))
    ]


def prepare_margins(tokenizer, rows: list[dict]) -> list[MarginItem]:
    prompts = chat_prompts(tokenizer, rows)
    raw_items = []
    for owner, (row, prompt) in enumerate(zip(rows, prompts, strict=True)):
        for answer in [row["target"], *row["alternatives"]]:
            raw_items.append((owner, prompt, prompt + answer))
    lengths = _token_lengths(tokenizer, [item[2] for item in raw_items])
    return [
        MarginItem(index, owner, prompt, text, length)
        for index, ((owner, prompt, text), length) in enumerate(zip(raw_items, lengths, strict=True))
    ]


def length_batches(items, batch_size):
    ordered = sorted(items, key=lambda item: item.token_length)
    return [ordered[start:start + batch_size] for start in range(0, len(ordered), batch_size)]


def baseline_fingerprint(rows: list[dict], max_new_tokens: int) -> str:
    """Identify every input that can affect deterministic baseline results."""
    payload = {
        "schema": 1,
        "model": MODEL_ID,
        "revision": REVISION,
        "max_new_tokens": max_new_tokens,
        "rows": rows,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def load_baseline_cache(path: Path, fingerprint: str) -> tuple[list[dict], list[dict]]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != 1 or payload.get("fingerprint") != fingerprint:
        raise ValueError(
            f"baseline cache does not match model, data, or generation settings: {path}"
        )
    return payload["predictions"], payload.get("margins", [])


def save_baseline_cache(
    path: Path, fingerprint: str, predictions: list[dict], margin_rows: list[dict],
) -> None:
    atomic_json({
        "schema": 1,
        "fingerprint": fingerprint,
        "predictions": predictions,
        "margins": margin_rows,
    }, path)


def scored_answer(prediction, row):
    if row.get("answer_format") == "first_line":
        return prediction.splitlines()[0] if prediction else ""
    if row.get("answer_format") == "final_answer":
        plain = prediction.replace("**", "").replace("`", "")
        matches = re.findall(r"(?im)^[ \t]*(?:#+[ \t]*)?Final answer[ \t]*:[ \t]*([^\n]+)", plain)
        answer = matches[-1].strip() if matches else ""
        return re.sub(r"</?answer>", "", answer, flags=re.IGNORECASE).strip()
    return prediction


def budget_batches(items, batch_size):
    """Never mix token budgets in one batch: generation runs until the longest."""
    budgets = sorted({item.max_new_tokens for item in items})
    return [
        (budget, batch)
        for budget in budgets
        for batch in length_batches([x for x in items if x.max_new_tokens == budget], batch_size)
    ]


def generate(model, tokenizer, items, batch_size, desc):
    predictions = [None] * len(items)
    progress = tqdm(total=len(items), desc=desc, unit="prompt")
    for max_new_tokens, batch in budget_batches(items, batch_size):
        inputs = tokenizer([item.prompt for item in batch], return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        texts = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for item, text in zip(batch, texts, strict=True):
            row = item.row
            prediction = text.strip()
            scored_prediction = scored_answer(prediction, row)
            predictions[item.index] = {
                "id": row["id"], "source_id": row["source_id"], "group_id": row["group_id"],
                "condition": row["condition"], "variant": row["variant"],
                "family": row.get("family", row["condition"]),
                "target": row["target"], "aliases": row["aliases"], "relation": row["relation"],
                "prediction": prediction, "normalized_prediction": normalize_answer(prediction),
                "scored_prediction": scored_prediction,
                "raw_exact_correct": is_correct(prediction, row["target"], row["aliases"]),
                "exact_correct": is_correct(scored_prediction, row["target"], row["aliases"]),
                "contains_answer": contains_accepted_answer(scored_prediction, row["target"], row["aliases"]),
            }
        progress.update(len(batch))
    progress.close()
    return predictions


def answer_token_logps(model, encoded, answer_mask):
    """Mean answer-token log probability, without projecting prompt positions."""
    shifted_mask = answer_mask[:, 1:] & encoded.attention_mask[:, 1:].bool()
    if not shifted_mask.any(dim=1).all():
        raise ValueError("an answer produced no scored tokens")
    with torch.inference_mode():
        hidden = model.model(**encoded, use_cache=False, return_dict=True).last_hidden_state[:, :-1]
        selected_logits = model.lm_head(hidden[shifted_mask]).float()
        selected_targets = encoded.input_ids[:, 1:][shifted_mask]
        selected_logps = selected_logits.log_softmax(-1).gather(
            -1, selected_targets.unsqueeze(-1)
        ).squeeze(-1)
    counts = shifted_mask.sum(1).tolist()
    return [chunk.mean().item() for chunk in selected_logps.split(counts)]


def batch_logps(model, tokenizer, items: list[MarginItem], batch_size: int) -> list[float]:
    values = [None] * len(items)
    for batch in length_batches(items, batch_size):
        prompts = [item.prompt for item in batch]
        encoded = tokenizer(
            [item.text for item in batch], return_tensors="pt", padding=True, add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")
        answer_mask = torch.stack([
            offsets[i, :, 1] > len(prompts[i]) for i in range(len(batch))
        ]).to(model.device)
        encoded = encoded.to(model.device)
        scores = answer_token_logps(model, encoded, answer_mask)
        for item, score in zip(batch, scores, strict=True):
            values[item.index] = score
    return values


def margins(model, tokenizer, rows, items, batch_size):
    logps = batch_logps(model, tokenizer, items, batch_size)
    grouped = defaultdict(list)
    for item, value in zip(items, logps, strict=True):
        grouped[item.owner].append(value)
    return [{
        "id": row["id"], "source_id": row["source_id"], "condition": row["condition"],
        "variant": row["variant"], "target": row["target"], "relation": row["relation"],
        "family": row.get("family", row["condition"]),
        "correct_logp": grouped[i][0],
        "alternative_mean_logp": sum(grouped[i][1:]) / len(grouped[i][1:]),
        "margin": grouped[i][0] - sum(grouped[i][1:]) / len(grouped[i][1:]),
    } for i, row in enumerate(rows)]


def aggregate(rows: list[dict]) -> dict:
    def one(selected):
        return {
            "count": len(selected),
            "exact_accuracy": sum(x["exact_correct"] for x in selected) / len(selected),
            "answer_containment": sum(x["contains_answer"] for x in selected) / len(selected),
        }
    report = {condition: one([x for x in rows if x["condition"] == condition]) for condition in CONDITIONS}
    report["by_variant"] = {
        variant: one([x for x in rows if x["variant"] == variant])
        for variant in sorted({x["variant"] for x in rows})
    }
    report["by_family"] = {
        family: {
            **one([x for x in rows if x.get("family", x["condition"]) == family]),
            "conditions": sorted({x["condition"] for x in rows if x.get("family", x["condition"]) == family}),
        }
        for family in sorted({x.get("family", x["condition"]) for x in rows})
    }
    for field in ("target", "relation"):
        units = sorted({x[field] for x in rows})
        per_unit = {unit: one([x for x in rows if x[field] == unit]) for unit in units}
        report[f"macro_exact_by_{field}"] = sum(x["exact_accuracy"] for x in per_unit.values()) / len(per_unit)
        report[f"macro_containment_by_{field}"] = sum(x["answer_containment"] for x in per_unit.values()) / len(per_unit)
        report[f"by_{field}"] = per_unit
    return report


def main() -> None:
    args = parse_args()
    base_records = read_jsonl(args.data)
    if any(row.get("split") == "test" for row in base_records):
        raise ValueError("robustness evaluation refuses the sealed test split")
    groups = sorted({row["group_id"] for row in base_records})
    if args.limit is not None:
        allowed = set(groups[:args.limit])
        base_records = [row for row in base_records if row["group_id"] in allowed]
    rows = expand_variants(base_records)
    model, tokenizer = load_baseline(args.device, local_files_only=args.offline)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    generation_items = prepare_generation(tokenizer, rows, args.max_new_tokens)
    margin_items = None if args.no_margins else prepare_margins(tokenizer, rows)

    stems = [path.stem for path in args.mask]
    modes = [("baseline", None), *[
        (path.stem if stems.count(path.stem) == 1 else f"{path.parent.name}_{path.stem}", path)
        for path in args.mask
    ]]
    if len({name for name, _ in modes}) != len(modes):
        raise ValueError("mask paths must have unique stems or parent-directory names")
    all_predictions, all_margins, summaries = [], [], {}
    fingerprint = baseline_fingerprint(rows, args.max_new_tokens)
    baseline_predictions: list[dict] = []
    baseline_margins: list[dict] = []
    if args.baseline_cache is not None and args.baseline_cache.exists():
        baseline_predictions, baseline_margins = load_baseline_cache(
            args.baseline_cache, fingerprint,
        )
        print(f"loaded baseline cache: {args.baseline_cache}", flush=True)
        if args.no_margins:
            baseline_margins = []

    cache_changed = False
    if not baseline_predictions:
        baseline_predictions = generate(
            model, tokenizer, generation_items, args.batch_size, desc="baseline",
        )
        cache_changed = True
    if not args.no_margins and not baseline_margins:
        baseline_margins = margins(model, tokenizer, rows, margin_items, args.batch_size)
        cache_changed = True
    row_by_id = {row["id"]: row for row in rows}
    for prediction in baseline_predictions:
        row = row_by_id[prediction["id"]]
        if row.get("answer_format"):
            value = scored_answer(prediction["prediction"], row)
            prediction["scored_prediction"] = value
            prediction["exact_correct"] = is_correct(value, row["target"], row["aliases"])
            prediction["contains_answer"] = contains_accepted_answer(value, row["target"], row["aliases"])
    if args.baseline_cache is not None and cache_changed:
        save_baseline_cache(
            args.baseline_cache, fingerprint, baseline_predictions, baseline_margins,
        )
        print(f"saved baseline cache: {args.baseline_cache}", flush=True)

    for item in baseline_predictions:
        item["mode"] = "baseline"
    for item in baseline_margins:
        item["mode"] = "baseline"
    all_predictions.extend(baseline_predictions)
    all_margins.extend(baseline_margins)
    summaries["baseline"] = aggregate(baseline_predictions)
    if baseline_margins:
        summaries["baseline"]["mean_margin"] = sum(
            x["margin"] for x in baseline_margins
        ) / len(baseline_margins)

    with ChannelGates(model, requires_grad=False) as gates:
        for mode, path in modes[1:]:
            payload = json.loads(path.read_text())
            if payload.get("model") != MODEL_ID or payload.get("revision") != REVISION:
                raise ValueError(f"mask model/revision mismatch: {path}")
            gates.mask(load_mask(path))
            predictions = generate(
                model, tokenizer, generation_items, args.batch_size, desc=mode,
            )
            for item in predictions:
                item["mode"] = mode
            all_predictions.extend(predictions)
            summaries[mode] = aggregate(predictions)
            if not args.no_margins:
                margin_rows = margins(model, tokenizer, rows, margin_items, args.batch_size)
                for item in margin_rows:
                    item["mode"] = mode
                all_margins.extend(margin_rows)
                summaries[mode]["mean_margin"] = sum(x["margin"] for x in margin_rows) / len(margin_rows)

    baseline_predictions = {x["id"]: x for x in baseline_predictions}
    baseline_margins = {x["id"]: x for x in baseline_margins}
    for mode, _ in modes[1:]:
        selected = [x for x in all_predictions if x["mode"] == mode]
        def paired(items):
            return {
                "exact_correct_to_wrong": sum(
                    baseline_predictions[x["id"]]["exact_correct"] and not x["exact_correct"]
                    for x in items
                ),
                "exact_wrong_to_correct": sum(
                    not baseline_predictions[x["id"]]["exact_correct"] and x["exact_correct"]
                    for x in items
                ),
                "content_correct_to_missing": sum(
                    baseline_predictions[x["id"]]["contains_answer"] and not x["contains_answer"]
                    for x in items
                ),
                "content_missing_to_correct": sum(
                    not baseline_predictions[x["id"]]["contains_answer"] and x["contains_answer"]
                    for x in items
                ),
            }

        summaries[mode]["paired"] = paired(selected)
        summaries[mode]["paired_by_condition"] = {
            condition: paired([x for x in selected if x["condition"] == condition])
            for condition in CONDITIONS
        }
        if baseline_margins:
            selected_margins = [x for x in all_margins if x["mode"] == mode]
            summaries[mode]["mean_margin_delta_vs_baseline"] = sum(
                x["margin"] - baseline_margins[x["id"]]["margin"] for x in selected_margins
            ) / len(selected_margins)
            summaries[mode]["mean_margin_delta_by_condition"] = {
                condition: sum(
                    x["margin"] - baseline_margins[x["id"]]["margin"]
                    for x in selected_margins if x["condition"] == condition
                ) / sum(x["condition"] == condition for x in selected_margins)
                for condition in CONDITIONS
            }
            for family, family_metrics in summaries[mode]["by_family"].items():
                subset = [x for x in selected_margins if x.get("family", x["condition"]) == family]
                family_metrics["margin_drop"] = sum(
                    baseline_margins[x["id"]]["margin"] - x["margin"] for x in subset
                ) / len(subset)
                family_metrics["ce_increase"] = sum(
                    baseline_margins[x["id"]]["correct_logp"] - x["correct_logp"] for x in subset
                ) / len(subset)

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "predictions.jsonl", all_predictions)
    if all_margins:
        write_jsonl(args.output / "margins.jsonl", all_margins)
    report = {
        "model": MODEL_ID, "revision": REVISION, "data": str(args.data),
        "groups": len({x["group_id"] for x in base_records}), "variant_rows_per_mode": len(rows),
        "generation": {
            "do_sample": False, "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
        },
        "test_split_used": False, "metrics": summaries,
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
