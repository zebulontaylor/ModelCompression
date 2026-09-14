#!/usr/bin/env python3
"""Evaluate the unchanged baseline and temporary MLP channel masks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from model_compression.channels import (
    CONDITIONS,
    ChannelGates,
    is_correct,
    load_mask,
    normalize_answer,
    read_jsonl,
    write_jsonl,
)
from model_compression.qwen import MODEL_ID, REVISION, load_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=Path("data/calibration/mquake_remastered_cf3k_baseline_known"),
    )
    parser.add_argument("--split", choices=["calibration", "validation", "test"], default="validation")
    parser.add_argument("--allow-test", action="store_true", help="explicitly permit final-test evaluation")
    parser.add_argument("--mask", type=Path, action="append", default=[], help="mask JSON; may be repeated")
    parser.add_argument("--output", type=Path, default=Path("artifacts/mask_evaluations/initial_validation"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, help="evaluate at most this many matched groups")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("the test split is sealed; pass --allow-test for an intentional final evaluation")
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("batch size and max new tokens must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def prompts_for(tokenizer, records: list[dict]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in records
    ]


def generate(model, tokenizer, records: list[dict], batch_size: int, max_new_tokens: int):
    predictions = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        inputs = tokenizer(prompts_for(tokenizer, batch), return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[:, inputs.input_ids.shape[1] :]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for row, text in zip(batch, texts, strict=True):
            prediction = text.strip()
            predictions.append({
                "id": row["id"],
                "group_id": row["group_id"],
                "condition": row["condition"],
                "target": row["target"],
                "aliases": row["aliases"],
                "prediction": prediction,
                "normalized_prediction": normalize_answer(prediction),
                "correct": is_correct(prediction, row["target"], row["aliases"]),
            })
        print(f"generated {min(start + len(batch), len(records))}/{len(records)}", flush=True)
    return predictions


def metrics(rows: list[dict]) -> dict:
    totals = Counter(row["condition"] for row in rows)
    correct = Counter(row["condition"] for row in rows if row["correct"])
    result = {
        condition: {
            "correct": correct[condition],
            "total": totals[condition],
            "accuracy": correct[condition] / totals[condition] if totals[condition] else None,
        }
        for condition in CONDITIONS
    }
    result["macro_accuracy"] = sum(result[c]["accuracy"] for c in CONDITIONS) / len(CONDITIONS)
    return result


def paired_transitions(baseline: list[dict], masked: list[dict]) -> dict:
    baseline_by_id = {row["id"]: row for row in baseline}
    result = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in masked if row["condition"] == condition]
        result[condition] = {
            "correct_to_wrong": sum(
                baseline_by_id[row["id"]]["correct"] and not row["correct"]
                for row in condition_rows
            ),
            "wrong_to_correct": sum(
                not baseline_by_id[row["id"]]["correct"] and row["correct"]
                for row in condition_rows
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.data / f"{args.split}.jsonl")
    groups = sorted({row["group_id"] for row in records})
    if args.limit is not None:
        allowed = set(groups[: args.limit])
        records = [row for row in records if row["group_id"] in allowed]
    condition_counts = Counter(row["condition"] for row in records)
    if any(condition_counts[c] == 0 for c in CONDITIONS):
        raise ValueError("evaluation input must contain every matched condition")

    model, tokenizer = load_baseline(args.device, local_files_only=args.offline)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_predictions = []
    summaries = {}
    mask_details = {}
    baseline = generate(model, tokenizer, records, args.batch_size, args.max_new_tokens)
    for row in baseline:
        row["mode"] = "baseline"
    all_predictions.extend(baseline)
    summaries["baseline"] = metrics(baseline)

    if args.mask:
        with ChannelGates(model, requires_grad=False) as gates:
            for path in args.mask:
                payload = json.loads(path.read_text())
                if payload.get("model") not in (None, MODEL_ID) or payload.get("revision") not in (None, REVISION):
                    raise ValueError(f"mask model/revision does not match the pinned baseline: {path}")
                gates.mask(load_mask(path))
                label = path.stem
                if label in summaries:
                    raise ValueError(f"duplicate evaluation label: {label}")
                print(f"evaluating {label} ({len(payload['channels'])} masked channels)", flush=True)
                rows = generate(model, tokenizer, records, args.batch_size, args.max_new_tokens)
                for row in rows:
                    row["mode"] = label
                all_predictions.extend(rows)
                summaries[label] = metrics(rows)
                summaries[label]["paired_transitions"] = paired_transitions(baseline, rows)
                mask_details[label] = {
                    "path": str(path),
                    "method": payload.get("method"),
                    "channels": len(payload["channels"]),
                    "predicted_effects": payload.get("predicted_effects"),
                }

    baseline_metrics = summaries["baseline"]
    for label, summary in summaries.items():
        if label == "baseline":
            continue
        summary["accuracy_delta_vs_baseline"] = {
            condition: summary[condition]["accuracy"] - baseline_metrics[condition]["accuracy"]
            for condition in CONDITIONS
        }

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "predictions.jsonl", all_predictions)
    report = {
        "model": MODEL_ID,
        "revision": REVISION,
        "split": args.split,
        "groups": len({row["group_id"] for row in records}),
        "examples_per_mode": len(records),
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "masks": mask_details,
        "metrics": summaries,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
