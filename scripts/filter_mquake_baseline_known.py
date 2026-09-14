#!/usr/bin/env python3
"""Retain MQuAKE groups whose recall item the frozen baseline answers correctly."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from model_compression.qwen import MODEL_ID, REVISION, load_baseline
from model_compression.channels import is_correct, normalize_answer


SPLITS = ("calibration", "validation", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/calibration/mquake_remastered_cf3k"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/calibration/mquake_remastered_cf3k_baseline_known"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("batch size and max new tokens must be positive")
    return args


def main() -> None:
    args = parse_args()
    split_records = {split: read_jsonl(args.input / f"{split}.jsonl") for split in SPLITS}
    recalls = [
        record
        for split in SPLITS
        for record in split_records[split]
        if record["condition"] == "recall"
    ]

    model, tokenizer = load_baseline(args.device, local_files_only=args.offline)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    predictions = []
    for start in range(0, len(recalls), args.batch_size):
        batch = recalls[start : start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": record["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for record in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[:, inputs.input_ids.shape[1] :]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for record, prediction in zip(batch, texts, strict=True):
            prediction = prediction.strip()
            predictions.append({
                "id": record["id"],
                "group_id": record["group_id"],
                "split": record["split"],
                "target": record["target"],
                "aliases": record["aliases"],
                "prediction": prediction,
                "normalized_prediction": normalize_answer(prediction),
                "correct": is_correct(prediction, record["target"], record["aliases"]),
            })
        print(f"scored {min(start + len(batch), len(recalls))}/{len(recalls)}", flush=True)

    accepted_groups = {row["group_id"] for row in predictions if row["correct"]}
    accepted_recall_ids = sorted(row["id"] for row in predictions if row["correct"])
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "recall_predictions.jsonl", predictions)
    (args.output / "accepted_recall_ids.json").write_text(
        json.dumps(accepted_recall_ids, indent=2) + "\n"
    )

    manifest = {
        "model": MODEL_ID,
        "revision": REVISION,
        "filter": "alias-aware normalized exact match on deterministic recall generation",
        "normalization": "Unicode NFKC, casefold, punctuation and English articles removed, whitespace collapsed",
        "generation": {
            "chat_template": True,
            "enable_thinking": False,
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
        },
        "accepted_recall_ids_sha256": "",
        "splits": {},
    }
    for split in SPLITS:
        source_path = args.input / f"{split}.jsonl"
        filtered = [row for row in split_records[split] if row["group_id"] in accepted_groups]
        destination = args.output / f"{split}.jsonl"
        write_jsonl(destination, filtered)
        split_predictions = [row for row in predictions if row["split"] == split]
        conditions = Counter(row["condition"] for row in filtered)
        manifest["splits"][split] = {
            "source_sha256": sha256(source_path),
            "candidate_groups": len(split_predictions),
            "accepted_groups": sum(row["correct"] for row in split_predictions),
            "examples": len(filtered),
            "conditions": dict(conditions),
            "output_sha256": sha256(destination),
        }
    manifest["accepted_recall_ids_sha256"] = sha256(args.output / "accepted_recall_ids.json")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
