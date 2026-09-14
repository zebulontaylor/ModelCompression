#!/usr/bin/env python3
"""Compute frozen-weight, answer-token MLP gate-gradient statistics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from model_compression.channels import (
    CONDITIONS,
    ChannelGates,
    ChannelStatistics,
    answer_token_batch,
    calibration_half,
    read_jsonl,
)
from model_compression.qwen import MODEL_ID, REVISION, load_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/calibration/mquake_remastered_cf3k_baseline_known/calibration.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/channel_scores/mquake_qwen3_1.7b"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--limit", type=int, help="score at most this many matched groups")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    return args


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    groups = sorted({row["group_id"] for row in records})
    if args.limit is not None:
        allowed = set(groups[: args.limit])
        records = [row for row in records if row["group_id"] in allowed]
    unexpected = sorted({row["condition"] for row in records} - set(CONDITIONS))
    if unexpected:
        raise ValueError(f"unknown conditions: {unexpected}")

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "checkpoint.pt"
    checkpoint = None
    processed: set[str] = set()
    if checkpoint_path.exists() and not args.no_resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["input"] != str(args.input) or checkpoint["limit"] != args.limit:
            raise ValueError("checkpoint input/limit differs; use another output or --no-resume")
        processed = set(checkpoint["processed_ids"])
        print(f"resuming after {len(processed)} examples", flush=True)

    model, tokenizer = load_baseline(args.device, local_files_only=args.offline)
    with ChannelGates(model) as gates:
        statistics = ChannelStatistics(
            (gates.layer_count, gates.width), model.device,
            checkpoint["accumulator"] if checkpoint else None,
        )
        pending = [row for row in records if row["id"] not in processed]
        for index, record in enumerate(pending, start=1):
            gates.zero_grad()
            batch = answer_token_batch(tokenizer, record["prompt"], record["target"], model.device)
            output = model(**batch, use_cache=False)
            output.loss.backward()
            if gates.values.grad is None or not torch.isfinite(gates.values.grad).all():
                raise RuntimeError(f"invalid gate gradient for {record['id']}")
            statistics.add(
                record["condition"], calibration_half(record["group_id"]), gates.values.grad
            )
            processed.add(record["id"])
            if index % args.checkpoint_every == 0 or index == len(pending):
                state = {
                    "model": MODEL_ID,
                    "revision": REVISION,
                    "input": str(args.input),
                    "limit": args.limit,
                    "processed_ids": sorted(processed),
                    "accumulator": statistics.state_dict(),
                }
                atomic_torch_save(state, checkpoint_path)
                print(f"scored {len(processed)}/{len(records)}", flush=True)

        if len(processed) != len(records):
            raise RuntimeError("checkpoint contains records outside the requested input")
        results = statistics.results()
        results.update({"model": MODEL_ID, "revision": REVISION, "input": str(args.input)})
        atomic_torch_save(results, args.output / "scores.pt")
        metadata = {
            "model": MODEL_ID,
            "revision": REVISION,
            "input": str(args.input),
            "examples": len(records),
            "groups": len({row["group_id"] for row in records}),
            "counts": results["counts"],
            "shape": list(results["shape"]),
            "objective": "mean cross-entropy on target-overlapping answer tokens",
            "recall_importance": "-mean(dL/dm)",
            "protected_sensitivity": "mean(abs(dL/dm)) per example",
            "half_assignment": "sha256(group_id) low bit",
        }
        (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
