#!/usr/bin/env python3
"""Small offline benchmark for robustness-evaluation and channel-scoring hot paths."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from model_compression.channels import ChannelGates, read_jsonl
from model_compression.qwen import load_baseline
from evaluate_robustness import (
    answer_token_logps,
    batch_logps,
    expand_variants,
    length_batches,
    prepare_generation,
    prepare_margins,
)
from score_channels_v2 import batched_loss_gradient, loss_gradient


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=Path("data/calibration/mquake_remastered_cf9k_v2/validation.jsonl"),
    )
    parser.add_argument(
        "--calibration-data", type=Path,
        default=Path("data/calibration/mquake_remastered_cf3k_v2/calibration.jsonl"),
    )
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.groups < 1 or args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("groups, batch size, and max new tokens must be positive")
    return args


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def timed(device, function):
    sync(device)
    start = time.perf_counter()
    result = function()
    sync(device)
    return result, time.perf_counter() - start


def padding_tokens(items, batch_size, bucketed):
    batches = length_batches(items, batch_size) if bucketed else [
        items[start:start + batch_size] for start in range(0, len(items), batch_size)
    ]
    return sum(max(item.token_length for item in batch) * len(batch) for batch in batches)


def legacy_logps(model, encoded, answer_mask):
    shifted_mask = answer_mask[:, 1:] & encoded.attention_mask[:, 1:].bool()
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False).logits[:, :-1].float()
        token_logps = logits.log_softmax(-1).gather(
            -1, encoded.input_ids[:, 1:].unsqueeze(-1),
        ).squeeze(-1)
    scores = (token_logps * shifted_mask).sum(1) / shifted_mask.sum(1)
    return scores.tolist()


def generate_batch(model, tokenizer, batch, max_new_tokens):
    inputs = tokenizer(
        [item.prompt for item in batch], return_tensors="pt", padding=True,
    ).to(model.device)
    with torch.inference_mode():
        model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )


def scoring_pass_counts(records):
    """Sequential distractor passes versus one batched pass per recall example."""
    recall = sum(row["condition"] == "recall" for row in records)
    alternatives = sum(
        len(row["alternatives"]) for row in records if row["condition"] == "recall"
    )
    return {
        "examples": len(records),
        "recall_examples": recall,
        "sequential_passes": len(records) + alternatives,
        "batched_passes": len(records) + recall,
    }


def sequential_alternative_gradient(model, tokenizer, gates, prompt, answers):
    """Pre-optimization path: one backward pass per distractor, then average."""
    total = None
    for answer in answers:
        gradient = loss_gradient(model, tokenizer, gates, prompt, answer)
        total = gradient if total is None else total.add_(gradient)
    return total.div_(len(answers))


def evaluation_round_cost(rows, generation_seconds, margin_seconds, sample_rows):
    """Per-candidate seconds before and after baseline caching plus --no-margins.

    Legacy evaluated the unmasked baseline and the candidate, each with
    generation and four-answer margins.  Screening now generates the candidate
    only, because the baseline is cached and margins are deferred to the final
    report.  Both measured totals scale linearly in the number of variant rows.
    """
    scale = rows / sample_rows
    generation = generation_seconds * scale
    margins = margin_seconds * scale
    legacy = 2 * (generation + margins)
    return {
        "variant_rows": rows,
        "legacy_seconds": legacy,
        "optimized_seconds": generation,
        "speedup": legacy / generation,
        "generation_seconds": generation,
        "margin_seconds": margins,
    }


def main():
    args = parse_args()
    records = read_jsonl(args.data)
    all_rows = expand_variants(records)
    allowed = set(sorted({row["group_id"] for row in records})[:args.groups])
    rows = expand_variants([row for row in records if row["group_id"] in allowed])
    model, tokenizer = load_baseline(args.device, local_files_only=args.offline)
    scoring_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    generation_items = prepare_generation(tokenizer, rows)
    margin_items = prepare_margins(tokenizer, rows)
    batch = margin_items[:args.batch_size]
    prompts = [item.prompt for item in batch]
    encoded = tokenizer(
        [item.text for item in batch], return_tensors="pt", padding=True,
        add_special_tokens=False, return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")
    answer_mask = torch.stack([
        offsets[index, :, 1] > len(prompts[index]) for index in range(len(batch))
    ]).to(model.device)
    encoded = encoded.to(model.device)

    # Warm up kernels before comparing the old and new scoring paths.
    answer_token_logps(model, encoded, answer_mask)
    legacy, legacy_seconds = timed(model.device, lambda: legacy_logps(model, encoded, answer_mask))
    optimized, optimized_seconds = timed(
        model.device, lambda: answer_token_logps(model, encoded, answer_mask),
    )

    generation_batches = length_batches(generation_items, args.batch_size)
    generate_batch(model, tokenizer, generation_batches[0], args.max_new_tokens)
    _, generation_seconds = timed(model.device, lambda: [
        generate_batch(model, tokenizer, batch, args.max_new_tokens)
        for batch in generation_batches
    ])
    _, margins_seconds = timed(model.device, lambda: batch_logps(
        model, tokenizer, margin_items, args.batch_size,
    ))

    calibration = read_jsonl(args.calibration_data)
    recall_row = next(row for row in calibration if row["condition"] == "recall")
    alternatives = recall_row["alternatives"]
    tokenizer.padding_side = scoring_padding_side
    with ChannelGates(model) as gates:
        sequential_alternative_gradient(
            model, tokenizer, gates, recall_row["prompt"], alternatives,
        )
        sequential, sequential_seconds = timed(model.device, lambda: (
            sequential_alternative_gradient(
                model, tokenizer, gates, recall_row["prompt"], alternatives,
            )
        ))
        batched, batched_seconds = timed(model.device, lambda: batched_loss_gradient(
            model, tokenizer, gates, recall_row["prompt"], alternatives,
        ))
    difference = (sequential - batched).abs().max().item()

    useful_generation = sum(item.token_length for item in generation_items)
    useful_margins = sum(item.token_length for item in margin_items)
    report = {
        "device": str(model.device),
        "attention": model.config._attn_implementation,
        "groups": args.groups,
        "variant_rows": len(rows),
        "batch_size": args.batch_size,
        "padding": {
            "generation_original_overhead": padding_tokens(generation_items, args.batch_size, False) / useful_generation - 1,
            "generation_bucketed_overhead": padding_tokens(generation_items, args.batch_size, True) / useful_generation - 1,
            "margins_original_overhead": padding_tokens(margin_items, args.batch_size, False) / useful_margins - 1,
            "margins_bucketed_overhead": padding_tokens(margin_items, args.batch_size, True) / useful_margins - 1,
        },
        "margin_batch": {
            "legacy_seconds": legacy_seconds,
            "optimized_seconds": optimized_seconds,
            "speedup": legacy_seconds / optimized_seconds,
            "max_abs_score_difference": max(abs(a - b) for a, b in zip(legacy, optimized, strict=True)),
        },
        "evaluation_round": {
            "measured_generation_seconds": generation_seconds,
            "measured_margin_seconds": margins_seconds,
            "sample": evaluation_round_cost(
                len(rows), generation_seconds, margins_seconds, len(rows),
            ),
            "full_validation": evaluation_round_cost(
                len(all_rows), generation_seconds, margins_seconds, len(rows),
            ),
        },
        "channel_scoring": {
            "sequential_alternatives_seconds": sequential_seconds,
            "batched_alternatives_seconds": batched_seconds,
            "speedup": sequential_seconds / batched_seconds,
            "alternatives": len(alternatives),
            "max_abs_gradient_difference": difference,
            "max_abs_gradient": sequential.abs().max().item(),
            "relative_gradient_difference": difference / sequential.abs().max().item(),
            "gradient_cosine_similarity": torch.nn.functional.cosine_similarity(
                sequential.flatten(), batched.flatten(), dim=0,
            ).item(),
            "passes": scoring_pass_counts(calibration),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
