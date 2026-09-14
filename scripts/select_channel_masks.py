#!/usr/bin/env python3
"""Select small channel cohorts for validating gradient predictions."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch

from model_compression.qwen import MODEL_ID, REVISION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/channel_masks/initial_validation"))
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--protected-quantile", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    if not 0 < args.protected_quantile < 1:
        parser.error("--protected-quantile must be between zero and one")
    return args


def unravel(indices: torch.Tensor, width: int) -> list[tuple[int, int]]:
    return [(int(index // width), int(index % width)) for index in indices.tolist()]


def predicted_effects(channels, recall, extraction, reasoning) -> dict:
    locations = tuple(zip(*channels, strict=True))
    return {
        "recall_loss_change_first_order": float(recall[locations].sum()),
        "extraction_first_order_abs_bound": float(extraction[locations].sum()),
        "reasoning_first_order_abs_bound": float(reasoning[locations].sum()),
    }


def write_mask(
    path: Path, method: str, channels: list[tuple[int, int]], extra: dict,
    recall: torch.Tensor, extraction: torch.Tensor, reasoning: torch.Tensor,
) -> None:
    payload = {
        "model": MODEL_ID,
        "revision": REVISION,
        "method": method,
        **extra,
        "predicted_effects": predicted_effects(channels, recall, extraction, reasoning),
        "channels": [
            {
                "layer": layer,
                "channel": channel,
                "recall_importance": float(recall[layer, channel]),
                "extraction_sensitivity": float(extraction[layer, channel]),
                "reasoning_sensitivity": float(reasoning[layer, channel]),
            }
            for layer, channel in channels
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def ranking_stability(scores: dict) -> dict:
    """Compare calibration-half rankings with Spearman and top-1% overlap."""
    definitions = {
        "recall_importance": ("recall_half0", "recall_half1", "signed_mean", -1),
        "extraction_sensitivity": ("extraction_half0", "extraction_half1", "abs_mean", 1),
        "reasoning_sensitivity": ("reasoning_half0", "reasoning_half1", "abs_mean", 1),
    }
    statistics = scores.get("statistics", {})
    result = {}
    for label, (half0, half1, field, sign) in definitions.items():
        if half0 not in statistics or half1 not in statistics:
            result[label] = None
            continue
        left = (statistics[half0][field] * sign).flatten().float()
        right = (statistics[half1][field] * sign).flatten().float()
        left_rank = torch.argsort(torch.argsort(left)).float()
        right_rank = torch.argsort(torch.argsort(right)).float()
        spearman = torch.corrcoef(torch.stack((left_rank, right_rank)))[0, 1]
        top_count = max(1, left.numel() // 100)
        left_top = set(torch.topk(left, top_count).indices.tolist())
        right_top = set(torch.topk(right, top_count).indices.tolist())
        result[label] = {
            "spearman": float(spearman),
            "top_1_percent_overlap": len(left_top & right_top) / top_count,
            "half0_examples": scores["counts"][half0],
            "half1_examples": scores["counts"][half1],
        }
    return result


def main() -> None:
    args = parse_args()
    scores = torch.load(args.scores, map_location="cpu", weights_only=False)
    recall = scores["recall_importance"].float()
    extraction = scores["extraction_sensitivity"].float()
    reasoning = scores["reasoning_sensitivity"].float()
    if recall.shape != extraction.shape or recall.shape != reasoning.shape:
        raise ValueError("score tensor shapes differ")
    layers, width = recall.shape
    total = recall.numel()
    if args.count > total:
        raise ValueError(f"--count exceeds the {total} available channels")

    extraction_cutoff = torch.quantile(extraction.flatten(), args.protected_quantile)
    reasoning_cutoff = torch.quantile(reasoning.flatten(), args.protected_quantile)
    eligible = (extraction <= extraction_cutoff) & (reasoning <= reasoning_cutoff)
    eligible_count = int(eligible.sum())
    if eligible_count < args.count:
        raise ValueError(f"only {eligible_count} channels meet both protected thresholds")
    contrastive_values = recall.masked_fill(~eligible, -torch.inf).flatten()
    contrastive = unravel(torch.topk(contrastive_values, args.count).indices, width)

    combined_protected = extraction / extraction.median().clamp_min(1e-12)
    combined_protected += reasoning / reasoning.median().clamp_min(1e-12)
    low_importance_value = recall.abs() / recall.abs().median().clamp_min(1e-12)
    low_importance_value += combined_protected
    low_importance = unravel(torch.topk(-low_importance_value.flatten(), args.count).indices, width)
    protected_important = unravel(torch.topk(combined_protected.flatten(), args.count).indices, width)

    selected = set(contrastive)
    by_layer = Counter(layer for layer, _ in contrastive)
    generator = random.Random(args.seed)
    random_matched = []
    for layer, count in sorted(by_layer.items()):
        candidates = [(layer, channel) for channel in range(width) if (layer, channel) not in selected]
        random_matched.extend(generator.sample(candidates, count))

    args.output.mkdir(parents=True, exist_ok=True)
    common = {
        "source_scores": str(args.scores),
        "count": args.count,
        "protected_quantile": args.protected_quantile,
    }
    write_mask(args.output / "contrastive.json", "high recall importance within low E/R thresholds", contrastive, common, recall, extraction, reasoning)
    write_mask(args.output / "low_importance.json", "low normalized total importance", low_importance, common, recall, extraction, reasoning)
    write_mask(args.output / "protected_important.json", "high normalized extraction/reasoning sensitivity", protected_important, common, recall, extraction, reasoning)
    write_mask(args.output / "random_layer_matched.json", "random with contrastive layer counts", random_matched, {**common, "seed": args.seed}, recall, extraction, reasoning)
    stability = ranking_stability(scores)
    (args.output / "stability.json").write_text(json.dumps(stability, indent=2) + "\n")
    report = {
        "output": str(args.output),
        "shape": [layers, width],
        "eligible_channels": eligible_count,
        "extraction_cutoff": float(extraction_cutoff),
        "reasoning_cutoff": float(reasoning_cutoff),
        "masks": ["contrastive", "low_importance", "protected_important", "random_layer_matched"],
        "ranking_stability": stability,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
