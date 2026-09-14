#!/usr/bin/env python3
"""Select balanced CE, margin, consensus, and random-control channel masks."""

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
    parser.add_argument("--output", type=Path, default=Path("artifacts/channel_masks/robust_validation"))
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--protected-quantile", type=float, default=0.25)
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    if args.count < 1 or not 0 < args.protected_quantile < 1:
        parser.error("count must be positive and protected quantile must be between zero and one")
    return args


def unravel(indices: torch.Tensor, width: int) -> list[tuple[int, int]]:
    return [(int(i // width), int(i % width)) for i in indices.tolist()]


def percentile_ranks(values: torch.Tensor) -> torch.Tensor:
    flat = values.flatten()
    return torch.argsort(torch.argsort(flat)).reshape(values.shape).float() / max(1, flat.numel() - 1)


def stability(left: torch.Tensor, right: torch.Tensor) -> dict:
    left, right = left.flatten().float(), right.flatten().float()
    left_rank = torch.argsort(torch.argsort(left)).float()
    right_rank = torch.argsort(torch.argsort(right)).float()
    top_count = max(1, left.numel() // 100)
    left_top = set(torch.topk(left, top_count).indices.tolist())
    right_top = set(torch.topk(right, top_count).indices.tolist())
    return {
        "spearman": float(torch.corrcoef(torch.stack((left_rank, right_rank)))[0, 1]),
        "top_1_percent_overlap": len(left_top & right_top) / top_count,
    }


def choose(values: torch.Tensor, eligible: torch.Tensor, count: int) -> list[tuple[int, int]]:
    ranked = values.masked_fill(~eligible, -torch.inf).flatten()
    return unravel(torch.topk(ranked, count).indices, values.shape[1])


def predicted(channels, ce, margin, extraction, reasoning) -> dict:
    locations = tuple(zip(*channels, strict=True))
    return {
        "recall_ce_loss_change_first_order": float(ce[locations].sum()),
        "recall_margin_drop_first_order": float(margin[locations].sum()),
        "extraction_first_order_abs_bound": float(extraction[locations].sum()),
        "reasoning_first_order_abs_bound": float(reasoning[locations].sum()),
    }


def write_mask(path, method, channels, scores, metadata) -> None:
    ce, margin, extraction, reasoning = scores
    payload = {
        "model": MODEL_ID,
        "revision": REVISION,
        "method": method,
        **metadata,
        "predicted_effects": predicted(channels, *scores),
        "channels": [
            {
                "layer": layer,
                "channel": channel,
                "recall_importance_ce": float(ce[layer, channel]),
                "recall_importance_margin": float(margin[layer, channel]),
                "extraction_sensitivity": float(extraction[layer, channel]),
                "reasoning_sensitivity": float(reasoning[layer, channel]),
            }
            for layer, channel in channels
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    payload = torch.load(args.scores, map_location="cpu", weights_only=False)
    tensors = tuple(payload[key].float() for key in (
        "recall_importance_ce", "recall_importance_margin",
        "extraction_sensitivity", "reasoning_sensitivity",
    ))
    ce, margin, extraction, reasoning = tensors
    if len({tuple(x.shape) for x in tensors}) != 1:
        raise ValueError("score tensor shapes differ")
    extraction_cutoff = torch.quantile(extraction.flatten(), args.protected_quantile)
    reasoning_cutoff = torch.quantile(reasoning.flatten(), args.protected_quantile)
    eligible = (extraction <= extraction_cutoff) & (reasoning <= reasoning_cutoff)
    if int(eligible.sum()) < args.count:
        raise ValueError("not enough channels meet both protected thresholds")

    ce_mask = choose(ce, eligible, args.count)
    margin_mask = choose(margin, eligible, args.count)
    consensus_score = (percentile_ranks(ce) + percentile_ranks(margin)) / 2
    consensus = choose(consensus_score, eligible, args.count)
    args.output.mkdir(parents=True, exist_ok=True)
    common = {
        "source_scores": str(args.scores),
        "count": args.count,
        "protected_quantile": args.protected_quantile,
    }
    write_mask(args.output / "contrastive_ce_balanced.json", "target-balanced CE contrast", ce_mask, tensors, common)
    write_mask(args.output / "contrastive_margin_balanced.json", "target-balanced answer-margin contrast", margin_mask, tensors, common)
    write_mask(args.output / "contrastive_consensus.json", "mean percentile rank of balanced CE and margin", consensus, tensors, common)

    layer_counts = Counter(layer for layer, _ in consensus)
    excluded = set(consensus)
    random_names = []
    for seed in args.random_seeds:
        generator = random.Random(seed)
        channels = []
        for layer, count in sorted(layer_counts.items()):
            candidates = [(layer, channel) for channel in range(ce.shape[1]) if (layer, channel) not in excluded]
            channels.extend(generator.sample(candidates, count))
        name = f"random_layer_matched_seed_{seed}"
        random_names.append(name)
        write_mask(args.output / f"{name}.json", "random matched to consensus layer counts", channels, tensors, {**common, "seed": seed})

    report = {
        "output": str(args.output),
        "shape": list(ce.shape),
        "eligible_channels": int(eligible.sum()),
        "extraction_cutoff": float(extraction_cutoff),
        "reasoning_cutoff": float(reasoning_cutoff),
        "masks": ["contrastive_ce_balanced", "contrastive_margin_balanced", "contrastive_consensus", *random_names],
        "ce_margin_overlap": len(set(ce_mask) & set(margin_mask)),
        "consensus_ce_overlap": len(set(consensus) & set(ce_mask)),
        "consensus_margin_overlap": len(set(consensus) & set(margin_mask)),
        "ranking_stability": {
            "recall_importance_ce": stability(
                -payload["ce"]["statistics"]["recall_half0"]["signed_mean"],
                -payload["ce"]["statistics"]["recall_half1"]["signed_mean"],
            ),
            "recall_importance_margin": stability(
                payload["margin"]["statistics"]["recall_half0"]["signed_mean"],
                payload["margin"]["statistics"]["recall_half1"]["signed_mean"],
            ),
            "extraction_sensitivity": stability(
                payload["ce"]["statistics"]["extraction_half0"]["abs_mean"],
                payload["ce"]["statistics"]["extraction_half1"]["abs_mean"],
            ),
            "reasoning_sensitivity": stability(
                payload["ce"]["statistics"]["reasoning_half0"]["abs_mean"],
                payload["ce"]["statistics"]["reasoning_half1"]["abs_mean"],
            ),
        },
    }
    (args.output / "selection_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
