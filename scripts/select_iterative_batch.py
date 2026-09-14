#!/usr/bin/env python3
"""Select the next percentage-based batch under a cumulative channel mask."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from model_compression.channels import load_mask
from model_compression.qwen import MODEL_ID, REVISION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("cumulative_mask", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    batch_size = parser.add_mutually_exclusive_group()
    batch_size.add_argument(
        "--fraction",
        type=float,
        default=0.01,
        help="fraction of the original MLP channels to delete this round (default: 0.01)",
    )
    batch_size.add_argument(
        "--count",
        type=int,
        help="fixed channels to delete this round (overrides --fraction)",
    )
    parser.add_argument(
        "--target-fraction",
        type=float,
        default=0.05,
        help="cap on cumulative deletion as a fraction of original channels (default: 0.05)",
    )
    parser.add_argument("--protected-quantile", type=float, default=0.25)
    parser.add_argument("--family-scores", type=Path, help="additional protected family sensitivities, measured under the same parent mask")
    parser.add_argument(
        "--recall-weight", type=float, default=0.05,
        help="secondary recall utility weight after continuous protected cost (default: 0.05)",
    )
    parser.add_argument(
        "--max-layer-fraction", type=float, default=0.15,
        help="maximum cumulative deletion fraction in any one layer (default: 0.15)",
    )
    args = parser.parse_args()
    if args.count is not None and args.count < 1:
        parser.error("count must be positive")
    if not 0 < args.fraction <= 1:
        parser.error("fraction must be greater than zero and at most one")
    if not 0 < args.target_fraction <= 1:
        parser.error("target fraction must be greater than zero and at most one")
    if not 0 < args.protected_quantile < 1:
        parser.error("protected quantile must be between zero and one")
    if args.recall_weight < 0:
        parser.error("recall weight must be nonnegative")
    if not 0 < args.max_layer_fraction <= 1:
        parser.error("max layer fraction must be greater than zero and at most one")
    return args


def percentile_ranks(values: torch.Tensor, surviving: torch.Tensor) -> torch.Tensor:
    """Rank surviving entries only, leaving deleted entries at negative infinity."""
    result = torch.full_like(values, -torch.inf, dtype=torch.float32)
    flat_values = values[surviving]
    ranks = torch.argsort(torch.argsort(flat_values)).float() / max(1, flat_values.numel() - 1)
    result[surviving] = ranks
    return result


def select_channels(
    ce: torch.Tensor,
    margin: torch.Tensor,
    extraction: torch.Tensor,
    reasoning: torch.Tensor,
    surviving: torch.Tensor,
    previous: list[tuple[int, int]],
    count: int,
    protected_quantile: float,
    recall_weight: float,
    max_layer_fraction: float,
    family_sensitivities: dict[str, torch.Tensor] | None = None,
) -> tuple[list[tuple[int, int]], dict[str, torch.Tensor | float | int]]:
    """Select low-cost channels, using recall utility only as a secondary term.

    Absolute protected gradients are converted to survivor percentile ranks so
    extraction and reasoning share a scale.  The larger rank is the channel's
    protected cost; this prevents a very cheap score on one protected task from
    hiding an expensive score on the other.
    """
    extraction_rank = percentile_ranks(extraction, surviving)
    reasoning_rank = percentile_ranks(reasoning, surviving)
    recall_rank = (
        percentile_ranks(ce, surviving) + percentile_ranks(margin, surviving)
    ) / 2
    protected_cost = torch.maximum(extraction_rank, reasoning_rank)

    extraction_cutoff = torch.quantile(extraction[surviving], protected_quantile)
    reasoning_cutoff = torch.quantile(reasoning[surviving], protected_quantile)
    eligible = surviving & (extraction <= extraction_cutoff) & (reasoning <= reasoning_cutoff)
    for family, sensitivity in (family_sensitivities or {}).items():
        if sensitivity.shape != ce.shape or not torch.isfinite(sensitivity).all():
            raise ValueError(f"invalid sensitivity for protected family {family}")
        protected_cost = torch.maximum(protected_cost, percentile_ranks(sensitivity, surviving))
        eligible &= sensitivity <= torch.quantile(sensitivity[surviving], protected_quantile)
    priority = protected_cost - recall_weight * recall_rank
    width = ce.shape[1]
    layer_cap = math.ceil(width * max_layer_fraction)
    layer_counts = [0] * ce.shape[0]
    for layer, _ in previous:
        layer_counts[layer] += 1

    # Stable sorting makes channel coordinates the deterministic tie-breaker.
    order = torch.argsort(priority.masked_fill(~eligible, torch.inf).flatten(), stable=True)
    selected = []
    for flat_index in order.tolist():
        if not eligible.flatten()[flat_index]:
            break
        layer, channel = divmod(flat_index, width)
        if layer_counts[layer] >= layer_cap:
            continue
        selected.append((layer, channel))
        layer_counts[layer] += 1
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(
            f"only {len(selected)} channels meet the protected thresholds and per-layer cap; "
            f"need {count}"
        )
    locations = tuple(zip(*selected, strict=True))
    diagnostics = {
        "eligible": eligible,
        "protected_cost": protected_cost,
        "recall_rank": recall_rank,
        "extraction_cutoff": float(extraction_cutoff),
        "reasoning_cutoff": float(reasoning_cutoff),
        "layer_cap": layer_cap,
        "layer_counts": layer_counts,
        "selected_protected_cost_sum": float(protected_cost[locations].sum()),
        "selected_protected_cost_max": float(protected_cost[locations].max()),
    }
    return sorted(selected), diagnostics


def main() -> None:
    args = parse_args()
    payload = torch.load(args.scores, map_location="cpu", weights_only=False)
    mask_payload = json.loads(args.cumulative_mask.read_text())
    for source, name in ((payload, "scores"), (mask_payload, "cumulative mask")):
        if source.get("model") != MODEL_ID or source.get("revision") != REVISION:
            raise ValueError(f"{name} model/revision mismatch")

    previous = sorted(load_mask(args.cumulative_mask))
    score_mask = sorted(tuple(x) for x in payload.get("masked_channels", []))
    if score_mask != previous:
        raise ValueError("scores were not recomputed under the supplied cumulative mask")

    keys = (
        "recall_importance_ce", "recall_importance_margin",
        "extraction_sensitivity", "reasoning_sensitivity",
    )
    ce, margin, extraction, reasoning = (payload[key].float() for key in keys)
    if len({tuple(payload[key].shape) for key in keys}) != 1:
        raise ValueError("score tensor shapes differ")

    total_channels = ce.numel()
    target_count = math.ceil(total_channels * args.target_fraction)
    requested_count = args.count or math.ceil(total_channels * args.fraction)
    count = min(requested_count, target_count - len(previous))
    if count <= 0:
        print(json.dumps({
            "status": "target_reached",
            "previous_count": len(previous),
            "total_channels": total_channels,
            "cumulative_fraction": len(previous) / total_channels,
            "target_count": target_count,
            "target_fraction": args.target_fraction,
        }, indent=2))
        return

    surviving = torch.ones_like(ce, dtype=torch.bool)
    for layer, channel in previous:
        surviving[layer, channel] = False
    families = None
    if args.family_scores is not None:
        family_payload = torch.load(args.family_scores, map_location="cpu", weights_only=False)
        if (family_payload.get("model"), family_payload.get("revision")) != (MODEL_ID, REVISION):
            raise ValueError("family score model/revision mismatch")
        if sorted(tuple(x) for x in family_payload.get("masked_channels", [])) != previous:
            raise ValueError("family scores were not measured under the supplied cumulative mask")
        families = family_payload["family_sensitivities"]
    batch, selection = select_channels(
        ce, margin, extraction, reasoning, surviving, previous, count,
        args.protected_quantile, args.recall_weight, args.max_layer_fraction, families,
    )
    locations = tuple(zip(*batch, strict=True))

    def channel_row(layer: int, channel: int) -> dict:
        return {
            "layer": layer,
            "channel": channel,
            "recall_importance_ce": float(ce[layer, channel]),
            "recall_importance_margin": float(margin[layer, channel]),
            "extraction_sensitivity": float(extraction[layer, channel]),
            "reasoning_sensitivity": float(reasoning[layer, channel]),
        }

    common = {
        "model": MODEL_ID,
        "revision": REVISION,
        "source_scores": str(args.scores),
        "family_scores": str(args.family_scores) if args.family_scores else None,
        "parent_mask": str(args.cumulative_mask),
        "protected_quantile": args.protected_quantile,
        "recall_weight": args.recall_weight,
        "max_layer_fraction": args.max_layer_fraction,
        "per_layer_cumulative_cap": selection["layer_cap"],
        "total_channels": total_channels,
        "requested_batch_count": requested_count,
        "target_count": target_count,
        "target_fraction": args.target_fraction,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    batch_payload = {
        **common,
        "method": "continuous protected percentile cost with secondary recall utility",
        "count": len(batch),
        "fraction_of_original": len(batch) / total_channels,
        "predicted_effects": {
            "recall_ce_loss_change_first_order": float(ce[locations].sum()),
            "recall_margin_drop_first_order": float(margin[locations].sum()),
            "extraction_first_order_abs_bound": float(extraction[locations].sum()),
            "reasoning_first_order_abs_bound": float(reasoning[locations].sum()),
            "protected_percentile_cost_sum": selection["selected_protected_cost_sum"],
            "protected_percentile_cost_max": selection["selected_protected_cost_max"],
        },
        "channels": [channel_row(*item) for item in batch],
    }
    cumulative_payload = {
        **common,
        "method": "cumulative accepted iterative deletion plus candidate batch",
        "accepted_parent_count": len(previous),
        "candidate_batch_count": len(batch),
        "count": len(previous) + len(batch),
        "fraction_of_original": (len(previous) + len(batch)) / total_channels,
        "channels": [
            {"layer": layer, "channel": channel}
            for layer, channel in sorted([*previous, *batch])
        ],
    }
    (args.output / "candidate_batch.json").write_text(json.dumps(batch_payload, indent=2) + "\n")
    (args.output / "candidate_cumulative_mask.json").write_text(
        json.dumps(cumulative_payload, indent=2) + "\n"
    )
    print(json.dumps({
        "previous_count": len(previous),
        "batch_count": len(batch),
        "candidate_cumulative_count": len(previous) + len(batch),
        "batch_fraction_of_original": len(batch) / total_channels,
        "candidate_cumulative_fraction": (len(previous) + len(batch)) / total_channels,
        "target_count": target_count,
        "target_fraction": args.target_fraction,
        "eligible_survivors": int(selection["eligible"].sum()),
        "extraction_cutoff": selection["extraction_cutoff"],
        "reasoning_cutoff": selection["reasoning_cutoff"],
        "selected_protected_cost_sum": selection["selected_protected_cost_sum"],
        "selected_protected_cost_max": selection["selected_protected_cost_max"],
        "cumulative_channels_by_layer": selection["layer_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
