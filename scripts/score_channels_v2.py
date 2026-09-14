#!/usr/bin/env python3
"""Score channels with target-balanced CE and answer-margin objectives."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm.auto import tqdm

from model_compression.channels import (
    CONDITIONS,
    ChannelGates,
    ChannelStatistics,
    answer_token_batch,
    calibration_half,
    load_mask,
    read_jsonl,
    target_balance_weights,
)
from model_compression.qwen import MODEL_ID, REVISION, load_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/calibration/mquake_remastered_cf3k_v2/calibration.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/channel_scores/mquake_qwen3_1.7b_v2"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--limit", type=int, help="score at most this many matched groups")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--mask", type=Path,
        help="cumulative channel mask to apply while recomputing survivor scores",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    return args


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def loss_gradient(model, tokenizer, gates, prompt: str, answer: str, completion_prefix: str = "") -> torch.Tensor:
    gates.zero_grad()
    batch = answer_token_batch(tokenizer, prompt, answer, model.device, completion_prefix)
    model(**batch, use_cache=False).loss.backward()
    gradient = gates.values.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError("invalid gate gradient")
    return gradient.detach().clone()


def batched_answer_losses(model, tokenizer, prompt: str, answers: list[str]) -> torch.Tensor:
    """Return one mean answer-token loss per completion in a padded batch."""
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    texts = [prompt_text + answer for answer in answers]
    encoded = tokenizer(
        texts, return_tensors="pt", padding=True, add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")
    answer_mask = (offsets[:, :, 1] > len(prompt_text)).to(model.device)
    encoded = encoded.to(model.device)
    shifted_mask = answer_mask[:, 1:] & encoded.attention_mask[:, 1:].bool()
    if not shifted_mask.any(dim=1).all():
        raise ValueError("an answer produced no supervised tokens")
    hidden = model.model(**encoded, use_cache=False, return_dict=True).last_hidden_state[:, :-1]
    selected_logits = model.lm_head(hidden[shifted_mask]).float()
    selected_targets = encoded.input_ids[:, 1:][shifted_mask]
    token_losses = torch.nn.functional.cross_entropy(
        selected_logits, selected_targets, reduction="none",
    )
    counts = shifted_mask.sum(1).tolist()
    return torch.stack([chunk.mean() for chunk in token_losses.split(counts)])


def batched_loss_gradient(
    model, tokenizer, gates, prompt: str, answers: list[str],
) -> torch.Tensor:
    """Gradient of the equally weighted mean loss for several completions."""
    if not answers:
        raise ValueError("at least one answer is required")
    gates.zero_grad()
    batched_answer_losses(model, tokenizer, prompt, answers).mean().backward()
    gradient = gates.values.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError("invalid gate gradient")
    return gradient.detach().clone()


def main() -> None:
    args = parse_args()
    masked_channels: list[tuple[int, int]] = []
    if args.mask is not None:
        mask_payload = json.loads(args.mask.read_text())
        if mask_payload.get("model") != MODEL_ID or mask_payload.get("revision") != REVISION:
            raise ValueError(f"mask model/revision mismatch: {args.mask}")
        masked_channels = sorted(load_mask(args.mask))
    records = read_jsonl(args.input)
    groups = sorted({row["group_id"] for row in records})
    if args.limit is not None:
        allowed = set(groups[:args.limit])
        records = [row for row in records if row["group_id"] in allowed]
    if sorted({row["condition"] for row in records}) != sorted(CONDITIONS):
        raise ValueError("input must contain all and only the three matched conditions")
    weights = target_balance_weights(records)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "checkpoint.pt"
    checkpoint = None
    processed: set[str] = set()
    if checkpoint_path.exists() and not args.no_resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected = (str(args.input), args.limit, masked_channels)
        actual = (
            checkpoint["input"], checkpoint["limit"],
            [tuple(item) for item in checkpoint.get("masked_channels", [])],
        )
        if actual != expected:
            raise ValueError("checkpoint input/limit/mask differs; use another output or --no-resume")
        processed = set(checkpoint["processed_ids"])
        print(f"resuming after {len(processed)} examples", flush=True)

    model, tokenizer = load_baseline(args.device, local_files_only=args.offline)
    with ChannelGates(model) as gates:
        gates.mask(masked_channels)
        shape = (gates.layer_count, gates.width)
        ce = ChannelStatistics(shape, model.device, checkpoint["ce"] if checkpoint else None)
        margin = ChannelStatistics(shape, model.device, checkpoint["margin"] if checkpoint else None)
        pending = [row for row in records if row["id"] not in processed]
        progress = tqdm(
            pending, initial=len(processed), total=len(records),
            desc="scoring", unit="example",
        )
        for index, row in enumerate(progress, 1):
            # Families that must show their working supply the full checked
            # derivation, so protection covers trace generation, not the final
            # token alone.
            correct_loss_gradient = loss_gradient(
                model, tokenizer, gates, row["prompt"], row.get("score_answer", row["target"])
            )
            weight, half_weight = weights[row["id"]]
            half = calibration_half(row["group_id"])
            ce.add(row["condition"], half, correct_loss_gradient, weight, half_weight)
            if row["condition"] == "recall":
                # Reuse the already-computed correct gradient; only distractors need new passes.
                alternatives = row.get("alternatives", [])
                if not alternatives:
                    raise ValueError(f"no relation-matched alternatives for {row['id']}")
                gradient = -correct_loss_gradient
                gradient.add_(batched_loss_gradient(
                    model, tokenizer, gates, row["prompt"], alternatives,
                ))
                margin.add("recall", half, gradient, weight, half_weight)
            processed.add(row["id"])
            if index % args.checkpoint_every == 0 or index == len(pending):
                atomic_save({
                    "model": MODEL_ID,
                    "revision": REVISION,
                    "input": str(args.input),
                    "limit": args.limit,
                    "masked_channels": masked_channels,
                    "processed_ids": sorted(processed),
                    "ce": ce.state_dict(),
                    "margin": margin.state_dict(),
                }, checkpoint_path)
        progress.close()

        ce_result = ce.results()
        margin_result = margin.summary()
        results = {
            "model": MODEL_ID,
            "revision": REVISION,
            "input": str(args.input),
            "mask": str(args.mask) if args.mask is not None else None,
            "masked_channels": masked_channels,
            "shape": shape,
            "counts": ce_result["counts"],
            "ce": ce_result,
            "margin": margin_result,
            "recall_importance_ce": ce_result["recall_importance"],
            "recall_importance_margin": margin_result["statistics"]["recall"]["signed_mean"],
            "extraction_sensitivity": ce_result["extraction_sensitivity"],
            "reasoning_sensitivity": ce_result["reasoning_sensitivity"],
        }
        atomic_save(results, args.output / "scores.pt")
        manifest = {
            "model": MODEL_ID,
            "revision": REVISION,
            "input": str(args.input),
            "mask": str(args.mask) if args.mask is not None else None,
            "masked_channel_count": len(masked_channels),
            "examples": len(records),
            "groups": len({row["group_id"] for row in records}),
            "shape": list(shape),
            "weighting": "equal normalized-target mass within condition and calibration half",
            "ce_recall_importance": "-weighted mean(dL_correct/dm)",
            "margin_recall_importance": "weighted mean(d[logp(correct)-mean(logp(relation-matched alternatives))]/dm)",
            "protected_sensitivity": "target-balanced weighted mean(abs(dL_correct/dm))",
            "scored_answer": "record score_answer when present, otherwise target",
            "test_split_used": False,
        }
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
