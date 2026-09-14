"""Friendly entry point for Colab and first-time runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model_compression.channels import ChannelGates, answer_token_batch, read_jsonl
from model_compression.qwen import MODEL_PRESETS, load_baseline, resolve_model


DATASET_PRESETS = {
    "mquake-cf3k": Path("data/calibration/mquake_remastered_cf3k_v2/calibration.jsonl"),
    "mquake-cf9k": Path("data/calibration/mquake_remastered_cf9k_v2/calibration.jsonl"),
    "mquake-openmath": Path("data/calibration/mquake_cf9k_openmath_v2/calibration.jsonl"),
    "openmath": Path("data/calibration/openmath_v2/calibration.jsonl"),
    "2wiki": Path("data/calibration/2wiki_v2/calibration.jsonl"),
}


def resolve_dataset(value: str) -> Path:
    return DATASET_PRESETS.get(value, Path(value))


def add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="qwen3-0.6b", help="preset or Hugging Face model ID")
    parser.add_argument("--revision", help="optional Hugging Face revision")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--offline", action="store_true")


def command_list() -> None:
    print("Models:")
    for name, (model_id, revision) in MODEL_PRESETS.items():
        suffix = f" @ {revision}" if revision else ""
        print(f"  {name:16} {model_id}{suffix}")
    print("\nDatasets:")
    for name, path in DATASET_PRESETS.items():
        print(f"  {name:16} {path}")


def command_generate(args: argparse.Namespace) -> None:
    model_id, revision = resolve_model(args.model, args.revision)
    model, tokenizer = load_baseline(
        args.device, args.offline, model_id=model_id, revision=revision,
    )
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
    except (AttributeError, TypeError, ValueError):
        text = args.prompt
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))


def command_score(args: argparse.Namespace) -> None:
    dataset = resolve_dataset(args.dataset)
    if not dataset.is_file():
        raise SystemExit(f"dataset not found: {dataset}")
    records = read_jsonl(dataset)
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise SystemExit("dataset contains no records")

    model_id, revision = resolve_model(args.model, args.revision)
    model, tokenizer = load_baseline(
        args.device, args.offline, model_id=model_id, revision=revision,
    )
    with ChannelGates(model) as gates:
        total = torch.zeros_like(gates.values)
        for index, row in enumerate(records, 1):
            gates.zero_grad()
            answer = row.get("score_answer", row["target"])
            batch = answer_token_batch(tokenizer, row["prompt"], answer, model.device)
            model(**batch, use_cache=False).loss.backward()
            if gates.values.grad is None or not torch.isfinite(gates.values.grad).all():
                raise RuntimeError(f"invalid gradient for {row.get('id', index)}")
            total.add_(gates.values.grad.detach().abs())
            print(f"scored {index}/{len(records)}", flush=True)

        scores = total.div(len(records)).cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model_id,
        "revision": revision,
        "dataset": str(dataset),
        "examples": len(records),
        "score": "mean absolute answer-loss gate gradient",
        "scores": scores,
    }, args.output)
    print(json.dumps({
        "output": str(args.output), "model": model_id,
        "dataset": str(dataset), "examples": len(records),
        "shape": list(scores.shape),
    }, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model-compression")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="show bundled model and dataset choices")

    generate = subparsers.add_parser("generate", help="smoke-test a model")
    add_runtime_options(generate)
    generate.add_argument("--prompt", default="What is 2 + 2? Answer briefly.")
    generate.add_argument("--max-new-tokens", type=int, default=64)

    score = subparsers.add_parser("score", help="score MLP channels on a dataset")
    add_runtime_options(score)
    score.add_argument("--dataset", default="mquake-cf3k", help="preset or JSONL path")
    score.add_argument("--limit", type=int, default=3, help="examples to score (0 means all)")
    score.add_argument("--output", type=Path, default=Path("outputs/channel_scores.pt"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "max_new_tokens", 1) < 1:
        parser.error("--max-new-tokens must be positive")
    if getattr(args, "limit", 0) < 0:
        parser.error("--limit cannot be negative")
    if args.command == "list":
        command_list()
    elif args.command == "generate":
        command_generate(args)
    else:
        command_score(args)


if __name__ == "__main__":
    main()
