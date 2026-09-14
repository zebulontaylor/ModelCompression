#!/usr/bin/env python3
"""Evaluate an MLP channel mask on external reasoning and factual-recall tasks.

Reasoning uses ARC-Challenge and GSM8K. Recall uses PopQA and CounterFact and
reports retention on the subset answered correctly by the unmasked model, which
separates deletion damage from facts the small baseline never knew.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from model_compression.channels import (
    ChannelGates,
    contains_accepted_answer,
    is_correct,
    load_mask,
)
from model_compression.qwen import MODEL_ID, REVISION, load_baseline


ARC_REVISION = "210d026faf9955653af8916fad021475a3f00453"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
DEFAULT_MASK = Path(
    "artifacts/iterative_deletion/cf9k_sweep_4/round_008_backoff_06/"
    "candidate_cumulative_mask.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/external_generalization/cf9k_sweep_4"),
    )
    parser.add_argument("--arc-sample", type=int, default=300)
    parser.add_argument("--gsm8k-sample", type=int, default=64)
    parser.add_argument("--popqa-sample", type=int, default=800)
    parser.add_argument("--counterfact-sample", type=int, default=800)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gsm8k-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    for name in ("arc_sample", "gsm8k_sample", "popqa_sample", "counterfact_sample"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


def sample_rows(rows: list[dict], count: int, seed: int) -> list[dict]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:count]


def load_arc(count: int, seed: int, offline: bool) -> list[dict]:
    path = hf_hub_download(
        "allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet",
        repo_type="dataset", revision=ARC_REVISION, local_files_only=offline,
    )
    output = []
    for row in sample_rows(pq.read_table(path).to_pylist(), count, seed):
        labels = row["choices"]["label"]
        choices = row["choices"]["text"]
        rendered = "\n".join(f"{label}. {text}" for label, text in zip(labels, choices, strict=True))
        output.append({
            "id": f"arc_challenge:{row['id']}",
            "benchmark": "arc_challenge",
            "kind": "reasoning",
            "prompt": (
                "Answer this multiple-choice science question. Give only the letter "
                "of the correct choice.\n\n"
                f"Question: {row['question']}\n{rendered}\nAnswer:"
            ),
            "target": row["answerKey"],
            "aliases": [],
            "thinking": False,
            "max_new_tokens": 8,
        })
    return output


def gsm8k_target(answer: str) -> str:
    return answer.rsplit("####", 1)[-1].strip().replace(",", "")


def load_gsm8k(count: int, seed: int, offline: bool) -> list[dict]:
    path = hf_hub_download(
        "openai/gsm8k", "main/test-00000-of-00001.parquet",
        repo_type="dataset", revision=GSM8K_REVISION, local_files_only=offline,
    )
    output = []
    for index, row in enumerate(sample_rows(pq.read_table(path).to_pylist(), count, seed + 1)):
        output.append({
            "id": f"gsm8k:{index}",
            "benchmark": "gsm8k",
            "kind": "reasoning",
            "prompt": (
                "Solve the following math word problem. Show a short calculation, using no "
                "more than 120 words, then end with `FINAL: <number>`.\n\n"
                f"Problem: {row['question']}"
            ),
            "target": gsm8k_target(row["answer"]),
            "aliases": [],
            "thinking": False,
            "max_new_tokens": 256,
        })
    return output


def load_popqa(path: Path, count: int, seed: int) -> list[dict]:
    output = []
    for row in sample_rows(pq.read_table(path).to_pylist(), count, seed + 2):
        try:
            answers = [str(value) for value in ast.literal_eval(row["possible_answers"])]
        except (SyntaxError, ValueError):
            answers = []
        target = row["obj"]
        output.append({
            "id": f"popqa:{row['id']}",
            "benchmark": "popqa",
            "kind": "recall",
            "prompt": f"Question: {row['question']}\nAnswer with only the answer:",
            "target": target,
            "aliases": [answer for answer in answers if answer != target],
            "thinking": False,
            "max_new_tokens": 32,
        })
    return output


def load_counterfact(path: Path, count: int, seed: int) -> list[dict]:
    output = []
    for row in sample_rows(pq.read_table(path).to_pylist(), count, seed + 3):
        rewrite = row["requested_rewrite"]
        cloze = rewrite["prompt"].replace("{}", rewrite["subject"])
        output.append({
            "id": f"counterfact:{row['case_id']}",
            "benchmark": "counterfact",
            "kind": "recall",
            "prompt": f"Complete with only the missing words.\n\n{cloze}",
            "target": rewrite["target_true"]["str"],
            "aliases": [],
            "thinking": False,
            "max_new_tokens": 32,
        })
    return output


def parse_arc(text: str) -> str:
    match = re.search(r"(?i)(?:answer\s*[:=]\s*)?\b([A-E])\b", text.strip())
    return match.group(1).upper() if match else ""


def parse_last_number(text: str) -> str:
    text = text.replace(",", "")
    finals = re.findall(
        r"(?i)FINAL\s*:\s*(?:\\boxed\{)?\$?\s*(-?(?:\d+(?:\.\d+)?|\.\d+))", text,
    )
    if finals:
        text = finals[-1]
    matches = re.findall(r"(?<![\w.])-?(?:\d+(?:\.\d+)?|\.\d+)(?![\w.])", text)
    if not matches:
        return ""
    value = matches[-1]
    try:
        numeric = float(value)
        return str(int(numeric)) if numeric.is_integer() else str(numeric)
    except ValueError:
        return value


def score_prediction(row: dict, prediction: str) -> tuple[bool, bool, str]:
    if row["benchmark"] == "arc_challenge":
        scored = parse_arc(prediction)
        correct = scored == row["target"].upper()
        return correct, correct, scored
    if row["benchmark"] == "gsm8k":
        scored = parse_last_number(prediction)
        correct = scored == row["target"]
        return correct, correct, scored
    exact = is_correct(prediction, row["target"], row["aliases"])
    contained = contains_accepted_answer(prediction, row["target"], row["aliases"])
    return exact, contained, prediction.strip()


def generate(
    model, tokenizer, rows: list[dict], batch_size: int, thinking_batch_size: int, desc: str,
) -> list[dict]:
    results = []
    # Separate by decoding settings; length-sort within each setting for efficient padding.
    buckets = defaultdict(list)
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=row["thinking"],
        )
        length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        buckets[(row["thinking"], row["max_new_tokens"])].append((length, row, prompt))

    progress = tqdm(total=len(rows), desc=desc, unit="example")
    for (thinking, max_new_tokens), items in buckets.items():
        items.sort(key=lambda item: item[0])
        effective_batch = thinking_batch_size if thinking else batch_size
        for start in range(0, len(items), effective_batch):
            batch = items[start:start + effective_batch]
            encoded = tokenizer(
                [item[2] for item in batch], return_tensors="pt", padding=True,
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            texts = tokenizer.batch_decode(
                generated[:, encoded.input_ids.shape[1]:], skip_special_tokens=True,
            )
            suffix = generated[:, encoded.input_ids.shape[1]:]
            eos_ids = tokenizer.eos_token_id
            eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
            eos = torch.tensor(eos_ids, device=suffix.device)
            terminated = torch.isin(suffix, eos).any(dim=1).tolist()
            for batch_index, ((_, row, _), text) in enumerate(zip(batch, texts, strict=True)):
                prediction = text.strip()
                exact, contained, scored = score_prediction(row, prediction)
                results.append({
                    **{key: value for key, value in row.items() if key != "prompt"},
                    "prediction": prediction,
                    "scored_prediction": scored,
                    "exact_correct": exact,
                    "contains_answer": contained,
                    "hit_token_limit": suffix.shape[1] >= max_new_tokens and not terminated[batch_index],
                })
            progress.update(len(batch))
    progress.close()
    return sorted(results, key=lambda item: item["id"])


def exact_mcnemar_p(losses: int, gains: int) -> float:
    discordant = losses + gains
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(losses, gains) + 1)) / 2**discordant
    return min(1.0, 2 * tail)


def summarize(baseline: list[dict], masked: list[dict]) -> dict:
    by_id = {row["id"]: row for row in baseline}
    report = {}
    for benchmark in sorted({row["benchmark"] for row in baseline}):
        base = [row for row in baseline if row["benchmark"] == benchmark]
        intervention = [row for row in masked if row["benchmark"] == benchmark]
        n = len(base)
        exact_losses = sum(by_id[row["id"]]["exact_correct"] and not row["exact_correct"] for row in intervention)
        exact_gains = sum(not by_id[row["id"]]["exact_correct"] and row["exact_correct"] for row in intervention)
        content_losses = sum(by_id[row["id"]]["contains_answer"] and not row["contains_answer"] for row in intervention)
        content_gains = sum(not by_id[row["id"]]["contains_answer"] and row["contains_answer"] for row in intervention)
        base_exact = sum(row["exact_correct"] for row in base)
        masked_exact = sum(row["exact_correct"] for row in intervention)
        base_content = sum(row["contains_answer"] for row in base)
        masked_content = sum(row["contains_answer"] for row in intervention)
        report[benchmark] = {
            "kind": base[0]["kind"],
            "count": n,
            "baseline_exact": base_exact / n,
            "masked_exact": masked_exact / n,
            "exact_delta": (masked_exact - base_exact) / n,
            "exact_correct_to_wrong": exact_losses,
            "exact_wrong_to_correct": exact_gains,
            "exact_mcnemar_p": exact_mcnemar_p(exact_losses, exact_gains),
            "baseline_containment": base_content / n,
            "masked_containment": masked_content / n,
            "containment_delta": (masked_content - base_content) / n,
            "content_correct_to_missing": content_losses,
            "content_missing_to_correct": content_gains,
            "baseline_known_count": base_content,
            "baseline_known_retention": (
                (base_content - content_losses) / base_content if base_content else None
            ),
            "containment_mcnemar_p": exact_mcnemar_p(content_losses, content_gains),
            "baseline_token_limit_rate": sum(row["hit_token_limit"] for row in base) / n,
            "masked_token_limit_rate": sum(row["hit_token_limit"] for row in intervention) / n,
        }
    return report


def markdown_report(report: dict, mask: Path, samples: dict[str, int]) -> str:
    lines = [
        "# External generalization check — CF9k sweep 4",
        "",
        f"Mask: `{mask}`",
        f"Deleted channels: **{report['mask']['channel_count']:,} ({report['mask']['fraction']:.2%})**",
        "",
        "All comparisons use the same deterministic sample and greedy decoding. Recall retention is",
        "conditioned on baseline containment-correct examples. McNemar p-values are exact, two-sided,",
        "and descriptive (no multiple-comparison correction).",
        "",
        "| benchmark | n | baseline | masked | delta | correct→wrong | wrong→correct | paired p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("arc_challenge", "gsm8k", "popqa", "counterfact"):
        row = report["benchmarks"].get(name)
        if row is None:
            continue
        metric = "exact" if row["kind"] == "reasoning" else "containment"
        lines.append(
            f"| {name} | {row['count']} | {row[f'baseline_{metric}']:.1%} | "
            f"{row[f'masked_{metric}']:.1%} | {row[f'{metric}_delta']:+.1%} | "
            f"{row['exact_correct_to_wrong' if metric == 'exact' else 'content_correct_to_missing']} | "
            f"{row['exact_wrong_to_correct' if metric == 'exact' else 'content_missing_to_correct']} | "
            f"{row[f'{metric}_mcnemar_p']:.4g} |"
        )
    lines.extend(["", "## Recall retention", "", "| benchmark | baseline-known | retained |", "|---|---:|---:|"])
    for name in ("popqa", "counterfact"):
        row = report["benchmarks"].get(name)
        if row:
            retention = row["baseline_known_retention"]
            rendered = f"{retention:.1%}" if retention is not None else "n/a"
            lines.append(f"| {name} | {row['baseline_known_count']} | {rendered} |")
    lines.extend([
        "", "## Token-limit diagnostic", "",
        "| benchmark | baseline | masked |", "|---|---:|---:|",
    ])
    for name in ("arc_challenge", "gsm8k", "popqa", "counterfact"):
        row = report["benchmarks"].get(name)
        if row:
            lines.append(
                f"| {name} | {row['baseline_token_limit_rate']:.1%} | "
                f"{row['masked_token_limit_rate']:.1%} |"
            )
    lines.extend([
        "", "ARC's eight-token cap only checks the leading answer letter; reaching it is not a",
        "failure. On GSM8K, the cap is diagnostic: prompts explicitly request a short calculation",
        "and baseline responses all terminate, while masked responses that hit the cap usually loop.",
    ])
    lines.extend([
        "", "## Reproducibility", "",
        f"Seed: `{report['seed']}`. Samples: `{json.dumps(samples, sort_keys=True)}`.",
        f"Model: `{MODEL_ID}` at `{REVISION}`.",
        f"ARC revision: `{ARC_REVISION}`. GSM8K revision: `{GSM8K_REVISION}`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    mask_payload = json.loads(args.mask.read_text())
    if mask_payload.get("model") != MODEL_ID or mask_payload.get("revision") != REVISION:
        raise ValueError("mask model/revision does not match the configured baseline")

    records = [
        *load_arc(args.arc_sample, args.seed, args.offline),
        *load_gsm8k(args.gsm8k_sample, args.seed, args.offline),
        *load_popqa(Path("data/raw/popqa.parquet"), args.popqa_sample, args.seed),
        *load_counterfact(Path("data/raw/counterfact.parquet"), args.counterfact_sample, args.seed),
    ]
    if not records:
        raise ValueError("at least one sample count must be positive")

    model, tokenizer = load_baseline(args.device, local_files_only=args.offline)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    baseline = generate(
        model, tokenizer, records, args.batch_size, args.gsm8k_batch_size, "baseline",
    )
    with ChannelGates(model, requires_grad=False) as gates:
        gates.mask(load_mask(args.mask))
        masked = generate(
            model, tokenizer, records, args.batch_size, args.gsm8k_batch_size, "sweep4-mask",
        )

    for row in baseline:
        row["mode"] = "baseline"
    for row in masked:
        row["mode"] = "masked"
    samples = {
        "arc_challenge": args.arc_sample,
        "gsm8k": args.gsm8k_sample,
        "popqa": args.popqa_sample,
        "counterfact": args.counterfact_sample,
    }
    report = {
        "schema": 1,
        "seed": args.seed,
        "model": {"id": MODEL_ID, "revision": REVISION},
        "mask": {
            "path": str(args.mask.resolve()),
            "channel_count": len(mask_payload["channels"]),
            "fraction": len(mask_payload["channels"]) / mask_payload["total_channels"],
        },
        "datasets": {
            "arc_challenge": {"repo": "allenai/ai2_arc", "revision": ARC_REVISION, "split": "test"},
            "gsm8k": {"repo": "openai/gsm8k", "revision": GSM8K_REVISION, "split": "test"},
            "popqa": {"path": str(Path("data/raw/popqa.parquet").resolve())},
            "counterfact": {"path": str(Path("data/raw/counterfact.parquet").resolve())},
        },
        "samples": samples,
        "benchmarks": summarize(baseline, masked),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "predictions.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in [*baseline, *masked]
    ))
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "REPORT.md").write_text(markdown_report(report, args.mask, samples))
    print(markdown_report(report, args.mask, samples))


if __name__ == "__main__":
    main()
