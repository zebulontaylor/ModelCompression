#!/usr/bin/env python3
"""Build a baseline-known protected math-reasoning family from OpenMathInstruct-2.

One pinned shard of `nvidia/OpenMathInstruct-2` is filtered to original (not
augmented) GSM8K/MATH problems with integer answers, then reduced to the
problems the frozen baseline already solves, free-running, in the pilot's
explicit-working format.  The baseline's own accepted derivation is kept as
`score_answer` so channel scoring protects trace generation rather than the
final token alone, and each record carries its own `max_new_tokens` budget.

The result is a protected `openmath` family of `condition: reasoning` records
sharing the v2 MQuAKE record fields, so it concatenates into the calibration
and validation corpora.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path

import torch

from model_compression.channels import (
    is_correct, normalize_answer, relation_matched_alternatives, write_jsonl,
)
from model_compression.qwen import MODEL_ID, REVISION, load_baseline


SOURCE_REPO = "nvidia/OpenMathInstruct-2"
SOURCE_REVISION = "469216e3f46f4dacf476b382e192485ea51a143e"
SOURCE_FILE = "data/train-00000-of-00032.parquet"
SOURCE_URL = (
    f"https://huggingface.co/datasets/{SOURCE_REPO}/resolve/{SOURCE_REVISION}/{SOURCE_FILE}"
)
SOURCE_SHA256 = "55bae5fa27c8956ab996a94d82fdcda683a650d0a16fe9ee6d26c831adc31087"
ORIGINAL_SOURCES = ("gsm8k", "math")
INTEGER_ANSWER = re.compile(r"^-?\d{1,9}$")
WORKING_INSTRUCTION = (
    "\nWork through the problem briefly. End with exactly 'Final answer: <answer>' "
    "on its own line. The final answer must contain only the requested number."
)
FINAL_ANSWER_LINE = re.compile(r"(?im)^[ \t]*(?:#+[ \t]*)?\**Final answer[ \t]*:[ \t]*[^\n]+")
FINAL_ANSWER_VALUE = re.compile(r"(?im)^[ \t]*(?:#+[ \t]*)?Final answer[ \t]*:[ \t]*([^\n]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_source(path: Path, expected: str | None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise RuntimeError(f"source checksum mismatch: expected {expected}, got {actual}")
        return actual
    partial = path.with_suffix(path.suffix + ".partial")
    urllib.request.urlretrieve(SOURCE_URL, partial)
    actual = sha256(partial)
    if expected is not None and actual != expected:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"source checksum mismatch: expected {expected}, got {actual}")
    partial.replace(path)
    return actual


def working_prompt(problem: str) -> str:
    return f"{problem}{WORKING_INSTRUCTION}"


def final_answer(prediction: str) -> str:
    """Extract the answer exactly as `evaluate_robustness` scores it."""
    plain = prediction.replace("**", "").replace("`", "")
    matches = FINAL_ANSWER_VALUE.findall(plain)
    answer = matches[-1].strip() if matches else ""
    return re.sub(r"</?answer>", "", answer, flags=re.IGNORECASE).strip()


def accepted_derivation(prediction: str) -> str | None:
    """Keep the checked trace through its last final-answer line, and no further."""
    matches = list(FINAL_ANSWER_LINE.finditer(prediction))
    if not matches:
        return None
    trace = prediction[: matches[-1].end()].strip()
    return trace or None


def answer_aliases(target: str) -> list[str]:
    """Accept the ordinary written forms of the same integer."""
    aliases = {f"${target}"}
    value = int(target)
    if abs(value) >= 1000:
        aliases.add(f"{value:,}")
        aliases.add(f"${value:,}")
    aliases.discard(target)
    return sorted(aliases)


def load_candidates(raw: Path, limit: int, seed: int, max_problem_characters: int) -> list[dict]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - environment guard
        raise SystemExit("pyarrow is required; run with `uv run --extra data ...`") from error

    table = parquet.read_table(
        raw, columns=["problem", "expected_answer", "problem_source"],
    )
    seen: set[str] = set()
    kept: list[dict] = []
    for index, row in enumerate(table.to_pylist()):
        source = row["problem_source"]
        answer = (row["expected_answer"] or "").strip()
        problem = " ".join((row["problem"] or "").split())
        if source not in ORIGINAL_SOURCES or not INTEGER_ANSWER.match(answer):
            continue
        if not 20 <= len(problem) <= max_problem_characters:
            continue
        key = normalize_answer(problem)
        if key in seen:
            continue
        seen.add(key)
        kept.append({
            "source_index": index, "problem": problem,
            "target": answer, "problem_source": source,
        })
    random.Random(seed).shuffle(kept)
    return kept[:limit]


def as_record(candidate: dict, split: str, ordinal: int, max_new_tokens: int) -> dict:
    identifier = f"openmath:{split}:{ordinal:05d}"
    prompt = working_prompt(candidate["problem"])
    return {
        "group_id": identifier,
        "source": f"{SOURCE_REPO}:{SOURCE_FILE}",
        "source_index": candidate["source_index"],
        "problem_source": candidate["problem_source"],
        "split": split,
        "id": f"{identifier}:reasoning",
        "condition": "reasoning",
        "family": "openmath",
        "relation": f"openmath_{candidate['problem_source']}",
        "prompt": prompt,
        "target": candidate["target"],
        "aliases": answer_aliases(candidate["target"]),
        "answer_format": "final_answer",
        "max_new_tokens": max_new_tokens,
        "facts": [],
        "question_variants": [candidate["problem"]],
        "prompt_variants": [{"label": "original", "prompt": prompt}],
    }


def generate(model, tokenizer, records, batch_size, max_new_tokens) -> list[dict]:
    predictions = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": record["prompt"]}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            for record in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        texts = tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
        )
        for record, text in zip(batch, texts, strict=True):
            prediction = text.strip()
            trace = accepted_derivation(prediction)
            scored = final_answer(prediction)
            predictions.append({
                "id": record["id"], "split": record["split"],
                "problem_source": record["problem_source"],
                "target": record["target"], "prediction": prediction,
                "scored_prediction": scored,
                "derivation": trace,
                "truncated": trace is None,
                "correct": trace is not None
                and is_correct(scored, record["target"], record["aliases"]),
            })
        print(f"scored {min(start + len(batch), len(records))}/{len(records)}", flush=True)
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", type=Path,
        default=Path("data/raw/OpenMathInstruct-2-train-00000-of-00032.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/calibration/openmath_v2"))
    parser.add_argument("--candidates", type=int, default=1200)
    parser.add_argument("--calibration-cap", type=int, default=320)
    parser.add_argument("--validation-cap", type=int, default=128)
    parser.add_argument("--calibration-fraction", type=float, default=0.65)
    parser.add_argument("--max-problem-characters", type=int, default=600)
    parser.add_argument("--alternatives", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--expected-sha256", default=None,
        help="verify the shard against this checksum; omitted on the first download",
    )
    args = parser.parse_args()
    if args.candidates < 1 or args.alternatives < 1:
        parser.error("candidate and alternative counts must be positive")
    if min(args.calibration_cap, args.validation_cap) < 1:
        parser.error("split caps must be positive")
    if not 0 < args.calibration_fraction < 1:
        parser.error("calibration fraction must be between zero and one")
    if min(args.batch_size, args.max_new_tokens, args.max_problem_characters) < 1:
        parser.error("batch size, token budget, and problem length must be positive")
    return args


def main() -> None:
    args = parse_args()
    expected = args.expected_sha256
    if expected is None and SOURCE_SHA256 != "0" * 64:
        expected = SOURCE_SHA256
    checksum = download_source(args.raw, expected)

    candidates = load_candidates(
        args.raw, args.candidates, args.seed, args.max_problem_characters,
    )
    boundary = round(len(candidates) * args.calibration_fraction)
    records = [
        as_record(candidate, "calibration", ordinal, args.max_new_tokens)
        for ordinal, candidate in enumerate(candidates[:boundary])
    ] + [
        as_record(candidate, "validation", ordinal, args.max_new_tokens)
        for ordinal, candidate in enumerate(candidates[boundary:])
    ]

    model, tokenizer = load_baseline(args.device, local_files_only=args.offline)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    predictions = generate(model, tokenizer, records, args.batch_size, args.max_new_tokens)
    derivations = {row["id"]: row["derivation"] for row in predictions if row["correct"]}
    accepted_ids = set(derivations)

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "candidates.jsonl", records)
    write_jsonl(args.output / "predictions.jsonl", predictions)
    (args.output / "accepted_ids.json").write_text(
        json.dumps(sorted(accepted_ids), indent=2) + "\n"
    )

    caps = {"calibration": args.calibration_cap, "validation": args.validation_cap}
    manifest = {
        "source": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "file": SOURCE_FILE,
        "file_sha256": checksum,
        "license": "CC BY 4.0",
        "model": MODEL_ID,
        "model_revision": REVISION,
        "selection": {
            "problem_sources": list(ORIGINAL_SOURCES),
            "answer_pattern": INTEGER_ANSWER.pattern,
            "max_problem_characters": args.max_problem_characters,
            "deduplicated_by": "normalized problem text",
            "seed": args.seed,
            "candidates": len(records),
        },
        "filter": "alias-aware normalized exact match on the final-answer line of a "
                  "deterministic free-running derivation",
        "generation": {
            "chat_template": True, "enable_thinking": False, "do_sample": False,
            "max_new_tokens": args.max_new_tokens, "variant": "original",
        },
        "score_answer": "the baseline's own accepted derivation, including its final-answer line",
        "caps": caps,
        "condition": "reasoning",
        "family": "openmath",
        "test_split_copied": False,
        "splits": {},
    }
    for split in ("calibration", "validation"):
        selected = [
            row for row in records if row["split"] == split and row["id"] in accepted_ids
        ][: caps[split]]
        alternatives = relation_matched_alternatives(selected, args.alternatives)
        for row in selected:
            row["alternatives"] = alternatives[row["id"]]
            row["score_answer"] = derivations[row["id"]]
        destination = args.output / f"{split}.jsonl"
        write_jsonl(destination, selected)
        attempted = [row for row in predictions if row["split"] == split]
        correct = sum(row["correct"] for row in attempted)
        manifest["splits"][split] = {
            "candidates": len(attempted),
            "baseline_known": correct,
            "acceptance_rate": correct / len(attempted) if attempted else 0.0,
            "kept": len(selected),
            "cap": caps[split],
            "truncated": sum(row["truncated"] for row in attempted),
            "by_problem_source": dict(Counter(row["problem_source"] for row in selected)),
            "mean_derivation_characters": (
                sum(len(row["score_answer"]) for row in selected) / len(selected)
                if selected else 0.0
            ),
            "targets": len({row["target"] for row in selected}),
            "records_with_full_alternative_count": sum(
                len(row["alternatives"]) == args.alternatives for row in selected
            ),
            "output_sha256": sha256(destination),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
