#!/usr/bin/env python3
"""Enrich the frozen accepted calibration/validation records for v2 screening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_compression.channels import read_jsonl, relation_matched_alternatives, write_jsonl


def fact_block(triples: list[list[str]]) -> str:
    body = "\n".join(f"- {subject} — {relation} — {obj}" for subject, relation, obj in triples)
    return f"Facts:\n{body}\n\n" if triples else ""


def formatted_prompts(row: dict, questions: list[str]) -> list[dict[str, str]]:
    facts = fact_block(row["facts"])
    variants = []
    for index, question in enumerate(questions):
        label = "original" if index == 0 else f"source_paraphrase_{index}"
        variants.append({
            "label": label,
            "prompt": f"{facts}Question: {question}\nAnswer with only the answer:\n",
        })
    variants.append({
        "label": "instruction_first",
        "prompt": f"Give only the answer.\n\n{facts}Question: {questions[0]}\n",
    })
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/MQuAKE-Remastered-CF3k.parquet"))
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/calibration/mquake_remastered_cf3k_baseline_known"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/calibration/mquake_remastered_cf3k_v2"),
    )
    parser.add_argument("--alternatives", type=int, default=3)
    parser.add_argument(
        "--use-record-metadata", action="store_true",
        help="take relation and question variants from the records themselves instead of --raw; "
             "required for sources without an MQuAKE-shaped parquet",
    )
    args = parser.parse_args()
    if args.alternatives < 1:
        parser.error("--alternatives must be positive")

    source: dict = {}
    if not args.use_record_metadata:
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise SystemExit("pyarrow is required; run with `uv run --extra data ...`") from error
        source = {row["case_id"]: row for row in parquet.read_table(args.raw).to_pylist()}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(args.input),
        "raw": None if args.use_record_metadata else str(args.raw),
        "metadata_source": "records" if args.use_record_metadata else "raw",
        "test_split_copied": False,
        "alternative_policy": "deterministic same-relation targets, excluding accepted answers",
        "splits": {},
    }
    for split in ("calibration", "validation"):
        records = read_jsonl(args.input / f"{split}.jsonl")
        enriched = []
        for row in records:
            if args.use_record_metadata:
                enriched.append({
                    **row,
                    "relation": row["relation"],
                    "question_variants": list(dict.fromkeys(row["question_variants"])),
                })
                continue
            raw = source[row["source_case_id"]]
            if row["condition"] == "recall":
                relation = raw["orig_triples_labeled"][-1][1]
                questions = [raw["single_hops"][-1]["question"]]
            elif row["condition"] == "extraction":
                relation = raw["new_triples_labeled"][-1][1]
                questions = [raw["new_single_hops"][-1]["question"]]
            else:
                relation = raw["new_triples_labeled"][-1][1]
                questions = list(dict.fromkeys(raw["questions"]))
            enriched.append({**row, "relation": relation, "question_variants": questions})
        alternatives = relation_matched_alternatives(enriched, args.alternatives)
        for row in enriched:
            row["alternatives"] = alternatives[row["id"]]
            row["prompt_variants"] = formatted_prompts(row, row["question_variants"])
        write_jsonl(args.output / f"{split}.jsonl", enriched)
        manifest["splits"][split] = {
            "records": len(enriched),
            "groups": len({row["group_id"] for row in enriched}),
            "relations": len({row["relation"] for row in enriched}),
            "targets": len({row["target"] for row in enriched}),
            "records_with_full_alternative_count": sum(
                len(row["alternatives"]) == args.alternatives for row in enriched
            ),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
