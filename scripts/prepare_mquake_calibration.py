#!/usr/bin/env python3
"""Build matched records from the corrected MQuAKE-Remastered CF3k split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_URL = (
    "https://huggingface.co/datasets/henryzhongsc/MQuAKE-Remastered/resolve/"
    "b54712d4b464d7e2d4edccd4022f95ddbcb719e7/"
    "data/CF3k-00000-of-00001.parquet"
)
SOURCE_SHA256 = "12c8bc1fe1a5e6b9edc15e2348898a92b4c6b50f6a0693793239af9c0cf9eee0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_source(path: Path, source_url: str, source_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and sha256(path) == source_sha256:
        return
    partial = path.with_suffix(path.suffix + ".partial")
    urllib.request.urlretrieve(source_url, partial)
    actual = sha256(partial)
    if actual != source_sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"source checksum mismatch: expected {source_sha256}, got {actual}")
    partial.replace(path)


def fact_block(triples: list[list[str]]) -> str:
    facts = "\n".join(f"- {subject} — {relation} — {obj}" for subject, relation, obj in triples)
    return f"Facts:\n{facts}\n\n"


def prompt(question: str, triples: list[list[str]] | None = None) -> str:
    prefix = fact_block(triples) if triples else ""
    return f"{prefix}Question: {question}\nAnswer with only the answer:\n"


def root_entity(record: dict) -> str:
    return record["orig_triples"][0][0]


def split_records(records: list[dict], seed: int, calibration_cases: int, validation_cases: int):
    """Keep repeated root subjects in one split to prevent the easiest entity leakage."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[root_entity(record)].append(record)
    ordered = list(groups.values())
    random.Random(seed).shuffle(ordered)
    result = {"calibration": [], "validation": [], "test": []}
    for group in ordered:
        if len(result["calibration"]) < calibration_cases:
            split = "calibration"
        elif len(result["validation"]) < validation_cases:
            split = "validation"
        else:
            split = "test"
        result[split].extend(group)
    return result


def make_examples(record: dict, split: str) -> list[dict]:
    case_id = record["case_id"]
    original_hop = record["single_hops"][-1]
    new_hop = record["new_single_hops"][-1]
    new_triples = record["new_triples_labeled"]
    common = {
        "group_id": f"mquake_remastered_cf3k:{case_id}",
        "source": "henryzhongsc/MQuAKE-Remastered:CF3k",
        "source_case_id": case_id,
        "split": split,
        "hop_count": len(new_triples),
        "relation": record["orig_triples_labeled"][-1][1],
    }
    return [
        {
            **common,
            "id": f"mquake_remastered_cf3k:{case_id}:recall",
            "condition": "recall",
            "prompt": prompt(original_hop["question"]),
            "target": original_hop["answer"],
            "aliases": original_hop.get("answer_alias", []),
            "facts": [],
            "question_variants": [original_hop["question"]],
        },
        {
            **common,
            "id": f"mquake_remastered_cf3k:{case_id}:extraction",
            "condition": "extraction",
            "prompt": prompt(new_hop["question"], [new_triples[-1]]),
            "target": new_hop["answer"],
            "aliases": new_hop.get("answer_alias", []),
            "facts": [new_triples[-1]],
            "relation": new_triples[-1][1],
            "question_variants": [new_hop["question"]],
        },
        {
            **common,
            "id": f"mquake_remastered_cf3k:{case_id}:reasoning",
            "condition": "reasoning",
            "prompt": prompt(record["questions"][0], new_triples),
            "target": record["new_answer"],
            "aliases": record.get("new_answer_alias", []),
            "facts": new_triples,
            "relation": new_triples[-1][1],
            "question_variants": list(dict.fromkeys(record["questions"])),
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", type=Path, default=Path("data/raw/MQuAKE-Remastered-CF3k.parquet")
    )
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--source-sha256", default=SOURCE_SHA256)
    parser.add_argument(
        "--output", type=Path, default=Path("data/calibration/mquake_remastered_cf3k")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-cases", type=int, default=1024)
    parser.add_argument("--validation-cases", type=int, default=512)
    args = parser.parse_args()
    if min(args.calibration_cases, args.validation_cases) < 1:
        parser.error("split sizes must be positive")

    download_source(args.raw, args.source_url, args.source_sha256)
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit("pyarrow is required; run with `uv run --extra data ...`") from error
    records = parquet.read_table(args.raw).to_pylist()
    if args.calibration_cases + args.validation_cases >= len(records):
        parser.error("calibration and validation sizes must leave a non-empty test split")

    splits = split_records(records, args.seed, args.calibration_cases, args.validation_cases)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_url": args.source_url,
        "source_sha256": args.source_sha256,
        "seed": args.seed,
        "split_policy": "deterministic shuffle, grouped by the first-hop subject Wikidata ID",
        "note": "Candidates must still be filtered to facts the frozen baseline answers correctly.",
        "splits": {},
    }
    for split, split_cases in splits.items():
        examples = [example for record in split_cases for example in make_examples(record, split)]
        destination = args.output / f"{split}.jsonl"
        with destination.open("w") as stream:
            for example in examples:
                stream.write(json.dumps(example, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {
            "cases": len(split_cases),
            "examples": len(examples),
            "conditions": dict(Counter(x["condition"] for x in examples)),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
