#!/usr/bin/env python3
"""Concatenate matched-condition corpora into one calibration/validation corpus.

Record identifiers and group identifiers must already be disjoint across the
inputs; the first corpus supplies the matched MQuAKE conditions and the rest
add protected families.  The sealed test split is never read or written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from model_compression.channels import CONDITIONS, read_jsonl, write_jsonl


SPLITS = ("calibration", "validation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, action="append", default=[],
                        help="directory holding calibration.jsonl and validation.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.corpus) < 2:
        parser.error("merging needs at least two corpora")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "sources": [str(path) for path in args.corpus],
        "test_split_copied": False,
        "splits": {},
    }
    for split in SPLITS:
        merged: list[dict] = []
        seen_ids: set[str] = set()
        seen_groups: dict[str, str] = {}
        per_corpus = {}
        for corpus in args.corpus:
            records = read_jsonl(corpus / f"{split}.jsonl")
            for row in records:
                if row.get("split") != split:
                    raise ValueError(f"record {row['id']} is not from the {split} split")
                if row["id"] in seen_ids:
                    raise ValueError(f"duplicate record id across corpora: {row['id']}")
                if seen_groups.get(row["group_id"], str(corpus)) != str(corpus):
                    raise ValueError(f"group id spans corpora: {row['group_id']}")
                seen_ids.add(row["id"])
                seen_groups[row["group_id"]] = str(corpus)
            per_corpus[str(corpus)] = len(records)
            merged.extend(records)
        present = {row["condition"] for row in merged}
        if present != set(CONDITIONS):
            raise ValueError(f"merged {split} split lacks the matched conditions: {sorted(present)}")
        destination = args.output / f"{split}.jsonl"
        write_jsonl(destination, merged)
        manifest["splits"][split] = {
            "records": len(merged),
            "groups": len({row["group_id"] for row in merged}),
            "by_corpus": per_corpus,
            "by_condition": dict(Counter(row["condition"] for row in merged)),
            "by_family": dict(Counter(row.get("family", row["condition"]) for row in merged)),
            "output_sha256": sha256(destination),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
