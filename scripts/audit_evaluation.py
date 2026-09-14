#!/usr/bin/env python3
"""Audit strict exact-match changes for answer-containing format effects."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from model_compression.channels import CONDITIONS, contains_accepted_answer, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path,
        default=Path("artifacts/mask_evaluations/initial_validation/predictions.jsonl"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.predictions.with_name("exact_match_audit.json")

    rows = read_jsonl(args.predictions)
    baseline = {row["id"]: row for row in rows if row["mode"] == "baseline"}
    modes = sorted({row["mode"] for row in rows} - {"baseline"})
    report = {
        "note": (
            "Containment is a diagnostic, not the primary metric: long or contradictory "
            "outputs can contain the target and still be wrong."
        ),
        "modes": {},
    }
    for mode in modes:
        report["modes"][mode] = {}
        for condition in CONDITIONS:
            selected = [row for row in rows if row["mode"] == mode and row["condition"] == condition]
            lost = [row for row in selected if baseline[row["id"]]["correct"] and not row["correct"]]
            gained = [row for row in selected if not baseline[row["id"]]["correct"] and row["correct"]]
            contained = [
                row for row in lost
                if contains_accepted_answer(row["prediction"], row["target"], row["aliases"])
            ]
            pairs = Counter((row["target"], row["prediction"]) for row in lost)
            report["modes"][mode][condition] = {
                "correct_to_wrong": len(lost),
                "wrong_to_correct": len(gained),
                "lost_but_contains_accepted_answer": len(contained),
                "lost_without_accepted_answer": len(lost) - len(contained),
                "most_common_lost_pairs": [
                    {"target": target, "prediction": prediction, "count": count}
                    for (target, prediction), count in pairs.most_common(10)
                ],
            }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
