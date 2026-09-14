#!/usr/bin/env python3
"""Build matched records from the 2WikiMultiHopQA chain-shaped cases.

2WikiMultiHopQA supplies explicit (subject, relation, object) evidence chains
alongside each composed question. Only the `compositional` and `inference`
types are chain-shaped; `comparison` and `bridge_comparison` compare two
independent entities and have no final triple to counterfactualize, so they are
dropped rather than pooled into the matched contrast.

As in the MQuAKE build, the recall condition asks the true final-hop fact with
no context, while extraction and reasoning supply a counterfactual replacement
for that fact. The counterfactual keeps the protected conditions from being
solvable by repeating stored real-world knowledge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_REVISION = "16852fde9d85cba158cf7e6517e7a3f9415a28c0"
SOURCE_URL = (
    "https://huggingface.co/datasets/voidful/2WikiMultihopQA/resolve/"
    f"{SOURCE_REVISION}/train.json"
)
SOURCE_SHA256 = "b3fddb4d5bb42cd797919cad67616545be51b24740e0a7dabdae7bf76b8f7bfa"

CHAIN_TYPES = ("compositional", "inference")

# Single-hop question templates for the final-hop relation. A relation without a
# template is dropped: the recall condition needs an unambiguous direct question.
TEMPLATES = {
    "director": "Who is the director of {subject}?",
    "date of birth": "When was {subject} born?",
    "father": "Who is the father of {subject}?",
    "date of death": "When did {subject} die?",
    "publication date": "When was {subject} published?",
    "country of citizenship": "Which country is {subject} a citizen of?",
    "place of birth": "Where was {subject} born?",
    "spouse": "Who is the spouse of {subject}?",
    "mother": "Who is the mother of {subject}?",
    "place of death": "Where did {subject} die?",
    "country of origin": "Which country is {subject} from?",
    "country": "Which country is {subject} in?",
    "performer": "Who is the performer of {subject}?",
    "composer": "Who is the composer of {subject}?",
    "educated at": "Where was {subject} educated?",
    "place of burial": "Where is {subject} buried?",
    "employer": "Who is the employer of {subject}?",
    "inception": "When was {subject} founded?",
    "award received": "Which award did {subject} receive?",
    "child": "Who is the child of {subject}?",
    "sibling": "Who is the sibling of {subject}?",
    "cause of death": "What was the cause of death of {subject}?",
    "founded by": "Who founded {subject}?",
    "producer": "Who is the producer of {subject}?",
    "publisher": "Who is the publisher of {subject}?",
    "occupation": "What is the occupation of {subject}?",
    "creator": "Who is the creator of {subject}?",
    "editor": "Who is the editor of {subject}?",
    "presenter": "Who is the presenter of {subject}?",
    "student of": "Who was {subject} a student of?",
    "place of detention": "Where was {subject} detained?",
    "manufacturer": "Who is the manufacturer of {subject}?",
    "doctoral advisor": "Who is the doctoral advisor of {subject}?",
}


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


def normalize(text: str) -> str:
    """Loose key for deduplication and counterfactual distinctness only."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(char for char in text if not unicodedata.category(char).startswith("P"))
    return " ".join(text.split())


def fact_block(triples: list[list[str]]) -> str:
    facts = "\n".join(f"- {subject} — {relation} — {obj}" for subject, relation, obj in triples)
    return f"Facts:\n{facts}\n\n"


def prompt(question: str, triples: list[list[str]] | None = None) -> str:
    prefix = fact_block(triples) if triples else ""
    return f"{prefix}Question: {question}\nAnswer with only the answer:\n"


def usable(record: dict) -> bool:
    """Keep only connected two-hop chains whose final relation has a template."""
    if record.get("type") not in CHAIN_TYPES:
        return False
    evidences = record.get("evidences") or []
    if len(evidences) != 2 or any(len(triple) != 3 for triple in evidences):
        return False
    if any(not str(field).strip() for triple in evidences for field in triple):
        return False
    if evidences[0][2] != evidences[1][0]:
        return False
    if evidences[-1][2] != record.get("answer"):
        return False
    return evidences[-1][1] in TEMPLATES


def title_aliases(record: dict, answer: str) -> list[str]:
    """2WikiMultiHopQA carries no alias lists; recover a Wikipedia title when it matches."""
    key = normalize(answer)
    aliases = []
    for title, _sentences in record.get("context") or []:
        if normalize(title) == key and title != answer:
            aliases.append(title)
    return sorted(dict.fromkeys(aliases))


def counterfactual_objects(records: list[dict], seed: int) -> dict[str, str]:
    """Deterministic same-relation replacement object for each case's final triple."""
    pool: dict[str, list[str]] = defaultdict(list)
    for record in records:
        relation, obj = record["evidences"][-1][1], record["evidences"][-1][2]
        pool[relation].append(obj)
    for relation in pool:
        pool[relation] = sorted(dict.fromkeys(pool[relation]))

    assigned: dict[str, str] = {}
    for record in records:
        case_id = record["_id"]
        relation, true_object = record["evidences"][-1][1], record["evidences"][-1][2]
        candidates = pool[relation]
        if len(candidates) < 2:
            continue
        # Seed from the case id so the choice does not depend on iteration order.
        digest = hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        excluded = normalize(true_object)
        for _ in range(16):
            choice = rng.choice(candidates)
            if normalize(choice) != excluded:
                assigned[case_id] = choice
                break
    return assigned


def recall_subject(record: dict) -> str:
    return record["evidences"][-1][0]


def split_records(records: list[dict], seed: int, calibration_cases: int, validation_cases: int):
    """Keep every case about one recall subject in a single split."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[normalize(recall_subject(record))].append(record)
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


def make_examples(record: dict, split: str, counterfactual: str) -> list[dict]:
    case_id = record["_id"]
    first_triple, final_triple = record["evidences"]
    subject, relation, true_object = final_triple
    question = TEMPLATES[relation].format(subject=subject)
    new_final = [subject, relation, counterfactual]
    new_triples = [list(first_triple), new_final]
    composed = record["question"]
    common = {
        "group_id": f"2wiki:{case_id}",
        "source": f"voidful/2WikiMultihopQA:train:{record['type']}",
        "source_case_id": case_id,
        "split": split,
        "hop_count": len(record["evidences"]),
        "relation": relation,
    }
    return [
        {
            **common,
            "id": f"2wiki:{case_id}:recall",
            "condition": "recall",
            "prompt": prompt(question),
            "target": true_object,
            "aliases": title_aliases(record, true_object),
            "facts": [],
            "question_variants": [question],
        },
        {
            **common,
            "id": f"2wiki:{case_id}:extraction",
            "condition": "extraction",
            "prompt": prompt(question, [new_final]),
            "target": counterfactual,
            "aliases": [],
            "facts": [new_final],
            "question_variants": [question],
        },
        {
            **common,
            "id": f"2wiki:{case_id}:reasoning",
            "condition": "reasoning",
            "prompt": prompt(composed, new_triples),
            "target": counterfactual,
            "aliases": [],
            "facts": new_triples,
            "question_variants": [composed],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/2wiki-train.json"))
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--source-sha256", default=SOURCE_SHA256)
    parser.add_argument("--output", type=Path, default=Path("data/calibration/2wiki"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-cases", type=int, default=24576)
    parser.add_argument("--validation-cases", type=int, default=6144)
    parser.add_argument(
        "--max-cases", type=int, default=0,
        help="cap the candidate pool after deduplication (0 keeps everything)",
    )
    args = parser.parse_args()
    if min(args.calibration_cases, args.validation_cases) < 1:
        parser.error("split sizes must be positive")

    download_source(args.raw, args.source_url, args.source_sha256)
    with args.raw.open(encoding="utf-8") as stream:
        source = json.load(stream)
    total_source = len(source)

    chain_records = [record for record in source if usable(record)]
    del source

    # One case per distinct final triple, so no recall fact is scored twice.
    seen: set[tuple[str, str, str]] = set()
    deduplicated = []
    for record in chain_records:
        subject, relation, obj = record["evidences"][-1]
        key = (normalize(subject), relation, normalize(obj))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(record)

    assigned = counterfactual_objects(deduplicated, args.seed)
    candidates = [record for record in deduplicated if record["_id"] in assigned]
    candidates.sort(key=lambda record: record["_id"])
    if args.max_cases:
        rng = random.Random(args.seed)
        rng.shuffle(candidates)
        candidates = sorted(candidates[: args.max_cases], key=lambda record: record["_id"])
    if args.calibration_cases + args.validation_cases >= len(candidates):
        parser.error("calibration and validation sizes must leave a non-empty test split")

    splits = split_records(candidates, args.seed, args.calibration_cases, args.validation_cases)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_url": args.source_url,
        "source_sha256": args.source_sha256,
        "source_revision": SOURCE_REVISION,
        "seed": args.seed,
        "kept_types": list(CHAIN_TYPES),
        "split_policy": "deterministic shuffle, grouped by the normalized recall subject",
        "counterfactual_policy": (
            "deterministic same-relation replacement object for the final hop, "
            "excluding the true object"
        ),
        "note": "Candidates must still be filtered to facts the frozen baseline answers correctly.",
        "selection": {
            "source_records": total_source,
            "chain_shaped": len(chain_records),
            "after_final_triple_deduplication": len(deduplicated),
            "with_counterfactual": len(candidates),
        },
        "splits": {},
    }
    for split, split_cases in splits.items():
        examples = [
            example
            for record in split_cases
            for example in make_examples(record, split, assigned[record["_id"]])
        ]
        destination = args.output / f"{split}.jsonl"
        with destination.open("w") as stream:
            for example in examples:
                stream.write(json.dumps(example, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {
            "cases": len(split_cases),
            "examples": len(examples),
            "conditions": dict(Counter(x["condition"] for x in examples)),
            "relations": len({x["relation"] for x in examples}),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
