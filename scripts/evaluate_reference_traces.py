#!/usr/bin/env python3
"""Post-hoc likelihood diagnostic on fixed, baseline-correct reasoning traces.

This complements free generation; it cannot establish reasoning retention alone.
It does not change masks or the pilot's predeclared acceptance decisions.
"""
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from model_compression.channels import ChannelGates, load_mask, read_jsonl, is_correct
from model_compression.compensation import MeanBiasCompensation
from model_compression.qwen import load_baseline, MODEL_ID, REVISION
from scripts.evaluate_robustness import scored_answer, prepare_margins, length_batches
from scripts.test_selection_mechanism import save


def chunked_logps(model, encoded, answer_mask, chunk_size=128):
    """Project trace-token positions in bounded chunks to avoid full-vocab OOM."""
    mask = answer_mask[:, 1:] & encoded.attention_mask[:, 1:].bool()
    counts = mask.sum(1).tolist()
    if not all(counts):
        raise ValueError("empty reference trace")
    with torch.inference_mode():
        hidden = model.model(**encoded, use_cache=False, return_dict=True).last_hidden_state[:, :-1][mask]
        targets = encoded.input_ids[:, 1:][mask]
        losses = []
        for start in range(0, len(hidden), chunk_size):
            logits = model.lm_head(hidden[start:start + chunk_size]).float()
            losses.append(torch.nn.functional.cross_entropy(logits, targets[start:start + chunk_size], reduction="none"))
        values = torch.cat(losses)
    return [-chunk.mean().item() for chunk in values.split(counts)]


def evaluate(model, tokenizer, rows, batch_size=16):
    items = prepare_margins(tokenizer, rows)
    results = {}
    for batch in length_batches(items, batch_size):
        encoded = tokenizer([x.text for x in batch], return_tensors="pt", padding=True,
                            add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded.pop("offset_mapping")
        answer_mask = torch.stack([offsets[i, :, 1] > len(item.prompt) for i, item in enumerate(batch)]).to(model.device)
        logps = chunked_logps(model, encoded.to(model.device), answer_mask)
        for item, value in zip(batch, logps, strict=True):
            row = rows[item.owner]
            results[row["id"]] = {"family": row["family"], "ce": -value}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path("artifacts/selection_mechanism_pilot"))
    args = parser.parse_args()
    root = args.output
    baseline = json.loads((root / "evaluations" / "baseline.json").read_text())
    predictions = {p["id"]: p for p in baseline["predictions"]}
    rows = []
    for row in read_jsonl(root / "validation.jsonl"):
        if not row["id"].startswith("synthetic:"):
            continue
        text = predictions[row["id"]]["prediction"]
        if is_correct(scored_answer(text, row), row["target"], row["aliases"]):
            rows.append({**row, "target": text, "alternatives": []})
    save(root / "reference_trace_manifest.json", {
        "model": MODEL_ID, "revision": REVISION, "baseline_correct_ids": [r["id"] for r in rows],
        "objective": "mean token CE of fixed, baseline-generated complete solutions",
        "verification": "final answers verified; intermediate steps not independently verified",
        "post_hoc": True, "used_for_mask_selection": False, "batch_size": 16, "projection_chunk_tokens": 128,
    })
    model, tokenizer = load_baseline(local_files_only=True)
    tokenizer.padding_side = "left"
    means = torch.load(root / "activation_means.pt", map_location="cpu", weights_only=False)
    modes = ["baseline", *[p.stem for p in sorted((root / "evaluations").glob("*.json")) if p.stem != "baseline"]]
    all_results = {}
    with ChannelGates(model, requires_grad=False) as gates:
        for mode in modes:
            path = root / "reference_traces" / f"{mode}.json"
            if path.exists():
                result = json.loads(path.read_text())
            else:
                channels = [] if mode == "baseline" else load_mask(root / "masks" / f"{mode.removesuffix('_compensated')}.json")
                gates.mask(channels)
                context = MeanBiasCompensation(model, channels, means) if mode.endswith("_compensated") else nullcontext()
                with context:
                    result = evaluate(model, tokenizer, rows)
                save(path, result)
            all_results[mode] = result
            print(f"reference traces: {mode} ({len(result)} solutions)", flush=True)
    summary = {}
    for mode, result in all_results.items():
        summary[mode] = {}
        for family in sorted({r["family"] for r in rows}):
            ids = [r["id"] for r in rows if r["family"] == family]
            summary[mode][family] = {
                "count": len(ids),
                "ce": sum(result[i]["ce"] for i in ids) / len(ids),
                "ce_increase": sum(result[i]["ce"] - all_results["baseline"][i]["ce"] for i in ids) / len(ids),
            }
    save(root / "reference_trace_metrics.json", summary)


if __name__ == "__main__":
    main()
