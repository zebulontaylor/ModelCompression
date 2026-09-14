#!/usr/bin/env python3
"""Measure what fraction of a candidate corpus the frozen baseline already knows.

This is the cheap precondition check that decides whether a source is usable as a
recall corpus at all. Run it on a sample before building a full matched corpus.
"""
from __future__ import annotations
import argparse, ast, json, random
from pathlib import Path
import torch
from model_compression.qwen import load_baseline
from model_compression.channels import is_correct, contains_accepted_answer


def popqa(path, sample, seed):
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:sample]:
        try:
            aliases = [a for a in ast.literal_eval(r["possible_answers"])]
        except Exception:
            aliases = []
        target = r["obj"]
        out.append({
            "prompt": f"Question: {r['question']}\nAnswer with only the answer:\n",
            "target": target,
            "aliases": [a for a in aliases if a != target],
            "relation": r["prop"],
            "s_pop": r.get("s_pop") or 0,
        })
    return out


def counterfact(path, sample, seed):
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:sample]:
        rw = r["requested_rewrite"]
        cloze = rw["prompt"].replace("{}", rw["subject"])
        out.append({
            "prompt": f"Complete the sentence with only the missing words.\n\n{cloze}",
            "target": rw["target_true"]["str"],
            "aliases": [],
            "relation": rw["relation_id"],
            "s_pop": 0,
        })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["popqa", "counterfact"], required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--sample", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args()

    records = (popqa if a.source == "popqa" else counterfact)(a.raw, a.sample, a.seed)
    model, tok = load_baseline("auto", local_files_only=a.offline)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    results = []
    for s in range(0, len(records), a.batch_size):
        batch = records[s : s + a.batch_size]
        prompts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                   tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in batch]
        inp = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        texts = tok.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)
        for r, pred in zip(batch, texts, strict=True):
            pred = pred.strip()
            results.append({**r, "prediction": pred,
                            "correct": is_correct(pred, r["target"], r["aliases"]),
                            "contained": contains_accepted_answer(pred, r["target"], r["aliases"])})
        print(f"  {min(s + len(batch), len(records))}/{len(records)}", flush=True)

    n = len(results)
    exact = sum(r["correct"] for r in results)
    cont = sum(r["contained"] for r in results)
    print(f"\n=== {a.source}: exact {exact}/{n} ({exact/n:.1%})  containment {cont}/{n} ({cont/n:.1%})")
    by = {}
    for r in results:
        by.setdefault(r["relation"], [0, 0])
        by[r["relation"]][0] += 1
        by[r["relation"]][1] += r["correct"]
    print("by relation:")
    for k, (t, c) in sorted(by.items(), key=lambda x: -x[1][0])[:15]:
        print(f"  {c:4d}/{t:4d}  {c/t:6.1%}  {k}")
    if a.source == "popqa":
        tiers = [(0, 100), (100, 1000), (1000, 10000), (10000, 10**9)]
        print("by subject popularity:")
        for lo, hi in tiers:
            sel = [r for r in results if lo <= r["s_pop"] < hi]
            if sel:
                c = sum(r["correct"] for r in sel)
                print(f"  s_pop [{lo},{hi}): {c}/{len(sel)} ({c/len(sel):.1%})")
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
