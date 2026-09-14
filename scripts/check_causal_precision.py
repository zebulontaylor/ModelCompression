#!/usr/bin/env python3
"""Repeat the pilot's finite-ablation diagnostic in FP32 on the same examples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model_compression.channels import ChannelGates
from model_compression.qwen import load_baseline
from scripts.score_channels_v2 import batched_answer_losses
import scripts.test_selection_mechanism as pilot


def selected_position_gradient(model, tokenizer, gates, prompt, answer):
    # Project only supervised answer positions. Full prompt logits in FP32 would
    # waste enough memory to make an 8 GiB card unnecessarily difficult to use.
    gates.zero_grad()
    batched_answer_losses(model, tokenizer, prompt, [answer]).mean().backward()
    gradient = gates.values.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError("invalid FP32 gradient")
    return gradient.detach().clone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path("artifacts/selection_mechanism_pilot"))
    args = parser.parse_args()
    root = args.output
    output = root / "causal_fp32"
    if (output / "causal.json").exists():
        print("FP32 causal control already complete")
        return
    config = json.loads((root / "configuration.json").read_text())
    scores = torch.load(config["scores"], map_location="cpu", weights_only=False)
    rows = pilot.sample_mquake(Path(config["validation"]), config["groups"])
    torch.backends.cuda.matmul.allow_tf32 = False
    model, tokenizer = load_baseline(device="cpu", local_files_only=True)
    model.to("cuda")
    tokenizer.padding_side = "left"
    pilot.loss_gradient = selected_position_gradient
    with ChannelGates(model, requires_grad=False) as gates:
        pilot.causal_check(model, tokenizer, gates, scores, rows, output, batch_size=4)
    print(f"FP32 control complete: {output}", flush=True)


if __name__ == "__main__":
    main()
