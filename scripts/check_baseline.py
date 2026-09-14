"""Check checkpoint structure and gate gradients without changing weights.

Run after download: uv run python scripts/check_baseline.py
"""

import json

import torch

from model_compression.qwen import load_baseline


def main():
    model, tokenizer = load_baseline(local_files_only=True)
    assert not any(p.requires_grad for p in model.parameters())
    assert len(model.model.layers) == 28
    width = model.config.intermediate_size
    hidden = model.config.hidden_size
    for layer in model.model.layers:
        assert layer.mlp.gate_proj.weight.shape == (width, hidden)
        assert layer.mlp.up_proj.weight.shape == (width, hidden)
        assert layer.mlp.down_proj.weight.shape == (hidden, width)

    # Gate one layer at the exact intermediate-channel boundary.
    gate = torch.nn.Parameter(torch.ones(width, device=model.device, dtype=model.dtype))
    def apply_gate(module, inputs):
        return (inputs[0] * gate,)

    handle = model.model.layers[0].mlp.down_proj.register_forward_pre_hook(apply_gate)
    try:
        inputs = tokenizer("The answer to two plus two is four.", return_tensors="pt").to(model.device)
        loss = model(**inputs, labels=inputs.input_ids, use_cache=False).loss
        loss.backward()
        assert gate.grad is not None and torch.isfinite(gate.grad).all()
        assert gate.grad.abs().sum() > 0
        assert all(p.grad is None for p in model.parameters())
        print(json.dumps({
            "status": "passed", "loss": loss.item(),
            "layers": len(model.model.layers), "hidden_size": hidden,
            "intermediate_size": width,
            "gate_gradient_l1": gate.grad.float().abs().sum().item(),
            "peak_cuda_memory_gib": (
                torch.cuda.max_memory_allocated() / 1024**3
                if model.device.type == "cuda" else None
            ),
        }, indent=2))
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
