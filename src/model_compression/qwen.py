"""Load a compatible frozen causal LM and run local generation.

The historical experiment default remains the pinned Qwen3-1.7B checkpoint.
Callers may select another Hugging Face model as long as it exposes the common
``model.layers[*].mlp.down_proj`` structure used by the channel gates.
"""

import argparse
import json

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

MODEL_ID = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

MODEL_PRESETS = {
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", None),
    "qwen3-1.7b": (MODEL_ID, REVISION),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", None),
}


def resolve_model(model: str | None = None, revision: str | None = None) -> tuple[str, str | None]:
    """Resolve a friendly preset or pass through a Hugging Face repository ID."""
    model = model or "qwen3-1.7b"
    preset_id, preset_revision = MODEL_PRESETS.get(model, (model, None))
    return preset_id, revision or preset_revision


def load_baseline(
    device="auto", local_files_only=False, attn_implementation="sdpa",
    model_id: str | None = None, revision: str | None = None,
):
    """Return (model, tokenizer); weights are frozen, autograd remains available.

    Use trainable auxiliary gates for gradient measurements. Frozen parameters
    alone do not create an autograd graph, so attach gates before scoring.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    resolved_id, resolved_revision = resolve_model(model_id, revision)
    options = dict(local_files_only=local_files_only)
    if resolved_revision is not None:
        options["revision"] = resolved_revision
    tokenizer = AutoTokenizer.from_pretrained(resolved_id, **options)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_id, dtype=dtype, device_map=device,
        attn_implementation=attn_implementation, **options,
    )
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="What is 2 + 2? Answer briefly.")
    parser.add_argument(
        "--model", default="qwen3-1.7b",
        help="model preset or Hugging Face repository ID",
    )
    parser.add_argument("--revision", help="optional Hugging Face revision override")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    model_id, revision = resolve_model(args.model, args.revision)
    if args.download_only:
        options = dict(local_files_only=args.offline)
        if revision is not None:
            options["revision"] = revision
        print(snapshot_download(model_id, **options))
        return
    set_seed(args.seed)
    model, tokenizer = load_baseline(
        args.device, args.offline, model_id=model_id, revision=revision,
    )
    print(json.dumps({
        "model": model_id, "revision": revision,
        "device": str(model.device), "dtype": str(model.dtype),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "layers": model.config.num_hidden_layers,
        "intermediate_size": model.config.intermediate_size,
    }, indent=2))
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=args.thinking,
        )
    except TypeError:
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}], tokenize=False,
                add_generation_prompt=True,
            )
        except (AttributeError, ValueError):
            text = args.prompt
    except (AttributeError, ValueError):
        text = args.prompt
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
            temperature=0.6 if args.thinking else 0.7,
            top_p=0.95 if args.thinking else 0.8, top_k=20,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
