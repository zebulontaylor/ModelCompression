"""Auxiliary MLP channel gates and experiment artifact helpers."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import torch


CONDITIONS = ("recall", "extraction", "reasoning")


def normalize_answer(text: str) -> str:
    """Normalize for alias-aware exact match without fuzzy matching."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(char for char in text if not unicodedata.category(char).startswith("P"))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def is_correct(prediction: str, target: str, aliases: list[str]) -> bool:
    normalized = normalize_answer(prediction)
    return normalized in {normalize_answer(answer) for answer in [target, *aliases]}


def contains_accepted_answer(prediction: str, target: str, aliases: list[str]) -> bool:
    """Diagnostic only: whether an accepted answer occurs on word boundaries."""
    normalized = normalize_answer(prediction)
    accepted = {normalize_answer(answer) for answer in [target, *aliases]}
    return any(
        answer and re.search(rf"(?<!\w){re.escape(answer)}(?!\w)", normalized)
        for answer in accepted
    )


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def calibration_half(group_id: str) -> int:
    """Return a stable group-level half assignment independent of row order."""
    return hashlib.sha256(group_id.encode()).digest()[0] & 1


def target_balance_weights(records: list[dict]) -> dict[str, tuple[float, float]]:
    """Give every normalized target equal mass within a condition and half.

    Returns ``record id -> (overall weight, calibration-half weight)``.  The
    absolute scale is irrelevant because the accumulator divides by total
    weight; inverse frequency prevents common answers from dominating.
    """
    overall = Counter((row["condition"], normalize_answer(row["target"])) for row in records)
    by_half = Counter(
        (row["condition"], calibration_half(row["group_id"]), normalize_answer(row["target"]))
        for row in records
    )
    return {
        row["id"]: (
            1.0 / overall[(row["condition"], normalize_answer(row["target"]))],
            1.0 / by_half[(
                row["condition"], calibration_half(row["group_id"]),
                normalize_answer(row["target"]),
            )],
        )
        for row in records
    }


def relation_matched_alternatives(
    records: list[dict], count: int = 3,
) -> dict[str, list[str]]:
    """Choose deterministic distinct distractors from the same relation."""
    pools: dict[str, set[str]] = defaultdict(set)
    for row in records:
        if "relation" not in row:
            raise ValueError(f"record lacks relation metadata: {row['id']}")
        pools[row["relation"]].add(row["target"])
    result = {}
    for row in records:
        accepted = {normalize_answer(x) for x in [row["target"], *row.get("aliases", [])]}
        candidates = [x for x in pools[row["relation"]] if normalize_answer(x) not in accepted]
        candidates.sort(key=lambda x: hashlib.sha256(f"{row['id']}\0{x}".encode()).digest())
        result[row["id"]] = candidates[:count]
    return result


def answer_token_batch(tokenizer, prompt: str, target: str, device, completion_prefix: str = "") -> dict[str, torch.Tensor]:
    """Tokenize a chat prompt and supervise only tokens overlapping the answer.

    Offset mappings avoid assuming that separately tokenized prompt IDs are an
    exact prefix: some tokenizers merge text across the prompt/answer boundary.
    """
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_text += completion_prefix
    full_text = prompt_text + target
    encoded = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")[0]
    labels = encoded["input_ids"].clone()
    supervised = offsets[:, 1] > len(prompt_text)
    labels[0, ~supervised] = -100
    if not supervised.any():
        raise ValueError("target produced no supervised tokens")
    encoded["labels"] = labels
    return {name: tensor.to(device) for name, tensor in encoded.items()}


class ChannelGates:
    """Instrument every Qwen MLP at the input to ``down_proj``.

    Gates are FP32 leaf tensors even when activations are BF16, improving the
    precision of accumulated gate gradients without changing model weights.
    """

    def __init__(self, model, requires_grad: bool = True):
        layers = model.model.layers
        device = next(model.parameters()).device
        widths = [layer.mlp.down_proj.weight.shape[1] for layer in layers]
        if len(set(widths)) != 1:
            raise ValueError("the scoring implementation currently requires a uniform MLP width")
        self.layer_count = len(layers)
        self.width = widths[0]
        self.values = torch.ones(
            self.layer_count,
            self.width,
            device=device,
            dtype=torch.float32,
            requires_grad=requires_grad,
        )
        self._handles = []
        for layer_index, layer in enumerate(layers):
            def apply_gate(module, inputs, index=layer_index):
                gate = self.values[index].to(dtype=inputs[0].dtype)
                return (inputs[0] * gate, *inputs[1:])

            self._handles.append(layer.mlp.down_proj.register_forward_pre_hook(apply_gate))

    def zero_grad(self) -> None:
        self.values.grad = None

    def reset(self) -> None:
        with torch.no_grad():
            self.values.fill_(1)

    def mask(self, channels: Iterable[tuple[int, int]]) -> None:
        self.reset()
        with torch.no_grad():
            for layer, channel in channels:
                if not 0 <= layer < self.layer_count:
                    raise ValueError(f"layer {layer} is outside [0, {self.layer_count})")
                if not 0 <= channel < self.width:
                    raise ValueError(f"channel {channel} is outside [0, {self.width})")
                self.values[layer, channel] = 0

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove()


class ChannelStatistics:
    """Online per-condition gate-gradient moments on the accelerator."""

    def __init__(self, shape: tuple[int, int], device, state: dict | None = None):
        self.shape = shape
        self.device = torch.device(device)
        self.counts: Counter[str] = Counter()
        self.weight_sums: Counter[str] = Counter()
        self.weight_sq_sums: Counter[str] = Counter()
        self.moments: dict[str, dict[str, torch.Tensor]] = {}
        if state is not None:
            self.load_state(state)

    def _bucket(self, key: str) -> dict[str, torch.Tensor]:
        if key not in self.moments:
            self.moments[key] = {
                name: torch.zeros(self.shape, device=self.device, dtype=torch.float32)
                for name in ("signed_sum", "signed_sq_sum", "abs_sum", "abs_sq_sum")
            }
        return self.moments[key]

    def add(
        self, condition: str, half: int, gradient: torch.Tensor,
        weight: float = 1.0, half_weight: float | None = None,
    ) -> None:
        gradient = gradient.detach().float()
        absolute = gradient.abs()
        half_weight = weight if half_weight is None else half_weight
        for key, item_weight in (
            (condition, weight), (f"{condition}_half{half}", half_weight),
        ):
            bucket = self._bucket(key)
            bucket["signed_sum"].add_(gradient, alpha=item_weight)
            bucket["signed_sq_sum"].addcmul_(gradient, gradient, value=item_weight)
            bucket["abs_sum"].add_(absolute, alpha=item_weight)
            bucket["abs_sq_sum"].addcmul_(absolute, absolute, value=item_weight)
            self.counts[key] += 1
            self.weight_sums[key] += item_weight
            self.weight_sq_sums[key] += item_weight * item_weight

    def state_dict(self) -> dict:
        return {
            "shape": self.shape,
            "counts": dict(self.counts),
            "weight_sums": dict(self.weight_sums),
            "weight_sq_sums": dict(self.weight_sq_sums),
            "moments": {
                key: {name: value.cpu() for name, value in bucket.items()}
                for key, bucket in self.moments.items()
            },
        }

    def load_state(self, state: dict) -> None:
        if tuple(state["shape"]) != self.shape:
            raise ValueError(f"checkpoint shape {state['shape']} does not match {self.shape}")
        self.counts.update(state["counts"])
        self.weight_sums.update(state.get("weight_sums", state["counts"]))
        self.weight_sq_sums.update(state.get("weight_sq_sums", state["counts"]))
        self.moments = {
            key: {name: value.to(self.device) for name, value in bucket.items()}
            for key, bucket in state["moments"].items()
        }

    def summary(self) -> dict:
        result = {
            "shape": self.shape,
            "counts": dict(self.counts),
            "weight_sums": dict(self.weight_sums),
            "effective_sample_sizes": {},
            "statistics": {},
        }
        for key, bucket in self.moments.items():
            weight_sum = self.weight_sums[key]
            effective_n = weight_sum * weight_sum / self.weight_sq_sums[key]
            signed_mean = bucket["signed_sum"] / weight_sum
            abs_mean = bucket["abs_sum"] / weight_sum
            signed_var = (bucket["signed_sq_sum"] / weight_sum - signed_mean.square()).clamp_min(0)
            abs_var = (bucket["abs_sq_sum"] / weight_sum - abs_mean.square()).clamp_min(0)
            result["effective_sample_sizes"][key] = effective_n
            result["statistics"][key] = {
                "signed_mean": signed_mean.cpu(),
                "abs_mean": abs_mean.cpu(),
                "signed_stderr": (signed_var / effective_n).sqrt().cpu(),
                "abs_stderr": (abs_var / effective_n).sqrt().cpu(),
            }
        return result

    def results(self) -> dict:
        result = self.summary()
        stats = result["statistics"]
        result["recall_importance"] = -stats["recall"]["signed_mean"]
        result["extraction_sensitivity"] = stats["extraction"]["abs_mean"]
        result["reasoning_sensitivity"] = stats["reasoning"]["abs_mean"]
        return result


def load_mask(path: Path) -> list[tuple[int, int]]:
    payload = json.loads(path.read_text())
    channels = [(int(row["layer"]), int(row["channel"])) for row in payload["channels"]]
    if len(channels) != len(set(channels)):
        raise ValueError(f"mask contains duplicate channels: {path}")
    return channels
