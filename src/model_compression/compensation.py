"""Calibration-mean bias compensation for masked MLP channels."""
from __future__ import annotations

import torch


class MeanBiasCompensation:
    """Replace removed channel contributions with a fixed calibration mean.

    For each layer, add W_down[:, deleted] @ mean_activation[deleted].
    Means are measured before down_proj in the unmasked parent. This adds a
    constant residual contribution and never modifies surviving weights.
    """

    def __init__(self, model, channels, mean_activations):
        self.handles = []
        layers = model.model.layers
        by_layer = {}
        for layer, channel in channels:
            by_layer.setdefault(layer, []).append(channel)
        for layer, indices in by_layer.items():
            projection = layers[layer].mlp.down_proj
            means = mean_activations[layer, indices].to(projection.weight.device).float()
            bias = projection.weight[:, indices].float() @ means
            bias = bias.to(projection.weight.dtype).detach()

            def compensate(module, inputs, output, value=bias):
                return output + value

            self.handles.append(projection.register_forward_hook(compensate))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
