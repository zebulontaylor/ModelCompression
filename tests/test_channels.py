import unittest
from types import SimpleNamespace

import torch

from model_compression.channels import (
    ChannelGates,
    ChannelStatistics,
    calibration_half,
    contains_accepted_answer,
    is_correct,
    normalize_answer,
    relation_matched_alternatives,
    target_balance_weights,
)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        layers = []
        for _ in range(2):
            mlp = SimpleNamespace(down_proj=torch.nn.Linear(3, 2, bias=False))
            layers.append(SimpleNamespace(mlp=mlp))
        self.model = SimpleNamespace(layers=layers)


class ChannelTests(unittest.TestCase):
    def test_answer_normalization_remains_alias_aware_exact_match(self):
        self.assertEqual(normalize_answer(" The District. "), "district")
        self.assertTrue(is_correct("D.C.", "Washington, D.C.", ["DC", "The District"]))
        self.assertFalse(is_correct("Washington DC area", "Washington, D.C.", ["DC"]))
        self.assertTrue(contains_accepted_answer("Continent: Europe", "Europe", []))
        self.assertTrue(contains_accepted_answer("Continental Europe", "Europe", []))
        self.assertFalse(contains_accepted_answer("Seattle", "Redmond", []))

    def test_half_assignment_is_stable(self):
        self.assertEqual(calibration_half("group:7"), calibration_half("group:7"))
        self.assertIn(calibration_half("group:7"), (0, 1))

    def test_gates_mask_only_requested_channels_and_remove_hooks(self):
        model = DummyModel()
        layer = model.model.layers[1].mlp.down_proj
        original = layer(torch.ones(1, 3))
        with ChannelGates(model, requires_grad=False) as gates:
            gates.mask([(1, 2)])
            expected = layer(torch.tensor([[1.0, 1.0, 0.0]]))
            actual = layer(torch.ones(1, 3))
            self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(layer(torch.ones(1, 3)), original))

    def test_statistics_use_directional_recall_and_absolute_protection(self):
        stats = ChannelStatistics((1, 2), "cpu")
        stats.add("recall", 0, torch.tensor([[2.0, -3.0]]))
        stats.add("extraction", 0, torch.tensor([[-2.0, 3.0]]))
        stats.add("reasoning", 1, torch.tensor([[4.0, -5.0]]))
        result = stats.results()
        self.assertTrue(torch.equal(result["recall_importance"], torch.tensor([[-2.0, 3.0]])))
        self.assertTrue(torch.equal(result["extraction_sensitivity"], torch.tensor([[2.0, 3.0]])))
        self.assertTrue(torch.equal(result["reasoning_sensitivity"], torch.tensor([[4.0, 5.0]])))

    def test_weighted_statistics_equalize_repeated_targets(self):
        stats = ChannelStatistics((1, 1), "cpu")
        stats.add("recall", 0, torch.tensor([[1.0]]), weight=0.5, half_weight=0.5)
        stats.add("recall", 0, torch.tensor([[3.0]]), weight=0.5, half_weight=0.5)
        stats.add("recall", 0, torch.tensor([[10.0]]), weight=1.0, half_weight=1.0)
        summary = stats.summary()
        self.assertTrue(torch.allclose(
            summary["statistics"]["recall"]["signed_mean"], torch.tensor([[6.0]])
        ))

    def test_target_weights_and_relation_alternatives_are_balanced(self):
        records = [
            {"id": "a", "group_id": "g1", "condition": "recall", "target": "Europe", "aliases": [], "relation": "continent"},
            {"id": "b", "group_id": "g2", "condition": "recall", "target": "Europe", "aliases": [], "relation": "continent"},
            {"id": "c", "group_id": "g3", "condition": "recall", "target": "Asia", "aliases": [], "relation": "continent"},
        ]
        weights = target_balance_weights(records)
        self.assertEqual(weights["a"][0], 0.5)
        self.assertEqual(weights["c"][0], 1.0)
        alternatives = relation_matched_alternatives(records, count=3)
        self.assertEqual(alternatives["a"], ["Asia"])
        self.assertEqual(alternatives["c"], ["Europe"])


if __name__ == "__main__":
    unittest.main()
