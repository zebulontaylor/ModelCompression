import unittest

import torch

from scripts.select_iterative_batch import select_channels


class IterativeSelectionTests(unittest.TestCase):
    def test_continuous_protected_cost_beats_high_recall_high_cost_channel(self):
        ce = torch.tensor([[0.1, 10.0, 2.0, 3.0]])
        margin = ce.clone()
        extraction = torch.tensor([[0.01, 0.20, 0.02, 0.03]])
        reasoning = torch.tensor([[0.01, 0.20, 0.02, 0.03]])
        surviving = torch.ones_like(ce, dtype=torch.bool)
        selected, diagnostics = select_channels(
            ce, margin, extraction, reasoning, surviving, [], 1,
            protected_quantile=0.75, recall_weight=0.05, max_layer_fraction=1.0,
        )
        self.assertEqual(selected, [(0, 0)])
        self.assertEqual(diagnostics["layer_counts"], [1])

    def test_per_layer_cap_prevents_concentrated_selection(self):
        ce = torch.zeros((2, 4))
        margin = ce.clone()
        extraction = torch.tensor([[0.01, 0.02, 0.03, 0.04], [0.10, 0.20, 0.30, 0.40]])
        reasoning = extraction.clone()
        surviving = torch.ones_like(ce, dtype=torch.bool)
        selected, _ = select_channels(
            ce, margin, extraction, reasoning, surviving, [], 2,
            protected_quantile=0.99, recall_weight=0.0, max_layer_fraction=0.25,
        )
        self.assertEqual([layer for layer, _ in selected], [0, 1])


if __name__ == "__main__":
    unittest.main()
