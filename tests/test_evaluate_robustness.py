import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from scripts.evaluate_robustness import (
    GenerationItem,
    answer_token_logps,
    baseline_fingerprint,
    budget_batches,
    length_batches,
    load_baseline_cache,
    prepare_generation,
    save_baseline_cache,
)
from scripts.evaluate_reference_traces import chunked_logps


class AttrDict(dict):
    __getattr__ = dict.__getitem__


class DummyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)

    def forward(self, input_ids, attention_mask, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class DummyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DummyBackbone()
        self.lm_head = torch.nn.Linear(4, 8, bias=False)


class RobustnessOptimizationTests(unittest.TestCase):
    def test_chunked_trace_projection_matches_full_projection_with_padding(self):
        torch.manual_seed(19)
        model = DummyCausalLM().eval()
        encoded = AttrDict(
            input_ids=torch.tensor([[0, 1, 2, 3, 4], [0, 0, 5, 6, 7]]),
            attention_mask=torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]]),
        )
        mask = torch.tensor([[False, False, True, True, True], [False, False, False, True, True]])
        expected = answer_token_logps(model, encoded, mask)
        for chunk_size in (1, 2, 128):
            actual = chunked_logps(model, encoded, mask, chunk_size)
            torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected))

    def test_length_batches_sort_by_length_without_losing_items(self):
        items = [
            GenerationItem(i, {}, str(i), length, 32)
            for i, length in enumerate([9, 2, 7, 3])
        ]
        batches = length_batches(items, 2)
        self.assertEqual([[x.token_length for x in batch] for batch in batches], [[2, 3], [7, 9]])
        self.assertEqual(sorted(x.index for batch in batches for x in batch), [0, 1, 2, 3])

    def test_batches_never_mix_generation_budgets(self):
        items = [
            GenerationItem(i, {}, str(i), length, budget)
            for i, (length, budget) in enumerate(
                [(9, 32), (2, 384), (7, 32), (3, 384), (5, 32)]
            )
        ]
        batches = budget_batches(items, 2)
        for budget, batch in batches:
            self.assertEqual({x.max_new_tokens for x in batch}, {budget})
        self.assertEqual(
            sorted(x.index for _, batch in batches for x in batch), [0, 1, 2, 3, 4],
        )

    def test_records_may_override_the_default_generation_budget(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"]

            def __call__(self, texts, **kwargs):
                return {"length": [len(text) for text in texts]}

        rows = [
            {"prompt": "short"},
            {"prompt": "working", "max_new_tokens": 384},
        ]
        items = prepare_generation(Tokenizer(), rows, 32)
        self.assertEqual([item.max_new_tokens for item in items], [32, 384])

    def test_selected_position_scores_match_full_logits(self):
        torch.manual_seed(7)
        model = DummyCausalLM().eval()
        encoded = AttrDict(
            input_ids=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
            attention_mask=torch.ones(2, 4, dtype=torch.long),
        )
        answer_mask = torch.tensor([
            [False, False, True, True],
            [False, False, False, True],
        ])
        actual = answer_token_logps(model, encoded, answer_mask)

        hidden = model.model(**encoded).last_hidden_state[:, :-1]
        logits = model.lm_head(hidden).float()
        token_logps = logits.log_softmax(-1).gather(
            -1, encoded.input_ids[:, 1:].unsqueeze(-1),
        ).squeeze(-1)
        shifted_mask = answer_mask[:, 1:]
        expected = [token_logps[i][shifted_mask[i]].mean().item() for i in range(2)]
        self.assertEqual(actual, expected)

    def test_baseline_cache_is_fingerprinted_and_round_trips(self):
        rows = [{"id": "row-1", "prompt": "Question?", "target": "Answer"}]
        fingerprint = baseline_fingerprint(rows, 32)
        predictions = [{"id": "row-1", "prediction": "Answer"}]
        margins = [{"id": "row-1", "margin": 1.5}]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            save_baseline_cache(path, fingerprint, predictions, margins)
            self.assertEqual(
                load_baseline_cache(path, fingerprint), (predictions, margins),
            )
            with self.assertRaises(ValueError):
                load_baseline_cache(path, baseline_fingerprint(rows, 64))


if __name__ == "__main__":
    unittest.main()
