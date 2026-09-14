import unittest
from types import SimpleNamespace

import torch

from model_compression.compensation import MeanBiasCompensation
from scripts.run_iterative_deletion import acceptance
from scripts.evaluate_robustness import scored_answer
from scripts.select_iterative_batch import select_channels
from scripts.test_selection_mechanism import synthetic_records
from scripts.summarize_selection_mechanism import binary_comparison


class MechanismTests(unittest.TestCase):
    def test_sparse_paired_changes_do_not_imply_a_decisive_binary_result(self):
        result = binary_comparison([1] * 4 + [0] * 28)
        self.assertEqual(result["paired_exact_two_sided_p"], .125)
        self.assertEqual(binary_comparison([0] * 32)["paired_exact_two_sided_p"], 1.)

    def test_final_answer_grading_ignores_numbers_in_the_derivation(self):
        row = {"answer_format": "final_answer"}
        self.assertEqual(scored_answer("The intermediate number is 42.\nFinal answer: **17**", row), "17")
        self.assertEqual(scored_answer("The result is 42 but I did not finish.", row), "")
        self.assertEqual(scored_answer("Final answer: <answer>Mira</answer>", row), "Mira")
        self.assertEqual(scored_answer("### Final answer:\nFinal answer: soft", row), "soft")
        self.assertEqual(scored_answer("### Final answer:\nFinal answer: <answer>", row), "")

    def test_compensation_restores_exact_mean_contribution_and_removes_hooks(self):
        projection = torch.nn.Linear(4, 3, bias=False)
        model = SimpleNamespace(model=SimpleNamespace(layers=[
            SimpleNamespace(mlp=SimpleNamespace(down_proj=projection))]))
        means = torch.tensor([[2., -1., 3., 4.]])
        original_weights = projection.weight.detach().clone()
        masked = means.clone()
        masked[:, [1, 3]] = 0
        expected = projection(means)
        uncorrected = projection(masked)
        with MeanBiasCompensation(model, [(0, 1), (0, 3)], means):
            torch.testing.assert_close(projection(masked), expected)
        torch.testing.assert_close(projection(masked), uncorrected)
        torch.testing.assert_close(projection.weight, original_weights)

    def test_additional_family_protects_a_channel_the_old_tasks_miss(self):
        ce = torch.tensor([[1., 2., 3., 4.]])
        sensitivity = torch.tensor([[.01, .02, .03, .04]])
        family = torch.tensor([[10., .01, .02, .03]])
        selected, _ = select_channels(
            ce, ce, sensitivity, sensitivity, torch.ones_like(ce, dtype=torch.bool),
            [], 1, .75, 0., 1., {"arithmetic": family})
        self.assertEqual(selected, [(0, 1)])

    def test_family_margin_rejects_despite_unchanged_accuracy(self):
        metrics = {c: {"exact_accuracy": 1., "answer_containment": 1.}
                   for c in ("recall", "extraction", "reasoning")}
        baseline = {**metrics, "by_family": {"arithmetic": {"exact_accuracy": 1.}}}
        candidate = {**metrics, "by_family": {"arithmetic": {"exact_accuracy": 1., "margin_drop": .3}}}
        report = {"metrics": {"baseline": baseline, "candidate": candidate}}
        accepted, result = acceptance(report, "candidate", {"protected_family_margin_drop": .1})
        self.assertFalse(accepted)
        self.assertEqual(result["family_drops"]["protected_family_margin_drop"]["arithmetic"], .3)
        del candidate["by_family"]["arithmetic"]["margin_drop"]
        with self.assertRaisesRegex(ValueError, "enable margin"):
            acceptance(report, "candidate", {"protected_family_margin_drop": .1})

    def test_synthetic_splits_are_reproducible_and_distinct(self):
        cal = synthetic_records("calibration", 32)
        val = synthetic_records("validation", 64)
        self.assertEqual(cal, synthetic_records("calibration", 32))
        self.assertFalse({r["prompt"] for r in cal} & {r["prompt"] for r in val})
        for row in cal + val:
            self.assertNotIn(row["target"], row["alternatives"])
            self.assertGreaterEqual(len(row["alternatives"]), 2)
            self.assertEqual(len(row["alternatives"]), len(set(row["alternatives"])))


if __name__ == "__main__":
    unittest.main()
