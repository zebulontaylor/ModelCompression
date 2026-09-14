import argparse
import unittest
from pathlib import Path

from scripts.run_iterative_deletion import (
    acceptance,
    evaluation_command,
    next_backoff_count,
)


class IterativeDeletionTests(unittest.TestCase):
    def report(self, extraction_drop=0.0, reasoning_drop=0.0):
        baseline = {
            "recall": {"exact_accuracy": 1.0, "answer_containment": 1.0},
            "extraction": {"exact_accuracy": 1.0, "answer_containment": 1.0},
            "reasoning": {"exact_accuracy": 1.0, "answer_containment": 1.0},
        }
        masked = {
            **baseline,
            "extraction": {
                "exact_accuracy": 1.0 - extraction_drop,
                "answer_containment": 1.0,
            },
            "reasoning": {
                "exact_accuracy": 1.0,
                "answer_containment": 1.0 - reasoning_drop,
            },
        }
        return {"metrics": {"baseline": baseline, "candidate": masked}}

    def test_accepts_metrics_inside_all_budgets(self):
        budgets = {
            "extraction_exact_drop": 0.01,
            "extraction_containment_drop": 0.0,
            "reasoning_exact_drop": 0.0,
            "reasoning_containment_drop": 0.02,
        }
        accepted, observed = acceptance(self.report(0.01, 0.02), "candidate", budgets)
        self.assertTrue(accepted)
        self.assertAlmostEqual(observed["drops"]["extraction_exact_drop"], 0.01)

    def test_rejects_when_any_protected_budget_is_exceeded(self):
        budgets = {
            "extraction_exact_drop": 0.0,
            "extraction_containment_drop": 0.0,
            "reasoning_exact_drop": 0.0,
            "reasoning_containment_drop": 0.01,
        }
        accepted, _ = acceptance(self.report(reasoning_drop=0.02), "candidate", budgets)
        self.assertFalse(accepted)

    def test_backoff_halves_until_the_minimum_is_rejected(self):
        self.assertEqual(next_backoff_count(100, 20), 50)
        self.assertEqual(next_backoff_count(30, 20), 20)
        self.assertIsNone(next_backoff_count(20, 20))

    def evaluation_args(self):
        return argparse.Namespace(
            validation_data=Path("data/validation.jsonl"), device="cuda",
            batch_size=8, max_new_tokens=32, offline=True, validation_limit=3,
        )

    def test_screening_skips_margins_and_shares_one_baseline_cache(self):
        cache = Path("/sweep/baseline_cache.json")
        screening = evaluation_command(
            self.evaluation_args(), "/sweep/round_001/mask.json",
            Path("/sweep/round_001/evaluation"), cache, margins=False,
        )
        final = evaluation_command(
            self.evaluation_args(), "/sweep/round_001/mask.json",
            Path("/sweep/final_evaluation"), cache, margins=True,
        )
        self.assertIn("--no-margins", screening)
        self.assertNotIn("--no-margins", final)
        for command in (screening, final):
            self.assertEqual(
                command[command.index("--baseline-cache") + 1], str(cache),
            )
            self.assertEqual(command[command.index("--limit") + 1], "3")
            self.assertIn("--offline", command)


if __name__ == "__main__":
    unittest.main()


class ProtectedFamilyAcceptanceTests(unittest.TestCase):
    """The math family gates acceptance beside the matched MQuAKE conditions."""

    def report(self, openmath_exact=1.0):
        def condition(exact=1.0, containment=1.0):
            return {"exact_accuracy": exact, "answer_containment": containment}

        baseline = {
            "recall": condition(), "extraction": condition(), "reasoning": condition(),
            "by_family": {
                "recall": {**condition(), "conditions": ["recall"]},
                "extraction": {**condition(), "conditions": ["extraction"]},
                "reasoning": {**condition(), "conditions": ["reasoning"]},
                "openmath": {**condition(), "conditions": ["reasoning"]},
            },
        }
        masked = {
            **{key: dict(value) for key, value in baseline.items() if key != "by_family"},
            "by_family": {
                **{key: dict(value) for key, value in baseline["by_family"].items()},
                "openmath": {**condition(openmath_exact), "conditions": ["reasoning"]},
            },
        }
        return {"metrics": {"baseline": baseline, "candidate": masked}}

    def budgets(self, family_budget):
        return {
            "extraction_exact_drop": 0.10,
            "extraction_containment_drop": 0.05,
            "reasoning_exact_drop": 0.10,
            "reasoning_containment_drop": 0.05,
            "protected_family_exact_drop": family_budget,
        }

    def test_rejects_a_mask_that_breaks_only_the_math_family(self):
        accepted, observed = acceptance(self.report(0.70), "candidate", self.budgets(0.20))
        self.assertFalse(accepted)
        self.assertAlmostEqual(observed["drops"]["protected_family_exact_drop"], 0.30)
        self.assertAlmostEqual(observed["family_drops"]["protected_family_exact_drop"]["openmath"], 0.30)

    def test_ignores_the_recall_family_when_gating(self):
        report = self.report(1.0)
        report["metrics"]["candidate"]["by_family"]["recall"]["exact_accuracy"] = 0.0
        accepted, observed = acceptance(report, "candidate", self.budgets(0.20))
        self.assertTrue(accepted)
        self.assertNotIn("recall", observed["family_drops"]["protected_family_exact_drop"])
