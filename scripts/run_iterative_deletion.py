#!/usr/bin/env python3
"""Run resumable select/evaluate/accept/re-score channel-deletion rounds."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

from model_compression.channels import load_mask


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initial-scores", type=Path,
        default=Path("artifacts/iterative_deletion/round_002/recomputed_scores/scores.pt"),
    )
    parser.add_argument(
        "--initial-mask", type=Path,
        default=Path("artifacts/iterative_deletion/round_002/candidate_cumulative_mask.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/iterative_deletion/continuous_cost_sweep"),
    )
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--target-fraction", type=float, default=0.05)
    parser.add_argument("--protected-quantile", type=float, default=0.25)
    parser.add_argument("--recall-weight", type=float, default=0.05)
    parser.add_argument("--max-layer-fraction", type=float, default=0.15)
    parser.add_argument(
        "--min-backoff-fraction", type=float, default=0.001,
        help="smallest batch tried after a protected-budget rejection (default: 0.001)",
    )
    parser.add_argument("--no-backoff", action="store_true")
    parser.add_argument("--max-extraction-exact-drop", type=float, default=0.005)
    parser.add_argument("--max-extraction-containment-drop", type=float, default=0.005)
    parser.add_argument("--max-reasoning-exact-drop", type=float, default=0.005)
    parser.add_argument("--max-reasoning-containment-drop", type=float, default=0.005)
    parser.add_argument("--max-protected-family-exact-drop", type=float)
    parser.add_argument("--max-protected-family-margin-drop", type=float, help="nats; enables margins in every acceptance run")
    parser.add_argument("--max-protected-family-ce-increase", type=float, help="nats per answer token; enables margins in every acceptance run")
    parser.add_argument("--calibration-data", type=Path, default=Path(
        "data/calibration/mquake_remastered_cf3k_v2/calibration.jsonl"
    ))
    parser.add_argument("--validation-data", type=Path, default=Path(
        "data/calibration/mquake_remastered_cf3k_v2/validation.jsonl"
    ))
    parser.add_argument("--calibration-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    if not 0 < args.fraction <= 1 or not 0 < args.target_fraction <= 1:
        parser.error("fraction and target fraction must be in (0, 1]")
    if not 0 < args.protected_quantile < 1:
        parser.error("protected quantile must be between zero and one")
    if args.recall_weight < 0:
        parser.error("recall weight must be nonnegative")
    if not 0 < args.max_layer_fraction <= 1:
        parser.error("max layer fraction must be in (0, 1]")
    if not 0 < args.min_backoff_fraction <= args.fraction:
        parser.error("minimum backoff fraction must be in (0, fraction]")
    budgets = (
        args.max_extraction_exact_drop, args.max_extraction_containment_drop,
        args.max_reasoning_exact_drop, args.max_reasoning_containment_drop,
    )
    if not all(0 <= value <= 1 for value in budgets):
        parser.error("acceptance budgets must be between zero and one")
    extra_budgets = (args.max_protected_family_exact_drop, args.max_protected_family_margin_drop,
                     args.max_protected_family_ce_increase)
    if any(x is not None and (not math.isfinite(x) or x < 0) for x in extra_budgets):
        parser.error("family budgets must be finite and nonnegative")
    if args.max_protected_family_exact_drop is not None and args.max_protected_family_exact_drop > 1:
        parser.error("family accuracy drop must be at most one")
    limits = (args.calibration_limit, args.validation_limit)
    if any(value is not None and value < 1 for value in limits):
        parser.error("limits must be positive")
    if min(args.batch_size, args.max_new_tokens, args.checkpoint_every) < 1:
        parser.error("batch size, max new tokens, and checkpoint frequency must be positive")
    return args


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def score_shape(path: Path) -> tuple[int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    shape = tuple(payload["recall_importance_ce"].shape)
    if len(shape) != 2:
        raise ValueError(f"expected two-dimensional channel scores: {path}")
    return shape


def acceptance(report: dict, mode: str, budgets: dict[str, float]) -> tuple[bool, dict]:
    metrics = report["metrics"]
    baseline = metrics["baseline"]
    masked = metrics[mode]
    drops = {
        "extraction_exact_drop": baseline["extraction"]["exact_accuracy"] - masked["extraction"]["exact_accuracy"],
        "extraction_containment_drop": baseline["extraction"]["answer_containment"] - masked["extraction"]["answer_containment"],
        "reasoning_exact_drop": baseline["reasoning"]["exact_accuracy"] - masked["reasoning"]["exact_accuracy"],
        "reasoning_containment_drop": baseline["reasoning"]["answer_containment"] - masked["reasoning"]["answer_containment"],
    }
    family_checks = {}
    for key, field in (("protected_family_exact_drop", "exact_accuracy"),
                       ("protected_family_margin_drop", "margin_drop"),
                       ("protected_family_ce_increase", "ce_increase")):
        if key not in budgets:
            continue
        if "by_family" not in masked or "by_family" not in baseline:
            raise ValueError("family budgets require evaluation with per-family metrics")
        values = {}
        for family, measured in masked["by_family"].items():
            if measured.get("conditions") == ["recall"] or family in ("recall", "mquake_recall"):
                continue
            if "recall" in measured.get("conditions", []):
                raise ValueError("protected family metrics must not mix recall and protected conditions")
            if field not in measured:
                raise ValueError(f"family budget requires {field}; enable margin evaluation")
            values[family] = (baseline["by_family"][family][field] - measured[field]
                              if field == "exact_accuracy" else measured[field])
            if not math.isfinite(values[family]):
                raise ValueError(f"nonfinite protected metric for {family}")
        if not values:
            raise ValueError("no protected families available for acceptance")
        drops[key] = max(values.values())
        family_checks[key] = values
    accepted = all(drops[key] <= budgets[key] + 1e-12 for key in budgets)
    observed = {
        "family_drops": family_checks,
        "drops": drops,
        "baseline": {condition: baseline[condition] for condition in ("recall", "extraction", "reasoning")},
        "masked": {condition: masked[condition] for condition in ("recall", "extraction", "reasoning")},
        "paired_by_condition": masked.get("paired_by_condition"),
        "mean_margin_delta_by_condition": masked.get("mean_margin_delta_by_condition"),
    }
    return accepted, observed


DROP_LABELS = {
    "extraction_exact_drop": "extraction exact",
    "extraction_containment_drop": "extraction containment",
    "reasoning_exact_drop": "reasoning exact",
    "reasoning_containment_drop": "reasoning containment",
}


def report_round(decision: dict, budgets: dict[str, float]) -> None:
    """One human- and grep-readable block per attempted round."""
    drops = decision["observed"]["drops"]
    over = [DROP_LABELS.get(key, key) for key in budgets if drops[key] > budgets[key] + 1e-12]
    print(
        f"round {decision['round']:03d}: {decision['decision'].upper()}  "
        f"batch {decision['batch_count']}  "
        f"cumulative {decision['cumulative_count']} "
        f"({decision['cumulative_fraction'] * 100:.3f}%)",
        flush=True,
    )
    print("  protected drops (pts, + is worse): " + ", ".join(
        f"{label} {drops[key] * 100:+.3f}" for key, label in DROP_LABELS.items()
    ), flush=True)
    if decision["observed"].get("family_drops"):
        print("  family checks: " + json.dumps(decision["observed"]["family_drops"]), flush=True)
    baseline_recall = decision["observed"]["baseline"]["recall"]["exact_accuracy"]
    masked_recall = decision["observed"]["masked"]["recall"]["exact_accuracy"]
    print(
        f"  recall exact: {baseline_recall * 100:.2f}% -> {masked_recall * 100:.2f}% "
        f"({(masked_recall - baseline_recall) * 100:+.2f} pts)",
        flush=True,
    )
    if over:
        print("  OVER BUDGET: " + ", ".join(over), flush=True)


def summary(state: dict, state_path: Path) -> str:
    accepted = state["accepted_rounds"]
    lines = [
        "",
        "=" * 62,
        f"status:            {state['status']}",
        f"rounds accepted:   {len(accepted)}"
        + (f"  ({', '.join(str(value) for value in accepted)})" if accepted else ""),
        f"channels deleted:  {state['current_count']} / {state['total_channels']}"
        f"  ({state['current_fraction'] * 100:.4f}%)",
        f"target was:        {state['target_count']}"
        f"  ({state['configuration']['target_fraction'] * 100:.4f}%)",
    ]
    if state.get("rejected_round") is not None:
        lines.append(f"stopped at round:  {state['rejected_round']} (protected budget)")
        lines.append(f"rejection:         {state['rejection']}")
    final_evaluation = state.get("final_evaluation")
    if final_evaluation is not None and Path(final_evaluation).is_file():
        report = json.loads(Path(final_evaluation).read_text())
        mode = Path(state["final_mask"]).stem
        _, observed = acceptance(report, mode, state["configuration"]["budgets"])
        lines.append("final protected drops (pts, + is worse):")
        for key, label in DROP_LABELS.items():
            lines.append(f"  {label + ':':<26}{observed['drops'][key] * 100:+.3f}")
        baseline_recall = observed["baseline"]["recall"]["exact_accuracy"]
        masked_recall = observed["masked"]["recall"]["exact_accuracy"]
        lines.append(
            f"  {'recall exact:':<26}{baseline_recall * 100:.2f}% -> "
            f"{masked_recall * 100:.2f}% ({(masked_recall - baseline_recall) * 100:+.2f} pts)"
        )
    lines.append(f"final mask:        {state.get('final_mask', state['current_mask'])}")
    if final_evaluation is not None:
        lines.append(f"final evaluation:  {final_evaluation}")
    lines.append(f"state:             {state_path}")
    lines.append("=" * 62)
    return "\n".join(lines)


def path_arg(path: Path) -> str:
    return str(path.resolve())


def evaluation_command(
    args: argparse.Namespace, mask: str, output_dir: Path, cache: Path, margins: bool,
) -> list[str]:
    """Screening rounds skip margins; only the final report pays for them."""
    command = [
        sys.executable, str(ROOT / "scripts/evaluate_robustness.py"),
        "--data", path_arg(args.validation_data), "--mask", mask,
        "--output", str(output_dir), "--device", args.device,
        "--batch-size", str(args.batch_size),
        "--max-new-tokens", str(args.max_new_tokens),
        "--baseline-cache", str(cache),
    ]
    if not margins:
        command.append("--no-margins")
    if args.offline:
        command.append("--offline")
    if args.validation_limit is not None:
        command.extend(["--limit", str(args.validation_limit)])
    return command


def next_backoff_count(proposed_count: int, minimum_count: int) -> int | None:
    """Halve a rejected batch, or stop after the minimum batch is rejected."""
    if proposed_count <= minimum_count:
        return None
    return max(minimum_count, proposed_count // 2)


def configuration(args: argparse.Namespace) -> dict:
    return {
        "initial_scores": path_arg(args.initial_scores),
        "initial_mask": path_arg(args.initial_mask),
        "fraction": args.fraction,
        "target_fraction": args.target_fraction,
        "protected_quantile": args.protected_quantile,
        "recall_weight": args.recall_weight,
        "max_layer_fraction": args.max_layer_fraction,
        "min_backoff_fraction": args.min_backoff_fraction,
        "backoff": not args.no_backoff,
        "budgets": {
            "extraction_exact_drop": args.max_extraction_exact_drop,
            "extraction_containment_drop": args.max_extraction_containment_drop,
            "reasoning_exact_drop": args.max_reasoning_exact_drop,
            "reasoning_containment_drop": args.max_reasoning_containment_drop,
            **{key: value for key, value in (
                ("protected_family_exact_drop", args.max_protected_family_exact_drop),
                ("protected_family_margin_drop", args.max_protected_family_margin_drop),
                ("protected_family_ce_increase", args.max_protected_family_ce_increase),
            ) if value is not None},
        },
        "calibration_data": path_arg(args.calibration_data),
        "validation_data": path_arg(args.validation_data),
        "calibration_limit": args.calibration_limit,
        "validation_limit": args.validation_limit,
        "device": args.device,
        "offline": args.offline,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "checkpoint_every": args.checkpoint_every,
    }


def new_state(args: argparse.Namespace, config: dict) -> dict:
    scores = args.initial_scores.resolve()
    mask = args.initial_mask.resolve()
    if not scores.is_file() or not mask.is_file():
        raise FileNotFoundError("initial scores and mask must both exist")
    shape = score_shape(scores)
    total_channels = math.prod(shape)
    current_count = len(load_mask(mask))
    return {
        "status": "running", "stage": "select", "round": 1,
        "configuration": config, "score_shape": list(shape),
        "total_channels": total_channels,
        "target_count": math.ceil(total_channels * args.target_fraction),
        "current_count": current_count,
        "current_fraction": current_count / total_channels,
        "current_mask": str(mask), "current_scores": str(scores),
        "accepted_rounds": [],
        "attempt": 0, "requested_count": None,
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    state_path = output / "state.json"
    config = configuration(args)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["configuration"] != config:
            raise ValueError("arguments differ from the saved sweep; resume with the original arguments")
        if state["status"] != "running" and state.get("stage") != "final_report":
            print(summary(state, state_path))
            return
        print(f"resuming round {state['round']} at {state['stage']} stage", flush=True)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"output exists without resumable state: {output}")
        state = new_state(args, config)
        atomic_json(state, state_path)

    while state["status"] == "running":
        if state["current_count"] >= state["target_count"]:
            state.update(status="complete", stage="final_report", final_mask=state["current_mask"])
            atomic_json(state, state_path)
            break

        suffix = "" if state.get("attempt", 0) == 0 else f"_backoff_{state['attempt']:02d}"
        round_dir = output / f"round_{state['round']:03d}{suffix}"
        candidate = round_dir / "candidate_cumulative_mask.json"
        evaluation_dir = round_dir / "evaluation"

        if state["stage"] == "select":
            if not candidate.exists():
                command = [
                    sys.executable, str(ROOT / "scripts/select_iterative_batch.py"),
                    state["current_scores"], state["current_mask"],
                    "--output", str(round_dir),
                    "--target-fraction", str(args.target_fraction),
                    "--protected-quantile", str(args.protected_quantile),
                    "--recall-weight", str(args.recall_weight),
                    "--max-layer-fraction", str(args.max_layer_fraction),
                ]
                if state.get("requested_count") is not None:
                    command.extend(["--count", str(state["requested_count"])])
                else:
                    command.extend(["--fraction", str(args.fraction)])
                run(command)
            state.update(stage="evaluate", candidate_mask=str(candidate))
            atomic_json(state, state_path)

        if state["stage"] == "evaluate":
            report_path = evaluation_dir / "report.json"
            if not report_path.exists():
                run(evaluation_command(
                    args, state["candidate_mask"], evaluation_dir,
                    output / "baseline_cache.json", margins=any(
                        key in config["budgets"] for key in
                        ("protected_family_margin_drop", "protected_family_ce_increase")
                    ),
                ))
            report = json.loads(report_path.read_text())
            mode = Path(state["candidate_mask"]).stem
            accepted, observed = acceptance(report, mode, config["budgets"])
            cumulative_count = len(load_mask(Path(state["candidate_mask"])))
            decision = {
                "round": state["round"], "decision": "accepted" if accepted else "rejected",
                "batch_count": cumulative_count - state["current_count"],
                "cumulative_count": cumulative_count,
                "cumulative_fraction": cumulative_count / state["total_channels"],
                "candidate_mask": state["candidate_mask"],
                "source_evaluation": str(report_path),
                "protected_budgets": config["budgets"], "observed": observed,
            }
            atomic_json(decision, round_dir / "acceptance.json")
            report_round(decision, config["budgets"])
            if not accepted:
                minimum_count = math.ceil(state["total_channels"] * args.min_backoff_fraction)
                retry_count = None if args.no_backoff else next_backoff_count(
                    cumulative_count - state["current_count"], minimum_count
                )
                if retry_count is not None:
                    state.update(
                        stage="select", attempt=state.get("attempt", 0) + 1,
                        requested_count=retry_count,
                    )
                    state.pop("candidate_mask", None)
                    atomic_json(state, state_path)
                    continue
                state.update(
                    status="stopped_protected_budget", stage="final_report",
                    rejected_round=state["round"], final_mask=state["current_mask"],
                    rejection=str(round_dir / "acceptance.json"),
                )
                atomic_json(state, state_path)
                break

            state["accepted_rounds"].append(state["round"])
            state.update(
                current_mask=state["candidate_mask"], current_count=cumulative_count,
                current_fraction=cumulative_count / state["total_channels"],
            )
            if cumulative_count >= state["target_count"]:
                state.update(status="complete", stage="final_report", final_mask=state["current_mask"])
                atomic_json(state, state_path)
                break
            state.update(
                stage="score", current_scores=str(round_dir / "recomputed_scores" / "scores.pt"),
                attempt=0, requested_count=None,
            )
            atomic_json(state, state_path)

        if state["stage"] == "score":
            scores_path = Path(state["current_scores"])
            if not scores_path.exists():
                command = [
                    sys.executable, str(ROOT / "scripts/score_channels_v2.py"),
                    "--input", path_arg(args.calibration_data), "--output", str(scores_path.parent),
                    "--mask", state["current_mask"], "--device", args.device,
                    "--checkpoint-every", str(args.checkpoint_every),
                ]
                if args.offline:
                    command.append("--offline")
                if args.calibration_limit is not None:
                    command.extend(["--limit", str(args.calibration_limit)])
                run(command)
            state.update(round=state["round"] + 1, stage="select")
            atomic_json(state, state_path)

    if state.get("stage") == "final_report":
        final_dir = output / "final_evaluation"
        if not (final_dir / "report.json").exists():
            run(evaluation_command(
                args, state["final_mask"], final_dir,
                output / "baseline_cache.json", margins=True,
            ))
        state.update(stage="done", final_evaluation=str(final_dir / "report.json"))
        atomic_json(state, state_path)

    print(summary(state, state_path))


if __name__ == "__main__":
    main()
