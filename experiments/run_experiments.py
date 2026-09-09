"""
experiments/run_experiments.py

Orchestrates the full experimental protocol from Handoff Sections 15 and 17:

    Configs (config/*.yaml, 7 total):
        full, prnu_only, ela_only, content_only, no_prnu, no_ela, no_content

    For each config: 5 independent training runs with different seeds
    (experiments/seeds.EXPERIMENT_SEEDS), each evaluated on the SAME
    held-out test set (guaranteed by every config sharing data.seed).

    Comparisons against the Full Model:
        SOP 1a: Full vs PRNU-only
        SOP 1b: Full vs ELA-only
        SOP 1c: Full vs Content-only
        SOP 2:  Full vs Full-without-PRNU   (no_prnu.yaml = ELA+Content)
        SOP 3:  Full vs Full-without-ELA    (no_ela.yaml  = PRNU+Content)
        SOP 4:  Full vs Full-without-Content(no_content.yaml = PRNU+ELA)

    Statistics per comparison (Handoff Section 17):
        - Paired t-test (alpha=0.05) on precision / recall / F1 across
          the 5 seed-aligned runs.
        - McNemar's test (alpha=0.05) on per-image predictions, since
          both models are evaluated on the same test images.

Outputs:
    results/summary.json    -- everything, machine-readable
    results/summary.md      -- human-readable table
    checkpoints/<config>/   -- per-seed best/last checkpoints (via training/train.py)
    results/<config>/       -- per-seed training history + test predictions

CLI:
    python -m experiments.run_experiments
    python -m experiments.run_experiments --configs full ela_only --epochs 1 --seeds 42
        (useful for a fast smoke test before committing to the full sweep)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from data.dataset import load_config
from experiments.seeds import EXPERIMENT_SEEDS
from training.evaluate import evaluate_checkpoint
from training.metrics import (
    RunResult,
    aggregate_metric,
    mcnemar_test,
    paired_ttest,
)
from training.train import resolve_device, train_model

CONFIG_DIR_DEFAULT = Path("config")

# All 7 configs required by Handoff Section 15's task table.
ALL_CONFIGS = ["full", "prnu_only", "ela_only", "content_only", "no_prnu", "no_ela", "no_content"]

BASELINE_CONFIG = "full"

# Handoff Section 15: SOP label -> config compared against the Full Model.
SOP_COMPARISONS: Dict[str, str] = {
    "SOP 1a (Full vs PRNU-only)": "prnu_only",
    "SOP 1b (Full vs ELA-only)": "ela_only",
    "SOP 1c (Full vs Content-only)": "content_only",
    "SOP 2 (Full vs Full-without-PRNU)": "no_prnu",
    "SOP 3 (Full vs Full-without-ELA)": "no_ela",
    "SOP 4 (Full vs Full-without-Content)": "no_content",
}

METRICS_FOR_TTEST = ("precision", "recall", "f1")


def run_single_config(
    config_path: Path,
    seeds: List[int],
    data_root: Optional[str],
    device: torch.device,
    checkpoint_root: Path,
    results_root: Path,
    epochs_override: Optional[int],
    quiet: bool,
) -> List[RunResult]:
    """Train + evaluate one config across all `seeds`, returning one RunResult per seed."""
    config = load_config(str(config_path))
    config_name = config.get("name", config_path.stem)

    runs: List[RunResult] = []
    for seed in seeds:
        run_name = f"{config_name}_seed{seed}"
        if not quiet:
            print(f"\n=== {config_name} | seed {seed} ===")

        train_summary = train_model(
            config,
            seed=seed,
            run_name=run_name,
            data_root=data_root,
            device=device,
            checkpoint_root=checkpoint_root,
            results_root=results_root,
            epochs_override=epochs_override,
            quiet=quiet,
        )

        eval_save_path = results_root / config_name / f"test_eval_seed{seed}.json"
        eval_result = evaluate_checkpoint(
            train_summary["checkpoint_path"],
            config=config,
            split="test",
            data_root=data_root,
            device=device,
            save_path=eval_save_path,
        )

        # Rebuild a ClassificationMetrics from the already-computed dict
        # (evaluate_checkpoint recomputes internally; reuse its metrics).
        from training.metrics import compute_classification_metrics
        metrics = compute_classification_metrics(eval_result["y_true"], eval_result["y_pred"])

        runs.append(RunResult(seed=seed, metrics=metrics, y_true=eval_result["y_true"], y_pred=eval_result["y_pred"]))

    return runs


def compare_to_baseline(baseline_runs: List[RunResult], comparison_runs: List[RunResult]) -> Dict:
    """
    Paired t-tests (per metric) + McNemar's test, per Handoff Section 17.

    McNemar's test is run once per seed-aligned run pair (5 tests, since
    each seed produces a different trained model and therefore different
    per-image predictions, even though the test IMAGES are identical
    across seeds/configs). Which single run's McNemar result should be
    treated as authoritative is NOT specified in the research doc, so
    all 5 are reported plus a "primary" one (the first seed) is
    highlighted as the headline number -- an implementation choice,
    flagged here rather than silently picked.
    """
    ttests = {
        metric: paired_ttest(baseline_runs, comparison_runs, metric).__dict__
        for metric in METRICS_FOR_TTEST
    }

    mcnemar_per_seed = []
    for base_run, comp_run in zip(baseline_runs, comparison_runs):
        assert base_run.seed == comp_run.seed
        # Both runs were evaluated on the same config.data.seed-derived
        # test split, so y_true should match exactly; assert that rather
        # than silently comparing misaligned predictions.
        assert base_run.y_true == comp_run.y_true, (
            "Baseline and comparison runs disagree on the test set's ground truth -- "
            "check that both configs share the same data.root/seed/val_ratio/test_ratio."
        )
        result = mcnemar_test(base_run.y_true, base_run.y_pred, comp_run.y_pred)
        mcnemar_per_seed.append({"seed": base_run.seed, **result.__dict__})

    return {
        "paired_ttests": ttests,
        "mcnemar_per_seed": mcnemar_per_seed,
        "mcnemar_primary": mcnemar_per_seed[0] if mcnemar_per_seed else None,
    }


def build_summary(all_runs: Dict[str, List[RunResult]], comparisons: Dict[str, Dict]) -> Dict:
    """Assemble the full JSON-serializable summary of aggregated metrics + statistical tests."""
    aggregates = {}
    for config_name, runs in all_runs.items():
        aggregates[config_name] = {
            "seeds": [r.seed for r in runs],
            "n_test_images": runs[0].metrics.n if runs else 0,
            **{metric: aggregate_metric(runs, metric).to_dict() for metric in ("precision", "recall", "f1", "accuracy")},
            "confusion_matrix_per_seed": [
                {"seed": r.seed, **r.metrics.to_dict()} for r in runs
            ],
        }

    return {
        "baseline": BASELINE_CONFIG,
        "aggregates": aggregates,
        "comparisons": comparisons,
    }


def format_markdown_summary(summary: Dict) -> str:
    lines = ["# Multi-Stream CNN — Experiment Summary\n"]
    lines.append("## Aggregate metrics (mean ± std across 5 seeded runs)\n")
    lines.append("| Config | Precision | Recall | F1 | N (test) |")
    lines.append("|---|---|---|---|---|")
    for name, agg in summary["aggregates"].items():
        p, r, f1 = agg["precision"], agg["recall"], agg["f1"]
        lines.append(
            f"| {name} | {p['mean']:.4f} ± {p['std']:.4f} | {r['mean']:.4f} ± {r['std']:.4f} "
            f"| {f1['mean']:.4f} ± {f1['std']:.4f} | {agg['n_test_images']} |"
        )

    lines.append("\n## Comparisons vs. Full Model (alpha = 0.05)\n")
    lines.append("| SOP | Metric | Paired t p-value | Significant | McNemar p-value (seed[0]) | Significant |")
    lines.append("|---|---|---|---|---|---|")
    for sop_label, result in summary["comparisons"].items():
        primary = result["mcnemar_primary"]
        mcnemar_p_str = f"{primary['p_value']:.4g}" if primary else "n/a"
        mcnemar_sig_str = str(primary["significant"]) if primary else "n/a"
        for metric, tres in result["paired_ttests"].items():
            lines.append(
                f"| {sop_label} | {metric} | {tres['p_value']:.4g} | {tres['significant']} "
                f"| {mcnemar_p_str} | {mcnemar_sig_str} |"
            )
    return "\n".join(lines)


def run_all_experiments(
    config_dir: Path = CONFIG_DIR_DEFAULT,
    configs: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    data_root: Optional[str] = None,
    device: Optional[torch.device] = None,
    checkpoint_root: Path = Path("checkpoints"),
    results_root: Path = Path("results"),
    epochs_override: Optional[int] = None,
    quiet: bool = False,
) -> Dict:
    """Run the full (or a subset of the) SOP protocol and write results/summary.{json,md}."""
    configs = configs or ALL_CONFIGS
    seeds = seeds or EXPERIMENT_SEEDS
    device = device or resolve_device()

    if BASELINE_CONFIG not in configs:
        raise ValueError(
            f"'{BASELINE_CONFIG}' (the Full Model) must be included in `configs` -- "
            "every SOP comparison is measured against it."
        )

    all_runs: Dict[str, List[RunResult]] = {}
    for name in configs:
        config_path = config_dir / f"{name}.yaml"
        runs = run_single_config(
            config_path, seeds, data_root, device, checkpoint_root, results_root, epochs_override, quiet
        )
        all_runs[name] = runs

    comparisons = {}
    for sop_label, comparison_name in SOP_COMPARISONS.items():
        if comparison_name not in all_runs:
            continue  # comparison config wasn't included in this run (e.g. a partial/subset sweep)
        comparisons[sop_label] = compare_to_baseline(all_runs[BASELINE_CONFIG], all_runs[comparison_name])

    summary = build_summary(all_runs, comparisons)
    summary.pop("seeds", None)
    summary["seeds"] = seeds

    results_root.mkdir(parents=True, exist_ok=True)
    with open(results_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(results_root / "summary.md", "w") as f:
        f.write(format_markdown_summary(summary))

    if not quiet:
        print(f"\nSummary written to {results_root / 'summary.json'} and {results_root / 'summary.md'}")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full multi-stream CNN experimental protocol (Handoff Sections 15, 17).")
    parser.add_argument("--config-dir", default=str(CONFIG_DIR_DEFAULT), help="Directory containing config/*.yaml.")
    parser.add_argument("--configs", nargs="+", default=None, choices=ALL_CONFIGS,
                         help="Subset of configs to run (default: all 7). 'full' must be included.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                         help=f"Override the 5 seeds (default: {EXPERIMENT_SEEDS}).")
    parser.add_argument("--data-root", default=None, help="Override config.data.root for every config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override config.training.epochs for every run (useful for smoke tests).")
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto-detect).")
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-epoch training logs.")
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    device = resolve_device(args.device)

    run_all_experiments(
        config_dir=Path(args.config_dir),
        configs=args.configs,
        seeds=args.seeds,
        data_root=args.data_root,
        device=device,
        checkpoint_root=Path(args.checkpoint_root),
        results_root=Path(args.results_root),
        epochs_override=args.epochs,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
