"""Build results/summary.json from saved checkpoint(s) without retraining.

This is the "use your own best.pt models" workflow for the project. It scans the
configured checkpoint root, evaluates each available model on the test split, and
writes the same JSON structure used by experiments/run_experiments.py so the
frontend benchmark page can render it directly.

Typical use:
    python -m experiments.build_benchmark_from_checkpoints \
        --config-dir config \
        --checkpoint-root checkpoints \
        --data-root data/raw \
        --device cpu

If a config has multiple checkpoint files (e.g. multiple seeds), all of them are
included and aggregated. If it has only a single best.pt model, the summary still
loads and renders as a single-run benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from data.dataset import load_config
from training.evaluate import evaluate_checkpoint
from training.metrics import RunResult, aggregate_metric, compute_classification_metrics

ALL_CONFIGS = [
    "full",
    "prnu_only",
    "ela_only",
    "content_only",
    "no_prnu",
    "no_ela",
    "no_content",
]

BASELINE_CONFIG = "full"

SOP_COMPARISONS: Dict[str, str] = {
    "SOP 1a (Full vs PRNU-only)": "prnu_only",
    "SOP 1b (Full vs ELA-only)": "ela_only",
    "SOP 1c (Full vs Content-only)": "content_only",
    "SOP 2 (Full vs Full-without-PRNU)": "no_prnu",
    "SOP 3 (Full vs Full-without-ELA)": "no_ela",
    "SOP 4 (Full vs Full-without-Content)": "no_content",
}

METRICS_FOR_TTEST = ("precision", "recall", "f1")


def _infer_seed_from_checkpoint(path: Path, fallback: int = 0) -> int:
    if path.parent.name.startswith("seed"):
        try:
            return int(path.parent.name.replace("seed", ""))
        except ValueError:
            pass
    if "seed" in path.stem.lower():
        try:
            return int(path.stem.lower().split("seed")[-1])
        except ValueError:
            pass
    return fallback


def _find_checkpoint_candidates(checkpoint_root: Path, config_name: str) -> List[Path]:
    candidates: List[Path] = []

    primary = checkpoint_root / config_name / "best.pt"
    if primary.exists():
        candidates.append(primary)

    for p in sorted(checkpoint_root.glob(f"{config_name}*/best.pt")):
        if p != primary and p not in candidates:
            candidates.append(p)

    for p in sorted(checkpoint_root.glob(f"**/{config_name}/best.pt")):
        if p != primary and p not in candidates:
            candidates.append(p)

    for p in sorted(checkpoint_root.glob(f"**/{config_name}/**/best.pt")):
        if p not in candidates:
            candidates.append(p)

    if not candidates:
        fallback = checkpoint_root / f"{config_name}.pt"
        if fallback.exists():
            candidates.append(fallback)

    return sorted(set(candidates))


def _paired_ttests_for_comparison(
    baseline_runs: Sequence[RunResult], comparison_runs: Sequence[RunResult]
) -> Dict[str, Dict[str, Optional[float] | bool]]:
    ttests: Dict[str, Dict[str, Optional[float] | bool]] = {}
    if len(baseline_runs) < 2 or len(comparison_runs) < 2:
        for metric in METRICS_FOR_TTEST:
            ttests[metric] = {"p_value": None, "significant": False}
        return ttests

    for metric in METRICS_FOR_TTEST:
        values_a = [getattr(r.metrics, metric) for r in baseline_runs]
        values_b = [getattr(r.metrics, metric) for r in comparison_runs]
        if len(values_a) != len(values_b):
            ttests[metric] = {"p_value": None, "significant": False}
            continue
        try:
            import numpy as np
            from scipy import stats

            arr_a = np.asarray(values_a, dtype=np.float64)
            arr_b = np.asarray(values_b, dtype=np.float64)
            _, p_value = stats.ttest_rel(arr_a, arr_b)
            ttests[metric] = {
                "p_value": float(p_value) if np.isfinite(p_value) else None,
                "significant": bool(p_value < 0.05) if np.isfinite(p_value) else False,
            }
        except Exception:
            ttests[metric] = {"p_value": None, "significant": False}

    return ttests


def _mcnemar_for_comparison(
    baseline_runs: Sequence[RunResult], comparison_runs: Sequence[RunResult]
) -> Optional[Dict[str, object]]:
    if len(baseline_runs) != len(comparison_runs):
        return None

    for base_run, comp_run in zip(baseline_runs, comparison_runs):
        if base_run.seed != comp_run.seed:
            return None

    if not baseline_runs:
        return None

    try:
        from training.metrics import mcnemar_test

        base_run = baseline_runs[0]
        comp_run = comparison_runs[0]
        result = mcnemar_test(base_run.y_true, base_run.y_pred, comp_run.y_pred)
        return {
            "p_value": float(result.p_value),
            "significant": bool(result.significant),
        }
    except Exception:
        return None


def _build_summary(all_runs: Dict[str, List[RunResult]], comparisons: Dict[str, Dict]) -> Dict:
    aggregates = {}
    for config_name, runs in all_runs.items():
        aggregates[config_name] = {
            "seeds": [r.seed for r in runs],
            "n_test_images": runs[0].metrics.n if runs else 0,
            **{
                metric: aggregate_metric(runs, metric).to_dict()
                for metric in ("precision", "recall", "f1", "accuracy")
            },
            "confusion_matrix_per_seed": [
                {"seed": r.seed, **r.metrics.to_dict()} for r in runs
            ],
        }

    return {
        "baseline": BASELINE_CONFIG,
        "aggregates": aggregates,
        "comparisons": comparisons,
    }


def _format_markdown_summary(summary: Dict) -> str:
    lines = ["# Multi-Stream CNN — Experiment Summary\n"]
    lines.append("## Aggregate metrics (mean ± std across available seeded runs)\n")
    lines.append("| Config | Precision | Recall | F1 | N (test) |")
    lines.append("|---|---|---|---|---|")
    for name, agg in summary["aggregates"].items():
        p, r, f1 = agg["precision"], agg["recall"], agg["f1"]
        lines.append(
            f"| {name} | {p['mean']:.4f} ± {p['std']:.4f} | {r['mean']:.4f} ± {r['std']:.4f} "
            f"| {f1['mean']:.4f} ± {f1['std']:.4f} | {agg['n_test_images']} |"
        )

    lines.append("\n## Comparisons vs. Full Model (alpha = 0.05)\n")
    lines.append("| SOP | Metric | Paired t p-value | Significant | McNemar p-value | Significant |")
    lines.append("|---|---|---|---|---|---|")
    for sop_label, result in summary["comparisons"].items():
        primary = result.get("mcnemar_primary")
        mcnemar_p_str = f"{primary['p_value']:.4g}" if primary and primary.get("p_value") is not None else "n/a"
        mcnemar_sig_str = str(primary["significant"]) if primary else "n/a"
        for metric, tres in result["paired_ttests"].items():
            p_value = tres["p_value"]
            p_str = f"{p_value:.4g}" if p_value is not None else "n/a"
            lines.append(
                f"| {sop_label} | {metric} | {p_str} | {tres['significant']} "
                f"| {mcnemar_p_str} | {mcnemar_sig_str} |"
            )
    return "\n".join(lines)


def build_summary_from_checkpoints(
    config_dir: Path,
    checkpoint_root: Path,
    results_root: Path,
    data_root: Optional[str],
    device: torch.device,
    configs: Optional[List[str]] = None,
) -> Dict:
    config_names = configs or ALL_CONFIGS

    all_runs: Dict[str, List[RunResult]] = {}
    for config_name in config_names:
        config_path = config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            continue
        config = load_config(str(config_path))
        checkpoint_paths = _find_checkpoint_candidates(checkpoint_root, config_name)
        if not checkpoint_paths:
            continue

        for checkpoint_path in checkpoint_paths:
            try:
                ckpt = torch.load(checkpoint_path, map_location=device)
            except Exception:
                continue

            eval_result = evaluate_checkpoint(
                str(checkpoint_path),
                config=config,
                split="test",
                data_root=data_root,
                device=device,
                save_path=None,
            )
            if not eval_result["y_true"]:
                raise ValueError(
                    f"No test samples were found for config '{config_name}' at '{checkpoint_path}'. "
                    "Check data.root, val_ratio/test_ratio, and that the raw dataset contains real/fake images."
                )
            metrics = compute_classification_metrics(eval_result["y_true"], eval_result["y_pred"])
            seed = ckpt.get("seed", _infer_seed_from_checkpoint(checkpoint_path))
            all_runs.setdefault(config_name, []).append(
                RunResult(seed=seed, metrics=metrics, y_true=eval_result["y_true"], y_pred=eval_result["y_pred"])
            )

    # Preserve the same summary shape as the training sweep, but omit
    # configs that have no saved checkpoint so the aggregate math stays valid.
    for config_name in list(all_runs):
        if not all_runs[config_name]:
            del all_runs[config_name]

    comparisons = {}
    for sop_label, comparison_name in SOP_COMPARISONS.items():
        if comparison_name not in all_runs:
            continue
        baseline_runs = all_runs.get(BASELINE_CONFIG, [])
        comp_runs = all_runs.get(comparison_name, [])
        if not baseline_runs or not comp_runs:
            continue

        paired = _paired_ttests_for_comparison(baseline_runs, comp_runs)
        primary = _mcnemar_for_comparison(baseline_runs, comp_runs)
        comparisons[sop_label] = {
            "paired_ttests": paired,
            "mcnemar_primary": primary,
        }

    summary = _build_summary(all_runs, comparisons)
    summary["seeds"] = sorted({seed for runs in all_runs.values() for seed in [r.seed for r in runs]})

    results_root.mkdir(parents=True, exist_ok=True)
    with open(results_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(results_root / "summary.md", "w", encoding="utf-8") as f:
        f.write(_format_markdown_summary(summary))

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create results/summary.json from saved .pt checkpoints without retraining."
    )
    parser.add_argument("--config-dir", default="config", help="Directory with the config/*.yaml files.")
    parser.add_argument("--checkpoint-root", default="checkpoints", help="Root directory containing saved .pt models.")
    parser.add_argument("--results-root", default="results", help="Root directory for summary files.")
    parser.add_argument("--data-root", default=None, help="Override config.data.root.")
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto-detect).")
    parser.add_argument("--configs", nargs="+", default=None, choices=ALL_CONFIGS, help="Subset of config names to include.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    from training.train import resolve_device

    device = resolve_device(args.device)
    summary = build_summary_from_checkpoints(
        config_dir=Path(args.config_dir),
        checkpoint_root=Path(args.checkpoint_root),
        results_root=Path(args.results_root),
        data_root=args.data_root,
        device=device,
        configs=args.configs,
    )

    print(f"Summary written to {Path(args.results_root) / 'summary.json'}")
    print(f"Available configs: {', '.join(sorted(summary['aggregates'])) if summary.get('aggregates') else 'none'}")


if __name__ == "__main__":
    main()
