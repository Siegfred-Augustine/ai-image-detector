"""
training/evaluate.py

Load a trained checkpoint and evaluate it on a data split (test by
default), per Handoff Section 16: Precision, Recall, F1, Confusion
Matrix. Also returns raw per-image predictions, which
experiments/run_experiments.py needs for McNemar's test (Section 17,
which explicitly compares models on the same test images).

Two entry points, mirroring training/train.py:
    - evaluate_checkpoint(...): plain function returning a result dict
      (used programmatically by experiments/run_experiments.py).
    - CLI: `python -m training.evaluate --config config/full.yaml --checkpoint checkpoints/full/best.pt`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch

from data.dataset import build_dataloaders, load_config
from models.multistream import MultiStreamModel
from training.metrics import compute_classification_metrics, format_confusion_matrix
from training.train import resolve_device, run_inference


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> MultiStreamModel:
    """
    Rebuild a MultiStreamModel from a checkpoint saved by training/train.py.
    The checkpoint stores the exact config used to train it, so the
    model architecture doesn't need to be reconstructed by hand.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = MultiStreamModel.from_config(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def evaluate_checkpoint(
    checkpoint_path: str,
    config: Optional[dict] = None,
    split: str = "test",
    data_root: Optional[str] = None,
    device: Optional[torch.device] = None,
    save_path: Optional[Path] = None,
) -> Dict:
    """
    Evaluate a saved checkpoint on one split ("train" / "val" / "test").

    Args:
        checkpoint_path: path to a .pt file saved by training/train.py.
        config: the config dict to build the dataloader from. If None,
            the config embedded in the checkpoint is reused (the
            checkpoint's config is authoritative for reproducing the
            exact data split that was used at training time — same
            data.seed as training/train.py used — since Handoff
            Section 17's McNemar's test requires the SAME test images
            across every model being compared).
        split: which dataloader split to evaluate on.
        data_root: override the data root path if the dataset moved.
        device: torch device; auto-detected if omitted.
        save_path: if given, dump the full result (metrics + raw
            per-image predictions) to this JSON path.

    Returns:
        {
          "metrics": {...},                 # from ClassificationMetrics.to_dict()
          "confusion_matrix_text": "...",
          "y_true": [...], "y_pred": [...], "y_prob": [...],
          "config_name": ..., "checkpoint_path": ...,
        }
    """
    device = device or resolve_device()
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = config or ckpt["config"]

    model = MultiStreamModel.from_config(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataloaders = build_dataloaders(config, data_root=data_root)
    if split not in dataloaders:
        raise ValueError(f"Unknown split '{split}'; expected one of {list(dataloaders)}")

    predictions = run_inference(model, dataloaders[split], device)
    metrics = compute_classification_metrics(predictions["y_true"], predictions["y_pred"])

    result = {
        "config_name": config.get("name", "model"),
        "checkpoint_path": str(checkpoint_path),
        "split": split,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_seed": ckpt.get("seed"),
        "metrics": metrics.to_dict(),
        "confusion_matrix_text": format_confusion_matrix(metrics),
        "y_true": predictions["y_true"],
        "y_pred": predictions["y_pred"],
        "y_prob": predictions["y_prob"],
    }

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint (Handoff Section 16).")
    parser.add_argument("--config", default=None, help="Path to a config/*.yaml file (default: reuse the checkpoint's own config).")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint from training/train.py.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-root", default=None, help="Override config.data.root.")
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto-detect).")
    parser.add_argument("--output", default=None, help="Optional path to save the result JSON.")
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config) if args.config else None
    device = resolve_device(args.device)

    save_path = Path(args.output) if args.output else None
    result = evaluate_checkpoint(
        args.checkpoint,
        config=config,
        split=args.split,
        data_root=args.data_root,
        device=device,
        save_path=save_path,
    )

    print(f"\n{result['config_name']} — {args.split} set ({result['metrics']['n']} images)")
    print(f"Precision: {result['metrics']['precision']:.4f}")
    print(f"Recall:    {result['metrics']['recall']:.4f}")
    print(f"F1:        {result['metrics']['f1']:.4f}")
    print(f"FN rate (AI accepted as real): {result['metrics']['fn_rate']:.4f}")
    print("\n" + result["confusion_matrix_text"])
    if save_path:
        print(f"Full results (incl. per-image predictions) saved to: {save_path}")


if __name__ == "__main__":
    main()
