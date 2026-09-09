"""
training/train.py

Training loop for one MultiStreamModel configuration (config/*.yaml),
implementing Handoff Section 12:

    Loss:            Cross Entropy with label smoothing = 0.1
    Optimizer:       AdamW
    LR scheduler:    Cosine Annealing
    Gradient clip:   max norm = 1.0 (stated purpose: prevent early
                     gradient spikes in the wavelet layer)

Learning rate, weight decay, batch size, epochs, and the cosine
scheduler's minimum LR are explicitly NOT specified in the research doc
(Handoff Section 19) -- this file reads them from each config's
`training:` section, where they're recorded as implementation defaults
(see config/*.yaml). CLI flags can override any of them for quick
experimentation without editing the YAML.

Two entry points:
    - train_model(...): plain Python function, returns a result dict.
      This is what experiments/run_experiments.py calls directly (no
      subprocess/CLI overhead) for the 7 configs x 5 seeds sweep.
    - main() / CLI: `python -m training.train --config config/full.yaml --seed 42`
      for a single one-off run.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataset import build_dataloaders, load_config
from experiments.seeds import set_seed
from models.multistream import MultiStreamModel
from training.metrics import ClassificationMetrics, compute_classification_metrics


def resolve_device(requested: Optional[str] = None) -> torch.device:
    """Pick the best available device unless the caller pins one explicitly."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def _move_batch(batch: Dict[str, torch.Tensor], active_branches: List[str], device: torch.device):
    inputs = {name: batch[name].to(device, non_blocking=True) for name in active_branches}
    labels = batch["label"].to(device, non_blocking=True)
    return inputs, labels


@torch.no_grad()
def run_inference(
    model: MultiStreamModel,
    dataloader,
    device: torch.device,
) -> Dict[str, List[int]]:
    """Run the model over a full dataloader and collect predictions + ground truth."""
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []  # P(AI-generated), for optional threshold analysis / calibration

    for batch in dataloader:
        inputs, labels = _move_batch(batch, model.active_branches, device)
        logits, _ = model(inputs)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        y_prob.extend(probs[:, 1].cpu().tolist())

    return {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}


@torch.no_grad()
def evaluate(
    model: MultiStreamModel,
    dataloader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> Dict:
    """Evaluate the model on a dataloader; returns loss (if criterion given) + classification metrics."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    y_true: List[int] = []
    y_pred: List[int] = []

    for batch in dataloader:
        inputs, labels = _move_batch(batch, model.active_branches, device)
        logits, _ = model(inputs)

        if criterion is not None:
            total_loss += criterion(logits, labels).item()
            n_batches += 1

        preds = logits.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    metrics = compute_classification_metrics(y_true, y_pred)
    result = {"metrics": metrics, "y_true": y_true, "y_pred": y_pred}
    if criterion is not None:
        result["loss"] = total_loss / max(n_batches, 1)
    return result


def train_one_epoch(
    model: MultiStreamModel,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    """One training epoch. Returns the mean training loss."""
    model.train()
    total_loss, n_batches = 0.0, 0

    for batch in dataloader:
        inputs, labels = _move_batch(batch, model.active_branches, device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()

        # Handoff Section 12: gradient clipping, max_norm = 1.0, to
        # prevent early gradient spikes in the (learnable) wavelet layer.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_model(
    config: dict,
    seed: int,
    run_name: Optional[str] = None,
    data_root: Optional[str] = None,
    device: Optional[torch.device] = None,
    checkpoint_root: Path = Path("checkpoints"),
    results_root: Path = Path("results"),
    epochs_override: Optional[int] = None,
    quiet: bool = False,
) -> Dict:
    """
    Train one MultiStreamModel configuration end to end and return a
    summary dict. This is the function experiments/run_experiments.py
    calls for every (config, seed) pair in the 7-configs x 5-seeds
    statistical protocol (Handoff Section 17).

    Layout produced:
        checkpoint_root/<run_name>/best.pt   -- best val-F1 checkpoint
        checkpoint_root/<run_name>/last.pt   -- final-epoch checkpoint
        results_root/<run_name>/history.json -- per-epoch train/val curves

    Returns:
        {
          "run_name": ..., "seed": ..., "config_name": ...,
          "best_val_f1": ..., "best_epoch": ..., "epochs_trained": ...,
          "checkpoint_path": ..., "history_path": ..., "history": [...]
        }
    """
    set_seed(seed, deterministic=True)

    config = copy.deepcopy(config)
    config_name = config.get("name", "model")
    run_name = run_name or f"{config_name}_seed{seed}"

    train_cfg = config.get("training", {})
    epochs = epochs_override or train_cfg.get("epochs", 30)          # NOT specified in doc -- see config/*.yaml
    lr = train_cfg.get("learning_rate", 1e-4)                        # NOT specified in doc
    weight_decay = train_cfg.get("weight_decay", 1e-4)               # NOT specified in doc
    lr_min = train_cfg.get("lr_min", 1e-6)                           # NOT specified in doc
    betas = tuple(train_cfg.get("optimizer_betas", (0.9, 0.999)))    # NOT specified in doc
    label_smoothing = train_cfg.get("label_smoothing", 0.1)          # Handoff Section 12: documented
    grad_clip_norm = train_cfg.get("grad_clip_norm", 1.0)            # Handoff Section 12: documented
    patience = train_cfg.get("early_stopping_patience", 7)           # NOT specified in doc

    device = device or resolve_device()

    dataloaders = build_dataloaders(config, data_root=data_root)
    model = MultiStreamModel.from_config(config).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    run_checkpoint_dir = checkpoint_root / config_name
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_results_dir = results_root / config_name
    run_results_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    best_epoch = -1
    epochs_since_improvement = 0
    history: List[Dict] = []

    start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, dataloaders["train"], optimizer, criterion, device, grad_clip_norm
        )
        val_result = evaluate(model, dataloaders["val"], device, criterion=criterion)
        val_metrics: ClassificationMetrics = val_result["metrics"]
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_result["loss"],
            "val_precision": val_metrics.precision,
            "val_recall": val_metrics.recall,
            "val_f1": val_metrics.f1,
            "val_accuracy": val_metrics.accuracy,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        if not quiet:
            print(
                f"[{run_name}] epoch {epoch:03d}/{epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_result['loss']:.4f} "
                f"val_f1={val_metrics.f1:.4f} lr={record['lr']:.2e}"
            )

        improved = val_metrics.f1 > best_val_f1
        if improved:
            best_val_f1 = val_metrics.f1
            best_epoch = epoch
            epochs_since_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "seed": seed,
                    "epoch": epoch,
                    "val_f1": best_val_f1,
                },
                run_checkpoint_dir / "best.pt",
            )
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            if not quiet:
                print(f"[{run_name}] early stopping at epoch {epoch} "
                      f"(no val_f1 improvement for {patience} epochs)")
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "seed": seed,
            "epoch": history[-1]["epoch"],
        },
        run_checkpoint_dir / "last.pt",
    )

    elapsed = time.time() - start
    history_path = run_results_dir / f"history_seed{seed}.json"
    with open(history_path, "w") as f:
        json.dump({"run_name": run_name, "seed": seed, "elapsed_seconds": elapsed, "history": history}, f, indent=2)

    return {
        "run_name": run_name,
        "seed": seed,
        "config_name": config_name,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "epochs_trained": history[-1]["epoch"],
        "checkpoint_path": str(run_checkpoint_dir / "best.pt"),
        "history_path": str(history_path),
        "history": history,
        "elapsed_seconds": elapsed,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one MultiStreamModel config (Handoff Section 12).")
    parser.add_argument("--config", required=True, help="Path to a config/*.yaml file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--run-name", default=None, help="Override the run's checkpoint/results folder name.")
    parser.add_argument("--data-root", default=None, help="Override config.data.root.")
    parser.add_argument("--epochs", type=int, default=None, help="Override config.training.epochs.")
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto-detect).")
    parser.add_argument("--checkpoint-root", default="checkpoints", help="Root directory for checkpoints.")
    parser.add_argument("--results-root", default="results", help="Root directory for result/history JSON.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    device = resolve_device(args.device)

    summary = train_model(
        config,
        seed=args.seed,
        run_name=args.run_name,
        data_root=args.data_root,
        device=device,
        checkpoint_root=Path(args.checkpoint_root),
        results_root=Path(args.results_root),
        epochs_override=args.epochs,
    )

    print(f"\nBest val F1: {summary['best_val_f1']:.4f} (epoch {summary['best_epoch']})")
    print(f"Checkpoint saved to: {summary['checkpoint_path']}")


if __name__ == "__main__":
    main()
