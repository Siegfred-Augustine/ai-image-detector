# AI-Generated Image Detector — Multi-Stream CNN

Detects AI-generated images using three forensic streams — Content, ELA
(Error Level Analysis), and PRNU (via a learnable wavelet-residual layer)
— fused with attention. 

## 1. Setup

```bash
cd ai-image-detector-main
pip install -r requirements.txt
```

Requires Python 3.10+. GPU is optional but strongly recommended for
real training runs (CPU works fine for smoke-testing).

## 2. Point it at your dataset

Your images need to be laid out as:

```
data/raw/real/*.jpg
data/raw/fake/*.jpg
```

If your data lives somewhere else, either move it, or override the
path per-command with `--data-root /path/to/your/data`, or edit it
directly in every `config/*.yaml`:

```yaml
data:
  root: "./data/raw"   # <- change this
```

**Important:** if you edit `data.root` (or `data.seed`, `val_ratio`,
`test_ratio`), change it identically in **all 7** config files. The
statistical comparisons between models (McNemar's test) only work if
every model was evaluated on the exact same test images.

## 3. Train just the Full Model

```bash
python -m training.train --config config/full.yaml --seed 42
```

Optional flags:

| Flag | Meaning |
|---|---|
| `--epochs N` | Override the config's epoch count |
| `--device cuda\|mps\|cpu` | Force a device (auto-detects otherwise) |
| `--data-root PATH` | Override `config.data.root` |
| `--checkpoint-root DIR` | Where checkpoints get saved (default `checkpoints/`) |
| `--results-root DIR` | Where training history gets saved (default `results/`) |
| `--run-name NAME` | Custom name for this run's output folder |

**Output:**
```
checkpoints/full/best.pt        # best validation-F1 checkpoint
checkpoints/full/last.pt        # final-epoch checkpoint
results/full/history_seed42.json  # per-epoch train/val loss + metrics
```

## 4. Evaluate a trained checkpoint

```bash
python -m training.evaluate \
  --checkpoint checkpoints/full/best.pt \
  --split test \
  --output results/full/test_eval.json
```

Prints precision / recall / F1 / confusion matrix to the terminal, and
saves the full result (including every per-image prediction) to the
JSON file.

`--split` can be `train`, `val`, or `test`.

## 5. Training one of the ablation configs

Same command, just point at a different config — no other changes needed:

```bash
python -m training.train --config config/prnu_only.yaml --seed 42
python -m training.train --config config/ela_only.yaml --seed 42
python -m training.train --config config/content_only.yaml --seed 42
python -m training.train --config config/no_prnu.yaml --seed 42
python -m training.train --config config/no_ela.yaml --seed 42
python -m training.train --config config/no_content.yaml --seed 42
```

## 6. Run the full statistical comparison (all 7 configs × 5 seeds)

This is the complete experimental protocol: trains every config 5 times
(different seeds), evaluates each on the test set, and runs the SOP1a–c
/ SOP2–4 paired t-test + McNemar's test comparisons against the Full
Model.

```bash
python -m experiments.run_experiments
```

This takes a while (7 configs × 5 seeds = 35 full training runs). To
sanity-check the whole pipeline quickly before committing to that, run
a tiny subset first:

```bash
python -m experiments.run_experiments --configs full ela_only --seeds 42 --epochs 1
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--configs full ela_only ...` | Subset of configs to run (must include `full`) |
| `--seeds 42 123 ...` | Override the 5 default seeds |
| `--epochs N` | Override epoch count for every run (for quick tests) |
| `--data-root PATH` | Override `config.data.root` for every config |
| `--device cuda\|mps\|cpu` | Force a device |
| `--quiet` | Suppress per-epoch training logs |

**Output:**
```
results/summary.json   # everything, machine-readable
results/summary.md     # human-readable table of results + p-values
checkpoints/<config>/  # every seed's best/last checkpoint, per config
results/<config>/      # every seed's training history + test predictions
```

## Notes

- Every hyperparameter not specified in the research handoff doc
  (learning rate, batch size, epoch count, the 5 random seeds, etc.) is
  filled in with a documented implementation default — search the
  codebase for `NOT SPECIFIED` to find every one, or read
  `AGENT_CHANGES.md` for the full list. Review these against the
  research team's intentions before treating results as final.
- The PRNU branch computes its wavelet residual internally (a learnable
  D4 wavelet layer, `models/wavelet_layer.py`) — you don't need to
  precompute anything for it; feeding it raw images is correct.