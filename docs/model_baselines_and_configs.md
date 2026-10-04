# Model Baselines and Configuration Reference

This document is the project-level reference for the three single-branch baselines and the full multi-stream model. It is intended to support thesis-level comparison, ablation analysis, and fair benchmarking.

## 1. Model family overview

The project uses a common `MultiStreamModel` class that builds different forensic configurations from the same architecture family:

- `content` branch: raw-image content CNN
- `ela` branch: ELA-based compression artifact CNN
- `prnu` branch: wavelet residual / PRNU CNN
- `full` model: all three branches + attention fusion
- ablations: `no_content`, `no_ela`, `no_prnu`
- single-branch baselines: `content_only`, `ela_only`, `prnu_only`

The canonical order is fixed as:

- `ela`
- `prnu`
- `content`

This is defined in `models/multistream.py` and ensures feature concatenation and attention weights remain consistent across runs.

---

## 2. Shared architecture and training setup

These settings are common across the repository unless a config overrides them.

### Shared model contract

- Feature dimension: `feature_dim = 128` by default
- Classifier hidden dimension: `classifier_hidden_dim = 64`
- Dropout: `0.5`
- Task: binary classification (`num_classes = 2`)
- Fusion: `AttentionFusion` when more than one branch is active
- Single-branch configs bypass fusion entirely and send the feature vector directly to the classifier

### Shared dataset / split contract

All configs should use the same data split to enable fair statistical comparisons:

- `data.root`: `./data/raw`
- `data.image_size`: `224`
- `data.val_ratio`: `0.15`
- `data.test_ratio`: `0.15`
- `data.seed`: `42`
- the same seed must be used across all configs for paired comparisons

Expected folder layout:

```text
data/raw/real/*.jpg
data/raw/fake/*.jpg
```

### Shared training defaults

- learning rate: `0.0001`
- weight decay: `0.0001`
- batch size: `32`
- epochs: `30`
- LR scheduler floor: `0.000001`
- optimizer betas: `[0.9, 0.999]`
- label smoothing: `0.1`
- grad clip norm: `1.0`
- early stopping patience: `7`
- workers: `4`

These are implementation defaults; the project documents them as defaults for a runnable pipeline, not as final research-approved values.

---

## 3. Configuration matrix

| Model                     | Config file                    | Active branches          | Input type                | Use case                            |
| ------------------------- | ------------------------------ | ------------------------ | ------------------------- | ----------------------------------- |
| Full model                | `config/full.yaml`             | `ela`, `prnu`, `content` | all three streams         | primary thesis model                |
| ELA baseline              | `config/ela_only.yaml`         | `ela`                    | ELA image                 | forensic compression baseline       |
| PRNU baseline             | `config/prnu_only.yaml`        | `prnu`                   | raw image / PRNU residual | sensor-noise baseline               |
| Content baseline          | `config/content_only.yaml`     | `content`                | raw image                 | raw-content baseline                |
| Content baseline (simple) | `config/content_baseline.yaml` | `content`                | raw image                 | classical content-only CNN baseline |
| No PRNU ablation          | `config/no_prnu.yaml`          | `ela`, `content`         | ELA + content             | isolate PRNU contribution           |
| No ELA ablation           | `config/no_ela.yaml`           | `prnu`, `content`        | PRNU + content            | isolate ELA contribution            |
| No content ablation       | `config/no_content.yaml`       | `prnu`, `ela`            | PRNU + ELA                | isolate content contribution        |

---

## 4. Single-branch baseline models

These are the three foundational baselines for your thesis comparison.

### 4.1 ELA-only baseline

Config: `config/ela_only.yaml`

- Active branches: `['ela']`
- Branch type: `ELABranch`
- Input: precomputed ELA image
- ELA quality: `95`
- ELA scale: `15.0`
- Data stream flags: `content: false`, `ela: true`, `prnu: false`

Branch architecture:

```text
Conv1 (3 -> 32, 3x3, padding=1)
-> BatchNorm
-> ReLU
-> Conv2 (32 -> 64, 3x3, padding=1)
-> BatchNorm
-> ReLU
-> GAP
-> Linear(64, 128)
```

Remarks:

- This baseline tests whether JPEG compression artifacts alone are sufficient to discriminate real vs AI-generated images.
- It is a direct compression-artifact detector rather than a richer multi-stream model.
- This is one of the clearest baselines for a thesis table.

### 4.2 PRNU-only baseline

Config: `config/prnu_only.yaml`

- Active branches: `['prnu']`
- Branch type: `PRNUBranch`
- Input: raw image tensor in the default hybrid mode
- Data stream flags: `content: false`, `ela: false`, `prnu: true`

Branch architecture (default learnable mode):

```text
Raw image
-> HybridWaveletLayer (D4 DWT, learnable soft threshold, IDWT)
-> residual W = X - D
-> Conv1 (3 -> 32, 3x3, padding=1)
-> ReLU
-> Conv2 (32 -> 64, 3x3, padding=1)
-> BatchNorm
-> ReLU
-> GAP
-> Linear(64, 128)
```

Important note:

- `PRNUBranch` has a special `use_hybrid_wavelet=False` path for classical baseline mode
- in that setting, the input is assumed to already be a precomputed residual and the learnable wavelet layer is skipped
- the project includes a classical PRNU path for comparison to published sensor-noise baselines

Remarks:

- This baseline isolates the sensor-noise and residual signal without the content or ELA streams.
- It is especially important for understanding whether the full model is capitalizing on PRNU-style artifacts or just content cues.

### 4.3 Content-only baseline

Config: `config/content_only.yaml`

- Active branches: `['content']`
- Branch type: `ContentBranch`
- Input: raw content-preprocessed image
- Data stream flags: `content: true`, `ela: false`, `prnu: false`

Default content branch architecture:

```text
ConvBlock1: Conv(3 -> 32) -> BN -> ReLU -> MaxPool
ConvBlock2: Conv(32 -> 64) -> BN -> ReLU -> MaxPool
ConvBlock3: Conv(64 -> 128) -> BN -> ReLU -> MaxPool
-> GAP
-> Linear(128, 128)
```

This is the main “single-stream content” baseline already implemented by the project.

### 4.4 Classical content baseline (separate comparison mode)

Config: `config/content_baseline.yaml`

- Active branches: `['content']`
- Branch type: `ContentBranch` with `baseline_mode: true`
- Purpose: a simpler raw RGB CNN baseline for comparison against classical image-forensics-style detectors

Simplified architecture:

```text
Conv(3 -> 32, 5x5, padding=2)
-> ReLU
-> MaxPool
-> Conv(32 -> 64, 3x3, padding=1)
-> ReLU
-> MaxPool
-> Conv(64 -> 128, 3x3, padding=1)
-> ReLU
-> MaxPool
-> GAP
-> Linear(128, 64)
```

This branch is intentionally distinct from the main thesis content branch so you can report both:

1. the project’s raw-content baseline
2. a literature-aligned simpler content CNN baseline

This is useful for a stronger thesis narrative if you want to compare against generic CNN-based generated-image detectors.

---

## 5. Full model and ablations

### 5.1 Full model

Config: `config/full.yaml`

- Active branches: `['ela', 'prnu', 'content']`
- Input: ELA map, raw PRNU input, and content image
- Fusion type: `AttentionFusion`
- Fusion output: weighted sum of branch features
- Final classifier: `Classifier(feature_dim=128, hidden_dim=64, dropout=0.5, num_classes=2)`

The attention gate is implemented as:

```text
concat(f_ela, f_prnu, f_content) -> Linear(3*n, 3) -> softmax -> weights
f_fused = sum_i weight_i * f_i
```

This is the reference model in the project and the main target for the thesis claim.

### 5.2 No PRNU ablation

Config: `config/no_prnu.yaml`

- Active branches: `['ela', 'content']`
- Purpose: measures the contribution of the PRNU stream

### 5.3 No ELA ablation

Config: `config/no_ela.yaml`

- Active branches: `['prnu', 'content']`
- Purpose: measures the contribution of the ELA stream

### 5.4 No content ablation

Config: `config/no_content.yaml`

- Active branches: `['prnu', 'ela']`
- Purpose: measures the contribution of the content stream

---

## 6. Branch-specific notes

### ELA branch

- Intended to learn compression artifacts from ELA preprocessing
- Uses `JPEG recompression at 95% quality` as the key ELA setup
- Uses a simple 2-layer CNN after precomputation

### PRNU branch

- Intended to learn residual noise patterns from the image sensor
- Default mode computes a learnable wavelet residual inside the model
- This is a key thesis novelty because the wavelet threshold is trainable end-to-end
- The classical mode exists for direct baseline comparison against non-learnable PRNU methods

### Content branch

- Intended to learn semantic / structural / generation traces directly from pixels
- The default implementation is a deeper CNN than the ELA and PRNU branches
- The project also includes a separate simpler content baseline mode for comparison to raw-image CNN baselines from the literature

---

## 7. Recommended thesis comparison structure

For a clean thesis chapter, present the models in this order:

1. `ela_only` — compression artifact baseline
2. `prnu_only` — sensor-noise baseline
3. `content_only` — raw-content baseline
4. `content_baseline` — simplified literature-style raw-content CNN baseline (optional but recommended for discussion)
5. `no_prnu`, `no_ela`, `no_content` — ablation analyses
6. `full` — proposed model

This structure makes the story very clear:

- singular branch baselines show whether each forensic signal is independently useful
- ablation models show which signal matters most when combined
- full model shows the final proposed architecture

---

## 8. Training and benchmarking commands

Train a single model:

```bash
python -m training.train --config config/ela_only.yaml --seed 42
python -m training.train --config config/prnu_only.yaml --seed 42
python -m training.train --config config/content_only.yaml --seed 42
python -m training.train --config config/content_baseline.yaml --seed 42
python -m training.train --config config/full.yaml --seed 42
```

Run the full experimental sweep:

```bash
python -m experiments.run_experiments
```

Evaluate a checkpoint:

```bash
python -m training.evaluate --checkpoint checkpoints/full/best.pt --split test --output results/full/test_eval.json
```

---

## 9. Summary

The repository already contains a strong baseline structure:

- `ela_only` = ELA single-branch baseline
- `prnu_only` = PRNU single-branch baseline
- `content_only` = content single-branch baseline
- `content_baseline` = simpler classical raw-content detector for literature comparison
- `full` + 3 ablations = full experimental matrix

This gives you a thesis-ready comparison stack without having to invent additional model families.
