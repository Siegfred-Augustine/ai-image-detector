# AI-Generated Image Detector: Presentation and Code Guide

This guide explains the behavior implemented in this repository, from image folders through model training and the web demo. Each component pairs its purpose and mechanism with the code that performs it. Code excerpts are shortened to the lines that demonstrate the behavior; the linked source files are authoritative.

## 1. Project in One Minute

The project is a binary image classifier. It labels an image as **real** (class `0`) or **AI-generated** (class `1`). Its full configuration processes each image through three parallel feature streams:

1. **Content:** a CNN learns visual and structural patterns from the RGB image.
2. **ELA:** the image is JPEG-recompressed and the pixel difference is amplified; a CNN learns patterns in that difference map.
3. **PRNU-style residual:** a differentiable wavelet layer denoises the input using learned thresholds; the difference between the image and its reconstruction is encoded by a CNN.

An attention gate combines the three feature vectors, and a classifier produces two logits. Training and experiment scripts compare the full model with single-stream baselines and models with one stream removed. A FastAPI service loads trained checkpoints for uploaded-image predictions, and a React frontend presents both the detector and benchmark dashboard.

**Important terminology:** in the full model, “PRNU” means a learned wavelet-residual proxy. It is not a measured camera fingerprint, and the program does not prove that an image came from a particular camera or generator.

## 2. End-to-End Flow

```text
data/raw/real + data/raw/fake
             |
             v
index files, assign labels, split train/validation/test
             |
             v
load RGB, resize, optionally flip once for all active streams
       /             |               \
      v              v                v
  content RGB       ELA map       minimally processed RGB
      |              |                |
 content CNN       ELA CNN      learned wavelet residual + CNN
       \             |               /
             attention fusion
                    |
             two-class head
                    |
        train / evaluate / predict
```

The central model is assembled in [`models/multistream.py`](../models/multistream.py). The YAML configuration controls which branches are active, so the same model and dataset classes serve the full model, baselines, and ablations.

## 3. Data and Preprocessing

### What it does

The dataset is expected at `data/raw/real/` and `data/raw/fake/`. Supported extensions are JPG, JPEG, PNG, BMP, and WebP. Each image is loaded as RGB; real images receive label `0`, fake/AI-generated images label `1`. The split is stratified by class. In the checked-in full config, images are resized to 224 × 224, with 15% assigned to validation and 15% to test.

For training, a random horizontal flip may be applied to the shared PIL image before individual stream transforms are computed. This ensures content, ELA, and PRNU inputs refer to the same transformed image. The content tensor is ImageNet-normalized. ELA maps and the learnable PRNU input are not ImageNet-normalized.

### How the code does it

[`data/preprocessing.py`](../data/preprocessing.py) indexes files, loads RGB, splits each class, and defines transforms. [`data/dataset.py`](../data/dataset.py) applies the config and creates only the enabled streams.

```python
# data/dataset.py, simplified from AIGeneratedImageDataset.__getitem__
pil_image = center_crop_or_resize(load_image(sample.path), self.image_size, self.center_crop)
if self.random_hflip and torch.rand(1).item() < 0.5:
    pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)

if self.use_content:
    item["content"] = self.content_transform(pil_image)
if self.use_ela:
    ela_map = compute_ela_image(pil_image, quality=self.ela_quality, scale=self.ela_scale)
    item["ela"] = to_unit_tensor(ela_map)
if self.use_prnu:
    item["prnu"] = self.prnu_transform(pil_image)
item["label"] = torch.tensor(sample.label, dtype=torch.long)
```

The full model config is [`config/full.yaml`](../config/full.yaml). Its ELA quality is 95 and scale is 15. `build_content_transform` resizes, converts to a tensor, and applies ImageNet mean/std. `build_prnu_transform` resizes and converts to a tensor in `[0, 1]` without normalization. `to_unit_tensor` converts H×W or H×W×C arrays to channel-first tensors and min-max rescales arrays outside `[-1, 1]`.

## 4. ELA Stream: Recompression-Difference Features

### What ELA can reveal

Error Level Analysis compares an image with a JPEG-recompressed version. The difference map highlights how strongly different pixels change under that particular recompression. The ELA CNN can learn whether spatial patterns in these differences correlate with the training labels. ELA is a feature source, not a forensic verdict: the map alone does not prove editing or AI generation, and results depend on the image's compression history and selected quality.

### How ELA is computed

[`data/ela.py`](../data/ela.py) encodes the RGB image as JPEG at the configured quality, decodes it, computes the absolute per-channel difference, amplifies it, and clips it to the display range.

```python
# data/ela.py, compute_ela_image
image.save(buffer, format="JPEG", quality=quality)
recompressed = Image.open(buffer).convert("RGB")
diff = ImageChops.difference(image, recompressed)
diff_arr = np.asarray(diff, dtype=np.float32)
return np.clip(diff_arr * scale, 0, 255)
```

The map is converted to a tensor and passed to [`models/ela_branch.py`](../models/ela_branch.py). That branch applies two Conv → BatchNorm → ReLU blocks, global average pooling, and a linear projection to the shared 128-dimensional feature vector.

```python
# models/ela_branch.py, forward
x = self.relu1(self.bn1(self.conv1(x)))
x = self.relu2(self.bn2(self.conv2(x)))
x = self.gap(x).flatten(1)
f_ela = self.fc(x)
```

The data function's standalone default quality is 90, but the checked-in full and ELA-only configs pass 95. The ELA branch itself does not calculate ELA; extraction happens upstream in the dataset or serving inference code.

## 5. Content Stream: RGB Structure

### What it contributes

This branch receives the resized RGB image and learns spatial patterns from its visual content. It can represent broad structures, textures, and image artifacts learned from examples. It does not use a pretrained backbone in the standard full configuration; its CNN weights are learned during this project's training.

### How the code does it

[`models/content_branch.py`](../models/content_branch.py) uses three convolution blocks. Each block performs Conv → BatchNorm → ReLU → 2×2 max pooling. Adaptive global average pooling collapses spatial dimensions, then a linear layer projects to the feature dimension.

```python
# models/content_branch.py, standard block and forward path
return self.pool(self.relu(self.bn(self.conv(x))))

x = self.blocks(x)
x = self.gap(x).flatten(1)
f_c = self.fc(x)
```

The standard blocks use 32, 64, and 128 output channels and produce a 128-dimensional vector. [`config/content_baseline.yaml`](../config/content_baseline.yaml) also defines a separate `baseline_mode` architecture with a 64-dimensional feature vector; it is not one of the seven configs in the default experiment sweep.

## 6. PRNU Stream: Learnable Wavelet Residual

### What it contributes, and what it does not

The stream attempts to emphasize high-frequency residual detail rather than ordinary RGB content. This project does that with a differentiable wavelet denoising-and-subtraction layer. Since it does not have known reference fingerprints for camera sensors, it should be called a **PRNU-style** or **wavelet-residual** stream, not validated camera PRNU extraction.

### How the code does it

The standard path starts with minimally processed RGB in `[0, 1]`. [`models/wavelet_layer.py`](../models/wavelet_layer.py) computes a one-level D4 wavelet decomposition into `LL`, `LH`, `HL`, and `HH` subbands. It soft-thresholds each subband, reconstructs a denoised image, then subtracts that reconstruction from the input:

$$
W = X - \operatorname{IDWT}(\operatorname{soft\_threshold}(\operatorname{DWT}(X), \tau))
$$

Each subband has a learned threshold constrained to be nonnegative with softplus. The residual `W` is then encoded by [`models/prnu_branch.py`](../models/prnu_branch.py) using two convolution stages, global average pooling, and a linear projection.

```python
# models/wavelet_layer.py, simplified from forward
subbands = self.dwt(x)
tau = self.tau
thresholded = {
    name: soft_threshold(subbands[name], tau[i])
    for i, name in enumerate(SUBBAND_ORDER)
}
denoised = self.idwt(thresholded, out_hw=(h, w))
residual = x - denoised
return residual, tau.detach()
```

The thresholds participate in the training computation; the returned detached values are for inspection/logging. [`models/prnu_branch.py`](../models/prnu_branch.py) applies the wavelet layer internally when `use_hybrid_wavelet=True`, which is the default. In the checked-in `full.yaml` and `prnu_only.yaml`, no override changes that default.

### Related utilities that are not the standard full-model path

- [`data/classical_prnu.py`](../data/classical_prnu.py) provides a grayscale BayesShrink wavelet-denoising residual. The checked-in `prnu_only.yaml` does not set `prnu.mode: classical`, so this function is not selected by that current config. Its own module documentation also cautions that it is not an exact reproduction of the cited Matlab toolbox.
- [`data/wavelet.py`](../data/wavelet.py) provides offline PyWavelets denoising and detail-map helpers. It is not called by the learnable model path.
- [`data/patches.py`](../data/patches.py) extracts regular-grid or random image patches and can rank them by variance. It is a utility for patch-level experiments, not a step in the default full-image training/inference path.

## 7. Fusion and Classification

### Attention fusion

Each enabled branch emits a 128-dimensional feature vector. For two or more active branches, [`models/attention.py`](../models/attention.py) concatenates their vectors, uses one linear layer to produce one score per branch, applies softmax, and takes the weighted sum. The weights sum to one for each image and can be returned for display.

```python
# models/attention.py, forward
f_concat = torch.cat(features, dim=1)
weights = torch.softmax(self.gate(f_concat), dim=1)
stacked = torch.stack(features, dim=1)
f_fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
```

These weights describe the learned gate's relative weighting for that input; they are not proof of causal importance or a complete explanation of the model. A single-branch configuration bypasses fusion and sends its branch feature directly to the classifier.

### Classifier head

[`models/classifier.py`](../models/classifier.py) maps the fused feature to two logits. The implementation is `Linear → ReLU → Dropout → Linear → Linear`; it does **not** apply softmax inside the module. Training passes logits to cross-entropy loss, while prediction code applies softmax to obtain class probabilities.

```python
# models/classifier.py, forward
x = self.relu(self.fc1(x))
x = self.dropout(x)
x = self.fc2(x)
logits = self.fc3(x)
return logits
```

For the standard configs, the feature dimension is 128, the hidden dimension is 64, dropout is 0.5, and there are two output classes. These dimensions and probabilities are implementation/config choices; a softmax probability should not be described as calibrated confidence unless a calibration procedure is separately applied.

## 8. Model Variants and Configurations

The seven configs in [`experiments/run_experiments.py`](../experiments/run_experiments.py) are:

| Config         | Active streams       | Experiment role                         |
| -------------- | -------------------- | --------------------------------------- |
| `full`         | ELA + PRNU + Content | Main attention-fused model and baseline |
| `prnu_only`    | PRNU                 | Single-stream baseline                  |
| `ela_only`     | ELA                  | Single-stream baseline                  |
| `content_only` | Content              | Single-stream baseline                  |
| `no_prnu`      | ELA + Content        | Full model with PRNU removed            |
| `no_ela`       | PRNU + Content       | Full model with ELA removed             |
| `no_content`   | ELA + PRNU           | Full model with Content removed         |

The model factory in `MultiStreamModel.from_config` builds only the configured branches, preserves a consistent ELA → PRNU → Content ordering, and enables fusion only if at least two branches are active. The shared dataset class similarly computes only enabled streams. This makes each variant an actual retrained architecture, rather than a full model with a display-only switch.

The extra `content_baseline` config is a legacy/alternate content-only architecture and is not part of the default seven-config sweep.

## 9. Training and Checkpoints

### Training behavior

[`training/train.py`](../training/train.py) constructs the loaders and model from YAML and trains one config. In the checked-in full config, the training defaults include:

- AdamW, learning rate `0.0001`, weight decay `0.0001`
- batch size 32, up to 50 epochs
- cross-entropy with label smoothing `0.1`
- cosine learning-rate decay down to `0.000001`
- gradient norm clipping at `1.0`
- early stopping after 7 epochs without validation-F1 improvement

These are values in the current config, not universal properties of the network. The source also supports an SGD-with-momentum option, but the full config uses AdamW. Device selection prefers CUDA, then Apple MPS, then CPU unless a device is explicitly selected.

```python
# training/train.py, essential update sequence
logits, _ = model(inputs)
loss = criterion(logits, labels)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
optimizer.step()
```

The best checkpoint is selected by validation F1. `best.pt` and `last.pt` each contain model state and the config used to build it; `best.pt` additionally records its best validation F1. Per-epoch loss and validation metrics are written to a history JSON. Experiment seeds affect initialization and stochastic training behavior. The dataset split uses `data.seed` from config, which is fixed at 42 in the checked-in configs so models can be compared on the same images.

## 10. Evaluation and Statistical Comparisons

[`training/evaluate.py`](../training/evaluate.py) reloads a checkpoint and its embedded config, evaluates train/validation/test data, and can save per-image predictions and probabilities. [`training/metrics.py`](../training/metrics.py) calculates precision, recall, F1, accuracy, and the confusion matrix with AI-generated as the positive class:

- **TP:** AI image predicted AI
- **FN:** AI image predicted real (the project calls this the dangerous miss)
- **FP:** real image predicted AI
- **TN:** real image predicted real

[`experiments/run_experiments.py`](../experiments/run_experiments.py) defaults to seven configs × three model seeds (`42`, `123`, and `2024`). It checks that the compared configs share the same split settings before training, evaluates each selected checkpoint on test, then computes:

- Mean and sample standard deviation across seeded runs for precision, recall, F1, and accuracy.
- Paired t-tests across seed-matched run-level precision, recall, and F1.
- McNemar tests comparing image-level correctness for each matched seed, because the two models are evaluated on the same test images.

The standard sweep writes `results/summary.json`, `results/summary.md`, per-run histories/evaluations, and checkpoint folders. [`experiments/build_benchmark_from_checkpoints.py`](../experiments/build_benchmark_from_checkpoints.py) is an alternate route: it finds existing checkpoints, evaluates them without retraining, and creates the same summary format.

Statistical outputs depend on valid pairing and distinct runs. A small p-value is not evidence of broad real-world generalization, and the experiment code does not establish performance on unseen generators, cameras, or image domains.

## 11. Serving and Web Interface

### Inference

[`serve/inference.py`](../serve/inference.py) defines `Detector`. It loads a checkpoint, reconstructs the model from the saved config, preprocesses an uploaded image using the configured size/ELA settings, executes active branches, and returns the predicted label, both probabilities, branch weights where applicable, checkpoint metadata, and visualizations.

The ELA map is encoded as a PNG. For PRNU, inference asks the branch for its residual and learned thresholds; the residual is converted to a per-image grayscale min-max visualization. This rendering is for inspection, not a calibrated forensic scale. A displayed prediction confidence is the larger of the two softmax probabilities, without a separate calibration step.

### HTTP API

[`serve/api.py`](../serve/api.py) exposes:

| Route            | Purpose                                                                  |
| ---------------- | ------------------------------------------------------------------------ |
| `GET /health`    | Report readiness and the number of discovered checkpoints                |
| `GET /models`    | List available `best.pt` checkpoints for the selector                    |
| `GET /benchmark` | Return `results/summary.json` or a 404 if it is absent                   |
| `POST /predict`  | Accept multipart image files plus `model_id`; return one result per file |

The backend discovers legacy and per-seed checkpoints under the configured checkpoint root. It loads the requested model on demand and keeps at most one detector resident, protected by a lock. The default checkpoint is `checkpoints/full/seed42/best.pt`; environment variables can override checkpoint root, checkpoint path, and device. CORS is configured to allow all origins, methods, and headers, so deployment should tighten that policy as appropriate.

### React frontend

[`frontend/src/App.tsx`](../frontend/src/App.tsx) switches between the Detector and Benchmark pages.

- [`frontend/src/pages/Detector.tsx`](../frontend/src/pages/Detector.tsx) checks backend health and fetches the model list, accepts one or multiple uploaded images, submits them to `/predict`, and displays label, class probabilities, branch weights, ELA/PRNU visualizations, and checkpoint metadata. The UI also tracks upload states and supports clearing the queue.
- Detector **Demo mode** fabricates random labels, probabilities, attention weights, and noise-like pictures entirely in the browser. It tests the presentation UI only; it does not run the classifier.
- [`frontend/src/pages/Benchmark.tsx`](../frontend/src/pages/Benchmark.tsx) fetches `/benchmark`, groups models into single-stream comparisons and ablations, displays precision/recall/F1 and significance badges, plots mean F1, and shows a selectable pooled confusion matrix.
- If `/benchmark` cannot provide a summary, the Benchmark page uses its local placeholder `MODEL_DATA`. Its preview must not be presented as a live experiment result. The page shows a status message describing whether it loaded the generated summary or fell back to placeholders.

## 12. Current Results and Presentation Caveats

The checked-in [`results/summary.md`](../results/summary.md) reports the full model at approximately **0.7909 ± 0.1072 F1** over 750 test examples. Treat this as a snapshot of the checked-in artifact, not a verified fresh sweep. The accompanying JSON records seeds `[42, 123, 2024, 42]`, so seed 42 appears twice; the aggregate therefore has four records rather than the default three. Resolve which checkpoint set should be included and regenerate the summary before making a definitive quantitative claim.

Other points to state accurately during a presentation:

- The checked-in configs use a 15% test split. The benchmark page's descriptive paragraph says “2,500 real + 2,500 AI-generated” test images, but the saved summary says `n_test_images: 750`; those statements conflict. The UI text should not be treated as the verified test count.
- ELA measures JPEG recompression differences; the stream learns associations in those differences and does not independently prove manipulation.
- The full model's PRNU branch is the differentiable wavelet-residual proxy, not a camera fingerprint matcher.
- Attention weights are learned mixture weights, not causal explanations.
- Softmax scores are not shown to be calibrated probabilities.
- The benchmark's fallback metrics and detector's Demo mode are fabricated preview data, not measured predictions.
- The current summary only reflects the specific dataset, split, seeds, and checkpoints represented in that file. It does not by itself demonstrate robustness on new generators or real-world images.

## 13. Suggested Presentation Sequence

1. **Problem and output:** binary classification, labels, and what the tool can and cannot claim.
2. **Three complementary views:** content RGB, JPEG recompression difference (ELA), and learned wavelet residual.
3. **Architecture:** one feature extractor per active stream; attention-weighted fusion; two-class head.
4. **Experimental design:** full model, three single-stream baselines, three leave-one-stream-out ablations; same data split across runs.
5. **Training and evaluation:** validation-F1 checkpoint selection, test metrics, seed variability, paired tests, and confusion-matrix false negatives.
6. **Working demo:** select a real checkpoint, upload images, inspect outputs; verify the API is connected and Demo mode is off.
7. **Limitations and next steps:** resolve duplicate seed and test-count mismatch, regenerate the benchmark, then test external domains and probability calibration.

### Short talk track

“This system classifies images as real or AI-generated by combining three learned views of the same image. A content CNN reads RGB structure, an ELA CNN reads JPEG recompression differences, and a wavelet branch learns a residual representation. When multiple branches are active, an attention gate weights their feature vectors before a two-class classifier. We train full, single-stream, and leave-one-stream-out models on matched data splits, then compare both aggregate metrics and per-image disagreements. The web app can show checkpoint-backed predictions and experiment summaries, but its preview modes are synthetic, and the current saved results need a clean rerun before being treated as final evidence.”

## 14. Source Map

| Area                                         | Main implementation                                                                                                                                                                                   |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset indexing, splits, transforms         | [`data/preprocessing.py`](../data/preprocessing.py), [`data/dataset.py`](../data/dataset.py)                                                                                                          |
| ELA map                                      | [`data/ela.py`](../data/ela.py)                                                                                                                                                                       |
| Content, ELA, PRNU branches                  | [`models/content_branch.py`](../models/content_branch.py), [`models/ela_branch.py`](../models/ela_branch.py), [`models/prnu_branch.py`](../models/prnu_branch.py)                                     |
| Learnable wavelet residual                   | [`models/wavelet_layer.py`](../models/wavelet_layer.py)                                                                                                                                               |
| Fusion, classifier, config assembly          | [`models/attention.py`](../models/attention.py), [`models/classifier.py`](../models/classifier.py), [`models/multistream.py`](../models/multistream.py)                                               |
| Training, metrics, evaluation                | [`training/train.py`](../training/train.py), [`training/metrics.py`](../training/metrics.py), [`training/evaluate.py`](../training/evaluate.py)                                                       |
| Experiment sweep and checkpoint-only summary | [`experiments/run_experiments.py`](../experiments/run_experiments.py), [`experiments/build_benchmark_from_checkpoints.py`](../experiments/build_benchmark_from_checkpoints.py)                        |
| API and inference                            | [`serve/api.py`](../serve/api.py), [`serve/inference.py`](../serve/inference.py)                                                                                                                      |
| Frontend                                     | [`frontend/src/App.tsx`](../frontend/src/App.tsx), [`frontend/src/pages/Detector.tsx`](../frontend/src/pages/Detector.tsx), [`frontend/src/pages/Benchmark.tsx`](../frontend/src/pages/Benchmark.tsx) |
| Dataset preparation helper                   | [`scripts/prepare_class_folder.py`](../scripts/prepare_class_folder.py)                                                                                                                               |
