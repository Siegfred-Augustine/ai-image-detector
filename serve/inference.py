"""
serve/inference.py

Core inference logic for the thesis's actual demo tool: load the
trained Full Model (or any single-config checkpoint) once, then run
predictions on uploaded images, returning not just a label but the
analytics needed to explain it -- per-branch attention weights, the
ELA map, and the learned PRNU wavelet-residual map.

Kept separate from serve/api.py (the FastAPI wrapper) so it can be
imported and tested directly without spinning up a server:

    from serve.inference import Detector
    d = Detector("checkpoints/full/best.pt")
    result = d.predict(open("some_image.jpg", "rb").read())
"""

from __future__ import annotations

import base64
import io
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from data.ela import compute_ela_image
from data.preprocessing import build_content_transform, build_prnu_transform, to_unit_tensor
from models.multistream import MultiStreamModel
from models.wavelet_layer import SUBBAND_ORDER
from training.train import resolve_device

# Handoff Section 1: 0 = Real, 1 = AI-generated.
LABELS = {0: "Real", 1: "AI-generated"}


def _array_to_png_base64(arr: np.ndarray) -> str:
    """arr: HxWx3 float array in [0, 255] -> base64-encoded PNG string (for embedding in JSON/<img src>)."""
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _residual_to_png_base64(residual: torch.Tensor) -> str:
    """
    residual: (C, H, W) wavelet residual tensor (small values centered
    around 0 -- NOT a displayable image on its own).

    Rendered as a single-channel heatmap for the UI: mean over color
    channels, then min-max normalized to [0, 255]. This normalization
    is purely for visualization -- the model itself consumes the raw
    (unnormalized) residual amplitude, per Handoff Section 4/7.
    """
    gray = residual.mean(dim=0).detach().cpu().numpy()
    lo, hi = float(gray.min()), float(gray.max())
    if hi - lo < 1e-8:
        norm = np.zeros_like(gray)
    else:
        norm = (gray - lo) / (hi - lo)
    img = Image.fromarray((norm * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Detector:
    """
    Loads a trained checkpoint once (expensive) and serves repeated
    `predict()` calls (cheap) -- instantiate this a single time at
    server startup, not per-request.
    """

    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        self.device = resolve_device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.config = ckpt["config"]

        self.model = MultiStreamModel.from_config(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        data_cfg = self.config.get("data", {})
        self.image_size = data_cfg.get("image_size", 224)

        ela_cfg = self.config.get("ela", {})
        self.ela_quality = ela_cfg.get("quality", 95)
        self.ela_scale = ela_cfg.get("scale", 15.0)

        self.content_transform = build_content_transform(self.image_size)
        self.prnu_transform = build_prnu_transform(self.image_size)

        self.active_branches = self.model.active_branches
        self.checkpoint_path = str(checkpoint_path)
        self.checkpoint_epoch = ckpt.get("epoch")
        self.checkpoint_seed = ckpt.get("seed")

    @torch.no_grad()
    def predict(self, image_bytes: bytes) -> Dict:
        """
        Args:
            image_bytes: raw bytes of an uploaded image file (jpg/png/etc).

        Returns a JSON-serializable dict:
            {
              "label": "AI-generated" | "Real",
              "label_index": 0 | 1,
              "confidence": float,                 # P(predicted label)
              "probabilities": {"Real": p0, "AI-generated": p1},
              "attention_weights": {branch: weight, ...} | None,
              "ela_image_base64": "..." | None,     # PNG, base64
              "prnu_residual_image_base64": "..." | None,   # PNG, base64
              "wavelet_tau": {"LL": t, "LH": t, "HL": t, "HH": t} | None,
              "active_branches": [...],
              "checkpoint": {"path": ..., "epoch": ..., "seed": ...},
            }
        """
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_image = pil_image.resize((self.image_size, self.image_size))

        inputs = {}
        ela_map = None

        if "content" in self.active_branches:
            inputs["content"] = self.content_transform(pil_image).unsqueeze(0).to(self.device)

        if "ela" in self.active_branches:
            ela_map = compute_ela_image(pil_image, quality=self.ela_quality, scale=self.ela_scale)
            inputs["ela"] = to_unit_tensor(ela_map).unsqueeze(0).to(self.device)

        if "prnu" in self.active_branches:
            inputs["prnu"] = self.prnu_transform(pil_image).unsqueeze(0).to(self.device)

        logits, attn = self.model(inputs)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax().item())
        confidence = float(probs[pred_idx].item())

        attention_weights = None
        if attn is not None:
            attn_values = attn[0].detach().cpu().tolist()
            attention_weights = {name: float(w) for name, w in zip(self.active_branches, attn_values)}

        ela_b64 = _array_to_png_base64(ela_map) if ela_map is not None else None

        prnu_b64 = None
        tau_dict = None
        if "prnu" in self.active_branches:
            _, (residual, tau) = self.model.branches["prnu"](inputs["prnu"], return_residual=True)
            prnu_b64 = _residual_to_png_base64(residual[0])
            tau_dict = {name: float(t) for name, t in zip(SUBBAND_ORDER, tau.detach().cpu().tolist())}

        return {
            "label": LABELS[pred_idx],
            "label_index": pred_idx,
            "confidence": confidence,
            "probabilities": {LABELS[0]: float(probs[0].item()), LABELS[1]: float(probs[1].item())},
            "attention_weights": attention_weights,
            "ela_image_base64": ela_b64,
            "prnu_residual_image_base64": prnu_b64,
            "wavelet_tau": tau_dict,
            "active_branches": self.active_branches,
            "checkpoint": {
                "path": self.checkpoint_path,
                "epoch": self.checkpoint_epoch,
                "seed": self.checkpoint_seed,
            },
        }
