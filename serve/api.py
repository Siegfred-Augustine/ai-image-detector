"""
serve/api.py

FastAPI wrapper around the image detector and benchmark summary.

Run with: uvicorn serve.api:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from serve.inference import Detector

CHECKPOINT_ROOT = Path(os.environ.get("CHECKPOINT_ROOT", "checkpoints"))
CHECKPOINT_PATH = Path(os.environ.get("CHECKPOINT_PATH", "checkpoints/full/seed42/best.pt"))
DEVICE = os.environ.get("DEVICE")

MODEL_NAMES = {
    "full": "Full Model",
    "prnu_only": "PRNU-only CNN",
    "ela_only": "ELA-only CNN",
    "content_only": "Content-only CNN",
    "no_prnu": "Full Model - PRNU",
    "no_ela": "Full Model - ELA",
    "no_content": "Full Model - Content",
}

app = FastAPI(
    title="AI-Generated Image Detector API",
    description="Serves selectable trained AI-image detection models.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_detector: Optional[Detector] = None
_detector_path: Optional[Path] = None
_detector_lock = threading.RLock()


def available_models() -> List[Dict[str, object]]:
    """List discovered checkpoints without loading model weights."""
    models: List[Dict[str, object]] = []
    for config_name, display_name in MODEL_NAMES.items():
        config_root = CHECKPOINT_ROOT / config_name
        legacy_path = config_root / "best.pt"
        if legacy_path.is_file():
            models.append({
                "id": f"{config_name}:default",
                "name": display_name,
                "config": config_name,
                "seed": None,
                "checkpoint": str(legacy_path),
            })
        for checkpoint in sorted(config_root.glob("seed*/best.pt")):
            seed_name = checkpoint.parent.name.removeprefix("seed")
            if seed_name.isdigit():
                models.append({
                    "id": f"{config_name}:seed{seed_name}",
                    "name": display_name,
                    "config": config_name,
                    "seed": int(seed_name),
                    "checkpoint": str(checkpoint),
                })

    configured_path = CHECKPOINT_PATH
    discovered_paths = {Path(str(model["checkpoint"])) for model in models}
    if configured_path.is_file() and configured_path not in discovered_paths:
        models.append({
            "id": "configured:default",
            "name": "Configured checkpoint",
            "config": "configured",
            "seed": None,
            "checkpoint": str(configured_path),
        })

    return sorted(
        models,
        key=lambda model: (
            str(model["name"]),
            model["seed"] is None,
            model["seed"] if model["seed"] is not None else -1,
        ),
    )


def resolve_model_checkpoint(model_id: str) -> Path:
    for model in available_models():
        if model["id"] == model_id:
            return Path(str(model["checkpoint"]))
    raise HTTPException(status_code=404, detail=f"No trained checkpoint is available for model '{model_id}'.")


def get_detector(model_id: str) -> Detector:
    """Load the requested checkpoint, keeping at most one model resident."""
    global _detector, _detector_path
    checkpoint_path = resolve_model_checkpoint(model_id)
    with _detector_lock:
        if _detector is None or _detector_path != checkpoint_path:
            _detector = Detector(str(checkpoint_path), device=DEVICE)
            _detector_path = checkpoint_path
        return _detector


@app.get("/health")
def health():
    models = available_models()
    return {
        "status": "ok" if models else "not_ready",
        "available_models": len(models),
        "detail": None if models else f"No checkpoints found under '{CHECKPOINT_ROOT}'.",
    }


@app.get("/models")
def models():
    """Return selectable trained checkpoints for the detector UI."""
    return {"models": available_models()}


@app.get("/benchmark")
def benchmark():
    """Serve the experiment summary JSON produced by the experiment runner."""
    summary_path = Path("results/summary.json")
    if not summary_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No benchmark summary found yet. Run the experiment sweep first: "
                "python -m experiments.run_experiments"
            ),
        )

    with summary_path.open("r", encoding="utf-8") as summary_file:
        summary = json.load(summary_file)
    return {"status": "ok", "summary": summary, "path": str(summary_path)}


@app.post("/predict")
async def predict(
    files: List[UploadFile] = File(...),
    model_id: str = Form(...),
):
    """Predict uploaded images using the selected available checkpoint."""
    results = []
    for file in files:
        image_bytes = await file.read()
        try:
            with _detector_lock:
                detector = get_detector(model_id)
                result = detector.predict(image_bytes)
            result["filename"] = file.filename
        except Exception as error:
            result = {"filename": file.filename, "error": str(error)}
        results.append(result)
    return {"results": results}
