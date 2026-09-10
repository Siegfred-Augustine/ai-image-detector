"""
serve/api.py

The thesis's actual demo tool: a small FastAPI server around
serve.inference.Detector. Upload one or more images, get back a
Real / AI-generated prediction from the trained Full Model plus the
analytics needed to explain *why* -- per-branch attention weights, the
ELA map, and the learned PRNU wavelet-residual map.

Run it:
    uvicorn serve.api:app --reload --port 8000

Then point the frontend's Detector page at it (see
frontend/src/pages/Detector.tsx -- default API_BASE is
http://localhost:8000).

Configuration (environment variables):
    CHECKPOINT_PATH   Which trained checkpoint to serve.
                       Default: checkpoints/full/best.pt
    DEVICE            "cuda" | "mps" | "cpu". Default: auto-detect.

The checkpoint is loaded lazily on the first request (not at import
time), so the server starts even before you've trained a model --
you'll just get a clear 503 telling you to train one first, instead of
a crash on boot.
"""

from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from serve.inference import Detector

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/full/best.pt")
DEVICE = os.environ.get("DEVICE")  # None -> auto-detect

app = FastAPI(
    title="AI-Generated Image Detector API",
    description="Serves the trained Multi-Stream CNN (ELA + PRNU + Content, attention fusion).",
)

# NOT specified which origin(s) the frontend will actually be served
# from -- wide open here for local development convenience. Restrict
# `allow_origins` before deploying this anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_detector: Optional[Detector] = None


def get_detector() -> Detector:
    """Lazily load the checkpoint on first use, then reuse it for every subsequent request."""
    global _detector
    if _detector is None:
        if not os.path.exists(CHECKPOINT_PATH):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"No checkpoint found at '{CHECKPOINT_PATH}'. Train the Full Model "
                    f"first: python -m training.train --config config/full.yaml --seed 42 "
                    f"(or set CHECKPOINT_PATH to point at an existing checkpoint)."
                ),
            )
        _detector = Detector(CHECKPOINT_PATH, device=DEVICE)
    return _detector


@app.get("/health")
def health():
    try:
        detector = get_detector()
        return {
            "status": "ok",
            "checkpoint": detector.checkpoint_path,
            "checkpoint_epoch": detector.checkpoint_epoch,
            "checkpoint_seed": detector.checkpoint_seed,
            "active_branches": detector.active_branches,
            "device": str(detector.device),
        }
    except HTTPException as e:
        return {"status": "not_ready", "detail": e.detail}


@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    """
    Accepts one or more image files (multipart/form-data, field name
    "files"). Returns one result per file, in the same order.
    """
    detector = get_detector()
    results = []

    for file in files:
        image_bytes = await file.read()
        try:
            result = detector.predict(image_bytes)
            result["filename"] = file.filename
        except Exception as e:  # keep one bad upload from failing the whole batch
            result = {"filename": file.filename, "error": str(e)}
        results.append(result)

    return {"results": results}
