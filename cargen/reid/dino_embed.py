"""Real appearance embedder: DINOv2 (Meta) via torch.hub.

INTEGRATION POINT
-----------------
Install:   pip install torch torchvision   (weights auto-download, ~90 MB for ViT-S/14)
License:   Apache-2.0 (DINOv2 code + weights).
VRAM:      <1 GB for ViT-S/14 — negligible alongside other stages.

Why DINOv2 rather than a ReID-specific model: its features are strongly
semantic and notably robust to lighting/exposure change, which is what the
duplicate-flag pass and (later) the render-based verifier both need — the
verifier compares a *rendered* view against a *photographed* one, a domain gap
that colour-space metrics handle badly and DINOv2 handles well.

Upgrade path for the full cascade: a VeRi-776-trained ReID head for viewpoint
robustness, plus ALPR for plate reads.
"""
from __future__ import annotations

import numpy as np

from cargen.reid.interface import Embedder

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoEmbedder(Embedder):
    def __init__(self, model_name: str = "dinov2_vits14", device: str = "cuda",
                 image_size: int = 224):
        import torch

        self._torch = torch
        self._device = device if torch.cuda.is_available() else "cpu"
        self._model = torch.hub.load("facebookresearch/dinov2", model_name)
        self._model = self._model.eval().to(self._device)
        # DINOv2 needs dimensions divisible by its patch size (14)
        self._image_size = (image_size // 14) * 14

    def embed(self, image_rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        import cv2

        img = image_rgb.astype(np.float32) / 255.0
        if mask is not None:
            # zero the background so the embedding describes the vehicle, not the street
            img = img * np.clip(mask, 0, 1)[..., None]
        img = cv2.resize(img, (self._image_size, self._image_size), interpolation=cv2.INTER_AREA)
        img = (img - np.asarray(_IMAGENET_MEAN, np.float32)) / np.asarray(_IMAGENET_STD, np.float32)

        tensor = self._torch.from_numpy(img).permute(2, 0, 1)[None].to(self._device)
        with self._torch.no_grad():
            features = self._model(tensor)
        vector = features.squeeze(0).float().cpu().numpy()
        norm = float(np.linalg.norm(vector))
        return (vector / norm if norm > 0 else vector).astype(np.float32)
