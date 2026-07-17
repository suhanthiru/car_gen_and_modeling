"""Color-histogram embedder — the dependency-free duplicate-flag baseline.

Deliberately weak: it captures paint colour and coarse tonal layout, which is
enough to *flag* candidate duplicates for the merge pass but nowhere near enough
to confirm identity. That split is the point — flagging is advisory and
(with auto_merge off) human-confirmed, so a weak signal here is safe. Identity
confirmation is the render-based verifier's job.

Swap in DINOv2 (`dino_embed.py`) at Milestone A for a much stronger signal.
"""
from __future__ import annotations

import cv2
import numpy as np

from cargen.reid.interface import Embedder


class HistogramEmbedder(Embedder):
    """Hue-saturation histogram over vehicle pixels, plus a coarse tonal grid."""

    def __init__(self, h_bins: int = 24, s_bins: int = 8, grid: int = 4):
        self._h_bins = h_bins
        self._s_bins = s_bins
        self._grid = grid

    def embed(self, image_rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        cv_mask = None
        if mask is not None:
            cv_mask = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
            if not cv_mask.any():
                return np.zeros(self._h_bins * self._s_bins + self._grid**2, np.float32)

        hist = cv2.calcHist(
            [hsv], [0, 1], cv_mask, [self._h_bins, self._s_bins], [0, 180, 0, 256]
        ).ravel()
        hist = hist / max(float(hist.sum()), 1e-6)

        # coarse spatial value grid — some shape/tonal layout signal
        value = hsv[..., 2].astype(np.float32) / 255.0
        if mask is not None:
            value = value * np.clip(mask, 0, 1)
        cells = cv2.resize(value, (self._grid, self._grid), interpolation=cv2.INTER_AREA).ravel()

        vector = np.concatenate([hist, cells]).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector
