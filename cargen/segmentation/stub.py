"""Stub segmenter: assumes the vehicle fills the central region of the frame."""
from __future__ import annotations

import numpy as np

from cargen.segmentation.interface import Segmenter


class StubSegmenter(Segmenter):
    """Returns a centered rectangle mask covering `coverage` of each dimension."""

    def __init__(self, coverage: float = 0.8):
        self.coverage = coverage

    def segment(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), np.float32)
        mh = int(h * (1 - self.coverage) / 2)
        mw = int(w * (1 - self.coverage) / 2)
        mask[mh : h - mh, mw : w - mw] = 1.0
        return mask
