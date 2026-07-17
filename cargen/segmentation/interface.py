"""Vehicle segmentation interface.

Contract: `segment` takes an RGB uint8 image (H, W, 3) and returns a float32
mask (H, W) in [0, 1] where 1 = vehicle. Downstream stages threshold at 0.5.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Segmenter(ABC):
    @abstractmethod
    def segment(self, image_rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 (H,W,3) → float32 mask (H,W) in [0,1], 1 = vehicle."""
