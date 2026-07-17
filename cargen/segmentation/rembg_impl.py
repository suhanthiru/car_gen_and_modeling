"""Real segmentation backend: rembg (U2-Net family / BiRefNet, ONNX runtime).

INTEGRATION POINT
-----------------
Install:   pip install rembg pillow           (CPU is fine; onnxruntime-gpu optional)
Weights:   auto-downloaded on first use to ~/.u2net/
Models:    "isnet-general-use" (default here — best object cutouts for cluttered
           outdoor photos), "u2net" (classic), "birefnet-general" (highest quality,
           slower). Configure via constructor.
VRAM:      none required (CPU inference ~1-3 s at 1280 px).

Contract matches Segmenter: RGB uint8 in, float32 [0,1] vehicle mask out.
"""
from __future__ import annotations

import numpy as np

from cargen.segmentation.interface import Segmenter


class RembgSegmenter(Segmenter):
    def __init__(self, model_name: str = "isnet-general-use"):
        from rembg import new_session  # lazy: heavy import + weight download

        self._session = new_session(model_name)

    def segment(self, image_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image
        from rembg import remove

        pil = Image.fromarray(image_rgb)
        result = remove(pil, session=self._session, only_mask=True)
        mask = np.asarray(result, np.float32) / 255.0
        return mask
