"""Chain of registrars: cheap and precise first, expensive and robust last.

PnP is fast but needs keypoint overlap; render-based re-ID (`render_reid.py`)
is far more expensive (it rasterizes many candidate poses) but tolerates much
larger viewpoint gaps. Trying PnP first keeps the common case — a walk-around
video's consecutive, heavily-overlapping frames — cheap, and only pays for the
expensive search on the frames that actually need it.
"""
from __future__ import annotations

import numpy as np

from cargen.core.camera import Intrinsics
from cargen.pose_estimation.interface import Registration, Registrar


class FallbackRegistrar(Registrar):
    def __init__(self, registrars: list[Registrar], min_confidence: float = 0.35):
        if not registrars:
            raise ValueError("FallbackRegistrar needs at least one registrar")
        self._registrars = registrars
        self._min_confidence = min_confidence

    def register(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray | None,
        intrinsics: Intrinsics,
        context: dict,
    ) -> Registration:
        best = Registration.rejected("no registrar in the chain was tried")
        for registrar in self._registrars:
            result = registrar.register(image_rgb, mask, intrinsics, context)
            if result.ok and result.confidence >= self._min_confidence:
                return result  # good enough — stop before paying for the next, pricier one
            if result.confidence > best.confidence:
                best = result
        return best  # everyone rejected; report whichever came closest
