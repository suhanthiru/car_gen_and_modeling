"""Re-identification: is this incoming photo the vehicle we already have?

WHY NOT CLASSIC ReID EMBEDDINGS ALONE
-------------------------------------
Standard ReID embeds two crops and compares them. Viewpoint is a dominant
nuisance factor: the front of car A often embeds closer to the front of car B
than to the side of car A. That is *the* angle problem, and no amount of
multi-view training fully removes it — you are asking a 2D network to be
invariant to a 3D transform it cannot observe.

THE 3D ANSWER (phase 2, `verify.py`)
------------------------------------
We hold an explicit 3D asset, so we never have to compare across angles: render
the asset over a sweep of azimuths at the query camera's elevation, degrade the
renders to match the sensor, and compare same-angle vs same-angle in feature
space. Identity is then measured with viewpoint held fixed by construction, and
the argmax azimuth falls out as a free pose initialization.

Two properties embeddings cannot give us:
  * comparison masked to OBSERVED regions — hallucinated geometry never votes,
    so a wrong guess can neither cause a false reject nor a false accept;
  * the flywheel — each confirmation widens the confirmed angle range, which
    makes the *next* camera more likely to match.

PHASE 1 SHIPS the cheap pre-filter tier of that cascade: an appearance embedding
used only to *flag* likely duplicates for the merge pass. It is deliberately
advisory — with `auto_merge` off, a human confirms. Full cascade later:
plate read (unique key + metric scale) → attributes → embedding → render-verify.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IdentityMatch:
    vehicle_id: str
    name: str
    score: float  # cosine similarity in [0, 1]


class Embedder(ABC):
    @abstractmethod
    def embed(self, image_rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Masked vehicle crop → L2-normalized float32 appearance vector."""

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity remapped to [0, 1]."""
        if a is None or b is None or a.size == 0 or b.size == 0:
            return 0.0
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 0:
            return 0.0
        return float(np.clip((float(np.dot(a, b)) / denom + 1.0) / 2.0, 0.0, 1.0))
