"""Registration interface — estimate where a new frame's camera was.

THE LOAD-BEARING CONTRACT OF THE WHOLE SYSTEM.

Registering a photo against a model whose unseen regions are *hallucinated* is
the hard problem: matches against guessed geometry are sparse, wrong, or
ambiguous, and cars are nearly bilaterally symmetric, so front/rear and
left/right confusions are the classic failure. Everything downstream is garbage
if the pose is wrong.

So registration MUST report its own confidence, and the fusion engine MUST
refuse to fuse below threshold. A rejected frame is a non-event; a wrongly
accepted frame corrupts the asset. Implementations must therefore:

  * match only against OBSERVED regions / real source photos, never against
    PRIOR-provenance geometry (see `registration.py`);
  * return `Registration.rejected(...)` rather than a low-quality guess.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from cargen.core.camera import CameraPose, Intrinsics


@dataclass(frozen=True)
class Registration:
    """Result of locating a frame's camera in the asset's canonical frame."""

    pose: CameraPose | None
    confidence: float          # [0, 1]
    inliers: int
    reprojection_error: float  # pixels; inf when unregistered
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.pose is not None

    @staticmethod
    def rejected(reason: str, inliers: int = 0) -> "Registration":
        return Registration(
            pose=None, confidence=0.0, inliers=inliers,
            reprojection_error=float("inf"), reason=reason,
        )

    def accepted_at(self, threshold: float) -> bool:
        return self.ok and self.confidence >= threshold


class Registrar(ABC):
    @abstractmethod
    def register(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray | None,
        intrinsics: Intrinsics,
        context: dict,
    ) -> Registration:
        """Locate this frame's camera in the asset's canonical frame.

        `context` carries what the implementation needs — typically
        {"asset": VehicleAsset, "prev_pose": CameraPose | None,
         "prev_frame": np.ndarray | None}. Kept loose so backends
        (PnP-vs-landmarks now; VGGT/MASt3R pointmap regression later) can each
        take what they need without changing orchestration.
        """
