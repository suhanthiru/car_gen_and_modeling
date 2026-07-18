"""Registrar backed by the render-based verifier (`cargen/reid/verify.py`).

PnP (`registration.py`) needs sparse 2D-3D keypoint correspondences, which
don't exist when a photo shares too little distinct visual structure with
anything already confirmed — the classic case is a new angle more than a
frame-to-frame video's worth of rotation away from the last confirmed view.
This registrar instead treats "where is the camera" as a coarse search over
candidate poses, scored by how well a *render* of the current model from each
candidate matches the photo — a holistic comparison that survives far more
viewpoint change than local keypoint matching.

It is deliberately not the default first choice: rendering N candidate poses
is much more expensive than one PnP solve. See `FallbackRegistrar`
(`fallback.py`), which tries PnP first and only reaches for this when PnP
comes back empty-handed.
"""
from __future__ import annotations

import numpy as np

from cargen.core.camera import Intrinsics
from cargen.pose_estimation.interface import Registration, Registrar
from cargen.reid.verify import RenderVerifier


class RenderReidRegistrar(Registrar):
    def __init__(self, verifier: RenderVerifier, min_score: float = 0.55):
        self._verifier = verifier
        self._min_score = min_score

    def register(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray | None,
        intrinsics: Intrinsics,
        context: dict,
    ) -> Registration:
        asset = context.get("asset")
        cloud = getattr(asset, "cloud", None)
        if cloud is None or cloud.n == 0:
            return Registration.rejected("no model to search against")

        result = self._verifier.verify(cloud, image_rgb, mask, intrinsics)
        if result.pose is None:
            return Registration.rejected(
                "no candidate angle had enough confirmed geometry to compare against"
            )
        if result.ambiguous:
            return Registration.rejected(
                f"best match (score {result.score:.2f}) is ambiguous with a "
                "far-away angle — likely front/rear or left/right symmetry"
            )
        if result.score < self._min_score:
            return Registration.rejected(
                f"best render match too weak (score {result.score:.2f} < {self._min_score:.2f})"
            )

        return Registration(
            pose=result.pose,
            confidence=result.score,
            inliers=result.observed_px,
            # not a reprojection error in pixels (this method has no keypoint
            # correspondences) — reported here only as an informational
            # "how far from a perfect match" proxy for logs/observation metadata
            reprojection_error=1.0 - result.score,
            reason=f"matched via render-based re-ID search (score {result.score:.2f})",
        )
