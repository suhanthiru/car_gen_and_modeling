"""Splat renderer interface.

Two consumers with different needs:
  * fusion — needs colour AND a per-pixel splat index map, so high-residual
    pixels can be traced back to the splats responsible ("which guesses does
    this photo disagree with?");
  * viewer/verifier — needs colour only.

The index map is what makes localized editing possible; a renderer that cannot
report which splat painted a pixel cannot drive this fusion loop.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from cargen.core.camera import CameraPose, Intrinsics
from cargen.core.splat import GaussianCloud


@dataclass(frozen=True)
class RenderResult:
    color: np.ndarray       # (H, W, 3) float32 in [0, 1]
    depth: np.ndarray       # (H, W) float32; inf where nothing was hit
    splat_index: np.ndarray  # (H, W) int32; -1 where nothing was hit
    alpha: np.ndarray       # (H, W) float32 in [0, 1]; coverage

    @property
    def hit_mask(self) -> np.ndarray:
        return self.splat_index >= 0


class SplatRenderer(ABC):
    #: Whether this renderer is differentiable, i.e. whether the localized
    #: optimizer can run against it. False for the CPU stand-in, which can only
    #: repaint splats; True for gsplat, which recovers geometry from gradients.
    #: Declared rather than probed so the pipeline's choice stays explicit.
    supports_gradients: bool = False

    @abstractmethod
    def render(
        self, cloud: GaussianCloud, pose: CameraPose, intrinsics: Intrinsics
    ) -> RenderResult:
        """Rasterize `cloud` from `pose`."""

    def unproject(
        self,
        cloud: GaussianCloud,
        pose: CameraPose,
        intrinsics: Intrinsics,
        uv: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pixels (N,2) → 3D points via rendered depth. Returns (points, valid).

        Used by the video tracker to lift the previous frame's keypoints into
        3D without re-running matching against the whole history.
        """
        result = self.render(cloud, pose, intrinsics)
        u = np.round(uv[:, 0]).astype(int)
        v = np.round(uv[:, 1]).astype(int)
        inside = (
            (u >= 0) & (u < intrinsics.width) & (v >= 0) & (v < intrinsics.height)
        )
        points = np.zeros((uv.shape[0], 3), np.float64)
        valid = np.zeros((uv.shape[0],), bool)
        if not inside.any():
            return points, valid

        idx = result.splat_index[v[inside], u[inside]]
        hit = idx >= 0
        # splat centers are already in the canonical frame — no unprojection math
        # needed, and it avoids depth-quantization error at silhouette edges
        rows = np.where(inside)[0][hit]
        points[rows] = cloud.positions[idx[hit]]
        valid[rows] = True
        return points, valid
