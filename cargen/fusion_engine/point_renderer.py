"""CPU point-splatting renderer — real, dependency-free, z-buffered.

Not a Gaussian rasterizer: each splat is drawn as an opaque disc of its
projected radius, resolved by depth. That is enough for everything the CPU path
needs — residual maps, dirty-splat attribution, the merge verifier, and the
synthetic demo — and it keeps the whole fusion loop runnable and testable with
no CUDA toolchain.

`gsplat_renderer.py` swaps in the real alpha-blended, view-dependent rasterizer
at Milestone B. The index map here reports the nearest splat per pixel; gsplat's
reports the highest-contribution one. Same contract.
"""
from __future__ import annotations

import numpy as np

from cargen.core.camera import CameraPose, Intrinsics
from cargen.core.splat import GaussianCloud
from cargen.fusion_engine.renderer import RenderResult, SplatRenderer


class PointRenderer(SplatRenderer):
    def __init__(self, background: float = 1.0, max_radius_px: int = 12,
                 min_radius_px: float = 0.6):
        self._background = background
        self._max_radius = max_radius_px
        self._min_radius = min_radius_px

    def render(
        self, cloud: GaussianCloud, pose: CameraPose, intrinsics: Intrinsics
    ) -> RenderResult:
        h, w = intrinsics.height, intrinsics.width
        color = np.full((h, w, 3), self._background, np.float32)
        depth = np.full((h, w), np.inf, np.float32)
        index = np.full((h, w), -1, np.int32)

        if cloud.n == 0:
            return RenderResult(color, depth, index, np.zeros((h, w), np.float32))

        uv, z = pose.project(cloud.positions, intrinsics)
        # world-space splat radius → pixels, via the pinhole scale at that depth
        radius_world = cloud.scales.max(axis=1)
        radius_px = np.clip(
            intrinsics.fx * radius_world / np.maximum(z, 1e-6),
            self._min_radius,
            self._max_radius,
        )
        visible = (
            (z > 1e-6)
            & (cloud.opacities > 0.05)
            & (uv[:, 0] > -radius_px)
            & (uv[:, 0] < w + radius_px)
            & (uv[:, 1] > -radius_px)
            & (uv[:, 1] < h + radius_px)
        )
        order = np.argsort(-z[visible])  # far → near, so near splats overwrite
        candidates = np.where(visible)[0][order]

        for i in candidates:
            self._draw_disc(
                color, depth, index, i, uv[i], float(z[i]), float(radius_px[i]),
                cloud.colors[i], h, w,
            )

        alpha = (index >= 0).astype(np.float32)
        return RenderResult(color, depth, index, alpha)

    def _draw_disc(self, color, depth, index, splat_id, center, z, radius,
                   rgb, h, w) -> None:
        r = max(int(np.ceil(radius)), 1)
        cu, cv = int(round(center[0])), int(round(center[1]))
        u0, u1 = max(cu - r, 0), min(cu + r + 1, w)
        v0, v1 = max(cv - r, 0), min(cv + r + 1, h)
        if u0 >= u1 or v0 >= v1:
            return

        vv, uu = np.mgrid[v0:v1, u0:u1]
        inside = (uu - center[0]) ** 2 + (vv - center[1]) ** 2 <= radius**2
        if not inside.any():
            return
        patch_depth = depth[v0:v1, u0:u1]
        nearer = inside & (z < patch_depth)
        if not nearer.any():
            return
        patch_depth[nearer] = z
        index[v0:v1, u0:u1][nearer] = splat_id
        color[v0:v1, u0:u1][nearer] = rgb
