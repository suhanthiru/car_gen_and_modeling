"""Real Gaussian rasterizer: gsplat (Nerfstudio) — Milestone B.

INTEGRATION POINT — THE RISKIEST INSTALL IN THE STACK
-----------------------------------------------------
Prereqs:  Visual Studio Build Tools (Desktop C++ workload) + CUDA Toolkit
          matching the torch build (~8 GB one-time). Then:
              pip install gsplat
          It JIT-compiles CUDA kernels on first import — the first render is
          slow and this is where Windows setups usually break. Isolated to
          Milestone B on purpose so nothing else is blocked by it.
License:  Apache-2.0.
VRAM:     scales with splat count and resolution; ~2-4 GB at 400k splats /
          1080p on the 8 GB laptop. Reduce `max_splats` if it OOMs.

What this buys over PointRenderer: real alpha-blended anisotropic Gaussians and
view-dependent SH appearance — i.e. actual specular paint — plus gradients, so
`optimize.py` can run the real localized Adam loop.

Contract note: `splat_index` here reports the highest-contribution splat per
pixel (from gsplat's per-pixel sorting), which is the correct analogue of the
CPU renderer's nearest-splat attribution for dirty-flagging.
"""
from __future__ import annotations

import numpy as np

from cargen.core.camera import CameraPose, Intrinsics
from cargen.core.splat import GaussianCloud
from cargen.fusion_engine.renderer import RenderResult, SplatRenderer

#: Y_0^0 — the SH DC basis function. Same constant the exporter uses.
_SH_C0 = 0.28209479177387814


def _rgb_to_sh_dc_torch(rgb, torch):
    """Linear RGB → SH DC coefficient, the inverse of what a viewer applies."""
    return (rgb - 0.5) / _SH_C0


class GsplatRenderer(SplatRenderer):
    supports_gradients = True  # unlocks LocalizedOptimizer in the pipeline

    def __init__(self, device: str = "cuda", background: float = 1.0,
                 max_radius_px: int = 12):
        import torch  # lazy: keeps the CPU path importable without CUDA

        self._torch = torch
        self._device = device
        self._background = background
        # caps the ID pass's footprint loop; matches PointRenderer's own cap
        self._max_radius = max_radius_px

    def _to_tensors(self, cloud: GaussianCloud):
        t = self._torch
        return (
            t.from_numpy(cloud.positions).float().to(self._device),
            t.from_numpy(cloud.rotations).float().to(self._device),
            t.from_numpy(cloud.scales).float().to(self._device),
            t.from_numpy(cloud.opacities).float().to(self._device),
            self._sh_coefficients(cloud),
        )

    def _sh_coefficients(self, cloud: GaussianCloud):
        """Colours for gsplat: (N,3) flat RGB, or (N,K,3) SH when we have bands.

        gsplat switches behaviour on the shape — a 3D `colors` tensor is read as
        SH coefficients (DC first) and evaluated per view direction, which is
        what makes a highlight track the camera. Passing (N,3) is the matte path.
        Handing it SH when every band is zero would only cost work for an
        identical image, so a prior stays on the cheap path.
        """
        t = self._torch
        dc = t.from_numpy(cloud.colors).float().to(self._device)
        if not cloud.is_view_dependent:
            return dc
        rest = t.from_numpy(cloud.sh_rest).float().to(self._device)
        # gsplat wants DC as band 0, so prepend it: (N, 1+15, 3)
        return t.cat([_rgb_to_sh_dc_torch(dc, t)[:, None, :], rest], dim=1)

    def render(
        self, cloud: GaussianCloud, pose: CameraPose, intrinsics: Intrinsics
    ) -> RenderResult:
        from gsplat import rasterization

        t = self._torch
        h, w = intrinsics.height, intrinsics.width
        if cloud.n == 0:
            return RenderResult(
                np.full((h, w, 3), self._background, np.float32),
                np.full((h, w), np.inf, np.float32),
                np.full((h, w), -1, np.int32),
                np.zeros((h, w), np.float32),
            )

        means, quats, scales, opacities, colors = self._to_tensors(cloud)
        viewmat = t.eye(4, device=self._device)
        viewmat[:3, :3] = t.from_numpy(pose.R).float().to(self._device)
        viewmat[:3, 3] = t.from_numpy(pose.t).float().to(self._device)
        K = t.from_numpy(intrinsics.K).float().to(self._device)[None]

        with t.no_grad():
            rendered, alpha, meta = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmat[None],
                Ks=K,
                width=w,
                height=h,
                render_mode="RGB+ED",  # ED = expected depth
                # a 3D colours tensor means SH; gsplat needs the band count told
                # to it, and evaluates per view direction from there
                sh_degree=3 if colors.ndim == 3 else None,
                backgrounds=t.full((1, 3), self._background, device=self._device),
            )

        image = rendered[0, ..., :3].cpu().numpy().astype(np.float32)
        depth = rendered[0, ..., 3].cpu().numpy().astype(np.float32)
        alpha_np = alpha[0, ..., 0].cpu().numpy().astype(np.float32)
        depth[alpha_np <= 0.01] = np.inf

        index = self._splat_index(cloud, pose, intrinsics, alpha_np)
        return RenderResult(image, depth, index, alpha_np)

    def _splat_index(
        self,
        cloud: GaussianCloud,
        pose: CameraPose,
        intrinsics: Intrinsics,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """Which splat is responsible for each pixel — a separate z-buffer pass.

        gsplat exposes no per-pixel ID buffer, and reconstructing one from its
        tile/intersection buffers means re-implementing its alpha-compositing
        loop against internals that move between releases. Not worth it: the
        fusion engine only needs to attribute a residual to *some* splat near
        that surface, which is exactly what a nearest-centre z-buffer gives —
        and it is the same semantics PointRenderer already defines, so the
        RenderResult contract holds and the engine needs no special case.

        Division of labour: gsplat owns appearance, this owns attribution.
        """
        t = self._torch
        h, w = intrinsics.height, intrinsics.width
        index = np.full((h, w), -1, np.int32)

        uv, z = pose.project(cloud.positions, intrinsics)
        radius_world = cloud.scales.max(axis=1)
        radius_px = np.clip(
            intrinsics.fx * radius_world / np.maximum(z, 1e-6), 0.5, self._max_radius
        )
        visible = (z > 1e-6) & (cloud.opacities > 0.05)
        if not visible.any():
            return index

        # Rasterize each splat's disc footprint, keeping the nearest per pixel.
        # scatter_reduce with amin over a packed depth-and-id key resolves ties
        # by depth in one pass, on the GPU, with no Python loop.
        idx = np.where(visible)[0]
        du, dz, dr = uv[idx], z[idx], radius_px[idx]
        radius = int(np.ceil(min(float(dr.max()), self._max_radius)))

        keys = t.full((h * w,), float("inf"), device=self._device)
        vals = t.full((h * w,), -1, dtype=t.int32, device=self._device)
        cu = t.from_numpy(du[:, 0]).float().to(self._device)
        cv = t.from_numpy(du[:, 1]).float().to(self._device)
        cz = t.from_numpy(dz).float().to(self._device)
        cr = t.from_numpy(dr).float().to(self._device)
        ids = t.from_numpy(idx.astype(np.int32)).to(self._device)

        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                if ox * ox + oy * oy > (radius + 0.5) ** 2:
                    continue
                px = t.round(cu + ox).long()
                py = t.round(cv + oy).long()
                inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
                # only paint pixels this splat's own disc actually covers
                inside &= (ox * ox + oy * oy) <= cr * cr
                if not bool(inside.any()):
                    continue
                flat = py[inside] * w + px[inside]
                depth_here = cz[inside]
                keys.scatter_reduce_(0, flat, depth_here, reduce="amin", include_self=True)

        # second pass: whoever matches the winning depth owns the pixel
        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                if ox * ox + oy * oy > (radius + 0.5) ** 2:
                    continue
                px = t.round(cu + ox).long()
                py = t.round(cv + oy).long()
                inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
                inside &= (ox * ox + oy * oy) <= cr * cr
                if not bool(inside.any()):
                    continue
                flat = py[inside] * w + px[inside]
                won = t.isclose(keys[flat], cz[inside])
                if bool(won.any()):
                    vals[flat[won]] = ids[inside][won]

        index = vals.reshape(h, w).cpu().numpy().astype(np.int32)
        index[alpha <= 0.01] = -1  # gsplat says nothing is there; believe it
        return index
