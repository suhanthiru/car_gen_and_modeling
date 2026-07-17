"""Localized splat optimization: L1 + D-SSIM, Adam, dirty region only — Milestone B.

INTEGRATION POINT (requires gsplat — see gsplat_renderer.py for the install)
----------------------------------------------------------------------------
This replaces the CPU engine's direct colour blend with real gradient descent
over position/scale/rotation/opacity/SH, which is what actually recovers
geometry (a dent's shape, not just its shading).

THE CRITICAL PROPERTY: only dirty splats carry gradients. Everything else is
detached, so a few hundred iterations cost milliseconds instead of retraining
the whole vehicle, and — more importantly — confirmed regions are mathematically
incapable of drifting. The dilation ring is included in the optimization but at
reduced loss weight, so the boundary blends instead of seaming.

Loss follows the 3DGS paper: (1 - λ)·L1 + λ·D-SSIM, λ = 0.2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cargen.core.camera import CameraPose, Intrinsics
from cargen.core.splat import GaussianCloud


#: Y_0^0 — the SH DC basis function, matching the exporter and gsplat renderer.
_SH_C0 = 0.28209479177387814


def _logit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def _rgb_to_sh_dc_torch(rgb):
    """Linear RGB → SH DC coefficient (differentiable; keeps the graph)."""
    return (rgb - 0.5) / _SH_C0


def _gaussian_window(size: int, sigma: float, device, torch):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def _weighted_l1_dssim(rendered, target, weight, ssim_lambda: float):
    """(1-λ)·L1 + λ·D-SSIM, both masked to the dirty region.

    The 3DGS paper's loss. L1 alone converges to a blurry mean; SSIM alone is
    indifferent to absolute colour. `weight` is what makes this *localized*:
    pixels outside the dirty region contribute zero, so nothing the frame does
    not dispute can pull on the optimizer.
    """
    import torch
    import torch.nn.functional as F

    denom = weight.sum().clamp_min(1e-8)
    l1 = ((rendered - target).abs() * weight).sum() / (denom * 3)

    # SSIM over an 11x11 Gaussian window, the standard setup
    x = rendered.permute(0, 3, 1, 2)
    y = target.permute(0, 3, 1, 2)
    win = _gaussian_window(11, 1.5, rendered.device, torch).expand(3, 1, 11, 11)
    mu_x = F.conv2d(x, win, padding=5, groups=3)
    mu_y = F.conv2d(y, win, padding=5, groups=3)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x = F.conv2d(x * x, win, padding=5, groups=3) - mu_x2
    sigma_y = F.conv2d(y * y, win, padding=5, groups=3) - mu_y2
    sigma_xy = F.conv2d(x * y, win, padding=5, groups=3) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    )
    w = weight.permute(0, 3, 1, 2)
    ssim = (ssim_map * w).sum() / (w.sum().clamp_min(1e-8) * 3)
    return (1 - ssim_lambda) * l1 + ssim_lambda * (1 - ssim)


@dataclass
class OptimizeConfig:
    iterations: int = 300
    lr_position: float = 1e-4
    lr_scale: float = 5e-3
    lr_rotation: float = 1e-3
    lr_opacity: float = 5e-2
    lr_color: float = 2.5e-3
    # The 3DGS paper trains higher SH bands 20x slower than the DC term: they
    # are a small view-dependent correction, and letting them move at full speed
    # lets the model explain away geometry error as "lighting".
    lr_sh_rest: float = 2.5e-3 / 20
    ssim_lambda: float = 0.2
    ring_weight: float = 0.3  # blending-ring pixels count less than core dirty ones


class LocalizedOptimizer:
    """Refines only the dirty splats against one observed frame."""

    def __init__(self, config: OptimizeConfig | None = None, device: str = "cuda"):
        import torch

        self._torch = torch
        self._device = device
        self.config = config or OptimizeConfig()

    def refine(
        self,
        cloud: GaussianCloud,
        dirty_indices: np.ndarray,
        image_rgb: np.ndarray,
        pixel_weight: np.ndarray,
        pose: CameraPose,
        intrinsics: Intrinsics,
    ) -> GaussianCloud:
        """Optimize `dirty_indices` toward `image_rgb`; return the updated cloud.

        `pixel_weight` (H, W) weights the loss per pixel — core dirty region at
        1.0, dilation ring at `ring_weight`, frozen/unseen elsewhere at 0.
        """
        from gsplat import rasterization

        t = self._torch
        if dirty_indices.size == 0:
            return cloud

        dirty = np.zeros(cloud.n, bool)
        dirty[dirty_indices] = True
        dev = self._device
        cfg = self.config

        def tensor(a, dtype=None):
            return t.from_numpy(np.ascontiguousarray(a)).to(dev, dtype or t.float32)

        # Frozen splats take part in the render (they occlude, they blend at the
        # boundary) but carry no gradient — that is the whole point of localized
        # refinement, and it is what keeps confirmed regions mathematically
        # incapable of drifting.
        frozen = {
            "means": tensor(cloud.positions[~dirty]),
            "quats": tensor(cloud.rotations[~dirty]),
            "scales": tensor(cloud.scales[~dirty]),
            "opacities": tensor(cloud.opacities[~dirty]),
            "colors": tensor(cloud.colors[~dirty]),
            "sh_rest": tensor(cloud.sh_rest[~dirty]),
        }

        # Optimize in the same activated space the exporter stores: log-scale and
        # logit-opacity, so the optimizer can't drive them negative or past 1.
        params = {
            "means": tensor(cloud.positions[dirty]).requires_grad_(True),
            "quats": tensor(cloud.rotations[dirty]).requires_grad_(True),
            "log_scales": tensor(np.log(np.maximum(cloud.scales[dirty], 1e-9))).requires_grad_(True),
            "logit_opacities": tensor(_logit(cloud.opacities[dirty])).requires_grad_(True),
            "colors": tensor(cloud.colors[dirty]).requires_grad_(True),
            "sh_rest": tensor(cloud.sh_rest[dirty]).requires_grad_(True),
        }
        optimizer = t.optim.Adam(
            [
                {"params": [params["means"]], "lr": cfg.lr_position},
                {"params": [params["quats"]], "lr": cfg.lr_rotation},
                {"params": [params["log_scales"]], "lr": cfg.lr_scale},
                {"params": [params["logit_opacities"]], "lr": cfg.lr_opacity},
                {"params": [params["colors"]], "lr": cfg.lr_color},
                {"params": [params["sh_rest"]], "lr": cfg.lr_sh_rest},
            ]
        )

        target = tensor(image_rgb.astype(np.float32) / 255.0
                        if image_rgb.dtype == np.uint8 else image_rgb)
        weight = tensor(pixel_weight)[None, ..., None]
        viewmat = t.eye(4, device=dev)
        viewmat[:3, :3] = tensor(pose.R)
        viewmat[:3, 3] = tensor(pose.t)
        K = tensor(intrinsics.K)[None]
        h, w = intrinsics.height, intrinsics.width

        for _ in range(cfg.iterations):
            optimizer.zero_grad(set_to_none=True)
            means = t.cat([frozen["means"], params["means"]])
            quats = t.cat([frozen["quats"], params["quats"]])
            scales = t.cat([frozen["scales"], t.exp(params["log_scales"])])
            opacities = t.cat([frozen["opacities"], t.sigmoid(params["logit_opacities"])])
            dc = t.cat([frozen["colors"], params["colors"].clamp(0, 1)])
            rest = t.cat([frozen["sh_rest"], params["sh_rest"]])
            # SH coefficients: DC as band 0, then bands 1-3 -> (N, 16, 3)
            colors = t.cat([_rgb_to_sh_dc_torch(dc)[:, None, :], rest], dim=1)

            rendered, _, _ = rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities,
                colors=colors, viewmats=viewmat[None], Ks=K, width=w, height=h,
                sh_degree=3, backgrounds=t.full((1, 3), 1.0, device=dev),
            )
            loss = _weighted_l1_dssim(
                rendered[..., :3], target[None], weight, cfg.ssim_lambda
            )
            loss.backward()
            optimizer.step()

        with t.no_grad():
            out_pos = cloud.positions.copy()
            out_rot = cloud.rotations.copy()
            out_scale = cloud.scales.copy()
            out_opac = cloud.opacities.copy()
            out_col = cloud.colors.copy()
            out_sh = cloud.sh_rest.copy()
            q = params["quats"].detach()
            q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-9)  # keep unit
            out_pos[dirty] = params["means"].detach().cpu().numpy()
            out_rot[dirty] = q.cpu().numpy()
            out_scale[dirty] = t.exp(params["log_scales"]).detach().cpu().numpy()
            out_opac[dirty] = t.sigmoid(params["logit_opacities"]).detach().cpu().numpy()
            out_col[dirty] = params["colors"].detach().clamp(0, 1).cpu().numpy()
            out_sh[dirty] = params["sh_rest"].detach().cpu().numpy()

        return GaussianCloud(
            positions=out_pos, scales=out_scale, rotations=out_rot,
            opacities=out_opac, colors=out_col, sh_rest=out_sh,
            provenance=cloud.provenance, confidence=cloud.confidence,
            view_count=cloud.view_count, last_seen_ts=cloud.last_seen_ts,
        )
