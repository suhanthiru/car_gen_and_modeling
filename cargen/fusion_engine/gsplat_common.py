"""Shared gsplat optimization helpers — Milestone B.

Factored out of `optimize.py` so `LocalizedOptimizer` (per-frame, localized)
and `Consolidator` (`consolidate.py`, joint multi-frame) share one copy of the
3DGS loss and the SH-DC/rest coefficient assembly, instead of each carrying
its own. Both callers optimize the same activated parameterization (log-scale,
logit-opacity) against the same `(1-λ)·L1 + λ·D-SSIM` loss from the 3DGS
paper; only the *scope* of what gets optimized differs (one frame's dirty
splats vs. every splat any frame in a joint batch has seen).

No `torch`/`gsplat` import at module scope — this stays importable on a
CPU-only machine. `torch` is threaded through as a parameter where needed
(mirroring `gsplat_renderer.py`'s convention) so callers keep ownership of the
lazy import.
"""
from __future__ import annotations

import numpy as np

#: Y_0^0 — the SH DC basis function, matching the exporter and gsplat renderer.
SH_C0 = 0.28209479177387814


def logit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Inverse sigmoid, clamped — the activation `LocalizedOptimizer` and
    `Consolidator` both invert to get a raw parameter for Adam to move."""
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def rgb_to_sh_dc_torch(rgb):
    """Linear RGB → SH DC coefficient (differentiable; keeps the graph)."""
    return (rgb - 0.5) / SH_C0


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numpy inverse of `logit` — de-activates an optimized opacity parameter
    back to [0, 1] without needing torch (used on the CPU-testable path)."""
    return 1.0 / (1.0 + np.exp(-x))


def sh_colors_from_dc_rest(dc, rest):
    """Assemble gsplat's (N, 16, 3) SH colour tensor from a DC band + rest bands.

    `dc` is (N, 3) linear RGB (band 0, converted to SH here); `rest` is
    (N, 15, 3), already in SH coefficient space. `dc`/`rest` are torch tensors;
    `torch` itself is imported lazily here (mirrors every other gsplat-facing
    module in this package) so this file stays importable with no torch
    installed.
    """
    import torch

    return torch.cat([rgb_to_sh_dc_torch(dc)[:, None, :], rest], dim=1)


def gaussian_window(size: int, sigma: float, device, torch):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def weighted_l1_dssim(rendered, target, weight, ssim_lambda: float):
    """(1-λ)·L1 + λ·D-SSIM, both masked to `weight`.

    The 3DGS paper's loss. L1 alone converges to a blurry mean; SSIM alone is
    indifferent to absolute colour. `weight` is what makes a call *localized*
    when the caller wants that: pixels outside the region of interest
    contribute zero. `LocalizedOptimizer` passes a dirty-region mask;
    `Consolidator` passes a uniform (or evidence-weighted) mask over the whole
    frame, since joint optimization has no single dirty region to confine to.
    """
    import torch
    import torch.nn.functional as F

    denom = weight.sum().clamp_min(1e-8)
    l1 = ((rendered - target).abs() * weight).sum() / (denom * 3)

    # SSIM over an 11x11 Gaussian window, the standard setup
    x = rendered.permute(0, 3, 1, 2)
    y = target.permute(0, 3, 1, 2)
    win = gaussian_window(11, 1.5, rendered.device, torch).expand(3, 1, 11, 11)
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
