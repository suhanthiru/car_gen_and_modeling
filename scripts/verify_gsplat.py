"""Manual acceptance check for the real gsplat toolchain — Milestone B.

Not a pytest file: it needs CUDA + a working gsplat build, neither of which the
CPU test suite has (see `tests/test_fusion.py::TestLocalizedOptimizer`'s fake
double for the CPU-only contract tests). Run this once after
`pip install gsplat lightglue` to confirm the install actually works end to
end, not just that `import gsplat` didn't crash — a JIT-compiled kernel that
silently no-ops (e.g. a stale/mismatched build) would still "import" fine and
still "render" fine, but would never actually move the optimizer, which is
exactly the failure mode step 5 below is built to catch.

Run: python scripts/verify_gsplat.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows consoles default to cp1252, which cannot encode the bar/rule glyphs.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cargen.core.camera import Intrinsics
from cargen.fusion_engine.gsplat_renderer import GsplatRenderer
from cargen.fusion_engine.optimize import LocalizedOptimizer, OptimizeConfig
from cargen.fusion_engine.point_renderer import PointRenderer
from demo.synthetic import build_prior_cloud, build_truth_cloud, orbit_pose, render_photo

WIDTH, HEIGHT = 64, 48


def _loss(renderer, cloud, target_rgb, pixel_weight, pose, intrinsics) -> float:
    """Weighted L1 against the target photo — the same core term optimize.py
    minimizes, used here only to observe whether refine() actually moved
    anything, not to reproduce its D-SSIM term."""
    rendered = renderer.render(cloud, pose, intrinsics)
    target = target_rgb.astype(np.float32) / 255.0
    weight = pixel_weight[..., None]
    denom = max(float(weight.sum()), 1e-8)
    return float((np.abs(rendered.color - target) * weight).sum()) / denom


def main() -> int:
    print("\n" + "=" * 78)
    print("verify_gsplat — manual acceptance check for the gsplat/CUDA toolchain")
    print("=" * 78)

    print("\n[1/5] Checking CUDA availability...")
    import torch

    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False — no CUDA GPU visible to this process.")
        return 1
    device_name = torch.cuda.get_device_name(0)
    print(f"  CUDA available: {device_name}")

    print("\n[2/5] Importing gsplat (JIT-compiles CUDA kernels — expect several minutes "
          "on first run, seconds after)...")
    t0 = time.time()
    from gsplat import rasterization  # noqa: F401 — import is the point; triggers the JIT build

    print(f"  gsplat imported in {time.time() - t0:.1f}s")

    print("\n[3/5] Building a tiny synthetic GaussianCloud + target photo...")
    intrinsics = Intrinsics.simple(WIDTH, HEIGHT)
    point_renderer = PointRenderer()
    truth = build_truth_cloud(intrinsics, point_renderer, n_points=300)
    prior = build_prior_cloud(truth)
    pose = orbit_pose(0.0)
    target_rgb, mask = render_photo(truth, pose, intrinsics, point_renderer, seed=1)
    print(f"  cloud: {prior.n} splats, image {intrinsics.width}x{intrinsics.height}")

    print("\n[4/5] Round-tripping through GsplatRenderer.render()...")
    gsplat_renderer = GsplatRenderer()
    result = gsplat_renderer.render(prior, pose, intrinsics)
    if not np.isfinite(result.color).all():
        print("FAIL: GsplatRenderer.render() produced non-finite pixels (NaN/Inf).")
        return 1
    print(f"  render OK: shape {result.color.shape}, finite everywhere")

    print("\n[5/5] Round-tripping through LocalizedOptimizer.refine() over 20 iterations...")
    dirty_indices = np.arange(prior.n)
    pixel_weight = mask.astype(np.float32)
    optimizer = LocalizedOptimizer(OptimizeConfig(iterations=20), device="cuda")

    before_loss = _loss(gsplat_renderer, prior, target_rgb, pixel_weight, pose, intrinsics)
    refined = optimizer.refine(prior, dirty_indices, target_rgb, pixel_weight, pose, intrinsics)
    after_loss = _loss(gsplat_renderer, refined, target_rgb, pixel_weight, pose, intrinsics)
    print(f"  loss before: {before_loss:.5f}   after 20 iters: {after_loss:.5f}")

    if not (after_loss < before_loss):
        print("FAIL: loss did not decrease — the optimizer step may be a silently-broken "
              "no-op (stale/mismatched JIT build), not a crash.")
        return 1

    print("\n" + "=" * 78)
    print(f"SUCCESS — gsplat renderer + LocalizedOptimizer verified working on {device_name}")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
