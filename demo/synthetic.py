"""Deterministic synthetic vehicle + capture trajectory for the demo and tests.

Everything here is CPU-only and seeded, so the demo's provenance numbers are
reproducible and assertable.

ON THE VISIBILITY CULL
----------------------
The procedural sedan is built from overlapping box primitives, so a third of its
sampled surface is *interior* faces — permanently occluded geometry no camera can
ever see. Left in, `observed%` would plateau near 50% and read as a broken
fusion loop when it is really just unreachable surface. Real priors (TRELLIS,
SF3D) emit outer shells, not stacked boxes, so culling here makes the demo
represent the real system rather than an artifact of the stand-in mesh.
"""
from __future__ import annotations

import numpy as np

from cargen.core.camera import CameraPose, Intrinsics
from cargen.core.splat import GaussianCloud
from cargen.fusion_engine.point_renderer import PointRenderer
from cargen.prior_generation.stub import build_sedan_mesh
from cargen.prior_generation.mesh_to_splats import mesh_to_splats
from cargen.segmentation.interface import Segmenter


class BackgroundSegmenter(Segmenter):
    """Exact vehicle mask for synthetic renders: anything not the white backdrop.

    The generic `StubSegmenter` returns a crude centered rectangle, which makes
    background pixels read as "vehicle with no geometry there" and sends the
    densifier spawning junk splats into empty space. Synthetic frames have a
    known flat background, so the demo can segment them exactly — and measure
    fusion rather than segmentation error.
    """

    def __init__(self, threshold: float = 0.96):
        self._threshold = threshold

    def segment(self, image_rgb: np.ndarray) -> np.ndarray:
        value = image_rgb.astype(np.float32) / 255.0
        return (value.min(axis=2) < self._threshold).astype(np.float32)

# The "real" car: red paint, black aftermarket rims — differs from the prior's
# silver guess, which is what fusion has to discover.
TRUTH_COLORS = {"paint": (0.72, 0.11, 0.10), "wheel": (0.06, 0.06, 0.07)}
PRIOR_COLORS = {"paint": (0.55, 0.58, 0.62)}


def orbit_pose(angle_rad: float, radius: float = 3.2, height: float = 1.25) -> CameraPose:
    """A camera on the walk-around ring, aimed at the vehicle's mid-body."""
    eye = (radius * np.cos(angle_rad), radius * np.sin(angle_rad), height)
    return CameraPose.look_at(eye=eye, target=(0.0, 0.0, 0.5))


def visible_splat_mask(
    cloud: GaussianCloud,
    intrinsics: Intrinsics,
    renderer: PointRenderer,
    n_views: int = 32,
    heights=(0.8, 1.5, 2.4),
) -> np.ndarray:
    """Which splats any camera on the capture hemisphere can ever see."""
    seen = np.zeros(cloud.n, bool)
    for angle in np.linspace(0, 2 * np.pi, n_views, endpoint=False):
        for height in heights:
            render = renderer.render(cloud, orbit_pose(angle, height=height), intrinsics)
            idx = render.splat_index[render.hit_mask]
            if idx.size:
                seen[np.unique(idx[idx >= 0])] = True
    return seen


def build_truth_cloud(
    intrinsics: Intrinsics,
    renderer: PointRenderer | None = None,
    n_points: int = 12_000,
    seed: int = 3,
) -> GaussianCloud:
    """The ground-truth vehicle, outer surface only (see module docstring)."""
    renderer = renderer or PointRenderer()
    dense = mesh_to_splats(build_sedan_mesh(TRUTH_COLORS), n_points=n_points, seed=seed)
    return dense.select(visible_splat_mask(dense, intrinsics, renderer))


def build_prior_cloud(truth: GaussianCloud, seed: int = 3) -> GaussianCloud:
    """The generative prior's guess: same shape, wrong (factory-default) paint.

    Sharing the geometry isolates what the demo is measuring — the provenance
    arbitration — from prior-shape error, which is a separate concern owned by
    the real image-to-3D backend.
    """
    rng = np.random.default_rng(seed)
    paint = np.asarray(PRIOR_COLORS["paint"], np.float32)
    # slight per-splat variation so the prior isn't suspiciously uniform
    colors = np.clip(paint + rng.normal(0, 0.02, size=(truth.n, 3)), 0, 1).astype(np.float32)
    return GaussianCloud.create(
        positions=truth.positions.copy(),
        colors=colors,
        scales=truth.scales.copy(),
        rotations=truth.rotations.copy(),
        opacities=truth.opacities.copy(),
        confidence=0.15,
    )


def render_photo(
    cloud: GaussianCloud,
    pose: CameraPose,
    intrinsics: Intrinsics,
    renderer: PointRenderer,
    exposure: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic 'photo' of the truth cloud → (RGB uint8, vehicle mask).

    `exposure` simulates the lighting change between capture sessions — the
    thing the fusion engine's exposure compensation has to absorb without
    flagging the whole car dirty.
    """
    render = renderer.render(cloud, pose, intrinsics)
    image = np.clip(render.color * exposure, 0, 1)
    if seed is not None:
        rng = np.random.default_rng(seed)
        image = np.clip(image + rng.normal(0, 0.01, image.shape), 0, 1)
    return (image * 255).astype(np.uint8), render.hit_mask.astype(np.float32)


def walkaround_frames(
    truth: GaussianCloud,
    intrinsics: Intrinsics,
    renderer: PointRenderer,
    n_frames: int = 24,
    start_angle: float = 0.0,
    sweep: float = 2 * np.pi,
    exposure: float = 1.0,
    fps: float = 12.0,
):
    """A mock walk-around video: (index, RGB, timestamp) tuples + true poses."""
    frames, poses = [], []
    for i in range(n_frames):
        angle = start_angle + sweep * i / n_frames
        pose = orbit_pose(angle)
        image, _ = render_photo(truth, pose, intrinsics, renderer, exposure, seed=100 + i)
        frames.append((i, image, i / fps))
        poses.append(pose)
    return frames, poses
