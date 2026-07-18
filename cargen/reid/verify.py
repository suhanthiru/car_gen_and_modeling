"""Render-based verifier — phase 2 of the identity/pose problem (see interface.py).

Locates a photo's camera by rendering the current model over a sweep of
candidate angles and finding the one whose render matches the photo in
embedding space, rather than by 2D-3D keypoint correspondence (`registration.py`).
This is what lets a photo register with only *partial* appearance overlap
against what's already confirmed — a gap sparse feature matching cannot bridge
because there may be too few (or zero) distinct keypoints in common, even
though a holistic render-vs-photo comparison still recognises the same car.

WHAT THIS DOES NOT FIX
-----------------------
Comparison is masked to OBSERVED-provenance pixels only (never PRIOR — see
`pose_estimation/interface.py`'s inviolable rule), so a candidate angle with no
confirmed geometry visible from it has nothing legitimate to compare against
and is skipped. Two photos of *literally opposite* sides of a brand-new
vehicle — zero confirmed overlap in either direction — still cannot register
against each other; that is the honest architectural limit, not a bug. What
this buys is registering photos with partial (not near-total) overlap with
something already confirmed, without needing a frame-to-frame video.

AMBIGUITY
---------
Cars are close to bilaterally symmetric, so the front and rear (or the two
sides) can render similarly. The fix is a margin check: the winning angle must
beat the best candidate more than `ambiguity_angle_deg` away by at least
`ambiguity_margin`, or the match is flagged ambiguous and rejected — a
plausible-looking but wrong pose is worse than no pose.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cargen.core.camera import CameraPose, Intrinsics
from cargen.core.splat import GaussianCloud, Provenance
from cargen.fusion_engine.renderer import RenderResult, SplatRenderer
from cargen.reid.interface import Embedder


@dataclass(frozen=True)
class VerifyResult:
    pose: CameraPose | None
    score: float          # best embedding similarity in [0, 1]; 0 if no candidate qualified
    ambiguous: bool        # a far-away candidate nearly matched the winner
    observed_px: int       # confirmed pixels the winning render actually compared


@dataclass(frozen=True)
class _Candidate:
    pose: CameraPose
    score: float
    observed_px: int


def _cloud_bounds(cloud: GaussianCloud) -> tuple[np.ndarray, float, tuple[float, float]]:
    """(center, orbit_radius, (min_z, max_z)) from the cloud's own extent —
    the prior's overall silhouette/scale is trustworthy even before anything
    is confirmed (see docs/SETUP.md on SF3D framing), so this needs no
    OBSERVED filtering."""
    lo, hi = cloud.positions.min(axis=0), cloud.positions.max(axis=0)
    center = (lo + hi) / 2
    radius = float(np.linalg.norm(hi[:2] - lo[:2])) * 1.1 + 0.5
    return center, radius, (float(lo[2]), float(hi[2]))


def _orbit_pose(center: np.ndarray, radius: float, angle: float, height: float) -> CameraPose:
    eye = center + np.array([radius * np.cos(angle), radius * np.sin(angle), height])
    return CameraPose.look_at(eye=eye, target=center)


def candidate_poses(
    cloud: GaussianCloud, n_azimuth: int, n_elevation: int
) -> list[CameraPose]:
    """A grid of orbit cameras around the cloud, aimed at its center."""
    center, radius, (z_lo, z_hi) = _cloud_bounds(cloud)
    heights = np.linspace(z_lo + 0.15 * (z_hi - z_lo), z_hi * 0.85, n_elevation) - center[2]
    return [
        _orbit_pose(center, radius, 2 * np.pi * i / n_azimuth, h)
        for h in heights
        for i in range(n_azimuth)
    ]


def _observed_mask(cloud: GaussianCloud, result: RenderResult) -> np.ndarray:
    """Which rendered pixels are painted by OBSERVED (not PRIOR) splats."""
    mask = np.zeros(result.splat_index.shape, np.float32)
    hit = result.hit_mask
    if not hit.any():
        return mask
    rows, cols = np.where(hit)
    observed = cloud.provenance[result.splat_index[hit]] == Provenance.OBSERVED
    mask[rows[observed], cols[observed]] = 1.0
    return mask


def _angle_between(a: CameraPose, b: CameraPose, center: np.ndarray) -> float:
    va = a.camera_center - center
    vb = b.camera_center - center
    va = va / max(np.linalg.norm(va), 1e-9)
    vb = vb / max(np.linalg.norm(vb), 1e-9)
    return float(np.arccos(np.clip(np.dot(va, vb), -1.0, 1.0)))


class RenderVerifier:
    """Sweeps candidate camera poses and scores a photo against each render.

    `renderer` need not support gradients (`PointRenderer` is fine and cheap);
    this only ever reads renders, never optimizes through one.
    """

    def __init__(
        self,
        renderer: SplatRenderer,
        embedder: Embedder,
        n_azimuth: int = 24,
        n_elevation: int = 3,
        min_observed_px: int = 400,
        ambiguity_angle_deg: float = 60.0,
        ambiguity_margin: float = 0.05,
        refine_steps: int = 7,
    ):
        self._renderer = renderer
        self._embedder = embedder
        self._n_azimuth = n_azimuth
        self._n_elevation = n_elevation
        self._min_observed_px = min_observed_px
        self._ambiguity_angle_deg = ambiguity_angle_deg
        self._ambiguity_margin = ambiguity_margin
        self._refine_steps = refine_steps

    def verify(
        self,
        cloud: GaussianCloud,
        image_rgb: np.ndarray,
        mask: np.ndarray | None,
        intrinsics: Intrinsics,
    ) -> VerifyResult:
        if cloud.n == 0 or not (cloud.provenance == Provenance.OBSERVED).any():
            return VerifyResult(None, 0.0, False, 0)  # nothing confirmed to match yet

        query = self._embedder.embed(image_rgb, mask)
        center, radius, _ = _cloud_bounds(cloud)

        coarse = self._score(cloud, candidate_poses(cloud, self._n_azimuth, self._n_elevation),
                              intrinsics, query)
        if not coarse:
            return VerifyResult(None, 0.0, False, 0)

        best = max(coarse, key=lambda c: c.score)
        ambiguous = self._is_ambiguous(coarse, best, center)

        # local refine: a finer sweep around the winning angle to cut down the
        # coarse grid's angular quantization error before handing off a pose
        if self._refine_steps > 1:
            step = 2 * np.pi / self._n_azimuth
            base_angle = np.arctan2(
                best.pose.camera_center[1] - center[1], best.pose.camera_center[0] - center[0]
            )
            height = best.pose.camera_center[2] - center[2]
            fine_poses = [
                _orbit_pose(center, radius, a, height)
                for a in np.linspace(base_angle - step, base_angle + step, self._refine_steps)
            ]
            fine = self._score(cloud, fine_poses, intrinsics, query)
            if fine:
                fine_best = max(fine, key=lambda c: c.score)
                if fine_best.score >= best.score:
                    best = fine_best

        return VerifyResult(best.pose, best.score, ambiguous, best.observed_px)

    def _score(
        self, cloud: GaussianCloud, poses: list[CameraPose], intrinsics: Intrinsics,
        query: np.ndarray,
    ) -> list[_Candidate]:
        out = []
        for pose in poses:
            result = self._renderer.render(cloud, pose, intrinsics)
            observed_mask = _observed_mask(cloud, result)
            n_observed = int(observed_mask.sum())
            if n_observed < self._min_observed_px:
                continue
            rendered_rgb = np.clip(result.color * 255.0, 0, 255).astype(np.uint8)
            candidate_embed = self._embedder.embed(rendered_rgb, observed_mask)
            score = Embedder.similarity(query, candidate_embed)
            out.append(_Candidate(pose, score, n_observed))
        return out

    def _is_ambiguous(
        self, candidates: list[_Candidate], best: _Candidate, center: np.ndarray
    ) -> bool:
        far_scores = [
            c.score for c in candidates
            if _angle_between(c.pose, best.pose, center) >= np.radians(self._ambiguity_angle_deg)
        ]
        if not far_scores:
            return False  # nothing far enough away to be a symmetric confusion
        return (best.score - max(far_scores)) < self._ambiguity_margin
