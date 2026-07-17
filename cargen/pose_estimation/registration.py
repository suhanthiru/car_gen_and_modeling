"""PnP registration against OBSERVED landmarks, with a confidence gate.

The chicken-and-egg problem this solves: to fuse a new photo you need its pose;
to get its pose you need 2D-3D correspondences against the model — but most of
the model is a hallucinated guess, and matching real pixels against invented
geometry yields garbage poses.

Resolution: never match against PRIOR geometry. Each observation contributes
descriptor-tagged 3D landmarks (triangulated / back-projected from confirmed
splats). New frames match against *those real landmarks only*. If a photo shows
only never-observed sides, there is nothing legitimate to match — registration
reports low confidence and the frame is queued rather than fused, until a video
walk-around bridges the gap.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from cargen.core.camera import CameraPose, Intrinsics
from cargen.feature_matching.interface import FeatureMatcher
from cargen.pose_estimation.interface import Registration, Registrar

MIN_PNP_POINTS = 6


@dataclass
class Landmark:
    """A 3D point in the canonical frame, tied to the image that observed it."""

    position: np.ndarray   # (3,)
    source_uv: np.ndarray  # (2,) pixel coords in the source image
    source_index: int      # index into the landmark store's image list


class LandmarkStore:
    """Real 3D landmarks + the images they came from, per asset.

    This is the registration anchor. Only confirmed observations land here, so
    matching against it can never drag in hallucinated geometry.
    """

    def __init__(self) -> None:
        self.images: list[np.ndarray] = []
        self.masks: list[np.ndarray | None] = []
        self.points_3d: list[np.ndarray] = []   # (N_i, 3) per image
        self.points_2d: list[np.ndarray] = []   # (N_i, 2) per image

    @property
    def n_views(self) -> int:
        return len(self.images)

    def add_view(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        points_3d: np.ndarray,
        points_2d: np.ndarray,
    ) -> None:
        if points_3d.shape[0] != points_2d.shape[0]:
            raise ValueError("landmark 3D/2D counts must match")
        self.images.append(image)
        self.masks.append(mask)
        self.points_3d.append(np.asarray(points_3d, np.float64))
        self.points_2d.append(np.asarray(points_2d, np.float64))


def solve_pnp(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    intrinsics: Intrinsics,
    reproj_threshold: float = 4.0,
) -> tuple[CameraPose | None, np.ndarray, float]:
    """RANSAC PnP → (pose, inlier_mask, mean_inlier_reprojection_error)."""
    if points_3d.shape[0] < MIN_PNP_POINTS:
        return None, np.zeros((0,), bool), float("inf")

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d.astype(np.float64).reshape(-1, 1, 3),
        points_2d.astype(np.float64).reshape(-1, 1, 2),
        intrinsics.K,
        None,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=reproj_threshold,
        confidence=0.999,
        iterationsCount=500,
    )
    if not ok or inliers is None or len(inliers) < MIN_PNP_POINTS:
        return None, np.zeros((points_3d.shape[0],), bool), float("inf")

    inlier_idx = inliers.ravel()
    # Refine on inliers only — RANSAC's minimal-set estimate is coarse.
    rvec, tvec = cv2.solvePnPRefineLM(
        points_3d[inlier_idx].astype(np.float64).reshape(-1, 1, 3),
        points_2d[inlier_idx].astype(np.float64).reshape(-1, 1, 2),
        intrinsics.K,
        None,
        rvec,
        tvec,
    )
    projected, _ = cv2.projectPoints(
        points_3d[inlier_idx].astype(np.float64), rvec, tvec, intrinsics.K, None
    )
    error = float(
        np.mean(np.linalg.norm(projected.reshape(-1, 2) - points_2d[inlier_idx], axis=1))
    )
    mask = np.zeros((points_3d.shape[0],), bool)
    mask[inlier_idx] = True
    R, _ = cv2.Rodrigues(rvec)
    return CameraPose(R=R, t=tvec.ravel()), mask, error


def registration_confidence(
    inliers: int, total_matches: int, reproj_error: float, reproj_threshold: float
) -> float:
    """Blend inlier count, inlier ratio, and reprojection error into [0, 1].

    All three matter independently: many matches with a poor ratio means the
    matcher latched onto symmetric structure; a great ratio on 7 points is
    luck; low error on few points is degenerate.
    """
    if inliers < MIN_PNP_POINTS or not np.isfinite(reproj_error):
        return 0.0
    # saturates around 60 inliers — beyond that, more points add little evidence
    count_score = min(inliers / 60.0, 1.0)
    ratio_score = inliers / max(total_matches, 1)
    error_score = max(0.0, 1.0 - reproj_error / max(reproj_threshold, 1e-6))
    return float(np.clip(count_score * 0.4 + ratio_score * 0.3 + error_score * 0.3, 0.0, 1.0))


class PnPRegistrar(Registrar):
    """Match the new frame against stored real views, lift matches to 3D via
    each view's landmarks, then PnP into the canonical frame."""

    def __init__(
        self,
        matcher: FeatureMatcher,
        reproj_threshold: float = 4.0,
        neighbor_radius_px: float = 6.0,
    ):
        self._matcher = matcher
        self._reproj_threshold = reproj_threshold
        self._neighbor_radius = neighbor_radius_px

    def register(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray | None,
        intrinsics: Intrinsics,
        context: dict,
    ) -> Registration:
        store: LandmarkStore | None = context.get("landmarks")
        if store is None or store.n_views == 0:
            return Registration.rejected("no observed landmarks to register against")

        points_3d, points_2d, total_matches = self._gather_correspondences(
            image_rgb, mask, store
        )
        if points_3d.shape[0] < MIN_PNP_POINTS:
            return Registration.rejected(
                f"only {points_3d.shape[0]} 2D-3D correspondences "
                f"(need {MIN_PNP_POINTS}) — likely an unobserved side",
                inliers=points_3d.shape[0],
            )

        pose, inlier_mask, error = solve_pnp(
            points_3d, points_2d, intrinsics, self._reproj_threshold
        )
        if pose is None:
            return Registration.rejected("PnP failed to find a consistent pose")

        inliers = int(inlier_mask.sum())
        confidence = registration_confidence(
            inliers, total_matches, error, self._reproj_threshold
        )
        return Registration(
            pose=pose,
            confidence=confidence,
            inliers=inliers,
            reprojection_error=error,
            reason="registered against observed landmarks",
        )

    def _gather_correspondences(
        self, image_rgb: np.ndarray, mask: np.ndarray | None, store: LandmarkStore
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Match against every stored view; lift each match to the 3D landmark
        nearest its source-image keypoint."""
        all_3d, all_2d, total = [], [], 0
        for i in range(store.n_views):
            result = self._matcher.match(
                store.images[i], image_rgb, store.masks[i], mask
            )
            total += result.count
            if result.count == 0:
                continue
            lm_2d, lm_3d = store.points_2d[i], store.points_3d[i]
            if lm_2d.shape[0] == 0:
                continue
            # nearest stored landmark to each match's source keypoint
            d = np.linalg.norm(
                result.points_a[:, None, :] - lm_2d[None, :, :], axis=2
            )
            nearest = np.argmin(d, axis=1)
            close = d[np.arange(len(nearest)), nearest] <= self._neighbor_radius
            if not close.any():
                continue
            all_3d.append(lm_3d[nearest[close]])
            all_2d.append(result.points_b[close].astype(np.float64))

        if not all_3d:
            return np.zeros((0, 3)), np.zeros((0, 2)), total
        return np.concatenate(all_3d), np.concatenate(all_2d), total
