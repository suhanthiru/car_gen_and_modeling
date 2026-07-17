"""Frame-to-frame tracking for video — cheaper than re-registering every frame.

Consecutive video frames have small baselines and high overlap, which is a much
easier problem than matching two disconnected photos. So only the first frame of
a clip pays for full registration against the landmark store; the rest chain
from their predecessor's pose via 2D-3D correspondences carried forward.

Drift is the cost of chaining. `reregister_every` forces a periodic global
registration to pull the chain back onto the canonical frame.
"""
from __future__ import annotations

import numpy as np

from cargen.core.camera import CameraPose, Intrinsics
from cargen.pose_estimation.interface import Registration, Registrar
from cargen.pose_estimation.registration import (
    MIN_PNP_POINTS,
    registration_confidence,
    solve_pnp,
)


class VideoTracker:
    """Tracks camera poses across a frame sequence.

    Needs a way to turn the previous frame's pixels into 3D points: the
    `depth_lookup` callable takes (pose, uv array) and returns (points_3d,
    valid_mask) — normally implemented by rendering the current splats and
    reading back depth.
    """

    def __init__(
        self,
        matcher,
        registrar: Registrar,
        intrinsics: Intrinsics,
        depth_lookup,
        min_confidence: float = 0.3,
        reregister_every: int = 12,
        reproj_threshold: float = 4.0,
    ):
        self._matcher = matcher
        self._registrar = registrar
        self._intrinsics = intrinsics
        self._depth_lookup = depth_lookup
        self._min_confidence = min_confidence
        self._reregister_every = reregister_every
        self._reproj_threshold = reproj_threshold
        self._prev_frame: np.ndarray | None = None
        self._prev_mask: np.ndarray | None = None
        self._prev_pose: CameraPose | None = None
        self._since_global = 0

    def reset(self) -> None:
        self._prev_frame = None
        self._prev_mask = None
        self._prev_pose = None
        self._since_global = 0

    def track(
        self, frame: np.ndarray, mask: np.ndarray | None, context: dict
    ) -> Registration:
        """Pose for one frame: chain from the previous, else register globally."""
        needs_global = (
            self._prev_pose is None or self._since_global >= self._reregister_every
        )
        if not needs_global:
            result = self._track_from_previous(frame, mask)
            if result.accepted_at(self._min_confidence):
                self._remember(frame, mask, result.pose, chained=True)
                return result
            # tracking lost (occlusion, fast motion) — fall back to global
        result = self._registrar.register(frame, mask, self._intrinsics, context)
        if result.accepted_at(self._min_confidence):
            self._remember(frame, mask, result.pose, chained=False)
        return result

    def _remember(self, frame, mask, pose, chained: bool) -> None:
        self._prev_frame, self._prev_mask, self._prev_pose = frame, mask, pose
        self._since_global = self._since_global + 1 if chained else 0

    def _track_from_previous(
        self, frame: np.ndarray, mask: np.ndarray | None
    ) -> Registration:
        result = self._matcher.match(self._prev_frame, frame, self._prev_mask, mask)
        if result.count < MIN_PNP_POINTS:
            return Registration.rejected(
                f"frame-to-frame matches too few ({result.count})", inliers=result.count
            )
        points_3d, valid = self._depth_lookup(self._prev_pose, result.points_a)
        if valid.sum() < MIN_PNP_POINTS:
            return Registration.rejected(
                f"only {int(valid.sum())} tracked points hit geometry",
                inliers=int(valid.sum()),
            )
        pose, inlier_mask, error = solve_pnp(
            points_3d[valid],
            result.points_b[valid].astype(np.float64),
            self._intrinsics,
            self._reproj_threshold,
        )
        if pose is None:
            return Registration.rejected("frame-to-frame PnP failed")
        inliers = int(inlier_mask.sum())
        confidence = registration_confidence(
            inliers, int(valid.sum()), error, self._reproj_threshold
        )
        return Registration(
            pose=pose,
            confidence=confidence,
            inliers=inliers,
            reprojection_error=error,
            reason="tracked from previous frame",
        )
