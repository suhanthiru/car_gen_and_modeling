"""Stub registrar: returns poses supplied by the caller.

Used by the synthetic demo and tests, where ground-truth camera poses are known
by construction, so fusion arbitration can be exercised without a real matcher.
"""
from __future__ import annotations

import numpy as np

from cargen.core.camera import CameraPose, Intrinsics
from cargen.pose_estimation.interface import Registration, Registrar


class StubRegistrar(Registrar):
    """Returns poses the caller already knows, instead of estimating them.

    Pose comes from `context["true_pose"]` if present, else from the `poses`
    sequence in call order — the latter lets tests drive multi-frame fusion
    through interfaces that have nowhere to put a pose (e.g. the HTTP API).

    `confidence` is configurable so tests can exercise the fusion engine's
    rejection path; `fail_after` makes registration start failing mid-sequence,
    which is how tracking-loss recovery gets tested.
    """

    def __init__(
        self,
        confidence: float = 0.9,
        fail_after: int | None = None,
        poses: list[CameraPose] | None = None,
    ):
        self._confidence = confidence
        self._fail_after = fail_after
        self._poses = poses
        self.calls = 0

    def register(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray | None,
        intrinsics: Intrinsics,
        context: dict,
    ) -> Registration:
        index = self.calls
        self.calls += 1
        if self._fail_after is not None and self.calls > self._fail_after:
            return Registration.rejected("stub configured to fail")

        pose: CameraPose | None = context.get("true_pose")
        if pose is None and self._poses:
            pose = self._poses[index % len(self._poses)]
        if pose is None:
            return Registration.rejected(
                "stub registrar needs context['true_pose'] or a poses sequence"
            )
        return Registration(
            pose=pose,
            confidence=self._confidence,
            inliers=100,
            reprojection_error=0.5,
            reason="stub: known pose",
        )
