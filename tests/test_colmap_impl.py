"""Tests for `cargen/pose_estimation/colmap_impl.py` that don't need pycolmap.

`run_colmap_sfm` needs a real pycolmap install (not present in this
environment) and is deliberately not covered here — same convention as
`gsplat_renderer.py`/`optimize.py`, which are excluded from coverage rather
than tested against a fake CUDA stack. What IS testable without pycolmap:

  * the module imports cleanly (the lazy `import pycolmap` inside
    `run_colmap_sfm` must not run at module scope);
  * `align_colmap_to_canonical`'s Sim(3) alignment math, against a synthetic
    `ColmapResult` + `LandmarkStore` built with a known ground-truth
    rotation/scale/translation.
"""
from __future__ import annotations

import numpy as np
import pytest

from cargen.core.camera import CameraPose, Intrinsics, Sim3
from cargen.pose_estimation.colmap_impl import ColmapResult, align_colmap_to_canonical
from cargen.pose_estimation.registration import LandmarkStore


def test_module_imports_without_pycolmap():
    """Importing the module must not require pycolmap to be installed —
    `import pycolmap` has to be lazy, inside `run_colmap_sfm`, not at module
    scope. If this test fails on an environment without pycolmap, the lazy
    import discipline has been broken."""
    import cargen.pose_estimation.colmap_impl as m

    assert hasattr(m, "run_colmap_sfm")
    assert hasattr(m, "align_colmap_to_canonical")
    assert hasattr(m, "ColmapResult")


def _make_ground_truth_transform() -> Sim3:
    # A deliberately non-trivial rotation (not identity/axis-aligned), scale,
    # and translation, so the test can't pass by accident on a degenerate
    # transform.
    theta = 0.4
    axis = np.array([0.2, 0.7, 0.68], np.float64)
    axis /= np.linalg.norm(axis)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return Sim3(s=2.3, R=R, t=np.array([5.0, -2.0, 1.5]))


def _synthetic_colmap_result(rng: np.random.Generator) -> tuple[ColmapResult, Sim3]:
    """Points/poses in an arbitrary 'COLMAP frame'; landmarks in the
    'canonical frame' are the same points run through a known Sim(3), so
    `align_colmap_to_canonical` should recover that exact transform."""
    n_points = 40
    colmap_points = rng.normal(scale=3.0, size=(n_points, 3))

    intrinsics = Intrinsics.simple(width=640, height=480)
    poses = {}
    for i in range(5):
        # simple look-at-ish poses scattered around the origin
        R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(R) < 0:
            R[0] *= -1
        t = rng.normal(scale=2.0, size=3)
        poses[i] = CameraPose(R=R, t=t)

    result = ColmapResult(
        poses=poses,
        intrinsics=intrinsics,
        sparse_points=colmap_points,
        registered_fraction=1.0,
    )
    return result, colmap_points


def test_align_colmap_to_canonical_recovers_known_sim3():
    rng = np.random.default_rng(0)
    colmap_result, colmap_points = _synthetic_colmap_result(rng)
    ground_truth = _make_ground_truth_transform()

    landmark_points = ground_truth.apply(colmap_points)
    store = LandmarkStore()
    # add_view requires matching 2D/3D counts; 2D values are irrelevant here.
    dummy_2d = np.zeros((landmark_points.shape[0], 2))
    store.add_view(
        image=np.zeros((4, 4, 3), np.uint8),
        mask=None,
        points_3d=landmark_points,
        points_2d=dummy_2d,
    )

    aligned = align_colmap_to_canonical(colmap_result, store)

    assert set(aligned.keys()) == set(colmap_result.poses.keys())
    for idx, original_pose in colmap_result.poses.items():
        expected_center = ground_truth.apply(
            original_pose.camera_center.reshape(1, 3)
        ).reshape(3)
        got_center = aligned[idx].camera_center
        np.testing.assert_allclose(got_center, expected_center, atol=1e-6)

        expected_R = original_pose.R @ ground_truth.R.T
        np.testing.assert_allclose(aligned[idx].R, expected_R, atol=1e-6)


def test_align_colmap_to_canonical_raises_with_too_few_landmarks():
    rng = np.random.default_rng(1)
    colmap_result, _ = _synthetic_colmap_result(rng)

    store = LandmarkStore()
    store.add_view(
        image=np.zeros((4, 4, 3), np.uint8),
        mask=None,
        points_3d=np.zeros((2, 3)),  # fewer than the 3 points Sim(3) needs
        points_2d=np.zeros((2, 2)),
    )

    with pytest.raises(ValueError):
        align_colmap_to_canonical(colmap_result, store)


def test_align_colmap_to_canonical_raises_with_no_landmarks():
    rng = np.random.default_rng(2)
    colmap_result, _ = _synthetic_colmap_result(rng)

    store = LandmarkStore()  # no views added at all

    with pytest.raises(ValueError):
        align_colmap_to_canonical(colmap_result, store)
