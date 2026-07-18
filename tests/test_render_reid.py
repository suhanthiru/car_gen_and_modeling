"""Render-based verifier: the fallback path for registering a photo that has
too little keypoint overlap with anything already confirmed for PnP to bridge.
"""
from __future__ import annotations

import numpy as np
import pytest

from cargen.core.camera import CameraPose, Intrinsics
from cargen.core.splat import Provenance
from cargen.fusion_engine.engine import FusionConfig, FusionEngine
from cargen.pose_estimation.fallback import FallbackRegistrar
from cargen.pose_estimation.interface import Registration
from cargen.pose_estimation.render_reid import RenderReidRegistrar
from cargen.pose_estimation.stub import StubRegistrar
from cargen.reid.histogram import HistogramEmbedder
from cargen.reid.verify import RenderVerifier, VerifyResult, candidate_poses
from demo.synthetic import orbit_pose, render_photo


@pytest.fixture
def confirmed_cloud(prior_cloud, truth_cloud, intrinsics, renderer):
    """prior_cloud with the angle=0 side fused in and confirmed OBSERVED."""
    engine = FusionEngine(renderer, FusionConfig())
    photo, mask = render_photo(truth_cloud, orbit_pose(0.0), intrinsics, renderer, seed=1)
    cloud, report = engine.fuse_frame(
        prior_cloud, photo, mask, orbit_pose(0.0), intrinsics,
        registration_confidence=0.9, timestamp=1.0,
    )
    assert report.accepted
    assert (cloud.provenance == Provenance.OBSERVED).any()
    return cloud


class TestCandidatePoses:
    def test_returns_full_grid(self, confirmed_cloud):
        poses = candidate_poses(confirmed_cloud, n_azimuth=12, n_elevation=3)
        assert len(poses) == 36
        assert all(isinstance(p, CameraPose) for p in poses)


class TestRenderVerifier:
    def test_rejects_when_nothing_is_confirmed_yet(self, prior_cloud, intrinsics, renderer):
        verifier = RenderVerifier(renderer, HistogramEmbedder(), n_azimuth=8, n_elevation=1)
        photo, mask = render_photo(prior_cloud, orbit_pose(0.0), intrinsics, renderer, seed=2)
        result = verifier.verify(prior_cloud, photo, mask, intrinsics)
        assert result.pose is None
        assert result.score == 0.0

    def test_rejects_on_an_empty_cloud(self, intrinsics, renderer):
        from cargen.core.splat import GaussianCloud

        verifier = RenderVerifier(renderer, HistogramEmbedder())
        photo = np.zeros((intrinsics.height, intrinsics.width, 3), np.uint8)
        result = verifier.verify(GaussianCloud.empty(), photo, None, intrinsics)
        assert result.pose is None

    def test_registers_a_moderately_offset_view_against_the_confirmed_side(
        self, confirmed_cloud, truth_cloud, intrinsics, renderer
    ):
        verifier = RenderVerifier(
            renderer, HistogramEmbedder(), n_azimuth=16, n_elevation=1, min_observed_px=30
        )
        query_angle = np.radians(35)  # partial overlap with the angle=0 confirmed side
        photo, mask = render_photo(truth_cloud, orbit_pose(query_angle), intrinsics, renderer, seed=5)
        result = verifier.verify(confirmed_cloud, photo, mask, intrinsics)

        assert result.pose is not None
        true_center = orbit_pose(query_angle).camera_center[:2]
        got_center = result.pose.camera_center[:2]
        cos_sim = float(
            np.dot(got_center, true_center)
            / (np.linalg.norm(got_center) * np.linalg.norm(true_center) + 1e-9)
        )
        assert cos_sim > 0.7  # roughly the right side of the car, not a wild miss

    def test_ambiguity_flag_catches_symmetric_confusion(self, intrinsics, renderer):
        """A cloud confirmed identically on two opposite sides (contrived, but
        exactly the bilateral-symmetry failure mode the module docstring
        names) must not silently hand back one of the two as if it were sure."""
        from cargen.core.splat import GaussianCloud

        rng = np.random.default_rng(0)
        n = 400
        # two mirrored blobs, identical color -> indistinguishable by appearance
        half = n // 2
        positions = np.zeros((n, 3), np.float32)
        positions[:half] = rng.normal([1.0, 0, 0.5], 0.3, size=(half, 3)).astype(np.float32)
        positions[half:] = rng.normal([-1.0, 0, 0.5], 0.3, size=(n - half, 3)).astype(np.float32)
        cloud = GaussianCloud.create(
            positions=positions,
            colors=np.full((n, 3), 0.6, np.float32),
            scales=np.full((n, 3), 0.08, np.float32),
            rotations=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
            opacities=np.ones(n, np.float32),
            confidence=1.0,
        )
        cloud.provenance[:] = Provenance.OBSERVED

        verifier = RenderVerifier(
            renderer, HistogramEmbedder(), n_azimuth=12, n_elevation=1,
            min_observed_px=20, ambiguity_angle_deg=90.0,
        )
        photo, mask = render_photo(cloud, orbit_pose(0.0), intrinsics, renderer, seed=9)
        result = verifier.verify(cloud, photo, mask, intrinsics)
        assert result.ambiguous


class _FakeVerifier:
    """Duck-typed RenderVerifier double, so RenderReidRegistrar's Registration
    mapping is testable independent of the real search's numeric behavior."""

    def __init__(self, result: VerifyResult):
        self._result = result

    def verify(self, cloud, image_rgb, mask, intrinsics):
        return self._result


class TestRenderReidRegistrar:
    def _asset(self, cloud):
        class _Asset:
            pass

        a = _Asset()
        a.cloud = cloud
        return a

    def test_maps_a_good_match_to_an_accepted_registration(self, small_cloud, intrinsics):
        pose = CameraPose.identity()
        registrar = RenderReidRegistrar(
            _FakeVerifier(VerifyResult(pose, 0.8, False, 500)), min_score=0.55
        )
        result = registrar.register(
            np.zeros((4, 4, 3), np.uint8), None, intrinsics,
            {"asset": self._asset(small_cloud)},
        )
        assert result.ok
        assert result.confidence == 0.8
        assert result.inliers == 500

    def test_rejects_when_no_pose_found(self, small_cloud, intrinsics):
        registrar = RenderReidRegistrar(_FakeVerifier(VerifyResult(None, 0.0, False, 0)))
        result = registrar.register(
            np.zeros((4, 4, 3), np.uint8), None, intrinsics,
            {"asset": self._asset(small_cloud)},
        )
        assert not result.ok

    def test_rejects_ambiguous_matches(self, small_cloud, intrinsics):
        pose = CameraPose.identity()
        registrar = RenderReidRegistrar(_FakeVerifier(VerifyResult(pose, 0.9, True, 500)))
        result = registrar.register(
            np.zeros((4, 4, 3), np.uint8), None, intrinsics,
            {"asset": self._asset(small_cloud)},
        )
        assert not result.ok
        assert "ambiguous" in result.reason

    def test_rejects_weak_matches_below_threshold(self, small_cloud, intrinsics):
        pose = CameraPose.identity()
        registrar = RenderReidRegistrar(
            _FakeVerifier(VerifyResult(pose, 0.3, False, 500)), min_score=0.55
        )
        result = registrar.register(
            np.zeros((4, 4, 3), np.uint8), None, intrinsics,
            {"asset": self._asset(small_cloud)},
        )
        assert not result.ok

    def test_rejects_when_asset_has_no_cloud(self, intrinsics):
        registrar = RenderReidRegistrar(_FakeVerifier(VerifyResult(None, 0, False, 0)))
        result = registrar.register(
            np.zeros((4, 4, 3), np.uint8), None, intrinsics, {"asset": None}
        )
        assert not result.ok


class TestFallbackRegistrar:
    def test_first_success_short_circuits(self, intrinsics):
        first = StubRegistrar(confidence=0.9, poses=[CameraPose.identity()])
        second = StubRegistrar(confidence=0.9, poses=[CameraPose.identity()])
        chain = FallbackRegistrar([first, second], min_confidence=0.35)
        result = chain.register(np.zeros((2, 2, 3), np.uint8), None, intrinsics, {})
        assert result.ok
        assert first.calls == 1
        assert second.calls == 0

    def test_falls_through_to_second_when_first_is_weak(self, intrinsics):
        first = StubRegistrar(confidence=0.1, poses=[CameraPose.identity()])
        second = StubRegistrar(confidence=0.9, poses=[CameraPose.identity()])
        chain = FallbackRegistrar([first, second], min_confidence=0.35)
        result = chain.register(np.zeros((2, 2, 3), np.uint8), None, intrinsics, {})
        assert result.ok
        assert result.confidence == 0.9
        assert first.calls == 1 and second.calls == 1

    def test_returns_best_reject_when_everyone_fails(self, intrinsics):
        # StubRegistrar without a pose source always rejects at confidence 0.0;
        # use fail_after=0 to force rejection while still recording the call.
        first = StubRegistrar(fail_after=0, poses=[CameraPose.identity()])
        second = StubRegistrar(fail_after=0, poses=[CameraPose.identity()])
        chain = FallbackRegistrar([first, second], min_confidence=0.35)
        result = chain.register(np.zeros((2, 2, 3), np.uint8), None, intrinsics, {})
        assert not result.ok
        assert first.calls == 1 and second.calls == 1

    def test_requires_at_least_one_registrar(self):
        with pytest.raises(ValueError):
            FallbackRegistrar([])
