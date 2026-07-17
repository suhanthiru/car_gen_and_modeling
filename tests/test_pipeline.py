"""Pipeline orchestration: bootstrap, photo fusion, video walk-arounds, tracking."""
from __future__ import annotations

import numpy as np
import pytest

from cargen.core.asset import VehicleAsset
from cargen.core.splat import Provenance
from cargen.pipeline import DEVICE_EVIDENCE_WEIGHT, Pipeline, evidence_weight_for
from cargen.pose_estimation.stub import StubRegistrar
from cargen.prior_generation.interface import PriorGenerator
from cargen.reid.histogram import HistogramEmbedder
from cargen.video.frame_sampler import FrameSampler
from demo.synthetic import BackgroundSegmenter, orbit_pose, render_photo, walkaround_frames


class _FixedPrior(PriorGenerator):
    """Stands in for the image-to-3D model so tests measure fusion, not priors."""

    def __init__(self, cloud):
        self._cloud = cloud

    def generate_splats(self, image_rgb, mask, n_points=20_000):
        return self._cloud

    def generate(self, image_rgb, mask):
        raise NotImplementedError


@pytest.fixture
def pipeline(prior_cloud, renderer, request):
    return Pipeline(
        segmenter=BackgroundSegmenter(),
        prior_generator=_FixedPrior(prior_cloud),
        matcher=None,
        renderer=renderer,
        embedder=HistogramEmbedder(),
        registrar=StubRegistrar(confidence=0.9),
        sampler=FrameSampler(motion_threshold=4.0, max_frames=10),
    )


class TestEvidenceWeights:
    def test_tiers_are_ordered(self):
        w = DEVICE_EVIDENCE_WEIGHT
        assert w["phone_ar"] > w["phone"] > w["pi"] > w["cctv"]

    def test_unknown_device_gets_middling_weight(self):
        assert evidence_weight_for(None) == evidence_weight_for("nonsense") == 0.5

    def test_case_insensitive(self):
        assert evidence_weight_for("CCTV") == DEVICE_EVIDENCE_WEIGHT["cctv"]


class TestBootstrap:
    def test_first_photo_creates_a_complete_model(
        self, pipeline, truth_cloud, intrinsics, renderer
    ):
        asset = VehicleAsset(name="test")
        photo, _ = render_photo(truth_cloud, orbit_pose(0.0), intrinsics, renderer)
        result = pipeline.ingest_photo(asset, photo, intrinsics=intrinsics, timestamp=0.0)

        assert result.created
        assert asset.cloud.n > 0
        # a whole car exists, but only the photographed side is real
        assert 0.0 < asset.cloud.observed_fraction() < 1.0
        assert len(asset.observations) == 1
        assert result.embedding is not None

    def test_bootstrap_records_landmarks_for_future_registration(
        self, pipeline, truth_cloud, intrinsics, renderer
    ):
        asset = VehicleAsset(name="test")
        photo, _ = render_photo(truth_cloud, orbit_pose(0.0), intrinsics, renderer)
        pipeline.ingest_photo(asset, photo, intrinsics=intrinsics, timestamp=0.0)
        store = pipeline.landmarks_for(asset)
        assert store.n_views == 1
        assert store.points_3d[0].shape[0] > 0

    def test_intrinsics_inferred_from_image(self, pipeline, truth_cloud, intrinsics, renderer):
        photo, _ = render_photo(truth_cloud, orbit_pose(0.0), intrinsics, renderer)
        asset = VehicleAsset(name="test")
        pipeline.ingest_photo(asset, photo, timestamp=0.0)  # no intrinsics passed
        assert asset.cloud.n > 0


class TestPhotoFusion:
    def test_known_pose_bypasses_registration(
        self, pipeline, prior_cloud, truth_cloud, intrinsics, renderer
    ):
        asset = VehicleAsset(name="test", cloud=prior_cloud)
        photo, _ = render_photo(truth_cloud, orbit_pose(0.7), intrinsics, renderer)
        result = pipeline.ingest_photo(
            asset, photo, intrinsics=intrinsics, timestamp=1.0,
            known_pose=orbit_pose(0.7),
        )
        assert result.frames_fused == 1
        assert asset.cloud.observed_fraction() > 0

    def test_unregisterable_photo_is_recorded_but_not_fused(
        self, pipeline, prior_cloud, truth_cloud, intrinsics, renderer
    ):
        """The asset must be unchanged, and the attempt must still be logged."""
        asset = VehicleAsset(name="test", cloud=prior_cloud)
        before = asset.cloud
        photo, _ = render_photo(truth_cloud, orbit_pose(2.0), intrinsics, renderer)
        result = pipeline.ingest_photo(asset, photo, intrinsics=intrinsics, timestamp=1.0)

        assert result.frames_rejected == 1
        assert result.frames_fused == 0
        assert asset.cloud is before
        assert asset.observations[-1]["rejected"] is True


class TestVideo:
    def test_walkaround_raises_observed_fraction(
        self, pipeline, prior_cloud, truth_cloud, intrinsics, renderer
    ):
        asset = VehicleAsset(name="test", cloud=prior_cloud)
        frames, poses = walkaround_frames(
            truth_cloud, intrinsics, renderer, n_frames=12
        )
        pipeline.registrar = StubRegistrar(confidence=0.9, poses=poses)
        result = pipeline.ingest_video(
            asset, iter(frames), intrinsics=intrinsics, timestamp=0.0
        )
        assert result.frames_sampled > 1
        assert result.frames_fused > 0
        assert asset.cloud.observed_fraction() > 0.3
        assert asset.observations[-1]["kind"] == "video"

    def test_video_can_bootstrap_from_nothing(
        self, pipeline, truth_cloud, intrinsics, renderer
    ):
        asset = VehicleAsset(name="test")
        frames, poses = walkaround_frames(truth_cloud, intrinsics, renderer, n_frames=8)
        pipeline.registrar = StubRegistrar(confidence=0.9, poses=poses)
        result = pipeline.ingest_video(
            asset, iter(frames), intrinsics=intrinsics, timestamp=0.0
        )
        assert result.created
        assert asset.cloud.n > 0

    def test_known_poses_are_used(self, pipeline, prior_cloud, truth_cloud, intrinsics, renderer):
        asset = VehicleAsset(name="test", cloud=prior_cloud)
        frames, poses = walkaround_frames(truth_cloud, intrinsics, renderer, n_frames=10)
        sampled = pipeline.sampler.sample(iter(frames))
        known = [poses[f.index] for f in sampled]
        result = pipeline.ingest_video(
            asset, iter(frames), intrinsics=intrinsics, timestamp=0.0, known_poses=known
        )
        assert result.frames_fused == len(sampled)
        assert result.frames_rejected == 0

    def test_empty_video_is_harmless(self, pipeline):
        asset = VehicleAsset(name="test")
        result = pipeline.ingest_video(asset, iter([]))
        assert result.frames_sampled == 0
        assert asset.cloud.n == 0

    def test_rejected_frames_are_counted_not_fused(
        self, pipeline, prior_cloud, truth_cloud, intrinsics, renderer
    ):
        asset = VehicleAsset(name="test", cloud=prior_cloud)
        frames, _ = walkaround_frames(truth_cloud, intrinsics, renderer, n_frames=8)
        pipeline.registrar = StubRegistrar(confidence=0.9)  # no poses → always rejects
        result = pipeline.ingest_video(
            asset, iter(frames), intrinsics=intrinsics, timestamp=0.0
        )
        assert result.frames_fused == 0
        assert result.frames_rejected > 0
        assert asset.cloud.observed_fraction() == pytest.approx(
            prior_cloud.observed_fraction()
        )


class TestIngestResult:
    def test_summary_shape(self, pipeline, truth_cloud, intrinsics, renderer):
        asset = VehicleAsset(name="test")
        photo, _ = render_photo(truth_cloud, orbit_pose(0.0), intrinsics, renderer)
        summary = pipeline.ingest_photo(
            asset, photo, intrinsics=intrinsics, timestamp=0.0
        ).summary()
        for key in ("created", "frames_fused", "splats", "observed_fraction"):
            assert key in summary
