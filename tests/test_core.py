"""Splat cloud, cameras, Sim(3), and asset persistence."""
from __future__ import annotations

import numpy as np
import pytest

from cargen.core.asset import FrameRecord, VehicleAsset
from cargen.core.camera import CameraPose, Intrinsics, Sim3, umeyama
from cargen.core.splat import SH_REST_COEFFS, GaussianCloud, Provenance


class TestGaussianCloud:
    def test_create_fills_defaults(self):
        cloud = GaussianCloud.create(
            positions=np.zeros((5, 3)), colors=np.ones((5, 3))
        )
        assert cloud.n == 5
        assert cloud.provenance.tolist() == [Provenance.PRIOR] * 5
        assert cloud.rotations.shape == (5, 4)
        assert np.allclose(cloud.rotations[:, 0], 1.0)  # identity quaternion

    @staticmethod
    def _fields(n=3, **override):
        fields = dict(
            positions=np.zeros((n, 3), np.float32),
            scales=np.ones((n, 3), np.float32),
            rotations=np.ones((n, 4), np.float32),
            opacities=np.ones(n, np.float32),
            colors=np.ones((n, 3), np.float32),
            sh_rest=np.zeros((n, SH_REST_COEFFS, 3), np.float32),
            provenance=np.zeros(n, np.uint8),
            confidence=np.zeros(n, np.float32),
            view_count=np.zeros(n, np.int32),
            last_seen_ts=np.zeros(n, np.float64),
        )
        fields.update(override)
        return fields

    def test_mismatched_shapes_rejected(self):
        with pytest.raises(ValueError, match="colors"):
            GaussianCloud(**self._fields(colors=np.ones((2, 3), np.float32)))

    def test_mismatched_sh_rest_rejected(self):
        with pytest.raises(ValueError, match="sh_rest"):
            GaussianCloud(**self._fields(sh_rest=np.zeros((3, 9, 3), np.float32)))

    def test_with_updates_does_not_mutate_source(self):
        cloud = GaussianCloud.create(np.zeros((4, 3)), np.zeros((4, 3)))
        updated = cloud.with_updates(
            np.array([0, 1]), confidence=np.array([0.9, 0.9], np.float32)
        )
        assert cloud.confidence == pytest.approx([0.1] * 4)  # original untouched
        assert updated.confidence[0] == pytest.approx(0.9)
        assert updated.confidence[2] == pytest.approx(0.1)

    def test_with_updates_rejects_unknown_field(self, small_cloud):
        with pytest.raises(ValueError, match="unknown fields"):
            small_cloud.with_updates(np.array([0]), bogus=np.array([1]))

    def test_select_and_concat(self, small_cloud):
        half = small_cloud.select(np.arange(50))
        assert half.n == 50
        assert half.concat(half).n == 100

    def test_observed_fraction(self):
        cloud = GaussianCloud.create(
            np.zeros((4, 3)), np.zeros((4, 3)),
            provenance=np.array([0, 1, 1, 0], np.uint8),
        )
        assert cloud.observed_fraction() == pytest.approx(0.5)
        assert cloud.stats()["observed"] == 2

    def test_empty_cloud_is_safe(self):
        empty = GaussianCloud.empty()
        assert empty.n == 0
        assert empty.observed_fraction() == 0.0
        assert empty.stats()["splats"] == 0


class TestCamera:
    def test_look_at_places_camera(self):
        pose = CameraPose.look_at(eye=(5, 0, 0), target=(0, 0, 0))
        assert np.allclose(pose.camera_center, [5, 0, 0], atol=1e-6)

    def test_projection_puts_target_at_principal_point(self, ):
        intr = Intrinsics.simple(100, 100)
        pose = CameraPose.look_at(eye=(3, 0, 0), target=(0, 0, 0))
        uv, z = pose.project(np.zeros((1, 3)), intr)
        assert uv[0] == pytest.approx([intr.cx, intr.cy], abs=1e-4)
        assert z[0] > 0  # in front of the camera

    def test_points_behind_camera_have_negative_depth(self):
        intr = Intrinsics.simple(100, 100)
        pose = CameraPose.look_at(eye=(3, 0, 0), target=(0, 0, 0))
        _, z = pose.project(np.array([[9.0, 0, 0]]), intr)
        assert z[0] < 0

    def test_intrinsics_scaled(self):
        intr = Intrinsics.simple(1000, 500).scaled(0.5)
        assert (intr.width, intr.height) == (500, 250)
        assert intr.cx == pytest.approx(250)


class TestSim3:
    def test_umeyama_recovers_known_transform(self):
        rng = np.random.default_rng(2)
        src = rng.normal(size=(40, 3))
        truth = Sim3(
            s=2.5,
            R=CameraPose.look_at((1, 1, 1), (0, 0, 0)).R,
            t=np.array([0.3, -0.2, 5.0]),
        )
        est = umeyama(src, truth.apply(src))
        assert est.s == pytest.approx(2.5, abs=1e-9)
        assert np.allclose(est.apply(src), truth.apply(src), atol=1e-9)

    def test_umeyama_without_scale(self):
        rng = np.random.default_rng(3)
        src = rng.normal(size=(20, 3))
        est = umeyama(src, src * 3.0, with_scale=False)
        assert est.s == 1.0

    def test_umeyama_needs_enough_points(self):
        with pytest.raises(ValueError):
            umeyama(np.zeros((2, 3)), np.zeros((2, 3)))

    def test_inverse_and_compose(self):
        t = Sim3(s=2.0, R=CameraPose.look_at((1, 2, 3), (0, 0, 0)).R, t=np.array([1.0, 2, 3]))
        pts = np.random.default_rng(4).normal(size=(10, 3))
        assert np.allclose(t.inverse().apply(t.apply(pts)), pts, atol=1e-9)
        assert np.allclose(t.compose(Sim3.identity()).apply(pts), t.apply(pts), atol=1e-9)


class TestVehicleAsset:
    def test_save_load_roundtrip(self, tmp_path, small_cloud):
        asset = VehicleAsset(name="Bob's Civic", cloud=small_cloud)
        asset.add_observation({"kind": "photo", "device": "phone"}, np.ones(8, np.float32))
        asset.aliases.append("the-red-one")
        asset.save(tmp_path / "v")

        loaded = VehicleAsset.load(tmp_path / "v")
        assert loaded.name == "Bob's Civic"
        assert loaded.vehicle_id == asset.vehicle_id
        assert loaded.aliases == ["the-red-one"]
        assert loaded.cloud.n == small_cloud.n
        assert np.allclose(loaded.cloud.positions, small_cloud.positions)
        assert np.array_equal(loaded.cloud.provenance, small_cloud.provenance)
        assert len(loaded.observations) == 1
        assert loaded.embeddings.shape == (1, 8)

    def test_embeddings_accumulate(self, small_cloud):
        asset = VehicleAsset(name="x", cloud=small_cloud)
        asset.add_observation({}, np.array([1.0, 0.0], np.float32))
        asset.add_observation({}, np.array([0.0, 1.0], np.float32))
        assert asset.embeddings.shape == (2, 2)
        mean = asset.mean_embedding()
        assert np.linalg.norm(mean) == pytest.approx(1.0, abs=1e-6)

    def test_mean_embedding_none_when_empty(self, small_cloud):
        assert VehicleAsset(name="x", cloud=small_cloud).mean_embedding() is None

    def test_is_asset_dir(self, tmp_path, small_cloud):
        assert not VehicleAsset.is_asset_dir(tmp_path)
        VehicleAsset(name="x", cloud=small_cloud).save(tmp_path / "v")
        assert VehicleAsset.is_asset_dir(tmp_path / "v")

    def test_loads_assets_written_before_sh_rest_existed(self, tmp_path, small_cloud):
        """Assets on disk predate the SH bands. They are real captures the user
        cannot re-take, so they must keep loading — with zeros, which is the
        honest value: those clouds genuinely had no view-dependent appearance."""
        VehicleAsset(name="old", cloud=small_cloud).save(tmp_path / "v")

        # rewrite cloud.npz without sh_rest, as an older build would have
        path = tmp_path / "v" / "cloud.npz"
        with np.load(path) as data:
            legacy = {k: data[k] for k in data.files if k != "sh_rest"}
        assert "sh_rest" not in legacy
        np.savez_compressed(path, **legacy)

        loaded = VehicleAsset.load(tmp_path / "v")
        assert loaded.cloud.n == small_cloud.n
        assert loaded.cloud.sh_rest.shape == (small_cloud.n, SH_REST_COEFFS, 3)
        assert not loaded.cloud.is_view_dependent

    def test_missing_required_field_still_raises(self, tmp_path, small_cloud):
        """The migration must not turn genuine corruption into silence."""
        VehicleAsset(name="x", cloud=small_cloud).save(tmp_path / "v")
        path = tmp_path / "v" / "cloud.npz"
        with np.load(path) as data:
            broken = {k: data[k] for k in data.files if k != "opacities"}
        np.savez_compressed(path, **broken)
        with pytest.raises(KeyError, match="opacities"):
            VehicleAsset.load(tmp_path / "v")

    def test_add_frame_returns_sequential_index(self, small_cloud):
        asset = VehicleAsset(name="x", cloud=small_cloud)
        image = np.zeros((4, 6, 3), np.uint8)
        pose = CameraPose.identity()
        intr = Intrinsics.simple(6, 4)
        assert asset.add_frame(image, None, pose, intr) == 0
        assert asset.add_frame(image, None, pose, intr) == 1
        assert len(asset.load_frames()) == 2

    def test_save_load_frames_roundtrip(self, tmp_path, small_cloud):
        asset = VehicleAsset(name="x", cloud=small_cloud)
        rng = np.random.default_rng(7)
        image = rng.integers(0, 255, size=(8, 10, 3), dtype=np.uint8)
        mask = rng.random((8, 10)).astype(np.float32)
        pose = CameraPose.look_at(eye=(3, 0, 0), target=(0, 0, 0))
        intr = Intrinsics.simple(10, 8)
        asset.add_frame(image, mask, pose, intr, evidence_weight=0.85)
        # a second frame with no mask, to confirm has_mask=False round-trips too
        asset.add_frame(image, None, pose, intr)
        asset.save(tmp_path / "v")

        loaded = VehicleAsset.load(tmp_path / "v")
        frames = loaded.load_frames()
        assert len(frames) == 2

        f0 = frames[0]
        assert isinstance(f0, FrameRecord)
        assert f0.index == 0
        assert np.array_equal(f0.image_rgb, image)
        assert f0.mask is not None
        assert np.allclose(f0.mask, mask, atol=1.0 / 255.0)
        assert np.allclose(f0.pose.R, pose.R, atol=1e-9)
        assert np.allclose(f0.pose.t, pose.t, atol=1e-9)
        assert f0.intrinsics.width == intr.width
        assert f0.intrinsics.fx == pytest.approx(intr.fx)
        assert f0.evidence_weight == pytest.approx(0.85)

        f1 = frames[1]
        assert f1.mask is None
        assert f1.evidence_weight == pytest.approx(1.0)

    def test_loads_assets_written_before_frames_existed(self, tmp_path, small_cloud):
        """Assets saved before frame/pose persistence existed have no `frames/`
        directory or `poses.json` at all. They must keep loading, with an
        empty frame list rather than an error — the same migration reasoning
        as `load_cloud_fields`'s missing `sh_rest`."""
        VehicleAsset(name="old", cloud=small_cloud).save(tmp_path / "v")
        import shutil
        shutil.rmtree(tmp_path / "v" / "frames")
        (tmp_path / "v" / "poses.json").unlink()

        loaded = VehicleAsset.load(tmp_path / "v")
        assert loaded.load_frames() == []
