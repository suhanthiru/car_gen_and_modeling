"""The HTTP surface: storage naming, ingest, queue, merge approval, downloads."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from cargen.core.asset import VehicleAsset
from cargen.core.splat import Provenance
from cargen.pipeline import Pipeline
from cargen.pose_estimation.stub import StubRegistrar
from cargen.prior_generation.interface import PriorGenerator
from cargen.reid.histogram import HistogramEmbedder
from demo.synthetic import BackgroundSegmenter, orbit_pose, render_photo
from cargen.core.camera import CameraPose, Intrinsics
from server.app import create_app, decode_image, should_autoconsolidate
from server.config import Config, sanitize_name
from server.events import Event, EventLog
from server.merge import merge_assets, merge_clouds, pick_primary
from server.queue import VehicleQueue
from server.store import VehicleStore


class _FixedPrior(PriorGenerator):
    """Stands in for the image-to-3D model. Subclasses the real interface so it
    inherits export_raw_mesh and can't drift from the contract."""

    def __init__(self, cloud):
        self._cloud = cloud

    def generate_splats(self, image_rgb, mask, n_points=20_000):
        return self._cloud

    def generate(self, image_rgb, mask):
        raise NotImplementedError


@pytest.fixture
def config(tmp_path):
    return Config(storage_root=tmp_path / "vehicles", auto_merge=False,
                  merge_threshold=0.9)


@pytest.fixture
def client(config, prior_cloud, renderer):
    pipeline = Pipeline(
        segmenter=BackgroundSegmenter(),
        prior_generator=_FixedPrior(prior_cloud),
        matcher=None,
        renderer=renderer,
        embedder=HistogramEmbedder(),
        registrar=StubRegistrar(confidence=0.9),
    )
    return TestClient(create_app(config=config, pipeline=pipeline))


@pytest.fixture
def jpeg(truth_cloud, intrinsics, renderer):
    def make(angle=0.0, cloud=None):
        img, _ = render_photo(
            cloud if cloud is not None else truth_cloud,
            orbit_pose(angle), intrinsics, renderer, seed=1,
        )
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        assert ok
        return buf.tobytes()

    return make


def post(client, jpeg_bytes, name, device="phone", filename="c.jpg"):
    return client.post(
        "/observations",
        files={"file": (filename, jpeg_bytes, "image/jpeg")},
        data={"name": name, "device": device},
    )


class TestNameSanitization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Bob's Civic!", "bobs-civic"),
            ("  My Car  ", "my-car"),
            ("CON", "vehicle-con"),             # Windows reserved device name
            ("....", "vehicle"),                # nothing usable left
            ("🚗", "vehicle"),
            ("_merged", "merged"),              # can't claim the internal folder
            ("..", "vehicle"),
        ],
    )
    def test_sanitize(self, raw, expected):
        assert sanitize_name(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        ["car/../etc", "..\\..\\windows", "/etc/passwd", "a/b/c", "..%2f..%2fx"],
    )
    def test_sanitize_cannot_escape_the_storage_root(self, raw, tmp_path):
        """The security property, not the cosmetics: whatever the user types must
        resolve to a single component inside the root."""
        slug = sanitize_name(raw)
        assert "/" not in slug and "\\" not in slug
        resolved = (tmp_path / slug).resolve()
        assert resolved.parent == tmp_path.resolve()

    def test_length_capped(self):
        assert len(sanitize_name("a" * 200)) <= 64

    def test_unique_folder_suffixes_on_collision(self, config):
        config.storage_root.mkdir(parents=True)
        assert config.unique_folder("my car") == "my-car"
        (config.storage_root / "my-car").mkdir()
        assert config.unique_folder("my car") == "my-car-2"


class TestIngest:
    def test_photo_creates_named_folder_with_exports(self, client, config, jpeg):
        response = post(client, jpeg(0.0), "Bob's Civic!")
        assert response.status_code == 200
        body = response.json()
        assert body["vehicle"]["folder"] == "bobs-civic"
        assert body["vehicle"]["name"] == "Bob's Civic!"  # original preserved
        assert body["result"]["created"] is True

        directory = config.storage_root / "bobs-civic"
        assert (directory / "manifest.json").exists()
        assert (directory / "cloud.npz").exists()
        assert {p.name for p in (directory / "exports").iterdir()} == {
            "model.ply", "model.splat", "model_provenance.ply"
        }
        # the raw capture is kept: re-fusable later with better models
        assert len(list((directory / "observations").iterdir())) == 1

    def test_second_photo_routes_to_same_asset(self, client, jpeg):
        post(client, jpeg(0.0), "civic")
        body = post(client, jpeg(0.3), "civic").json()
        assert body["vehicle"]["folder"] == "civic"
        assert body["vehicle"]["observations"] == 2

    def test_name_is_required(self, client, jpeg):
        assert post(client, jpeg(), "   ").status_code == 400

    def test_unsupported_type_rejected(self, client):
        response = client.post(
            "/observations",
            files={"file": ("notes.txt", b"hello", "text/plain")},
            data={"name": "car"},
        )
        assert response.status_code == 400

    def test_oversized_upload_rejected(self, client, config, jpeg):
        config.max_upload_mb = 0
        assert post(client, jpeg(), "car").status_code == 413

    def test_undecodable_image_is_a_client_error(self, client):
        response = client.post(
            "/observations",
            files={"file": ("broken.jpg", b"not-a-jpeg", "image/jpeg")},
            data={"name": "car"},
        )
        assert response.status_code in (400, 500)


class TestReads:
    def test_list_and_detail(self, client, jpeg):
        post(client, jpeg(0.0), "civic")
        vehicles = client.get("/vehicles").json()["vehicles"]
        assert [v["folder"] for v in vehicles] == ["civic"]

        detail = client.get("/vehicles/civic").json()
        assert detail["splats"] > 0
        assert len(detail["observations_log"]) == 1

    def test_resolve_by_display_name_and_id(self, client, jpeg):
        post(client, jpeg(0.0), "Bob's Civic!")
        assert client.get("/vehicles/Bob's Civic!").status_code == 200
        vid = client.get("/vehicles/bobs-civic").json()["vehicle_id"]
        assert client.get(f"/vehicles/{vid}").status_code == 200

    @pytest.mark.parametrize("fmt", ["splat", "ply", "provenance"])
    def test_model_download(self, client, jpeg, fmt):
        post(client, jpeg(0.0), "civic")
        response = client.get("/vehicles/civic/model", params={"fmt": fmt})
        assert response.status_code == 200
        assert len(response.content) > 0

    def test_bad_format_rejected(self, client, jpeg):
        post(client, jpeg(0.0), "civic")
        assert client.get("/vehicles/civic/model", params={"fmt": "obj"}).status_code == 422

    def test_consolidation_state_reported_and_defaults_to_none(self, client, jpeg):
        post(client, jpeg(0.0), "civic")
        for endpoint in ("/vehicles", "/vehicles/civic"):
            data = client.get(endpoint).json()
            v = data["vehicles"][0] if endpoint == "/vehicles" else data
            assert v["consolidation"] == "none"  # nothing consolidated yet
            assert v["frames"] >= 0 and v["consolidated_frames"] == 0

    def test_before_snapshot_is_404_until_consolidation_runs(self, client, jpeg):
        post(client, jpeg(0.0), "civic")
        assert client.get("/vehicles/civic/model", params={"fmt": "before"}).status_code == 404

    def test_missing_vehicle_404s(self, client):
        assert client.get("/vehicles/ghost").status_code == 404

    def test_root_redirects_to_capture_mount(self, client):
        # the capture page loads app.js by relative URL, which only resolves
        # under /capture/ — so "/" must redirect there, not serve bare HTML
        r = client.get("/", follow_redirects=False)
        assert r.status_code in (307, 308)
        assert r.headers["location"] == "/capture/"
        assert client.get("/vehicles/ghost/model").status_code == 404

    def test_health(self, client):
        assert client.get("/health").json()["ok"] is True


class TestMergeFlow:
    def test_duplicate_is_flagged_not_merged_when_auto_merge_off(self, client, jpeg):
        post(client, jpeg(0.0), "civic")
        response = post(client, jpeg(0.1), "mystery")
        assert [e["kind"] for e in response.json()["merges"]] == ["merge_pending"]
        # both still exist: nothing merged without a human
        assert len(client.get("/vehicles").json()["vehicles"]) == 2

        pending = client.get("/merges/pending").json()["pending"]
        assert len(pending) == 1
        assert pending[0]["data"]["score"] >= 0.9

    def test_approve_merges_and_keeps_the_established_name(self, client, config, jpeg):
        """The newcomer must not hijack the folder the user named."""
        post(client, jpeg(0.0), "civic")
        post(client, jpeg(0.1), "mystery")
        pending_id = client.get("/merges/pending").json()["pending"][0]["event_id"]

        response = client.post(f"/merges/{pending_id}/approve")
        assert response.status_code == 200

        vehicles = client.get("/vehicles").json()["vehicles"]
        assert [v["folder"] for v in vehicles] == ["civic"]
        assert "mystery" in vehicles[0]["aliases"]
        # the duplicate is archived, never deleted: a wrong merge must be undoable
        assert (config.storage_root / "_merged" / "mystery").exists()

    def test_reject_keeps_them_separate(self, client, jpeg):
        post(client, jpeg(0.0), "civic")
        post(client, jpeg(0.1), "mystery")
        pending_id = client.get("/merges/pending").json()["pending"][0]["event_id"]

        assert client.post(f"/merges/{pending_id}/reject").status_code == 200
        assert len(client.get("/vehicles").json()["vehicles"]) == 2
        assert client.get("/merges/pending").json()["pending"] == []

    def test_double_approve_is_a_conflict(self, client, jpeg):
        post(client, jpeg(0.0), "civic")
        post(client, jpeg(0.1), "mystery")
        pending_id = client.get("/merges/pending").json()["pending"][0]["event_id"]
        client.post(f"/merges/{pending_id}/approve")
        assert client.post(f"/merges/{pending_id}/approve").status_code in (404, 409)

    def test_unknown_merge_404s(self, client):
        assert client.post("/merges/nope/approve").status_code == 404

    def test_auto_merge_on_merges_immediately(self, client, config, jpeg):
        client.post("/settings/auto-merge", data={"enabled": "true"})
        assert client.get("/settings/auto-merge").json()["auto_merge"] is True

        post(client, jpeg(0.0), "civic")
        response = post(client, jpeg(0.1), "mystery")
        assert [e["kind"] for e in response.json()["merges"]] == ["merge"]
        assert len(client.get("/vehicles").json()["vehicles"]) == 1

    def test_distinct_vehicles_are_not_flagged(self, client, jpeg, truth_cloud):
        blue = truth_cloud.with_updates(
            np.arange(truth_cloud.n),
            colors=np.tile(np.array([0.1, 0.2, 0.85], np.float32), (truth_cloud.n, 1)),
        )
        post(client, jpeg(0.0), "red-car")
        response = post(client, jpeg(0.0, cloud=blue), "blue-car")
        assert response.json()["merges"] == []
        assert len(client.get("/vehicles").json()["vehicles"]) == 2


class TestMergeMechanics:
    def test_merge_clouds_takes_observed_discards_guesses(self, prior_cloud):
        marked = prior_cloud.with_updates(
            np.arange(10),
            provenance=np.full(10, Provenance.OBSERVED, np.uint8),
        )
        merged = merge_clouds(prior_cloud, marked)
        assert merged.n == prior_cloud.n + 10  # only the 10 real ones crossed over

    def test_merge_clouds_with_no_evidence_is_a_noop(self, prior_cloud):
        assert merge_clouds(prior_cloud, prior_cloud) is prior_cloud

    def test_merge_assets_unions_history_and_aliases(self, prior_cloud):
        a = VehicleAsset(name="civic", cloud=prior_cloud)
        a.add_observation({"kind": "photo"}, np.ones(4, np.float32))
        b = VehicleAsset(name="mystery", cloud=prior_cloud)
        b.aliases.append("unknown-car")
        b.add_observation({"kind": "photo"}, np.ones(4, np.float32))

        merged = merge_assets(a, b)
        assert len(merged.observations) == 2
        assert set(merged.aliases) == {"mystery", "unknown-car"}
        assert merged.embeddings.shape == (2, 4)
        assert merged.name == "civic"

    def test_pick_primary_keeps_the_established_identity(self, prior_cloud, config):
        """The older asset wins regardless of argument order."""
        store = VehicleStore(config)
        VehicleAsset(name="old", cloud=prior_cloud, created_ts=100.0).save(
            config.vehicle_dir("old")
        )
        VehicleAsset(name="new", cloud=prior_cloud, created_ts=200.0).save(
            config.vehicle_dir("new")
        )
        assert pick_primary(store, "new", "old") == ("old", "new")
        assert pick_primary(store, "old", "new") == ("old", "new")

    def test_pick_primary_ignores_coverage(self, prior_cloud, config):
        """A newcomer with more confirmed splats must NOT steal the user's name.

        Its evidence merges in either way, so letting a few splats of luck flip
        the surviving folder name would be pure downside.
        """
        store = VehicleStore(config)
        VehicleAsset(name="old", cloud=prior_cloud, created_ts=100.0).save(
            config.vehicle_dir("old")
        )
        richer = prior_cloud.with_updates(
            np.arange(50), provenance=np.full(50, Provenance.OBSERVED, np.uint8)
        )
        VehicleAsset(name="new", cloud=richer, created_ts=200.0).save(
            config.vehicle_dir("new")
        )
        assert pick_primary(store, "new", "old") == ("old", "new")


class TestStore:
    def test_internal_folders_are_hidden(self, config, prior_cloud):
        store = VehicleStore(config)
        VehicleAsset(name="a", cloud=prior_cloud).save(config.vehicle_dir("a"))
        VehicleAsset(name="b", cloud=prior_cloud).save(config.merged_root / "b")
        assert store.folders() == ["a"]

    def test_resolve_returns_none_when_missing(self, config):
        assert VehicleStore(config).resolve("ghost") is None

    def test_save_upload_avoids_overwrite(self, config):
        store = VehicleStore(config)
        folder = store.create_folder("car")
        first = store.save_upload(folder, "a.jpg", b"one")
        second = store.save_upload(folder, "a.jpg", b"two")
        assert first != second
        assert first.read_bytes() == b"one"

    def test_manifest_of_missing_folder(self, config):
        assert VehicleStore(config).manifest("ghost") is None


class TestEventLog:
    def test_append_and_read_newest_first(self, tmp_path):
        log = EventLog(tmp_path / "e.jsonl")
        log.append(Event(kind="observation", message="one"))
        log.append(Event(kind="observation", message="two"))
        assert [e["message"] for e in log.all()] == ["two", "one"]
        assert len(log.all(limit=1)) == 1

    def test_missing_file_is_empty(self, tmp_path):
        assert EventLog(tmp_path / "nope.jsonl").all() == []

    def test_pending_excludes_resolved(self, tmp_path):
        log = EventLog(tmp_path / "e.jsonl")
        pending = log.append(Event(kind="merge_pending", message="?", data={}))
        assert len(log.pending_merges()) == 1
        log.append(Event(kind="merge", message="done",
                         data={"pending_id": pending.event_id}))
        assert log.pending_merges() == []

    def test_find(self, tmp_path):
        log = EventLog(tmp_path / "e.jsonl")
        event = log.append(Event(kind="observation", message="x"))
        assert log.find(event.event_id)["message"] == "x"
        assert log.find("nope") is None


class TestQueue:
    def test_runs_work_and_returns_result(self):
        queue = VehicleQueue()
        assert queue.run_sync("car", lambda: 42) == 42
        assert queue.stats.completed == 1
        queue.shutdown()

    def test_same_vehicle_never_runs_concurrently(self):
        """Fusion is read-modify-write; overlap would silently lose evidence."""
        import threading
        import time

        queue = VehicleQueue()
        overlaps, active, lock = [], [0], threading.Lock()

        def work():
            with lock:
                active[0] += 1
                overlaps.append(active[0])
            time.sleep(0.02)
            with lock:
                active[0] -= 1

        jobs = [queue.submit("same-car", work) for _ in range(6)]
        for job in jobs:
            job.future.result()
        assert max(overlaps) == 1, "two jobs touched one vehicle at once"
        queue.shutdown()

    def test_different_vehicles_run_in_parallel(self):
        import threading
        import time

        queue = VehicleQueue(max_workers=4)
        seen, lock = [], threading.Lock()

        def work():
            with lock:
                seen.append(time.time())
            time.sleep(0.05)

        jobs = [queue.submit(f"car-{i}", work) for i in range(4)]
        for job in jobs:
            job.future.result()
        assert max(seen) - min(seen) < 0.05, "vehicles were serialized against each other"
        queue.shutdown()

    def test_failure_is_recorded_and_reraised(self):
        queue = VehicleQueue()

        def boom():
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            queue.run_sync("car", boom)
        assert queue.stats.failed == 1
        queue.shutdown()


class TestDecodeImage:
    def test_downscales_wide_images(self):
        big = np.zeros((1000, 4000, 3), np.uint8)
        ok, buf = cv2.imencode(".png", big)
        assert ok
        out = decode_image(buf.tobytes(), max_width=1280)
        assert out.shape[1] == 1280
        assert out.shape[0] == 320  # aspect preserved

    def test_small_images_pass_through(self):
        small = np.zeros((10, 20, 3), np.uint8)
        ok, buf = cv2.imencode(".png", small)
        assert ok
        assert decode_image(buf.tobytes(), max_width=1280).shape == (10, 20, 3)

    def test_garbage_raises(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            decode_image(b"garbage", max_width=1280)


class TestAutoConsolidateGate:
    """The pure decision that gates the background photorealism pass. The pass
    itself is CUDA-only (see scripts/consolidate_vehicle.py); only the gating is
    tested here, so it runs on any machine."""

    def _asset_with_frames(self, n_frames: int, consolidated: int = 0):
        asset = VehicleAsset(name="car")
        asset.consolidated_frames = consolidated
        image = np.zeros((4, 6, 3), np.uint8)
        pose, intr = CameraPose.identity(), Intrinsics.simple(6, 4)
        for _ in range(n_frames):
            asset.add_frame(image, None, pose, intr)
        return asset

    def test_fires_once_enough_new_frames_and_gpu_present(self):
        cfg = Config(auto_consolidate=True, consolidate_min_frames=24)
        asset = self._asset_with_frames(24)
        assert should_autoconsolidate(asset, cfg, gsplat_ok=True) is True

    def test_skipped_without_gsplat(self):
        cfg = Config(auto_consolidate=True, consolidate_min_frames=24)
        asset = self._asset_with_frames(50)
        assert should_autoconsolidate(asset, cfg, gsplat_ok=False) is False

    def test_skipped_when_disabled(self):
        cfg = Config(auto_consolidate=False, consolidate_min_frames=24)
        asset = self._asset_with_frames(50)
        assert should_autoconsolidate(asset, cfg, gsplat_ok=True) is False

    def test_skipped_below_frame_threshold(self):
        cfg = Config(auto_consolidate=True, consolidate_min_frames=24)
        asset = self._asset_with_frames(23)
        assert should_autoconsolidate(asset, cfg, gsplat_ok=True) is False

    def test_skipped_when_no_new_frames_since_last_pass(self):
        cfg = Config(auto_consolidate=True, consolidate_min_frames=24)
        # already consolidated at 30 frames, still 30 -> nothing new to do
        asset = self._asset_with_frames(30, consolidated=30)
        assert should_autoconsolidate(asset, cfg, gsplat_ok=True) is False

    def test_fires_again_after_more_frames_arrive(self):
        cfg = Config(auto_consolidate=True, consolidate_min_frames=24)
        asset = self._asset_with_frames(40, consolidated=30)  # 10 new since last
        assert should_autoconsolidate(asset, cfg, gsplat_ok=True) is True
