"""Fusion arbitration — the rules that decide what may overwrite what.

These are the tests that matter most: every failure here is silent data
corruption in a user's model, not a crash.
"""
from __future__ import annotations

import numpy as np
import pytest

from cargen.core.splat import GaussianCloud, Provenance
from cargen.fusion_engine.engine import FusionConfig, FusionEngine
from cargen.fusion_engine.residual import compensate_exposure, dilate_mask, residual_map
from demo.synthetic import orbit_pose, render_photo


@pytest.fixture
def engine(renderer):
    return FusionEngine(renderer, FusionConfig())


def _photo(truth, angle, intrinsics, renderer, **kw):
    return render_photo(truth, orbit_pose(angle), intrinsics, renderer, seed=1, **kw)


class TestRegistrationGate:
    """A pose we don't trust must never touch the asset."""

    def test_low_confidence_frame_is_a_non_event(
        self, engine, prior_cloud, intrinsics, renderer, truth_cloud
    ):
        photo, mask = _photo(truth_cloud, 0.0, intrinsics, renderer)
        after, report = engine.fuse_frame(
            prior_cloud, photo, mask, orbit_pose(0.0), intrinsics,
            registration_confidence=0.1, timestamp=1.0,
        )
        assert not report.accepted
        assert after is prior_cloud  # identical object, not merely equal
        assert "queued, not fused" in report.reason

    def test_threshold_boundary(self, engine, prior_cloud, intrinsics, renderer, truth_cloud):
        photo, mask = _photo(truth_cloud, 0.0, intrinsics, renderer)
        floor = engine.config.min_registration_confidence
        _, just_below = engine.fuse_frame(
            prior_cloud, photo, mask, orbit_pose(0.0), intrinsics,
            registration_confidence=floor - 0.01, timestamp=1.0,
        )
        _, at_floor = engine.fuse_frame(
            prior_cloud, photo, mask, orbit_pose(0.0), intrinsics,
            registration_confidence=floor, timestamp=1.0,
        )
        assert not just_below.accepted
        assert at_floor.accepted

    def test_empty_cloud_rejected(self, engine, intrinsics):
        photo = np.zeros((intrinsics.height, intrinsics.width, 3), np.uint8)
        after, report = engine.fuse_frame(
            GaussianCloud.empty(), photo, None, orbit_pose(0.0), intrinsics,
            registration_confidence=1.0, timestamp=0.0,
        )
        assert not report.accepted
        assert after.n == 0


class TestProvenanceArbitration:
    def test_first_view_replaces_guess_entirely(
        self, engine, prior_cloud, truth_cloud, intrinsics, renderer
    ):
        """A PRIOR splat carries no evidence, so reality must replace it whole.

        Blending would leave a fraction of the hallucination permanently baked
        in — and, because the small leftover residual falls under the dirty
        threshold, the splat would then be declared 'close enough' and never
        corrected again.
        """
        photo, mask = _photo(truth_cloud, 0.0, intrinsics, renderer)
        fused, report = engine.fuse_frame(
            prior_cloud, photo, mask, orbit_pose(0.0), intrinsics,
            registration_confidence=0.9, timestamp=1.0, evidence_weight=0.5,
        )
        touched = fused.provenance == Provenance.OBSERVED
        assert touched.sum() > 0
        # despite evidence_weight=0.5, the guess is gone: colours match truth
        error = np.abs(fused.colors[touched] - truth_cloud.colors[touched]).mean()
        assert error < 0.05, f"guess survived replacement (error {error:.3f})"

    def test_prior_to_observed_only_where_seen(
        self, engine, prior_cloud, truth_cloud, intrinsics, renderer
    ):
        """Splats the frame cannot see must stay frozen."""
        pose = orbit_pose(0.0)
        photo, mask = _photo(truth_cloud, 0.0, intrinsics, renderer)
        render = renderer.render(prior_cloud, pose, intrinsics)
        visible = np.unique(render.splat_index[render.hit_mask])
        visible = set(visible[visible >= 0].tolist())

        fused, _ = engine.fuse_frame(
            prior_cloud, photo, mask, pose, intrinsics,
            registration_confidence=0.9, timestamp=1.0,
        )
        promoted = set(np.where(fused.provenance == Provenance.OBSERVED)[0].tolist())
        assert promoted <= visible, "a splat outside the view was modified"
        assert promoted, "nothing was confirmed at all"

    def test_confirmed_splats_resist_weak_contradiction(
        self, engine, prior_cloud, truth_cloud, intrinsics, renderer
    ):
        """A blurry CCTV frame must not vandalize a clean phone capture."""
        pose = orbit_pose(0.0)
        photo, mask = _photo(truth_cloud, 0.0, intrinsics, renderer)
        # confirm the model hard with several strong views
        cloud = prior_cloud
        for i in range(4):
            cloud, _ = engine.fuse_frame(
                cloud, photo, mask, pose, intrinsics,
                registration_confidence=0.95, timestamp=float(i), evidence_weight=1.0,
            )
        confirmed = (cloud.provenance == Provenance.OBSERVED) & (cloud.confidence > 0.8)
        assert confirmed.sum() > 0
        before = cloud.colors[confirmed].copy()

        # a genuinely contradicting frame at CCTV weight
        green = truth_cloud.with_updates(
            np.arange(truth_cloud.n),
            colors=np.tile(np.array([0.1, 0.8, 0.2], np.float32), (truth_cloud.n, 1)),
        )
        bad_photo, bad_mask = _photo(green, 0.0, intrinsics, renderer)
        after, _ = engine.fuse_frame(
            cloud, bad_photo, bad_mask, pose, intrinsics,
            registration_confidence=0.95, timestamp=99.0, evidence_weight=0.3,
        )
        drift = np.abs(after.colors[: cloud.n][confirmed] - before).max()
        assert drift == pytest.approx(0.0, abs=1e-6), f"weak evidence overwrote (drift {drift})"

    def test_strong_evidence_does_get_through(
        self, engine, prior_cloud, truth_cloud, intrinsics, renderer
    ):
        """Resistance must not become immunity — a good capture still corrects."""
        pose = orbit_pose(0.0)
        photo, mask = _photo(truth_cloud, 0.0, intrinsics, renderer)
        cloud = prior_cloud
        for i in range(4):
            cloud, _ = engine.fuse_frame(
                cloud, photo, mask, pose, intrinsics,
                registration_confidence=0.95, timestamp=float(i), evidence_weight=1.0,
            )
        confirmed = (cloud.provenance == Provenance.OBSERVED) & (cloud.confidence > 0.8)
        before = cloud.colors[confirmed].copy()

        green = truth_cloud.with_updates(
            np.arange(truth_cloud.n),
            colors=np.tile(np.array([0.1, 0.8, 0.2], np.float32), (truth_cloud.n, 1)),
        )
        bad_photo, bad_mask = _photo(green, 0.0, intrinsics, renderer)
        after, report = engine.fuse_frame(
            cloud, bad_photo, bad_mask, pose, intrinsics,
            registration_confidence=0.95, timestamp=99.0, evidence_weight=1.0,
        )
        drift = np.abs(after.colors[: cloud.n][confirmed] - before).max()
        assert report.dirty > 0
        assert drift > 0.1, "strong contradicting evidence was ignored"

    def test_confidence_and_view_count_accumulate(
        self, engine, prior_cloud, truth_cloud, intrinsics, renderer
    ):
        pose = orbit_pose(0.0)
        photo, mask = _photo(truth_cloud, 0.0, intrinsics, renderer)
        cloud, _ = engine.fuse_frame(
            cloud := prior_cloud, photo, mask, pose, intrinsics,
            registration_confidence=0.9, timestamp=5.0,
        )
        seen = cloud.view_count > 0
        assert seen.sum() > 0
        assert (cloud.confidence[seen] > prior_cloud.confidence[seen]).all()
        assert (cloud.last_seen_ts[seen] == 5.0).all()


class TestConvergence:
    def test_walkaround_raises_observed_fraction(
        self, engine, prior_cloud, truth_cloud, intrinsics, renderer
    ):
        cloud = prior_cloud
        assert cloud.observed_fraction() == 0.0
        for i, angle in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
            photo, mask = _photo(truth_cloud, angle, intrinsics, renderer)
            cloud, _ = engine.fuse_frame(
                cloud, photo, mask, orbit_pose(angle), intrinsics,
                registration_confidence=0.9, timestamp=float(i),
            )
        assert cloud.observed_fraction() > 0.5

    def test_surprise_falls_as_model_learns(
        self, engine, prior_cloud, truth_cloud, intrinsics, renderer
    ):
        pose, residuals = orbit_pose(0.0), []
        photo, mask = _photo(truth_cloud, 0.0, intrinsics, renderer)
        cloud = prior_cloud
        for i in range(3):
            cloud, report = engine.fuse_frame(
                cloud, photo, mask, pose, intrinsics,
                registration_confidence=0.9, timestamp=float(i),
            )
            residuals.append(report.mean_residual)
        assert residuals[-1] < residuals[0], f"model did not converge: {residuals}"


class TestDensification:
    def test_evidence_without_geometry_spawns_splats(
        self, engine, prior_cloud, truth_cloud, intrinsics, renderer
    ):
        """Aftermarket parts the prior never guessed must be able to appear."""
        pose = orbit_pose(0.0)
        photo, mask = _photo(truth_cloud, 0.0, intrinsics, renderer)
        # claim vehicle everywhere: pixels with no geometry behind them
        wide_mask = np.ones(mask.shape, np.float32)
        _, report = engine.fuse_frame(
            prior_cloud, photo, wide_mask, pose, intrinsics,
            registration_confidence=0.9, timestamp=1.0,
        )
        assert report.densified > 0
        assert report.splats_after > report.splats_before

    def test_densified_splats_are_observed(
        self, engine, prior_cloud, truth_cloud, intrinsics, renderer
    ):
        photo, mask = _photo(truth_cloud, 0.0, intrinsics, renderer)
        fused, report = engine.fuse_frame(
            prior_cloud, photo, np.ones(mask.shape, np.float32),
            orbit_pose(0.0), intrinsics,
            registration_confidence=0.9, timestamp=1.0,
        )
        new = fused.select(np.arange(prior_cloud.n, fused.n))
        assert (new.provenance == Provenance.OBSERVED).all()
        assert (new.view_count == 1).all()

    def test_never_invents_geometry_in_open_space(
        self, engine, prior_cloud, truth_cloud, intrinsics, renderer
    ):
        """A sloppy mask must not be able to paint a slab across the scene.

        Regression: with a rectangular mask calling the background "vehicle",
        densification used to spawn a splat for every empty pixel at the global
        median hit depth — a flat sheet of splats through the model. New splats
        must stay within reach of geometry that actually exists.
        """
        pose = orbit_pose(0.0)
        photo, _ = _photo(truth_cloud, 0.0, intrinsics, renderer)
        rectangle = np.zeros(photo.shape[:2], np.float32)
        rectangle[5:-5, 5:-5] = 1.0  # the StubSegmenter's crude claim

        fused, report = engine.fuse_frame(
            prior_cloud, photo, rectangle, pose, intrinsics,
            registration_confidence=0.9, timestamp=1.0,
        )
        new = fused.select(np.arange(prior_cloud.n, fused.n))
        if new.n == 0:
            return  # nothing spawned is also acceptable

        # New splats may extend beyond the model by at most the densify reach,
        # converted from pixels to world units at the viewing distance (plus a
        # margin for grid quantisation). Anything past that is invented depth.
        depth = float(np.linalg.norm(pose.camera_center - prior_cloud.positions.mean(0)))
        reach_world = engine.config.densify_reach * depth / intrinsics.fx
        margin = reach_world * 1.6

        lo = prior_cloud.positions.min(axis=0) - margin
        hi = prior_cloud.positions.max(axis=0) + margin
        assert (new.positions >= lo).all() and (new.positions <= hi).all(), (
            f"densifier invented geometry {margin:.2f} beyond the model: "
            f"{new.positions.min(axis=0)}..{new.positions.max(axis=0)} "
            f"vs prior {prior_cloud.positions.min(axis=0)}..{prior_cloud.positions.max(axis=0)}"
        )
        # and the slab was thousands of splats across the whole frame
        assert new.n < prior_cloud.n * 0.25, f"{new.n} new splats looks like a slab"

    def test_no_densify_without_any_geometry(self, engine, prior_cloud, intrinsics, renderer):
        """Nothing rendered means no depth context anywhere — spawn nothing."""
        away = orbit_pose(0.0)
        empty_cloud = prior_cloud.with_updates(
            np.arange(prior_cloud.n), opacities=np.zeros(prior_cloud.n, np.float32)
        )
        photo = np.full((intrinsics.height, intrinsics.width, 3), 90, np.uint8)
        fused, report = engine.fuse_frame(
            empty_cloud, photo, np.ones(photo.shape[:2], np.float32), away, intrinsics,
            registration_confidence=0.9, timestamp=1.0,
        )
        assert report.densified == 0

    def test_max_splats_is_respected(self, prior_cloud, truth_cloud, intrinsics, renderer):
        engine = FusionEngine(renderer, FusionConfig(max_splats=prior_cloud.n))
        photo, mask = _photo(truth_cloud, 0.0, intrinsics, renderer)
        fused, report = engine.fuse_frame(
            prior_cloud, photo, np.ones(mask.shape, np.float32),
            orbit_pose(0.0), intrinsics,
            registration_confidence=0.9, timestamp=1.0,
        )
        assert report.densified == 0
        assert fused.n <= prior_cloud.n


class TestPruning:
    def test_transparent_splats_pruned(self, truth_cloud, intrinsics, renderer):
        # symmetry/smoothing off: this test is about the prune step
        # specifically, and smoothing could nudge a zero-opacity outlier's
        # opacity toward its local median before pruning ever sees it — see
        # tests/test_symmetry.py for that behavior's own tests.
        engine = FusionEngine(
            renderer, FusionConfig(mirror_symmetry=False, smooth_bands=False)
        )
        cloud = truth_cloud.with_updates(
            np.arange(10), opacities=np.zeros(10, np.float32)
        )
        photo, mask = _photo(truth_cloud, 0.0, intrinsics, renderer)
        fused, report = engine.fuse_frame(
            cloud, photo, mask, orbit_pose(0.0), intrinsics,
            registration_confidence=0.9, timestamp=1.0,
        )
        assert report.pruned == 10
        assert fused.n == cloud.n - 10


class TestLocalizedOptimizer:
    """The optimizer is the real refinement path (gsplat); these tests use a
    recording fake so they run without CUDA and pin the contract the engine
    relies on."""

    class _FakeOptimizer:
        """Records what it was handed and nudges only the dirty splats."""

        def __init__(self, ring_weight=0.3):
            from cargen.fusion_engine.optimize import OptimizeConfig

            self.config = OptimizeConfig(ring_weight=ring_weight)
            self.calls = []

        def refine(self, cloud, dirty_indices, image, pixel_weight, pose, intrinsics):
            self.calls.append(
                {"dirty": dirty_indices.copy(), "weight": pixel_weight.copy()}
            )
            colors = cloud.colors.copy()
            colors[dirty_indices] = 0.0  # an unmistakable mark
            return cloud.with_updates(
                np.arange(cloud.n), colors=colors.astype(np.float32)
            )

    def test_engine_skips_optimizer_when_absent(self, engine):
        assert engine.optimizer is None  # default: CPU blend stand-in

    def test_frozen_splats_are_untouched(
        self, renderer, prior_cloud, truth_cloud, intrinsics
    ):
        """The property the whole localized design rests on: a frame may only
        alter splats it disputes. Everything else must come out identical.

        Symmetry/smoothing off: those passes deliberately touch PRIOR splats
        outside the dispute region too (that's their whole point — propagate
        confirmed evidence to the still-guessed side) — a separate, later
        mechanism with its own tests in tests/test_symmetry.py, orthogonal to
        the dirty-region contract this test pins down."""
        fake = self._FakeOptimizer()
        engine = FusionEngine(
            renderer,
            FusionConfig(mirror_symmetry=False, smooth_bands=False),
            optimizer=fake,
        )
        photo, mask = _photo(truth_cloud, 0.0, intrinsics, renderer)
        before = prior_cloud

        after, report = engine.fuse_frame(
            before, photo, mask, orbit_pose(0.0), intrinsics,
            registration_confidence=0.9, timestamp=1.0,
        )
        assert fake.calls, "optimizer was never invoked"
        dirty = fake.calls[0]["dirty"]
        frozen = np.setdiff1d(np.arange(before.n), dirty)
        # positions/scales/rotations of frozen splats must be bit-identical
        assert np.array_equal(after.positions[frozen], before.positions[frozen])
        assert np.array_equal(after.scales[frozen], before.scales[frozen])
        assert np.array_equal(after.rotations[frozen], before.rotations[frozen])

    def test_pixel_weight_is_zero_outside_the_dispute(
        self, renderer, prior_cloud, truth_cloud, intrinsics
    ):
        fake = self._FakeOptimizer(ring_weight=0.3)
        engine = FusionEngine(renderer, FusionConfig(), optimizer=fake)
        photo, mask = _photo(truth_cloud, 0.0, intrinsics, renderer)
        engine.fuse_frame(
            prior_cloud, photo, mask, orbit_pose(0.0), intrinsics,
            registration_confidence=0.9, timestamp=1.0,
        )
        weight = fake.calls[0]["weight"]
        assert weight.max() == pytest.approx(1.0), "core region must pull fully"
        assert set(np.unique(weight)).issubset({0.0, 0.3, 1.0}), np.unique(weight)
        assert (weight == 0).any(), "undisputed pixels must contribute nothing"

    def test_partial_dispute_gets_a_blend_ring(
        self, renderer, truth_cloud, intrinsics
    ):
        """Where a disputed patch borders an agreeing region, the optimizer must
        get a soft ring — otherwise the edit seams against the frozen splats.

        (A dispute covering the whole visible surface has no such boundary, so
        this needs a model that is right almost everywhere and wrong in a patch.)
        """
        fake = self._FakeOptimizer(ring_weight=0.3)
        engine = FusionEngine(renderer, FusionConfig(), optimizer=fake)
        photo, mask = _photo(truth_cloud, 0.0, intrinsics, renderer)

        # a model that matches the photo except for one localized patch
        patch = truth_cloud.positions[:, 0] > 0.4
        wrong = truth_cloud.with_updates(
            np.where(patch)[0],
            colors=np.tile(
                np.array([0.1, 0.9, 0.2], np.float32), (int(patch.sum()), 1)
            ),
        )
        engine.fuse_frame(
            wrong, photo, mask, orbit_pose(0.0), intrinsics,
            registration_confidence=0.9, timestamp=1.0,
        )
        weight = fake.calls[0]["weight"]
        assert np.isclose(weight, 1.0).any(), "no fully-weighted core"
        assert np.isclose(weight, 0.3).any(), "no blend ring around the dirty core"
        assert (weight == 0).any(), "agreeing region must contribute nothing"

    def test_not_called_when_nothing_is_dirty(
        self, renderer, truth_cloud, intrinsics
    ):
        """A frame the model already agrees with must not trigger an optimize."""
        fake = self._FakeOptimizer()
        engine = FusionEngine(renderer, FusionConfig(), optimizer=fake)
        photo, mask = _photo(truth_cloud, 0.0, intrinsics, renderer)
        # fuse the truth against itself: nothing to dispute
        engine.fuse_frame(
            truth_cloud, photo, mask, orbit_pose(0.0), intrinsics,
            registration_confidence=0.9, timestamp=1.0,
        )
        for call in fake.calls:
            assert call["dirty"].size > 0, "optimizer called with an empty dirty set"


class TestExposureCompensation:
    def test_lighting_shift_is_absorbed_not_flagged(
        self, engine, prior_cloud, truth_cloud, intrinsics, renderer
    ):
        """An overcast day must not read as 'the whole car changed'."""
        pose = orbit_pose(0.0)
        photo, mask = _photo(truth_cloud, 0.0, intrinsics, renderer)
        cloud = prior_cloud
        for i in range(5):  # confirm hard so splats pass the exposure-trust bar
            cloud, _ = engine.fuse_frame(
                cloud, photo, mask, pose, intrinsics,
                registration_confidence=0.95, timestamp=float(i), evidence_weight=1.0,
            )
        _, bright = engine.fuse_frame(
            cloud, photo, mask, pose, intrinsics,
            registration_confidence=0.95, timestamp=10.0,
        )
        dim_photo, dim_mask = _photo(truth_cloud, 0.0, intrinsics, renderer, exposure=0.7)
        _, dim = engine.fuse_frame(
            cloud, dim_photo, dim_mask, pose, intrinsics,
            registration_confidence=0.95, timestamp=11.0,
        )
        # a 30% exposure drop should not multiply the dirty region
        assert dim.dirty <= bright.dirty + 0.15 * cloud.n

    def test_compensate_needs_enough_reference(self):
        rendered = np.full((20, 20, 3), 0.5, np.float32)
        observed = np.full((20, 20, 3), 0.2, np.float32)
        tiny = np.zeros((20, 20), bool)
        tiny[0, 0] = True
        # too few reference pixels to fit anything trustworthy → leave it alone
        assert np.allclose(compensate_exposure(rendered, observed, tiny), observed)

    def test_compensate_recovers_a_gain(self):
        rng = np.random.default_rng(0)
        rendered = rng.random((64, 64, 3)).astype(np.float32)
        observed = np.clip(rendered * 0.5, 0, 1)  # a uniform exposure drop
        mask = np.ones((64, 64), bool)
        fixed = compensate_exposure(rendered, observed, mask)
        assert np.abs(fixed - rendered).mean() < np.abs(observed - rendered).mean()

    def test_compensate_handles_flat_reference(self):
        rendered = np.full((32, 32, 3), 0.6, np.float32)
        observed = np.full((32, 32, 3), 0.3, np.float32)
        fixed = compensate_exposure(rendered, observed, np.ones((32, 32), bool))
        assert np.allclose(fixed, 0.6, atol=1e-5)  # brightness matched, no blow-up


class TestResidualHelpers:
    def test_residual_zero_for_identical(self):
        img = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
        assert residual_map(img, img).max() == pytest.approx(0.0, abs=1e-6)

    def test_residual_detects_difference(self):
        a = np.zeros((32, 32, 3), np.float32)
        b = np.ones((32, 32, 3), np.float32)
        assert residual_map(a, b).mean() == pytest.approx(1.0, abs=1e-3)

    def test_dilate_grows_mask(self):
        mask = np.zeros((21, 21), bool)
        mask[10, 10] = True
        assert dilate_mask(mask, 2).sum() > 1
        assert dilate_mask(mask, 0).sum() == 1
