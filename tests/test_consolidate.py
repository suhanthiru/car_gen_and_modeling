"""Joint multi-view consolidation — CPU-only, no gsplat/CUDA needed.

Mirrors `test_fusion.py::TestLocalizedOptimizer`'s pattern: a recording fake
stands in for the real (torch/gsplat) numeric core, so these tests pin the
contract `Consolidator` promises — visibility partitioning, provenance
promotion, batching coverage, and report plumbing — without ever importing
torch.
"""
from __future__ import annotations

import numpy as np
import pytest

from cargen.core.camera import CameraPose, Intrinsics
from cargen.core.splat import GaussianCloud, Provenance
from cargen.fusion_engine.consolidate import (
    Consolidator,
    ConsolidationConfig,
    ConsolidationReport,
    FrameObservation,
)
from cargen.fusion_engine.renderer import RenderResult

WIDTH, HEIGHT = 4, 4


def _make_cloud(n: int, provenance: int | np.ndarray = Provenance.PRIOR) -> GaussianCloud:
    positions = np.arange(n * 3, dtype=np.float32).reshape(n, 3) * 0.1
    colors = np.full((n, 3), 0.5, np.float32)
    return GaussianCloud.create(positions, colors, provenance=provenance)


def _frame(image_rgb=None, mask=None, evidence_weight=1.0) -> FrameObservation:
    if image_rgb is None:
        image_rgb = np.full((HEIGHT, WIDTH, 3), 128, np.uint8)
    return FrameObservation(
        image_rgb=image_rgb,
        mask=mask,
        pose=CameraPose.identity(),
        intrinsics=Intrinsics.simple(WIDTH, HEIGHT),
        evidence_weight=evidence_weight,
    )


def _render_result(hit_ids: set[int], color: float = 0.5) -> RenderResult:
    """A canned RenderResult: `hit_ids` are scattered one-per-pixel down the
    diagonal so each hit splat owns exactly one pixel — enough for
    `_visibility_mask`'s "which splats did this frame see" bookkeeping."""
    index = np.full((HEIGHT, WIDTH), -1, np.int32)
    for i, splat_id in enumerate(sorted(hit_ids)):
        index[i % HEIGHT, i % WIDTH] = splat_id
    alpha = (index >= 0).astype(np.float32)
    color_arr = np.full((HEIGHT, WIDTH, 3), color, np.float32)
    depth = np.where(index >= 0, 1.0, np.inf).astype(np.float32)
    return RenderResult(color_arr, depth, index, alpha)


class _FakeDifferentiableRenderer:
    """Deterministic double for `GsplatRenderer.render()` — no torch/gsplat.

    Hands back one canned `RenderResult` per call, in order, so a test can
    control exactly which splats each frame "sees" without real projection
    math or a CUDA rasterizer.
    """

    def __init__(self, results: list[RenderResult]):
        self._results = list(results)
        self.calls: list[dict] = []

    def render(self, cloud, pose, intrinsics) -> RenderResult:
        self.calls.append({"pose": pose, "intrinsics": intrinsics, "n": cloud.n})
        return self._results[len(self.calls) - 1]


class _FakeConsolidator(Consolidator):
    """Overrides only `_step` — the one method that genuinely needs
    torch/gsplat — so the rest of `consolidate()`'s orchestration (visibility,
    batching, densify cadence, provenance promotion, report assembly) runs for
    real and gets exercised end to end without ever importing torch.

    `params`/`frozen` are deliberately left as plain numpy dicts (never
    tensorized), which is what keeps `_maybe_densify`'s early-return and
    `_assemble_cloud`'s numpy branch both torch-free too.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_calls: list[dict] = []

    def _step(self, params, frozen, frames_batch, sh_degree) -> float:
        self.step_calls.append(
            {
                "frame_indices": [i for i, _ in frames_batch],
                "sh_degree": sh_degree,
                "lr_position": self.config.lr_position,
                "lr_scale": self.config.lr_scale,
                "lr_sh_rest": self.config.lr_sh_rest,
            }
        )
        # A tiny deterministic nudge so _assemble_cloud has something real to
        # copy back, while staying pure numpy.
        params["means"] = params["means"] + 0.001
        return max(0.5 - 0.05 * len(self.step_calls), 0.01)


class TestVisibilityMask:
    def test_counts_hits_across_frames(self):
        cloud = _make_cloud(5)
        frames = [_frame(), _frame(), _frame()]
        results = [
            _render_result({0, 1}),
            _render_result({1, 2}),
            _render_result(set()),
        ]
        fake = _FakeDifferentiableRenderer(results)
        consolidator = Consolidator(renderer=fake, device="cpu")

        counts = consolidator._visibility_mask(cloud, frames)

        assert counts.tolist() == [1, 2, 1, 0, 0]
        assert len(fake.calls) == 3

    def test_no_frames_means_nothing_visible(self):
        cloud = _make_cloud(3)
        consolidator = Consolidator(renderer=_FakeDifferentiableRenderer([]), device="cpu")
        counts = consolidator._visibility_mask(cloud, [])
        assert counts.tolist() == [0, 0, 0]

    def test_empty_cloud_short_circuits(self):
        consolidator = Consolidator(renderer=_FakeDifferentiableRenderer([]), device="cpu")
        counts = consolidator._visibility_mask(GaussianCloud.empty(), [_frame()])
        assert counts.shape == (0,)


class TestPromoteProvenance:
    def test_seen_enough_promotes_prior_to_observed(self):
        cloud = _make_cloud(4, provenance=np.array(
            [Provenance.PRIOR, Provenance.PRIOR, Provenance.OBSERVED, Provenance.PRIOR],
            np.uint8,
        ))
        cfg = ConsolidationConfig(min_views_for_confirm=2)
        consolidator = Consolidator(cfg, device="cpu")
        seen_counts = np.array([1, 2, 3, 0], np.int32)

        promoted = consolidator._promote_provenance(cloud, seen_counts)

        assert promoted.provenance.tolist() == [
            Provenance.PRIOR,      # seen once, below threshold
            Provenance.OBSERVED,   # seen twice, promoted
            Provenance.OBSERVED,   # already observed, untouched but still observed
            Provenance.PRIOR,      # never seen, untouched
        ]

    def test_never_seen_splats_are_bit_identical(self):
        cloud = _make_cloud(4)
        cfg = ConsolidationConfig(min_views_for_confirm=1)
        consolidator = Consolidator(cfg, device="cpu")
        seen_counts = np.array([1, 0, 1, 0], np.int32)

        promoted = consolidator._promote_provenance(cloud, seen_counts)

        untouched = [1, 3]
        assert np.array_equal(promoted.positions[untouched], cloud.positions[untouched])
        assert np.array_equal(promoted.colors[untouched], cloud.colors[untouched])
        assert promoted.provenance[untouched].tolist() == [Provenance.PRIOR, Provenance.PRIOR]

    def test_no_confirmations_returns_cloud_unchanged(self):
        cloud = _make_cloud(3)
        consolidator = Consolidator(ConsolidationConfig(min_views_for_confirm=5), device="cpu")
        promoted = consolidator._promote_provenance(cloud, np.array([1, 2, 3], np.int32))
        assert promoted is cloud


class TestShDegreeSchedule:
    def test_starts_at_lowest_band_and_reaches_highest(self):
        cfg = ConsolidationConfig(iterations=8, sh_degree_schedule=(0, 1, 2, 3))
        consolidator = Consolidator(cfg, device="cpu")
        assert consolidator._sh_degree_for_iteration(0) == 0
        assert consolidator._sh_degree_for_iteration(cfg.iterations - 1) == 3

    def test_monotonically_non_decreasing(self):
        cfg = ConsolidationConfig(iterations=20, sh_degree_schedule=(0, 1, 2, 3))
        consolidator = Consolidator(cfg, device="cpu")
        degrees = [consolidator._sh_degree_for_iteration(i) for i in range(cfg.iterations)]
        assert degrees == sorted(degrees)
        assert set(degrees) <= set(cfg.sh_degree_schedule)


class TestConsolidateEndToEnd:
    """Exercises `consolidate()`'s full orchestration through `_FakeConsolidator`
    — everything except the real `_step` numeric core."""

    def _run(self, n_splats=6, n_frames=5, **cfg_kwargs):
        cloud = _make_cloud(n_splats)
        frames = [_frame() for _ in range(n_frames)]
        # 4 frames train, 1 held out for n_frames=5 (see _split_holdout: 5//5=1).
        # Visibility renders happen for train frames only, in order, then one
        # held-out render for PSNR — same order as `frames` overall.
        train_hits = [{0, 1}, {1, 2}, {2, 3}, {0, 3}]
        held_out_hit = {0, 1, 2}
        results = [_render_result(h) for h in train_hits] + [_render_result(held_out_hit)]
        fake_renderer = _FakeDifferentiableRenderer(results)
        cfg = ConsolidationConfig(
            iterations=cfg_kwargs.pop("iterations", 6),
            batch_size=cfg_kwargs.pop("batch_size", 2),
            min_views_for_confirm=cfg_kwargs.pop("min_views_for_confirm", 2),
            **cfg_kwargs,
        )
        consolidator = _FakeConsolidator(cfg, device="cpu", renderer=fake_renderer)
        cloud_out, report = consolidator.consolidate(cloud, frames)
        return consolidator, cloud, cloud_out, report

    def test_every_train_frame_gets_batched_at_least_once(self):
        consolidator, _, _, report = self._run()
        seen = set()
        for call in consolidator.step_calls:
            seen.update(call["frame_indices"])
        assert seen == {0, 1, 2, 3}, f"not every frame was batched: {seen}"
        assert report.iterations_run == 6

    def test_config_lrs_reach_the_step_calls(self):
        cfg_overrides = dict(lr_position=9.9e-3, lr_scale=1.23e-2, lr_sh_rest=4.56e-4)
        consolidator, *_ = self._run(**cfg_overrides)
        assert consolidator.step_calls, "step was never called"
        for call in consolidator.step_calls:
            assert call["lr_position"] == pytest.approx(9.9e-3)
            assert call["lr_scale"] == pytest.approx(1.23e-2)
            assert call["lr_sh_rest"] == pytest.approx(4.56e-4)

    def test_never_seen_splat_is_frozen_bit_identical(self):
        # splat 4 and 5 never appear in any train_hits set above.
        consolidator, before, after, _report = self._run(n_splats=6)
        frozen = [4, 5]
        assert np.array_equal(after.positions[frozen], before.positions[frozen])
        assert np.array_equal(after.colors[frozen], before.colors[frozen])
        assert np.array_equal(after.scales[frozen], before.scales[frozen])
        assert np.array_equal(after.rotations[frozen], before.rotations[frozen])
        assert after.provenance[frozen].tolist() == before.provenance[frozen].tolist()

    def test_seen_splats_get_promoted_per_threshold(self):
        # splat 0: seen in train frames {0,1} and {0,3} -> count 2 -> promoted.
        # splat 1: seen in {0,1} and {1,2} -> count 2 -> promoted.
        # splat 2: seen in {1,2} and {2,3} -> count 2 -> promoted.
        # splat 3: seen in {2,3} and {0,3} -> count 2 -> promoted.
        _, before, after, report = self._run(min_views_for_confirm=2)
        assert before.provenance[[0, 1, 2, 3]].tolist() == [Provenance.PRIOR] * 4
        assert after.provenance[[0, 1, 2, 3]].tolist() == [Provenance.OBSERVED] * 4
        assert report.promoted_to_observed == 4

    def test_report_fields_populated(self):
        _, _, _, report = self._run()
        assert isinstance(report, ConsolidationReport)
        assert report.frames_used == 5
        assert report.iterations_run == 6
        assert report.final_loss > 0.0
        assert report.promoted_to_observed >= 0
        assert report.elapsed_seconds >= 0.0
        assert set(report.psnr_by_frame.keys()) == {0}  # one held-out frame

    def test_held_out_psnr_reflects_render_vs_target(self):
        # Held-out frame's canned render color is 0.5 everywhere; make the
        # target image match closely so PSNR comes out high.
        cloud = _make_cloud(6)
        frames = [_frame() for _ in range(5)]
        frames[-1] = _frame(image_rgb=np.full((HEIGHT, WIDTH, 3), 128, np.uint8))
        train_hits = [{0, 1}, {1, 2}, {2, 3}, {0, 3}]
        results = [_render_result(h) for h in train_hits] + [_render_result({0}, color=0.5019608)]
        fake_renderer = _FakeDifferentiableRenderer(results)
        cfg = ConsolidationConfig(iterations=2, batch_size=2)
        consolidator = _FakeConsolidator(cfg, device="cpu", renderer=fake_renderer)

        _, report = consolidator.consolidate(cloud, frames)

        assert report.psnr_by_frame[0] > 40.0  # near-exact match -> high PSNR

    def test_empty_frames_is_a_noop(self):
        cloud = _make_cloud(4)
        consolidator = _FakeConsolidator(device="cpu", renderer=_FakeDifferentiableRenderer([]))
        out, report = consolidator.consolidate(cloud, [])
        assert out is cloud
        assert report.frames_used == 0
        assert report.iterations_run == 0

    def test_empty_cloud_is_a_noop(self):
        consolidator = _FakeConsolidator(device="cpu", renderer=_FakeDifferentiableRenderer([]))
        out, report = consolidator.consolidate(GaussianCloud.empty(), [_frame()])
        assert out.n == 0
        assert report.iterations_run == 0

    def test_nothing_visible_is_a_noop(self):
        cloud = _make_cloud(3)
        frames = [_frame()]
        fake_renderer = _FakeDifferentiableRenderer([_render_result(set())])
        consolidator = _FakeConsolidator(device="cpu", renderer=fake_renderer)
        out, report = consolidator.consolidate(cloud, frames)
        assert np.array_equal(out.positions, cloud.positions)
        assert report.iterations_run == 0
