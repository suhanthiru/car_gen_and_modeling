"""Joint multi-view splat optimization: the photorealism pass — Milestone B.

INTEGRATION POINT (requires gsplat — see gsplat_renderer.py for the install)
----------------------------------------------------------------------------
`FusionEngine.fuse_frame` (engine.py) is deliberately *localized*: one frame in,
only the splats it disputes get touched, and confirmed splats are frozen by
construction. That is the right shape for incremental "someone sent one more
photo" updates, but it never sees two views at once, so nothing forces the
model to be multi-view *consistent* — two photos can each look plausible while
disagreeing about the car's actual shape.

`Consolidator` is the other half named in `docs/ROADMAP.md`'s "What stands
between here and photorealism": ~7k-30k joint Adam iterations over every
frame together (or a batch each step), so multi-view disagreement shows up as
gradient signal instead of being invisible one frame at a time. This is what
recovers real geometry (dents, panel gaps) and fills in `sh_rest` (the SH
bands that make a highlight slide across the paint as the camera orbits) —
neither of which a single localized pass, or a single photo's prior, can ever
produce.

THE SAME NOVELTY AS THE REST OF THIS ENGINE, restated for the joint case:
standard 3D Gaussian Splatting starts from COLMAP's sparse points and can only
reconstruct what a camera actually photographed. cargen instead seeds this
optimizer with the generative prior's *complete* guess, including the side of
the car no frame here has seen, and only touches splats at least one frame in
this batch actually saw (`_visibility_mask`). Never-seen splats are excluded
from the optimizable parameter set entirely — bit-identical in, bit-identical
out — exactly the contract `LocalizedOptimizer` gives non-dirty splats, just
applied across the whole frame set instead of one frame's dirty region.
Provenance rides along: splats seen by enough views (`min_views_for_confirm`)
graduate PRIOR -> OBSERVED; everything else keeps its existing provenance.

This module has no `torch`/`gsplat` import at module scope, so it stays
importable on a CPU-only machine (see `tests/test_consolidate.py`, which
exercises `_visibility_mask`, `_promote_provenance`, batching, and report
plumbing against a fake differentiable-renderer double). Only actually
*calling* `consolidate()`'s real training loop needs gsplat/CUDA.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from cargen.core.camera import CameraPose, Intrinsics
from cargen.core.splat import GaussianCloud, Provenance
from cargen.fusion_engine.gsplat_common import (
    logit as _logit,
    sh_colors_from_dc_rest,
    sigmoid as _sigmoid,
    weighted_l1_dssim,
)
from cargen.fusion_engine.renderer import SplatRenderer


@dataclass
class ConsolidationConfig:
    iterations: int = 15_000            # roadmap's 7k-30k range
    batch_size: int = 4
    lr_position: float = 1.6e-4
    lr_scale: float = 5e-3
    lr_rotation: float = 1e-3
    lr_opacity: float = 5e-2
    lr_color: float = 2.5e-3
    # Same 20x-slower rationale as OptimizeConfig: letting SH-rest move at full
    # speed lets the model explain away geometry error as "lighting".
    lr_sh_rest: float = 2.5e-3 / 20
    ssim_lambda: float = 0.2
    densify_every: int = 500
    opacity_reset_every: int = 3000
    min_views_for_confirm: int = 2
    # Coarse-to-fine SH band unlock, the standard 3DGS trick: fitting
    # view-dependence before geometry has converged lets the optimizer explain
    # away shape error as a lighting effect instead of fixing the shape.
    sh_degree_schedule: tuple[int, ...] = (0, 1, 2, 3)


@dataclass
class ConsolidationReport:
    frames_used: int
    iterations_run: int
    final_loss: float
    psnr_by_frame: dict[int, float]
    promoted_to_observed: int
    elapsed_seconds: float
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d.pop("extras", None)
        d.update(self.extras)
        return d


@dataclass
class FrameObservation:
    image_rgb: np.ndarray
    mask: np.ndarray | None
    pose: CameraPose
    intrinsics: Intrinsics
    evidence_weight: float = 1.0


class Consolidator:
    """Joint Adam optimization over every splat any supplied frame has seen.

    `renderer` is an injectable `SplatRenderer` (defaults to lazily building a
    real `GsplatRenderer` on `device`) used for the non-differentiable passes
    (`_visibility_mask`, `_held_out_psnr`). It is a deliberate small departure
    from the plan's `__init__(config, device)` shape: `LocalizedOptimizer`
    gets swapped wholesale by a fake in `FusionEngine`'s tests because it is
    *injected* into another object; `Consolidator` has no outer object to
    inject it into, so the same "test without CUDA" goal needs its own
    renderer to be injectable here instead. Passing nothing preserves the
    documented `Consolidator(config, device="cuda")` call shape exactly.
    """

    def __init__(
        self,
        config: ConsolidationConfig | None = None,
        device: str = "cuda",
        renderer: SplatRenderer | None = None,
    ):
        self.config = config or ConsolidationConfig()
        self._device = device
        self._renderer = renderer
        # Lazily populated on first real (non-overridden) `_step` call, so
        # __init__ and the CPU-testable methods never import torch.
        self._torch = None
        self._adam = None

    # -- orchestration --------------------------------------------------------

    def consolidate(
        self, cloud: GaussianCloud, frames: list[FrameObservation]
    ) -> tuple[GaussianCloud, ConsolidationReport]:
        """Jointly refine every splat seen by >=1 frame; return (cloud, report)."""
        start = time.time()
        report = ConsolidationReport(
            frames_used=len(frames), iterations_run=0, final_loss=0.0,
            psnr_by_frame={}, promoted_to_observed=0, elapsed_seconds=0.0,
        )
        if not frames or cloud.n == 0:
            report.elapsed_seconds = time.time() - start
            return cloud, report

        train_frames, held_out_frames = self._split_holdout(frames)
        seen_counts = self._visibility_mask(cloud, train_frames)
        optimizable = seen_counts > 0
        if not optimizable.any():
            report.elapsed_seconds = time.time() - start
            return cloud, report

        params, frozen = self._build_params(cloud, optimizable)
        self._adam = None  # fresh optimizer state for this run

        cfg = self.config
        n_train = len(train_frames)
        loss = 0.0
        for iteration in range(cfg.iterations):
            batch = self._next_batch(iteration, train_frames, cfg.batch_size)
            sh_degree = self._sh_degree_for_iteration(iteration)
            loss = self._step(params, frozen, batch, sh_degree)
            grads = self._position_grads(params)
            self._maybe_densify(iteration, params, grads)
            report.iterations_run = iteration + 1

        cloud = self._assemble_cloud(cloud, optimizable, params)
        cloud, promoted = self._promote_and_count(cloud, seen_counts)
        report.promoted_to_observed = promoted
        report.final_loss = float(loss)
        report.psnr_by_frame = self._held_out_psnr(cloud, held_out_frames)
        report.elapsed_seconds = time.time() - start
        return cloud, report

    @staticmethod
    def _split_holdout(
        frames: list[FrameObservation],
    ) -> tuple[list[FrameObservation], list[FrameObservation]]:
        """Reserve ~20% of frames (at least one, when there are enough to spare)
        for held-out PSNR reporting; training always keeps at least one frame."""
        if len(frames) <= 1:
            return list(frames), []
        n_holdout = max(1, len(frames) // 5)
        n_holdout = min(n_holdout, len(frames) - 1)
        return list(frames[:-n_holdout]), list(frames[-n_holdout:])

    @staticmethod
    def _next_batch(
        iteration: int, train_frames: list[FrameObservation], batch_size: int
    ) -> list[tuple[int, FrameObservation]]:
        """Deterministic round-robin batch — every frame gets visited within
        `n_train` iterations, not merely in expectation from random sampling."""
        n = len(train_frames)
        start_i = (iteration * batch_size) % n
        size = min(batch_size, n)
        idxs = [(start_i + k) % n for k in range(size)]
        return [(i, train_frames[i]) for i in idxs]

    def _sh_degree_for_iteration(self, iteration: int) -> int:
        """Unlock schedule[k] once iteration >= iterations * k/len(schedule)."""
        cfg = self.config
        schedule = cfg.sh_degree_schedule
        frac = iteration / max(cfg.iterations, 1)
        stage = min(int(frac * len(schedule)), len(schedule) - 1)
        return schedule[stage]

    # -- CPU-only (no torch/gsplat import) ------------------------------------

    def _visibility_mask(
        self, cloud: GaussianCloud, frames: list[FrameObservation]
    ) -> np.ndarray:
        """Per-splat count of how many `frames` see it (int32, shape (cloud.n,)).

        0 = never seen -> excluded from the optimizable set entirely, frozen
        exactly like `LocalizedOptimizer` freezes non-dirty splats. >=1 = at
        least one frame in this batch has evidence about it, so it is
        optimizable; `min_views_for_confirm` (checked separately, in
        `_promote_provenance`) governs provenance promotion, not eligibility.
        """
        counts = np.zeros(cloud.n, dtype=np.int32)
        if cloud.n == 0 or not frames:
            return counts
        renderer = self._get_renderer()
        for frame in frames:
            render = renderer.render(cloud, frame.pose, frame.intrinsics)
            idx = render.splat_index[render.hit_mask]
            if idx.size:
                seen = np.unique(idx[idx >= 0])
                counts[seen] += 1
        return counts

    def _build_params(self, cloud: GaussianCloud, optimizable: np.ndarray):
        """Numpy-only split into optimizable/frozen splats. Same activated
        parameterization `LocalizedOptimizer` uses (log-scale, logit-opacity),
        so a splat's raw numbers mean the same thing regardless of which
        optimizer last touched it."""
        frozen = {
            "means": cloud.positions[~optimizable].copy(),
            "quats": cloud.rotations[~optimizable].copy(),
            "scales": cloud.scales[~optimizable].copy(),
            "opacities": cloud.opacities[~optimizable].copy(),
            "colors": cloud.colors[~optimizable].copy(),
            "sh_rest": cloud.sh_rest[~optimizable].copy(),
        }
        params = {
            "means": cloud.positions[optimizable].copy(),
            "quats": cloud.rotations[optimizable].copy(),
            "log_scales": np.log(np.maximum(cloud.scales[optimizable], 1e-9)),
            "logit_opacities": _logit(cloud.opacities[optimizable]),
            "colors": cloud.colors[optimizable].copy(),
            "sh_rest": cloud.sh_rest[optimizable].copy(),
        }
        return params, frozen

    def _promote_and_count(
        self, cloud: GaussianCloud, seen_counts: np.ndarray
    ) -> tuple[GaussianCloud, int]:
        promoted_cloud = self._promote_provenance(cloud, seen_counts)
        promoted = int(
            np.sum(
                (promoted_cloud.provenance == Provenance.OBSERVED)
                & (cloud.provenance == Provenance.PRIOR)
            )
        )
        return promoted_cloud, promoted

    def _promote_provenance(
        self, cloud: GaussianCloud, seen_counts: np.ndarray
    ) -> GaussianCloud:
        """Splats seen >= `min_views_for_confirm` times: PRIOR -> OBSERVED.
        Everything else (including already-OBSERVED splats) is untouched."""
        confirmed = seen_counts >= self.config.min_views_for_confirm
        idx = np.where(confirmed & (cloud.provenance == Provenance.PRIOR))[0]
        if idx.size == 0:
            return cloud
        return cloud.with_updates(
            idx, provenance=np.full(idx.size, Provenance.OBSERVED, np.uint8)
        )

    def _assemble_cloud(
        self, cloud: GaussianCloud, optimizable: np.ndarray, params: dict
    ) -> GaussianCloud:
        """Write optimized values back at `optimizable`; frozen splats copy
        through bit-identical. Works whether `params` holds numpy arrays (a
        test double that never touched torch) or torch tensors (the real
        `_step` path)."""
        def _np(v):
            return v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)

        out_pos = cloud.positions.copy()
        out_rot = cloud.rotations.copy()
        out_scale = cloud.scales.copy()
        out_opac = cloud.opacities.copy()
        out_col = cloud.colors.copy()
        out_sh = cloud.sh_rest.copy()

        quats = _np(params["quats"])
        norm = np.linalg.norm(quats, axis=1, keepdims=True)
        quats = quats / np.clip(norm, 1e-9, None)

        out_pos[optimizable] = _np(params["means"])
        out_rot[optimizable] = quats
        out_scale[optimizable] = np.exp(_np(params["log_scales"]))
        out_opac[optimizable] = _sigmoid(_np(params["logit_opacities"]))
        out_col[optimizable] = np.clip(_np(params["colors"]), 0, 1)
        out_sh[optimizable] = _np(params["sh_rest"])

        return GaussianCloud(
            positions=out_pos.astype(np.float32),
            scales=out_scale.astype(np.float32),
            rotations=out_rot.astype(np.float32),
            opacities=out_opac.astype(np.float32),
            colors=out_col.astype(np.float32),
            sh_rest=out_sh.astype(np.float32),
            provenance=cloud.provenance, confidence=cloud.confidence,
            view_count=cloud.view_count, last_seen_ts=cloud.last_seen_ts,
        )

    def _held_out_psnr(
        self, cloud: GaussianCloud, held_out_frames: list[FrameObservation]
    ) -> dict[int, float]:
        """PSNR per held-out frame (keyed by its index within `held_out_frames`),
        rendered with no gradient tracking needed."""
        out: dict[int, float] = {}
        if not held_out_frames or cloud.n == 0:
            return out
        renderer = self._get_renderer()
        for i, frame in enumerate(held_out_frames):
            render = renderer.render(cloud, frame.pose, frame.intrinsics)
            target = (
                frame.image_rgb.astype(np.float32) / 255.0
                if frame.image_rgb.dtype == np.uint8
                else frame.image_rgb.astype(np.float32)
            )
            mse = float(np.mean((render.color - target) ** 2))
            out[i] = 99.0 if mse < 1e-10 else float(10.0 * np.log10(1.0 / mse))
        return out

    def _get_renderer(self) -> SplatRenderer:
        if self._renderer is None:
            from cargen.fusion_engine.gsplat_renderer import GsplatRenderer

            self._renderer = GsplatRenderer(device=self._device)
        return self._renderer

    @staticmethod
    def _position_grads(params: dict) -> np.ndarray | None:
        """Position-gradient magnitude from the last `_step`, for `_maybe_densify`.
        None on the CPU-testable path, where `params["means"]` never became a
        torch tensor with a populated `.grad`."""
        means = params.get("means")
        grad = getattr(means, "grad", None)
        if grad is None:
            return None
        return grad.detach().norm(dim=1).cpu().numpy()

    # -- requires torch/gsplat -------------------------------------------------

    def _require_torch(self):
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def _tensorize(self, params: dict, frozen: dict, t, dev: str) -> None:
        """Convert `params`/`frozen` (numpy) to persistent torch tensors, in
        place, so Adam's momentum survives across `_step` calls."""

        def tensor(a):
            return t.from_numpy(np.ascontiguousarray(a)).to(dev, t.float32)

        for key in list(frozen):
            frozen[key] = tensor(frozen[key])
        for key in ("means", "quats", "log_scales", "logit_opacities", "colors", "sh_rest"):
            params[key] = tensor(params[key]).requires_grad_(True)

    def _build_optimizer(self, params: dict, t):
        cfg = self.config
        return t.optim.Adam(
            [
                {"params": [params["means"]], "lr": cfg.lr_position},
                {"params": [params["quats"]], "lr": cfg.lr_rotation},
                {"params": [params["log_scales"]], "lr": cfg.lr_scale},
                {"params": [params["logit_opacities"]], "lr": cfg.lr_opacity},
                {"params": [params["colors"]], "lr": cfg.lr_color},
                {"params": [params["sh_rest"]], "lr": cfg.lr_sh_rest},
            ]
        )

    def _step(
        self,
        params: dict,
        frozen: dict,
        frames_batch: list[tuple[int, FrameObservation]],
        sh_degree: int,
    ) -> float:
        """One Adam step averaged over `frames_batch`.

        Calls `rasterization` directly (not `GsplatRenderer.render`, which
        wraps the call in `no_grad` for the inference-only path) — the same
        choice `LocalizedOptimizer.refine` makes, for the same reason: this is
        the one place that actually needs gradients to flow.
        """
        from gsplat import rasterization

        t = self._require_torch()
        dev = self._device
        if not isinstance(params["means"], t.Tensor):
            self._tensorize(params, frozen, t, dev)
            self._adam = self._build_optimizer(params, t)

        optimizer = self._adam
        optimizer.zero_grad(set_to_none=True)

        means = t.cat([frozen["means"], params["means"]])
        quats = t.cat([frozen["quats"], params["quats"]])
        scales = t.cat([frozen["scales"], t.exp(params["log_scales"])])
        opacities = t.cat([frozen["opacities"], t.sigmoid(params["logit_opacities"])])
        dc = t.cat([frozen["colors"], params["colors"].clamp(0, 1)])
        rest = t.cat([frozen["sh_rest"], params["sh_rest"]])
        colors_full = sh_colors_from_dc_rest(dc, rest)
        k = (sh_degree + 1) ** 2
        colors_active = colors_full[:, :k, :]

        cfg = self.config
        losses = []
        for _, frame in frames_batch:
            target = self._frame_tensor(frame.image_rgb, t, dev)
            weight = self._weight_tensor(frame, t, dev)
            viewmat = t.eye(4, device=dev)
            viewmat[:3, :3] = t.from_numpy(np.ascontiguousarray(frame.pose.R)).float().to(dev)
            viewmat[:3, 3] = t.from_numpy(np.ascontiguousarray(frame.pose.t)).float().to(dev)
            K = t.from_numpy(frame.intrinsics.K).float().to(dev)[None]
            h, w = frame.intrinsics.height, frame.intrinsics.width

            rendered, _, _ = rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities,
                colors=colors_active, viewmats=viewmat[None], Ks=K, width=w, height=h,
                # single camera -> gsplat squeezes the camera dim, so a plain-RGB
                # render expects a 1-D (channels,) background, not (1, channels).
                sh_degree=sh_degree, backgrounds=t.full((3,), 1.0, device=dev),
            )
            frame_loss = weighted_l1_dssim(
                rendered[..., :3], target[None], weight, cfg.ssim_lambda
            )
            losses.append(frame_loss * frame.evidence_weight)

        loss = t.stack(losses).mean()
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())

    def _frame_tensor(self, image_rgb: np.ndarray, t, dev: str):
        arr = (
            image_rgb.astype(np.float32) / 255.0
            if image_rgb.dtype == np.uint8
            else image_rgb.astype(np.float32)
        )
        return t.from_numpy(np.ascontiguousarray(arr)).to(dev)

    def _weight_tensor(self, frame: FrameObservation, t, dev: str):
        if frame.mask is not None:
            w = np.clip(frame.mask, 0, 1).astype(np.float32)
        else:
            w = np.ones(frame.image_rgb.shape[:2], np.float32)
        return t.from_numpy(np.ascontiguousarray(w)).to(dev)[None, ..., None]

    def _maybe_densify(self, iteration: int, params: dict, grads) -> None:
        """Opacity-reset cadence only, v1 — NOT CPU-testable (needs real
        gradients over real geometry; see docs/DEVELOPMENT.md's
        manual-acceptance-run note).

        Full clone/split densification (spawning new splats where gradient
        signal is high, standard 3DGS) is deferred: it needs gradient
        *accumulation* across many steps to avoid densifying on a single
        noisy batch, which `grads` (one step's magnitude) does not give it.
        Periodic opacity reset alone still guards against floaters
        accumulating unbounded confidence over a long run.
        """
        cfg = self.config
        if iteration == 0 or iteration % cfg.opacity_reset_every != 0:
            return
        t = self._require_torch()
        logit_opacities = params.get("logit_opacities")
        if logit_opacities is None or not hasattr(logit_opacities, "clamp_"):
            return
        floor = float(np.log(0.05 / 0.95))
        with t.no_grad():
            logit_opacities.clamp_(max=floor)
