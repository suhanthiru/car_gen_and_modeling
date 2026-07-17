"""Incremental fusion — the arbitration core.

Per frame:
  1. Render the current splats from the frame's estimated pose.
  2. Exposure-compensate, then diff against the real frame → surprise map.
  3. Flag splats dirty where surprise is high AND the frame is trusted there;
     freeze everything else.
  4. Densify where real evidence exists but no geometry does.
  5. Update the dirty splats toward the observation (a localized optimization
     stands in for gsplat's Adam loop until Milestone B).
  6. Update provenance/confidence/view_count; prune dead splats.

THE ARBITRATION RULES, which are the point of the whole design:
  * A frame whose registration confidence is below threshold is REJECTED
    outright. A bad pose corrupts everything downstream, so a rejected frame
    must be a non-event, not a best-effort fuse.
  * PRIOR splats are cheap to overwrite — they are guesses.
  * OBSERVED splats resist: evidence must beat their accumulated confidence to
    change them. This is what stops a blurry CCTV frame from degrading a clean
    phone capture, and it is why `evidence_weight` (device tier / registration
    quality) is a first-class input.
  * Splats not visible in this frame are never touched.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from cargen.core.camera import CameraPose, Intrinsics
from cargen.core.splat import GaussianCloud, Provenance
from cargen.fusion_engine.renderer import SplatRenderer
from cargen.fusion_engine.residual import compensate_exposure, dilate_mask, residual_map


@dataclass
class FusionConfig:
    min_registration_confidence: float = 0.35
    residual_threshold: float = 0.10   # surprise above this is a candidate for dirty
    dilation_radius: int = 3           # blending ring around dirty regions
    densify_grid: int = 6              # px between densified splats
    # How far (px) a new splat may be spawned from existing geometry. This is
    # the guard against inventing depth in open space — see _densify.
    densify_reach: int = 8
    learning_rate: float = 0.6         # how far a CONFIRMED splat moves toward evidence
    confidence_gain: float = 0.35      # confidence added per confirming view
    prune_opacity: float = 0.05
    max_splats: int = 400_000
    observed_confidence_floor: float = 0.25  # evidence must beat this to alter OBSERVED
    # A PRIOR splat is a guess carrying no evidentiary weight, so the first real
    # look at it REPLACES its colour outright rather than averaging reality with
    # a hallucination. Only confirmed splats earn gentle refinement.
    prior_learning_rate: float = 1.0
    # Exposure may only be fit on splats confirmed by MULTIPLE views. One view
    # promotes a splat to OBSERVED while its colour is still half-guess; trusting
    # those would let the fit map reality onto the model's error. See
    # residual.compensate_exposure.
    exposure_trust_confidence: float = 0.75


@dataclass
class FusionReport:
    """What one frame actually changed — the audit trail for the merge/event log."""

    accepted: bool
    reason: str = ""
    registration_confidence: float = 0.0
    evidence_weight: float = 0.0
    splats_before: int = 0
    splats_after: int = 0
    dirty: int = 0
    densified: int = 0
    pruned: int = 0
    promoted: int = 0  # PRIOR → OBSERVED this frame
    mean_residual: float = 0.0
    observed_fraction_before: float = 0.0
    observed_fraction_after: float = 0.0
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d.pop("extras", None)
        d.update(self.extras)
        return d


class FusionEngine:
    def __init__(
        self,
        renderer: SplatRenderer,
        config: FusionConfig | None = None,
        optimizer=None,
    ):
        self._renderer = renderer
        self.config = config or FusionConfig()
        # LocalizedOptimizer when gsplat is available. The colour blend in
        # _recolor_dirty was always a stand-in for this: it can only repaint
        # splats, while the optimizer recovers geometry — a dent's *shape*, not
        # just its shading. Absent one, the CPU path still works end to end.
        self.optimizer = optimizer

    def fuse_frame(
        self,
        cloud: GaussianCloud,
        image_rgb: np.ndarray,
        mask: np.ndarray | None,
        pose: CameraPose,
        intrinsics: Intrinsics,
        registration_confidence: float,
        timestamp: float,
        evidence_weight: float = 1.0,
    ) -> tuple[GaussianCloud, FusionReport]:
        """Fuse one registered frame. Returns the new cloud and a report.

        `evidence_weight` in (0, 1] scales how authoritative this frame is:
        phone-with-AR-pose ≈ 1.0, handheld phone ≈ 0.8, Pi ≈ 0.6, CCTV ≈ 0.3.
        """
        cfg = self.config
        report = FusionReport(
            accepted=False,
            registration_confidence=registration_confidence,
            evidence_weight=evidence_weight,
            splats_before=cloud.n,
            splats_after=cloud.n,
            observed_fraction_before=cloud.observed_fraction(),
            observed_fraction_after=cloud.observed_fraction(),
        )

        if registration_confidence < cfg.min_registration_confidence:
            report.reason = (
                f"registration confidence {registration_confidence:.2f} < "
                f"{cfg.min_registration_confidence:.2f} — frame queued, not fused"
            )
            return cloud, report
        if cloud.n == 0:
            report.reason = "empty cloud — nothing to fuse against"
            return cloud, report

        observed = image_rgb.astype(np.float32) / 255.0
        render = self._renderer.render(cloud, pose, intrinsics)
        vehicle = (
            np.clip(mask, 0, 1) > 0.5
            if mask is not None
            else np.ones(observed.shape[:2], bool)
        )

        # Fit exposure on pixels we already trust, so lighting shifts don't
        # masquerade as change (see residual.py).
        trusted = self._trusted_pixels(cloud, render, vehicle)
        observed = compensate_exposure(render.color, observed, trusted)

        surprise = residual_map(render.color, observed)
        report.mean_residual = float(surprise[vehicle].mean()) if vehicle.any() else 0.0

        (
            agreeing_idx,
            disputed_idx,
            disputed_region,
            core_region,
        ) = self._partition_visible_splats(render, surprise, vehicle)
        dirty_idx = self._resist(cloud, disputed_idx, evidence_weight)

        cloud, dirty_idx, recolor_promoted = self._recolor_dirty(
            cloud, render, observed, disputed_region, dirty_idx, evidence_weight
        )
        if self.optimizer is not None and dirty_idx.size:
            cloud = self._refine(
                cloud, dirty_idx, observed, disputed_region, core_region,
                pose, intrinsics,
            )
        cloud, confirm_promoted = self._confirm_views(
            cloud, agreeing_idx, dirty_idx, timestamp, evidence_weight
        )
        report.dirty = int(dirty_idx.size)
        report.promoted = recolor_promoted + confirm_promoted

        cloud, densified = self._densify(
            cloud, render, surprise, observed, vehicle, pose, intrinsics,
            timestamp, evidence_weight,
        )
        report.densified = densified

        cloud, pruned = self._prune(cloud)
        report.pruned = pruned

        report.accepted = True
        report.reason = "fused"
        report.splats_after = cloud.n
        report.observed_fraction_after = cloud.observed_fraction()
        return cloud, report

    # -- steps ---------------------------------------------------------------

    def _trusted_pixels(self, cloud, render, vehicle) -> np.ndarray:
        """Pixels painted by splats several views agree on — the exposure reference.

        The bar is `exposure_trust_confidence`, not merely OBSERVED provenance:
        a splat confirmed once is still part guess, and fitting exposure against
        it would explain away real evidence (see FusionConfig).
        """
        trusted = np.zeros(vehicle.shape, bool)
        hit = render.hit_mask & vehicle
        if not hit.any():
            return trusted
        idx = render.splat_index[hit]
        believable = (cloud.provenance[idx] == Provenance.OBSERVED) & (
            cloud.confidence[idx] >= self.config.exposure_trust_confidence
        )
        trusted[hit] = believable
        return trusted

    def _partition_visible_splats(
        self, render, surprise, vehicle
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split the splats this frame can see into agreeing vs disputed.

        Returns (agreeing_idx, disputed_idx, disputed_region, core_region).
        `core_region` is the pixels that genuinely disagreed; `disputed_region`
        adds the blending ring around them. Both are needed downstream — the
        optimizer weights the core fully and the ring softly, so the boundary
        blends instead of seaming.

        Splats outside `hit` are invisible in this view and stay frozen — the
        engine never touches what the frame cannot see.
        """
        empty = np.zeros((0,), int)
        blank = np.zeros(vehicle.shape, bool)
        hit = render.hit_mask & vehicle
        if not hit.any():
            return empty, empty, blank, blank

        core = hit & (surprise > self.config.residual_threshold)
        disputed = dilate_mask(core, self.config.dilation_radius) & hit

        return (
            np.unique(render.splat_index[hit & ~disputed]),
            np.unique(render.splat_index[disputed]),
            disputed,
            core,
        )

    def _recolor_dirty(
        self, cloud, render, observed, region, dirty, evidence_weight
    ) -> tuple[GaussianCloud, np.ndarray, int]:
        """Pull dirty splats toward the observed colour and mark them OBSERVED."""
        if dirty.size == 0:
            return cloud, dirty, 0

        target = self._mean_observed_color(render.splat_index, observed, dirty, region)
        valid = ~np.isnan(target[:, 0])
        dirty = dirty[valid]
        if dirty.size == 0:
            return cloud, dirty, 0

        was_prior = cloud.provenance[dirty] == Provenance.PRIOR
        # Guesses are replaced outright; confirmed splats are nudged.
        #
        # Note evidence_weight scales only the OBSERVED path. A PRIOR splat is a
        # hallucination with zero evidentiary value, so even weak imagery beats
        # it and should replace it whole — down-weighting the replacement would
        # leave a permanent fraction of the guess alive, and (because the small
        # residual then falls under residual_threshold) the splat would be
        # declared "close enough" and never corrected again. How much we trust
        # the new observation is recorded in `confidence`, which governs who may
        # overwrite it next; it does not belong in how much guess survives.
        lr = np.where(
            was_prior,
            self.config.prior_learning_rate,
            self.config.learning_rate * evidence_weight,
        ).astype(np.float32)
        lr = np.clip(lr, 0.0, 1.0)[:, None]

        blended = (1 - lr) * cloud.colors[dirty] + lr * target[valid]
        cloud = cloud.with_updates(
            dirty,
            colors=np.clip(blended, 0, 1).astype(np.float32),
            provenance=np.full(dirty.size, Provenance.OBSERVED, np.uint8),
        )
        return cloud, dirty, int(was_prior.sum())

    def _refine(
        self, cloud, dirty, observed, region, core, pose, intrinsics
    ) -> GaussianCloud:
        """Run the localized optimizer over the dirty splats only.

        The pixel weights are what confine it: pixels that genuinely disagreed
        pull at full strength, the surrounding blend ring pulls softly, and
        everything else is exactly zero — a frame cannot pull on what it does
        not dispute.
        """
        weight = np.zeros(region.shape, np.float32)
        weight[region] = self.optimizer.config.ring_weight
        weight[core] = 1.0
        return self.optimizer.refine(cloud, dirty, observed, weight, pose, intrinsics)

    def _confirm_views(
        self, cloud, agreeing, dirty, timestamp, evidence_weight
    ) -> tuple[GaussianCloud, int]:
        """Bank the view: every splat this frame saw gains confidence and a count.

        Splats that already agreed with the model get promoted to OBSERVED once
        enough views have confirmed them — that is how a walk-around turns the
        prior's lucky guesses into real evidence without recolouring anything.
        """
        touched = np.unique(np.concatenate([agreeing, dirty])) if (
            agreeing.size or dirty.size
        ) else np.zeros((0,), int)
        touched = touched[touched >= 0]
        if touched.size == 0:
            return cloud, 0

        gain = self.config.confidence_gain * evidence_weight
        cloud = cloud.with_updates(
            touched,
            confidence=np.clip(cloud.confidence[touched] + gain, 0, 1).astype(np.float32),
            view_count=(cloud.view_count[touched] + 1).astype(np.int32),
            last_seen_ts=np.full(touched.size, timestamp, np.float64),
        )

        confirmed = np.setdiff1d(touched, dirty)
        confirmed = confirmed[
            cloud.confidence[confirmed] >= self.config.observed_confidence_floor
        ]
        if confirmed.size == 0:
            return cloud, 0
        promoted = int(np.sum(cloud.provenance[confirmed] == Provenance.PRIOR))
        cloud = cloud.with_updates(
            confirmed, provenance=np.full(confirmed.size, Provenance.OBSERVED, np.uint8)
        )
        return cloud, promoted

    def _resist(
        self, cloud: GaussianCloud, candidates: np.ndarray, evidence_weight: float
    ) -> np.ndarray:
        """Drop candidates whose accumulated confidence outweighs this evidence.

        PRIOR splats never resist — they are guesses, and overwriting them is the
        entire point. OBSERVED splats do: this is what protects a good capture
        from being degraded by a weak one.
        """
        candidates = candidates[candidates >= 0]
        if candidates.size == 0:
            return candidates
        is_prior = cloud.provenance[candidates] == Provenance.PRIOR
        beats_existing = evidence_weight > cloud.confidence[candidates] * (
            1.0 - self.config.observed_confidence_floor
        )
        return candidates[is_prior | beats_existing]

    @staticmethod
    def _mean_observed_color(
        splat_index: np.ndarray,
        observed: np.ndarray,
        splat_ids: np.ndarray,
        region: np.ndarray,
    ) -> np.ndarray:
        """Mean observed colour per splat over `region`; NaN where unseen."""
        out = np.full((splat_ids.size, 3), np.nan, np.float32)
        flat_idx = splat_index[region]
        flat_rgb = observed[region]
        if flat_idx.size == 0:
            return out
        order = np.argsort(flat_idx)
        sorted_idx, sorted_rgb = flat_idx[order], flat_rgb[order]
        bounds = np.searchsorted(sorted_idx, splat_ids)
        ends = np.searchsorted(sorted_idx, splat_ids, side="right")
        for i, (start, end) in enumerate(zip(bounds, ends)):
            if end > start:
                out[i] = sorted_rgb[start:end].mean(axis=0)
        return out

    def _densify(
        self, cloud, render, surprise, observed, vehicle, pose, intrinsics,
        timestamp, evidence_weight,
    ) -> tuple[GaussianCloud, int]:
        """Grow geometry where the photo shows vehicle the model doesn't have.

        This is how aftermarket parts, spoilers, and roof boxes enter a model
        whose prior never guessed them.

        DELIBERATELY CONSERVATIVE — it only extends *outward from geometry we
        already have*, never into open space. A pixel with nothing behind it has
        no recoverable depth from a single monocular view, so spawning there
        means inventing a distance. Doing that at a global median depth paints a
        flat slab of splats across the frame at whatever the camera happened to
        be looking at, which is exactly what a sloppy segmentation mask (a
        rectangle that calls the background "vehicle") will ask for.

        Restricting spawns to a `densify_reach` ring around real geometry means
        every new splat inherits depth from an actual neighbouring surface, and
        a bad mask can at worst fatten the silhouette slightly instead of
        wrecking the model. Genuinely detached structure has to wait for a view
        that connects it to something — or for Milestone B's optimizer, which
        recovers depth from gradients across multiple views instead of guessing.
        """
        cfg = self.config
        empty = vehicle & ~render.hit_mask
        if not empty.any() or cloud.n >= cfg.max_splats or not render.hit_mask.any():
            return cloud, 0

        # Nearest surface depth per pixel: erode = min-filter, so each empty
        # pixel picks up the closest depth within `densify_reach`. Pixels with
        # no geometry in reach stay +inf and are skipped.
        reach = cfg.densify_reach
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach * 2 + 1,) * 2)
        finite = np.where(render.hit_mask, render.depth, np.float32(1e9))
        local_depth = cv2.erode(finite, kernel)
        has_context = local_depth < 1e8

        # sparse sampling on a grid — one splat per cell, not per pixel
        grid = np.zeros_like(empty)
        grid[:: cfg.densify_grid, :: cfg.densify_grid] = True
        spawn = empty & grid & has_context
        if not spawn.any():
            return cloud, 0

        vs, us = np.nonzero(spawn)
        budget = cfg.max_splats - cloud.n
        if vs.size > budget:
            vs, us = vs[:budget], us[:budget]

        positions = self._unproject_pixels(
            us, vs, local_depth[vs, us].astype(np.float64), pose, intrinsics
        )
        colors = observed[vs, us].astype(np.float32)

        scale = float(np.median(cloud.scales)) if cloud.n else 0.01
        new = GaussianCloud.create(
            positions=positions.astype(np.float32),
            colors=np.clip(colors, 0, 1),
            default_scale=scale,
            provenance=int(Provenance.OBSERVED),
            confidence=float(np.clip(0.3 * evidence_weight, 0.05, 1.0)),
            view_count=1,
            last_seen_ts=timestamp,
        )
        return cloud.concat(new), int(new.n)

    @staticmethod
    def _unproject_pixels(us, vs, depth, pose: CameraPose, intr: Intrinsics) -> np.ndarray:
        """Pixels at per-pixel camera-frame depths → world points."""
        depth = np.broadcast_to(np.asarray(depth, np.float64), us.shape)
        x = (us - intr.cx) * depth / intr.fx
        y = (vs - intr.cy) * depth / intr.fy
        cam = np.stack([x, y, depth], axis=1)
        return (cam - pose.t) @ pose.R

    def _prune(self, cloud: GaussianCloud) -> tuple[GaussianCloud, int]:
        keep = cloud.opacities > self.config.prune_opacity
        pruned = int((~keep).sum())
        return (cloud.select(keep), pruned) if pruned else (cloud, 0)
