"""Pipeline orchestration — photos and videos in, updated VehicleAsset out.

Deliberately knows nothing about which models are installed: every stage arrives
via `backends.py`, so this file is identical whether it is driving stubs on a
laptop or TRELLIS + gsplat on the production box.

Two entry points, both operating on a persistent asset:
  * `ingest_photo`  — bootstraps the prior on first sight, else fuses.
  * `ingest_video`  — samples frames by motion, then runs the same fusion loop
                      per sampled frame with cheap frame-to-frame tracking.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from cargen.core.asset import VehicleAsset
from cargen.core.camera import CameraPose, Intrinsics
from cargen.fusion_engine.engine import FusionConfig, FusionEngine, FusionReport
from cargen.pose_estimation.fallback import FallbackRegistrar
from cargen.pose_estimation.interface import Registration, Registrar
from cargen.pose_estimation.registration import LandmarkStore, PnPRegistrar
from cargen.pose_estimation.render_reid import RenderReidRegistrar
from cargen.pose_estimation.video_tracker import VideoTracker
from cargen.reid.verify import RenderVerifier
from cargen.video.frame_sampler import FrameSampler

# How authoritative each device tier's imagery is (see FusionEngine.fuse_frame).
# A blurry wide-angle CCTV grab must never overwrite a clean phone capture.
DEVICE_EVIDENCE_WEIGHT = {
    "phone_ar": 1.0,   # phone with ARCore/ARKit pose — best evidence available
    "phone": 0.85,
    "camera": 0.8,
    "pi": 0.6,
    "cctv": 0.3,
    "unknown": 0.5,
}


def evidence_weight_for(device: str | None) -> float:
    return DEVICE_EVIDENCE_WEIGHT.get((device or "unknown").lower(), 0.5)


@dataclass
class IngestResult:
    asset: VehicleAsset
    created: bool = False
    reports: list[FusionReport] = field(default_factory=list)
    frames_sampled: int = 0
    frames_fused: int = 0
    frames_rejected: int = 0
    embedding: np.ndarray | None = None

    def summary(self) -> dict:
        return {
            "created": self.created,
            "frames_sampled": self.frames_sampled,
            "frames_fused": self.frames_fused,
            "frames_rejected": self.frames_rejected,
            **self.asset.cloud.stats(),
        }


class Pipeline:
    def __init__(
        self,
        segmenter=None,
        prior_generator=None,
        matcher=None,
        renderer=None,
        embedder=None,
        registrar: Registrar | None = None,
        optimizer=None,
        fusion_config: FusionConfig | None = None,
        # 20k splats over a whole vehicle is ~4cm per splat — coarse enough to
        # read as gravel rather than bodywork. 120k is the quality floor for a
        # car; real 3DGS assets run 200k-1M. Costs CPU-path render time (that
        # renderer loops in Python); gsplat removes the ceiling.
        prior_points: int = 120_000,
        sampler: FrameSampler | None = None,
    ):
        from cargen import backends

        self.segmenter = segmenter or backends.build_segmenter()
        self.prior_generator = prior_generator or backends.build_prior_generator()
        self.matcher = matcher or backends.build_matcher()
        self.renderer = renderer or backends.build_renderer()
        self.embedder = embedder or backends.build_embedder()
        # PnP is the default: cheap, precise when there's keypoint overlap, and
        # it never blocks. The render-based re-ID fallback (render_reid.py) is
        # opt-in via CARGEN_RENDER_REID=1 because a candidate-pose sweep on the
        # *CPU* PointRenderer is minutes per photo on a real (120k-splat) cloud
        # — it only makes sense against a GPU renderer. When enabled it slots in
        # behind PnP so it only fires when PnP can't register the frame.
        if registrar is not None:
            self.registrar = registrar
        elif os.environ.get("CARGEN_RENDER_REID", "0") == "1":
            self.registrar = FallbackRegistrar([
                PnPRegistrar(self.matcher),
                RenderReidRegistrar(RenderVerifier(self.renderer, self.embedder)),
            ])
        else:
            self.registrar = PnPRegistrar(self.matcher)
        # A differentiable renderer earns the real refinement loop; the CPU
        # stand-in only repaints splats. Driven off the renderer's own
        # declaration so the capability is explicit, not probed.
        if optimizer is None and self.renderer.supports_gradients:
            from cargen.fusion_engine.optimize import LocalizedOptimizer

            optimizer = LocalizedOptimizer()
        self.optimizer = optimizer
        self.engine = FusionEngine(self.renderer, fusion_config, optimizer=optimizer)
        self.sampler = sampler or FrameSampler()
        self.prior_points = prior_points
        # Landmarks are per-asset but rebuilt from observations on load; kept
        # in-process for now (they are a cache, not source of truth).
        self._landmarks: dict[str, LandmarkStore] = {}

    def landmarks_for(self, asset: VehicleAsset) -> LandmarkStore:
        return self._landmarks.setdefault(asset.vehicle_id, LandmarkStore())

    def embed(self, image_rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        return self.embedder.embed(image_rgb, mask)

    def export_raw_prior(self, path) -> bool:
        """Write the prior backend's own mesh alongside our splat exports."""
        return self.prior_generator.export_raw_mesh(path)

    # -- photo ---------------------------------------------------------------

    def ingest_photo(
        self,
        asset: VehicleAsset,
        image_rgb: np.ndarray,
        device: str = "phone",
        intrinsics: Intrinsics | None = None,
        timestamp: float | None = None,
        known_pose: CameraPose | None = None,
    ) -> IngestResult:
        """Fuse one photo; bootstrap the asset's prior if it has none yet."""
        timestamp = time.time() if timestamp is None else timestamp
        intrinsics = intrinsics or Intrinsics.simple(*image_rgb.shape[1::-1])
        mask = self.segmenter.segment(image_rgb)
        embedding = self.embed(image_rgb, mask)

        if asset.cloud.n == 0:
            return self._bootstrap(
                asset, image_rgb, mask, intrinsics, device, timestamp, embedding
            )

        registration = self._register(
            asset, image_rgb, mask, intrinsics, known_pose
        )
        return self._fuse_single(
            asset, image_rgb, mask, intrinsics, registration, device, timestamp, embedding
        )

    def _bootstrap(
        self, asset, image_rgb, mask, intrinsics, device, timestamp, embedding
    ) -> IngestResult:
        """First photo: generate the prior and anchor the canonical frame.

        The prior is generated *from* this photo, so this camera's pose relative
        to it is known by construction — the one registration we never have to
        estimate. That anchor is what every later photo registers against.
        """
        cloud = self.prior_generator.generate_splats(
            image_rgb, mask, n_points=self.prior_points
        )
        pose = self._canonical_bootstrap_pose(cloud, intrinsics)
        asset.cloud = cloud

        # Confirm the side the photo actually shows: real pixels replace the
        # prior's guess there, and those splats become the first landmarks.
        cloud, report = self.engine.fuse_frame(
            cloud, image_rgb, mask, pose, intrinsics,
            registration_confidence=1.0,
            timestamp=timestamp,
            evidence_weight=evidence_weight_for(device),
        )
        asset.cloud = cloud
        self._record_landmarks(asset, image_rgb, mask, pose, intrinsics)
        asset.add_observation(
            {
                "kind": "photo",
                "device": device,
                "ts": timestamp,
                "bootstrap": True,
                "report": report.as_dict(),
            },
            embedding,
        )
        return IngestResult(
            asset=asset, created=True, reports=[report],
            frames_sampled=1, frames_fused=1, embedding=embedding,
        )

    def _canonical_bootstrap_pose(
        self, cloud, intrinsics: Intrinsics
    ) -> CameraPose:
        """Where the first camera sits relative to a prior generated from it.

        The prior arrives in the canonical frame (+x forward, +z up), oriented
        so the generating view looks at the vehicle's front-three-quarter. We
        place the camera to frame the cloud's extent — no estimation involved.
        """
        center = cloud.positions.mean(axis=0)
        radius = float(np.linalg.norm(cloud.positions - center, axis=1).max())
        # back off far enough that the whole vehicle fits the frame
        distance = radius / np.tan(np.arctan(0.5 * intrinsics.width / intrinsics.fx)) * 1.4
        eye = center + np.array([distance * 0.82, -distance * 0.52, radius * 0.55])
        return CameraPose.look_at(eye=eye, target=center)

    def _register(
        self, asset, image_rgb, mask, intrinsics, known_pose: CameraPose | None
    ) -> Registration:
        """Locate a new photo. A supplied AR pose skips estimation entirely."""
        if known_pose is not None:
            return Registration(
                pose=known_pose, confidence=1.0, inliers=0,
                reprojection_error=0.0, reason="pose supplied by device (AR)",
            )
        return self.registrar.register(
            image_rgb, mask, intrinsics,
            {"asset": asset, "landmarks": self.landmarks_for(asset)},
        )

    def _fuse_single(
        self, asset, image_rgb, mask, intrinsics, registration, device, timestamp, embedding
    ) -> IngestResult:
        if not registration.ok:
            asset.add_observation(
                {
                    "kind": "photo", "device": device, "ts": timestamp,
                    "rejected": True, "reason": registration.reason,
                },
                embedding,
            )
            return IngestResult(
                asset=asset, frames_sampled=1, frames_rejected=1, embedding=embedding
            )

        cloud, report = self.engine.fuse_frame(
            asset.cloud, image_rgb, mask, registration.pose, intrinsics,
            registration_confidence=registration.confidence,
            timestamp=timestamp,
            evidence_weight=evidence_weight_for(device),
        )
        if report.accepted:
            asset.cloud = cloud
            self._record_landmarks(asset, image_rgb, mask, registration.pose, intrinsics)
            asset.add_frame(
                image_rgb, mask, registration.pose, intrinsics,
                evidence_weight=evidence_weight_for(device),
            )
        asset.add_observation(
            {
                "kind": "photo", "device": device, "ts": timestamp,
                "registration": {
                    "confidence": registration.confidence,
                    "inliers": registration.inliers,
                    "reason": registration.reason,
                },
                "report": report.as_dict(),
            },
            embedding,
        )
        return IngestResult(
            asset=asset,
            reports=[report],
            frames_sampled=1,
            frames_fused=int(report.accepted),
            frames_rejected=int(not report.accepted),
            embedding=embedding,
        )

    def _record_landmarks(self, asset, image_rgb, mask, pose, intrinsics) -> None:
        """Bank this view's confirmed geometry as registration anchors.

        Only splats this view actually saw are stored, so future photos can
        never register against hallucinated regions (see registration.py).
        """
        render = self.renderer.render(asset.cloud, pose, intrinsics)
        hit = render.hit_mask
        if mask is not None:
            hit = hit & (np.clip(mask, 0, 1) > 0.5)
        if not hit.any():
            return
        vs, us = np.nonzero(hit)
        # subsample: a few thousand landmarks per view is plenty for PnP
        if vs.size > 3000:
            step = vs.size // 3000
            vs, us = vs[::step], us[::step]
        idx = render.splat_index[vs, us]
        keep = idx >= 0
        if not keep.any():
            return
        self.landmarks_for(asset).add_view(
            image_rgb,
            mask,
            asset.cloud.positions[idx[keep]].astype(np.float64),
            np.stack([us[keep], vs[keep]], axis=1).astype(np.float64),
        )

    # -- video ---------------------------------------------------------------

    def ingest_video(
        self,
        asset: VehicleAsset,
        frames,
        device: str = "phone",
        intrinsics: Intrinsics | None = None,
        timestamp: float | None = None,
        known_poses: list[CameraPose] | None = None,
    ) -> IngestResult:
        """Fuse a walk-around clip.

        `frames` is an iterable of (index, RGB, timestamp_seconds) — normally
        `iter_video_frames(path)`. Only frames carrying new viewpoint
        information are fused; the rest are near-duplicates that cost time and
        add nothing.
        """
        base_ts = time.time() if timestamp is None else timestamp
        sampled = self.sampler.sample(frames)
        if not sampled:
            return IngestResult(asset=asset)

        first = sampled[0].image
        intrinsics = intrinsics or Intrinsics.simple(*first.shape[1::-1])
        result = IngestResult(asset=asset, frames_sampled=len(sampled))

        if asset.cloud.n == 0:
            # no model yet: the clip's first frame bootstraps one, then the rest
            # of the walk-around confirms it
            boot = self.ingest_photo(
                asset, first, device=device, intrinsics=intrinsics, timestamp=base_ts
            )
            result.created = True
            result.reports.extend(boot.reports)
            result.frames_fused += boot.frames_fused
            result.embedding = boot.embedding
            sampled = sampled[1:]

        tracker = VideoTracker(
            matcher=self.matcher,
            registrar=self.registrar,
            intrinsics=intrinsics,
            depth_lookup=lambda pose, uv: self.renderer.unproject(
                asset.cloud, pose, intrinsics, uv
            ),
            min_confidence=self.engine.config.min_registration_confidence,
        )

        for i, frame in enumerate(sampled):
            mask = self.segmenter.segment(frame.image)
            registration = (
                Registration(
                    pose=known_poses[i], confidence=1.0, inliers=0,
                    reprojection_error=0.0, reason="pose supplied by device (AR)",
                )
                if known_poses is not None and i < len(known_poses)
                else tracker.track(
                    frame.image, mask,
                    {"asset": asset, "landmarks": self.landmarks_for(asset)},
                )
            )
            if not registration.ok:
                result.frames_rejected += 1
                continue

            cloud, report = self.engine.fuse_frame(
                asset.cloud, frame.image, mask, registration.pose, intrinsics,
                registration_confidence=registration.confidence,
                timestamp=base_ts + frame.timestamp,
                evidence_weight=evidence_weight_for(device),
            )
            result.reports.append(report)
            if report.accepted:
                asset.cloud = cloud
                result.frames_fused += 1
                self._record_landmarks(
                    asset, frame.image, mask, registration.pose, intrinsics
                )
            else:
                result.frames_rejected += 1

        embedding = result.embedding
        if embedding is None:
            embedding = self.embed(first, self.segmenter.segment(first))
            result.embedding = embedding
        asset.add_observation(
            {
                "kind": "video", "device": device, "ts": base_ts,
                "frames_sampled": result.frames_sampled,
                "frames_fused": result.frames_fused,
                "frames_rejected": result.frames_rejected,
            },
            embedding,
        )
        return result

    # -- consolidation ---------------------------------------------------

    def consolidate(self, asset: VehicleAsset, config=None):
        """Joint multi-view refinement over every persisted frame — Milestone B.

        Unlike `ingest_photo`/`ingest_video`'s per-frame `fuse_frame` loop, this
        runs one optimizer over ALL of `asset.load_frames()` at once (~7k-30k
        iterations), which is what actually forces multi-view-consistent,
        photoreal geometry. Requires a real GPU + gsplat (see
        `scripts/verify_gsplat.py`); there is no CPU fallback, since a
        Python-loop CPU version of this would take hours.

        Mutates `asset.cloud` in place and returns the `ConsolidationReport`.
        Caller is responsible for `asset.save(...)` afterward.
        """
        from cargen.fusion_engine.consolidate import Consolidator, FrameObservation

        frames = [
            FrameObservation(
                image_rgb=f.image_rgb,
                mask=f.mask,
                pose=f.pose,
                intrinsics=f.intrinsics,
                evidence_weight=f.evidence_weight,
            )
            for f in asset.load_frames()
        ]
        consolidator = Consolidator(config=config, device="cuda")
        refined_cloud, report = consolidator.consolidate(asset.cloud, frames)
        asset.cloud = refined_cloud
        return report
