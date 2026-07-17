"""End-to-end fusion demo — the CPU regression for the whole arbitration loop.

Story it tells, with numbers:
  1. One "photo" produces a complete model whose unseen regions are guesses.
  2. A walk-around video confirms region after region: %OBSERVED climbs,
     photometric surprise falls, and the silver guess turns red.
  3. A later low-quality session cannot damage what a good one confirmed.
  4. A badly-registered frame changes nothing at all.

Run: python demo/run_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows consoles default to cp1252, which cannot encode the bar/rule glyphs.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cargen.core.asset import VehicleAsset
from cargen.core.camera import Intrinsics
from cargen.core.splat import Provenance
from cargen.export.exporter import export_all
from cargen.fusion_engine.engine import FusionConfig
from cargen.fusion_engine.point_renderer import PointRenderer
from cargen.pipeline import Pipeline
from cargen.pose_estimation.stub import StubRegistrar
from cargen.prior_generation.interface import PriorGenerator
from cargen.reid.histogram import HistogramEmbedder
from cargen.video.frame_sampler import FrameSampler
from demo.synthetic import (
    PRIOR_COLORS,
    TRUTH_COLORS,
    BackgroundSegmenter,
    build_prior_cloud,
    build_truth_cloud,
    orbit_pose,
    render_photo,
    walkaround_frames,
)

WIDTH, HEIGHT = 256, 192


class _DemoPrior(PriorGenerator):
    """Stands in for the image-to-3D model: returns the pre-built prior cloud.

    Real backends (TRELLIS/SF3D) hallucinate this from the photo; here it is
    fixed so the demo measures fusion, not prior quality.
    """

    def __init__(self, cloud):
        self._cloud = cloud

    def generate_splats(self, image_rgb, mask, n_points=20_000):
        return self._cloud

    def generate(self, image_rgb, mask):
        raise NotImplementedError("demo prior returns splats directly")


def _bar(fraction: float, width: int = 24) -> str:
    filled = int(round(fraction * width))
    return "█" * filled + "·" * (width - filled)


def _row(label: str, cloud, extra: str = "") -> str:
    stats = cloud.stats()
    return (
        f"  {label:<22} {_bar(stats['observed_fraction'])} "
        f"{stats['observed_fraction']*100:5.1f}% observed  "
        f"{stats['splats']:6d} splats  conf {stats['mean_confidence']:.2f}  {extra}"
    )


def main() -> int:
    intrinsics = Intrinsics.simple(WIDTH, HEIGHT)
    renderer = PointRenderer()

    print("\n" + "=" * 78)
    print("cargen — dynamic vehicle reconstruction demo (synthetic, CPU, seeded)")
    print("=" * 78)

    truth = build_truth_cloud(intrinsics, renderer)
    prior = build_prior_cloud(truth)
    print(f"\nGround truth: {truth.n} splats, paint {TRUTH_COLORS['paint']} (red)")
    print(f"Prior guess : {prior.n} splats, paint {PRIOR_COLORS['paint']} (silver)")
    print("The prior has the right shape but the wrong paint — fusion must discover")
    print("the real colour from imagery, region by region, without being told.\n")

    pipeline = Pipeline(
        segmenter=BackgroundSegmenter(),
        prior_generator=_DemoPrior(prior),
        matcher=None,  # poses come from the stub registrar; no matching needed
        renderer=renderer,
        embedder=HistogramEmbedder(),
        registrar=StubRegistrar(confidence=0.9),
        fusion_config=FusionConfig(),
        sampler=FrameSampler(motion_threshold=6.0, max_frames=40),
    )

    # ---- 1. first photo -----------------------------------------------------
    print("─" * 78)
    print("STEP 1 — one photo arrives, model is created")
    print("─" * 78)
    asset = VehicleAsset(name="demo-car")
    photo, _ = render_photo(truth, orbit_pose(0.0), intrinsics, renderer, seed=1)
    result = pipeline.ingest_photo(asset, photo, device="phone", intrinsics=intrinsics,
                                   timestamp=0.0)
    print(_row("after first photo", asset.cloud))
    print("  → a complete car exists; only the photographed side is real.\n")

    # ---- 2. walk-around video ----------------------------------------------
    print("─" * 78)
    print("STEP 2 — a walk-around video arrives, guesses become evidence")
    print("─" * 78)
    frames, poses = walkaround_frames(truth, intrinsics, renderer, n_frames=24)
    print(f"  video: {len(frames)} frames at 12fps")

    sampled = pipeline.sampler.sample(iter(frames))
    print(f"  sampler kept {len(sampled)}/{len(frames)} frames "
          f"(near-duplicates skipped by motion magnitude)\n")

    for i, frame in enumerate(sampled):
        pose = poses[frame.index]
        mask = pipeline.segmenter.segment(frame.image)
        cloud, report = pipeline.engine.fuse_frame(
            asset.cloud, frame.image, mask, pose, intrinsics,
            registration_confidence=0.9, timestamp=10.0 + frame.timestamp,
            evidence_weight=0.85,
        )
        asset.cloud = cloud
        if i % 3 == 0 or i == len(sampled) - 1:
            print(_row(f"frame {frame.index:2d}", cloud,
                       f"surprise {report.mean_residual:.3f}"))
    print("  → %OBSERVED climbs as the camera reveals each region; surprise falls")
    print("    as the model stops disagreeing with reality.\n")

    # Compare only the splats that came from the prior: densification appends
    # new ones past truth.n, which have no ground-truth counterpart to check.
    original = asset.cloud.select(np.arange(truth.n))
    confirmed = original.provenance == Provenance.OBSERVED
    mean_color = original.colors[confirmed].mean(axis=0)
    truth_mean = truth.colors[confirmed].mean(axis=0)
    print(f"  mean colour of confirmed splats : {np.round(mean_color, 3)}")
    print(f"  same splats in ground truth     : {np.round(truth_mean, 3)}")
    print(f"  colour error                    : "
          f"{float(np.abs(mean_color - truth_mean).mean()):.4f}")
    print(f"  (silver guess was {PRIOR_COLORS['paint']} — fusion moved it to the"
          f" real paint)\n")

    # ---- 3. arbitration: weak evidence cannot damage confirmed regions ------
    print("─" * 78)
    print("STEP 3 — arbitration: a bad frame must not damage a good model")
    print("─" * 78)
    before = asset.cloud
    confirmed = (before.provenance == Provenance.OBSERVED) & (before.confidence > 0.8)
    colors_before = before.colors[confirmed].copy()
    view = orbit_pose(0.0)

    def drift_on_confirmed(cloud) -> float:
        # densification appends past before.n, so compare the original range
        return float(np.abs(cloud.colors[: before.n][confirmed] - colors_before).max())

    # (a) a pose we don't trust must be a non-event
    junk = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    unchanged, rejected = pipeline.engine.fuse_frame(
        before, junk, None, view, intrinsics,
        registration_confidence=0.05, timestamp=99.0,
    )
    print(f"  (a) badly registered frame: {rejected.reason}")
    print(f"      cloud object untouched: {unchanged is before}")

    # (b) same car, different light: exposure compensation must absorb it rather
    #     than flag the whole vehicle dirty
    dim_photo, dim_mask = render_photo(
        truth, view, intrinsics, renderer, exposure=0.7, seed=7
    )
    dim_cloud, dim_report = pipeline.engine.fuse_frame(
        before, dim_photo, dim_mask, view, intrinsics,
        registration_confidence=0.9, timestamp=100.0, evidence_weight=0.85,
    )
    print(f"  (b) same car at 0.7x exposure (a cloudy day):")
    print(f"      surprise {dim_report.mean_residual:.3f}, "
          f"{dim_report.dirty} splats touched, "
          f"drift on confirmed {drift_on_confirmed(dim_cloud):.4f}")

    # (c) weak evidence that genuinely CONTRADICTS the model: a green car at
    #     CCTV tier. Confirmed splats must refuse it; guesses must not.
    green = truth.with_updates(
        np.arange(truth.n),
        colors=np.tile(np.array([0.1, 0.7, 0.2], np.float32), (truth.n, 1)),
    )
    green_photo, green_mask = render_photo(green, view, intrinsics, renderer, seed=8)
    cctv_cloud, cctv_report = pipeline.engine.fuse_frame(
        before, green_photo, green_mask, view, intrinsics,
        registration_confidence=0.9, timestamp=101.0,
        evidence_weight=0.3,  # CCTV tier
    )
    print(f"  (c) a CONTRADICTING green frame at CCTV weight 0.3:")
    print(f"      surprise {cctv_report.mean_residual:.3f}, "
          f"{cctv_report.dirty} splats altered, "
          f"drift on {int(confirmed.sum())} confirmed splats "
          f"{drift_on_confirmed(cctv_cloud):.4f}")

    # ...but the same contradicting frame at phone weight DOES get through
    phone_cloud, phone_report = pipeline.engine.fuse_frame(
        before, green_photo, green_mask, view, intrinsics,
        registration_confidence=0.9, timestamp=102.0, evidence_weight=0.85,
    )
    print(f"      same frame at phone weight 0.85: "
          f"{phone_report.dirty} splats altered, "
          f"drift {drift_on_confirmed(phone_cloud):.4f}")
    print("  → rejected frames are non-events; lighting shifts get absorbed;")
    print("    confirmed regions refuse weak contradictions but yield to strong ones.\n")

    # ---- 4. export ----------------------------------------------------------
    print("─" * 78)
    print("STEP 4 — export")
    print("─" * 78)
    out = Path(__file__).resolve().parent.parent / "data" / "demo"
    paths = export_all(asset.cloud, out)
    for fmt, path in paths.items():
        print(f"  {fmt:16s} {Path(path).name:28s} "
              f"{Path(path).stat().st_size / 1024:8.1f} KB")

    stats = asset.cloud.stats()
    print("\n" + "=" * 78)
    print(f"RESULT  {stats['observed']}/{stats['splats']} splats confirmed "
          f"({stats['observed_fraction']*100:.1f}%) from 1 photo + "
          f"{len(sampled)} video frames")
    print(f"        {stats['prior']} splats remain factory-default guesses, "
          f"awaiting angles nobody has photographed yet.")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
