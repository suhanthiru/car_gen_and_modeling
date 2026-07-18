"""Run the Milestone-B joint multi-view consolidation pass on a stored vehicle.

Not a pytest file: it needs CUDA + a working gsplat build (see
`scripts/verify_gsplat.py`) and a real vehicle with persisted frames — neither
of which the CPU test suite has (see `tests/test_consolidate.py`'s fake
renderer for the CPU-only contract tests).

This is the actual photorealism pass: unlike the per-photo fusion loop that
runs automatically on every upload, this jointly optimizes ALL of a vehicle's
stored frames together for ~7k-30k iterations, which is what forces
multi-view-consistent geometry (see docs/ROADMAP.md). Run it once you have a
vehicle with a real walk-around video ingested (50+ frames recommended for the
roadmap's >30dB PSNR target — fewer frames still runs, just won't hit that bar).

Usage:
    python scripts/consolidate_vehicle.py <vehicle-name> [--iterations N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows + Python 3.8+ resolves a .pyd's dependent DLLs via the default search
# order (which does NOT include PATH) — see verify_gsplat.py for the same guard.
import os

if os.name == "nt":
    for _var in ("CUDA_PATH", "CUDA_HOME"):
        _root = os.environ.get(_var)
        if _root:
            for _sub in ("bin", os.path.join("bin", "x64")):
                _dll_dir = os.path.join(_root, _sub)
                if os.path.isdir(_dll_dir):
                    os.add_dll_directory(_dll_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vehicle", help="Vehicle folder/display name, e.g. bobs-civic")
    parser.add_argument("--iterations", type=int, default=None,
                         help="Override ConsolidationConfig.iterations (default 15000)")
    args = parser.parse_args()

    from cargen.fusion_engine.consolidate import ConsolidationConfig
    from server.config import CONFIG
    from server.store import VehicleStore

    store = VehicleStore(CONFIG)
    folder = store.find_by_name(args.vehicle) or args.vehicle
    if not store.exists(folder):
        print(f"FAIL: no vehicle found matching {args.vehicle!r} under {CONFIG.storage_root}")
        return 1

    asset = store.load(folder)
    frames = asset.load_frames()
    print(f"Vehicle {asset.name!r}: {asset.cloud.n} splats, {len(frames)} persisted frames")
    if not frames:
        print("FAIL: this vehicle has no persisted frames yet — ingest a photo/video first "
              "(frames are only recorded for accepted, registered observations).")
        return 1
    if len(frames) < 20:
        print(f"NOTE: only {len(frames)} frames — docs/ROADMAP.md's >30dB PSNR target assumes "
              "50+ good frames. This will still run, just won't hit photoreal quality yet.")

    config = ConsolidationConfig()
    if args.iterations is not None:
        config.iterations = args.iterations

    from cargen.pipeline import Pipeline

    pipe = Pipeline()
    print(f"Consolidating {len(frames)} frames over {config.iterations} iterations "
          f"(this runs on GPU and can take a while)...")
    t0 = time.time()
    report = pipe.consolidate(asset, config=config)
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s — iterations_run={report.iterations_run}, "
          f"final_loss={report.final_loss:.5f}, promoted_to_observed={report.promoted_to_observed}")
    if report.psnr_by_frame:
        mean_psnr = sum(report.psnr_by_frame.values()) / len(report.psnr_by_frame)
        print(f"mean held-out PSNR: {mean_psnr:.1f} dB")

    store.save(folder, asset)
    print(f"Saved refined cloud + re-exported .ply/.splat to {CONFIG.vehicle_dir(folder)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
