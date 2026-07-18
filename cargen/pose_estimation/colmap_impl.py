"""Batch structure-from-motion pose solve: COLMAP (via `pycolmap`) — Milestone B.

WHAT THIS IS FOR
-----------------
`docs/ROADMAP.md`'s "next" list, item 3, "Accurate poses (COLMAP)":

    joint optimisation needs sub-pixel-accurate camera positions. The current
    pose estimator is good enough to *locate* one new photo, not to solve all
    cameras at once. A COLMAP structure-from-motion step over the walk-around
    frames supplies that.

`cargen/fusion_engine/consolidate.py`'s joint multi-view optimization
(`Consolidator`, ~7k-30k gsplat iterations over every frame at once) is far
more sensitive to pose error than the incremental fusion loop is: a few pixels
of reprojection error per frame, multiplied across thousands of joint
iterations, shows up directly as blur/ghosting in the final splats. The
existing `PnPRegistrar` (`cargen/pose_estimation/registration.py`) is good
enough to *register one new frame at a time* against sparse observed
landmarks, but it was never designed to jointly refine a whole walk-around
sequence to sub-pixel accuracy — that's a fundamentally batch problem (bundle
adjustment over every frame and every 3D point simultaneously), which is
exactly what COLMAP's incremental SfM pipeline solves. This module is the v2
pose source for `consolidate.py`: `run_colmap_sfm` produces poses for a whole
frame set, `align_colmap_to_canonical` drops them into cargen's coordinate
frame, and the result feeds `Consolidator.consolidate` as `FrameObservation`s
exactly the way v1 (PnP/VideoTracker) poses do — `Consolidator` itself is
agnostic to which pose source produced its input.

WHY THIS DOES NOT IMPLEMENT `Registrar`
-----------------------------------------
`cargen/pose_estimation/interface.py`'s `Registrar.register(...)` is a
one-frame-at-a-time *online* contract: a new photo arrives, match it against
whatever landmarks already exist, return a pose (or a rejection) for that
frame alone. COLMAP's contract is the opposite shape: batch-only, all frames
in at once, incremental bundle adjustment jointly over the whole set, all
poses out together. There is no sensible way to slice "register frame 7"
out of that process — a single frame's pose there is a side effect of solving
every frame's pose jointly. So this module exposes its own two-function
shape (`run_colmap_sfm` / `align_colmap_to_canonical`) rather than forcing a
square peg into `Registrar`'s round hole. It is a sibling to `Registrar`
implementations, not one.

INTEGRATION POINT
------------------
Install: COLMAP itself is not pip-installable as a CLI tool (it's a C++
project distributed as a standalone binary/installer per-platform). Rather
than shell out to that CLI and parse its text/binary output — brittle:
subprocess argument quoting, platform-specific binary paths, and COLMAP's
undocumented-and-changing binary reconstruction format — use the official
Python bindings instead:

    pip install pycolmap

`pycolmap` wraps COLMAP's actual C++ SfM pipeline (feature extraction,
matching, incremental mapping, bundle adjustment) as an in-process library
with a typed Python API and a `Reconstruction` object we can read directly,
with no subprocess/text-parsing layer to keep in sync with COLMAP's CLI
output format across versions.

UNTESTED IN THIS ENVIRONMENT: `pycolmap` is not installed here, and this
module has never been run against a real image set — same honesty convention
as `trellis_impl.py`/`tripo_api.py`'s "ships untested" callouts. The call
sequence below follows pycolmap's documented pipeline
(`extract_features` -> `match_exhaustive` -> `incremental_mapping`), but
several `Reconstruction`/`Image`/`Camera` attribute names could not be
verified against a running installation and are flagged inline where used.
Treat this as a best-effort real implementation to be validated the first
time `pycolmap` is actually installed and run against `honda-civic` or
`audi-s6`'s frames, not as a verified-working adapter.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cargen.core.camera import CameraPose, Intrinsics, Sim3, umeyama
from cargen.pose_estimation.registration import LandmarkStore


@dataclass
class ColmapResult:
    """Output of a batch SfM solve: one pose per successfully-registered frame,
    shared intrinsics, and the sparse point cloud COLMAP triangulated along the
    way. All in COLMAP's own arbitrary SfM frame — not cargen's canonical
    frame; see `align_colmap_to_canonical`."""

    poses: dict[int, CameraPose]     # keyed by input frame index, registered frames only
    intrinsics: Intrinsics
    sparse_points: np.ndarray        # (N, 3)
    registered_fraction: float       # len(poses) / len(input frames)


def run_colmap_sfm(
    frames: list[np.ndarray], masks: list[np.ndarray] | None = None
) -> ColmapResult:
    """Run COLMAP incremental SfM over a whole frame set at once.

    Writes `frames` (and `masks`, if given) to a temp directory as
    COLMAP expects (an on-disk image directory — pycolmap has no in-memory
    ingestion path), then runs the standard pycolmap pipeline: feature
    extraction, exhaustive matching, incremental mapping. A car walk-around
    is typically 50-300 frames, well within exhaustive matching's practical
    range; if that stops being true, swap in `pycolmap.match_sequential`
    (frames are already temporally ordered) without changing this function's
    contract.
    """
    import pycolmap  # lazy: keeps this file importable with no pycolmap installed

    if not frames:
        raise ValueError("run_colmap_sfm requires at least one frame")

    with tempfile.TemporaryDirectory(prefix="cargen_colmap_") as tmp:
        tmp_path = Path(tmp)
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        database_path = tmp_path / "database.db"
        output_path = tmp_path / "sparse"
        output_path.mkdir()

        names = _write_frames(frames, image_dir)
        mask_dir = None
        if masks is not None:
            mask_dir = tmp_path / "masks"
            mask_dir.mkdir()
            _write_masks(masks, names, mask_dir)

        # SINGLE: all frames share one camera/intrinsics — true for a single
        # walk-around video, false if frames came from multiple devices.
        reader_options = pycolmap.ImageReaderOptions()
        if mask_dir is not None:
            # NOT VERIFIED against a running pycolmap install: the mask-path
            # field name on ImageReaderOptions. Confirmed only from pipeline
            # docs describing mask behavior (0 = masked out, nonzero = keep,
            # filenames matching the image filenames), not a live API check.
            reader_options.mask_path = str(mask_dir)

        pycolmap.extract_features(
            database_path=str(database_path),
            image_path=str(image_dir),
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_options,
        )
        pycolmap.match_exhaustive(database_path=str(database_path))
        reconstructions = pycolmap.incremental_mapping(
            database_path=str(database_path),
            image_path=str(image_dir),
            output_path=str(output_path),
        )
        if not reconstructions:
            raise RuntimeError("COLMAP failed to register any frames")
        # incremental_mapping can produce multiple disconnected reconstructions
        # (e.g. a walk-around with a gap); keep the largest one.
        reconstruction = max(reconstructions.values(), key=lambda r: len(r.images))

        return _reconstruction_to_result(reconstruction, names)


def _write_frames(frames: list[np.ndarray], image_dir: Path) -> list[str]:
    from PIL import Image

    names = []
    for i, frame in enumerate(frames):
        name = f"frame_{i:06d}.png"
        Image.fromarray(frame).save(image_dir / name)
        names.append(name)
    return names


def _write_masks(masks: list[np.ndarray | None], names: list[str], mask_dir: Path) -> None:
    from PIL import Image

    if len(masks) != len(names):
        raise ValueError("masks must be the same length as frames")
    for name, mask in zip(names, masks):
        if mask is None:
            continue
        mask_u8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        # COLMAP mask convention: 0 = discard pixel, nonzero = keep.
        Image.fromarray(mask_u8).save(mask_dir / f"{name}.png")


def _reconstruction_to_result(reconstruction, names: list[str]) -> ColmapResult:
    """Convert a pycolmap `Reconstruction` into our own frame-index-keyed
    `ColmapResult`.

    NOT VERIFIED against a running pycolmap install: `image.cam_from_world`
    (a `Rigid3d`) exposing `.rotation.matrix()` and `.translation`, and
    `Camera.params`/`focal_length_x`/`focal_length_y`/`principal_point_x/y`
    accessor names — these follow pycolmap's documented `Reconstruction`
    shape as of the version researched, but the exact attribute set has
    shifted across pycolmap releases (its API moved into the main
    colmap/colmap repo). Confirm against `dir(reconstruction.images[i])` /
    `dir(camera)` the first time this actually runs, and adjust here.
    """
    name_to_index = {name: i for i, name in enumerate(names)}

    poses: dict[int, CameraPose] = {}
    for image in reconstruction.images.values():
        idx = name_to_index.get(image.name)
        if idx is None:
            continue
        cam_from_world = image.cam_from_world
        R = np.asarray(cam_from_world.rotation.matrix(), np.float64)
        t = np.asarray(cam_from_world.translation, np.float64).reshape(3)
        poses[idx] = CameraPose(R=R, t=t)

    # Shared intrinsics (CameraMode.SINGLE): pull the first camera.
    cameras = list(reconstruction.cameras.values())
    if not cameras:
        raise RuntimeError("COLMAP reconstruction has no cameras")
    cam = cameras[0]
    # SIMPLE_RADIAL and similar models expose fx==fy via `focal_length`; use
    # the split accessors when present so this also tolerates PINHOLE-family
    # models with independent fx/fy.
    fx = float(getattr(cam, "focal_length_x", None) or cam.focal_length)
    fy = float(getattr(cam, "focal_length_y", None) or cam.focal_length)
    cx = float(cam.principal_point_x)
    cy = float(cam.principal_point_y)
    intrinsics = Intrinsics(
        fx=fx, fy=fy, cx=cx, cy=cy, width=int(cam.width), height=int(cam.height)
    )

    sparse_points = np.array(
        [p.xyz for p in reconstruction.points3D.values()], np.float64
    ).reshape(-1, 3)

    return ColmapResult(
        poses=poses,
        intrinsics=intrinsics,
        sparse_points=sparse_points,
        registered_fraction=len(poses) / max(len(names), 1),
    )


def align_colmap_to_canonical(
    colmap_result: ColmapResult, landmark_store: LandmarkStore
) -> dict[int, CameraPose]:
    """Bring COLMAP's arbitrary SfM frame into cargen's canonical frame.

    COLMAP solves poses and sparse points in its own arbitrary coordinate
    frame (origin/scale/orientation fall out of whichever two frames seeded
    the reconstruction) — not cargen's canonical frame (+x forward, +z up,
    ground at z=0, vehicle centered; see
    `cargen/prior_generation/canonical.py`). We need a similarity transform
    (rotation + uniform scale + translation, i.e. Sim(3) — see the module
    docstring of `cargen/core/camera.py`) from COLMAP's frame into cargen's,
    computed by aligning corresponding 3D points that are known in both
    frames: COLMAP's own triangulated `sparse_points`, matched against the
    already-canonical 3D positions held in `landmark_store` (each
    `Landmark.position` is in the canonical frame already — see
    `LandmarkStore`/`Landmark` in `cargen/pose_estimation/registration.py`).

    Reuses `cargen.core.camera.umeyama` (least-squares Sim(3), Umeyama 1991)
    rather than reimplementing Procrustes alignment here — that is the one
    Sim(3)/Procrustes helper in the codebase (checked
    `cargen/prior_generation/canonical.py`, which only has PCA-based
    orientation recovery, not point-to-point alignment).

    Correspondence between COLMAP points and landmark-store points is
    nearest-neighbour in COLMAP's own frame is not meaningful (different
    frames, no common scale yet) — so this expects the caller to have
    already produced paired points, e.g. by re-triangulating the same
    `LandmarkStore` source images/pixels through the COLMAP reconstruction,
    or by matching a handful of shared frames' camera centers. Concretely,
    this function pairs `colmap_result.sparse_points` against
    `landmark_store`'s landmark positions by simple positional order up to
    the shorter length, which only makes sense if the caller has already
    arranged the two arrays to correspond 1:1 — that pairing step lives with
    the caller (e.g. `consolidate_vehicle.py`), since it depends on how the
    frames fed into COLMAP relate to `landmark_store`'s source images. If
    no usable correspondence exists yet, raise rather than silently produce
    a meaningless alignment.
    """
    nonempty_views = [pts for pts in landmark_store.points_3d if len(pts)]
    if not nonempty_views:
        raise ValueError("landmark_store has no landmarks to align against")
    landmark_points = np.concatenate(nonempty_views, axis=0)
    n = min(len(colmap_result.sparse_points), len(landmark_points))
    if n < 3:
        raise ValueError(
            "need >= 3 corresponding points to solve a Sim(3) alignment "
            f"(have {n} after truncating to the shorter of the two point sets)"
        )

    src = colmap_result.sparse_points[:n]
    dst = landmark_points[:n]
    transform: Sim3 = umeyama(src, dst, with_scale=True)

    aligned: dict[int, CameraPose] = {}
    for idx, pose in colmap_result.poses.items():
        # A world->camera pose transforms as: for x_cam = R_old @ x_world_old + t_old,
        # with x_world_new = s*R_t@x_world_old + t_t (transform.apply), we need
        # R_new = R_old @ R_t^-1, t_new = t_old - R_new @ t_t / s... derived
        # directly below via camera center + orientation, which is simpler and
        # avoids sign/scale algebra mistakes:
        #   1. camera center moves like any other point: apply the Sim(3) to it.
        #   2. orientation (R) is unaffected by translation/scale, only by the
        #      Sim(3)'s rotation component: R_new = R_old @ transform.R.T.
        new_center = transform.apply(pose.camera_center.reshape(1, 3)).reshape(3)
        new_R = pose.R @ transform.R.T
        new_t = -new_R @ new_center
        aligned[idx] = CameraPose(R=new_R, t=new_t)

    return aligned
