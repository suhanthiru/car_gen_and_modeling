"""Export splats in the standard 3DGS formats.

`.ply` uses the exact field layout the original 3DGS release defined, so the
files open in SuperSplat, PlayCanvas, Blender's 3DGS addons, and every other
splat tool — the user's assets are plain, portable files, not a private format.

`.splat` is the compact binary the web viewer streams (32 bytes/splat).

`model_provenance.ply` is a debug twin colouring PRIOR vs OBSERVED, which drives
the viewer's overlay toggle.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cargen.core.splat import SH_REST_COEFFS, GaussianCloud, Provenance

# Inverse of the activations 3DGS applies when loading: scales are stored as
# log, opacity as logit, colour as SH DC coefficients.
_SH_C0 = 0.28209479177387814

PRIOR_TINT = np.array([0.95, 0.35, 0.25], np.float32)      # red-ish = a guess
OBSERVED_TINT = np.array([0.25, 0.80, 0.40], np.float32)   # green = confirmed


def _logit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def _rgb_to_sh_dc(colors: np.ndarray) -> np.ndarray:
    return (colors - 0.5) / _SH_C0


def write_ply(cloud: GaussianCloud, path: str | Path, sh_rest: bool = True) -> Path:
    """Standard 3DGS binary .ply.

    Writes SH degree 3 (`f_rest_0..44`) whenever the cloud has non-zero higher
    bands, and degree 0 otherwise — a prior has nothing but DC, and emitting 45
    zero floats per splat would triple the file for no information.

    `f_rest` ordering is CHANNEL-MAJOR — all 15 red coefficients, then green,
    then blue — because the reference implementation stores `(N, 15, 3)` and
    writes `.transpose(1, 2).flatten(1)`. Get this wrong and viewers still load
    the file, but the colours smear as you orbit. The field names/order are a
    contract with SuperSplat, PlayCanvas and Blender, not an internal detail.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = cloud.n
    with_rest = sh_rest and cloud.is_view_dependent

    dtype = [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
    ]
    if with_rest:
        dtype += [(f"f_rest_{i}", "<f4") for i in range(SH_REST_COEFFS * 3)]
    dtype += [
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ]

    data = np.zeros(n, dtype=dtype)
    data["x"], data["y"], data["z"] = cloud.positions.T
    sh = _rgb_to_sh_dc(cloud.colors)
    data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = sh.T
    if with_rest:
        # (N, 15, 3) -> (N, 3, 15) -> (N, 45): red block, green block, blue block
        flat = np.transpose(cloud.sh_rest, (0, 2, 1)).reshape(n, -1)
        for i in range(SH_REST_COEFFS * 3):
            data[f"f_rest_{i}"] = flat[:, i]
    data["opacity"] = _logit(cloud.opacities)
    log_scales = np.log(np.maximum(cloud.scales, 1e-9))
    data["scale_0"], data["scale_1"], data["scale_2"] = log_scales.T
    data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"] = cloud.rotations.T

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {name}\n" for name, _ in dtype)
        + "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    return path


def write_provenance_ply(cloud: GaussianCloud, path: str | Path) -> Path:
    """Debug twin: colour by provenance, brightness by confidence.

    Makes "what does the model actually know vs. guess?" directly visible —
    the single most useful view when judging whether fusion is working.
    """
    tint = np.where(
        (cloud.provenance == Provenance.OBSERVED)[:, None], OBSERVED_TINT, PRIOR_TINT
    )
    shade = (0.35 + 0.65 * np.clip(cloud.confidence, 0, 1))[:, None]
    recolored = GaussianCloud(
        positions=cloud.positions, scales=cloud.scales, rotations=cloud.rotations,
        opacities=cloud.opacities, colors=(tint * shade).astype(np.float32),
        # deliberately flat: this view answers "what is real?", and a view-
        # dependent tint would make provenance shimmer as the camera moves
        sh_rest=np.zeros_like(cloud.sh_rest),
        provenance=cloud.provenance, confidence=cloud.confidence,
        view_count=cloud.view_count, last_seen_ts=cloud.last_seen_ts,
    )
    return write_ply(recolored, path)


def write_splat(cloud: GaussianCloud, path: str | Path) -> Path:
    """`.splat`: 32 bytes/splat — pos(12) scale(12) rgba(4) quat(4).

    Sorted by a size/opacity significance heuristic so a truncated prefix of the
    file is still a usable low-detail model — that is what makes progressive
    streaming work in the viewer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    significance = cloud.opacities * cloud.scales.prod(axis=1)
    order = np.argsort(-significance)

    buffer = np.zeros((cloud.n, 32), np.uint8)
    positions = cloud.positions[order].astype("<f4")
    scales = cloud.scales[order].astype("<f4")
    buffer[:, 0:12] = positions.view(np.uint8).reshape(-1, 12)
    buffer[:, 12:24] = scales.view(np.uint8).reshape(-1, 12)
    buffer[:, 24:27] = np.clip(cloud.colors[order] * 255, 0, 255).astype(np.uint8)
    buffer[:, 27] = np.clip(cloud.opacities[order] * 255, 0, 255).astype(np.uint8)
    # quaternion packed to bytes: (q * 128 + 128), the .splat convention
    quats = cloud.rotations[order]
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.where(norms > 0, norms, 1.0)
    buffer[:, 28:32] = np.clip(quats * 128 + 128, 0, 255).astype(np.uint8)

    path.write_bytes(buffer.tobytes())
    return path


def export_all(cloud: GaussianCloud, directory: str | Path, stem: str = "model") -> dict:
    """Write the full export set; returns {format: path}."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "ply": str(write_ply(cloud, directory / f"{stem}.ply")),
        "splat": str(write_splat(cloud, directory / f"{stem}.splat")),
        "provenance_ply": str(
            write_provenance_ply(cloud, directory / f"{stem}_provenance.ply")
        ),
    }
