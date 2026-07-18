"""Normalization into the canonical object frame, shared by every prior backend.

Generative image→3D models emit their own conventions (usually y-up, unit-ish
cube, arbitrary centering). The canonical frame the rest of cargen assumes is:
+x forward, +y left, +z up, ground plane at z=0, vehicle length ≈ 2.0.

Metric scale is deliberately discarded — Sim(3) registration absorbs per-session
scale, so a consistent normalized frame is what matters.
"""
from __future__ import annotations

import numpy as np

CANONICAL_LENGTH = 2.0

# y-up/z-forward (SF3D, TRELLIS, Tripo) → z-up/x-forward (canonical).
_YUP_TO_ZUP = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], np.float32)


def canonicalize_orientation(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a vehicle onto its own axes: length→x, width→y, height→z.

    WHY THIS IS NOT OPTIONAL. Image-to-3D models emit a *view-aligned* frame,
    not a canonical object frame — measured: the same car photographed at
    azimuth 0.0 and 1.2 rad comes back with its length axis along -z and -x
    respectively, tilted by the camera's elevation. Merely rotating y-up→z-up
    (as `normalize_to_canonical` does) cannot fix that, so without this step:
      * the vehicle sits tilted and oversized on the viewer's floor (its z
        extent inflates because a tilted car is "taller");
      * every capture session lands in a different orientation, so merging two
        scans of one car — the whole point of the asset model — fuses garbage.

    Cars are reliably length > width > height, so PCA recovers the axes. The
    up-sign is ambiguous under PCA (an eigenvector negated is still an
    eigenvector), resolved by the fact that a car's underside spreads wider than
    its roof.

    Returns (rotated_vertices, rotation) so callers can apply the same rotation
    to companion data (normals, per-splat rotations).
    """
    v = np.asarray(vertices, np.float64)
    centred = v - v.mean(axis=0)
    # eigh returns ascending eigenvalues; take the largest-spread axis first
    _, evecs = np.linalg.eigh(np.cov(centred.T))
    axes = evecs[:, ::-1]  # columns: length, width, height

    rotation = axes.T  # rows map world → canonical
    if np.linalg.det(rotation) < 0:
        rotation[2] *= -1  # keep it a rotation, not a reflection
    out = centred @ rotation.T

    # Disambiguate up: a car narrows in WIDTH toward the roof — the beltline and
    # fenders are wider than the greenhouse. Compare width (y) spread of the top
    # vs bottom decile, NOT length+width: a fastback/SUV roofline can be as long
    # as the base, so including the length axis lets it swamp the real signal and
    # flip the car upside down (measured on a crossover).
    z = out[:, 2]
    low, high = np.percentile(z, 10), np.percentile(z, 90)
    width_low = np.ptp(out[z <= low, 1]) if (z <= low).any() else 0.0
    width_high = np.ptp(out[z >= high, 1]) if (z >= high).any() else 0.0
    if width_high > width_low:  # roof wider than base → we're upside down
        rotation[1:] *= -1  # flip y and z together to stay right-handed
        out = centred @ rotation.T

    return out.astype(np.float32), rotation.astype(np.float32)


def normalize_to_canonical(
    vertices: np.ndarray, from_y_up: bool = True, use_pca: bool = False
) -> tuple[np.ndarray, float]:
    """Rotate, center, ground, and rescale points into the canonical frame.

    `use_pca=True` recovers the vehicle's own axes first (see
    `canonicalize_orientation`) and supersedes `from_y_up`. Real image-to-3D
    backends need it — their output frame follows the input camera. The stub
    prior does not: it is built in the canonical frame already, and running PCA
    on it would only add noise.

    Returns the transformed points and the uniform scale factor applied, so
    callers holding companion quantities in the same units (e.g. a Gaussian's
    own scales) can apply the same factor.
    """
    v = np.asarray(vertices, np.float32)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"expected (N,3) points, got {v.shape}")
    if v.shape[0] == 0:
        raise ValueError("cannot normalize an empty point set")

    if use_pca:
        v, _ = canonicalize_orientation(v)
    elif from_y_up:
        v = v @ _YUP_TO_ZUP.T
    v = v - v.mean(axis=0)
    v[:, 2] -= v[:, 2].min()

    length = float(v[:, 0].max() - v[:, 0].min())
    scale = CANONICAL_LENGTH / max(length, 1e-6)
    return (v * scale).astype(np.float32), scale
