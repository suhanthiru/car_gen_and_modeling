"""Symbolic geometric priors: a car has known shape constraints beyond what
any single photo shows — it's bilaterally symmetric, and it has two known
material bands (tinted glass in the greenhouse, smooth painted panel below).

Both passes here are deliberately narrow and bounded:
  * they only ever touch PRIOR (guessed) splats — an OBSERVED splat, backed by
    a real photo, is never overwritten, since real asymmetry (one-sided
    damage, badges, aftermarket parts) is common enough that mirroring or
    smoothing over it would be a regression, not an improvement;
  * they don't change `provenance` or `confidence` — a mirrored/smoothed
    splat is exactly as overwritable by a future real photo as the raw guess
    it replaced, because it IS still just a guess, only a better-informed one.

See docs/ROADMAP.md-adjacent design note (this session): "neurosymbolic AI is
overkill for cleaning up a single bad prior" — these are the cheap version of
that idea: hard-coded, well-understood constraints, not a learned model.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from cargen.core.splat import GaussianCloud, Provenance

# Canonical frame: x=length, y=width (mirror axis), z=height (see
# cargen/prior_generation/canonical.py). Splats within this distance of a
# reflected OBSERVED point are considered "the same feature, mirrored" and
# eligible for retargeting. Canonical car length is ~2.0 units.
DEFAULT_MIRROR_RADIUS = 0.05

# Height bands as a fraction of the cloud's own z-extent (post-grounding, so
# z=0 is the wheel-contact floor). Beltline/roofline proportions are rough
# but conservative -- these are estimation bands, not exact panel lines.
GLASS_BAND = (0.55, 0.90)
BODY_BAND = (0.05, 0.55)
SMOOTH_K_NEIGHBORS = 12
SMOOTH_COLOR_OUTLIER = 0.15  # RGB distance (0-1 scale) beyond which a splat is "noise"
SMOOTH_BLEND = 0.7  # how far an outlier moves toward its local neighborhood median


def _mirror_quaternion(rotations: np.ndarray) -> np.ndarray:
    """Reflect a batch of wxyz quaternions across the y=0 plane.

    Derived by conjugating the rotation matrix R by M=diag(1,-1,1) (the y-flip
    reflection) term-by-term and matching against the standard quaternion->
    matrix formula: R' = M R M corresponds exactly to negating the x and z
    quaternion components, w and y unchanged. Round-trips: mirroring twice
    returns the original quaternion.
    """
    w, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    return np.stack([w, -x, y, -z], axis=1).astype(np.float32)


def mirror_confirmed(
    cloud: GaussianCloud, radius: float = DEFAULT_MIRROR_RADIUS
) -> tuple[GaussianCloud, int]:
    """Fill still-guessed splats with a mirror of the confirmed opposite side.

    Cars are bilaterally symmetric; a photo of one side gives no benefit to
    the other side's guess quality today. This reflects every OBSERVED splat
    across the car's y=0 centerline and, where a PRIOR splat already exists
    near that reflected position, overwrites its geometry and appearance
    (never its provenance/confidence — see module docstring). Splats with no
    nearby match are left alone; this pass never inserts new splats.

    Returns (new_cloud, n_mirrored).
    """
    observed = np.where(cloud.provenance == Provenance.OBSERVED)[0]
    if observed.size == 0 or cloud.n == 0:
        return cloud, 0

    mirrored_pos = cloud.positions[observed].copy()
    mirrored_pos[:, 1] *= -1
    mirrored_rot = _mirror_quaternion(cloud.rotations[observed])

    tree = cKDTree(cloud.positions)
    dist, nearest = tree.query(mirrored_pos, k=1)

    is_prior = cloud.provenance[nearest] == Provenance.PRIOR
    within_radius = dist <= radius
    eligible = within_radius & is_prior
    if not eligible.any():
        return cloud, 0

    targets = nearest[eligible]
    n = np.count_nonzero(eligible)
    new_cloud = cloud.with_updates(
        targets,
        positions=mirrored_pos[eligible],
        rotations=mirrored_rot[eligible],
        scales=cloud.scales[observed][eligible],
        opacities=cloud.opacities[observed][eligible],
        colors=cloud.colors[observed][eligible],
        # SH bands 1-3 encode view-dependent gloss; reflecting them correctly
        # needs a proper spherical-harmonic reflection, real complexity this
        # doesn't need yet -- zero matches any splat's state this early in a
        # capture anyway (see cargen/core/splat.py's GaussianCloud docstring).
        sh_rest=np.zeros((n, cloud.sh_rest.shape[1], 3), np.float32),
    )
    return new_cloud, n


def smooth_material_bands(cloud: GaussianCloud) -> tuple[GaussianCloud, int]:
    """Suppress color/opacity outliers within a car's known material bands.

    Glass (tinted, low variance) and painted body panel (smooth, low
    variance) are each locally coherent materials. A single-view guess can
    produce speckled, high-variance noise within a band where the real
    surface should read as one clean material. This is an outlier-suppression
    pass, not a blur: only splats whose color departs sharply from their own
    local neighborhood are touched, and they're pulled toward THIS car's own
    local median (never a hardcoded "glass is grey" constant), so a
    well-behaved region -- including anything OBSERVED -- is untouched by
    construction.
    """
    if cloud.n == 0:
        return cloud, 0

    z = cloud.positions[:, 2]
    z_range = float(z.max() - z.min())
    if z_range <= 1e-6:
        return cloud, 0
    frac = (z - z.min()) / z_range

    new_colors = cloud.colors.copy()
    new_opacities = cloud.opacities.copy()
    total_touched = 0

    for lo, hi in (GLASS_BAND, BODY_BAND):
        in_band = np.where((frac >= lo) & (frac < hi))[0]
        if in_band.size < SMOOTH_K_NEIGHBORS + 1:
            continue

        band_pos = cloud.positions[in_band]
        tree = cKDTree(band_pos)
        k = min(SMOOTH_K_NEIGHBORS + 1, in_band.size)  # +1: query includes self
        _, neighbor_idx = tree.query(band_pos, k=k)

        neighbor_colors = cloud.colors[in_band][neighbor_idx]  # (n_band, k, 3)
        local_median = np.median(neighbor_colors, axis=1)
        dist = np.linalg.norm(cloud.colors[in_band] - local_median, axis=1)

        is_prior = cloud.provenance[in_band] == Provenance.PRIOR
        outlier = is_prior & (dist > SMOOTH_COLOR_OUTLIER)
        if not outlier.any():
            continue

        idx = in_band[outlier]
        blend = SMOOTH_BLEND
        new_colors[idx] = (
            (1 - blend) * cloud.colors[idx] + blend * local_median[outlier]
        ).astype(np.float32)
        local_median_opacity = np.median(
            cloud.opacities[in_band][neighbor_idx], axis=1
        )
        new_opacities[idx] = (
            (1 - blend) * cloud.opacities[idx] + blend * local_median_opacity[outlier]
        ).astype(np.float32)
        total_touched += int(outlier.sum())

    if total_touched == 0:
        return cloud, 0

    touched_idx = np.where(np.any(new_colors != cloud.colors, axis=1))[0]
    new_cloud = cloud.with_updates(
        touched_idx,
        colors=new_colors[touched_idx],
        opacities=new_opacities[touched_idx],
    )
    return new_cloud, total_touched
