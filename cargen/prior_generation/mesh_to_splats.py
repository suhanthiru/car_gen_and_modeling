"""Convert a prior mesh into an initial Gaussian cloud (provenance=PRIOR).

THE TWO THINGS THAT MAKE THIS LOOK LIKE A CAR RATHER THAN A BALL-PIT
--------------------------------------------------------------------
1. **Splats must be flat and surface-aligned.** A Gaussian with equal scales is a
   sphere; a few hundred thousand spheres scattered on a surface read as gravel,
   never as bodywork. Each splat here gets two axes tangent to its source
   triangle and a third collapsed along the normal — a disc lying *in* the
   surface — plus the rotation that puts it there. This is the single biggest
   determinant of whether the output looks like an object.

2. **Sample the texture, not the vertices.** A baked albedo map holds ~1M texels;
   a mesh's vertex colours hold one sample per vertex (~12k for SF3D). Averaging
   the map down to vertex colours and interpolating those back throws away ~80x
   the detail — badges, panel gaps, tail lights all smear into flat paint. When
   the mesh carries uv+texture we barycentrically interpolate UVs and sample the
   map directly; vertex colours are the fallback only.
"""
from __future__ import annotations

import numpy as np

from cargen.core.splat import GaussianCloud, Provenance
from cargen.prior_generation.interface import Mesh


def _face_normals(tri: np.ndarray) -> np.ndarray:
    """(F,3,3) triangles → (F,3) unit normals."""
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    # degenerate slivers have no normal; park them on +z rather than emit NaN
    return np.where(length > 1e-12, n / np.maximum(length, 1e-12), np.array([0.0, 0.0, 1.0]))


def _tangent_frames(normals: np.ndarray) -> np.ndarray:
    """(N,3) unit normals → (N,3,3) rotation matrices with columns [t1, t2, n].

    Local +z maps to the normal, so a scale of (r, r, r*thin) yields a disc lying
    in the surface. The seed axis is chosen per-normal to avoid the degenerate
    cross product when the normal happens to be parallel to it.
    """
    n = normals
    seed = np.where(
        (np.abs(n[:, 2:3]) < 0.9), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    )
    t1 = np.cross(seed, n)
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12)
    t2 = np.cross(n, t1)
    return np.stack([t1, t2, n], axis=2)  # columns


def _matrix_to_quaternion(m: np.ndarray) -> np.ndarray:
    """(N,3,3) rotations → (N,4) quaternions, w first.

    w-first matches the 3DGS .ply field order (rot_0..rot_3) that the exporter
    writes and GaussianCloud.create's identity default ([1,0,0,0]).
    Branchless Shepperd's method: pick the largest diagonal term per row to keep
    the square root away from zero.
    """
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = np.zeros((m.shape[0], 4), np.float64)

    # case 0: trace is comfortably positive
    big = trace > 0
    if big.any():
        s = np.sqrt(trace[big] + 1.0) * 2.0
        q[big, 0] = 0.25 * s
        q[big, 1] = (m[big, 2, 1] - m[big, 1, 2]) / s
        q[big, 2] = (m[big, 0, 2] - m[big, 2, 0]) / s
        q[big, 3] = (m[big, 1, 0] - m[big, 0, 1]) / s

    rest = ~big
    if rest.any():
        d = np.stack([m[rest, 0, 0], m[rest, 1, 1], m[rest, 2, 2]], axis=1)
        pick = np.argmax(d, axis=1)
        sub = m[rest]
        qr = np.zeros((sub.shape[0], 4), np.float64)
        for axis in (0, 1, 2):
            sel = pick == axis
            if not sel.any():
                continue
            i, j, k = axis, (axis + 1) % 3, (axis + 2) % 3
            s = np.sqrt(1.0 + sub[sel, i, i] - sub[sel, j, j] - sub[sel, k, k]) * 2.0
            qr[sel, 0] = (sub[sel, k, j] - sub[sel, j, k]) / s
            qr[sel, 1 + i] = 0.25 * s
            qr[sel, 1 + j] = (sub[sel, j, i] + sub[sel, i, j]) / s
            qr[sel, 1 + k] = (sub[sel, k, i] + sub[sel, i, k]) / s
        q[rest] = qr

    return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)


def _sample_texture(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinearly sample (H,W,3) at (N,2) UVs in [0,1]. v=0 is the bottom row."""
    h, w = texture.shape[:2]
    x = np.clip(uv[:, 0], 0.0, 1.0) * (w - 1)
    y = (1.0 - np.clip(uv[:, 1], 0.0, 1.0)) * (h - 1)  # image rows run top-down
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return (
        texture[y0, x0] * (1 - fx) * (1 - fy)
        + texture[y0, x1] * fx * (1 - fy)
        + texture[y1, x0] * (1 - fx) * fy
        + texture[y1, x1] * fx * fy
    )


def mesh_to_splats(
    mesh: Mesh,
    n_points: int = 20_000,
    prior_confidence: float = 0.15,
    scale_factor: float = 1.6,
    thin_ratio: float = 0.1,
    seed: int = 7,
) -> GaussianCloud:
    """Area-weighted surface sampling into flat, surface-aligned Gaussians."""
    rng = np.random.default_rng(seed)
    v, f = mesh.vertices.astype(np.float64), mesh.faces
    tri = v[f]  # (F, 3, 3)
    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    total_area = float(areas.sum())
    if total_area <= 0 or f.shape[0] == 0:
        raise ValueError("mesh has no surface area to sample")
    probs = areas / total_area
    face_idx = rng.choice(f.shape[0], size=n_points, p=probs)

    # uniform barycentric sampling
    r1 = np.sqrt(rng.random(n_points))
    r2 = rng.random(n_points)
    w0, w1, w2 = 1 - r1, r1 * (1 - r2), r1 * r2
    t = tri[face_idx]
    positions = w0[:, None] * t[:, 0] + w1[:, None] * t[:, 1] + w2[:, None] * t[:, 2]

    if mesh.is_textured:
        uvs = mesh.uv[f[face_idx]]  # (n, 3 verts, 2)
        uv = w0[:, None] * uvs[:, 0] + w1[:, None] * uvs[:, 1] + w2[:, None] * uvs[:, 2]
        colors = _sample_texture(mesh.texture, uv)
    else:
        vc = mesh.vertex_colors[f[face_idx]]  # (n, 3 verts, 3 rgb)
        colors = w0[:, None] * vc[:, 0] + w1[:, None] * vc[:, 1] + w2[:, None] * vc[:, 2]

    # Lay each splat flat in its triangle's plane: two tangent axes at the radius
    # that tiles the surface, one collapsed along the normal.
    normals = _face_normals(tri)[face_idx]
    rotations = _matrix_to_quaternion(_tangent_frames(normals))
    radius = scale_factor * np.sqrt(total_area / n_points / np.pi)
    scales = np.tile(
        np.array([radius, radius, radius * thin_ratio], np.float32), (n_points, 1)
    )

    return GaussianCloud.create(
        positions=positions.astype(np.float32),
        colors=np.clip(colors, 0, 1).astype(np.float32),
        scales=scales,
        rotations=rotations.astype(np.float32),
        provenance=int(Provenance.PRIOR),
        confidence=prior_confidence,
    )
