"""Stub prior: a procedural sedan mesh (also the synthetic demo's ground truth).

Deterministic, CPU-only, distinctly colored panels so fusion results are
visually interpretable in the viewer.
"""
from __future__ import annotations

import numpy as np

from cargen.prior_generation.interface import Mesh, PriorGenerator

DEFAULT_COLORS = {
    "paint": (0.55, 0.58, 0.62),   # noncommittal silver — the "factory default" guess
    "glass": (0.15, 0.20, 0.26),
    "wheel": (0.10, 0.10, 0.11),
    "trim": (0.22, 0.22, 0.24),
}


def _box(center, size, color) -> Mesh:
    cx, cy, cz = center
    sx, sy, sz = (s / 2 for s in size)
    v = np.array(
        [
            [cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz],
            [cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz],
            [cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz],
            [cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz],
        ],
        np.float32,
    )
    f = np.array(
        [
            [0, 2, 1], [0, 3, 2],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # -y
            [2, 3, 7], [2, 7, 6],  # +y
            [1, 2, 6], [1, 6, 5],  # +x
            [3, 0, 4], [3, 4, 7],  # -x
        ],
        np.int32,
    )
    colors = np.tile(np.asarray(color, np.float32), (8, 1))
    return Mesh(v, f, colors)


def _wheel(center, radius, width, color, segments: int = 14) -> Mesh:
    """Cylinder with axis along y (car's lateral axis)."""
    cx, cy, cz = center
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.stack([np.cos(angles) * radius, np.zeros(segments), np.sin(angles) * radius], 1)
    left = ring + [cx, cy - width / 2, cz]
    right = ring + [cx, cy + width / 2, cz]
    centers = np.array([[cx, cy - width / 2, cz], [cx, cy + width / 2, cz]], np.float32)
    v = np.concatenate([left, right, centers]).astype(np.float32)
    li, ri = np.arange(segments), np.arange(segments) + segments
    lc, rc = 2 * segments, 2 * segments + 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces += [[li[i], ri[i], ri[j]], [li[i], ri[j], li[j]]]   # tread
        faces += [[lc, li[j], li[i]], [rc, ri[i], ri[j]]]          # caps
    colors = np.tile(np.asarray(color, np.float32), (v.shape[0], 1))
    return Mesh(v, np.array(faces, np.int32), colors)


def build_sedan_mesh(colors: dict | None = None) -> Mesh:
    """Sedan in the canonical frame: +x forward, +z up, ground z=0, length 2.0."""
    c = {**DEFAULT_COLORS, **(colors or {})}
    paint = np.asarray(c["paint"], np.float32)
    parts = [
        _box((0.0, 0.0, 0.31), (2.00, 0.82, 0.26), paint),                 # chassis
        _box((0.62, 0.0, 0.42), (0.72, 0.80, 0.06), paint * 1.12),         # hood
        _box((-0.72, 0.0, 0.42), (0.52, 0.80, 0.06), paint * 0.95),        # trunk lid
        _box((-0.08, 0.0, 0.56), (0.94, 0.70, 0.22), c["glass"]),          # cabin/glass
        _box((-0.08, 0.0, 0.685), (0.86, 0.62, 0.035), paint * 0.85),      # roof
        _box((1.01, 0.0, 0.26), (0.06, 0.78, 0.16), c["trim"]),            # front bumper
        _box((-1.01, 0.0, 0.26), (0.06, 0.78, 0.16), c["trim"]),           # rear bumper
        _wheel((0.62, -0.44, 0.16), 0.16, 0.10, c["wheel"]),
        _wheel((0.62, 0.44, 0.16), 0.16, 0.10, c["wheel"]),
        _wheel((-0.62, -0.44, 0.16), 0.16, 0.10, c["wheel"]),
        _wheel((-0.62, 0.44, 0.16), 0.16, 0.10, c["wheel"]),
    ]
    mesh = parts[0]
    for part in parts[1:]:
        mesh = mesh.concat(part)
    return Mesh(mesh.vertices, mesh.faces, np.clip(mesh.vertex_colors, 0, 1))


class StubPriorGenerator(PriorGenerator):
    """Returns the procedural sedan; paint tinted from the photo's mean color
    when a photo is provided (a crude stand-in for the generative model
    conditioning its guess on the input view)."""

    def generate(self, image_rgb: np.ndarray | None, mask: np.ndarray | None) -> Mesh:
        colors = None
        if image_rgb is not None:
            m = mask > 0.5 if mask is not None else np.ones(image_rgb.shape[:2], bool)
            if m.any():
                mean = image_rgb[m].mean(axis=0) / 255.0
                colors = {"paint": tuple(np.clip(mean, 0.05, 0.95))}
        return build_sedan_mesh(colors)
