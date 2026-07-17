"""Image→3D prior generation interface.

Contract: `generate` takes the (masked) vehicle photo and returns a complete
textured triangle mesh in the canonical object frame — the "factory default"
guess including regions the photo never saw. The pipeline converts it to a
Gaussian cloud tagged provenance=PRIOR.

Canonical object frame: vehicle centered at origin, +x forward, +y left,
+z up, ground plane at z=0, overall length ≈ 2.0 units (normalized — metric
scale is deliberately not tracked; Sim(3) registration absorbs it).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Mesh:
    """A prior mesh, optionally UV-textured.

    `uv` + `texture` are optional but matter enormously when present: a real
    backend bakes a 1024x1024 albedo map (~1M texels), and collapsing that to one
    colour per vertex — which is all `vertex_colors` can hold — throws away ~80x
    the appearance detail. `vertex_colors` stays required as the fallback for
    backends (and the stub) that have nothing better.
    """

    vertices: np.ndarray       # (V, 3) float32
    faces: np.ndarray          # (F, 3) int32
    vertex_colors: np.ndarray  # (V, 3) float32 in [0, 1]
    uv: np.ndarray | None = None       # (V, 2) float32, [0,1], origin bottom-left
    texture: np.ndarray | None = None  # (H, W, 3) float32 in [0, 1]

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must be (V,3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must be (F,3)")
        if self.vertex_colors.shape != self.vertices.shape:
            raise ValueError("vertex_colors must match vertices shape")
        if self.uv is not None:
            if self.uv.ndim != 2 or self.uv.shape[1] != 2:
                raise ValueError("uv must be (V,2)")
            if self.uv.shape[0] != self.vertices.shape[0]:
                raise ValueError("uv must have one entry per vertex")
        if self.texture is not None:
            if self.texture.ndim != 3 or self.texture.shape[2] != 3:
                raise ValueError("texture must be (H,W,3)")
            if self.uv is None:
                raise ValueError("texture without uv cannot be sampled")

    @property
    def is_textured(self) -> bool:
        return self.uv is not None and self.texture is not None

    def concat(self, other: "Mesh") -> "Mesh":
        """Join two meshes. UV/texture are dropped — two meshes have two atlases,
        and merging them would need a repack we have no reason to do here."""
        offset = self.vertices.shape[0]
        return Mesh(
            vertices=np.concatenate([self.vertices, other.vertices]),
            faces=np.concatenate([self.faces, other.faces + offset]),
            vertex_colors=np.concatenate([self.vertex_colors, other.vertex_colors]),
        )


class PriorGenerator(ABC):
    @abstractmethod
    def generate(self, image_rgb: np.ndarray | None, mask: np.ndarray | None) -> Mesh:
        """Vehicle photo (RGB uint8) + mask → complete canonical-frame mesh."""

    def generate_splats(
        self,
        image_rgb: np.ndarray | None,
        mask: np.ndarray | None,
        n_points: int = 20_000,
    ):
        """Photo → GaussianCloud (provenance=PRIOR), the pipeline's real entry point.

        Default path: generate a mesh, then area-weighted surface-sample it.
        Backends whose model emits Gaussians natively (TRELLIS, LGM) override
        this to skip the lossy mesh round-trip and keep the model's own
        opacities/anisotropic scales/rotations.
        """
        from cargen.prior_generation.mesh_to_splats import mesh_to_splats

        return mesh_to_splats(self.generate(image_rgb, mask), n_points=n_points)

    def export_raw_mesh(self, path) -> bool:
        """Write the backend's own mesh, texture and all, for reference.

        Our splats are a resampling of this, so it is the ceiling they can reach:
        if the .glb looks right and the splats don't, the conversion is at fault,
        not the model. Also serves as the lightweight/mobile .glb asset.

        Default no-op — backends with no mesh of their own (or no exporter)
        simply return False.
        """
        return False
