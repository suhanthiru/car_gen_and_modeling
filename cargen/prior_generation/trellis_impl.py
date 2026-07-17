"""Real prior backend: TRELLIS (Microsoft) — local image→3D, native Gaussians.

Strongest local option and the best architectural fit: TRELLIS's SLAT
representation decodes *directly* to 3D Gaussians (opacity, anisotropic
scales, rotations, SH), so `generate_splats` bypasses the mesh→sample
round-trip that every other backend pays. `generate` (mesh) is still provided
for the .glb export path.

INTEGRATION POINT
-----------------
Install (Windows, the painful one — budget a couple of hours):
    git clone --recurse-submodules https://github.com/microsoft/TRELLIS
    pip install spconv-cu120 xformers        # prebuilt wheels; avoids flash-attn
    pip install -e ./TRELLIS                 # builds nvdiffrast + rasterizer
    setx ATTN_BACKEND xformers               # sidestep flash-attn entirely
    setx SPCONV_ALGO native                  # skip per-run algo autotuning
Weights:  huggingface.co/microsoft/TRELLIS-image-large (~2 GB, auto-download)
License:  MIT (code and weights) — the most permissive of the prior backends.
VRAM:     Microsoft recommends 16 GB. Comfortable on the future 12-16 GB box.
          On this 8 GB 4060: plausible ONLY with `low_vram=True` below, which
          keeps the staged submodels on CPU and pages each onto the GPU for its
          turn. Expect ~1 min/asset and some risk of OOM at high
          `slat_sampler_steps`. If it OOMs on the laptop, fall back to
          CARGEN_PRIOR_BACKEND=sf3d; TRELLIS stays the default on the big box.

Verified empirically at Milestone A — do not assume 8 GB works until measured.
"""
from __future__ import annotations

import numpy as np

from cargen.core.splat import GaussianCloud, Provenance
from cargen.prior_generation.canonical import normalize_to_canonical
from cargen.prior_generation.interface import Mesh, PriorGenerator

# 3DGS spherical-harmonics DC band → linear RGB (Y_0^0 = 0.2820947917738781).
_SH_C0 = 0.28209479177387814


class TrellisPriorGenerator(PriorGenerator):
    def __init__(
        self,
        model_name: str = "microsoft/TRELLIS-image-large",
        seed: int = 1,
        low_vram: bool = False,
        sparse_structure_steps: int = 12,
        slat_steps: int = 12,
    ):
        import torch
        from trellis.pipelines import TrellisImageTo3DPipeline  # lazy heavy import

        self._torch = torch
        self._pipeline = TrellisImageTo3DPipeline.from_pretrained(model_name)
        self._low_vram = low_vram
        if not low_vram:
            self._pipeline.cuda()
        self._seed = seed
        self._sparse_structure_steps = sparse_structure_steps
        self._slat_steps = slat_steps

    def _run(self, image_rgb: np.ndarray, mask: np.ndarray | None, formats: list[str]):
        from PIL import Image

        if image_rgb is None:
            raise ValueError("TRELLIS requires an input photo")
        # TRELLIS keys off alpha for the object cutout; reuse our segmentation mask
        # rather than its bundled rembg pass.
        alpha = (
            (np.clip(mask, 0, 1) * 255).astype(np.uint8)
            if mask is not None
            else np.full(image_rgb.shape[:2], 255, np.uint8)
        )
        pil = Image.fromarray(np.dstack([image_rgb, alpha]), mode="RGBA")

        if self._low_vram:
            self._pipeline.cuda()
        try:
            outputs = self._pipeline.run(
                pil,
                seed=self._seed,
                formats=formats,
                preprocess_image=False,  # already masked/cutout by our segmenter
                sparse_structure_sampler_params={"steps": self._sparse_structure_steps},
                slat_sampler_params={"steps": self._slat_steps},
            )
        finally:
            if self._low_vram:
                self._pipeline.cpu()
            self._torch.cuda.empty_cache()
        return outputs

    def generate_splats(
        self,
        image_rgb: np.ndarray | None,
        mask: np.ndarray | None,
        n_points: int = 20_000,
    ) -> GaussianCloud:
        """Native Gaussian path — no mesh round-trip.

        `n_points` is an upper bound here (TRELLIS decides its own count from
        the SLAT grid); we farthest-subsample only if it overshoots.
        """
        outputs = self._run(image_rgb, mask, formats=["gaussian"])
        g = outputs["gaussian"][0]

        with self._torch.no_grad():
            xyz = g.get_xyz.detach().cpu().numpy().astype(np.float32)
            scales = g.get_scaling.detach().cpu().numpy().astype(np.float32)
            rotations = g.get_rotation.detach().cpu().numpy().astype(np.float32)
            opacities = g.get_opacity.detach().cpu().numpy().astype(np.float32).reshape(-1)
            features_dc = g.get_features.detach().cpu().numpy().astype(np.float32)
        colors = np.clip(0.5 + _SH_C0 * features_dc.reshape(len(xyz), -1)[:, :3], 0, 1)

        # The Gaussians' own scales live in the model's units, so they take the
        # same uniform factor the positions do.
        canonical, factor = normalize_to_canonical(xyz)
        scales = scales * factor

        if len(canonical) > n_points:
            rng = np.random.default_rng(self._seed)
            keep = rng.choice(len(canonical), size=n_points, replace=False)
            canonical, scales, rotations = canonical[keep], scales[keep], rotations[keep]
            opacities, colors = opacities[keep], colors[keep]

        return GaussianCloud.create(
            positions=canonical,
            colors=colors,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            provenance=int(Provenance.PRIOR),
            confidence=0.15,
        )

    def generate(self, image_rgb: np.ndarray | None, mask: np.ndarray | None) -> Mesh:
        """Mesh path — used for the .glb export fallback, not the splat pipeline."""
        outputs = self._run(image_rgb, mask, formats=["mesh"])
        m = outputs["mesh"][0]
        vertices, _ = normalize_to_canonical(
            m.vertices.detach().cpu().numpy().astype(np.float32)
        )
        faces = m.faces.detach().cpu().numpy().astype(np.int32)
        return Mesh(vertices, faces, np.full((len(vertices), 3), 0.5, np.float32))
