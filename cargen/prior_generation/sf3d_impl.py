"""Real prior backend: Stable-Fast-3D (Stability AI) — local image→3D.

INTEGRATION POINT
-----------------
Install (verified on Windows 11 / RTX 4060 / py3.11):

    git clone https://github.com/Stability-AI/stable-fast-3d third_party/stable-fast-3d
    # SF3D's own pins — note numpy==1.26.4 and rembg==2.0.57 are HARD:
    #   rembg>=2.0.76 requires numpy>=2.3, which SF3D rejects. Keep rembg at
    #   2.0.57 (SF3D's own pin) or the two backends cannot coexist.
    pip install numpy==1.26.4 einops==0.7.0 jaxtyping==0.2.31 omegaconf==2.3.0 \
        transformers==4.42.3 open_clip_torch==2.24.0 trimesh==4.4.1 \
        huggingface-hub==0.23.4 rembg==2.0.57 pynanoinstantmeshes==0.0.3 gpytoolbox==0.2.0

    # The two C++ extensions. They are imported at module scope by sf3d/system.py,
    # so SF3D cannot run without them. Needs VS Build Tools (C++ workload), and
    # MUST build inside the MSVC dev shell or cl.exe is not on PATH:
    cmd /c 'call "...\VC\Auxiliary\Build\vcvars64.bat" && set USE_CUDA=0 && ^
            python -m pip install ./uv_unwrapper ./texture_baker --no-build-isolation'

    # USE_CUDA=0 builds texture_baker as a plain CppExtension -> no CUDA Toolkit
    # needed (only ~4 GB of Build Tools instead of ~10 GB with nvcc). Baking then
    # runs on CPU, which is fine: it happens once per photo.
    # --no-build-isolation is required: setup.py imports torch, and pip's isolated
    # build env has no torch.

Weights:   huggingface.co/stabilityai/stable-fast-3d — GATED. Accept the license
           on the model page, then `huggingface-cli login`. ~2 GB, auto-downloads.
License:   Stability AI Community License — free for personal / <$1M revenue.
VRAM:      ~6-7 GB peak against this laptop's 8 GB (7.4 GB free) — genuinely
           tight. Run alone; the pipeline's load->run->unload discipline and the
           per-vehicle queue keep it that way.
Output:    textured mesh; we resolve the texture to vertex colors for the
           canonical Mesh contract, then normalize into the canonical frame.

SF3D ships no setup.py, so it is not pip-installable; `_ensure_importable` puts
its checkout on sys.path. Override the location with CARGEN_SF3D_PATH.

Alternates behind the same interface: TRELLIS (native gaussians, MIT,
trellis_impl.py), Tripo3D API (tripo_api.py), custom (custom_slot.py).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from cargen.prior_generation.canonical import normalize_to_canonical
from cargen.prior_generation.interface import Mesh, PriorGenerator

_DEFAULT_SF3D_PATH = (
    Path(__file__).resolve().parents[2] / "third_party" / "stable-fast-3d"
)


def _install_cpu_baker_bridge(model) -> bool:
    """Let a CPU-built texture_baker serve SF3D's CUDA tensors.

    `USE_CUDA=0` builds texture_baker's kernels for CPU only (which is what
    spares us the ~3 GB CUDA Toolkit), but SF3D runs on the GPU and hands the
    baker CUDA tensors — so the dispatcher raises NotImplementedError for the
    CUDA backend. Only `rasterize` and `interpolate` call the custom op
    (`get_mask` is plain torch), so bridging those two is enough: move to CPU,
    bake, hand the result back on the caller's device.

    Costs two transfers plus CPU rasterization per photo — a few seconds, once
    per capture. Returns True if the bridge was installed; if texture_baker was
    built WITH CUDA the native path is already fine and we leave it alone.
    """
    import torch
    from texture_baker import TextureBaker

    probe = torch.zeros((3, 2), device="cuda")
    faces = torch.zeros((1, 3), dtype=torch.int32, device="cuda")
    try:
        TextureBaker().rasterize(probe, faces, 8)
        return False  # CUDA kernels present — nothing to bridge
    except NotImplementedError:
        pass
    except Exception:
        return False  # some other failure; don't paper over it

    class _CpuBakerBridge(TextureBaker):
        def rasterize(self, uv, face_indices, bake_resolution):
            device = uv.device
            return super().rasterize(
                uv.cpu(), face_indices.cpu(), bake_resolution
            ).to(device)

        def interpolate(self, attr, rast, face_indices):
            device = attr.device
            return super().interpolate(
                attr.cpu(), rast.cpu(), face_indices.cpu()
            ).to(device)

    model.baker = _CpuBakerBridge()
    return True


def _extract_texture(tm) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Pull SF3D's baked albedo atlas + UVs off the trimesh visual.

    This is the payload `texture_baker` exists to produce — roughly a million
    texels of panel gaps, badges and lights. Losing it here (which is what
    `visual.to_color()` does) is the difference between a car and a blob, so the
    fallback is deliberately last-resort.
    """
    visual = getattr(tm, "visual", None)
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    image = getattr(material, "baseColorTexture", None) if material is not None else None
    if uv is None or image is None:
        return None, None
    texture = np.asarray(image.convert("RGB"), np.float32) / 255.0
    return np.asarray(uv, np.float32), texture


def _vertex_colors(tm) -> np.ndarray:
    """Per-vertex colours: the fallback when there is no atlas, and the value
    Mesh always carries so untextured backends keep working."""
    n = len(tm.vertices)
    visual = getattr(tm, "visual", None)
    if visual is None:
        return np.full((n, 3), 0.5, np.float32)
    try:
        return np.asarray(visual.to_color().vertex_colors, np.float32)[:, :3] / 255.0
    except Exception:
        # some trimesh visuals can't be colour-converted; grey is honest here
        return np.full((n, 3), 0.5, np.float32)


def _ensure_importable() -> Path:
    """Put the SF3D checkout on sys.path.

    Kept inside this adapter rather than set as a global PYTHONPATH: the backend
    owns its own integration mess, so nothing else in cargen (or the tests, which
    never touch SF3D) has to know it exists.
    """
    root = Path(os.environ.get("CARGEN_SF3D_PATH", _DEFAULT_SF3D_PATH))
    if not (root / "sf3d").is_dir():
        raise FileNotFoundError(
            f"SF3D checkout not found at {root}. Clone it there, or set "
            f"CARGEN_SF3D_PATH. See this module's docstring for the full recipe."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


class SF3DPriorGenerator(PriorGenerator):
    def __init__(
        self,
        device: str = "cuda",
        texture_resolution: int = 1024,
        # SF3D's own run.py default. The model is trained exclusively on square,
        # centred, object-filling images — see _frame_subject.
        foreground_ratio: float = 0.85,
    ):
        _ensure_importable()
        import torch
        from sf3d.system import SF3D  # lazy heavy import (pulls the C++ extensions)

        self._torch = torch
        self._model = (
            SF3D.from_pretrained(
                "stabilityai/stable-fast-3d",
                config_name="config.yaml",
                weight_name="model.safetensors",
            )
            .eval()
            .to(device)
        )
        self._device = device
        self._texture_resolution = texture_resolution
        self._foreground_ratio = foreground_ratio
        self.cpu_baker = _install_cpu_baker_bridge(self._model)
        self.last_mesh = None  # most recent raw trimesh, for export_raw_mesh
        self.last_input = None  # the framed RGBA actually fed to the model

    def _frame_subject(self, rgba):
        """Crop square, centre the vehicle, scale it to fill the frame.

        NOT optional, and skipping it is why our first results looked nothing
        like SF3D's published demos. The model is trained exclusively on
        object-centric images: square, subject centred, filling ~85% of the
        frame. A raw phone photo of a parked car is none of those — the vehicle
        sits off-centre at ~20% of a 4:3 frame — so almost no image tokens land
        on it and the model's implicit framing assumption is violated. It
        answers anyway, with a blob.

        We call SF3D's own `resize_foreground` rather than reimplementing the
        crop, so our framing is identical to theirs by construction.
        """
        _ensure_importable()
        from sf3d.utils import resize_foreground

        return resize_foreground(rgba, self._foreground_ratio)

    def export_raw_mesh(self, path) -> bool:
        """SF3D's own textured mesh as .glb — untouched by our resampling.

        NOTE this is in SF3D's native (view-aligned) frame, not the canonical one:
        it is a reference for appearance, not a drop-in for the splat asset.
        """
        if self.last_mesh is None:
            return False
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.last_mesh.export(str(target))
        return True

    def generate(self, image_rgb: np.ndarray | None, mask: np.ndarray | None) -> Mesh:
        import trimesh
        from PIL import Image

        if image_rgb is None:
            raise ValueError("SF3D requires an input photo")
        rgba = np.dstack(
            [image_rgb, (np.clip(mask, 0, 1) * 255).astype(np.uint8)]
            if mask is not None
            else [image_rgb, np.full(image_rgb.shape[:2], 255, np.uint8)]
        )
        framed = self._frame_subject(Image.fromarray(rgba, mode="RGBA"))
        self.last_input = framed
        with self._torch.no_grad():
            trimesh_obj, _ = self._model.run_image(
                framed, bake_resolution=self._texture_resolution
            )
        tm: trimesh.Trimesh = trimesh_obj
        self.last_mesh = tm  # kept so the pipeline can export the raw .glb

        uv, texture = _extract_texture(tm)
        colors = _vertex_colors(tm)
        # use_pca: SF3D's output frame follows the input camera, so the vehicle
        # arrives rotated by wherever the photo was taken from and tilted by the
        # camera's elevation. PCA puts it back on its own axes. Rotating the
        # positions leaves UVs untouched (they index the atlas, not space).
        vertices, _ = normalize_to_canonical(
            np.asarray(tm.vertices, np.float32), use_pca=True
        )
        # free VRAM for the next pipeline stage
        self._torch.cuda.empty_cache()
        return Mesh(
            vertices=vertices,
            faces=np.asarray(tm.faces, np.int32),
            vertex_colors=colors,
            uv=uv,
            texture=texture,
        )
