"""Prior backend adapter: Tripo3D cloud API (image→3D).

INTEGRATION POINT (ships untested — no API key configured yet, user-approved)
------------------------------------------------------------------------------
Setup:     create an account at platform.tripo3d.ai, generate an API key,
           set env var TRIPO_API_KEY. Free starter credits, then pay-per-task.
Privacy:   the vehicle photo is uploaded to Tripo's cloud.
Quality:   highest of the available prior backends; no local VRAM used.
Docs:      https://platform.tripo3d.ai/docs

Flow: upload image → create image_to_model task → poll → download GLB →
bake to vertex colors → canonical frame.
"""
from __future__ import annotations

import io
import os
import time

import numpy as np

from cargen.prior_generation.canonical import normalize_to_canonical
from cargen.prior_generation.interface import Mesh, PriorGenerator

_API = "https://api.tripo3d.ai/v2/openapi"


class TripoPriorGenerator(PriorGenerator):
    def __init__(self, api_key: str | None = None, poll_interval_s: float = 3.0,
                 timeout_s: float = 600.0):
        self._key = api_key or os.environ.get("TRIPO_API_KEY")
        if not self._key:
            raise RuntimeError("TRIPO_API_KEY not set — see module docstring")
        self._poll = poll_interval_s
        self._timeout = timeout_s

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._key}"}

    def generate(self, image_rgb: np.ndarray | None, mask: np.ndarray | None) -> Mesh:
        import httpx
        import trimesh
        from PIL import Image

        if image_rgb is None:
            raise ValueError("Tripo requires an input photo")
        buf = io.BytesIO()
        Image.fromarray(image_rgb).save(buf, format="PNG")
        buf.seek(0)

        with httpx.Client(timeout=60) as client:
            up = client.post(
                f"{_API}/upload", headers=self._headers(),
                files={"file": ("vehicle.png", buf, "image/png")},
            )
            up.raise_for_status()
            token = up.json()["data"]["image_token"]

            task = client.post(
                f"{_API}/task", headers=self._headers(),
                json={"type": "image_to_model",
                      "file": {"type": "png", "file_token": token}},
            )
            task.raise_for_status()
            task_id = task.json()["data"]["task_id"]

            deadline = time.time() + self._timeout
            model_url = None
            while time.time() < deadline:
                status = client.get(f"{_API}/task/{task_id}", headers=self._headers())
                status.raise_for_status()
                data = status.json()["data"]
                if data["status"] == "success":
                    model_url = data["output"]["model"]
                    break
                if data["status"] in ("failed", "cancelled", "banned"):
                    raise RuntimeError(f"Tripo task {task_id} ended: {data['status']}")
                time.sleep(self._poll)
            if model_url is None:
                raise TimeoutError(f"Tripo task {task_id} did not finish in {self._timeout}s")

            glb = client.get(model_url).content

        tm = trimesh.load(io.BytesIO(glb), file_type="glb", force="mesh")
        colors = (
            np.asarray(tm.visual.to_color().vertex_colors, np.float32)[:, :3] / 255.0
            if tm.visual is not None
            else np.full((len(tm.vertices), 3), 0.5, np.float32)
        )
        vertices, _ = normalize_to_canonical(np.asarray(tm.vertices, np.float32))
        return Mesh(vertices, np.asarray(tm.faces, np.int32), colors)
