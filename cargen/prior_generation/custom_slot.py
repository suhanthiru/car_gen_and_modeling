"""Custom prior backend slot: plug in your own local model without touching
pipeline code.

INTEGRATION POINT
-----------------
Write a module exposing a callable:

    def generate(image_rgb: np.ndarray | None, mask: np.ndarray | None) -> Mesh: ...

returning a `cargen.prior_generation.interface.Mesh` in the canonical frame
(+x forward, +z up, ground z=0, length ≈ 2.0 — see interface.py). Then set:

    CARGEN_PRIOR_BACKEND=custom
    CARGEN_CUSTOM_PRIOR=your_package.your_module:generate

Useful for e.g. LGM / TripoSR / a future fine-tuned vehicle-specific model on
the stronger production box (12-16 GB GPU class).
"""
from __future__ import annotations

import importlib

import numpy as np

from cargen.prior_generation.interface import Mesh, PriorGenerator


class CustomPriorGenerator(PriorGenerator):
    def __init__(self, target: str):
        module_name, _, attr = target.partition(":")
        if not module_name or not attr:
            raise ValueError("custom prior target must look like 'package.module:callable'")
        self._fn = getattr(importlib.import_module(module_name), attr)

    def generate(self, image_rgb: np.ndarray | None, mask: np.ndarray | None) -> Mesh:
        mesh = self._fn(image_rgb, mask)
        if not isinstance(mesh, Mesh):
            raise TypeError("custom prior callable must return cargen Mesh")
        return mesh
