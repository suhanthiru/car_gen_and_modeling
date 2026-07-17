"""Backend registry — the one place real models get swapped in.

Every ML-heavy stage resolves through here by name, so orchestration never
imports a model directly and stubs/real backends are a config flip:

    CARGEN_SEGMENTER=rembg CARGEN_PRIOR_BACKEND=trellis CARGEN_RENDERER=gsplat

Imports are lazy and per-name: selecting `stub` never imports torch, which is
what keeps the CPU test suite fast and installable with no ML dependencies.
"""
from __future__ import annotations

import os

from cargen.feature_matching.interface import FeatureMatcher
from cargen.fusion_engine.renderer import SplatRenderer
from cargen.prior_generation.interface import PriorGenerator
from cargen.reid.interface import Embedder
from cargen.segmentation.interface import Segmenter


def build_segmenter(name: str | None = None) -> Segmenter:
    # rembg is the default: the stub's rectangle mask calls the background
    # "vehicle", which tints the prior's paint toward the sky/road colour and
    # feeds the densifier junk. Tests and the demo pass an explicit segmenter,
    # so this default only governs the server — which should do the real thing.
    # Set CARGEN_SEGMENTER=stub to run with no ML weights at all.
    name = (name or os.environ.get("CARGEN_SEGMENTER", "rembg")).lower()
    if name == "stub":
        from cargen.segmentation.stub import StubSegmenter

        return StubSegmenter()
    if name == "rembg":
        from cargen.segmentation.rembg_impl import RembgSegmenter

        return RembgSegmenter(os.environ.get("CARGEN_REMBG_MODEL", "isnet-general-use"))
    raise ValueError(f"unknown segmenter: {name!r} (stub|rembg)")


def build_prior_generator(name: str | None = None) -> PriorGenerator:
    # sf3d is the default for the same reason rembg is (see build_segmenter):
    # tests and the demo pass an explicit generator, so this only governs the
    # server, which should produce real geometry rather than a procedural box.
    # Set CARGEN_PRIOR_BACKEND=stub to run with no ML weights at all.
    name = (name or os.environ.get("CARGEN_PRIOR_BACKEND", "sf3d")).lower()
    if name == "stub":
        from cargen.prior_generation.stub import StubPriorGenerator

        return StubPriorGenerator()
    if name == "trellis":
        from cargen.prior_generation.trellis_impl import TrellisPriorGenerator

        # low_vram pages the staged submodels on/off the GPU — required on the
        # 8 GB laptop, unnecessary on the 12-16 GB box.
        return TrellisPriorGenerator(
            low_vram=os.environ.get("CARGEN_TRELLIS_LOW_VRAM", "0") == "1"
        )
    if name == "sf3d":
        from cargen.prior_generation.sf3d_impl import SF3DPriorGenerator

        return SF3DPriorGenerator()
    if name == "tripo":
        from cargen.prior_generation.tripo_api import TripoPriorGenerator

        return TripoPriorGenerator()
    if name == "custom":
        from cargen.prior_generation.custom_slot import CustomPriorGenerator

        target = os.environ.get("CARGEN_CUSTOM_PRIOR")
        if not target:
            raise ValueError("CARGEN_CUSTOM_PRIOR must be set for the custom backend")
        return CustomPriorGenerator(target)
    raise ValueError(f"unknown prior backend: {name!r} (stub|trellis|sf3d|tripo|custom)")


def build_matcher(name: str | None = None) -> FeatureMatcher:
    name = (name or os.environ.get("CARGEN_MATCHER", "orb")).lower()
    if name == "orb":
        from cargen.feature_matching.orb import OrbMatcher

        return OrbMatcher()
    if name == "lightglue":
        from cargen.feature_matching.lightglue_impl import LightGlueMatcher

        return LightGlueMatcher(os.environ.get("CARGEN_LIGHTGLUE_EXTRACTOR", "aliked"))
    raise ValueError(f"unknown matcher: {name!r} (orb|lightglue)")


def build_renderer(name: str | None = None) -> SplatRenderer:
    name = (name or os.environ.get("CARGEN_RENDERER", "point")).lower()
    if name == "point":
        from cargen.fusion_engine.point_renderer import PointRenderer

        return PointRenderer()
    if name == "gsplat":
        from cargen.fusion_engine.gsplat_renderer import GsplatRenderer

        return GsplatRenderer()
    raise ValueError(f"unknown renderer: {name!r} (point|gsplat)")


def build_embedder(name: str | None = None) -> Embedder:
    name = (name or os.environ.get("CARGEN_EMBEDDER", "histogram")).lower()
    if name == "histogram":
        from cargen.reid.histogram import HistogramEmbedder

        return HistogramEmbedder()
    if name == "dino":
        from cargen.reid.dino_embed import DinoEmbedder

        return DinoEmbedder()
    raise ValueError(f"unknown embedder: {name!r} (histogram|dino)")
