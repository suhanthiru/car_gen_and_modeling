"""Shared fixtures. Everything CPU-only and seeded — no ML weights, no GPU."""
from __future__ import annotations

import numpy as np
import pytest

from cargen.core.camera import Intrinsics
from cargen.fusion_engine.point_renderer import PointRenderer
from cargen.prior_generation.mesh_to_splats import mesh_to_splats
from cargen.prior_generation.stub import build_sedan_mesh
from demo.synthetic import build_prior_cloud, build_truth_cloud

# Small enough that the whole suite runs in seconds; large enough that
# projection, occlusion, and dirty-region logic all get exercised.
WIDTH, HEIGHT = 128, 96


@pytest.fixture(scope="session")
def intrinsics() -> Intrinsics:
    return Intrinsics.simple(WIDTH, HEIGHT)


@pytest.fixture(scope="session")
def renderer() -> PointRenderer:
    return PointRenderer()


@pytest.fixture(scope="session")
def truth_cloud(intrinsics, renderer):
    return build_truth_cloud(intrinsics, renderer, n_points=2500)


@pytest.fixture(scope="session")
def prior_cloud(truth_cloud):
    return build_prior_cloud(truth_cloud)


@pytest.fixture
def small_cloud():
    """A tiny hand-made cloud for exercising splat mechanics directly."""
    rng = np.random.default_rng(0)
    return mesh_to_splats(build_sedan_mesh(), n_points=200, seed=1)
