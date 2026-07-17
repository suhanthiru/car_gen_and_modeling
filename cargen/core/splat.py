"""Gaussian splat cloud: SoA numpy arrays + per-splat fusion metadata.

Every splat carries `provenance` (PRIOR = generative guess, OBSERVED = confirmed
by real imagery), `confidence` in [0, 1], `view_count`, and `last_seen_ts`.
This metadata is what lets real evidence overwrite guesses cheaply while
confirmed regions resist being churned by noise.

APPEARANCE is spherical harmonics: `colors` is the DC band (the flat, view-
independent colour) and `sh_rest` holds bands 1-3. The higher bands are what
make a highlight *slide across the paint* as the camera orbits — with DC alone a
splat has one colour from every direction, i.e. the vehicle is matte by
construction and can never look photoreal. `sh_rest` is zero until a
multi-view optimisation has something real to put there; a single-image prior
has no way to know how a surface changes with viewing angle.

GaussianCloud is immutable: all operations return new instances.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np

#: SH bands 1-3 = 15 coefficients (degree 3 has 16 total; the DC term is `colors`).
SH_REST_COEFFS = 15


class Provenance(IntEnum):
    PRIOR = 0
    OBSERVED = 1


_FIELDS = (
    "positions", "scales", "rotations", "opacities", "colors", "sh_rest",
    "provenance", "confidence", "view_count", "last_seen_ts",
)


@dataclass(frozen=True)
class GaussianCloud:
    positions: np.ndarray     # (N, 3) float32, canonical object frame
    scales: np.ndarray        # (N, 3) float32, linear (not log)
    rotations: np.ndarray     # (N, 4) float32, quaternion wxyz, normalized
    opacities: np.ndarray     # (N,)  float32 in [0, 1]
    colors: np.ndarray        # (N, 3) float32 in [0, 1] (SH DC band)
    sh_rest: np.ndarray       # (N, 15, 3) float32, SH bands 1-3; zero = matte
    provenance: np.ndarray    # (N,)  uint8 (Provenance)
    confidence: np.ndarray    # (N,)  float32 in [0, 1]
    view_count: np.ndarray    # (N,)  int32
    last_seen_ts: np.ndarray  # (N,)  float64 unix seconds

    def __post_init__(self) -> None:
        n = self.positions.shape[0]
        expected = {
            "positions": (n, 3), "scales": (n, 3), "rotations": (n, 4),
            "opacities": (n,), "colors": (n, 3), "sh_rest": (n, SH_REST_COEFFS, 3),
            "provenance": (n,), "confidence": (n,), "view_count": (n,),
            "last_seen_ts": (n,),
        }
        for name, shape in expected.items():
            arr = getattr(self, name)
            if arr.shape != shape:
                raise ValueError(f"{name}: expected shape {shape}, got {arr.shape}")

    @property
    def n(self) -> int:
        return self.positions.shape[0]

    @property
    def is_view_dependent(self) -> bool:
        """Whether any splat's appearance actually changes with viewing angle.

        False for anything a single-image prior produced: it has one photo and
        therefore no evidence about how a surface looks from elsewhere. Only a
        multi-view optimisation can fill these bands, which is why gloss is
        gated on consolidation rather than on a better prior.
        """
        return self.n > 0 and bool(np.any(self.sh_rest))

    @staticmethod
    def empty() -> "GaussianCloud":
        return GaussianCloud.create(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32))

    @staticmethod
    def create(
        positions: np.ndarray,
        colors: np.ndarray,
        scales: np.ndarray | None = None,
        rotations: np.ndarray | None = None,
        opacities: np.ndarray | None = None,
        sh_rest: np.ndarray | None = None,
        provenance: int | np.ndarray = Provenance.PRIOR,
        confidence: float | np.ndarray = 0.1,
        view_count: int | np.ndarray = 0,
        last_seen_ts: float | np.ndarray = 0.0,
        default_scale: float = 0.01,
    ) -> "GaussianCloud":
        """Build a cloud, filling unspecified attributes with sane defaults."""
        positions = np.asarray(positions, np.float32).reshape(-1, 3)
        n = positions.shape[0]
        colors = np.asarray(colors, np.float32).reshape(-1, 3)
        if scales is None:
            scales = np.full((n, 3), default_scale, np.float32)
        if rotations is None:
            rotations = np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))
        if opacities is None:
            opacities = np.full((n,), 0.9, np.float32)
        if sh_rest is None:
            # matte until a multi-view optimisation earns the higher bands
            sh_rest = np.zeros((n, SH_REST_COEFFS, 3), np.float32)

        def _fill(value, dtype):
            arr = np.asarray(value)
            if arr.ndim == 0:
                return np.full((n,), value, dtype)
            return arr.astype(dtype)

        return GaussianCloud(
            positions=positions,
            scales=np.asarray(scales, np.float32).reshape(-1, 3),
            rotations=np.asarray(rotations, np.float32).reshape(-1, 4),
            opacities=np.asarray(opacities, np.float32).reshape(-1),
            colors=colors,
            sh_rest=np.asarray(sh_rest, np.float32).reshape(-1, SH_REST_COEFFS, 3),
            provenance=_fill(provenance, np.uint8),
            confidence=_fill(confidence, np.float32),
            view_count=_fill(view_count, np.int32),
            last_seen_ts=_fill(last_seen_ts, np.float64),
        )

    def select(self, mask_or_indices: np.ndarray) -> "GaussianCloud":
        """New cloud containing only the selected splats."""
        return GaussianCloud(**{f: getattr(self, f)[mask_or_indices] for f in _FIELDS})

    def concat(self, other: "GaussianCloud") -> "GaussianCloud":
        """New cloud with `other`'s splats appended."""
        return GaussianCloud(**{
            f: np.concatenate([getattr(self, f), getattr(other, f)]) for f in _FIELDS
        })

    def with_updates(self, indices: np.ndarray, **updates: np.ndarray) -> "GaussianCloud":
        """New cloud with `updates` (field -> values) written at `indices`."""
        unknown = set(updates) - set(_FIELDS)
        if unknown:
            raise ValueError(f"unknown fields: {unknown}")
        new = {}
        for f in _FIELDS:
            arr = getattr(self, f)
            if f in updates:
                arr = arr.copy()
                arr[indices] = updates[f]
            new[f] = arr
        return GaussianCloud(**new)

    def observed_fraction(self) -> float:
        if self.n == 0:
            return 0.0
        return float(np.mean(self.provenance == Provenance.OBSERVED))

    def stats(self) -> dict:
        return {
            "splats": self.n,
            "observed": int(np.sum(self.provenance == Provenance.OBSERVED)),
            "prior": int(np.sum(self.provenance == Provenance.PRIOR)),
            "observed_fraction": round(self.observed_fraction(), 4),
            "mean_confidence": round(float(np.mean(self.confidence)) if self.n else 0.0, 4),
        }
