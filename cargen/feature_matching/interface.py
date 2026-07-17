"""Feature matching interface.

Contract: `match` takes two RGB images (+ optional masks restricting attention
to vehicle pixels) and returns corresponding pixel coordinates plus per-match
confidence. Downstream pose estimation treats `confidence` as a weight and the
match count as its primary registration-quality signal.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatchResult:
    points_a: np.ndarray   # (M, 2) float32 pixel coords in image A
    points_b: np.ndarray   # (M, 2) float32 pixel coords in image B
    confidence: np.ndarray  # (M,) float32 in [0, 1]

    def __post_init__(self) -> None:
        if self.points_a.shape != self.points_b.shape:
            raise ValueError("points_a and points_b must have the same shape")
        if self.points_a.shape[0] != self.confidence.shape[0]:
            raise ValueError("confidence must have one entry per match")

    @property
    def count(self) -> int:
        return self.points_a.shape[0]

    @staticmethod
    def empty() -> "MatchResult":
        return MatchResult(
            np.zeros((0, 2), np.float32),
            np.zeros((0, 2), np.float32),
            np.zeros((0,), np.float32),
        )

    def top_k(self, k: int) -> "MatchResult":
        if self.count <= k:
            return self
        keep = np.argsort(-self.confidence)[:k]
        return MatchResult(self.points_a[keep], self.points_b[keep], self.confidence[keep])


class FeatureMatcher(ABC):
    @abstractmethod
    def match(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        mask_a: np.ndarray | None = None,
        mask_b: np.ndarray | None = None,
    ) -> MatchResult:
        """RGB uint8 images (+ optional [0,1] masks) → pixel correspondences."""
