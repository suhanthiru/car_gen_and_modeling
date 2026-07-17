"""ORB + brute-force Hamming matching (OpenCV) — the cheap real matcher.

Good enough for video frame-to-frame tracking, where consecutive frames have
small baselines and high overlap. Wide-baseline matching (a new photo against
the observation history) is where ORB gets brittle on cars — glossy, textureless
panels and bilateral symmetry — so that path should prefer LightGlue
(lightglue_impl.py) once Milestone B is installed.
"""
from __future__ import annotations

import cv2
import numpy as np

from cargen.feature_matching.interface import FeatureMatcher, MatchResult


class OrbMatcher(FeatureMatcher):
    def __init__(
        self,
        n_features: int = 4000,
        ratio: float = 0.75,
        cross_check_symmetry: bool = True,
    ):
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._ratio = ratio
        self._symmetry = cross_check_symmetry

    def _detect(self, image: np.ndarray, mask: np.ndarray | None):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        cv_mask = None
        if mask is not None:
            cv_mask = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        return self._orb.detectAndCompute(gray, cv_mask)

    def _ratio_matches(self, desc_a: np.ndarray, desc_b: np.ndarray):
        """Lowe's ratio test → (query_idx, train_idx, confidence) triples.

        Ambiguous matches are the main source of the left/right and front/rear
        confusions that wreck car pose estimates, so this filter matters more
        here than in general scenes.
        """
        out = []
        for pair in self._bf.knnMatch(desc_a, desc_b, k=2):
            if len(pair) < 2:
                continue
            best, second = pair
            if best.distance >= self._ratio * second.distance:
                continue
            confidence = 1.0 - (best.distance / max(second.distance, 1e-6)) * self._ratio
            out.append((best.queryIdx, best.trainIdx, confidence))
        return out

    def match(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        mask_a: np.ndarray | None = None,
        mask_b: np.ndarray | None = None,
    ) -> MatchResult:
        kp_a, desc_a = self._detect(image_a, mask_a)
        kp_b, desc_b = self._detect(image_b, mask_b)
        if desc_a is None or desc_b is None or len(kp_a) < 2 or len(kp_b) < 2:
            return MatchResult.empty()

        forward = self._ratio_matches(desc_a, desc_b)
        if not forward:
            return MatchResult.empty()

        if self._symmetry:
            # b→a must agree with a→b: kills one-sided matches onto repeated
            # structure (both wheels, both door handles).
            backward = {b: a for a, b, _ in self._ratio_matches(desc_b, desc_a)}
            forward = [(a, b, c) for a, b, c in forward if backward.get(b) == a]
            if not forward:
                return MatchResult.empty()

        idx_a = np.fromiter((m[0] for m in forward), int, len(forward))
        idx_b = np.fromiter((m[1] for m in forward), int, len(forward))
        conf = np.fromiter((m[2] for m in forward), np.float32, len(forward))
        pts_a = np.asarray([kp_a[i].pt for i in idx_a], np.float32)
        pts_b = np.asarray([kp_b[i].pt for i in idx_b], np.float32)
        return MatchResult(pts_a, pts_b, np.clip(conf, 0.0, 1.0))
