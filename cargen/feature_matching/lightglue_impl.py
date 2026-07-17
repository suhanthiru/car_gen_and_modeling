"""Real wide-baseline matcher: LightGlue + ALIKED/SuperPoint (Milestone B).

INTEGRATION POINT
-----------------
Install:   pip install lightglue @ git+https://github.com/cvg/LightGlue.git
           (torch CUDA build first; weights auto-download, ~50 MB)
License:   Apache-2.0 (LightGlue + ALIKED). SuperPoint's weights are
           non-commercial — ALIKED is the default here to avoid that.
VRAM:      <2 GB — fits alongside other stages on the 8 GB laptop.

Why this over ORB for registration: a new photo of a car from a different angle
is exactly the wide-baseline, low-texture, specular case where ORB's ratio test
collapses. LightGlue is far more robust, and its per-match scores feed the
registration confidence gate directly.
"""
from __future__ import annotations

import numpy as np

from cargen.feature_matching.interface import FeatureMatcher, MatchResult


class LightGlueMatcher(FeatureMatcher):
    def __init__(self, extractor: str = "aliked", max_keypoints: int = 2048,
                 device: str = "cuda", min_score: float = 0.1):
        import torch
        from lightglue import ALIKED, LightGlue, SuperPoint

        self._torch = torch
        self._device = device
        self._min_score = min_score
        if extractor == "aliked":
            self._extractor = ALIKED(max_num_keypoints=max_keypoints).eval().to(device)
        elif extractor == "superpoint":
            self._extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        else:
            raise ValueError(f"unknown extractor: {extractor}")
        self._matcher = LightGlue(features=extractor).eval().to(device)

    def _prep(self, image: np.ndarray, mask: np.ndarray | None):
        """RGB uint8 (H,W,3) → torch (1,3,H,W) float in [0,1], background zeroed."""
        img = image.astype(np.float32) / 255.0
        if mask is not None:
            img = img * np.clip(mask, 0, 1)[..., None]
        tensor = self._torch.from_numpy(img).permute(2, 0, 1)[None]
        return tensor.to(self._device)

    def match(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        mask_a: np.ndarray | None = None,
        mask_b: np.ndarray | None = None,
    ) -> MatchResult:
        from lightglue.utils import rbd

        with self._torch.no_grad():
            feats_a = self._extractor.extract(self._prep(image_a, mask_a))
            feats_b = self._extractor.extract(self._prep(image_b, mask_b))
            matches01 = self._matcher({"image0": feats_a, "image1": feats_b})
        feats_a, feats_b, matches01 = (rbd(x) for x in (feats_a, feats_b, matches01))

        idx = matches01["matches"].cpu().numpy()
        if idx.size == 0:
            return MatchResult.empty()
        scores = matches01["scores"].detach().cpu().numpy().astype(np.float32)
        kp_a = feats_a["keypoints"].cpu().numpy()
        kp_b = feats_b["keypoints"].cpu().numpy()

        keep = scores >= self._min_score
        if not keep.any():
            return MatchResult.empty()
        return MatchResult(
            kp_a[idx[keep, 0]].astype(np.float32),
            kp_b[idx[keep, 1]].astype(np.float32),
            np.clip(scores[keep], 0.0, 1.0),
        )
