"""Photometric residual with exposure compensation — the "surprise" map.

WHY THIS IS NOT A PLAIN RGB DIFF
--------------------------------
Naive differencing conflates *change* with *lighting*. Photograph a car on an
overcast day after capturing it in sun and the residual lights up everywhere:
the whole vehicle reads as "dirty", and confirmed regions get churned by what is
really just an exposure shift. Glossy paint makes it worse — view-dependent
reflections are legitimate appearance change that RGB diffing misreads as
geometry change.

Two defenses live here:
  1. `compensate_exposure` fits a per-channel affine (gain + bias) transform on
     the pixels the model *already agrees about*, using only high-confidence
     regions as the reference. Global lighting shifts are absorbed before the
     diff; genuine local differences survive it.
  2. `residual_map` blurs before differencing, so sub-pixel misregistration
     doesn't masquerade as surprise.

The remaining specular false-positives are filtered temporally by the engine
(reflections move with viewpoint across frames; dents do not).

Upgrade path: compute residuals in DINOv2 feature space rather than RGB —
near lighting-invariant, still structure-sensitive.
"""
from __future__ import annotations

import cv2
import numpy as np


def compensate_exposure(
    rendered: np.ndarray,
    observed: np.ndarray,
    reference_mask: np.ndarray,
    max_gain: float = 3.0,
    min_reference_px: int = 256,
) -> np.ndarray:
    """Affine-match `observed` to `rendered`'s photometry over `reference_mask`.

    `reference_mask` MUST cover only pixels the model has *repeatedly* confirmed
    — not merely pixels it has seen once. This is the sharpest edge in the whole
    engine: fit on pixels where the model is still wrong and the fit will
    happily map reality onto the model's mistake, erasing the very disagreement
    fusion exists to detect. When in doubt, compensate nothing: a missed
    exposure correction costs some extra churn, a wrong one destroys evidence.

    Gain/bias are fit on interquartile statistics rather than mean/variance, so
    a minority of genuinely-changed pixels inside the reference region (a fresh
    dent on an otherwise-confirmed panel) cannot drag the estimate.
    """
    adjusted = observed.astype(np.float32).copy()
    if reference_mask.sum() < min_reference_px:
        return adjusted  # too little confirmed area to fit anything trustworthy

    for channel in range(3):
        obs = observed[..., channel][reference_mask].astype(np.float64)
        ref = rendered[..., channel][reference_mask].astype(np.float64)
        obs_q1, obs_med, obs_q3 = np.percentile(obs, [25, 50, 75])
        ref_q1, ref_med, ref_q3 = np.percentile(ref, [25, 50, 75])
        obs_spread = obs_q3 - obs_q1
        if obs_spread < 1e-3:
            # flat reference region: gain is unidentifiable, so correct
            # brightness only and leave contrast alone
            adjusted[..., channel] = np.clip(
                adjusted[..., channel] + (ref_med - obs_med), 0.0, 1.0
            )
            continue
        gain = float(np.clip((ref_q3 - ref_q1) / obs_spread, 1.0 / max_gain, max_gain))
        bias = float(ref_med - gain * obs_med)
        adjusted[..., channel] = np.clip(adjusted[..., channel] * gain + bias, 0.0, 1.0)
    return adjusted


def residual_map(
    rendered: np.ndarray, observed: np.ndarray, blur_sigma: float = 1.5
) -> np.ndarray:
    """Per-pixel L1 photometric disagreement in [0, 1], blurred before diffing."""
    if blur_sigma > 0:
        ksize = int(blur_sigma * 4) | 1  # odd kernel
        rendered = cv2.GaussianBlur(rendered, (ksize, ksize), blur_sigma)
        observed = cv2.GaussianBlur(observed, (ksize, ksize), blur_sigma)
    return np.abs(rendered - observed).mean(axis=2).astype(np.float32)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow a boolean pixel mask — the blending ring around dirty regions."""
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
