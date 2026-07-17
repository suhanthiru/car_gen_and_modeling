"""Frame sampling by camera-motion magnitude, not fixed interval.

A 30fps walk-around is mostly redundant: standing still produces dozens of
near-identical frames that cost fusion time and add no evidence, while a quick
pan produces the large viewpoint changes that actually matter. Fixed-interval
sampling gets both wrong.

So: accumulate optical-flow magnitude between candidate frames and emit one only
once the camera has moved enough. Near-duplicates are skipped; fast motion is
sampled densely. `max_frames` caps the work per clip.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class SampledFrame:
    index: int          # index in the source video
    image: np.ndarray   # RGB uint8
    motion: float       # accumulated flow magnitude (px) since the previous sample
    timestamp: float    # seconds into the clip


def iter_video_frames(path: str, max_read: int | None = None) -> Iterator[tuple[int, np.ndarray, float]]:
    """Yield (index, RGB frame, timestamp_seconds) from a video file."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise IOError(f"cannot open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        index = 0
        while max_read is None or index < max_read:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            yield index, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), index / fps
            index += 1
    finally:
        capture.release()


class FrameSampler:
    """Emits frames once accumulated camera motion exceeds `motion_threshold`."""

    def __init__(
        self,
        motion_threshold: float = 12.0,
        max_frames: int = 60,
        flow_downscale: int = 8,
        min_gap: int = 2,
    ):
        self._motion_threshold = motion_threshold
        self._max_frames = max_frames
        self._flow_downscale = flow_downscale
        self._min_gap = min_gap

    def _small_gray(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        h, w = gray.shape[:2]
        return cv2.resize(
            gray,
            (max(w // self._flow_downscale, 16), max(h // self._flow_downscale, 16)),
            interpolation=cv2.INTER_AREA,
        )

    def motion_between(self, image_a: np.ndarray, image_b: np.ndarray) -> float:
        """Mean dense optical-flow magnitude, in full-resolution pixels."""
        flow = cv2.calcOpticalFlowFarneback(
            self._small_gray(image_a), self._small_gray(image_b),
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        return float(np.mean(np.linalg.norm(flow, axis=2))) * self._flow_downscale

    def sample(self, frames: Iterator[tuple[int, np.ndarray, float]]) -> list[SampledFrame]:
        """Pick the frames worth fusing from a frame stream."""
        sampled: list[SampledFrame] = []
        previous: np.ndarray | None = None
        accumulated = 0.0
        gap = 0

        for index, image, timestamp in frames:
            if previous is None:
                # first frame is always worth having: it anchors the clip
                sampled.append(SampledFrame(index, image, 0.0, timestamp))
                previous = image
                continue

            gap += 1
            accumulated += self.motion_between(previous, image)
            previous = image
            if accumulated >= self._motion_threshold and gap >= self._min_gap:
                sampled.append(SampledFrame(index, image, accumulated, timestamp))
                accumulated = 0.0
                gap = 0
                if len(sampled) >= self._max_frames:
                    break
        return sampled

    def sample_video(self, path: str, max_read: int | None = None) -> list[SampledFrame]:
        return self.sample(iter_video_frames(path, max_read=max_read))
