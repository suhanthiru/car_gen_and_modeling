"""Camera intrinsics, world→camera poses, and Sim(3) similarity transforms.

Sim(3) (rotation + translation + scale, 7-DoF) is used for registration because
every capture session arrives at its own arbitrary monocular scale; plain SE(3)
cannot absorb that.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], np.float64
        )

    @staticmethod
    def simple(width: int, height: int, fov_deg: float = 55.0) -> "Intrinsics":
        """Pinhole guess from image size — the default for phone photos w/o EXIF."""
        f = 0.5 * width / np.tan(np.radians(fov_deg) / 2)
        return Intrinsics(fx=f, fy=f, cx=width / 2, cy=height / 2, width=width, height=height)

    def scaled(self, factor: float) -> "Intrinsics":
        return Intrinsics(
            fx=self.fx * factor, fy=self.fy * factor,
            cx=self.cx * factor, cy=self.cy * factor,
            width=int(round(self.width * factor)), height=int(round(self.height * factor)),
        )

    def to_dict(self) -> dict:
        return {
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "width": self.width, "height": self.height,
        }

    @staticmethod
    def from_dict(d: dict) -> "Intrinsics":
        return Intrinsics(
            fx=d["fx"], fy=d["fy"], cx=d["cx"], cy=d["cy"],
            width=d["width"], height=d["height"],
        )


@dataclass(frozen=True)
class CameraPose:
    """World→camera rigid transform: x_cam = R @ x_world + t."""

    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)

    @property
    def camera_center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @staticmethod
    def identity() -> "CameraPose":
        return CameraPose(np.eye(3), np.zeros(3))

    @staticmethod
    def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> "CameraPose":
        """Camera at `eye` looking toward `target` (OpenCV convention: +z forward,
        +x right, +y down)."""
        eye = np.asarray(eye, np.float64)
        forward = np.asarray(target, np.float64) - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(up, np.float64))
        right = right / np.linalg.norm(right)
        down = np.cross(forward, right)
        R = np.stack([right, down, forward])  # rows: camera axes in world coords
        return CameraPose(R=R, t=-R @ eye)

    def transform(self, points: np.ndarray) -> np.ndarray:
        """World points (N,3) → camera frame."""
        return points @ self.R.T + self.t

    def project(self, points: np.ndarray, intr: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
        """World points (N,3) → pixel coords (N,2) and camera-frame depths (N,).

        Points behind the camera get depth <= 0; callers must mask on depth.
        """
        cam = self.transform(points)
        z = cam[:, 2]
        safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
        u = intr.fx * cam[:, 0] / safe_z + intr.cx
        v = intr.fy * cam[:, 1] / safe_z + intr.cy
        return np.stack([u, v], axis=1), z

    def to_dict(self) -> dict:
        return {"R": self.R.tolist(), "t": self.t.tolist()}

    @staticmethod
    def from_dict(d: dict) -> "CameraPose":
        return CameraPose(R=np.array(d["R"], np.float64), t=np.array(d["t"], np.float64))


@dataclass(frozen=True)
class Sim3:
    """Similarity transform: y = s * R @ x + t."""

    s: float
    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)

    @staticmethod
    def identity() -> "Sim3":
        return Sim3(1.0, np.eye(3), np.zeros(3))

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.s * (points @ self.R.T) + self.t

    def compose(self, other: "Sim3") -> "Sim3":
        """Returns T such that T(x) = self(other(x))."""
        return Sim3(
            s=self.s * other.s,
            R=self.R @ other.R,
            t=self.s * (self.R @ other.t) + self.t,
        )

    def inverse(self) -> "Sim3":
        R_inv = self.R.T
        s_inv = 1.0 / self.s
        return Sim3(s=s_inv, R=R_inv, t=-s_inv * (R_inv @ self.t))


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Sim3:
    """Least-squares Sim(3) aligning src → dst point sets (N,3), Umeyama 1991."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("umeyama needs matching point sets with >= 3 points")
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = dst_c.T @ src_c / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scale:
        var_src = (src_c ** 2).sum() / src.shape[0]
        s = float(np.trace(np.diag(D) @ S) / var_src)
    else:
        s = 1.0
    t = mu_dst - s * R @ mu_src
    return Sim3(s=s, R=R, t=t)
