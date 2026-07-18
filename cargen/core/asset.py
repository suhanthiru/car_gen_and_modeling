"""VehicleAsset: the persistent per-vehicle aggregate.

One asset per physical vehicle, accumulating observations (photos/video
sessions from any device) over arbitrarily long time spans. Persisted as a
directory: manifest.json (identity, aliases, observation log, stats) +
cloud.npz (splat arrays) + embeddings.npy (re-ID vectors, one per observation)
+ frames/ + poses.json (accepted, registered frame imagery and resolved
camera poses — the durable input `consolidate.py`'s joint multi-view
optimization needs; everything else in this file is a cache of *derived*
state, but frames/poses are themselves source data worth keeping).

The numpy payloads (GaussianCloud) are immutable; the asset is the one mutable
aggregate root that swaps them.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cargen.core.camera import CameraPose, Intrinsics
from cargen.core.splat import _FIELDS, SH_REST_COEFFS, GaussianCloud

MANIFEST_NAME = "manifest.json"
CLOUD_NAME = "cloud.npz"
EMBEDDINGS_NAME = "embeddings.npy"
FRAMES_DIRNAME = "frames"
POSES_NAME = "poses.json"


@dataclass(frozen=True)
class FrameRecord:
    """One accepted, registered observation, ready for joint optimization.

    `index` matches the frame's position in `poses.json`/its filename under
    `frames/` — the stable key `consolidate.py` uses to report per-frame
    metrics (e.g. `psnr_by_frame`).
    """

    index: int
    image_rgb: np.ndarray       # (H, W, 3) uint8
    mask: np.ndarray | None     # (H, W) float32 [0, 1], or None
    pose: CameraPose
    intrinsics: Intrinsics
    evidence_weight: float = 1.0


@dataclass
class VehicleAsset:
    name: str
    vehicle_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    aliases: list[str] = field(default_factory=list)
    created_ts: float = field(default_factory=time.time)
    updated_ts: float = field(default_factory=time.time)
    cloud: GaussianCloud = field(default_factory=GaussianCloud.empty)
    observations: list[dict] = field(default_factory=list)
    embeddings: np.ndarray | None = None  # (n_obs, dim) float32, order matches observations
    frames: list[FrameRecord] = field(default_factory=list)

    def add_frame(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray | None,
        pose: CameraPose,
        intrinsics: Intrinsics,
        evidence_weight: float = 1.0,
    ) -> int:
        """Record an accepted, registered observation for later joint optimization.

        Returns the frame's index (its key in `poses.json` / filename under
        `frames/`).
        """
        index = len(self.frames)
        self.frames.append(
            FrameRecord(
                index=index,
                image_rgb=np.asarray(image_rgb, np.uint8),
                mask=None if mask is None else np.asarray(mask, np.float32),
                pose=pose,
                intrinsics=intrinsics,
                evidence_weight=float(evidence_weight),
            )
        )
        self.updated_ts = time.time()
        return index

    def load_frames(self) -> list[FrameRecord]:
        """All persisted frames, in index order (already resident after `load()`)."""
        return list(self.frames)

    def add_observation(self, meta: dict, embedding: np.ndarray | None = None) -> None:
        """Record an observation's metadata (kind, device, ts, file, report)."""
        self.observations.append(dict(meta))
        if embedding is not None:
            emb = np.asarray(embedding, np.float32).reshape(1, -1)
            if self.embeddings is None or self.embeddings.size == 0:
                self.embeddings = emb
            else:
                self.embeddings = np.concatenate([self.embeddings, emb])
        self.updated_ts = time.time()

    def mean_embedding(self) -> np.ndarray | None:
        if self.embeddings is None or self.embeddings.size == 0:
            return None
        mean = self.embeddings.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm > 0 else mean

    def stats(self) -> dict:
        return {
            "name": self.name,
            "vehicle_id": self.vehicle_id,
            "aliases": list(self.aliases),
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
            "observations": len(self.observations),
            "frames": len(self.frames),
            **self.cloud.stats(),
        }

    # -- persistence ---------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "vehicle_id": self.vehicle_id,
            "name": self.name,
            "aliases": self.aliases,
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
            "observations": self.observations,
            "stats": self.cloud.stats(),
        }
        (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
        np.savez_compressed(
            directory / CLOUD_NAME, **{f: getattr(self.cloud, f) for f in _FIELDS}
        )
        if self.embeddings is not None:
            np.save(directory / EMBEDDINGS_NAME, self.embeddings)
        self._save_frames(directory)
        return directory

    def _save_frames(self, directory: Path) -> None:
        """Write frames/<idx>.png (+ <idx>_mask.png) and poses.json.

        Only called from `save()`. Frame images are written unconditionally
        (even for an asset with zero frames) so re-saving an asset never
        leaves a stale frame from a prior save lying around.
        """
        import cv2

        frames_dir = directory / FRAMES_DIRNAME
        frames_dir.mkdir(parents=True, exist_ok=True)
        for old in frames_dir.glob("*"):
            old.unlink()

        poses: dict[str, dict] = {}
        for rec in self.frames:
            key = f"{rec.index:04d}"
            bgr = cv2.cvtColor(rec.image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frames_dir / f"{key}.png"), bgr)
            has_mask = rec.mask is not None
            if has_mask:
                mask_u8 = np.clip(rec.mask * 255.0, 0, 255).astype(np.uint8)
                cv2.imwrite(str(frames_dir / f"{key}_mask.png"), mask_u8)
            poses[key] = {
                "pose": rec.pose.to_dict(),
                "intrinsics": rec.intrinsics.to_dict(),
                "evidence_weight": rec.evidence_weight,
                "has_mask": has_mask,
            }
        (directory / POSES_NAME).write_text(json.dumps(poses, indent=2))

    @staticmethod
    def _load_frames(directory: Path) -> list[FrameRecord]:
        """Read frames/ + poses.json back, in ascending index order.

        Assets saved before this schema addition have neither `frames/` nor
        `poses.json` — those are real captures too, they simply predate frame
        persistence, so they load with an empty frame list rather than an
        error. Mirrors `load_cloud_fields`'s "missing sh_rest -> zeros"
        migration reasoning: "missing frames/ -> empty list".
        """
        frames_dir = directory / FRAMES_DIRNAME
        poses_path = directory / POSES_NAME
        if not frames_dir.exists() or not poses_path.exists():
            return []

        import cv2

        poses = json.loads(poses_path.read_text())
        records = []
        for key in sorted(poses, key=lambda k: int(k)):
            entry = poses[key]
            bgr = cv2.imread(str(frames_dir / f"{key}.png"), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"frame image missing or unreadable: {key}.png")
            image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask = None
            if entry.get("has_mask"):
                mask_path = frames_dir / f"{key}_mask.png"
                if mask_path.exists():
                    mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    mask = mask_u8.astype(np.float32) / 255.0
            records.append(
                FrameRecord(
                    index=int(key),
                    image_rgb=image_rgb,
                    mask=mask,
                    pose=CameraPose.from_dict(entry["pose"]),
                    intrinsics=Intrinsics.from_dict(entry["intrinsics"]),
                    evidence_weight=entry.get("evidence_weight", 1.0),
                )
            )
        return records

    @staticmethod
    def load_cloud_fields(data) -> dict:
        """Read splat arrays from a .npz, filling fields older files predate.

        Assets written before SH bands 1-3 existed have no `sh_rest`. Those are
        real captures the user cannot re-take, so they must keep loading — and
        zero is the honest value: a cloud from that era genuinely had no
        view-dependent appearance. This is a migration, not a fallback.
        """
        fields = {}
        for f in _FIELDS:
            if f in data:
                fields[f] = data[f]
            elif f == "sh_rest":
                n = data["positions"].shape[0]
                fields[f] = np.zeros((n, SH_REST_COEFFS, 3), np.float32)
            else:
                raise KeyError(f"cloud.npz is missing required field {f!r}")
        return fields

    @staticmethod
    def load(directory: str | Path) -> "VehicleAsset":
        directory = Path(directory)
        manifest = json.loads((directory / MANIFEST_NAME).read_text())
        with np.load(directory / CLOUD_NAME) as data:
            cloud = GaussianCloud(**VehicleAsset.load_cloud_fields(data))
        emb_path = directory / EMBEDDINGS_NAME
        embeddings = np.load(emb_path) if emb_path.exists() else None
        return VehicleAsset(
            name=manifest["name"],
            vehicle_id=manifest["vehicle_id"],
            aliases=manifest.get("aliases", []),
            created_ts=manifest["created_ts"],
            updated_ts=manifest["updated_ts"],
            cloud=cloud,
            observations=manifest.get("observations", []),
            embeddings=embeddings,
            frames=VehicleAsset._load_frames(directory),
        )

    @staticmethod
    def is_asset_dir(directory: str | Path) -> bool:
        directory = Path(directory)
        return (directory / MANIFEST_NAME).exists() and (directory / CLOUD_NAME).exists()
