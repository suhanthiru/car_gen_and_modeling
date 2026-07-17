"""VehicleAsset: the persistent per-vehicle aggregate.

One asset per physical vehicle, accumulating observations (photos/video
sessions from any device) over arbitrarily long time spans. Persisted as a
directory: manifest.json (identity, aliases, observation log, stats) +
cloud.npz (splat arrays) + embeddings.npy (re-ID vectors, one per observation).

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

from cargen.core.splat import _FIELDS, SH_REST_COEFFS, GaussianCloud

MANIFEST_NAME = "manifest.json"
CLOUD_NAME = "cloud.npz"
EMBEDDINGS_NAME = "embeddings.npy"


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
        return directory

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
        )

    @staticmethod
    def is_asset_dir(directory: str | Path) -> bool:
        directory = Path(directory)
        return (directory / MANIFEST_NAME).exists() and (directory / CLOUD_NAME).exists()
