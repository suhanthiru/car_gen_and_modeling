"""Vehicle storage: named folders on disk, one per vehicle.

Layout (`data/vehicles/<car-name>/`):
    manifest.json          identity, aliases, observation log, stats
    cloud.npz              the splats
    embeddings.npy         re-ID vectors, one per observation
    observations/          the raw uploads, exactly as captured
    exports/model.ply      standard 3DGS — opens in SuperSplat/Blender
    exports/model.splat    what the web viewer streams

The folder is named by what the user called the car, so the files are findable
without the app. `manifest.json` holds the immutable UUID, so renames and merges
never break references.
"""
from __future__ import annotations

import json
from pathlib import Path

from cargen.core.asset import VehicleAsset
from cargen.export.exporter import export_all
from server.config import Config


class VehicleStore:
    def __init__(self, config: Config):
        self.config = config
        self.config.storage_root.mkdir(parents=True, exist_ok=True)

    # -- lookup --------------------------------------------------------------

    def folders(self) -> list[str]:
        """Live vehicles. Folders starting with '_' are internal (e.g. _merged)."""
        return sorted(
            p.name
            for p in self.config.storage_root.iterdir()
            if p.is_dir() and not p.name.startswith("_") and VehicleAsset.is_asset_dir(p)
        )

    def exists(self, folder: str) -> bool:
        return VehicleAsset.is_asset_dir(self.config.vehicle_dir(folder))

    def load(self, folder: str) -> VehicleAsset:
        return VehicleAsset.load(self.config.vehicle_dir(folder))

    def load_all(self) -> dict[str, VehicleAsset]:
        return {f: self.load(f) for f in self.folders()}

    def find_by_name(self, name: str) -> str | None:
        """Folder whose asset answers to `name` — display name or alias."""
        for folder in self.folders():
            manifest = self.manifest(folder)
            if not manifest:
                continue
            names = {manifest.get("name", ""), *manifest.get("aliases", [])}
            if name in names or name.lower() in {n.lower() for n in names if n}:
                return folder
        return None

    def resolve(self, key: str) -> str | None:
        """Accept a folder name, display name, alias, or vehicle_id."""
        if self.exists(key):
            return key
        by_name = self.find_by_name(key)
        if by_name:
            return by_name
        for folder in self.folders():
            manifest = self.manifest(folder)
            if manifest and manifest.get("vehicle_id") == key:
                return folder
        return None

    def manifest(self, folder: str) -> dict | None:
        path = self.config.vehicle_dir(folder) / "manifest.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

    # -- write ---------------------------------------------------------------

    def create_folder(self, name: str) -> str:
        folder = self.config.unique_folder(name)
        (self.config.vehicle_dir(folder) / "observations").mkdir(
            parents=True, exist_ok=True
        )
        return folder

    def save(self, folder: str, asset: VehicleAsset, export: bool = True) -> dict:
        directory = self.config.vehicle_dir(folder)
        asset.save(directory)
        if not export:
            return {}
        return export_all(asset.cloud, directory / "exports")

    def save_upload(self, folder: str, filename: str, payload: bytes) -> Path:
        """Keep the raw capture: re-fusable later with better models."""
        directory = self.config.vehicle_dir(folder) / "observations"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        stem, suffix, n = path.stem, path.suffix, 2
        while path.exists():
            path = directory / f"{stem}-{n}{suffix}"
            n += 1
        path.write_bytes(payload)
        return path

    def export_path(self, folder: str, filename: str) -> Path:
        return self.config.vehicle_dir(folder) / "exports" / filename

    def summary(self, folder: str) -> dict:
        manifest = self.manifest(folder) or {}
        stats = manifest.get("stats", {})
        return {
            "folder": folder,
            "name": manifest.get("name", folder),
            "vehicle_id": manifest.get("vehicle_id", ""),
            "aliases": manifest.get("aliases", []),
            "observations": len(manifest.get("observations", [])),
            "updated_ts": manifest.get("updated_ts", 0),
            "splats": stats.get("splats", 0),
            "observed_fraction": stats.get("observed_fraction", 0.0),
            "mean_confidence": stats.get("mean_confidence", 0.0),
        }
