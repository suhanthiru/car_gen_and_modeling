"""Server configuration — every knob env-overridable (CARGEN_*).

Moving to the production box should be: install, copy `data/`, set the env vars
for the stronger GPU's backends. Nothing here is hard-coded to this laptop.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Windows reserves these regardless of extension; a car named "con" would break
# the whole storage layer.
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}
_UNSAFE = re.compile(r"[^a-z0-9._-]+")


def sanitize_name(name: str, fallback: str = "vehicle") -> str:
    """User-given car name → a safe directory name.

    Users type "Bob's Civic 🚗"; the filesystem gets "bobs-civic". The original
    string stays in the manifest, so nothing is lost — this only governs the
    folder name.
    """
    slug = _UNSAFE.sub("-", name.strip().lower().replace("'", ""))
    # A leading underscore marks internal folders (_merged); a car must not
    # be able to claim one, and '..' must never escape the storage root.
    slug = slug.strip("-._")
    if not slug or slug in _WINDOWS_RESERVED:
        slug = f"{fallback}-{slug}" if slug else fallback
    return slug[:64]


@dataclass
class Config:
    storage_root: Path = field(
        default_factory=lambda: Path(
            os.environ.get("CARGEN_STORAGE", PROJECT_ROOT / "data" / "vehicles")
        )
    )
    host: str = os.environ.get("CARGEN_HOST", "0.0.0.0")  # LAN-visible, not public
    port: int = int(os.environ.get("CARGEN_PORT", "8000"))

    # Duplicate handling. Default OFF: suspected duplicates are only flagged for
    # human approval. The verifier is unproven, and a wrong auto-merge silently
    # fuses two different cars into one asset — expensive to notice, worse to undo.
    auto_merge: bool = os.environ.get("CARGEN_AUTO_MERGE", "0") == "1"
    merge_threshold: float = float(os.environ.get("CARGEN_MERGE_THRESHOLD", "0.92"))

    # See Pipeline.prior_points: 20k reads as gravel on a car-sized object.
    prior_points: int = int(os.environ.get("CARGEN_PRIOR_POINTS", "120000"))
    max_upload_mb: int = int(os.environ.get("CARGEN_MAX_UPLOAD_MB", "256"))
    # Full-res phone photos are far larger than fusion needs; downscaling keeps
    # the CPU path responsive and matching stable.
    max_image_width: int = int(os.environ.get("CARGEN_MAX_IMAGE_WIDTH", "1280"))

    @property
    def merged_root(self) -> Path:
        """Where merged-away assets are archived — outside the live listing."""
        return self.storage_root / "_merged"

    def vehicle_dir(self, folder: str) -> Path:
        return self.storage_root / folder

    def unique_folder(self, name: str) -> str:
        """Sanitized folder for `name`, suffixed if that folder is taken."""
        base = sanitize_name(name)
        candidate, n = base, 2
        while (self.storage_root / candidate).exists():
            candidate = f"{base}-{n}"
            n += 1
        return candidate


CONFIG = Config()
