"""Deferred duplicate detection and asset merging.

THE PROBLEM THIS EXISTS FOR
---------------------------
It is the *expected* failure, not an edge case: someone scans a car's rear and
creates asset B, not knowing asset A is the same car scanned from the front.
Re-ID could not match them because their observed regions did not overlap — no
shared evidence existed to compare. Only later, once a walk-around bridges the
two, does the match become visible.

So detection cannot be a one-shot check at ingest. It re-runs whenever an asset
gains coverage, comparing against assets it could not previously be compared to.

MERGE POLICY
------------
The asset with more confirmed coverage wins and stays primary. The duplicate's
OBSERVED splats are fused in through the same confidence arbitration everything
else uses; its PRIOR splats are discarded outright (the primary's guesses are no
worse, and two sets of hallucinations do not average into truth). Observation
histories are unioned; the duplicate's name is kept as an alias so the user's
own label for it never disappears.

`auto_merge` defaults OFF: detection only *flags*, and a human approves. A wrong
auto-merge silently fuses two different cars into one asset — cheap to prevent,
expensive to notice, worse to undo.

PHASE 1 uses appearance embeddings (histogram now, DINOv2 at Milestone A) as the
similarity signal. That is the cheap pre-filter tier only. The render-based
verifier — sweep-render the asset at the query's viewpoint and compare same-angle
in feature space, masked to OBSERVED regions — is what makes this trustworthy
enough to enable auto_merge, and it lands in phase 2 (see reid/interface.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cargen.core.asset import VehicleAsset
from cargen.core.splat import GaussianCloud, Provenance
from cargen.reid.interface import Embedder
from server.config import Config
from server.events import Event, EventLog
from server.store import VehicleStore


@dataclass(frozen=True)
class DuplicateCandidate:
    folder: str
    other_folder: str
    score: float


def similarity(a: VehicleAsset, b: VehicleAsset) -> float:
    """Appearance similarity between two assets' mean embeddings."""
    ea, eb = a.mean_embedding(), b.mean_embedding()
    if ea is None or eb is None:
        return 0.0
    return Embedder.similarity(ea, eb)


def find_duplicates(
    store: VehicleStore, folder: str, threshold: float
) -> list[DuplicateCandidate]:
    """Assets that look like the same vehicle as `folder`, best first."""
    if not store.exists(folder):
        return []
    asset = store.load(folder)
    out = []
    for other_folder in store.folders():
        if other_folder == folder:
            continue
        score = similarity(asset, store.load(other_folder))
        if score >= threshold:
            out.append(DuplicateCandidate(folder, other_folder, score))
    return sorted(out, key=lambda c: -c.score)


def merge_clouds(primary: GaussianCloud, duplicate: GaussianCloud) -> GaussianCloud:
    """Fold the duplicate's real evidence into the primary.

    Only OBSERVED splats cross over: the duplicate's guesses carry no
    information the primary lacks. Kept deliberately additive — the localized
    optimizer reconciles overlaps at Milestone B; averaging positions here would
    smear two views of the same panel into a blur.
    """
    keep = duplicate.provenance == Provenance.OBSERVED
    if not keep.any():
        return primary
    return primary.concat(duplicate.select(keep))


def merge_assets(primary: VehicleAsset, duplicate: VehicleAsset) -> VehicleAsset:
    """Merge `duplicate` into `primary`, returning the updated primary.

    FRAME ASSUMPTION — read before enabling auto_merge. Both assets are assumed
    to sit in the same canonical frame. That is only true because
    `canonicalize_orientation` puts each prior on its own PCA axes: the
    image-to-3D backends emit a *view-aligned* frame, so without it a car
    scanned from the front and the same car scanned from the side arrive rotated
    ~90° apart and this merge fuses two interpenetrating shells.

    PCA fixes the gross orientation but is not exact: it cannot distinguish
    front from back (a car is roughly symmetric end-to-end), and its axes wobble
    with reconstruction noise. So a merge can still be off by 180° in yaw. The
    real fix is the render-based verifier's argmax azimuth feeding a Sim(3)
    alignment (phase 2) — until then, this is the main reason auto_merge
    defaults to off.
    """
    primary.cloud = merge_clouds(primary.cloud, duplicate.cloud)
    primary.observations.extend(duplicate.observations)
    if duplicate.embeddings is not None and duplicate.embeddings.size:
        primary.embeddings = (
            duplicate.embeddings
            if primary.embeddings is None or primary.embeddings.size == 0
            else np.concatenate([primary.embeddings, duplicate.embeddings])
        )
    for alias in (duplicate.name, *duplicate.aliases):
        if alias and alias != primary.name and alias not in primary.aliases:
            primary.aliases.append(alias)
    primary.updated_ts = max(primary.updated_ts, duplicate.updated_ts)
    return primary


def _coverage(asset: VehicleAsset) -> int:
    """Confirmed splats — how much of this asset is real rather than guessed."""
    return int(np.sum(asset.cloud.provenance == Provenance.OBSERVED))


def pick_primary(store: VehicleStore, a: str, b: str) -> tuple[str, str]:
    """(primary, duplicate) — the asset whose folder and name survive the merge.

    **The older asset always wins.** No evidence rides on this choice: the
    duplicate's OBSERVED splats are folded in either way (see merge_clouds), so
    the only thing at stake is which name and folder survive — and the older
    asset is the established identity the user named and everything else
    already refers to.

    Coverage deliberately does NOT decide this. It reads as the "fairer" rule,
    but confirmed-splat counts between two scans of the same car differ by a
    handful of splats of pure luck, which would let a throwaway capture
    ("mystery-car") hijack the folder of an established vehicle and demote the
    user's own name to an alias. Identity should not hinge on a coin flip.
    """
    asset_a, asset_b = store.load(a), store.load(b)
    return (a, b) if asset_a.created_ts <= asset_b.created_ts else (b, a)


class MergeManager:
    def __init__(self, store: VehicleStore, events: EventLog, config: Config):
        self.store = store
        self.events = events
        self.config = config

    def scan(self, folder: str) -> list[Event]:
        """Re-check `folder` against every other asset. Merge or flag."""
        emitted = []
        for candidate in find_duplicates(
            self.store, folder, self.config.merge_threshold
        ):
            if not (self.store.exists(candidate.folder)
                    and self.store.exists(candidate.other_folder)):
                continue  # already merged away by an earlier candidate
            primary, duplicate = pick_primary(
                self.store, candidate.folder, candidate.other_folder
            )
            if self.config.auto_merge:
                emitted.append(self.apply(primary, duplicate, candidate.score))
            else:
                emitted.append(
                    self.events.append(
                        Event(
                            kind="merge_pending",
                            vehicle=primary,
                            message=(
                                f"'{duplicate}' looks like the same vehicle as "
                                f"'{primary}' (score {candidate.score:.3f}). "
                                f"Approve to merge."
                            ),
                            data={
                                "primary": primary,
                                "duplicate": duplicate,
                                "score": candidate.score,
                            },
                        )
                    )
                )
        return emitted

    def apply(
        self, primary: str, duplicate: str, score: float, pending_id: str = ""
    ) -> Event:
        """Perform the merge and retire the duplicate's folder."""
        primary_asset = self.store.load(primary)
        duplicate_asset = self.store.load(duplicate)
        before = primary_asset.cloud.n

        merged = merge_assets(primary_asset, duplicate_asset)
        self.store.save(primary, merged)
        self._retire(duplicate)

        return self.events.append(
            Event(
                kind="merge",
                vehicle=primary,
                message=(
                    f"'{duplicate}' was identified as '{primary}' and merged "
                    f"({before} → {merged.cloud.n} splats)."
                ),
                data={
                    "primary": primary, "duplicate": duplicate, "score": score,
                    "splats_before": before, "splats_after": merged.cloud.n,
                    "pending_id": pending_id,
                },
            )
        )

    def reject(self, pending_id: str, primary: str, duplicate: str) -> Event:
        return self.events.append(
            Event(
                kind="merge_rejected",
                vehicle=primary,
                message=f"'{duplicate}' confirmed as a different vehicle from '{primary}'.",
                data={"primary": primary, "duplicate": duplicate,
                      "pending_id": pending_id},
            )
        )

    def _retire(self, folder: str) -> None:
        """Archive the merged-away folder instead of deleting it.

        Moved under `_merged/` (outside the listing, so it can never be served
        or re-matched) rather than removed: the user named this folder and its
        raw captures are irreplaceable, so a wrong merge must be recoverable.
        """
        directory = self.config.vehicle_dir(folder)
        if not directory.exists():
            return
        archive = self.config.merged_root
        archive.mkdir(parents=True, exist_ok=True)
        target, n = archive / folder, 2
        while target.exists():
            target = archive / f"{folder}-{n}"
            n += 1
        directory.rename(target)
