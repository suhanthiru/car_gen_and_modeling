"""Bilateral mirror-fill and material-band smoothing — the cheap symbolic
geometric priors: cars are bilaterally symmetric, and have known glass/panel
material bands. Both passes only ever touch PRIOR splats."""
from __future__ import annotations

import numpy as np
import pytest

from cargen.core.splat import GaussianCloud, Provenance
from cargen.fusion_engine.symmetry import (
    BODY_BAND,
    GLASS_BAND,
    _mirror_quaternion,
    mirror_confirmed,
    smooth_material_bands,
)


def _identity_quats(n: int) -> np.ndarray:
    return np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))


class TestMirrorQuaternion:
    def test_round_trips_to_identity(self):
        rng = np.random.default_rng(0)
        q = rng.normal(size=(50, 4)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        twice = _mirror_quaternion(_mirror_quaternion(q))
        assert np.allclose(twice, q, atol=1e-6)

    def test_stays_a_unit_quaternion(self):
        rng = np.random.default_rng(1)
        q = rng.normal(size=(50, 4)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        mirrored = _mirror_quaternion(q)
        norms = np.linalg.norm(mirrored, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_reflects_a_known_rotation_matrix(self):
        """Build a known rotation from a quaternion, reflect it via matrix
        conjugation R'=MRM (M=diag(1,-1,1)) directly, and confirm
        _mirror_quaternion's closed form produces the same matrix."""
        def quat_to_matrix(q):
            w, x, y, z = q
            return np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ])

        rng = np.random.default_rng(2)
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = quat_to_matrix(q)
        M = np.diag([1.0, -1.0, 1.0])
        expected = M @ R @ M

        q_mirrored = _mirror_quaternion(q[None, :].astype(np.float32))[0]
        actual = quat_to_matrix(q_mirrored)
        assert np.allclose(actual, expected, atol=1e-5)


class TestMirrorConfirmed:
    def _asymmetric_cloud(self, rng, n=400):
        """A cloud with OBSERVED splats only on the +y (right) side and PRIOR
        splats mirrored on both sides, so the -y side is pure guess."""
        pos = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
        pos[:, 1] = np.abs(pos[:, 1])  # start everyone on the +y side
        # mirror every point onto -y too, so both sides have geometry to match
        pos = np.concatenate([pos, pos * np.array([1, -1, 1], np.float32)])
        n2 = pos.shape[0]
        colors = np.full((n2, 3), 0.5, np.float32)
        provenance = np.full(n2, Provenance.PRIOR, np.uint8)
        # confirm the +y half with a distinct color, leave -y as the guess
        right = pos[:, 1] > 0
        provenance[right] = Provenance.OBSERVED
        colors[right] = [0.9, 0.1, 0.1]  # red = confirmed
        cloud = GaussianCloud.create(
            positions=pos, colors=colors,
            rotations=_identity_quats(n2),
            provenance=provenance,
            confidence=np.where(provenance == Provenance.OBSERVED, 0.8, 0.15).astype(np.float32),
        )
        return cloud, right

    def test_mirrors_confirmed_color_onto_the_guessed_side(self):
        rng = np.random.default_rng(3)
        cloud, right = self._asymmetric_cloud(rng)
        out, n_mirrored = mirror_confirmed(cloud, radius=0.2)

        assert n_mirrored > 0
        left_prior = (~right) & (cloud.provenance == Provenance.PRIOR)
        # at least some left-side guesses should now look red (mirrored), not
        # the original flat 0.5 grey
        moved = np.abs(out.colors[left_prior] - 0.5).sum(axis=1) > 0.05
        assert moved.any(), "no PRIOR splat on the guessed side was updated"

    def test_never_overwrites_an_observed_splat(self):
        rng = np.random.default_rng(4)
        cloud, right = self._asymmetric_cloud(rng)
        out, _ = mirror_confirmed(cloud, radius=0.2)
        # the confirmed side's own data must be bit-identical
        assert np.array_equal(out.colors[right], cloud.colors[right])
        assert np.array_equal(out.positions[right], cloud.positions[right])
        assert np.all(out.provenance[right] == Provenance.OBSERVED)

    def test_provenance_and_confidence_are_never_changed(self):
        """A mirrored splat is still just a guess -- same epistemic status as
        the raw prior it replaced, so a real future photo can still win."""
        rng = np.random.default_rng(5)
        cloud, right = self._asymmetric_cloud(rng)
        out, n_mirrored = mirror_confirmed(cloud, radius=0.2)
        assert n_mirrored > 0
        assert np.array_equal(out.provenance, cloud.provenance)
        assert np.array_equal(out.confidence, cloud.confidence)

    def test_empty_cloud_is_a_noop(self):
        cloud = GaussianCloud.empty()
        out, n = mirror_confirmed(cloud)
        assert n == 0
        assert out.n == 0

    def test_no_observed_splats_is_a_noop(self):
        rng = np.random.default_rng(6)
        n = 50
        cloud = GaussianCloud.create(
            positions=rng.uniform(-1, 1, size=(n, 3)).astype(np.float32),
            colors=np.full((n, 3), 0.5, np.float32),
            rotations=_identity_quats(n),
            provenance=np.full(n, Provenance.PRIOR, np.uint8),
        )
        out, n_mirrored = mirror_confirmed(cloud)
        assert n_mirrored == 0
        assert np.array_equal(out.colors, cloud.colors)


class TestSmoothMaterialBands:
    def _banded_cloud(self, rng, n_per_band=300):
        """Points spread across z so GLASS_BAND and BODY_BAND each get
        coverage, all PRIOR, all one coherent color -- then one outlier
        injected into each band."""
        parts = []
        for lo, hi in (GLASS_BAND, BODY_BAND):
            z = rng.uniform(lo + 0.01, hi - 0.01, n_per_band)
            x = rng.uniform(-1, 1, n_per_band)
            y = rng.uniform(-0.4, 0.4, n_per_band)
            parts.append(np.stack([x, y, z], axis=1))
        pos = np.concatenate(parts).astype(np.float32)
        n = pos.shape[0]
        colors = np.full((n, 3), 0.4, np.float32)  # one coherent color everywhere
        opacities = np.full(n, 0.9, np.float32)
        provenance = np.full(n, Provenance.PRIOR, np.uint8)
        cloud = GaussianCloud.create(
            positions=pos, colors=colors, opacities=opacities,
            rotations=_identity_quats(n), provenance=provenance,
        )
        return cloud

    def test_pulls_a_color_outlier_toward_its_local_neighborhood(self):
        rng = np.random.default_rng(7)
        cloud = self._banded_cloud(rng)
        # inject one glaring outlier splat into the glass band
        outlier_idx = 0
        colors = cloud.colors.copy()
        colors[outlier_idx] = [0.95, 0.95, 0.95]  # far from the 0.4 baseline
        cloud = cloud.with_updates(np.array([outlier_idx]), colors=colors[[outlier_idx]])

        out, n_touched = smooth_material_bands(cloud)
        assert n_touched > 0
        assert np.abs(out.colors[outlier_idx] - 0.4).sum() < np.abs(
            cloud.colors[outlier_idx] - 0.4
        ).sum(), "outlier was not pulled toward the local median"

    def test_well_behaved_splats_are_untouched(self):
        rng = np.random.default_rng(8)
        cloud = self._banded_cloud(rng)  # perfectly uniform color, no outliers
        out, n_touched = smooth_material_bands(cloud)
        assert n_touched == 0
        assert np.array_equal(out.colors, cloud.colors)
        assert np.array_equal(out.opacities, cloud.opacities)

    def test_observed_splats_are_never_touched(self):
        rng = np.random.default_rng(9)
        cloud = self._banded_cloud(rng)
        # make the outlier OBSERVED instead of PRIOR -- must survive untouched
        outlier_idx = 0
        colors = cloud.colors.copy()
        colors[outlier_idx] = [0.95, 0.95, 0.95]
        provenance = cloud.provenance.copy()
        provenance[outlier_idx] = Provenance.OBSERVED
        cloud = cloud.with_updates(
            np.array([outlier_idx]),
            colors=colors[[outlier_idx]],
            provenance=provenance[[outlier_idx]],
        )
        out, _ = smooth_material_bands(cloud)
        assert np.array_equal(out.colors[outlier_idx], cloud.colors[outlier_idx])

    def test_empty_cloud_is_a_noop(self):
        cloud = GaussianCloud.empty()
        out, n = smooth_material_bands(cloud)
        assert n == 0
        assert out.n == 0
