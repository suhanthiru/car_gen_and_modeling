"""Export formats. These files must open in tools we don't control, so the
layout is a contract, not an implementation detail."""
from __future__ import annotations

import numpy as np
import pytest

from cargen.core.splat import SH_REST_COEFFS, GaussianCloud, Provenance
from cargen.export.exporter import (
    _logit,
    _rgb_to_sh_dc,
    export_all,
    write_ply,
    write_provenance_ply,
    write_splat,
)

_SH_C0 = 0.28209479177387814


def read_ply_header(path):
    with open(path, "rb") as f:
        blob = f.read(2048)
    return blob.split(b"end_header")[0].decode("ascii")


class TestPly:
    def test_header_matches_3dgs_layout(self, tmp_path, small_cloud):
        """Field names/order are what SuperSplat, Blender addons, and every
        other 3DGS tool expect. Renaming any of them silently breaks import."""
        path = write_ply(small_cloud, tmp_path / "m.ply")
        header = read_ply_header(path)
        assert "format binary_little_endian 1.0" in header
        assert f"element vertex {small_cloud.n}" in header
        for field in (
            "x", "y", "z", "nx", "ny", "nz",
            "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
            "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3",
        ):
            assert f"property float {field}\n" in header, f"missing {field}"

    def test_payload_size_is_exact(self, tmp_path, small_cloud):
        path = write_ply(small_cloud, tmp_path / "m.ply")
        header_len = len(read_ply_header(path)) + len("end_header\n")
        expected = header_len + small_cloud.n * 17 * 4  # 17 float32 fields
        assert path.stat().st_size == expected

    def test_values_roundtrip_through_3dgs_activations(self, tmp_path):
        """3DGS applies sigmoid(opacity), exp(scale), and SH→RGB on load, so we
        must store the inverses or every viewer shows the wrong thing."""
        cloud = GaussianCloud.create(
            positions=np.array([[1.0, 2.0, 3.0]]),
            colors=np.array([[0.25, 0.5, 0.75]]),
            opacities=np.array([0.8]),
            scales=np.array([[0.1, 0.2, 0.3]]),
        )
        path = write_ply(cloud, tmp_path / "m.ply")
        header_len = len(read_ply_header(path)) + len("end_header\n")
        with open(path, "rb") as f:
            f.seek(header_len)
            v = np.frombuffer(f.read(17 * 4), "<f4")

        assert v[0:3] == pytest.approx([1, 2, 3])
        # decode the way a viewer would
        assert 0.5 + _SH_C0 * v[6:9] == pytest.approx([0.25, 0.5, 0.75], abs=1e-5)
        assert 1 / (1 + np.exp(-v[9])) == pytest.approx(0.8, abs=1e-4)
        assert np.exp(v[10:13]) == pytest.approx([0.1, 0.2, 0.3], abs=1e-5)

    def test_empty_cloud_writes_valid_file(self, tmp_path):
        path = write_ply(GaussianCloud.empty(), tmp_path / "m.ply")
        assert "element vertex 0" in read_ply_header(path)


class TestSphericalHarmonics:
    """SH bands 1-3 are what make a highlight move as you orbit. Degree 0 is a
    matte object. The f_rest layout is a contract with SuperSplat/Blender: get
    the ordering wrong and viewers still load the file, but colours smear."""

    @staticmethod
    def _glossy(n=4):
        rng = np.random.default_rng(0)
        return GaussianCloud.create(
            positions=rng.normal(size=(n, 3)),
            colors=rng.random((n, 3)),
            sh_rest=rng.normal(size=(n, SH_REST_COEFFS, 3)) * 0.1,
        )

    def test_prior_is_matte_and_omits_f_rest(self, tmp_path, small_cloud):
        """A single-image prior has no evidence about view-dependence, so
        writing 45 zero floats per splat would triple the file for nothing."""
        assert not small_cloud.is_view_dependent
        header = read_ply_header(write_ply(small_cloud, tmp_path / "m.ply"))
        assert "f_rest_0" not in header

    def test_view_dependent_cloud_writes_all_45(self, tmp_path):
        cloud = self._glossy()
        assert cloud.is_view_dependent
        header = read_ply_header(write_ply(cloud, tmp_path / "m.ply"))
        for i in range(SH_REST_COEFFS * 3):
            assert f"property float f_rest_{i}\n" in header
        assert "f_rest_45" not in header  # degree 3 is exactly 45

    def test_f_rest_is_channel_major(self, tmp_path):
        """The reference impl stores (N,15,3) and writes .transpose(1,2).flatten:
        all 15 red coefficients, then green, then blue. Not interleaved."""
        cloud = self._glossy(n=1)
        path = write_ply(cloud, tmp_path / "m.ply")
        header_len = len(read_ply_header(path)) + len("end_header\n")
        with open(path, "rb") as f:
            f.seek(header_len)
            row = np.frombuffer(f.read((9 + 45 + 8) * 4), "<f4")
        rest = row[9:54]  # after xyz, normals, f_dc
        expected = np.transpose(cloud.sh_rest, (0, 2, 1)).reshape(-1)
        assert rest == pytest.approx(expected, abs=1e-6)
        # red block must equal the red channel of every coefficient
        assert rest[:15] == pytest.approx(cloud.sh_rest[0, :, 0], abs=1e-6)
        assert rest[15:30] == pytest.approx(cloud.sh_rest[0, :, 1], abs=1e-6)

    def test_sh_survives_select_and_concat(self):
        cloud = self._glossy(n=6)
        half = cloud.select(np.arange(3))
        assert half.sh_rest.shape == (3, SH_REST_COEFFS, 3)
        assert np.array_equal(half.sh_rest, cloud.sh_rest[:3])
        assert half.concat(half).sh_rest.shape == (6, SH_REST_COEFFS, 3)

    def test_provenance_view_is_deliberately_flat(self, tmp_path):
        """Provenance answers 'what is real?'; a view-dependent tint would make
        that shimmer as the camera moves."""
        path = write_provenance_ply(self._glossy(), tmp_path / "p.ply")
        assert "f_rest_0" not in read_ply_header(path)

    def test_logit_clamps_extremes(self):
        assert np.isfinite(_logit(np.array([0.0, 1.0]))).all()

    def test_rgb_to_sh_is_invertible(self):
        rgb = np.array([[0.1, 0.5, 0.9]], np.float32)
        assert 0.5 + _SH_C0 * _rgb_to_sh_dc(rgb) == pytest.approx(rgb, abs=1e-6)


class TestSplat:
    def test_size_is_32_bytes_per_splat(self, tmp_path, small_cloud):
        path = write_splat(small_cloud, tmp_path / "m.splat")
        assert path.stat().st_size == small_cloud.n * 32

    def test_sorted_by_significance_for_progressive_loading(self, tmp_path):
        """A truncated prefix must still be a usable model — that's what makes
        the viewer's progressive load show a coarse shell first."""
        rng = np.random.default_rng(0)
        cloud = GaussianCloud.create(
            positions=rng.normal(size=(50, 3)),
            colors=rng.random((50, 3)),
            scales=rng.random((50, 3)) * 0.1,
            opacities=rng.random(50).astype(np.float32),
        )
        path = write_splat(cloud, tmp_path / "m.splat")
        data = np.frombuffer(path.read_bytes(), np.uint8).reshape(-1, 32)
        scales = data[:, 12:24].copy().view("<f4").reshape(-1, 3)
        alpha = data[:, 27].astype(np.float32) / 255.0
        significance = alpha * scales.prod(axis=1)
        assert (np.diff(significance) <= 1e-6).all(), "not sorted big-to-small"

    def test_positions_survive_roundtrip(self, tmp_path):
        cloud = GaussianCloud.create(
            positions=np.array([[1.5, -2.5, 3.5]]), colors=np.ones((1, 3))
        )
        path = write_splat(cloud, tmp_path / "m.splat")
        data = np.frombuffer(path.read_bytes(), np.uint8).reshape(-1, 32)
        assert data[:, 0:12].copy().view("<f4")[0] == pytest.approx([1.5, -2.5, 3.5])


class TestProvenanceExport:
    def test_colors_encode_provenance(self, tmp_path):
        cloud = GaussianCloud.create(
            positions=np.zeros((2, 3)), colors=np.zeros((2, 3)),
            provenance=np.array([Provenance.PRIOR, Provenance.OBSERVED], np.uint8),
            confidence=np.array([1.0, 1.0], np.float32),
        )
        path = write_provenance_ply(cloud, tmp_path / "p.ply")
        header_len = len(read_ply_header(path)) + len("end_header\n")
        with open(path, "rb") as f:
            f.seek(header_len)
            rows = np.frombuffer(f.read(2 * 17 * 4), "<f4").reshape(2, 17)
        rgb = 0.5 + _SH_C0 * rows[:, 6:9]
        assert rgb[0, 0] > rgb[0, 1]  # prior reads red
        assert rgb[1, 1] > rgb[1, 0]  # observed reads green

    def test_source_cloud_not_mutated(self, tmp_path, small_cloud):
        before = small_cloud.colors.copy()
        write_provenance_ply(small_cloud, tmp_path / "p.ply")
        assert np.array_equal(small_cloud.colors, before)


def test_export_all_writes_every_format(tmp_path, small_cloud):
    paths = export_all(small_cloud, tmp_path / "exports")
    assert set(paths) == {"ply", "splat", "provenance_ply"}
    for path in paths.values():
        assert __import__("pathlib").Path(path).stat().st_size > 0
