"""Frame sampling, matching, registration gating, renderer, embeddings, priors."""
from __future__ import annotations

import numpy as np
import pytest

from cargen.backends import (
    build_embedder,
    build_matcher,
    build_prior_generator,
    build_renderer,
    build_segmenter,
)
from cargen.core.camera import CameraPose, Intrinsics
from cargen.feature_matching.interface import MatchResult
from cargen.feature_matching.orb import OrbMatcher
from cargen.fusion_engine.point_renderer import PointRenderer
from cargen.pose_estimation.registration import (
    MIN_PNP_POINTS,
    LandmarkStore,
    PnPRegistrar,
    registration_confidence,
    solve_pnp,
)
from cargen.pose_estimation.stub import StubRegistrar
from cargen.prior_generation.canonical import (
    CANONICAL_LENGTH,
    canonicalize_orientation,
    normalize_to_canonical,
)
from cargen.prior_generation.interface import Mesh
from cargen.prior_generation.mesh_to_splats import mesh_to_splats
from cargen.prior_generation.stub import StubPriorGenerator, build_sedan_mesh
from cargen.reid.histogram import HistogramEmbedder
from cargen.reid.interface import Embedder
from cargen.segmentation.stub import StubSegmenter
from cargen.video.frame_sampler import FrameSampler, SampledFrame
from demo.synthetic import BackgroundSegmenter, orbit_pose, render_photo


class TestCanonicalFrame:
    def test_normalizes_length_and_ground(self):
        rng = np.random.default_rng(0)
        pts, factor = normalize_to_canonical(rng.normal(size=(100, 3)) * 7)
        assert np.ptp(pts[:, 0]) == pytest.approx(CANONICAL_LENGTH, abs=1e-4)
        assert pts[:, 2].min() == pytest.approx(0.0, abs=1e-6)
        assert factor > 0

    def test_scale_factor_is_applicable_to_companions(self):
        """Native-Gaussian backends must scale their splat sizes by the same
        factor the positions got, or the splats end up the wrong size."""
        pts = np.array([[0, 0, 0], [4, 0, 0], [0, 1, 2]], np.float32)
        out, factor = normalize_to_canonical(pts, from_y_up=False)
        assert np.ptp(out[:, 0]) == pytest.approx(CANONICAL_LENGTH, abs=1e-5)
        assert factor == pytest.approx(CANONICAL_LENGTH / 4.0, rel=1e-5)

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            normalize_to_canonical(np.zeros((0, 3)))
        with pytest.raises(ValueError):
            normalize_to_canonical(np.zeros((5, 2)))


class TestPcaCanonicalisation:
    """Image-to-3D backends emit a VIEW-ALIGNED frame — the vehicle arrives
    rotated by wherever the photo was taken. Without PCA, two scans of one car
    land in different orientations and merging fuses garbage."""

    @staticmethod
    def _car(rng, n=4000):
        """A car-shaped blob: length 4 > width 1.6 > height 1.2, sitting flat."""
        return rng.uniform(-1, 1, size=(n, 3)) * np.array([2.0, 0.8, 0.6])

    @staticmethod
    def _rotate(points, yaw=0.0, pitch=0.0):
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        return points @ (rz @ ry).T

    def test_recovers_axes_from_any_view_angle(self):
        """The measured failure: same car, different photo azimuth, different
        output orientation. After PCA the proportions must agree."""
        rng = np.random.default_rng(0)
        car = self._car(rng)
        shapes = []
        for yaw in (0.0, 0.6, 1.2, 2.5):
            rotated = self._rotate(car, yaw=yaw, pitch=0.25)  # camera elevation tilt
            out, _ = canonicalize_orientation(rotated)
            extent = np.ptp(out, axis=0)
            shapes.append(extent / extent[0])  # normalise out scale
        for other in shapes[1:]:
            assert np.allclose(shapes[0], other, atol=0.05), (
                f"orientations disagree across views: {shapes}"
            )

    def test_longest_axis_becomes_x(self):
        rng = np.random.default_rng(1)
        out, _ = canonicalize_orientation(self._rotate(self._car(rng), yaw=1.0))
        extent = np.ptp(out, axis=0)
        assert extent[0] > extent[1] > extent[2], f"axes not sorted: {extent}"

    def test_returns_a_proper_rotation(self):
        rng = np.random.default_rng(2)
        _, rotation = canonicalize_orientation(self._car(rng))
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4), "not a rotation"
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4)

    @staticmethod
    def _tapered_car(rng, n=6000):
        """Wide chassis, narrower roof — the asymmetry the up-heuristic reads."""
        z = rng.uniform(0, 1, n)
        x = rng.uniform(-2.0, 2.0, n)
        y = rng.uniform(-1, 1, n) * 0.8 * (1.0 - 0.6 * z)  # narrows with height
        return np.stack([x, y, z * 0.6], axis=1)

    @staticmethod
    def _base_width(points):
        """Horizontal spread of the lowest decile — same measure the code uses."""
        low = points[points[:, 2] <= np.percentile(points[:, 2], 10)]
        high = points[points[:, 2] >= np.percentile(points[:, 2], 90)]
        return (
            float(np.ptp(low[:, :2], axis=0).sum()),
            float(np.ptp(high[:, :2], axis=0).sum()),
        )

    def test_up_sign_resolved_by_wider_underside(self):
        """PCA can't tell up from down (a negated eigenvector is still an
        eigenvector). A car's chassis is wider than its roof, so an
        upside-down input must be flipped back."""
        rng = np.random.default_rng(3)
        car = self._tapered_car(rng)

        upright, _ = canonicalize_orientation(car)
        flipped, _ = canonicalize_orientation(car * np.array([1, 1, -1]))

        for tag, out in (("upright", upright), ("flipped", flipped)):
            base, roof = self._base_width(out)
            assert base > roof, f"{tag}: landed upside down (base {base:.3f} <= roof {roof:.3f})"

    def test_normalize_with_pca_grounds_and_scales(self):
        rng = np.random.default_rng(4)
        rotated = self._rotate(self._car(rng), yaw=0.9, pitch=0.3)
        out, scale = normalize_to_canonical(rotated, use_pca=True)
        assert np.ptp(out[:, 0]) == pytest.approx(CANONICAL_LENGTH, abs=1e-4)
        assert out[:, 2].min() == pytest.approx(0.0, abs=1e-6)
        assert scale > 0


class TestPrior:
    def test_sedan_mesh_is_well_formed(self):
        mesh = build_sedan_mesh()
        assert mesh.faces.max() < len(mesh.vertices)
        assert mesh.vertex_colors.shape == mesh.vertices.shape
        assert mesh.vertices[:, 2].min() >= -1e-6  # sits on the ground

    def test_generate_splats_default_path_samples_mesh(self):
        cloud = StubPriorGenerator().generate_splats(None, None, n_points=500)
        assert cloud.n == 500
        assert (cloud.provenance == 0).all()  # everything is a guess

    def test_prior_tints_from_photo(self):
        red = np.zeros((16, 16, 3), np.uint8)
        red[..., 0] = 220
        mesh = StubPriorGenerator().generate(red, np.ones((16, 16), np.float32))
        assert mesh.vertex_colors[:, 0].max() > mesh.vertex_colors[:, 1].max()

    def test_mesh_to_splats_is_deterministic(self):
        mesh = build_sedan_mesh()
        a = mesh_to_splats(mesh, n_points=300, seed=5)
        b = mesh_to_splats(mesh, n_points=300, seed=5)
        assert np.array_equal(a.positions, b.positions)

    def test_mesh_to_splats_rejects_degenerate(self):
        mesh = build_sedan_mesh()
        flat = type(mesh)(
            vertices=np.zeros_like(mesh.vertices),
            faces=mesh.faces,
            vertex_colors=mesh.vertex_colors,
        )
        with pytest.raises(ValueError, match="surface area"):
            mesh_to_splats(flat, n_points=10)


def _quat_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


@pytest.fixture
def flat_quad():
    """A unit square in the z=0 plane — every splat must lie flat in it."""
    return Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32),
        faces=np.array([[0, 1, 2], [0, 2, 3]], np.int32),
        vertex_colors=np.full((4, 3), 0.5, np.float32),
    )


class TestSurfaceAlignedSplats:
    """A Gaussian with equal scales is a sphere, and a cloud of spheres reads as
    gravel, never as bodywork. Splats must be flat discs lying IN the surface."""

    def test_splats_are_flat_not_spherical(self, flat_quad):
        cloud = mesh_to_splats(flat_quad, n_points=400, thin_ratio=0.1, seed=1)
        sx, sy, sz = cloud.scales.T
        assert np.allclose(sx, sy), "the two tangent axes should match"
        assert (sz < sx * 0.2).all(), f"splats are not flat: {cloud.scales[0]}"

    def test_thin_axis_lies_along_the_surface_normal(self, flat_quad):
        cloud = mesh_to_splats(flat_quad, n_points=200, seed=1)
        for q in cloud.rotations[:40]:
            local_z = _quat_to_matrix(q) @ np.array([0.0, 0.0, 1.0])
            # quad normal is +z; sign is free, the axis is what matters
            assert abs(abs(float(local_z[2])) - 1.0) < 1e-5, local_z

    def test_rotations_are_unit_quaternions(self, small_cloud):
        norms = np.linalg.norm(small_cloud.rotations, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_normals_follow_a_tilted_face(self):
        """A face tilted 45 degrees must produce splats tilted with it."""
        s = np.sqrt(0.5)
        mesh = Mesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, s, s]], np.float32),
            faces=np.array([[0, 1, 2]], np.int32),
            vertex_colors=np.full((3, 3), 0.5, np.float32),
        )
        cloud = mesh_to_splats(mesh, n_points=50, seed=1)
        expected = np.array([0.0, -s, s])  # normal of that triangle
        local_z = _quat_to_matrix(cloud.rotations[0]) @ np.array([0.0, 0.0, 1.0])
        assert min(
            np.abs(local_z - expected).max(), np.abs(local_z + expected).max()
        ) < 1e-5, local_z

    def test_axis_aligned_normals_do_not_produce_nan(self):
        """The tangent frame's seed axis must dodge the parallel-cross-product
        degeneracy, or splats facing straight up come out NaN."""
        mesh = Mesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
            faces=np.array([[0, 1, 2]], np.int32),
            vertex_colors=np.full((3, 3), 0.5, np.float32),
        )
        cloud = mesh_to_splats(mesh, n_points=20, seed=1)
        assert np.isfinite(cloud.rotations).all()
        assert np.isfinite(cloud.scales).all()


class TestTextureSampling:
    """The baked atlas holds ~1M texels; vertex colours hold one per vertex.
    Collapsing to vertex colours is ~80x detail thrown away."""

    @staticmethod
    def _ramp_texture():
        tex = np.zeros((64, 64, 3), np.float32)
        tex[:, :, 0] = np.linspace(0, 1, 64)[None, :]  # red across u
        tex[:, :, 2] = np.linspace(0, 1, 64)[:, None]  # blue down v
        return tex

    def test_texture_drives_colour_when_present(self, flat_quad):
        textured = Mesh(
            vertices=flat_quad.vertices,
            faces=flat_quad.faces,
            vertex_colors=np.full((4, 3), 0.5, np.float32),  # deliberately flat
            uv=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32),
            texture=self._ramp_texture(),
        )
        cloud = mesh_to_splats(textured, n_points=400, seed=1)
        # flat vertex colours could never produce variation — so any spread here
        # proves the texture, not the vertices, coloured these splats
        assert cloud.colors.std() > 0.1, "texture was ignored"
        assert textured.is_textured

    def test_falls_back_to_vertex_colours(self, flat_quad):
        cloud = mesh_to_splats(flat_quad, n_points=200, seed=1)
        assert not flat_quad.is_textured
        assert cloud.colors.std() < 1e-6  # flat vertex colours -> flat splats

    def test_uv_outside_range_is_clamped_not_wrapped(self, flat_quad):
        textured = Mesh(
            vertices=flat_quad.vertices,
            faces=flat_quad.faces,
            vertex_colors=np.full((4, 3), 0.5, np.float32),
            uv=np.array([[-5, -5], [5, -5], [5, 5], [-5, 5]], np.float32),
            texture=self._ramp_texture(),
        )
        cloud = mesh_to_splats(textured, n_points=100, seed=1)
        assert np.isfinite(cloud.colors).all()
        assert ((cloud.colors >= 0) & (cloud.colors <= 1)).all()


class TestObjectCentricFraming:
    """Image-to-3D models are trained on square, centred, subject-filling images.
    Handing one a raw photo (subject small and off-centre) is a severe
    distribution shift that returns a blob rather than an error — measured, it
    inflated a sedan's W/L from 0.446 to 0.524. This pins the framing helper's
    contract without needing SF3D's weights."""

    @staticmethod
    def _photo_with_offset_subject():
        """A 4:3 'photo' with the subject small and off to one side."""
        from PIL import Image

        rgba = np.zeros((300, 400, 4), np.uint8)
        rgba[..., :3] = 200
        rgba[180:240, 40:140, 3] = 255  # subject: 100x60, low-left, ~5% of frame
        rgba[180:240, 40:140, 0] = 255
        return Image.fromarray(rgba, mode="RGBA")

    def test_resize_foreground_squares_and_centres(self):
        """Uses SF3D's own helper, so this doubles as a check that our
        assumption about its behaviour still holds after an upstream bump."""
        from cargen.prior_generation.sf3d_impl import _ensure_importable

        try:
            _ensure_importable()  # puts the checkout on sys.path; needs no weights
        except FileNotFoundError:
            pytest.skip("SF3D checkout not present")
        from sf3d.utils import resize_foreground

        framed = resize_foreground(self._photo_with_offset_subject(), 0.85)
        assert framed.width == framed.height, "model needs a square frame"

        alpha = np.array(framed)[..., 3]
        ys, xs = np.nonzero(alpha > 127)
        assert ys.size, "subject vanished during framing"

        # subject should now dominate the frame, not occupy ~5% of it
        longest = max(np.ptp(xs), np.ptp(ys)) + 1
        assert longest / framed.width > 0.7, "subject does not fill the frame"

        # and be centred
        cx, cy = xs.mean() / framed.width, ys.mean() / framed.height
        assert abs(cx - 0.5) < 0.1 and abs(cy - 0.5) < 0.1, (cx, cy)


class TestMeshContract:
    def test_rejects_uv_that_does_not_match_vertices(self, flat_quad):
        with pytest.raises(ValueError, match="uv"):
            Mesh(
                vertices=flat_quad.vertices,
                faces=flat_quad.faces,
                vertex_colors=flat_quad.vertex_colors,
                uv=np.zeros((2, 2), np.float32),
            )

    def test_rejects_texture_without_uv(self, flat_quad):
        with pytest.raises(ValueError, match="uv"):
            Mesh(
                vertices=flat_quad.vertices,
                faces=flat_quad.faces,
                vertex_colors=flat_quad.vertex_colors,
                texture=np.zeros((8, 8, 3), np.float32),
            )

    def test_rejects_malformed_texture(self, flat_quad):
        with pytest.raises(ValueError, match="texture"):
            Mesh(
                vertices=flat_quad.vertices,
                faces=flat_quad.faces,
                vertex_colors=flat_quad.vertex_colors,
                uv=np.zeros((4, 2), np.float32),
                texture=np.zeros((8, 8), np.float32),
            )

    def test_concat_drops_uv_rather_than_fabricating(self, flat_quad):
        textured = Mesh(
            vertices=flat_quad.vertices, faces=flat_quad.faces,
            vertex_colors=flat_quad.vertex_colors,
            uv=np.zeros((4, 2), np.float32),
            texture=np.zeros((8, 8, 3), np.float32),
        )
        joined = textured.concat(flat_quad)
        assert not joined.is_textured  # two atlases can't merge without a repack
        assert joined.vertices.shape[0] == 8

    def test_export_raw_mesh_defaults_to_noop(self):
        assert StubPriorGenerator().export_raw_mesh("nowhere.glb") is False


class TestSegmenters:
    def test_stub_covers_center(self):
        mask = StubSegmenter(coverage=0.5).segment(np.zeros((100, 100, 3), np.uint8))
        assert mask[50, 50] == 1.0
        assert mask[2, 2] == 0.0

    def test_background_segmenter_isolates_object(self, truth_cloud, intrinsics, renderer):
        photo, true_mask = render_photo(truth_cloud, orbit_pose(0.0), intrinsics, renderer)
        mask = BackgroundSegmenter().segment(photo)
        overlap = (mask > 0.5) & (true_mask > 0.5)
        assert overlap.sum() / max((true_mask > 0.5).sum(), 1) > 0.9


class TestRenderer:
    def test_renders_and_attributes_pixels_to_splats(
        self, prior_cloud, intrinsics, renderer
    ):
        result = renderer.render(prior_cloud, orbit_pose(0.0), intrinsics)
        assert result.color.shape == (intrinsics.height, intrinsics.width, 3)
        assert result.hit_mask.sum() > 0
        idx = result.splat_index[result.hit_mask]
        assert idx.min() >= 0 and idx.max() < prior_cloud.n

    def test_empty_cloud_renders_background(self, intrinsics, renderer):
        from cargen.core.splat import GaussianCloud

        result = renderer.render(GaussianCloud.empty(), orbit_pose(0.0), intrinsics)
        assert not result.hit_mask.any()
        assert np.isinf(result.depth).all()

    def test_depth_ordering_near_occludes_far(self, intrinsics):
        from cargen.core.splat import GaussianCloud

        renderer = PointRenderer()
        cloud = GaussianCloud.create(
            positions=np.array([[0.0, 0, 0.5], [1.0, 0, 0.5]]),  # near/far along +x
            colors=np.array([[1.0, 0, 0], [0.0, 0, 1.0]]),
            scales=np.full((2, 3), 0.3, np.float32),
        )
        pose = CameraPose.look_at(eye=(4, 0, 0.5), target=(0, 0, 0.5))
        result = renderer.render(cloud, pose, intrinsics)
        centre = result.splat_index[intrinsics.height // 2, intrinsics.width // 2]
        assert centre == 1, "the nearer splat must win the centre pixel"

    def test_unproject_returns_splat_positions(self, prior_cloud, intrinsics, renderer):
        pose = orbit_pose(0.0)
        result = renderer.render(prior_cloud, pose, intrinsics)
        vs, us = np.nonzero(result.hit_mask)
        uv = np.stack([us[:20], vs[:20]], axis=1).astype(np.float64)
        points, valid = renderer.unproject(prior_cloud, pose, intrinsics, uv)
        assert valid.sum() > 0
        expected = prior_cloud.positions[result.splat_index[vs[:20], us[:20]]]
        assert np.allclose(points[valid], expected[valid], atol=1e-5)

    def test_unproject_ignores_out_of_frame(self, prior_cloud, intrinsics, renderer):
        uv = np.array([[-50.0, -50.0], [9999.0, 9999.0]])
        _, valid = renderer.unproject(prior_cloud, orbit_pose(0.0), intrinsics, uv)
        assert not valid.any()


class TestFrameSampler:
    def test_skips_near_duplicates(self):
        """Standing still must not spend fusion time on identical frames."""
        still = np.full((64, 64, 3), 120, np.uint8)
        frames = [(i, still, i / 30.0) for i in range(30)]
        sampled = FrameSampler(motion_threshold=5.0).sample(iter(frames))
        assert len(sampled) == 1  # only the anchor frame

    def test_samples_when_camera_moves(self, truth_cloud, intrinsics, renderer):
        frames = []
        for i, angle in enumerate(np.linspace(0, 2 * np.pi, 16, endpoint=False)):
            photo, _ = render_photo(truth_cloud, orbit_pose(angle), intrinsics, renderer)
            frames.append((i, photo, i / 12.0))
        sampled = FrameSampler(motion_threshold=4.0).sample(iter(frames))
        assert 1 < len(sampled) <= 16
        assert sampled[0].index == 0

    def test_respects_max_frames(self, truth_cloud, intrinsics, renderer):
        frames = []
        for i, angle in enumerate(np.linspace(0, 4 * np.pi, 40)):
            photo, _ = render_photo(truth_cloud, orbit_pose(angle), intrinsics, renderer)
            frames.append((i, photo, i / 12.0))
        sampled = FrameSampler(motion_threshold=1.0, max_frames=3).sample(iter(frames))
        assert len(sampled) == 3

    def test_empty_input(self):
        assert FrameSampler().sample(iter([])) == []

    def test_motion_between_grows_with_displacement(self, truth_cloud, intrinsics, renderer):
        sampler = FrameSampler()
        a, _ = render_photo(truth_cloud, orbit_pose(0.0), intrinsics, renderer)
        near, _ = render_photo(truth_cloud, orbit_pose(0.05), intrinsics, renderer)
        far, _ = render_photo(truth_cloud, orbit_pose(0.8), intrinsics, renderer)
        assert sampler.motion_between(a, near) < sampler.motion_between(a, far)

    def test_bad_video_path_raises(self):
        with pytest.raises(IOError):
            FrameSampler().sample_video("does-not-exist.mp4")


class TestMatching:
    def test_match_result_validates(self):
        with pytest.raises(ValueError):
            MatchResult(np.zeros((3, 2)), np.zeros((2, 2)), np.zeros(3))
        with pytest.raises(ValueError):
            MatchResult(np.zeros((3, 2)), np.zeros((3, 2)), np.zeros(2))

    def test_top_k(self):
        result = MatchResult(
            np.zeros((5, 2), np.float32), np.zeros((5, 2), np.float32),
            np.array([0.1, 0.9, 0.5, 0.2, 0.7], np.float32),
        )
        assert result.top_k(2).confidence.tolist() == pytest.approx([0.9, 0.7])
        assert result.top_k(99).count == 5

    def test_orb_matches_a_frame_to_itself(self, truth_cloud, intrinsics, renderer):
        photo, mask = render_photo(truth_cloud, orbit_pose(0.0), intrinsics, renderer)
        result = OrbMatcher(n_features=500).match(photo, photo, mask, mask)
        assert result.count > 0
        assert np.allclose(result.points_a, result.points_b, atol=1.0)

    def test_orb_on_textureless_returns_empty(self):
        blank = np.full((64, 64, 3), 200, np.uint8)
        assert OrbMatcher().match(blank, blank).count == 0


class TestRegistrationGating:
    def test_confidence_zero_below_min_points(self):
        assert registration_confidence(MIN_PNP_POINTS - 1, 100, 1.0, 4.0) == 0.0

    def test_confidence_zero_on_infinite_error(self):
        assert registration_confidence(50, 100, float("inf"), 4.0) == 0.0

    def test_confidence_rewards_inliers_and_accuracy(self):
        good = registration_confidence(60, 65, 0.5, 4.0)
        poor = registration_confidence(10, 200, 3.9, 4.0)
        assert good > poor
        assert 0.0 <= poor <= good <= 1.0

    def test_solve_pnp_recovers_a_known_pose(self, intrinsics):
        rng = np.random.default_rng(0)
        points = rng.uniform(-1, 1, size=(60, 3))
        truth = CameraPose.look_at(eye=(3.0, 0.5, 1.0), target=(0, 0, 0))
        uv, z = truth.project(points, intrinsics)
        front = z > 0
        pose, inliers, error = solve_pnp(points[front], uv[front], intrinsics)
        assert pose is not None
        assert error < 1.0
        assert np.allclose(pose.camera_center, truth.camera_center, atol=0.05)

    def test_solve_pnp_needs_enough_points(self, intrinsics):
        pose, _, error = solve_pnp(np.zeros((3, 3)), np.zeros((3, 2)), intrinsics)
        assert pose is None and np.isinf(error)

    def test_registrar_rejects_without_landmarks(self, intrinsics):
        registrar = PnPRegistrar(OrbMatcher())
        result = registrar.register(
            np.zeros((32, 32, 3), np.uint8), None, intrinsics, {}
        )
        assert not result.ok
        assert result.confidence == 0.0
        assert "no observed landmarks" in result.reason

    def test_registrar_rejects_unobserved_side(self, intrinsics, truth_cloud, renderer):
        """A photo of a side nobody has scanned must fail, not guess."""
        store = LandmarkStore()
        photo, mask = render_photo(truth_cloud, orbit_pose(0.0), intrinsics, renderer)
        store.add_view(photo, mask, np.zeros((0, 3)), np.zeros((0, 2)))
        other, other_mask = render_photo(
            truth_cloud, orbit_pose(np.pi), intrinsics, renderer
        )
        result = PnPRegistrar(OrbMatcher()).register(other, other_mask, intrinsics, {
            "landmarks": store
        })
        assert not result.ok

    def test_landmark_store_validates(self):
        with pytest.raises(ValueError):
            LandmarkStore().add_view(None, None, np.zeros((3, 3)), np.zeros((2, 2)))

    def test_registration_rejected_helper(self):
        from cargen.pose_estimation.interface import Registration

        r = Registration.rejected("nope")
        assert not r.ok and not r.accepted_at(0.0) and np.isinf(r.reprojection_error)

    def test_stub_registrar_pose_sequence(self, intrinsics):
        poses = [orbit_pose(0.0), orbit_pose(1.0)]
        stub = StubRegistrar(poses=poses)
        img = np.zeros((8, 8, 3), np.uint8)
        assert stub.register(img, None, intrinsics, {}).pose is poses[0]
        assert stub.register(img, None, intrinsics, {}).pose is poses[1]

    def test_stub_registrar_fail_after(self, intrinsics):
        stub = StubRegistrar(poses=[orbit_pose(0.0)], fail_after=1)
        img = np.zeros((8, 8, 3), np.uint8)
        assert stub.register(img, None, intrinsics, {}).ok
        assert not stub.register(img, None, intrinsics, {}).ok

    def test_stub_registrar_without_pose_rejects(self, intrinsics):
        result = StubRegistrar().register(np.zeros((8, 8, 3), np.uint8), None, intrinsics, {})
        assert not result.ok


class TestEmbedder:
    def test_same_image_scores_higher_than_different(self):
        red = np.zeros((64, 64, 3), np.uint8); red[..., 0] = 200
        blue = np.zeros((64, 64, 3), np.uint8); blue[..., 2] = 200
        embedder = HistogramEmbedder()
        a, b, c = embedder.embed(red), embedder.embed(red), embedder.embed(blue)
        assert Embedder.similarity(a, b) > Embedder.similarity(a, c)

    def test_embedding_is_normalized(self):
        rng = np.random.default_rng(0)
        vec = HistogramEmbedder().embed(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)

    def test_empty_mask_returns_zero_vector(self):
        vec = HistogramEmbedder().embed(
            np.zeros((32, 32, 3), np.uint8), np.zeros((32, 32), np.float32)
        )
        assert not vec.any()

    def test_similarity_handles_degenerate(self):
        assert Embedder.similarity(np.zeros(4), np.zeros(4)) == 0.0
        assert Embedder.similarity(None, np.ones(4)) == 0.0


class TestBackendRegistry:
    def test_stub_defaults_need_no_ml(self):
        assert build_segmenter("stub") is not None
        assert build_prior_generator("stub") is not None
        assert build_renderer("point") is not None
        assert build_matcher("orb") is not None
        assert build_embedder("histogram") is not None

    @pytest.mark.parametrize(
        "builder,name",
        [
            (build_segmenter, "nope"), (build_prior_generator, "nope"),
            (build_matcher, "nope"), (build_renderer, "nope"), (build_embedder, "nope"),
        ],
    )
    def test_unknown_backend_raises(self, builder, name):
        with pytest.raises(ValueError, match="unknown"):
            builder(name)

    def test_custom_prior_requires_target(self, monkeypatch):
        monkeypatch.delenv("CARGEN_CUSTOM_PRIOR", raising=False)
        with pytest.raises(ValueError, match="CARGEN_CUSTOM_PRIOR"):
            build_prior_generator("custom")

    def test_env_var_selects_backend(self, monkeypatch):
        monkeypatch.setenv("CARGEN_SEGMENTER", "stub")
        assert isinstance(build_segmenter(), StubSegmenter)
