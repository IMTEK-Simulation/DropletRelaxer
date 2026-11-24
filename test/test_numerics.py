"""Discrete operators and mesh operations: they must not change the geometry
(or only by an amount that is bounded and tested)."""
import numpy as np
import pytest

from DropletRelaxer.energy import area_and_grad, volume_and_grad, build_laplacian
from DropletRelaxer.mesh import (make_disk_cone, subdivide_mesh, refine_rivara, collapse_short_edges,
                  tangential_smooth, max_edge_length, mean_edge_length)
from DropletRelaxer.wetting import unique_edges


def _dome(rng=None, n_rings=6, noise=0.0):
    v, t, b = make_disk_cone(n_rings, 1.0, 0.8)
    if noise and rng is not None:
        inner = ~np.isin(np.arange(len(v)), b)
        v = v.copy()
        v[inner, 2] += noise * rng.standard_normal(inner.sum())
    return v, t, b


def _euler_characteristic(v, t):
    e = unique_edges(t)
    return len(v) - len(e) + len(t)


def test_graph_laplacian_properties():
    v, t, _ = _dome()
    L = build_laplacian(len(v), t)
    assert (L - L.T).nnz == 0 or abs(L - L.T).max() == 0.0                      # symmetric
    assert np.allclose(L @ np.ones(len(v)), 0.0)                                # constants in the kernel
    ev = np.linalg.eigvalsh(L.toarray())
    assert ev.min() > -1e-10                                                     # positive semi-definite
    assert np.sum(ev < 1e-10) == 1                                               # connected mesh: one zero mode
    # row i: degree on the diagonal, -1 per neighbour
    e = unique_edges(t)
    deg = np.bincount(e.ravel(), minlength=len(v))
    assert np.allclose(L.diagonal(), deg)


@pytest.mark.parametrize("alpha", [0.5, 3.0, 20.0])
def test_sobolev_solve_is_a_low_pass_filter(alpha, rng):
    v, t, _ = _dome()
    L = build_laplacian(len(v), t)
    M = (L * alpha).toarray() + np.eye(len(v))
    smooth = np.ones(len(v))
    rough = rng.standard_normal(len(v))
    rough -= rough.mean()
    for g in (smooth, rough):
        gs = np.linalg.solve(M, g)
        # (I + alpha L) g_sob = g  and  ||g_sob|| <= ||g||
        assert np.allclose(M @ gs, g)
        assert np.linalg.norm(gs) <= np.linalg.norm(g) + 1e-12
    # a constant field is left untouched, noise is damped
    assert np.allclose(np.linalg.solve(M, smooth), smooth)
    assert np.linalg.norm(np.linalg.solve(M, rough)) < 0.9 * np.linalg.norm(rough)


def test_subdivision_keeps_geometry_exactly(rng):
    v, t, b = _dome(rng, noise=0.05)
    v2, t2, b2 = subdivide_mesh(v, t, b)
    assert len(t2) == 4 * len(t)
    assert area_and_grad(v2, t2)[0] == pytest.approx(area_and_grad(v, t)[0], rel=1e-12)
    assert volume_and_grad(v2, t2)[0] == pytest.approx(volume_and_grad(v, t)[0], rel=1e-12)
    assert _euler_characteristic(v2, t2) == _euler_characteristic(v, t)
    assert len(b2) == 2 * len(b) and np.allclose(v2[b2, 2], 0.0)


def test_rivara_refinement_keeps_geometry_exactly(rng):
    v, t, b = _dome(rng, noise=0.05)
    target = 0.5 * max_edge_length(v, t)
    v2, t2, b2 = refine_rivara(v, t, b, target)
    assert len(t2) > len(t)
    assert max_edge_length(v2, t2) < max_edge_length(v, t)
    assert area_and_grad(v2, t2)[0] == pytest.approx(area_and_grad(v, t)[0], rel=1e-12)
    assert volume_and_grad(v2, t2)[0] == pytest.approx(volume_and_grad(v, t)[0], rel=1e-12)
    assert _euler_characteristic(v2, t2) == _euler_characteristic(v, t)
    assert np.allclose(v2[b2, 2], 0.0)
    # conforming mesh: no edge shared by more than two triangles
    e = np.sort(np.vstack([t2[:, [0, 1]], t2[:, [1, 2]], t2[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    assert cnt.max() <= 2


def test_rivara_float32_uses_same_float64_working_mesh(monkeypatch):
    from DropletRelaxer import _fast
    from kernel_parity_helpers import assert_bits

    if not _fast.numba_available():
        pytest.skip("requires Numba")
    v, t, b = _dome(n_rings=3)
    v = v.astype(np.float32)
    args = (v, t, b, 0.4, np.array([0, 0, 0.7], dtype=np.float32), 0.3, 0.2)
    with _fast.kernel_mode(native=True):
        expected = refine_rivara(*args)
    assert expected[0].dtype == np.float64 and len(expected[1]) > len(t)
    monkeypatch.setattr(_fast, "_native_requested", False)
    for got, want in zip(refine_rivara(*args), expected, strict=True):
        assert_bits(got, want)


def test_edge_collapse_keeps_topology_and_nearly_the_geometry(rng):
    v, t, b = _dome(rng, n_rings=8, noise=0.02)
    v, t, b = subdivide_mesh(v, t, b)
    L = mean_edge_length(v, t)
    v2, t2, b2 = collapse_short_edges(v, t, b, 0.8 * L)
    assert len(v2) < len(v) and len(t2) < len(t)
    assert _euler_characteristic(v2, t2) == _euler_characteristic(v, t)        # still a disc
    A1, A2 = area_and_grad(v, t)[0], area_and_grad(v2, t2)[0]
    V1, V2 = volume_and_grad(v, t)[0], volume_and_grad(v2, t2)[0]
    assert abs(A2 - A1) / A1 < 2e-2
    assert abs(V2 - V1) / V1 < 2e-2
    assert np.allclose(v2[b2, 2], 0.0)
    # no triangle degenerates or flips: all dome normals keep pointing upward-outward
    p1, p2, p3 = v2[t2[:, 0]], v2[t2[:, 1]], v2[t2[:, 2]]
    assert np.all(np.linalg.norm(np.cross(p2 - p1, p3 - p1), axis=1) > 1e-12)


def test_tangential_smoothing_changes_neither_area_nor_volume_noticeably(rng):
    v, t, b = _dome(rng, n_rings=8, noise=0.0)
    # shuffle interior vertices tangentially to create a poor mesh
    inner = ~np.isin(np.arange(len(v)), b)
    vp = v.copy()
    vp[inner, :2] += 0.02 * rng.standard_normal((inner.sum(), 2))
    vs = tangential_smooth(vp, t, b, strength=0.3, passes=3)
    A0, A1 = area_and_grad(vp, t)[0], area_and_grad(vs, t)[0]
    V0, V1 = volume_and_grad(vp, t)[0], volume_and_grad(vs, t)[0]
    assert abs(A1 - A0) / A0 < 1e-2
    assert abs(V1 - V0) / V0 < 1e-2
    assert np.allclose(vs[b], vp[b])                                           # ring is fixed
    # the point of it: triangles become more uniform
    assert (np.std(np.linalg.norm(vs[t[:, 0]] - vs[t[:, 1]], axis=1))
            <= np.std(np.linalg.norm(vp[t[:, 0]] - vp[t[:, 1]], axis=1)))


def test_float32_relax_uses_native_promotion(monkeypatch):
    from DropletRelaxer import _fast
    from DropletRelaxer.config import SolverConfig
    from DropletRelaxer.mesh import build_dof_mask
    from DropletRelaxer.solver import relax
    from kernel_parity_helpers import assert_bits

    if not _fast.numba_available():
        pytest.skip("requires Numba")
    v, t, ring = _dome(n_rings=3)
    v = v.astype(np.float32)
    args = (v, t, ring, build_dof_mask(len(v), ring), SolverConfig(n_iterations=2))
    monkeypatch.setattr(_fast, "_native_requested", False)
    actual = []
    for _, coords, area, volume, _, _ in relax(*args):
        assert not _fast.use_native_kernels()  # no mode leak while the generator is suspended
        actual.append((coords, area, volume))
    with _fast.kernel_mode(native=True):
        expected = [(coords, area, volume) for _, coords, area, volume, _, _ in relax(*args)]
    assert len(actual) == len(expected) == 3
    assert actual[1][0].dtype == np.float64
    for frame, ref in zip(actual, expected):
        for got, want in zip(frame, ref):
            assert_bits(got, want)


@pytest.mark.parametrize("do_collapse", [False, True])
def test_collapse_with_int32_triangles_preserves_numpy_dtypes(do_collapse, monkeypatch):
    from DropletRelaxer import _fast
    from kernel_parity_helpers import assert_bits

    if not _fast.numba_available():
        pytest.skip("requires Numba")
    v, t, ring = _dome()
    threshold = mean_edge_length(v, t) if do_collapse else 0.0
    args = (v, t.astype(np.int32), ring, threshold)
    monkeypatch.setattr(_fast, "_native_requested", False)
    actual = collapse_short_edges(*args)
    with _fast.kernel_mode(native=True):
        expected = collapse_short_edges(*args)
    assert (len(actual[1]) < len(t)) == do_collapse
    for got, want in zip(actual, expected):
        assert_bits(got, want)


def test_moving_wetting_sphere_matches_at_every_yield(monkeypatch):
    from DropletRelaxer import _fast
    from DropletRelaxer.solver import relax
    from kernel_parity_helpers import assert_bits
    from runhelpers import scaled_setup, static_cfg

    if not _fast.numba_available():
        pytest.skip("requires Numba")

    cfg = static_cfg(angle_deg=120.0, sphere=True, n_iter=128)
    cfg.obstacle.center_end = (0.2, *cfg.obstacle.center_start[1:])
    cfg.obstacle.wetting_contact_angle_deg = 120.0
    diag = {}

    def states():
        diag.clear()
        for state in relax(v, tris, ring, mask, cfg.solver, cfg.obstacle, V_target=volume, diag=diag):
            yield (*state, diag.get("force"))

    with _fast.kernel_mode(native=True):
        v, tris, ring, mask, volume = scaled_setup(cfg)
        reference = list(states())
    assert [state[0] for state in reference] == list(range(cfg.solver.n_iterations + 1))
    assert not np.array_equal(reference[0][4], reference[-1][4])
    assert any(loop is not None and np.any(force) for _, _, _, _, _, loop, force in reference)

    monkeypatch.setattr(_fast, "_native_requested", False)
    for actual, expected in zip(states(), reference, strict=True):
        for field, got, want in zip(("iteration", "verts", "area", "volume", "center", "loop", "force"),
                                    actual, expected, strict=True):
            try:
                if got is None or want is None:
                    assert got is want
                else:
                    assert_bits(got, want)
            except AssertionError as err:
                raise AssertionError(f"iteration {expected[0]}: {field} differs") from err


@pytest.mark.parametrize("case", ["sessile", "wetting-sphere", "anomaly", "roughness"])
def test_native_kernels_flag_is_bit_identical(case, monkeypatch):
    from DropletRelaxer import _fast
    from DropletRelaxer.driver import Driver
    from kernel_parity_helpers import assert_bits
    from test_driver import Recorder, short_cfg

    if not _fast.numba_available():
        pytest.skip("requires Numba")

    def run():
        cfg = short_cfg(430 if case == "sessile" else 200)
        if case == "wetting-sphere":
            cfg.obstacle.enabled = True
            cfg.obstacle.center_end = cfg.obstacle.center_start
            cfg.obstacle.wetting_contact_angle_deg = 120.0
            cfg.output.every = 10
        elif case == "anomaly":
            cfg.anomaly.enabled = True
            cfg.anomaly.contact_angle_deg = 75.0
            cfg.anomaly.radius = 40.0
            cfg.anomaly.center_xy = (15.0, -4.0)
        elif case == "roughness":
            cfg.roughness.enabled = True
            cfg.roughness.mode = "chemical"
        rec = Recorder()
        result = Driver(cfg, sinks=[rec]).run()
        assert result["iteration"] == cfg.solver.n_iterations
        if case == "sessile":
            assert result["remeshes"] >= 2
        if case == "wetting-sphere":
            assert any(frame.wetted_loop is not None for frame in rec.frames)
            assert np.any(rec.frames[-1].force != 0.0)
        return rec.frames

    def unexpected_fast_solver(*args):
        pytest.fail("native solver mode entered the fast solver adapter")

    monkeypatch.setattr(_fast, "_native_requested", False)
    fast = run()
    _fast.set_native_kernels(True)
    monkeypatch.setattr(_fast, "numba_kernels", unexpected_fast_solver)
    native = run()
    assert len(fast) == len(native)
    keys = ("verts", "tris", "energy", "area", "volume", "force", "boundary_idx", "wetted_loop")
    for accelerated, reference in zip(fast, native):
        assert (accelerated.iteration, accelerated.epoch) == (reference.iteration, reference.epoch)
        for key in keys:
            a, b = getattr(accelerated, key), getattr(reference, key)
            if a is None or b is None:
                assert a is b, key
            else:
                assert_bits(a, b)
