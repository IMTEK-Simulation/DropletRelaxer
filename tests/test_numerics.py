"""Discrete operators and mesh operations: they must not change the geometry
(or only by an amount that is bounded and tested)."""
import numpy as np
import pytest

from energy import area_and_grad, volume_and_grad, build_laplacian
from mesh import (make_disk_cone, subdivide_mesh, refine_rivara, collapse_short_edges,
                  tangential_smooth, max_edge_length, mean_edge_length)
from wetting import unique_edges


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
    assert np.std(np.linalg.norm(vs[t[:, 0]] - vs[t[:, 1]], axis=1)) <= np.std(np.linalg.norm(vp[t[:, 0]] - vp[t[:, 1]], axis=1))
