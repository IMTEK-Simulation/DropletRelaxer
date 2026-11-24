"""Exact parity of compiled solver primitives with the vanilla operations."""

import numpy as np
import pytest
from scipy import sparse

from DropletRelaxer import _fast
from DropletRelaxer.energy import row_norm
from DropletRelaxer.wetting import _stencil_colors
from kernel_parity_helpers import assert_bits

pytestmark = pytest.mark.skipif(not _fast.numba_available(), reason="requires Numba")
k = _fast.numba_kernels() if _fast.numba_available() else None


@pytest.mark.parametrize("bad", [None, np.inf, -np.inf, np.nan])
def test_check_verts_bounds(bad):
    verts = np.array([[1, -9, 0], [4, 6, -0.0]], dtype=float)
    if bad is not None:
        verts[1, 1] = bad
    finite, largest = k.check_verts_bounds_kernel(verts)
    assert finite == np.isfinite(verts).all()
    if finite:
        assert_bits(largest, np.abs(verts).max())
    else:
        assert np.isnan(largest)  # the guard returns a sentinel for nonfinite input


@pytest.mark.parametrize("n", [1, 2, 7, 8, 99, 100])
def test_quickselect_median(n, rng):
    lengths = rng.integers(0, 20, size=n).astype(float) / 7
    assert_bits(k.quickselect_median(lengths.copy()), np.median(lengths))


@pytest.mark.parametrize("frac", [0.0, 0.12, 1.0])
@pytest.mark.parametrize("order", ["C", "F"])
def test_trust_region_step(frac, order, rng):
    v = rng.standard_normal((8, 3))
    v[1] = v[0]  # zero-length edge; vertex 7 is isolated
    e0 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    e1 = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    bnd, base = np.array([1, 3], dtype=np.int64), np.array([6], dtype=np.int64)
    gs = np.array(rng.standard_normal(v.shape), order=order)
    gs[4] = 0
    gs[7] = [float.fromhex(x) for x in ('0x1.b1fbd2a0f8925p-1', '0x1.9adaec27c415ep-2', '0x1.1b4396d6d910ep-1')]
    lr = 1.0
    want_gs = gs.copy(order='K')
    want_gs[bnd, 2] = 0.0
    want_gs[base] = 0.0
    step = lr * want_gs
    if frac:
        lengths = row_norm(v[e0] - v[e1])
        shortest = np.full(len(v), np.inf)
        np.minimum.at(shortest, e0, lengths)
        np.minimum.at(shortest, e1, lengths)
        cap = frac * np.maximum(shortest, 0.25 * np.median(lengths))
        step *= np.minimum(1, cap / np.maximum(row_norm(step), 1e-300))[:, None]
    want_v = v - step
    want_v[bnd, 2] = 0.0
    got = k.trust_region_step_kernel(v, gs, lr, frac, e0, e1, bnd, base)
    assert_bits(gs, want_gs)
    assert_bits(v, want_v)
    assert_bits(got, row_norm(step).max())


@pytest.mark.parametrize("mode", ["empty", "single", "patch"])
def test_wetted_radial_project(mode, rng):
    v = rng.standard_normal((80, 3))
    center = np.array([1.2, -0.2, 2.4])
    v[0] = center      # exercises the 1e-12 denominator clamp
    wet = rng.random(len(v)) > 0.3 if mode == "patch" else np.zeros(len(v), bool)
    if mode == "single":
        wet[0] = True
    grad = rng.standard_normal(v.shape)
    want = grad.copy()
    d = v[wet] - center
    normal = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    radial = np.sum(grad[wet] * normal, axis=1, keepdims=True) * normal
    want[wet] -= radial
    got_force = k.wetted_radial_project_kernel(v, grad, wet, center)
    assert_bits(grad, want)
    assert_bits(got_force, radial.sum(axis=0))


@pytest.mark.parametrize("mode", ["empty", "patch"])
def test_wetted_snap(mode, rng):
    v = rng.standard_normal((80, 3))
    center = np.array([1.2, -0.2, 2.4])
    v[0] = center
    wet = rng.random(len(v)) > 0.3 if mode == "patch" else np.zeros(len(v), bool)
    if mode == "patch":
        wet[0] = True
    radius = 1.72
    want = v.copy()
    d = v[wet] - center
    want[wet] = center + radius * d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    k.wetted_snap_kernel(v, wet, center, radius)
    assert_bits(v, want)


@pytest.mark.parametrize("mode", ["empty", "all", "partial"])
def test_wetted_interior(mode, rng):
    v = rng.standard_normal((7, 3))
    src = [0, 1, 2, 2, 3, 4]
    dst = [1, 2, 3, 4, 4, 5]
    adj = sparse.csr_matrix((np.ones(12), (src + dst, dst + src)), shape=(7, 7))
    deg = np.asarray(adj.sum(axis=1)).ravel()
    wet = {"empty": np.zeros(7, bool), "all": np.ones(7, bool),
           "partial": np.array([1, 1, 1, 1, 0, 0, 1], bool)}[mode]
    interior = np.flatnonzero(wet & (adj @ wet.astype(float) >= deg - 0.5)).astype(np.int64)
    got_interior = k.wetted_interior_indices_kernel(wet, adj.indptr, adj.indices, deg)
    assert_bits(got_interior, interior)
    want = v.copy()
    centroid = (adj @ v) / np.maximum(deg, 1)[:, None]
    want[interior] = 0.5 * v[interior] + 0.5 * centroid[interior]
    k.wetted_interior_smooth_kernel(v, got_interior, adj.indptr, adj.indices, deg)
    assert_bits(v, want)


def _numpy_turn_components(u):
    left = u - np.roll(u, 1, axis=0)
    right = np.roll(u, -1, axis=0) - u
    left -= np.sum(left * u, axis=1, keepdims=True) * u
    right -= np.sum(right * u, axis=1, keepdims=True) * u
    left /= np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return np.sum(u * np.cross(left, right), axis=1), np.sum(left * right, axis=1)


@pytest.mark.parametrize("n", [7, 24, 52, 137])
def test_wetting_angle_and_gradient_stages(n, rng):
    """Test intermediate bits as well as the public wetting calculation."""
    raw = rng.standard_normal((n, 3)) + [0.0, 0.0, 4.0]
    idx = np.arange(n, dtype=np.int64) * 2
    colors = _stencil_colors(n)
    nc = int(colors.max()) + 1
    eps, scale = 1.23e-6, 0.73
    u = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    y, x = _numpy_turn_components(u)
    cy, cx = k.turning_angle_components_kernel(u)
    assert_bits(cy, y)
    assert_bits(cx, x)
    ys, xs = k.sphere_wetting_components_kernel(raw, colors, nc, eps)
    assert_bits(ys[0], y)
    assert_bits(xs[0], x)
    ref_grad = np.zeros((2 * n, 3))
    for c in range(nc):
        sel = np.flatnonzero(colors == c)
        touched = np.unique(np.r_[(sel - 1) % n, sel, (sel + 1) % n])
        for d in range(3):
            angles = []
            for sign in (0, 1):
                pert = raw.copy()
                pert[sel, d] += eps if sign == 0 else -eps
                normalized = pert / np.linalg.norm(pert, axis=1, keepdims=True)
                y, x = _numpy_turn_components(normalized)
                row = 1 + 6 * c + 2 * d + sign
                assert_bits(ys[row, touched], y[touched])
                assert_bits(xs[row, touched], x[touched])
                angles.append(np.arctan2(y, x))
            dt = (angles[0] - angles[1]) / (2 * eps)
            ref_grad[idx[sel], d] = scale * (dt + np.roll(dt, 1) + np.roll(dt, -1))[sel]
    got_grad = k.sphere_wetting_gradient_kernel(np.arctan2(ys, xs, out=ys), colors, idx, 2 * n, 2 * eps, scale)
    assert_bits(got_grad, ref_grad)
