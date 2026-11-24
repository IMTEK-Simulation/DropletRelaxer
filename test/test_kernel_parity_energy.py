"""Bit parity against the independent NumPy/SciPy energy paths."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from DropletRelaxer import _fast
from DropletRelaxer.anomaly import disc_integral_and_grad
from DropletRelaxer.energy import (area_and_grad, build_laplacian, row_norm, substrate_penetration_energy_and_grad,
                                   substrate_wall_energy_and_grad, volume_and_grad)
from DropletRelaxer.mesh import tangential_smooth_step
from DropletRelaxer.obstacle import sphere_contact_energy_and_grad
from kernel_parity_helpers import assert_bits
from test_roughness import _field, _loop

pytestmark = pytest.mark.skipif(not _fast.numba_available(), reason="requires Numba")
k = _fast.numba_kernels() if _fast.numba_available() else None


@pytest.fixture
def compare_backend(monkeypatch):
    def compare(fn, *args, accelerated=True):
        dispatched = False

        def backend():
            nonlocal dispatched
            dispatched = True
            return k

        with monkeypatch.context() as patch:
            patch.setattr(_fast, "numba_kernels", backend)
            patch.setattr(_fast, "_native_requested", False)
            fast = fn(*args)
            assert dispatched == accelerated
            dispatched = False
            patch.setattr(_fast, "_native_requested", True)
            native = fn(*args)
            assert not dispatched
        return fast, native

    return compare


@pytest.mark.parametrize("seed", [0, 3, 17, 12345])
@pytest.mark.parametrize("n", [0, 1, 7, 8, 9, 17, 127, 128, 129, 255, 256, 257, 1023, 8193])
def test_pairwise_sum(n, seed):
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(n) * 10.0 ** rng.integers(-6, 7, size=n)
    values[::5] = -0.0
    assert_bits(k.pairwise_sum_1d(values), values.sum())


@pytest.mark.parametrize("mode", ["empty", "ordinary", "weighted", "weighted-oriented-base"])
def test_area_and_gradient(mode, rng, compare_backend):
    v = rng.normal(size=(92, 3))
    v[:6, 2] = 0
    t = rng.integers(6, len(v), size=(204, 3), dtype=np.int64)
    t[:3] = [[0, 1, 2], [2, 3, 4], [3, 2, 5]]
    if mode == "empty":
        t = t[:0]
    weights = rng.normal(size=len(t)) if mode.startswith("weighted") else None
    flat = np.arange(len(t)) < 3 if mode == "weighted-oriented-base" else None
    fast, native = compare_backend(area_and_grad, v, t, weights, flat)
    for got, want in zip(fast, native):
        assert_bits(got, want)


@pytest.mark.parametrize("n", [0, 8, 159, 633])
def test_volume_and_gradient(n, rng, compare_backend):
    v = rng.normal(size=(90, 3))
    t = rng.integers(0, len(v), size=(n, 3), dtype=np.int64)
    fast, native = compare_backend(volume_and_grad, v, t)
    for got, want in zip(fast, native):
        assert_bits(got, want)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("mode", ["area", "weighted-area", "volume"])
def test_non_float64_geometry_matches_native(dtype, mode, compare_backend):
    verts = np.array([[.1, .1, .1], [1.3, .1, .1], [.1, 1.2, .1], [.1, .1, 1.]], dtype=dtype)
    tris = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    if mode == "volume":
        fast, native = compare_backend(volume_and_grad, verts, tris, accelerated=False)
    else:
        weights = np.array([.3, -.6, 1., .5], dtype=dtype) if mode == "weighted-area" else None
        fast, native = compare_backend(area_and_grad, verts, tris, weights, accelerated=False)
    assert fast[1].dtype == dtype
    for got, want in zip(fast, native):
        assert_bits(got, want)


def test_float32_contact_energies_match_native(compare_backend):
    verts = np.array([[.1, .1, -.1], [1.3, .1, .1], [.1, 1.2, .1], [.1, .1, 1.]], dtype=np.float32)
    for fn, args in (
        (substrate_wall_energy_and_grad, (np.arange(3), .7)),
        (substrate_penetration_energy_and_grad, (.3,)),
        (sphere_contact_energy_and_grad, (np.zeros(3, dtype=np.float32), 1.1, .4, .2, .7)),
    ):
        fast, native = compare_backend(fn, verts, *args, accelerated=False)
        for got, want in zip(fast, native):
            assert_bits(got, want)


@pytest.mark.parametrize("mode", ["open", "reversed"])
def test_substrate_wall(mode, compare_backend):
    v, ring = _loop(137)
    if mode == "reversed":
        ring = ring[::-1]
    fast, native = compare_backend(substrate_wall_energy_and_grad, v, ring, -0.73)
    for got, want in zip(fast, native):
        assert_bits(got, want)


@pytest.mark.parametrize("mode", ["none", "mixed"])
def test_substrate_penetration(mode, rng, compare_backend):
    v = rng.normal(size=(1001, 3))
    v[0, 2] = -0.0
    if mode == "none":
        v[:, 2] = np.abs(v[:, 2])
    fast, native = compare_backend(substrate_penetration_energy_and_grad, v, 0.713)
    for got, want in zip(fast, native):
        assert_bits(got, want)


@pytest.mark.parametrize("adhesion", [0.0, 0.77], ids=["repulsion", "capture"])
def test_sphere_contact(adhesion, rng, compare_backend):
    v = rng.normal(size=(1031, 3))
    center = np.array([0.2, 0.3, -0.7])
    v[:3] = [center, center + [1.25, 0, 0], center + [1.625, 0, 0]]
    fast, native = compare_backend(sphere_contact_energy_and_grad, v, center, 1.25, 0.39, adhesion, 0.375)
    for got, want in zip(fast, native):
        assert_bits(got, want)


@pytest.mark.parametrize("mode", ["topography", "chemical"])
def test_roughness_quadrature(mode, compare_backend):
    opts = dict(rms_slope=0.5) if mode == "topography" else dict(rms_kappa=0.08)
    field = _field(mode, **opts)
    v, ring = _loop(107, c=(-1.5, 0.92), R=13.7)
    if mode == "chemical":
        ring = ring[::-1]
    fast, native = compare_backend(field.wetted_integral_and_grad, v, ring)
    for got, want in zip(fast, native):
        assert_bits(got, want)


def test_disc_radii(rng):
    starts = rng.uniform(-4, 4, (97, 2))
    edges = rng.normal(size=starts.shape)
    nodes = np.r_[0.0, np.linspace(0.03, 0.97, 30), 1.0]
    center = np.array([0.17, -1.09])
    starts[0] = center  # zero radial distance at the first quadrature node
    rho0, h = 1.73, 0.28
    want = np.linalg.norm(starts[:, None, :] + nodes[None, :, None] * edges[:, None, :] - center, axis=2)
    rho, u = k.disc_radii_kernel(starts, edges, *center, rho0, h, nodes)
    assert_bits(rho, want)
    assert_bits(u, np.clip((want - rho0) / (2 * h), 0.0, 1.0))


@pytest.mark.parametrize("region", ["inside", "transition", "outside"])
def test_disc_quadrature(region, compare_backend):
    v, ring = _loop(91, c=(0.05, -3.01), R=2.1)
    r = {"inside": 8.0, "transition": 2.2, "outside": 0.55}[region]
    fast, native = compare_backend(disc_integral_and_grad, v, ring, (0.02, -3.11), r, 0.17, 32)
    for got, want in zip(fast, native):
        assert_bits(got, want)


def test_tangential_normals(rng):
    v = rng.normal(size=(36, 3))
    t = rng.integers(0, 35, size=(100, 3), dtype=np.int64)  # includes a vertex with no faces
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    fn = np.cross(b - a, c - a)
    vn = np.zeros_like(v)
    for j in range(3):
        scat = np.zeros_like(v)
        np.add.at(scat, t[:, j], fn)
        vn += scat
    vn /= np.maximum(row_norm(vn), 1e-300)[:, None]
    assert_bits(k.tangential_smooth_normals_kernel(v, t), vn)


@pytest.mark.parametrize("strength", [0.0, 0.31])
def test_tangential_smooth(rng, compare_backend, strength):
    v = rng.normal(size=(36, 3))
    v[0, 2] = -0.0  # native adds a positive zero to fixed vertices
    t = rng.integers(0, 35, size=(100, 3), dtype=np.int64)
    a, b = t.ravel(), np.roll(t, -1, axis=1).ravel()
    adj = csr_matrix((rng.uniform(0.2, 0.8, len(a)), (a, b)), shape=(len(v), len(v)))
    deg = np.asarray(adj.sum(axis=1)).ravel()
    fixed = rng.random(len(v)) < 0.3
    fixed[0] = True
    fast, native = compare_backend(tangential_smooth_step, v, t, adj, deg, fixed, strength)
    assert_bits(fast, native)


@pytest.mark.parametrize("weighted", [False, True])
def test_lv_area_sum(weighted, rng):
    a = rng.uniform(0.01, 15.0, 721)
    mask = rng.random(len(a)) < 0.78
    w = (rng.random(len(a)) < 0.76).astype(float) if weighted else np.empty(0)
    assert_bits(k.lv_area_sum_kernel(a, mask, weighted, w), (a * (mask * w if weighted else mask)).sum())


@pytest.mark.parametrize("a,mask,weights", [
    pytest.param([1.0, np.inf], [True, False], None, id="unweighted-masked-inf"),
    pytest.param([1.0, np.nan], [True, False], None, id="unweighted-masked-nan"),
    pytest.param([-0.0] * 8, [False] * 8, None, id="unweighted-masked-signed-zero"),
    pytest.param([1.0, np.nan], [True, False], [1.0, 1.0], id="weighted-masked-nan-area"),
    pytest.param([1.0, 2.0], [True, False], [1.0, np.inf], id="weighted-masked-inf-weight"),
    pytest.param([1.0, 1e300], [True, False], [1.0, 1e300], id="weighted-mask-before-area"),
    pytest.param([-0.0] * 8, [False] * 8, [1.0] * 8, id="weighted-masked-signed-zero"),
])
def test_lv_area_sum_preserves_mask_multiplication(a, mask, weights):
    a, mask = np.asarray(a), np.asarray(mask)
    w = np.asarray(weights) if weights is not None else np.empty(0)
    with np.errstate(invalid="ignore"):
        expected = (a * (mask * w if weights is not None else mask)).sum()
    assert_bits(k.lv_area_sum_kernel(a, mask, weights is not None, w), expected)


@pytest.mark.parametrize("count", [0, 70])
def test_laplacian(count, rng):
    tris = rng.choice(50, size=(count, 3), replace=True).astype(np.int64)
    native = build_laplacian(51, tris)
    fast = _fast.laplacian_fast(51, k.unique_edges_kernel(tris))
    for attr in ("indptr", "indices", "data"):
        assert_bits(getattr(fast, attr), getattr(native, attr))


def test_base_rhs(rng):
    rows = rng.integers(0, 63, size=115)
    cols = rng.integers(0, 101, size=115)
    adj = csr_matrix((rng.uniform(0.11, 1.1, len(rows)), (rows, cols)), shape=(65, 101))
    v = rng.normal(size=(101, 3))
    inside = rng.random(len(v)) < 0.4
    xy = v[:, :2].copy()
    xy[inside] = 0.0
    assert_bits(k.base_rhs_kernel(adj.indptr, adj.indices, adj.data, v, inside), adj @ xy)
