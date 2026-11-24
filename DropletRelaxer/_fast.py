"""Lazy backend selection, caches, and Python adapters for the fast path."""

from contextlib import contextmanager
from functools import cache, cached_property, lru_cache
import os

import numpy as np
import scipy.sparse as sp


_native_requested = os.environ.get("DROPLET_NATIVE_KERNELS", "").strip().lower() in ("1", "true", "yes")


@cache
def numba_available():
    try:
        from numba import njit, version_info
        return callable(njit) and version_info[:2] >= (0, 63)
    except Exception:
        # An optional installation can fail to import (e.g. incompatible llvmlite).
        # Probe only Numba here; errors in our own kernels must still surface.
        return False


def use_native_kernels():
    return _native_requested or not numba_available()


def set_native_kernels(enabled=True):
    """Select NumPy/SciPy, or allow the optional backend when installed."""
    global _native_requested
    _native_requested = bool(enabled)


@contextmanager
def kernel_mode(*, native=False):
    previous = _native_requested
    if native:
        set_native_kernels(True)
    try:
        yield
    finally:
        set_native_kernels(previous)


def native_iterations(iterations):
    """Use native helpers for each step without changing the caller's mode between yields."""
    while True:
        with kernel_mode(native=True):
            try:
                frame = next(iterations)
            except StopIteration:
                return
        yield frame


@cache
def numba_kernels():
    """Load the compiled implementation only from an accelerated call site."""
    from . import kernels
    return kernels


class MeshTopology:
    """Derived data for one driver-owned mesh, with fixed triangles and ring.

    Keep this object only while the connectivity is unchanged. It is not
    included in checkpoints; a restarted run builds it from its restored mesh.
    """

    def __init__(self, verts, tris, boundary_idx):
        from .substrate import base_triangle_mask

        self.tris = tris
        self.boundary_idx = boundary_idx
        self.n_verts = len(verts)
        self.base_tri = base_triangle_mask(verts, tris, boundary_idx).copy()
        self.base_tri.setflags(write=False)
        self.dome_tris = tris[~self.base_tri] if self.base_tri.any() else tris

    @cached_property
    def dome_edges(self):
        from .wetting import unique_edges

        edges = unique_edges(self.dome_tris)
        edges.setflags(write=False)
        return edges

    @cached_property
    def embedder(self):
        from .substrate import BaseEmbedder

        if not self.base_tri.any():
            return None
        return BaseEmbedder(self.tris, self.boundary_idx, self.base_tri, self.n_verts)


def base_rhs_fast(embedder, verts):
    """Prepare compiled CSR buffers once, without doing this work in native mode."""
    csr = getattr(embedder, "_fast_csr", None)
    if csr is None:
        matrix = embedder._A_int
        csr = (np.ascontiguousarray(matrix.indptr, dtype=np.int64),
               np.ascontiguousarray(matrix.indices, dtype=np.int64),
               np.ascontiguousarray(matrix.data, dtype=np.float64))
        embedder._fast_csr = csr
    return numba_kernels().base_rhs_kernel(*csr, verts, embedder._is_int)


def collapse_short_edges(verts, tris, boundary_idx, min_len, protect_center, protect_radius, protect_min_len):
    verts_c = np.ascontiguousarray(verts, dtype=np.float64)
    tris_c = np.ascontiguousarray(tris, dtype=np.int64)
    bnd_c = np.ascontiguousarray(boundary_idx, dtype=np.int64)
    has_prot = protect_center is not None
    prot_c = np.ascontiguousarray(protect_center, dtype=np.float64) if has_prot else np.zeros(3, dtype=np.float64)
    has_prot_min = protect_min_len is not None
    p_min = float(protect_min_len) if has_prot_min else 0.0
    return numba_kernels().collapse_short_edges_kernel(
        verts_c, tris_c, bnd_c, float(min_len),
        has_prot, prot_c, float(protect_radius), has_prot_min, p_min,
    )


def tangential_smooth_step(verts, tris, adj, deg, fixed, strength):
    return numba_kernels().tangential_smooth_step_kernel(
        np.ascontiguousarray(verts, dtype=np.float64),
        np.ascontiguousarray(tris, dtype=np.int64),
        np.ascontiguousarray(adj.indptr, dtype=np.int64),
        np.ascontiguousarray(adj.indices, dtype=np.int64),
        np.ascontiguousarray(adj.data, dtype=np.float64),
        np.ascontiguousarray(deg, dtype=np.float64),
        np.ascontiguousarray(fixed, dtype=np.bool_),
        float(strength),
    )


def sphere_wetting_energy_and_grad(raw, idx, n_verts, radius, kappa_sphere):
    from .wetting import _stencil_colors

    colors = _stencil_colors(len(idx))
    eps = 1e-6 * radius
    kernels = numba_kernels()
    y, x = kernels.sphere_wetting_components_kernel(raw, colors, int(colors.max()) + 1, eps)
    # The vector ufunc and pairwise sum can round differently from Numba's
    # scalar atan2 and serial sum. Leave both in NumPy, but batch the calls.
    angles = np.arctan2(y, x, out=y)
    area = radius ** 2 * (2.0 * np.pi - angles[0].sum())
    grad = kernels.sphere_wetting_gradient_kernel(angles, colors, idx, n_verts, 2 * eps, kappa_sphere * radius ** 2)
    return -kappa_sphere * area, grad


def area_and_grad_fast(verts, tris, weights, flat_down):
    verts = np.ascontiguousarray(verts, dtype=np.float64)
    tris = np.ascontiguousarray(tris, dtype=np.int64)
    has_w = weights is not None
    w_arr = np.ascontiguousarray(weights, dtype=np.float64) if has_w else np.empty(0, dtype=np.float64)
    has_fd = flat_down is not None and bool(np.any(flat_down))
    fd_arr = np.ascontiguousarray(flat_down, dtype=np.bool_) if has_fd else np.empty(0, dtype=np.bool_)
    total, grad, tri_area = numba_kernels().area_and_grad_kernel(verts, tris, has_w, w_arr, has_fd, fd_arr)
    return float(total), grad, tri_area


def volume_and_grad_fast(verts, tris):
    verts = np.ascontiguousarray(verts, dtype=np.float64)
    tris = np.ascontiguousarray(tris, dtype=np.int64)
    V, grad = numba_kernels().volume_and_grad_kernel(verts, tris)
    return float(V), grad


def substrate_wall_fast(verts, boundary_loop, kappa):
    verts = np.ascontiguousarray(verts, dtype=np.float64)
    idx = np.ascontiguousarray(boundary_loop, dtype=np.int64)
    return numba_kernels().substrate_wall_kernel(verts, idx, float(kappa))


def substrate_penetration_fast(verts, stiffness):
    return numba_kernels().substrate_penetration_kernel(np.ascontiguousarray(verts, dtype=np.float64), float(stiffness))


def sphere_contact_fast(verts, center, radius, stiffness, adhesion, capture_radius):
    verts = np.ascontiguousarray(verts, dtype=np.float64)
    center = np.ascontiguousarray(center, dtype=np.float64)
    return numba_kernels().sphere_contact_kernel(verts, center, float(radius), float(stiffness),
                                                 float(adhesion), float(capture_radius))


def laplacian_fast(n_verts, edges):
    e0, e1 = edges[:, 0], edges[:, 1]
    deg = np.bincount(edges.ravel(), minlength=n_verts).astype(float)
    diag = np.arange(n_verts, dtype=np.int64)
    rows = np.concatenate([e0, e1, diag])
    cols = np.concatenate([e1, e0, diag])
    vals = np.concatenate([np.full(2 * len(edges), -1.0), deg])
    return sp.coo_matrix((vals, (rows, cols)), shape=(n_verts, n_verts)).tocsr()


@lru_cache(maxsize=64)
def _anomaly_profile(r, frac):
    from .anomaly import _profile
    return _profile(r, frac)


@lru_cache(maxsize=6)
def _anomaly_gauss01(n):
    xg, wg = np.polynomial.legendre.leggauss(n)
    return 0.5 * (xg + 1.0), 0.5 * wg


def disc_integral_fast(verts, ring, center_xy, r, frac, n_gauss=None):
    idx = np.ascontiguousarray(ring, dtype=np.int64)
    a = verts[idx, :2]
    d = np.roll(a, -1, axis=0) - a
    L = np.linalg.norm(d, axis=1)
    rho0, h, Sp = _anomaly_profile(r, frac)
    if n_gauss is None:
        n_gauss = int(np.clip(16 * np.ceil(L.max() / (h / 1.5) / 16.0), 16, 96))
    tg, wg = _anomaly_gauss01(n_gauss)
    cx, cy = (float(x) for x in center_xy)
    kernels = numba_kernels()
    rho, u = kernels.disc_radii_kernel(a, d, cx, cy, rho0, h, tg)
    # Keep NumPy's powers: a factored polynomial changes rounding.
    indicator = 1.0 - (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5)
    I, grad = kernels.disc_integral_kernel(a, d, L, idx, len(verts), cx, cy, rho0, Sp.coef, tg, wg, rho, u, indicator)
    return float(I), grad


def orient_wetted_loop(loop, verts, center):
    oriented = numba_kernels().wetted_loop_orient_kernel(loop, verts, center)
    if oriented is None:
        from .wetting import _orient_wetted_loop
        oriented = _orient_wetted_loop(loop, verts, center)
    return np.empty(0, dtype=np.int64) if oriented is None else oriented


def wetted_boundary_loop_fast(wetted_mask, tris, verts, center):
    loop = numba_kernels().wetted_boundary_unoriented_kernel(wetted_mask, tris)
    if len(loop) == 0:
        return loop
    return orient_wetted_loop(loop, verts, center)


def project_wetted_gradient(verts, grad, wet, center, force):
    """Project the gradient in place and add the corresponding sphere reaction."""
    fx, fy, fz = numba_kernels().wetted_radial_project_kernel(verts, grad, wet, center)
    force[0] -= fx
    force[1] -= fy
    force[2] -= fz


class RelaxKernels:
    """Adapters and topology-local scratch for one accelerated relax segment."""

    def __init__(self, tris, edges, adj, deg, boundary_idx, base_interior, lr, max_step_frac):
        self.tris = np.ascontiguousarray(tris, dtype=np.int64)
        self.e0 = np.ascontiguousarray(edges[:, 0], dtype=np.int64)
        self.e1 = np.ascontiguousarray(edges[:, 1], dtype=np.int64)
        self.indptr = np.ascontiguousarray(adj.indptr, dtype=np.int64)
        self.indices = np.ascontiguousarray(adj.indices, dtype=np.int64)
        self.deg = np.ascontiguousarray(deg, dtype=np.float64)
        self.boundary = np.ascontiguousarray(boundary_idx, dtype=np.int64)
        self.base_interior = np.ascontiguousarray(base_interior, dtype=np.int64)
        self.lr = float(lr)
        self.max_step_frac = float(max_step_frac)
        self._empty_weights = np.empty(0, dtype=np.float64)
        self._empty_indices = np.empty(0, dtype=np.int64)
        self._wet = self._tri_w = self._loop = None
        self._interior = self._empty_indices

    def wetted_patch(self, verts, wet, center, min_vertices):
        """Cache the topology of the patch, but orient it using current positions."""
        if self._wet is None or not np.array_equal(wet, self._wet):
            self._wet = wet
            self._tri_w = self._loop = None
            self._interior = self._empty_indices
            if wet.sum() >= min_vertices:
                inside = wet[self.tris].all(axis=1)
                if inside.any():
                    self._tri_w = np.where(inside, 0.0, 1.0)
                    kernels = numba_kernels()
                    self._loop = kernels.wetted_boundary_unoriented_kernel(wet, self.tris)
                    self._interior = kernels.wetted_interior_indices_kernel(wet, self.indptr, self.indices, self.deg)
        loop = None
        if self._loop is not None and len(self._loop):
            oriented = orient_wetted_loop(self._loop, verts, center)
            if len(oriented):
                loop = oriented
        return self._tri_w, loop

    def liquid_vapor_area(self, tri_area, lv_mask, tri_w):
        return float(numba_kernels().lv_area_sum_kernel(tri_area, lv_mask, tri_w is not None,
                                        tri_w if tri_w is not None else self._empty_weights))

    def take_step(self, verts, grad):
        return float(numba_kernels().trust_region_step_kernel(
            verts, grad, self.lr, self.max_step_frac, self.e0, self.e1, self.boundary, self.base_interior
        ))

    def smooth_and_snap(self, verts, wet, center, radius):
        if len(self._interior):
            numba_kernels().wetted_interior_smooth_kernel(verts, self._interior, self.indptr, self.indices, self.deg)
        numba_kernels().wetted_snap_kernel(verts, wet, center, radius)
