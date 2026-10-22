"""Circular wettability anomaly on the substrate.

The wall energy of the flat substrate is  E_wall = -gamma_lv * cos(theta_w(x, y)) * A_sl
(Young), i.e. an integral over the wetted region of the local wettability
cos(theta_w).  With a homogeneous substrate (angle theta_0, kappa_0 = cos theta_0)
this term is carried by the weight of the base triangles / the shoelace area
(see solver.py).  A circular anomaly with another contact angle theta_a
(kappa_a = cos theta_a) adds only the DIFFERENCE:

    E_anom = -gamma_lv * (kappa_a - kappa_0) * Integral_{wetted region} s(x, y) dA,

with a smooth indicator s of the disc (centre p, radius r):

    s = 1 for rho <= r - h,  s = 0 for rho >= r + h,  quintic "smootherstep"
    in between (C2), rho = |x - p|, total transition width 2h =
    transition_fraction * r.

The integral over the wetted region is evaluated EXACTLY for the polygon of
the contact line (no base mesh needed) as a boundary flux (divergence
theorem):  s = div P  with the radial field  P = S(rho)/rho^2 (x - p),
S(rho) = Integral_0^rho s(r') r' dr'  (closed-form polynomial), so
Integral s dA = sum over contact-line edges of  Integral P . n dl  (Gauss-
Legendre quadrature along each edge).  The gradient with respect to a
contact-line vertex is the exact shape derivative of the polygon integral,
Integral_edges s(x) n phi_a(x) dl, with phi_a the linear hat function of the
vertex along the adjacent edges (same quadrature, so energy and force belong
together).  The term only depends on the contact-line vertices (in-plane),
not on the sphere, so it adds no force on the sphere.
"""

import numpy as np
from numpy.polynomial import Polynomial as _P


def _profile(r, frac):
    """(rho0, h, S_poly) of the smooth disc indicator: transition from
    rho0 = r - h to r + h, h = frac * r / 2 (clamped to <= r)."""
    h = min(0.5 * frac * r, r)
    h = max(h, 1e-9 * max(r, 1e-12))
    rho0 = r - h
    u = _P([0.0, 1.0])
    q = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5      # smootherstep: q(0)=0, q(1)=1
    integrand = (1 - q) * (rho0 + 2 * h * u) * (2 * h)
    return rho0, h, rho0 ** 2 / 2 + integrand.integ()


def s_profile(rho, r, frac):
    """The smooth indicator s(rho) (1 inside, 0 outside)."""
    rho0, h, _ = _profile(r, frac)
    u = np.clip((rho - rho0) / (2 * h), 0.0, 1.0)
    return 1.0 - (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5)


def _S(rho, r, frac):
    rho0, h, Sp = _profile(r, frac)
    u = np.clip((rho - rho0) / (2 * h), 0.0, 1.0)
    return np.where(rho <= rho0, 0.5 * rho ** 2, Sp(u))


def disc_integral_and_grad(verts, ring, center_xy, r, frac, n_gauss=None):
    """I = Integral over the polygon of the contact line (xy projection of
    `verts[ring]`, cyclic order) of the smooth disc indicator, and its
    gradient (n_verts x 3, only the xy components of ring vertices)."""
    idx = np.asarray(ring, dtype=int)
    a = verts[idx, :2]
    b = np.roll(a, -1, axis=0)
    d = b - a
    L = np.linalg.norm(d, axis=1)
    t = d / np.maximum(L, 1e-300)[:, None]
    shoe = 0.5 * np.sum(a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1])
    sg = 1.0 if shoe >= 0 else -1.0                     # ring orientation
    nrm = sg * np.column_stack([t[:, 1], -t[:, 0]])     # outward normal per edge
    _, h, _ = _profile(r, frac)
    if n_gauss is None:
        n_gauss = int(np.clip(16 * np.ceil(L.max() / (h / 1.5) / 16.0), 16, 96))  # steps of 16: no jitter
    xg, wg = np.polynomial.legendre.leggauss(n_gauss)
    tg, wg = 0.5 * (xg + 1.0), 0.5 * wg                 # nodes on [0, 1]
    p = a[:, None, :] + tg[None, :, None] * d[:, None, :]          # (E, G, 2)
    rel = p - np.asarray(center_xy, dtype=float)
    rho = np.linalg.norm(rel, axis=2)
    # flux density P . n  with  P = S(rho)/rho^2 * rel
    fac = np.where(rho > 1e-12, _S(rho, r, frac) / np.maximum(rho, 1e-12) ** 2, 0.5)
    Pn = fac * np.sum(rel * nrm[:, None, :], axis=2)               # (E, G)
    w = wg[None, :] * L[:, None]
    I = float(np.sum(Pn * w))

    s = s_profile(rho, r, frac)                                    # (E, G)
    # hat functions: vertex a (start of edge) 1 - t, vertex b (end) t
    start_c = np.sum((s * (1 - tg)[None, :] * w)[:, :, None] * nrm[:, None, :], axis=1)  # (E, 2)
    end_c = np.sum((s * tg[None, :] * w)[:, :, None] * nrm[:, None, :], axis=1)
    grad = np.zeros_like(verts)
    np.add.at(grad[:, :2], idx, start_c)
    np.add.at(grad[:, :2], np.roll(idx, -1), end_c)
    return I, grad


def anomaly_wall_energy_and_grad(verts, ring, anomaly_cfg, gamma, kappa0, want_grad=True):
    """(E_anom, grad) of the anomaly term (see module docstring). Returns
    (0, zeros) if the anomaly is disabled or has no contrast."""
    if anomaly_cfg is None or not getattr(anomaly_cfg, "enabled", False):
        return 0.0, np.zeros_like(verts)
    dk = np.cos(np.radians(anomaly_cfg.contact_angle_deg)) - kappa0
    if abs(dk) < 1e-12:
        return 0.0, np.zeros_like(verts)
    I, g = disc_integral_and_grad(verts, ring, anomaly_cfg.center_xy,
                                  anomaly_cfg.radius, anomaly_cfg.transition_fraction)
    return -gamma * dk * I, -gamma * dk * g
