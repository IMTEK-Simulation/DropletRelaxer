"""Sphere wetting (solid angle), sphere penalty, wetted-set detection, anomaly."""
import numpy as np
import pytest
from types import SimpleNamespace

from conftest import fd_gradient, rel_err
from wetting import (patch_solid_angle, sphere_wetting_energy_and_grad, compute_wetted_set,
                     wetted_boundary_loop)
from obstacle import sphere_contact_energy_and_grad
from anomaly import (s_profile, disc_integral_and_grad, anomaly_wall_energy_and_grad)


# ------------------------------------------------------------------ sphere
def _cap_loop(alpha, n, R=1.0, c=(0.0, 0.0, 0.0)):
    ph = np.linspace(0, 2 * np.pi, n, endpoint=False)
    u = np.c_[np.sin(alpha) * np.cos(ph), np.sin(alpha) * np.sin(ph), np.cos(alpha) * np.ones(n)]
    return np.asarray(c) + R * u


@pytest.mark.parametrize("alpha_deg", [10, 30, 60, 90, 120])
def test_solid_angle_of_a_spherical_cap(alpha_deg):
    a = np.radians(alpha_deg)
    u = _cap_loop(a, 400)
    # counter-clockwise seen from outside around the +z axis
    omega = patch_solid_angle(u)
    assert omega == pytest.approx(2 * np.pi * (1 - np.cos(a)), rel=2e-3)


def test_solid_angle_is_rotation_invariant(rng):
    u = _cap_loop(np.radians(40), 120)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    assert patch_solid_angle(u @ q.T) == pytest.approx(patch_solid_angle(u), rel=1e-9)


def test_sphere_wetting_energy_value_and_gradient(rng):
    R, c = 100.0, np.array([5.0, -3.0, 120.0])
    loop = _cap_loop(np.radians(25), 60, R, c)
    loop = loop + 0.02 * rng.standard_normal(loop.shape)      # radial and tangential noise
    idx = np.arange(60)
    kappa = np.cos(np.radians(120.0))
    E, g = sphere_wetting_energy_and_grad(loop, idx, c, R, kappa)
    u = (loop - c) / np.linalg.norm(loop - c, axis=1, keepdims=True)
    assert E == pytest.approx(-kappa * R ** 2 * patch_solid_angle(u), rel=1e-12)
    gfd = fd_gradient(lambda x: sphere_wetting_energy_and_grad(x, idx, c, R, kappa)[0], loop, h=1e-4 * 1.0)
    assert rel_err(g, gfd) < 1e-4
    # the energy depends on directions only, so the force is tangential to the sphere
    radial = np.sum(g * (loop - c), axis=1) / np.linalg.norm(loop - c, axis=1)
    assert np.max(np.abs(radial)) < 1e-6 * np.max(np.linalg.norm(g, axis=1))


def test_sphere_wetting_energy_is_zero_for_90_degrees():
    loop = _cap_loop(np.radians(25), 40, 10.0)
    E, g = sphere_wetting_energy_and_grad(loop, np.arange(40), np.zeros(3), 10.0, 0.0)
    assert E == 0.0 and np.all(g == 0.0)


def test_sphere_penalty_gradient_and_one_sidedness(rng):
    c, R = np.array([0.0, 0.0, 0.0]), 1.0
    d = rng.standard_normal((30, 3))
    v = c + (R + 0.3 * rng.standard_normal(30))[:, None] * d / np.linalg.norm(d, axis=1, keepdims=True)
    E, g = sphere_contact_energy_and_grad(v, c, R, 80.0, 0.0, 0.25)
    gfd = fd_gradient(lambda x: sphere_contact_energy_and_grad(x, c, R, 80.0, 0.0, 0.25)[0], v, h=1e-6)
    assert rel_err(g, gfd) < 1e-6
    gap = np.linalg.norm(v - c, axis=1) - R
    assert E == pytest.approx(80.0 * np.sum(np.minimum(gap, 0) ** 2))
    # adhesion on: the pull-in acts only inside the capture radius
    E2, _ = sphere_contact_energy_and_grad(v, c, R, 80.0, 5.0, 0.25)
    inside = (gap >= 0) & (gap < 0.25)
    assert E2 == pytest.approx(E + 5.0 * np.sum(gap[inside] ** 2))


def test_wetted_set_and_boundary_loop_of_a_patch():
    """A flat-ish triangulated patch lying exactly on a sphere: detection must
    return its boundary loop, counter-clockwise seen from outside."""
    R = 10.0
    n_r, n_p = 5, 24
    pts, tris = [np.array([0.0, 0.0, R])], []
    a_max = np.radians(20)
    for i in range(1, n_r + 1):
        a = a_max * i / n_r
        for j in range(n_p):
            ph = 2 * np.pi * j / n_p
            pts.append(R * np.array([np.sin(a) * np.cos(ph), np.sin(a) * np.sin(ph), np.cos(a)]))
    ring = lambda i, j: 1 + (i - 1) * n_p + (j % n_p)
    for j in range(n_p):
        tris.append([0, ring(1, j), ring(1, j + 1)])
    for i in range(1, n_r):
        for j in range(n_p):
            tris.append([ring(i, j), ring(i + 1, j), ring(i + 1, j + 1)])
            tris.append([ring(i, j), ring(i + 1, j + 1), ring(i, j + 1)])
    v, t = np.array(pts), np.array(tris)
    # add a non-wetted tail so the patch has a boundary inside the mesh
    far = np.array([[0.0, 0.0, R + 5.0]])
    c = np.zeros(3)
    mask = compute_wetted_set(v, t, c, R, 1e-6)
    assert mask.all()
    # shrink the wetted set to the inner rings: loop must be ring 3
    v2 = v.copy()
    v2[np.arange(1 + 3 * n_p, len(v))] *= 1.2      # lift the outer rings off the sphere
    mask2 = compute_wetted_set(v2, t, c, R, 1e-6)
    assert mask2.sum() == 1 + 3 * n_p
    loop = wetted_boundary_loop(mask2, t, v2, c)
    assert loop is not None and len(loop) == n_p
    assert set(loop) == set(range(1 + 2 * n_p, 1 + 3 * n_p))
    u = v2[loop] / R
    assert patch_solid_angle(u) == pytest.approx(2 * np.pi * (1 - np.cos(3 * a_max / n_r)), rel=2e-2)


# ------------------------------------------------------------------ anomaly
def _polygon(n=80, r=1.0, center=(0.0, 0.0), rng=None, jitter=0.0):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rr = r * (1 + (jitter * rng.standard_normal(n) if rng is not None else 0.0))
    return np.c_[center[0] + rr * np.cos(th), center[1] + rr * np.sin(th), np.zeros(n)]


def _shoelace(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def test_profile_properties():
    r, frac = 20.0, 0.1
    h = 0.5 * frac * r
    assert s_profile(np.array([0.0, r - h - 1e-9]), r, frac) == pytest.approx([1.0, 1.0])
    assert s_profile(np.array([r + h, r + 5.0]), r, frac) == pytest.approx([0.0, 0.0])
    assert s_profile(np.array([r]), r, frac)[0] == pytest.approx(0.5)
    rho = np.linspace(r - h, r + h, 401)
    s = s_profile(rho, r, frac)
    assert np.all(np.diff(s) <= 1e-12)                    # monotone decreasing
    # C2: s'' equals the analytic second derivative -q''(u)/(2h)^2 of the smootherstep
    # and vanishes at both ends of the transition
    u = (rho - (r - h)) / (2 * h)
    d2_exact = -(60 * u - 180 * u ** 2 + 120 * u ** 3) / (2 * h) ** 2
    d2 = np.gradient(np.gradient(s, rho), rho)
    assert np.max(np.abs(d2[3:-3] - d2_exact[3:-3])) < 2e-3 * np.max(np.abs(d2_exact))
    assert abs(d2_exact[0]) < 1e-12 and abs(d2_exact[-1]) < 1e-12


def test_polygon_inside_the_disc_gives_its_area_and_shoelace_gradient(rng):
    v = _polygon(60, r=1.0, center=(5.0, 1.0), rng=rng, jitter=0.03)
    ring = np.arange(60)
    I, g = disc_integral_and_grad(v, ring, (5.0, 1.0), 10.0, 0.1)
    assert I == pytest.approx(_shoelace(v), rel=1e-10)
    # same gradient as the shoelace area
    x, y = v[:, 0], v[:, 1]
    gs = np.zeros_like(v)
    gs[:, 0] = 0.5 * (np.roll(y, -1) - np.roll(y, 1))
    gs[:, 1] = 0.5 * (np.roll(x, 1) - np.roll(x, -1))
    assert rel_err(g, gs) < 1e-9


def test_polygon_outside_the_disc_gives_zero(rng):
    v = _polygon(60, r=1.0, center=(50.0, 0.0), rng=rng, jitter=0.03)
    I, g = disc_integral_and_grad(v, np.arange(60), (0.0, 0.0), 10.0, 0.1)
    assert abs(I) < 1e-12 and np.max(np.abs(g)) < 1e-12


@pytest.mark.parametrize("frac", [0.05, 0.1, 0.4])
def test_disc_integral_against_brute_force_quadrature(frac, rng):
    r, cxy = 3.0, np.array([0.7, 0.2])
    v = _polygon(90, r=3.4, center=(0.0, 0.0), rng=rng, jitter=0.02)   # polygon straddling the disc edge
    I, _ = disc_integral_and_grad(v, np.arange(90), cxy, r, frac)
    n = 1600
    xs = np.linspace(-4.5, 4.5, n)
    h = xs[1] - xs[0]
    X, Y = np.meshgrid(xs, xs)
    # point in polygon by ray casting
    px, py = v[:, 0], v[:, 1]
    inside = np.zeros_like(X, dtype=bool)
    j = len(px) - 1
    for i in range(len(px)):
        cond = ((py[i] > Y) != (py[j] > Y)) & (X < (px[j] - px[i]) * (Y - py[i]) / (py[j] - py[i] + 1e-300) + px[i])
        inside ^= cond
        j = i
    s = s_profile(np.hypot(X - cxy[0], Y - cxy[1]), r, frac)
    ref = np.sum(s * inside) * h * h
    assert I == pytest.approx(ref, rel=3e-3)


@pytest.mark.parametrize("frac", [0.1, 0.4])
def test_disc_integral_gradient_matches_finite_differences(frac, rng):
    v = _polygon(40, r=3.4, rng=rng, jitter=0.03)
    ring = np.arange(40)
    cxy = (0.7, 0.2)
    I, g = disc_integral_and_grad(v, ring, cxy, 3.0, frac)
    idx = [3 * i + k for i in range(0, 40, 3) for k in (0, 1)]
    gfd = fd_gradient(lambda x: disc_integral_and_grad(x, ring, cxy, 3.0, frac)[0], v, h=1e-5, idx=idx)
    sel = np.zeros(v.size, bool)
    sel[idx] = True
    assert rel_err(g.reshape(-1)[sel], gfd.reshape(-1)[sel]) < 1e-4     # default node count
    # the remaining difference is quadrature error: it vanishes with more Gauss nodes
    g96 = disc_integral_and_grad(v, ring, cxy, 3.0, frac, n_gauss=96)[1]
    gfd96 = fd_gradient(lambda x: disc_integral_and_grad(x, ring, cxy, 3.0, frac, n_gauss=96)[0], v, h=1e-5, idx=idx)
    assert rel_err(g96.reshape(-1)[sel], gfd96.reshape(-1)[sel]) < 1e-7


def test_anomaly_energy_formula_and_gradient(rng):
    cfg = SimpleNamespace(enabled=True, center_xy=(0.7, 0.2), radius=3.0, contact_angle_deg=120.0,
                          transition_fraction=0.1)
    v = _polygon(50, r=3.3, rng=rng, jitter=0.02)
    gamma, kappa0 = 0.88, np.cos(np.radians(90.0))
    E, g = anomaly_wall_energy_and_grad(v, np.arange(50), cfg, gamma, kappa0)
    I = disc_integral_and_grad(v, np.arange(50), cfg.center_xy, cfg.radius, cfg.transition_fraction)[0]
    assert E == pytest.approx(-gamma * (np.cos(np.radians(120.0)) - kappa0) * I, rel=1e-12)
    # same contact angle as the wall: no contrast, no energy
    cfg.contact_angle_deg = 90.0
    E0, g0 = anomaly_wall_energy_and_grad(v, np.arange(50), cfg, gamma, kappa0)
    assert E0 == 0.0 and np.all(g0 == 0.0)
    cfg.enabled = False
    assert anomaly_wall_energy_and_grad(v, np.arange(50), cfg, gamma, kappa0)[0] == 0.0
