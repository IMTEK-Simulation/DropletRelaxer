"""roughness.py: spectrum, statistics, exact contact-line integral and gradient."""
import math

import numpy as np
import pytest

from DropletRelaxer.config import RoughnessConfig
from DropletRelaxer.roughness import make_roughness_field

KAPPA0 = math.cos(math.radians(60.0))


def _loop(n=120, c=(3.0, -2.0), R=12.0):
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rad = R * (1 + 0.15 * np.sin(3 * phi) + 0.05 * np.cos(5 * phi))
    v = np.zeros((n + 5, 3))
    v[:n, 0] = c[0] + rad * np.cos(phi)
    v[:n, 1] = c[1] + rad * np.sin(phi)
    return v, np.arange(n)


def _field(mode="topography", **kw):
    cfg = RoughnessConfig(enabled=True, mode=mode, box_size=128.0, seed=5, **kw)
    return make_roughness_field(cfg, 1.0, KAPPA0)


def _inside(poly, X, Y):
    """Even-odd ray casting."""
    ins = np.zeros(X.shape, bool)
    px, py = poly[:, 0], poly[:, 1]
    for k in range(len(px)):
        x1, y1, x2, y2 = px[k], py[k], px[(k + 1) % len(px)], py[(k + 1) % len(px)]
        cond = (y1 > Y) != (y2 > Y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = x1 + (Y - y1) * (x2 - x1) / (y2 - y1)
        ins ^= cond & (X < xc)
    return ins


def test_spectrum_band_limited_and_normalised():
    f = _field(rms_slope=0.3)
    H = np.fft.fft2(f.h)
    q = 2 * np.pi * np.fft.fftfreq(f.N, d=f.dx)
    Q = np.hypot(*np.meshgrid(q, q))
    P = np.abs(H) ** 2
    assert P[Q > 1.001 * f.q_c].max() < 1e-12 * P.max()      # hard cutoff
    assert P[Q < 0.999 * f.q_L].max() < 1e-12 * P.max()      # no q = 0 / below q_L
    assert abs(f.info["rms_slope"] - 0.3) < 1e-9             # normalize_sample
    sh_theory = 0.3 * math.sqrt(2.0 / (f.q_c ** 2 + f.q_L ** 2))
    assert abs(f.info["rms_h"] / sh_theory - 1) < 0.1


def test_plateau_flat():
    f = _field(rms_slope=0.3)
    C = (f.dx ** 2 * np.abs(np.fft.fft2(f.h))) ** 2 / f.L ** 2
    q = 2 * np.pi * np.fft.fftfreq(f.N, d=f.dx)
    Q = np.hypot(*np.meshgrid(q, q))
    band = (Q > 1.5 * f.q_L) & (Q < 0.97 * f.q_c)
    # exponential (chi^2_2) statistics per mode: the mean over the band equals C0
    assert abs(C[band].mean() / f.info["C0"] - 1) < 0.05


@pytest.mark.parametrize("mode,kw", [("topography", dict(rms_slope=0.5)), ("chemical", dict(rms_kappa=0.08))])
def test_gradient_is_derivative_of_energy(mode, kw, rng):
    f = _field(mode, **kw)
    v, loop = _loop()
    E, g = f.energy_and_grad(v, loop, 0.88)
    d = rng.standard_normal(v.shape); d[:, 2] = 0; d[len(loop):] = 0
    d /= np.linalg.norm(d)
    eps = 1e-5
    fd = (f.energy_and_grad(v + eps * d, loop, 0.88)[0] - f.energy_and_grad(v - eps * d, loop, 0.88)[0]) / (2 * eps)
    assert abs(np.sum(g * d) - fd) < 1e-6 * abs(fd)
    assert np.all(g[:, 2] == 0) and np.all(g[len(loop):] == 0)


def test_orientation_invariance():
    f = _field(rms_slope=0.5)
    v, loop = _loop()
    E, g = f.energy_and_grad(v, loop, 0.88)
    Er, gr = f.energy_and_grad(v, loop[::-1], 0.88)
    assert abs(E - Er) < 1e-10 and np.abs(g - gr).max() < 1e-10


def test_integral_vs_brute_force():
    f = _field(rms_slope=0.5)
    v, loop = _loop()
    xs = np.linspace(-12, 20, 1600); ys = np.linspace(-18, 14, 1600)
    X, Y = np.meshgrid(xs, ys)
    ins = _inside(v[loop, :2], X, Y)
    I_bf = np.sum(f.delta_kappa(X, Y) * ins) * (xs[1] - xs[0]) * (ys[1] - ys[0])
    I_ex, _ = f.wetted_integral_and_grad(v, loop)
    assert abs(I_ex - I_bf) < 1e-3 * abs(I_bf)


def test_energy_sign_and_gamma_scaling():
    f = _field(rms_slope=0.4)
    v, loop = _loop()
    I, _ = f.wetted_integral_and_grad(v, loop)
    E, _ = f.energy_and_grad(v, loop, 2.5)
    assert E == pytest.approx(-2.5 * I, rel=1e-12)


def test_topography_is_wenzel_and_neutral_at_90deg():
    f = _field(rms_slope=0.4)                 # theta_Y = 60: r >= 1 pushes kappa up (clipped at 1)
    assert np.all(f.kappa_eff >= KAPPA0 - 1e-12)
    cfg = RoughnessConfig(enabled=True, mode="topography", rms_slope=0.4, box_size=128.0)
    f90 = make_roughness_field(cfg, 1.0, 0.0)
    assert np.abs(f90.kappa_eff).max() == 0.0


def test_clipping_reported():
    f = _field("chemical", rms_kappa=0.6)
    assert f.kappa_eff.max() <= 1.0 and f.kappa_eff.min() >= -1.0
    assert f.info["clipped"] > 0


def test_resolution_message_warns_when_too_coarse():
    f = _field(rms_slope=0.3)
    v, loop = _loop(n=12)                      # edges ~ 6 >> lambda_c / 2 = 2
    assert "WARNING" in f.check_resolution(v, loop)
    assert "WARNING" not in f.check_resolution(*_loop(n=2000))


def test_invalid_mode_and_cutoff():
    with pytest.raises(ValueError):
        make_roughness_field(RoughnessConfig(mode="nope"), 1.0, KAPPA0)
    with pytest.raises(ValueError):
        make_roughness_field(RoughnessConfig(box_size=4.0), 1.0, KAPPA0)


def test_surface_free_energy_contains_roughness_term():
    import DropletRelaxer.analysis as M
    from runhelpers import fresh_config, scaled_setup
    cfg = fresh_config(); cfg.anomaly.enabled = False
    verts, tris, b, mask = scaled_setup(cfg)[:4]
    A = 1.0
    E0 = M.surface_free_energy(cfg, verts, tris, b, A)
    cfg.roughness_field = make_roughness_field(
        RoughnessConfig(enabled=True, mode="chemical", rms_kappa=0.1, box_size=400.0), 5.0, KAPPA0)
    E1 = M.surface_free_energy(cfg, verts, tris, b, A)
    assert E1 - E0 == pytest.approx(cfg.roughness_field.energy_and_grad(verts, b, cfg.solver.gamma_lv)[0])
    assert E1 != E0


def test_solver_uses_roughness_term():
    from runhelpers import static_cfg, scaled_setup, run_solver
    cfg = static_cfg(60.0)
    verts, tris, b, mask, V0 = scaled_setup(cfg)
    last0, _ = run_solver(cfg, verts, tris, b, mask, V0, 200)
    ring = verts[b, :2]
    edge = float(np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1).mean())
    cfg.roughness_field = make_roughness_field(
        RoughnessConfig(enabled=True, mode="chemical", rms_kappa=0.3, box_size=400.0), edge, KAPPA0)
    last1, diag = run_solver(cfg, verts, tris, b, mask, V0, 200)
    v0, v1 = last0[1], last1[1]
    assert np.all(np.isfinite(v1))
    assert np.abs(v1[b, :2] - v0[b, :2]).max() > 1e-3        # the contact line feels the disorder
