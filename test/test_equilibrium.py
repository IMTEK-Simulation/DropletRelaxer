"""SLOW: real relaxations on a fixed mesh, compared with analytic geometry.

A droplet on a homogeneous wall must become a spherical cap with the Young
angle. The cap with volume V and angle theta has R = (3V / (pi (2+cos)(1-cos)^2))^(1/3),
contact radius a = R sin(theta), centre height -R cos(theta). Measured tolerances
(2000-8000 iterations, ~600 vertices, no remeshing): contact radius within
0.6 %, sphericity <0.9 %, fitted contact angle within ~0.5 deg."""
import numpy as np
import pytest

import DropletRelaxer.analysis as M
from runhelpers import (static_cfg, scaled_setup, run_solver, cap_prediction, refine_global)
from DropletRelaxer.solver import relax
from DropletRelaxer.substrate import base_triangle_mask

pytestmark = pytest.mark.slow


def _dome_vertices(v, tris, bidx):
    m = base_triangle_mask(v, tris, bidx)
    used = np.zeros(len(v), bool)
    used[tris[~m].ravel()] = True
    return v[used]


def _fit_sphere(p):
    A = np.c_[2 * p, np.ones(len(p))]
    b = np.sum(p ** 2, axis=1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    c = sol[:3]
    R = np.sqrt(sol[3] + c @ c)
    return c, R


@pytest.fixture(scope="module", params=[60.0, 90.0, 120.0], ids=lambda a: f"{a:g}deg")
def cap(request):
    ang = request.param
    cfg = static_cfg(ang, False, 4000)
    v, t, b, mask, V0 = scaled_setup(cfg)
    last, _ = run_solver(cfg, v, t, b, mask, V0, 4000)
    return ang, cfg, last, t, b, V0


def test_static_droplet_is_a_spherical_cap_with_the_young_angle(cap):
    ang, cfg, last, t, b, V0 = cap
    i, v, A, V, _, _ = last
    R, a, h, zc = cap_prediction(V, ang)           # prediction for the ACTUAL (penalty-reduced) volume
    a_meas = np.hypot(v[b, 0], v[b, 1]).mean()
    assert a_meas == pytest.approx(a, rel=1.5e-2)
    assert v[:, 2].max() == pytest.approx(h, rel=1.5e-2)
    p = _dome_vertices(v, t, b)
    d = np.linalg.norm(p - np.array([0.0, 0.0, zc]), axis=1)
    assert (d.max() - d.min()) / R < 2e-2                       # all dome vertices on one sphere
    c, Rfit = _fit_sphere(p)
    assert abs(c[0]) < 0.01 * R and abs(c[1]) < 0.01 * R         # axisymmetric
    ang_fit = np.degrees(np.arccos(np.clip(-c[2] / Rfit, -1, 1)))
    assert ang_fit == pytest.approx(ang, abs=1.5)
    assert abs(V - V0) / V0 < 0.05                               # soft volume penalty: a few percent


def test_young_laplace_pressure_is_constant_mean_curvature(cap):
    ang, cfg, last, t, b, V0 = cap
    i, v, A, V, _, _ = last
    from DropletRelaxer.energy import mean_curvature
    H = mean_curvature(v, t, b)
    m = base_triangle_mask(v, t, b)
    interior = np.zeros(len(v), bool)
    interior[t[~m].ravel()] = True
    interior[b] = False
    R, a, h, zc = cap_prediction(V, ang)
    Hi = H[interior]
    assert np.median(Hi) == pytest.approx(1.0 / R, rel=0.05)     # H = 1/R for a sphere
    assert np.percentile(np.abs(Hi - 1.0 / R), 90) < 0.15 / R


def test_remeshing_in_the_middle_keeps_the_equilibrium():
    cfg = static_cfg(120.0, False, 5000)
    v, t, b, mask, V0 = scaled_setup(cfg)
    last, _ = run_solver(cfg, v, t, b, mask, V0, 2500)
    v, t, b, mask = refine_global(last[1], t, b)
    last, _ = run_solver(cfg, v, t, b, mask, V0, 2500, start_iter=2500)
    i, v, A, V, _, _ = last
    R, a, h, zc = cap_prediction(V, 120.0)
    assert np.hypot(v[b, 0], v[b, 1]).mean() == pytest.approx(a, rel=1.5e-2)


def test_equilibrium_does_not_depend_on_the_numerical_parameters():
    """Sobolev alpha, learning rate and trust region change the path, not the minimizer."""
    results = []
    for kw, n in [({}, 4000), ({"sobolev_alpha": 1.0}, 4000), ({"sobolev_alpha": 8.0}, 4000),
                  ({"max_step_fraction": 0.1}, 4000), ({"learning_rate": 0.1}, 9000)]:
        cfg = static_cfg(120.0, False, n)
        for k, val in kw.items():
            setattr(cfg.solver, k, val)
        v, t, b, mask, V0 = scaled_setup(cfg)
        last, _ = run_solver(cfg, v, t, b, mask, V0, n)
        i, vv, A, V, _, _ = last
        results.append((np.hypot(vv[b, 0], vv[b, 1]).mean(), M.surface_free_energy(cfg, vv, t, b, A, None, None)))
    a, E = np.array(results).T
    assert np.ptp(a) / a.mean() < 3e-3
    assert np.ptp(E) / E.mean() < 3e-4


def test_energy_decreases_monotonically_during_relaxation():
    cfg = static_cfg(120.0, False, 3000)
    v, t, b, mask, V0 = scaled_setup(cfg)
    k = cfg.solver.volume_penalty
    es = []
    for out in relax(v.copy(), t, b, mask, cfg.solver, None, start_iter=0, stop_iter=3000,
                     V_target=V0, anomaly_cfg=cfg.anomaly):
        i, vv, A, V, _, _ = out
        if i % 10 == 0:
            es.append(M.surface_free_energy(cfg, vv, t, b, A, None, None) + k * (V - V0) ** 2)
    es = np.array(es)
    assert es[-1] < es[0]
    # tangential smoothing every 10 iterations is not a descent step: allow tiny rises only
    assert np.max(np.diff(es)) < 2e-6 * abs(es[0])


def test_volume_deficit_scales_like_inverse_penalty_weight():
    """The quadratic penalty leaves a volume deficit proportional to 1/k."""
    deficits = []
    for k in (50.0, 100.0):
        cfg = static_cfg(120.0, False, 8000)
        cfg.solver.learning_rate = 0.12                       # stiffer penalty needs a smaller step
        cfg.solver.volume_penalty = k
        v, t, b, mask, V0 = scaled_setup(cfg)
        last, _ = run_solver(cfg, v, t, b, mask, V0, 8000)
        deficits.append((V0 - last[3]) / V0)
    assert deficits[0] / deficits[1] == pytest.approx(2.0, rel=0.1)
