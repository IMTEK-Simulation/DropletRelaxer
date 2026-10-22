"""SLOW: force on the sphere against the numerical derivative of the equilibrium energy.

F = -dE*/dc is computed inside every iteration from force balance; here it is
compared with central differences of the total interface energy of
RE-EQUILIBRATED states at z0 +- 2 (warm start, one global refinement, 2500
iterations each). Measured agreement: 0.04 % (wetting angle 90 deg), 4-6 %
(120 deg, where the contact line carries more weight)."""
import numpy as np
import pytest

import DropletRelaxer.analysis as M
from runhelpers import static_cfg, scaled_setup, refine_global
from DropletRelaxer.solver import relax

pytestmark = pytest.mark.slow


def _stage(cfg, v, t, b, mask, V0, c, n):
    cfg.obstacle.center_start = c
    cfg.obstacle.center_end = c
    diag, forces, last = {}, [], None
    for out in relax(v.copy(), t, b, mask, cfg.solver, cfg.obstacle, start_iter=0, stop_iter=n, V_target=V0,
                     diag=diag, anomaly_cfg=cfg.anomaly):
        last = out
        if out[0] > n - 200:
            forces.append(diag["force"].copy())
    i, vv, A, V, sc, wl = last
    E = M.surface_free_energy(cfg, vv, t, b, A, sc, wl)
    f = np.array(forces)
    return vv, E, f.mean(axis=0), np.ptp(f, axis=0).max()


@pytest.mark.parametrize("angle,z0,tol", [(90.0, 122.0, 0.03), (120.0, 128.0, 0.10)], ids=["90deg", "120deg"])
def test_force_equals_minus_energy_derivative(angle, z0, tol):
    cfg = static_cfg(angle, True, 2500)
    cfg.obstacle.wetting_contact_angle_deg = angle
    v, t, b, mask, V0 = scaled_setup(cfg)
    v, t, b, mask = refine_global(v, t, b)
    v, _, _, _ = _stage(cfg, v, t, b, mask, V0, (0.0, 0.0, z0 + 30.0), 1500)     # settle the droplet first
    states = {}
    cur = v
    for dz in (2.0, 0.0, -2.0):
        cur, E, F, drift = _stage(cfg, cur, t, b, mask, V0, (0.0, 0.0, z0 + dz), 2500)
        states[dz] = (E, F, drift)
        assert drift < 0.05 * abs(F[2])                       # equilibrated: the force no longer changes
    dEdz = (states[2.0][0] - states[-2.0][0]) / 4.0
    Fz = states[0.0][1][2]
    assert Fz == pytest.approx(-dEdz, rel=tol)
    # homogeneous wall + sphere on the axis: no lateral force by symmetry
    Fxy = states[0.0][1][:2]
    assert np.max(np.abs(Fxy)) < 2e-3 * abs(Fz)
