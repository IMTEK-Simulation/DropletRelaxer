"""Sphere path and event-driven (quasi-static) stepping."""
import numpy as np
import pytest
from types import SimpleNamespace

from obstacle import sphere_center_at, path_length
from quasistatic import QuasiStatic


def _ocfg(**kw):
    d = dict(center_start=(-40.0, 0.0, 125.0), center_end=(100.0, 0.0, 125.0))
    d.update(kw)
    return SimpleNamespace(**d)


def _qcfg(**kw):
    d = dict(step=1.0, adaptive=False, min_step_factor=0.125, min_iter_per_step=10,
             max_iter_per_step=100, check_every=10, tol_energy=1e-6, tol_force=1e-6, tol_step=1e-6)
    d.update(kw)
    return SimpleNamespace(**d)


def test_straight_path_is_linear_and_clamped():
    c = _ocfg()
    for f in (0.0, 0.25, 0.5, 1.0):
        assert sphere_center_at(c, f) == pytest.approx(np.array([-40.0 + 140.0 * f, 0.0, 125.0]))
    assert sphere_center_at(c, -3.0)[0] == pytest.approx(-40.0)
    assert sphere_center_at(c, 7.0)[0] == pytest.approx(100.0)
    assert path_length(c) == pytest.approx(140.0)
    assert path_length(_ocfg(center_end=(-40.0, 30.0, 165.0))) == pytest.approx(50.0)


def _drive(q, E_of, F_of, n, verts=None):
    """Feeds the convergence logic with scripted E and F for n iterations."""
    verts = np.zeros((3, 3)) if verts is None else verts
    out = []
    for k in range(n):
        if q.converged_now:
            q.advance(verts)
            if q.finished:
                break
        q.energy_fn = lambda v, A, c, w, k=k: E_of(q.n_it)
        st = q.observe(verts, F_of(q.n_it), 0.0, False, 0.0, None, None)
        out.append(st)
        if st == "rollback":
            q.rollback()
    return out


def test_converges_only_when_energy_and_force_are_stationary():
    q = QuasiStatic(_qcfg(), _ocfg(), 1.0, 1000.0)
    # E keeps changing -> never converged within the budget but accepted at the limit
    _drive(q, lambda n: 1.0 + 1e-3 * n, lambda n: np.zeros(3), 100)
    assert q.frames == 1 and q.unconverged == 1 and q.frame_info["step_iters"] == 100
    q2 = QuasiStatic(_qcfg(), _ocfg(), 1.0, 1000.0)
    _drive(q2, lambda n: 1.0, lambda n: np.zeros(3), 20)
    assert q2.frames >= 1 and q2.unconverged == 0 and q2.frame_info["converged"]
    assert q2.frame_info["step_iters"] == 20       # test at n=10 has no reference, at n=20 it passes
    # force still drifting while the energy is flat -> not converged
    q3 = QuasiStatic(_qcfg(max_iter_per_step=10 ** 6), _ocfg(), 1.0, 1000.0)
    _drive(q3, lambda n: 1.0, lambda n: np.array([1e-2 * n, 0, 0]), 200)
    assert q3.frames == 0


def test_path_advances_by_step_and_finishes_at_the_end():
    q = QuasiStatic(_qcfg(step=1.0), _ocfg(center_start=(0.0, 0, 125.0), center_end=(2.5, 0, 125.0)), 1.0, 1000.0)
    xs = []
    for _ in range(200):
        _drive(q, lambda n: 1.0, lambda n: np.zeros(3), 1)
        if q.converged_now:
            xs.append(q.center()[0])
        if q.finished:
            break
        if q.converged_now:
            q.advance(np.zeros((3, 3)))
            if q.finished:
                break
    assert xs == pytest.approx([0.0, 1.0, 2.0, 2.5])      # last step clipped to the path end
    assert q.progress() == pytest.approx(1.0)
    z = QuasiStatic(_qcfg(), _ocfg(center_end=(-40.0, 0.0, 125.0)), 1.0, 1000.0)
    _drive(z, lambda n: 1.0, lambda n: np.zeros(3), 30)
    z.advance(np.zeros((3, 3)))
    assert z.finished                                        # zero-length path: one equilibrium


def test_adaptive_rolls_back_with_half_step_and_grows_again():
    q = QuasiStatic(_qcfg(adaptive=True, step=1.0, max_iter_per_step=90),
                    _ocfg(center_start=(0.0, 0, 125.0), center_end=(10.0, 0, 125.0)), 1.0, 1000.0)
    _drive(q, lambda n: 1.0, lambda n: np.zeros(3), 20)      # first frame at u = 0
    q.advance(np.zeros((3, 3)))
    assert q.center()[0] == pytest.approx(1.0)
    # this step never converges -> rollback with half the step
    st = _drive(q, lambda n: 1.0 + 1e-2 * n, lambda n: np.zeros(3), 90)
    assert "rollback" in st and q.rollbacks == 1
    assert q.delta == pytest.approx(0.5) and q.center()[0] == pytest.approx(0.5)
    # converges quickly -> frame at 0.5, next step doubles back to 1.0
    _drive(q, lambda n: 1.0, lambda n: np.zeros(3), 20)
    assert q.converged_now and q.frame_info["converged"]
    q.advance(np.zeros((3, 3)))
    assert q.delta == pytest.approx(1.0)


def test_adaptive_accepts_unconverged_step_at_the_smallest_step():
    q = QuasiStatic(_qcfg(adaptive=True, step=1.0, min_step_factor=0.5, max_iter_per_step=30),
                    _ocfg(center_start=(0.0, 0, 125.0), center_end=(10.0, 0, 125.0)), 1.0, 1000.0)
    _drive(q, lambda n: 1.0, lambda n: np.zeros(3), 20)
    q.advance(np.zeros((3, 3)))
    for _ in range(3):
        _drive(q, lambda n: 1.0 + 1e-2 * n, lambda n: np.zeros(3), 30)
        if q.converged_now:
            break
    assert q.converged_now and not q.frame_info["converged"] and q.unconverged == 1
    assert q.delta == pytest.approx(0.5)


def test_remesh_drops_rollback_state_but_keeps_the_convergence_window():
    q = QuasiStatic(_qcfg(adaptive=True), _ocfg(), 1.0, 1000.0)
    _drive(q, lambda n: 1.0, lambda n: np.zeros(3), 20)
    q.advance(np.zeros((3, 3)))
    assert q.ckpt is not None
    _drive(q, lambda n: 1.0, lambda n: np.zeros(3), 10)
    assert q.prev_E is not None
    q.on_remesh()
    assert q.ckpt is None and q.prev_E is not None
    q.reset_window()
    assert q.prev_E is None


def test_solver_steps_the_sphere_only_after_convergence():
    import copy
    import main as M
    from solver import relax
    from tests.runhelpers import static_cfg, scaled_setup
    cfg = static_cfg(90.0, sphere=True)
    cfg.obstacle.center_start = (0.0, 0.0, 125.0)
    cfg.obstacle.center_end = (2.0, 0.0, 125.0)
    cfg.quasi.step, cfg.quasi.min_iter_per_step, cfg.quasi.check_every = 1.0, 10, 10
    cfg.quasi.tol_energy = cfg.quasi.tol_force = cfg.quasi.tol_step = 1e9
    verts, tris, b, mask, V0 = scaled_setup(cfg)
    q = QuasiStatic(cfg.quasi, cfg.obstacle, cfg.solver.gamma_lv, V0)
    q.energy_fn = lambda v, A, c, w: M.surface_free_energy(cfg, v, tris, b, A, c, w)
    frames = []
    for i, v, A, V, c, w in relax(verts.copy(), tris, b, mask, cfg.solver, cfg.obstacle, start_iter=0,
                                  stop_iter=1000, V_target=V0, diag={}, quasi=q):
        if q.converged_now:
            frames.append((i, c[0]))
    assert [x for _, x in frames] == pytest.approx([0.0, 1.0, 2.0])
    assert [i for i, _ in frames] == [20, 40, 60]          # 20 iterations per step
    assert q.finished
