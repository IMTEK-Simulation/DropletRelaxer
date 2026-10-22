"""
Event-driven (quasi-static) motion of the sphere.

The sphere stands still at path position u (0..1 along center_start ->
center_end) until the liquid bridge has relaxed. Then a frame is recorded
(`converged_now`), the sphere advances by one step and the relaxation continues
from the relaxed state (warm start).

Convergence test (every `check_every` iterations of a step, at least
`min_iter_per_step` iterations): all three of

    |E - E_prev|            / (gamma_lv V0^(2/3)) < tol_energy
    max_k |F_k - F_prev,k|  / (gamma_lv V0^(1/3)) < tol_force
    max vertex move of one iteration in the window / V0^(1/3) < tol_step

hold (E_prev, F_prev = values at the previous test; F = force of the liquid on
the sphere, E = total surface free energy). The raw gradient criterion
`SolverConfig.grad_tol` also counts as converged.

At equilibrium the force on the sphere is F = -dE*/dc (envelope argument), so
the recorded (E, F) pairs are consistent -- unlike during a fast motion, where
the force contains the lag of the gradient flow.

This module only holds state and decisions; the iteration itself is in
solver.relax, the frames are recorded by main.py.
"""
import time

import numpy as np

from .obstacle import path_length, sphere_center_at


class QuasiStatic:
    def __init__(self, qcfg, obstacle_cfg, gamma, V0, clock=time.perf_counter):
        self.q, self.ocfg = qcfg, obstacle_cfg
        self.length = path_length(obstacle_cfg)
        self.delta_max = float(qcfg.step)
        self.delta = self.delta_max
        self.delta_min = self.delta_max * float(qcfg.min_step_factor)
        R = float(V0) ** (1.0 / 3.0)
        self.E_scale = gamma * R * R
        self.F_scale = gamma * R
        self.L_scale = R
        self.clock = clock
        self.u = 0.0                 # path position being relaxed
        self.u_done = None           # path position of the last recorded frame
        self.n_it = 0                # iterations of the current step
        self.converged_now = False   # True from the converging iteration until the next one starts
        self.finished = False
        self.energy_fn = None        # callable(verts, A, sphere_center, wetted_loop) -> E, set by main
        self.ckpt = None             # (u, verts) of the last converged state (adaptive rollback)
        self.frames = 0
        self.unconverged = 0
        self.rollbacks = 0
        self.frame_info = {}
        self.total_iters_in_frames = 0
        self._reset_window()
        self.t0 = clock()
        self.t_last = self.t0
        self.rate = None             # EMA of wall seconds per unit path length
        self.last_ok = True

    # ------------------------------------------------------------------ path
    def center(self):
        return sphere_center_at(self.ocfg, self.u)

    def progress(self):
        """Fraction of the path covered by the last recorded frame (0..1)."""
        if self.u_done is None:
            return 0.0
        return 1.0 if self.length == 0.0 else float(self.u_done)

    def eta(self):
        """Estimated remaining wall time in seconds (None while unknown)."""
        if self.finished:
            return 0.0
        if self.rate is None:
            return None
        return self.rate * (1.0 - self.progress()) * self.length

    # ------------------------------------------------------------ iteration
    def _reset_window(self):
        self.prev_E = None
        self.prev_F = None
        self.step_max_run = 0.0

    def on_remesh(self):
        """Call after a remesh: the rollback state belongs to the old mesh. The
        convergence window is kept (a remesh only perturbs E and F slightly; if it
        exceeds the tolerance the test simply fails and is repeated)."""
        self.ckpt = None

    def reset_window(self):
        """Call after a remesh: the comparison values belong to the old mesh,
        the rollback state too."""
        self._reset_window()
        self.ckpt = None

    def advance(self, verts):
        """Called at the start of the iteration that follows a recorded frame:
        moves the sphere to the next step position (or finishes the run)."""
        self.converged_now = False
        if self.length == 0.0 or self.u >= 1.0:
            self.finished = True
            return
        n_used = self.n_it
        if (self.q.adaptive and self.last_ok and self.delta < self.delta_max
                and n_used < self.q.max_iter_per_step / 3.0):
            self.delta = min(2.0 * self.delta, self.delta_max)
        self.ckpt = (self.u, verts.copy())
        self.u = min(1.0, self.u + self.delta / self.length)
        self.n_it = 0
        self._reset_window()

    def observe(self, verts, F, step_max, gnorm_converged, A, sphere_center, wetted_loop):
        """Called once per iteration (after the update). Returns "run",
        "converged" or "rollback" (the solver then restores `ckpt`)."""
        q = self.q
        self.n_it += 1
        self.step_max_run = max(self.step_max_run, float(step_max))
        timeout = self.n_it >= q.max_iter_per_step
        if not (gnorm_converged or self.n_it % q.check_every == 0 or timeout):
            return "run"
        E = float(self.energy_fn(verts, A, sphere_center, wetted_loop))
        F = np.asarray(F, dtype=float).copy()
        ok = bool(gnorm_converged)
        if (not ok and self.prev_E is not None and self.n_it >= q.min_iter_per_step):
            ok = (abs(E - self.prev_E) / self.E_scale < q.tol_energy
                  and float(np.abs(F - self.prev_F).max()) / self.F_scale < q.tol_force
                  and self.step_max_run / self.L_scale < q.tol_step)
        self.prev_E, self.prev_F, self.step_max_run = E, F, 0.0
        if ok:
            self._frame(True)
            return "converged"
        if timeout:
            if q.adaptive and self.ckpt is not None and self.delta > self.delta_min * (1 + 1e-9):
                return "rollback"
            self._frame(False)
            return "converged"
        return "run"

    def _frame(self, ok):
        now = self.clock()
        dt = now - self.t_last
        self.t_last = now
        step_len = (self.u - (self.u_done if self.u_done is not None else self.u)) * self.length
        if step_len > 0:
            r = dt / step_len
            self.rate = r if self.rate is None else 0.7 * self.rate + 0.3 * r
        self.last_ok = ok
        self.u_done = self.u
        self.frames += 1
        self.unconverged += 0 if ok else 1
        self.total_iters_in_frames += self.n_it
        self.frame_info = {"step_iters": self.n_it, "converged": ok, "u": self.u}
        self.converged_now = True

    # Everything that influences the numerics (and the counters printed at
    # the end); the wall-clock rate for the ETA is not part of it.
    _STATE = ("u", "u_done", "n_it", "converged_now", "finished", "delta", "frames", "unconverged",
              "rollbacks", "total_iters_in_frames", "prev_E", "step_max_run", "last_ok")

    def get_state(self):
        """(dict of scalars, rollback verts or None) for a restart checkpoint."""
        d = {k: getattr(self, k) for k in self._STATE}
        d["prev_F"] = None if self.prev_F is None else [float(x) for x in self.prev_F]
        d["frame_info"] = dict(self.frame_info)
        d["ckpt_u"] = None if self.ckpt is None else float(self.ckpt[0])
        return d, (None if self.ckpt is None else self.ckpt[1])

    def set_state(self, d, ckpt_verts=None):
        """Inverse of get_state."""
        for k in self._STATE:
            setattr(self, k, d[k])
        self.prev_F = None if d["prev_F"] is None else np.array(d["prev_F"], dtype=float)
        self.frame_info = dict(d["frame_info"])
        self.ckpt = None if d["ckpt_u"] is None else (d["ckpt_u"], np.array(ckpt_verts, dtype=float))

    def rollback(self):
        """Back to the last converged state with half the step. Returns the
        verts to restore."""
        u_prev, verts = self.ckpt
        self.delta = max(self.delta * 0.5, self.delta_min)
        self.u = min(1.0, u_prev + self.delta / self.length)
        self.n_it = 0
        self.rollbacks += 1
        self._reset_window()
        return verts.copy()
