"""The run loop: start mesh -> relax() -> remesh when triggered -> relax() ...

The driver builds the scaled start mesh, then runs one `solver.relax()`
generator until either the iteration budget is used up or a remesh trigger
fires ("split", "collapse", "contact_collapse"; checked every
`mesh.remesh_check_every` iterations). On a trigger it remeshes and starts a
NEW relax() generator (which rebuilds the sparse factorization); every remesh
renumbers the vertices and starts a new topology "epoch".

Outputs are sinks (io.frame.FrameSink): the driver hands them a RunInfo at the
start, a Frame per recorded state, Events and a cheap tick every iteration.
Nothing a sink does feeds back into the numerics except the StopRequest, so a
run gives bit-identical results with or without a window attached.
"""

import math
import time
from types import SimpleNamespace

import numpy as np

from .analysis import (com_xy_offset_to_sphere, format_frame_table, sphere_gap, surface_free_energy,
                       wetted_loop_indices)
from .config import apply_volume_scale
from .energy import area_and_grad, center_of_mass, mean_curvature, volume_and_grad
from .io.frame import Event, Frame, RunInfo, RunStatus, StopRequest, readonly
from .mesh import (build_dof_mask, collapse_short_edges, long_edges_near, make_start_mesh, max_edge_length,
                   mean_edge_length, refine_rivara, resolve_volume_scale, short_edge_fraction, short_edges_near,
                   subdivide_mesh, tangential_smooth)
from .quasistatic import QuasiStatic
from .roughness import make_roughness_field
from .solver import SolverDiverged, relax
from .substrate import add_base_mesh, base_triangle_mask, strip_base


class Driver:
    """Runs one relaxation of `cfg` (a config.Config, unscaled) and feeds the
    sinks. `run()` returns a summary dict of the final state."""

    def __init__(self, cfg, sinks=(), stop=None):
        self.cfg_in = cfg
        self.sinks = list(sinks)
        self.stop = stop if stop is not None else StopRequest()
        self._n_frames = 0
        self._last_frame_iter = None
        self._table = []          # small per-frame records for the quasi-static console table
        self._topo_cache = {}     # epoch -> read-only (tris, boundary_idx)

    # -- sink fan-out -------------------------------------------------------

    def _log(self, msg):
        e = Event("log", -1, message=msg)
        for s in self.sinks:
            s.event(e)

    def _event(self, e):
        for s in self.sinks:
            s.event(e)

    def _topology(self, epoch, tris, boundary_idx):
        t = self._topo_cache.get(epoch)
        if t is None:
            t = (readonly(tris, np.int64), readonly(boundary_idx, np.int64))
            self._topo_cache = {epoch: t}    # older epochs are never asked for again
        return t

    def _emit_frame(self, i, v, tris, boundary_idx, A, V, sphere_center, wetted_loop, epoch, diag, quasi):
        cfg = self.cfg
        energy = surface_free_energy(cfg, v, tris, boundary_idx, A, sphere_center, wetted_loop)
        force = np.array(diag.get("force", np.zeros(3)), dtype=float)
        com = center_of_mass(v, tris)
        x = float(sphere_center[0]) if (quasi is not None and sphere_center is not None) else float(i)
        t_ro, b_ro = self._topology(epoch, tris, boundary_idx)
        info = quasi.frame_info if quasi is not None else {}
        f = Frame(
            index=self._n_frames, iteration=int(i), x=x, epoch=int(epoch),
            verts=readonly(v, np.float64), tris=t_ro, boundary_idx=b_ro,
            wetted_loop=readonly(wetted_loop, np.int64),
            sphere_center=readonly(sphere_center, np.float64),
            energy=float(energy), area=float(A), volume=float(V),
            force=readonly(force), com=readonly(com),
            com_offset_xy=readonly(com_xy_offset_to_sphere(com, sphere_center)),
            gap=sphere_gap(sphere_center, cfg.obstacle.radius) if cfg.obstacle.enabled else float("nan"),
            curvature=readonly(mean_curvature(v, tris, boundary_idx)),
            step_iters=int(info.get("step_iters", -1)), converged=bool(info.get("converged", True)),
        )
        self._n_frames += 1
        self._last_frame_iter = int(i)
        if quasi is not None and sphere_center is not None:
            self._table.append(SimpleNamespace(sphere_center=f.sphere_center, step_iters=f.step_iters,
                                               converged=f.converged, energy=f.energy, force=f.force))
        for s in self.sinks:
            s.frame(f)
        return f

    # -- one relax() generator ----------------------------------------------

    def _run_until_remesh_or_done(self, verts, tris, boundary_idx, mask, start_iter, total_iters,
                                  V0, L0, check_split, check_collapse, epoch, quasi, t_start,
                                  cc_block_until, diag):
        """Runs a single relax() generator from start_iter to total_iters,
        stopping EARLY only if a remesh trigger fires (checked every
        remesh_check_every iterations, past iteration 0): either the mesh
        has stretched too much (split needed, while check_split) or a
        large-enough fraction of it has become unnecessarily fine (collapse
        needed, while check_collapse). relax() -- and the sparse Laplacian
        factorization it builds once at the top -- is only ever re-created
        when a remesh actually happens (rare), not on a fixed periodic
        schedule regardless of need.

        Returns (last, kind), where last = (i, verts, A, V, sphere_center,
        wetted_loop) is the most recently yielded state and kind is "split",
        "collapse", "contact_collapse", "diverged", "stopped" or None (ran to
        total_iters)."""
        cfg = self.cfg
        solver_gen = relax(verts, tris, boundary_idx, mask, cfg.solver, cfg.obstacle,
                           start_iter=start_iter, stop_iter=total_iters, V_target=V0, diag=diag,
                           anomaly_cfg=cfg.anomaly, roughness=cfg.roughness_field, quasi=quasi)
        if quasi is not None:
            quasi.energy_fn = lambda v_, A_, c_, w_: surface_free_energy(cfg, v_, tris, boundary_idx, A_, c_, w_)
        # The remesh triggers look at the free liquid/vapor surface (the dome)
        # only; the flat base on the substrate (substrate.py) is rebuilt from
        # the contact-line ring after every remesh and never triggers one.
        tris_d = tris[~base_triangle_mask(verts, tris, boundary_idx)]
        last = (start_iter, verts, None, None, None, None)
        local_split = cfg.mesh.remesh_split_mode == "local"
        contact_radius = 2.0 * cfg.obstacle.radius
        contact_len = cfg.mesh.remesh_contact_len_ratio * L0
        every = max(int(cfg.output.every), 1)

        try:
            for i, v, A, V, sphere_center, wetted_loop in solver_gen:
                last = (i, v, A, V, sphere_center, wetted_loop)
                self._tick(i, quasi, total_iters, t_start)

                remesh_kind = None
                if i > 0 and i % cfg.mesh.remesh_check_every == 0:
                    if check_split and (
                            max_edge_length(v, tris_d) / L0 > cfg.mesh.remesh_trigger_ratio
                            or (local_split and sphere_center is not None and long_edges_near(
                                v, tris_d, sphere_center, contact_radius, 1.5 * contact_len))):
                        remesh_kind = "split"
                    elif check_collapse and short_edge_fraction(
                            v, tris_d, cfg.mesh.remesh_min_edge_ratio * L0) > cfg.mesh.remesh_min_edge_fraction:
                        remesh_kind = "collapse"
                    elif (local_split and check_collapse and sphere_center is not None and i >= cc_block_until
                            and short_edges_near(v, tris_d, sphere_center, contact_radius,
                                                 0.4 * contact_len)):
                        # The GLOBAL collapse trigger above never fires from a
                        # few pathologically tiny edges concentrated at the
                        # contact patch (local refinement, being local, rarely
                        # pushes the mesh-wide short-edge FRACTION over
                        # remesh_min_edge_fraction) -- so local mode needs its
                        # own, contact-only trigger for exactly that case (see
                        # mesh.py:short_edges_near and the protect_min_len
                        # collapse below).
                        remesh_kind = "contact_collapse"

                if quasi is not None:
                    # quasi-static: a frame exactly when a sphere step has converged
                    record = quasi.converged_now
                else:
                    record = i % every == 0 or remesh_kind is not None
                if record:
                    self._emit_frame(i, v, tris, boundary_idx, A, V, sphere_center, wetted_loop, epoch, diag, quasi)

                if remesh_kind is not None:
                    return last, remesh_kind
                if self.stop.requested:
                    return last, "stopped"
        except SolverDiverged as err:
            # `last` is the last state the solver yielded successfully --
            # never a NaN one -- so outputs keep a valid droplet instead of it
            # vanishing, and the cause is stated plainly.
            self._log(f"\n!!! {err}")
            self._log("!!! Stopping here and keeping the last valid state. See README, "
                      "'Why can the droplet vanish?', for what usually causes this.")
            self._event(Event("diverged", int(err.iteration), message=str(err)))
            return last, "diverged"

        return last, None

    def _tick(self, i, quasi, total_iters, t_start):
        if not self.sinks:
            return
        if quasi is not None:
            frac, eta = quasi.progress(), quasi.eta()
        else:
            frac = min(max(i / max(total_iters, 1), 0.0), 1.0)
            el = time.perf_counter() - t_start
            eta = el * (1.0 - frac) / frac if frac > 0 else None
        for s in self.sinks:
            s.tick(i, frac, eta)

    # -- the whole run --------------------------------------------------------

    def run(self):
        cfg = self.cfg_in
        mesh_scale = resolve_volume_scale(cfg.mesh)
        cfg = self.cfg = apply_volume_scale(cfg, mesh_scale)
        if mesh_scale != 1.0:
            self._log(f"MeshConfig.target_volume={cfg.mesh.target_volume:.6g}: scaling the mesh "
                      f"(and volume_penalty/wetting_snap_distance/capture_radius to match) by {mesh_scale:.4g}.")

        verts, tris, boundary_idx = make_start_mesh(cfg.mesh)
        mask = build_dof_mask(len(verts), boundary_idx)

        base_tri0 = base_triangle_mask(verts, tris, boundary_idx)
        # Liquid/vapor area only: with a closed mesh (base on the substrate,
        # see substrate.py) the base triangles are not part of it.
        A0 = float(area_and_grad(verts, tris)[2][~base_tri0].sum())
        V0, _ = volume_and_grad(verts, tris)
        r0 = np.hypot(verts[boundary_idx, 0], verts[boundary_idx, 1]).mean()
        self._log(f"Start:  {len(verts)} vertices, {len(tris)} triangles"
                  + (f" (closed mesh: {int((~base_tri0).sum())} dome + {int(base_tri0.sum())} base)"
                     if base_tri0.any() else " (open dome, no base mesh)"))
        self._log(f"        Area A0={A0:.5f}, Volume V0={V0:.5f}, contact radius r0={r0:.5f}, "
                  f"gamma_lv={cfg.solver.gamma_lv:g}")

        if cfg.anomaly.enabled:
            _ring = verts[boundary_idx, :2]
            _edge = float(np.linalg.norm(np.roll(_ring, -1, axis=0) - _ring, axis=1).mean())
            _w = cfg.anomaly.transition_fraction * cfg.anomaly.radius
            self._log(f"Wettability anomaly: disc centre {tuple(cfg.anomaly.center_xy)}, radius "
                      f"{cfg.anomaly.radius:g}, theta = {cfg.anomaly.contact_angle_deg:g} deg (wall: "
                      f"{cfg.solver.contact_angle_deg:g} deg), transition width {_w:.3g}; initial "
                      f"contact-line edge length {_edge:.3g}"
                      + ("  -- NOTE: the transition is narrower than the contact-line edge length; "
                         "the contact line resolves it only coarsely." if _w < _edge else "."))
        if cfg.roughness.enabled:
            _ring = verts[boundary_idx, :2]
            _edge = float(np.linalg.norm(np.roll(_ring, -1, axis=0) - _ring, axis=1).mean())
            cfg.roughness_field = make_roughness_field(
                cfg.roughness, _edge, math.cos(math.radians(cfg.solver.contact_angle_deg)))
            self._log("Substrate roughness:")
            self._log("  " + cfg.roughness_field.describe().replace("\n", "\n  "))
            self._log("  " + cfg.roughness_field.check_resolution(verts, boundary_idx))
        # Reference edge-length scale for the remesh triggers (see
        # MeshConfig.remesh_trigger_ratio / remesh_min_edge_ratio) --
        # computed once, from the ORIGINAL mesh, so it doesn't shift as the
        # mesh itself gets refined or pruned. Dome edges only (the base does
        # not set the scale).
        L0 = mean_edge_length(verts, tris[~base_tri0])

        total_iters = cfg.solver.n_iterations
        use_quasi = cfg.quasi.enabled and cfg.obstacle.enabled
        quasi = (QuasiStatic(cfg.quasi, cfg.obstacle, cfg.solver.gamma_lv, V0) if use_quasi else None)
        if use_quasi:
            c0, c1 = cfg.obstacle.center_start, cfg.obstacle.center_end
            n_steps = int(math.ceil(quasi.length / quasi.delta_max - 1e-9)) if quasi.length > 0 else 0
            self._log(f"Quasi-static sphere motion: path {tuple(c0)} -> {tuple(c1)} (length {quasi.length:g}), "
                      f"step {cfg.quasi.step:g}{' (adaptive)' if cfg.quasi.adaptive else ''}, "
                      f"{n_steps + 1} equilibrium frames; a frame is recorded when the liquid has relaxed "
                      f"(tolerances: energy {cfg.quasi.tol_energy:g}, force {cfg.quasi.tol_force:g}, "
                      f"move {cfg.quasi.tol_step:g}; at most {cfg.quasi.max_iter_per_step} iterations per step).")

        self.run_info = RunInfo(config=cfg, mesh_scale=float(mesh_scale), V0=float(V0), A0=A0,
                                verts=readonly(verts), tris=readonly(tris), boundary_idx=readonly(boundary_idx),
                                total_iters=int(total_iters), quasi_static=bool(use_quasi),
                                roughness_field=cfg.roughness_field, stop=self.stop)
        for s in self.sinks:
            s.start(self.run_info)

        split_count = 0
        collapse_exhausted = False  # set once a collapse pass makes no
        # further progress (e.g. the only remaining short edges are the
        # mesh's permanent apex "spokes", which collapse_short_edges
        # correctly refuses to touch) -- stops re-triggering the same
        # no-op pass forever.
        epoch = 0
        diag = {}
        t0 = time.perf_counter()
        start_iter = 0
        last = (0, verts, A0, V0, None, None)
        tris_last, boundary_last, epoch_last = tris, boundary_idx, epoch
        end_kind = None

        t_start = time.perf_counter()
        cc_block_until = 0   # no-op contact collapses are not re-triggered before this iteration
        while start_iter < total_iters and not (quasi is not None and quasi.finished):
            if cfg.mesh.remesh_split_mode == "local":
                check_split = cfg.mesh.remesh_enabled and len(tris) < cfg.mesh.remesh_max_triangles
            else:
                check_split = cfg.mesh.remesh_enabled and split_count < cfg.mesh.remesh_max_count
            # Collapsing only ever makes sense as pruning something a
            # split added, and only once we know it's actually effective
            # here (see collapse_exhausted above).
            check_collapse = (cfg.mesh.remesh_enabled and cfg.mesh.remesh_min_edge_ratio > 0
                              and split_count >= 1 and not collapse_exhausted)
            last, remesh_kind = self._run_until_remesh_or_done(
                verts, tris, boundary_idx, mask, start_iter, total_iters,
                V0, L0, check_split, check_collapse, epoch, quasi, t_start, cc_block_until, diag,
            )
            start_iter, verts, _, _, _, _ = last
            # topology that belongs to `last` (a remesh below renumbers)
            tris_last, boundary_last, epoch_last = tris, boundary_idx, epoch

            if remesh_kind in ("diverged", "stopped"):
                end_kind = remesh_kind
                break

            if remesh_kind in ("split", "collapse", "contact_collapse"):
                # Remesh the open dome only; the flat base on the substrate
                # (substrate.py) is dropped here and rebuilt from the
                # contact-line ring right after (see the end of this block).
                orig_state = (verts, tris, boundary_idx)   # restored if the remesh turns out to be a no-op
                n_verts_before, n_tris_before = len(verts), len(tris)
                verts, tris, boundary_idx = strip_base(verts, tris, boundary_idx)

            if remesh_kind == "split":
                distortion = max_edge_length(verts, tris) / L0
                n_before = len(tris)
                if cfg.mesh.remesh_split_mode == "local":
                    sc = last[4]
                    verts, tris, boundary_idx = refine_rivara(
                        verts, tris, boundary_idx, cfg.mesh.remesh_local_target_ratio * L0,
                        fine_center=sc, fine_radius=2.0 * cfg.obstacle.radius,
                        fine_len=cfg.mesh.remesh_contact_len_ratio * L0,
                    )
                    if sc is not None:
                        # A local split, being local, fires almost every check
                        # while the sphere is active, so the dedicated
                        # "contact_collapse" trigger (checked only when NO split
                        # is needed) rarely gets a turn. Prune pathologically
                        # tiny contact-patch edges (see
                        # short_edges_near/protect_min_len) right here instead,
                        # after every local split, not only when one is absent.
                        n_pruned = len(tris)
                        for _ in range(4):
                            # One pass freezes every touched vertex (to avoid
                            # cascading through a whole neighborhood in one
                            # go -- see collapse_short_edges), so a single
                            # call leaves some collapsible edges untouched
                            # simply because their neighbor was already frozen
                            # this round. A handful of repeats cleans up what
                            # one pass alone cannot; each pass no-ops cheaply
                            # once nothing is left to collapse.
                            n_here = len(tris)
                            verts, tris, boundary_idx = collapse_short_edges(
                                verts, tris, boundary_idx, 0.0,
                                protect_center=sc, protect_radius=2.0 * cfg.obstacle.radius,
                                protect_min_len=0.4 * cfg.mesh.remesh_contact_len_ratio * L0,
                            )
                            if len(tris) == n_here:
                                break
                        if len(tris) != n_pruned:
                            self._log(f"        [contact collapse after split: "
                                      f"{n_pruned} -> {len(tris)} triangles]")
                    kind = "local split"
                else:
                    verts, tris, boundary_idx = subdivide_mesh(verts, tris, boundary_idx)
                    kind = "global split"
                mask = build_dof_mask(len(verts), boundary_idx)
                split_count += 1
                self._log(f"        [{kind} #{split_count} at iteration {start_iter}: "
                          f"distortion {distortion:.2f}x, {n_before} -> {len(tris)} triangles]")
            elif remesh_kind == "collapse":
                n_before = len(tris)
                sphere_center = last[4]
                protect_center = sphere_center if sphere_center is not None else None
                protect_radius = cfg.obstacle.radius * 2.5 if protect_center is not None else 0.0
                # Even right at the sphere, an edge can end up pathologically
                # tiny (the wetted patch's contact line gets pinched by
                # competing forces as it moves) -- a blanket protection would
                # leave it stuck there forever, stalling the trust region and
                # bloating the triangle count. Allow collapsing DOWN TO the
                # deliberate contact-refinement target itself (a fraction of
                # it, so genuine over-refinement is pruned without eating into
                # triangles that are merely as fine as intended).
                protect_min_len = (0.4 * cfg.mesh.remesh_contact_len_ratio * L0
                                   if protect_center is not None else None)
                for _ in range(4):  # see the comment on the split-branch version above
                    n_here = len(tris)
                    verts, tris, boundary_idx = collapse_short_edges(
                        verts, tris, boundary_idx, cfg.mesh.remesh_min_edge_ratio * L0,
                        protect_center=protect_center, protect_radius=protect_radius,
                        protect_min_len=protect_min_len,
                    )
                    if len(tris) == n_here:
                        break
                if len(tris) == n_before:
                    # No collapse was actually safe to perform (only the
                    # permanent apex spokes remain below threshold) --
                    # don't keep re-triggering this no-op.
                    collapse_exhausted = True
                    self._log(f"        [collapse at iteration {start_iter}: no safe collapses "
                              f"found (likely just the apex spokes) -- disabling further collapse checks]")
                else:
                    self._log(f"        [collapse at iteration {start_iter}: "
                              f"{n_before} -> {len(tris)} triangles]")
                mask = build_dof_mask(len(verts), boundary_idx)
            elif remesh_kind == "contact_collapse":
                # Same collapse, but restricted to ONLY the contact patch (see
                # short_edges_near/protect_min_len): min_len=0 leaves the rest
                # of the mesh untouched, protect_min_len prunes edges there
                # that have shrunk well below the deliberate contact target.
                n_before = len(tris)
                sphere_center = last[4]
                for _ in range(4):  # see the comment on the split-branch version above
                    n_here = len(tris)
                    verts, tris, boundary_idx = collapse_short_edges(
                        verts, tris, boundary_idx, 0.0,
                        protect_center=sphere_center, protect_radius=cfg.obstacle.radius * 2.5,
                        protect_min_len=0.4 * cfg.mesh.remesh_contact_len_ratio * L0,
                    )
                    if len(tris) == n_here:
                        break
                if len(tris) == n_before:
                    # nothing was collapsed: keep the mesh as it is (no cleanup, no new epoch) and do
                    # not re-trigger for a while -- every trigger restarts the solver (new factorization)
                    # and, in quasi-static mode, interrupts the convergence test of the step
                    cc_block_until = start_iter + 10 * cfg.mesh.remesh_check_every
                    remesh_kind = None
                    verts, tris, boundary_idx = orig_state
                else:
                    self._log(f"        [contact collapse at iteration {start_iter}: "
                              f"{n_before} -> {len(tris)} triangles]")
                mask = build_dof_mask(len(verts), boundary_idx)

            if remesh_kind is not None:
                # Mesh-quality cleanup: straighten out the skinny/sheared
                # triangles a split or collapse leaves behind (especially
                # visible right under the sphere, where contact shears the
                # surface) without materially changing the shape itself --
                # see mesh.py:tangential_smooth.
                verts = tangential_smooth(verts, tris, boundary_idx)
                if cfg.mesh.base_mesh_enabled:
                    verts, tris, boundary_idx = add_base_mesh(verts, tris, boundary_idx)
                mask = build_dof_mask(len(verts), boundary_idx)
                epoch += 1
                if quasi is not None:
                    quasi.on_remesh()      # the rollback state belongs to the old mesh
                sc_now = last[4]
                self._event(Event("remesh", int(start_iter), message=remesh_kind, data={
                    "kind": remesh_kind, "epoch": epoch,
                    "n_verts_before": n_verts_before, "n_verts_after": len(verts),
                    "n_tris_before": n_tris_before, "n_tris_after": len(tris),
                    "verts": readonly(verts), "tris": readonly(tris), "boundary_idx": readonly(boundary_idx),
                    "sphere_center": readonly(sc_now),
                    "wetted_loop": readonly(wetted_loop_indices(cfg, verts, tris, sc_now)),
                }))

        elapsed = time.perf_counter() - t0
        i, verts_final, Af, Vf, c_fin, loop_fin = last
        if Af is None:   # stopped before the first iteration was yielded
            i, verts_final, Af, Vf = start_iter, verts, A0, V0
        if self._last_frame_iter != int(i):
            # make sure the final state is recorded (e.g. after a stop, or when
            # n_iterations is not a multiple of output.every)
            self._emit_frame(i, verts_final, tris_last, boundary_last, Af, Vf, c_fin, loop_fin,
                             epoch_last, diag, quasi)

        if quasi is not None:
            n_bad = quasi.unconverged
            self._log(f"Quasi-static run: {quasi.frames} frames, {quasi.total_iters_in_frames} iterations in the "
                      f"steps, {n_bad} step(s) not converged"
                      + (f", {quasi.rollbacks} rollback(s)" if cfg.quasi.adaptive else "")
                      + ("" if quasi.finished else "  -- the path end was NOT reached (iteration limit, "
                                                   "stop or divergence)")
                      + ".")
            if self._table:
                self._log(format_frame_table(self._table))

        rf = np.hypot(verts_final[boundary_last, 0], verts_final[boundary_last, 1]).mean()
        self._log(f"End:    {i} iterations in {elapsed:.2f}s"
                  + (f"  (stopped: {self.stop.reason})" if end_kind == "stopped" else ""))
        self._log(f"        Area Af={Af:.5f}  (decrease: {A0 - Af:.5f})")
        self._log(f"        Volume Vf={Vf:.5f}  (deviation: {abs(V0 - Vf):.2e})")
        self._log(f"        Contact radius rf={rf:.5f}  (target angle: {cfg.solver.contact_angle_deg:.1f} deg)")
        if epoch > 0:
            n_base = int(base_triangle_mask(verts_final, tris_last, boundary_last).sum())
            self._log(f"        Remeshed {epoch}x, final resolution: "
                      f"{len(verts_final)} vertices, {len(tris_last)} triangles"
                      + (f" ({len(tris_last) - n_base} dome + {n_base} base)" if n_base else ""))

        summary = {
            "iteration": int(i),
            "energy": surface_free_energy(cfg, verts_final, tris_last, boundary_last, Af, c_fin, loop_fin),
            "area": float(Af), "volume": float(Vf), "contact_radius": float(rf), "remeshes": int(epoch),
            "verts": verts_final, "tris": tris_last, "boundary_idx": boundary_last,
            "force": None if quasi is None or quasi.prev_F is None else quasi.prev_F.tolist(),
            "quasi_frames": None if quasi is None else int(quasi.frames),
            "frames": self._n_frames,
        }
        status = RunStatus(end_kind or "finished", int(i), self.stop.reason or "")
        for s in self.sinks:
            s.finish(status)
        return summary
