"""
Main program: cone -> area-minimal shape at fixed volume, via Sobolev
gradient descent, with optional live 3D visualization (including a
live surface-energy chart docked at the top of the window, a substrate
grid, an optional movable rigid-sphere obstacle the droplet deforms
around, periodic adaptive remeshing (split where too stretched,
collapse where no longer needed) if the obstacle distorts the mesh,
and a history slider to scrub back through the whole run afterwards).

Run:
    python main.py

All parameters are collected in config.py.
"""

import math
import time

import numpy as np

from DropletRelaxer.config import CONFIG
from DropletRelaxer.mesh import (
    make_start_mesh, resolve_volume_scale, build_dof_mask, subdivide_mesh, collapse_short_edges,
    mean_edge_length, max_edge_length, short_edge_fraction, tangential_smooth,
    refine_rivara, long_edges_near, short_edges_near,
)
from DropletRelaxer.energy import area_and_grad, volume_and_grad
from DropletRelaxer.solver import relax, SolverDiverged
from DropletRelaxer.quasistatic import QuasiStatic
from DropletRelaxer.obstacle import sphere_center_at
from visualize import make_anomaly_disc, make_roughness_surface
from visualize import (
    make_plotter, update_mesh, replace_mesh, make_energy_chart, update_energy_chart,
    make_obstacle_mesh, update_obstacle_mesh, make_substrate_mesh, enable_right_button_pan,
    make_history_slider, update_wetting_outline, ProgressBar, update_volume_text, update_ring_outline,
)
from DropletRelaxer.substrate import base_triangle_mask, add_base_mesh, strip_base
from DropletRelaxer.wetting import compute_wetted_set, wetted_boundary_loop
from DropletRelaxer.energy import center_of_mass, mean_curvature
from visualize import update_com_text, _capillary_force, set_gap_offset
import export
from DropletRelaxer.roughness import make_roughness_field
from DropletRelaxer.analysis import surface_free_energy


class _Progress:
    """One-line terminal progress bar (end_only mode): fraction, iteration,
    elapsed time, ETA and number of remeshings. Redraws at most every 0.25 s."""

    def __init__(self, total, width=30):
        self.total, self.width = max(int(total), 1), width
        self.t0 = time.perf_counter()
        self.last = 0.0

    @staticmethod
    def _fmt(sec):
        sec = int(sec)
        return f"{sec // 3600}h{sec % 3600 // 60:02d}m{sec % 60:02d}s" if sec >= 3600 else \
            f"{sec // 60}m{sec % 60:02d}s"

    def update(self, i, remeshes, force=False, frac=None, eta=None):
        """`frac` (0..1) and `eta` (seconds, None = unknown) come from the
        quasi-static path progress; without them iteration / total is used."""
        now = time.perf_counter()
        if not force and now - self.last < 0.25:
            return
        self.last = now
        el = now - self.t0
        if frac is None:
            frac = min(max(i / self.total, 0.0), 1.0)
            eta = el * (1.0 - frac) / frac if frac > 0 else None
        done = int(round(frac * self.width))
        eta_s = self._fmt(eta) if eta is not None else "--"
        print(f"\r[{'#' * done}{'-' * (self.width - done)}] {100 * frac:5.1f}%  "
              f"iter {i}  elapsed {self._fmt(el)}  ETA {eta_s}  remeshes {remeshes}   ",
              end="", flush=True)


def _progress_text(frac, elapsed, eta):
    """Text inside the window's progress bar."""
    eta_s = "..." if eta is None else "~" + _Progress._fmt(eta)
    return f"Progress {100 * frac:.0f} %   |   elapsed {_Progress._fmt(elapsed)}   |   ETA {eta_s}"


def _tick(live_ctx, quasi, i, epoch, t0, force=False):
    """Progress display between frames (called every iteration, acts at most
    once per second): console bar in end_only mode, in-window bar otherwise.
    Refreshing the bar redraws the (unchanged) scene and lets the window handle
    its events; the mesh itself is only redrawn when a step has converged."""
    if live_ctx is None:
        return
    now = time.perf_counter()
    if not force and now - live_ctx.get("t_tick", 0.0) < 1.0:
        return
    live_ctx["t_tick"] = now
    if quasi is not None:
        frac, eta = quasi.progress(), quasi.eta()
    else:
        frac = min(max(i / max(live_ctx.get("total", 1), 1), 0.0), 1.0)
        el = now - t0
        eta = el * (1.0 - frac) / frac if frac > 0 else None
    prog = live_ctx.get("progress")
    if prog is not None:
        prog.update(i, epoch, force=True, frac=frac if quasi is not None else None, eta=eta)
    bar = live_ctx.get("bar")
    if bar is not None:
        bar.update(frac, _progress_text(frac, now - t0, eta))
        bar.render()


def _wetted_loop_points(cfg, verts, tris, sphere_center):
    """Vertex positions of the current three-phase loop on the sphere for a given mesh
    (same detection as solver.relax), or None without a contact patch. Used to redraw the
    contact line after a remesh, when the loop of the last yielded state no longer matches
    the vertex numbering."""
    oc = cfg.obstacle
    if not (oc.enabled and oc.wetting_enabled):
        return None
    wet = compute_wetted_set(verts, tris, sphere_center, oc.radius, oc.wetting_snap_distance)
    if wet.sum() < oc.wetting_min_patch_vertices or not wet[tris].all(axis=1).any():
        return None
    loop = wetted_boundary_loop(wet, tris, verts, sphere_center)
    return verts[loop] if loop is not None else None


def _log(live_ctx, msg):
    """Remesh/diagnostic line; suppressed in end_only mode, where the terminal
    shows a progress bar instead."""
    if live_ctx is not None and live_ctx.get("headless", False):
        return
    print(msg)


def _run_until_remesh_or_done(cfg, verts, tris, boundary_idx, mask, start_iter, total_iters,
                               V0, L0, check_split, check_collapse, live_ctx,
                               iters_hist, area_hist, center_hist, history, epoch,
                               quasi=None, x_hist=None, frame_rows=None, t_start=0.0, cc_block_until=0):
    """Runs a single relax() generator from start_iter to total_iters,
    stopping EARLY only if a remesh trigger fires (checked every
    remesh_check_every iterations, past iteration 0): either the mesh
    has stretched too much (split needed, while check_split) or a
    large-enough fraction of it has become unnecessarily fine (collapse
    needed, while check_collapse). relax() -- and the sparse Laplacian
    factorization it builds once at the top -- is only ever re-created
    when a remesh actually happens (rare), not on a fixed periodic
    schedule regardless of need.

    Returns (last, remesh_kind), where last = (i, verts, A, V,
    sphere_center, wetted_loop) is the most recently yielded state and
    remesh_kind is "split", "collapse", or None (ran to total_iters).

    `live_ctx`, when not None, is a dict (see main()) carrying the live
    PyVista objects; it is mutated in place (e.g. "wetting_actor" is
    replaced each time the contact-line overlay is redrawn)."""
    diag = live_ctx["diag"] if live_ctx is not None else None
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

    try:
        for i, v, A, V, sphere_center, wetted_loop in solver_gen:
            last = (i, v, A, V, sphere_center, wetted_loop)
            _tick(live_ctx, quasi, i, epoch, t_start)

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
                do_render = quasi.converged_now
            else:
                do_render = i % cfg.vis.render_every == 0 or remesh_kind is not None
            if live_ctx is not None and do_render:
                # `headless` (VisConfig.live_mode == "end_only"): only record the
                # frame for charts/history slider, draw nothing until the run is over.
                headless = live_ctx.get("headless", False)
                if not headless:
                    update_mesh(live_ctx["mesh"], v, tris, boundary_idx)
                    update_obstacle_mesh(live_ctx["obstacle_mesh"], sphere_center,
                                         live_ctx["obstacle_base_points"])
                    loop_points = v[wetted_loop] if wetted_loop is not None else None
                    live_ctx["wetting_actor"] = update_wetting_outline(
                        live_ctx["plotter"], live_ctx["wetting_actor"], loop_points)
                    live_ctx["ring_actor"] = update_ring_outline(
                        live_ctx["plotter"], live_ctx["ring_actor"], v[boundary_idx])
                x_now = float(sphere_center[0]) if (quasi is not None and sphere_center is not None) else float(i)
                iters_hist.append(i)
                x_hist.append(x_now)
                com = center_of_mass(v, tris)
                live_ctx["com_hist"].append(com)
                force = np.array(live_ctx["diag"].get("force", np.zeros(3)))
                live_ctx["force_hist"].append(force)
                if not headless:
                    update_com_text(live_ctx["plotter"], com, sphere_center)
                area_hist.append(surface_free_energy(cfg, v, tris, boundary_idx, A, sphere_center, wetted_loop))
                center_hist.append(sphere_center)
                if quasi is not None:
                    info = quasi.frame_info
                    frame_rows.append({"x": x_now, "y": float(sphere_center[1]), "z": float(sphere_center[2]),
                                       "iteration": i, "step_iters": info["step_iters"],
                                       "converged": info["converged"], "energy": float(area_hist[-1]),
                                       "Fx": float(force[0]), "Fy": float(force[1]), "Fz": float(force[2])})
                    _log(live_ctx, f"  frame {len(frame_rows):3d}: tip x = {x_now:7.2f}  "
                          f"{info['step_iters']:6d} its  F = ({force[0]:.3f}, {force[1]:.3f}, {force[2]:.3f})"
                          + ("" if info["converged"] else "  -- NOT CONVERGED within max_iter_per_step"))
                if not headless:
                    update_energy_chart(live_ctx["chart"], live_ctx["line"], x_hist, area_hist,
                                         marker=live_ctx["marker"], dchart=live_ctx["dchart"],
                                         dline=live_ctx["dline"], dmarker=live_ctx["dmarker"],
                                         forces=live_ctx["force_hist"])
                    update_volume_text(live_ctx["volume_text"], V, V0)
                    if cfg.vis.save_gif:
                        live_ctx["plotter"].write_frame()
                    if not cfg.vis.off_screen:
                        live_ctx["plotter"].update()
                if history is not None:
                    # Cache the curvature just computed above (by
                    # update_mesh, into live_ctx["mesh"]["curvature"])
                    # alongside this frame, so the history slider/Play
                    # button can reuse it instead of recomputing
                    # mean_curvature from scratch on every scrub/Play
                    # tick -- see visualize.py:update_mesh.
                    history.append({"iteration": i, "x": x_now, "verts": v.copy(), "epoch": epoch,
                                     "sphere_center": sphere_center, "wetted_loop": wetted_loop,
                                     "area": area_hist[-1], "volume": V, "com": com, "force": force,
                                     "curvature": (mean_curvature(v, tris, boundary_idx) if headless
                                                   else np.asarray(live_ctx["mesh"]["curvature"]).copy())})

            if remesh_kind is not None:
                return last, remesh_kind
    except SolverDiverged as err:
        # `last` is the last state the solver yielded successfully --
        # never a NaN one -- so the window keeps showing a valid droplet
        # instead of it vanishing, and the cause is stated plainly.
        print(f"\n!!! {err}")
        print("!!! Stopping here and keeping the last valid state. See README, "
              "'Why can the droplet vanish?', for what usually causes this.")
        return last, "diverged"

    return last, None


def main(cfg=None):
    """Runs one relaxation with `cfg` (default: config.CONFIG) and returns a
    summary dict of the final state (used by tools/reference_run.py)."""
    cfg = CONFIG if cfg is None else cfg

    mesh_scale = resolve_volume_scale(cfg.mesh)
    if mesh_scale != 1.0:
        # See the long comment on MeshConfig.target_volume: propagate the
        # SAME length-scale factor to every other config value that is
        # expressed in absolute physical units rather than as a ratio of
        # the mesh's own size, so the run behaves the same (same
        # stability, same rough iteration/remesh counts) at whatever
        # physical scale target_volume asked for. volume_penalty gets
        # s^-4 (its gradient contribution scales as length^5, one power
        # of length short of the area gradient's length^1 -- see
        # solver.py's gE, so a length^-4 coefficient is what keeps that
        # term the same size relative to the others under rescaling).
        # collision_stiffness/adhesion_strength/substrate_stiffness are
        # NOT touched: their gradient (2*k*gap*n_hat) is already
        # length^1, matching the area gradient with a dimensionless k,
        # same as before rescaling.
        #
        # curvature_clim is the odd one out: mean curvature is 1/length,
        # so it scales as mesh_scale^-1 -- the INVERSE direction from
        # everything above. Skipping this rescale is exactly what caused
        # the "fluid renders completely black" symptom at large scale
        # factors: with target_volume=130000 (mesh_scale ~38x), actual
        # curvature values drop to ~1/38 of their original size, landing
        # far below the low end of a clim tuned for the unscaled mesh --
        # under the "inferno" colormap that low end IS black, so nearly
        # the whole surface (everywhere but the sharpest contact-line
        # curvature) got clamped to black regardless of triangle size.
        cfg.solver.volume_penalty /= mesh_scale ** 4
        cfg.obstacle.wetting_snap_distance *= mesh_scale
        cfg.obstacle.capture_radius *= mesh_scale
        cfg.vis.substrate_size *= mesh_scale
        if cfg.vis.curvature_clim:
            lo, hi = cfg.vis.curvature_clim
            cfg.vis.curvature_clim = (lo / mesh_scale, hi / mesh_scale)
        print(f"MeshConfig.target_volume={cfg.mesh.target_volume:.6g}: scaling the mesh "
              f"(and volume_penalty/wetting_snap_distance/capture_radius/substrate_size to "
              f"match, and curvature_clim by the inverse factor) by {mesh_scale:.4g}.")

    verts, tris, boundary_idx = make_start_mesh(
        cfg.mesh
    )
    mask = build_dof_mask(len(verts), boundary_idx)

    base_tri0 = base_triangle_mask(verts, tris, boundary_idx)
    # Liquid/vapor area only: with a closed mesh (base on the substrate,
    # see substrate.py) the base triangles are not part of it.
    A0 = float(area_and_grad(verts, tris)[2][~base_tri0].sum())
    V0, _ = volume_and_grad(verts, tris)
    r0 = np.hypot(verts[boundary_idx, 0], verts[boundary_idx, 1]).mean()
    print(f"Start:  {len(verts)} vertices, {len(tris)} triangles"
          + (f" (closed mesh: {int((~base_tri0).sum())} dome + {int(base_tri0.sum())} base)"
             if base_tri0.any() else " (open dome, no base mesh)"))
    print(f"        Area A0={A0:.5f}, Volume V0={V0:.5f}, contact radius r0={r0:.5f}, "
          f"gamma_lv={cfg.solver.gamma_lv:g}")

    # Reference edge-length scale for the remesh triggers (see
    # MeshConfig.remesh_trigger_ratio / remesh_min_edge_ratio) --
    # computed once, from the ORIGINAL mesh, so it doesn't shift as the
    # mesh itself gets refined or pruned.
    if cfg.anomaly.enabled:
        _ring = verts[boundary_idx, :2]
        _edge = float(np.linalg.norm(np.roll(_ring, -1, axis=0) - _ring, axis=1).mean())
        _w = cfg.anomaly.transition_fraction * cfg.anomaly.radius
        print(f"Wettability anomaly: disc centre {tuple(cfg.anomaly.center_xy)}, radius "
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
        print("Substrate roughness:")
        print("  " + cfg.roughness_field.describe().replace("\n", "\n  "))
        print("  " + cfg.roughness_field.check_resolution(verts, boundary_idx))
    L0 = mean_edge_length(verts, tris[~base_tri0])  # dome edges only (the base does not set the scale)

    total_iters = cfg.solver.n_iterations
    use_quasi = cfg.quasi.enabled and cfg.obstacle.enabled
    quasi = (QuasiStatic(cfg.quasi, cfg.obstacle, cfg.solver.gamma_lv, V0) if use_quasi else None)
    if use_quasi:
        c0, c1 = cfg.obstacle.center_start, cfg.obstacle.center_end
        n_steps = int(math.ceil(quasi.length / quasi.delta_max - 1e-9)) if quasi.length > 0 else 0
        print(f"Quasi-static sphere motion: path {tuple(c0)} -> {tuple(c1)} (length {quasi.length:g}), "
              f"step {cfg.quasi.step:g}{' (adaptive)' if cfg.quasi.adaptive else ''}, "
              f"{n_steps + 1} equilibrium frames; a frame is recorded when the liquid has relaxed "
              f"(tolerances: energy {cfg.quasi.tol_energy:g}, force {cfg.quasi.tol_force:g}, "
              f"move {cfg.quasi.tol_step:g}; at most {cfg.quasi.max_iter_per_step} iterations per step).")
        x0c, x1c = float(c0[0]), float(c1[0])
        chart_xrange = ((x0c - 1.0, x0c + 1.0) if x1c == x0c
                        else (x0c - 0.03 * (x1c - x0c), x1c + 0.03 * (x1c - x0c)))
        chart_xlabel = "Tip x-coordinate"
    else:
        chart_xrange, chart_xlabel = (0.0, max(total_iters, 1)), "Iteration"
    frame_rows = []
    split_count = 0
    collapse_exhausted = False  # set once a collapse pass makes no
    # further progress (e.g. the only remaining short edges are the
    # mesh's permanent apex "spokes", which collapse_short_edges
    # correctly refuses to touch) -- stops re-triggering the same
    # no-op pass forever.

    t0 = time.perf_counter()
    live_ctx = None

    # History (for the post-run scrubber slider) and the topology of
    # each "epoch" (the (tris, boundary_idx) pair currently in use --
    # a new epoch starts each time a remesh changes the vertex/triangle
    # count). Only collected in a real interactive session; it's
    # otherwise unused and would just cost memory.
    collect_history = cfg.vis.live and not cfg.vis.off_screen and cfg.vis.enable_history_slider
    history = [] if collect_history else None
    topologies = [(tris, boundary_idx)]
    epoch = 0

    def build_window(v_, t_, b_):
        """Create the PyVista window, charts and scene objects for the mesh
        (v_, t_, b_); returns (live_ctx, mesh, actor)."""
        plotter, mesh, actor, volume_text = make_plotter(v_, t_, b_, cfg.vis)
        if cfg.vis.show_substrate:
            make_substrate_mesh(plotter, cfg.vis.substrate_size, cfg.vis.substrate_grid_resolution)
        if cfg.roughness_field is not None and cfg.vis.show_roughness:
            make_roughness_surface(plotter, cfg.roughness_field, cfg.vis.substrate_size,
                                   cfg.vis.roughness_opacity, cfg.vis.roughness_colormap)
        if cfg.anomaly.enabled:
            make_anomaly_disc(plotter, cfg.anomaly, cfg.solver.contact_angle_deg, cfg.vis.substrate_size)
        set_gap_offset(cfg.obstacle.radius if cfg.obstacle.enabled else 0.0)
        chart, line, marker, dchart, dline, dmarker = make_energy_chart(plotter, chart_xrange, chart_xlabel)
        obstacle_mesh = make_obstacle_mesh(plotter, cfg.obstacle)
        obstacle_base_points = obstacle_mesh.points.copy() if obstacle_mesh is not None else None
        if obstacle_mesh is not None:
            # the sphere mesh is built at the origin; put it at its start position right away
            # (frames -- and with them update_obstacle_mesh -- only come after the first equilibration)
            update_obstacle_mesh(obstacle_mesh, sphere_center_at(cfg.obstacle, 0.0), obstacle_base_points)

        if not cfg.vis.off_screen:
            # Opens the real interactive window (only meaningful with a
            # display, i.e. on your own machine). auto_close=False and
            # interactive_update=True let the Python code keep running
            # while the window stays open.
            plotter.show(title="Droplet Relaxer", interactive_update=True, auto_close=False)
            if cfg.vis.pan_with_right_button:
                enable_right_button_pan(plotter)

        ctx = {
            "plotter": plotter, "mesh": mesh, "obstacle_mesh": obstacle_mesh,
            "obstacle_base_points": obstacle_base_points, "chart": chart, "line": line,
            "marker": marker, "dchart": dchart, "dline": dline, "dmarker": dmarker,
            "wetting_actor": None, "volume_text": volume_text, "ring_actor": None,
            "com_hist": [] if use_quasi else [center_of_mass(v_, t_)],
            "force_hist": [] if use_quasi else [np.zeros(3)], "diag": {},
            "bar": ProgressBar(plotter), "total": total_iters,
        }
        update_com_text(plotter, center_of_mass(v_, t_), None)
        ctx["ring_actor"] = update_ring_outline(plotter, None, v_[b_])
        update_volume_text(volume_text, V0, V0)
        return ctx, mesh, actor

    # VisConfig.live_mode == "end_only": run without any window, record the
    # same frames, and open the interactive window (charts, history slider)
    # only when the relaxation is finished. Without a display it is the same
    # as live=False.
    end_only = cfg.vis.live and cfg.vis.live_mode == "end_only" and not cfg.vis.off_screen
    actor = None
    if end_only:
        live_ctx = {"headless": True, "diag": {},
                    "com_hist": [] if use_quasi else [center_of_mass(verts, tris)],
                    "force_hist": [] if use_quasi else [np.zeros(3)],
                    "progress": _Progress(total_iters), "total": total_iters}
    elif cfg.vis.live:
        live_ctx, mesh, actor = build_window(verts, tris, boundary_idx)

    if use_quasi:
        # only equilibrated frames are recorded (no frame for the starting cone)
        iters_hist, area_hist, center_hist, x_hist = [], [], [], []
    else:
        iters_hist, area_hist, center_hist = [0], [surface_free_energy(cfg, verts, tris, boundary_idx, A0)], [None]
        x_hist = [0.0]
    if history is not None and not use_quasi:
        history.append({"iteration": 0, "x": 0.0, "verts": verts.copy(), "epoch": epoch,
                         "sphere_center": None, "wetted_loop": None,
                         "area": area_hist[0], "volume": V0, "com": center_of_mass(verts, tris), "force": np.zeros(3),
                         "curvature": (mean_curvature(verts, tris, boundary_idx) if end_only
                                       else np.asarray(live_ctx["mesh"]["curvature"]).copy())})
    start_iter = 0
    last = (0, verts, A0, V0, None, None)

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
        last, remesh_kind = _run_until_remesh_or_done(
            cfg, verts, tris, boundary_idx, mask, start_iter, total_iters,
            V0, L0, check_split, check_collapse, live_ctx, iters_hist, area_hist, center_hist,
            history, epoch, quasi=quasi, x_hist=x_hist, frame_rows=frame_rows, t_start=t_start,
            cc_block_until=cc_block_until,
        )
        start_iter, verts, _, _, _, _ = last
        boundary_last = boundary_idx  # ring indices that belong to `last` (a remesh below renumbers)
        tris_last = tris

        if remesh_kind == "diverged":
            break

        if remesh_kind in ("split", "collapse", "contact_collapse"):
            # Remesh the open dome only; the flat base on the substrate
            # (substrate.py) is dropped here and rebuilt from the
            # contact-line ring right after (see the end of this block).
            orig_state = (verts, tris, boundary_idx)   # restored if the remesh turns out to be a no-op
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
                    # "contact_collapse" trigger below (checked only when
                    # NO split is needed) rarely gets a turn. Prune
                    # pathologically tiny contact-patch edges (see
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
                        _log(live_ctx, f"        [contact collapse after split: "
                              f"{n_pruned} -> {len(tris)} triangles]")
                kind = "local split"
            else:
                verts, tris, boundary_idx = subdivide_mesh(verts, tris, boundary_idx)
                kind = "global split"
            mask = build_dof_mask(len(verts), boundary_idx)
            split_count += 1
            _log(live_ctx, f"        [{kind} #{split_count} at iteration {start_iter}: "
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
                _log(live_ctx, f"        [collapse at iteration {start_iter}: no safe collapses "
                      f"found (likely just the apex spokes) -- disabling further collapse checks]")
            else:
                _log(live_ctx, f"        [collapse at iteration {start_iter}: "
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
                _log(live_ctx, f"        [contact collapse at iteration {start_iter}: "
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
            topologies.append((tris, boundary_idx))
            if live_ctx is not None and not live_ctx.get("headless", False):
                mesh, actor = replace_mesh(live_ctx["plotter"], actor, verts, tris, boundary_idx, cfg.vis)
                live_ctx["mesh"] = mesh
                # The remesh redraws the surface at its current (not yet recorded) state, so the
                # sphere and BOTH contact lines must follow it, otherwise they stay at the old
                # frame (ring at the substrate, no sphere line) while the bridge forms.
                sc_now = last[4]
                if sc_now is not None:
                    update_obstacle_mesh(live_ctx["obstacle_mesh"], sc_now, live_ctx["obstacle_base_points"])
                    live_ctx["wetting_actor"] = update_wetting_outline(
                        live_ctx["plotter"], live_ctx["wetting_actor"],
                        _wetted_loop_points(cfg, verts, tris, sc_now))
                live_ctx["ring_actor"] = update_ring_outline(
                    live_ctx["plotter"], live_ctx["ring_actor"], verts[boundary_idx])
                if not cfg.vis.off_screen:
                    live_ctx["plotter"].update()

    elapsed = time.perf_counter() - t0
    if end_only and live_ctx is not None and live_ctx.get("progress") is not None:
        live_ctx["progress"].update(last[0], epoch, force=True,
                                    frac=(1.0 if quasi.finished else None) if quasi is not None else None,
                                    eta=0.0 if quasi is not None else None)
        print()  # end of the progress line
    if quasi is not None:
        n_bad = quasi.unconverged
        print(f"Quasi-static run: {quasi.frames} frames, {quasi.total_iters_in_frames} iterations in the steps, "
              f"{n_bad} step(s) not converged" + (f", {quasi.rollbacks} rollback(s)" if cfg.quasi.adaptive else "")
              + ("" if quasi.finished else "  -- the path end was NOT reached (iteration limit or divergence)")
              + ".")
        if frame_rows:
            print(export.format_frame_table(frame_rows))
    if live_ctx is not None and live_ctx.get("bar") is not None:
        live_ctx["bar"].hide()   # the history slider takes its place

    if end_only and live_ctx is not None:
        # Build the window now and show the finished run: full chart data,
        # final mesh, sphere and contact lines; the history slider follows below.
        rec = live_ctx
        _, v_last, _, V_fin, c_fin, loop_fin = last
        v_fin = verts  # matches `tris`/`boundary_idx` even if a remesh ended the run
        if len(v_last) != len(v_fin):
            loop_fin = None  # indices of the contact loop belong to the pre-remesh mesh
        live_ctx, mesh, actor = build_window(v_fin, tris, boundary_idx)
        live_ctx["bar"].hide()
        live_ctx["com_hist"], live_ctx["force_hist"] = rec["com_hist"], rec["force_hist"]
        plotter = live_ctx["plotter"]
        update_mesh(mesh, v_fin, tris, boundary_idx)
        update_obstacle_mesh(live_ctx["obstacle_mesh"], c_fin, live_ctx["obstacle_base_points"])
        live_ctx["wetting_actor"] = update_wetting_outline(
            plotter, None, v_fin[loop_fin] if loop_fin is not None else None)
        live_ctx["ring_actor"] = update_ring_outline(plotter, live_ctx["ring_actor"], v_fin[boundary_idx])
        update_volume_text(live_ctx["volume_text"], V_fin, V0)
        if x_hist:
            update_com_text(plotter, rec["com_hist"][-1], c_fin)
            update_energy_chart(live_ctx["chart"], live_ctx["line"], x_hist, area_hist,
                                 marker=live_ctx["marker"], dchart=live_ctx["dchart"],
                                 dline=live_ctx["dline"], dmarker=live_ctx["dmarker"],
                                 forces=live_ctx["force_hist"])

    if live_ctx is not None:
        plotter, mesh = live_ctx["plotter"], live_ctx["mesh"]
        if cfg.vis.off_screen:
            # No display attached (automated test/server run) -- nothing
            # to keep open, so close and release resources right away.
            plotter.close()
        else:
            def _save_callback(record_video=None):
                import os
                import DropletRelaxer.config as _config_module
                name = time.strftime("droplet_run_%Y%m%d_%H%M%S")
                path = export.ask_save_path(name)
                if not path:
                    print("Save cancelled.")
                    return
                folder = export.save_run(
                    path, cfg, iters_hist, area_hist, center_hist, live_ctx["com_hist"],
                    forces=live_ctx["force_hist"], frame_rows=frame_rows,
                    config_source=getattr(_config_module, "__file__", None),
                    force_fn=_capillary_force, avg_window=1,
                    gap_offset=cfg.obstacle.radius if cfg.obstacle.enabled else 0.0)
                print(f"Run saved to {os.path.abspath(folder)}")
                if record_video is not None and cfg.vis.save_video:
                    print("Recording video (the window shows the replay) ...")
                    try:
                        vid = record_video(os.path.join(folder, "run.mp4"))
                        print(f"Video saved to {os.path.abspath(vid)}")
                    except Exception as err:
                        print(f"Video not saved: {err}")

            if history is not None and len(history) > 1:
                make_history_slider(plotter, history, topologies, mesh, actor,
                                     live_ctx["obstacle_mesh"], live_ctx["obstacle_base_points"],
                                     cfg.vis,
                                     initial_wetting_actor=live_ctx["wetting_actor"],
                                     initial_ring_actor=live_ctx["ring_actor"],
                                     marker=live_ctx["marker"], dmarker=live_ctx["dmarker"],
                                     volume_text=live_ctx["volume_text"], V0=V0,
                                     save_callback=_save_callback, x_label=chart_xlabel)
            # Keep the window open after the relaxation finishes: hand
            # control back to PyVista's blocking interactive loop, so
            # you can keep rotating/zooming (and scrubbing through the
            # history slider) until you close the window yourself (or
            # press 'q' inside it).
            print("Relaxation finished. The window stays open -- "
                  "rotate/zoom freely, close it (or press 'q') when done.")
            if cfg.vis.pan_with_right_button:
                # Belt-and-suspenders re-assertion (see
                # enable_right_button_pan's docstring for why this is
                # needed at all): idempotent, since the function reuses
                # the same style instance and self-healing observer
                # instead of creating a new one each time.
                enable_right_button_pan(plotter)
            plotter.show()

    i, verts_final, Af, Vf, _, _ = last
    rf = np.hypot(verts_final[boundary_last, 0], verts_final[boundary_last, 1]).mean()
    print(f"End:    {i} iterations in {elapsed:.2f}s")
    print(f"        Area Af={Af:.5f}  (decrease: {A0 - Af:.5f})")
    print(f"        Volume Vf={Vf:.5f}  (deviation: {abs(V0 - Vf):.2e})")
    print(f"        Contact radius rf={rf:.5f}  (target angle: {cfg.solver.contact_angle_deg:.1f} deg)")
    if epoch > 0:
        n_base = int(base_triangle_mask(verts_final, tris, boundary_idx).sum())
        print(f"        Remeshed {epoch}x, final resolution: "
              f"{len(verts_final)} vertices, {len(tris)} triangles"
              + (f" ({len(tris) - n_base} dome + {n_base} base)" if n_base else ""))
    return {"iteration": int(i), "energy": surface_free_energy(cfg, verts_final, tris_last, boundary_last, Af,
                                                              last[4], last[5]),
            "area": float(Af), "volume": float(Vf), "contact_radius": float(rf), "remeshes": int(epoch),
            "verts": verts_final, "tris": tris_last, "boundary_idx": boundary_last,
            "force": None if quasi is None or quasi.prev_F is None else quasi.prev_F.tolist(),
            "quasi_frames": None if quasi is None else int(quasi.frames)}


if __name__ == "__main__":
    main()
