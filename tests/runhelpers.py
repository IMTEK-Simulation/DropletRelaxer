"""Helpers for the slow tests: set up the same scaled problem as main.py and
run the solver on a FIXED mesh (no remeshing) so that results are repeatable."""
import copy

import numpy as np

import config as _config
from energy import area_and_grad, volume_and_grad
from mesh import make_start_mesh, build_dof_mask, resolve_volume_scale
from solver import relax
from substrate import base_triangle_mask


def fresh_config():
    return copy.deepcopy(_config.CONFIG)


def scaled_setup(cfg):
    """Mesh + the config rescaling of main.main() (volume_penalty, snap distance,
    capture radius, substrate size)."""
    s = resolve_volume_scale(cfg.mesh)
    if s != 1.0:
        cfg.solver.volume_penalty /= s ** 4
        cfg.obstacle.wetting_snap_distance *= s
        cfg.obstacle.capture_radius *= s
        cfg.vis.substrate_size *= s
    verts, tris, bidx = make_start_mesh(cfg.mesh)
    mask = build_dof_mask(len(verts), bidx)
    V0 = volume_and_grad(verts, tris)[0]
    return verts, tris, bidx, mask, V0


def run_solver(cfg, verts, tris, bidx, mask, V0, n_iter, start_iter=0, diag=None):
    """Runs relax() for `n_iter` iterations of a run of cfg.solver.n_iterations,
    returns the last yielded tuple and the diag dict."""
    diag = {} if diag is None else diag
    last = None
    for out in relax(verts.copy(), tris, bidx, mask, cfg.solver, cfg.obstacle, start_iter=start_iter,
                     stop_iter=start_iter + n_iter, V_target=V0, diag=diag, anomaly_cfg=cfg.anomaly,
                     roughness=getattr(cfg, "roughness_field", None)):
        last = out
    return last, diag


def static_cfg(angle_deg=90.0, sphere=False, n_iter=3000):
    cfg = fresh_config()
    cfg.solver.contact_angle_deg = angle_deg
    cfg.solver.n_iterations = n_iter
    cfg.anomaly.enabled = False
    cfg.obstacle.enabled = sphere
    cfg.quasi.enabled = False   # these tests drive relax() with a fixed iteration count
    cfg.vis.live = False
    return cfg


def cap_prediction(V, theta_deg):
    """Spherical cap of volume V and contact angle theta on a flat wall:
    returns (R, contact radius a, height h, centre height zc)."""
    th = np.radians(theta_deg)
    R = (3 * V / (np.pi * (2 + np.cos(th)) * (1 - np.cos(th)) ** 2)) ** (1 / 3)
    return R, R * np.sin(th), R * (1 - np.cos(th)), -R * np.cos(th)


def refine_global(verts, tris, bidx, mask_fn=build_dof_mask):
    """One global 1-to-4 subdivision of the dome, with the base rebuilt (what
    main.py does at a remesh)."""
    from mesh import subdivide_mesh, tangential_smooth
    from substrate import strip_base, add_base_mesh
    v, t, b = strip_base(verts, tris, bidx)
    v, t, b = subdivide_mesh(v, t, b)
    v = tangential_smooth(v, t, b)
    v, t, b = add_base_mesh(v, t, b)
    return v, t, b, mask_fn(len(v), b)
