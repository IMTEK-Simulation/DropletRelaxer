"""Per-frame physical post-processing of a relaxation state: surface free
energy, centre of mass and its offset from the sphere. Everything a viewer
plots is computed here (by the driver), not in the viewer."""

import numpy as np

from .anomaly import anomaly_wall_energy_and_grad
from .energy import cross3
from .substrate import base_triangle_mask
from .wetting import compute_wetted_set, patch_solid_angle, wetted_boundary_loop


def surface_free_energy(cfg, verts, tris, boundary_idx, A_lv, sphere_center=None, wetted_loop=None):
    """Total interface energy of the closed droplet,

        E = gamma_lv*A_lv - cos(theta)*gamma_lv*A_sl  [ - cos(theta_s)*gamma_lv*A_sphere ],

    A_lv = liquid/vapor area (`A_lv`, as yielded by the solver: dome
    without the sphere-wetted facets), A_sl = area of the flat base on the
    substrate, A_sphere = R^2 * solid angle of the wetted patch on the
    sphere (only when the sphere is in contact; it vanishes for the default
    sphere angle of 90 deg). Evaluated from the yielded state only, so it is
    exactly the energy the solver's gradient belongs to."""
    g = cfg.solver.gamma_lv
    E = g * A_lv
    kappa = np.cos(np.radians(cfg.solver.contact_angle_deg))
    if abs(kappa) > 1e-12:
        bt = base_triangle_mask(verts, tris, boundary_idx)
        if bt.any():
            t = tris[bt]
            # signed (normal down): the shoelace area even for a folded base
            A_sl = -0.5 * cross3(verts[t[:, 1]] - verts[t[:, 0]],
                                 verts[t[:, 2]] - verts[t[:, 0]])[:, 2].sum()
            E -= kappa * g * A_sl
    if getattr(cfg, "anomaly", None) is not None and cfg.anomaly.enabled:
        E += anomaly_wall_energy_and_grad(verts, boundary_idx, cfg.anomaly, g, kappa)[0]
    if getattr(cfg, "roughness_field", None) is not None:
        E += cfg.roughness_field.energy_and_grad(verts, boundary_idx, g)[0]
    if (wetted_loop is not None and sphere_center is not None
            and cfg.obstacle.wetting_enabled):
        ks = np.cos(np.radians(cfg.obstacle.wetting_contact_angle_deg))
        if abs(ks) > 1e-12:
            u = verts[wetted_loop] - sphere_center
            u = u / np.linalg.norm(u, axis=1, keepdims=True)
            E -= ks * g * cfg.obstacle.radius ** 2 * patch_solid_angle(u)
    return float(E)


def com_xy_offset_to_sphere(com, sphere_center=None):
    """Offset of the centre of mass from the sphere centre in the xy plane,
    returned as the vector (dx, dy) = COM_xy - sphere_xy. Without a sphere
    (None) the reference is the axis x = y = 0."""
    c = np.asarray(com, dtype=float)
    ref = np.zeros(2) if sphere_center is None else np.asarray(sphere_center, dtype=float)[:2]
    return c[:2] - ref


def sphere_gap(sphere_center, radius):
    """Height of the sphere's lowest point above the substrate, z_center -
    radius (NaN without a sphere)."""
    return float("nan") if sphere_center is None else float(sphere_center[2]) - float(radius)


def wetted_loop_indices(cfg, verts, tris, sphere_center):
    """Vertex indices of the current three-phase loop on the sphere for a
    given mesh (same detection as solver.relax), or None without a contact
    patch. Used to report the contact line right after a remesh, when the loop
    of the last yielded state no longer matches the vertex numbering."""
    oc = cfg.obstacle
    if not (oc.enabled and oc.wetting_enabled) or sphere_center is None:
        return None
    wet = compute_wetted_set(verts, tris, sphere_center, oc.radius, oc.wetting_snap_distance)
    if wet.sum() < oc.wetting_min_patch_vertices or not wet[tris].all(axis=1).any():
        return None
    return wetted_boundary_loop(wet, tris, verts, sphere_center)


def format_frame_table(frames):
    """Console table of quasi-static frames (one per sphere step): tip
    position, iterations used by the step, whether it converged, surface free
    energy and the force of the liquid on the sphere."""
    if not frames:
        return ""
    lines = ["Equilibrium frames (tip position, force of the liquid on the sphere):",
             f"{'x':>8}{'y':>8}{'z':>8}{'step its':>10}{'conv':>6}{'energy':>14}{'Fx':>11}{'Fy':>11}{'Fz':>11}"]
    for f in frames:
        c, F = f.sphere_center, f.force
        lines.append(f"{c[0]:>8.2f}{c[1]:>8.2f}{c[2]:>8.2f}{f.step_iters:>10}"
                     f"{'yes' if f.converged else 'NO':>6}{f.energy:>14.4f}"
                     f"{F[0]:>11.4f}{F[1]:>11.4f}{F[2]:>11.4f}")
    return "\n".join(lines)
