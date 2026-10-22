"""Per-frame physical post-processing of a relaxation state: surface free
energy, centre of mass and its offset from the sphere. Everything a viewer
plots is computed here (by the driver), not in the viewer."""

import numpy as np

from .anomaly import anomaly_wall_energy_and_grad
from .energy import cross3
from .substrate import base_triangle_mask
from .wetting import patch_solid_angle


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
