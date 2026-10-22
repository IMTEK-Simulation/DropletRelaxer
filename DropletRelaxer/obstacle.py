"""
Movable rigid-sphere obstacle.

This is a *prototype*, not a full physical treatment: it gives the
droplet a smooth, differentiable penalty energy that (a) keeps mesh
vertices from penetrating the sphere and (b) weakly pulls nearby
vertices onto the sphere once they get close enough, so the surface
"sticks" and wraps around it instead of just sliding off -- an
approximate stand-in for a genuine contact-angle/wetting energy.

Why only approximate, unlike the substrate wall energy in energy.py:
the flat-substrate wetting term is an *exact* line integral (the
enclosed area of the contact line's projection, via Stokes' theorem),
because a plane has a simple global vector potential for its normal
field. A sphere doesn't have an equally simple closed-form line
integral for an *arbitrary* contact loop (this is the same
"holonomy"/solid-angle subtlety behind the Gauss-Bonnet theorem), so
this module uses a local, per-vertex penalty instead. It behaves
qualitatively like a contact angle (the `adhesion_strength` and
`capture_radius` parameters control how eagerly the droplet wets the
sphere), but it is not derived from Young's equation the way the
substrate term is. (Where the surface actually wets the sphere,
wetting.py takes over with an exact Young-angle contact-line energy;
this module only provides collision and long-range pull-in.) It does
not change the mesh's topology: the droplet deforms around the sphere,
but it cannot pinch off into a separate liquid bridge.
"""

import numpy as np


def path_length(obstacle_cfg):
    """Length of the straight sphere path center_start -> center_end."""
    return float(np.linalg.norm(np.array(obstacle_cfg.center_end, dtype=float)
                                - np.array(obstacle_cfg.center_start, dtype=float)))


def sphere_center_at(obstacle_cfg, frac):
    """Sphere centre at path fraction `frac` in [0, 1]: the straight line from
    `center_start` to `center_end`. Quasi-static mode (quasistatic.py) steps
    `frac` forward once the liquid has relaxed; the old iteration-driven mode
    uses frac = iteration / n_iterations."""
    frac = min(max(float(frac), 0.0), 1.0)
    c0 = np.array(obstacle_cfg.center_start, dtype=float)
    c1 = np.array(obstacle_cfg.center_end, dtype=float)
    return c0 + frac * (c1 - c0)


def sphere_contact_energy_and_grad(verts, center, radius, stiffness, adhesion, capture_radius):
    """Penalty energy against a rigid sphere at `center` with the given
    `radius`. Let gap = ||v - center|| - radius (the signed distance
    from a vertex to the sphere surface, negative when penetrating):

      - gap < 0 (penetrating): quadratic push-out penalty, weight
        `stiffness` -- same style as SolverConfig.volume_penalty.
      - 0 <= gap < capture_radius (nearby, outside): quadratic pull-in
        "adhesion" penalty, weight `adhesion` -- pulls the surface onto
        the sphere once it's close enough to be considered "in
        contact", instead of only ever pushing it away.

    Both terms share the same quadratic form k*gap^2, so their
    gradient is +-2*k*gap in the radial direction -- continuous at
    gap=0 (where both are zero), so the combined force has no
    discontinuity right at the sphere surface."""
    d = verts - center
    dist = np.linalg.norm(d, axis=1)
    n_hat = d / np.maximum(dist, 1e-12)[:, None]
    gap = dist - radius

    grad = np.zeros_like(verts)

    penetrating = gap < 0
    grad[penetrating] += (2.0 * stiffness * gap[penetrating])[:, None] * n_hat[penetrating]
    E_pen = stiffness * np.sum(gap[penetrating] ** 2)

    contact = (~penetrating) & (gap < capture_radius)
    grad[contact] += (2.0 * adhesion * gap[contact])[:, None] * n_hat[contact]
    E_adh = adhesion * np.sum(gap[contact] ** 2)

    return E_pen + E_adh, grad
