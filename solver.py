"""
Sobolev-preconditioned gradient descent for area minimization at fixed
volume (penalty formulation).

The solver is implemented as a generator: `relax()` yields the current
state after each iteration, so main.py can either visualize it live or
simply run it through to the end.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from energy import (
    area_and_grad, volume_and_grad, build_laplacian, row_norm,
    substrate_wall_energy_and_grad, substrate_penetration_energy_and_grad,
)
from mesh import tangential_smooth_step
from anomaly import anomaly_wall_energy_and_grad
from substrate import base_triangle_mask, BaseEmbedder
from obstacle import sphere_center_at, sphere_contact_energy_and_grad
from wetting import (
    unique_edges, compute_wetted_set, wetted_boundary_loop, sphere_wetting_energy_and_grad,
)


class SolverDiverged(RuntimeError):
    """Raised when the relaxation produced non-finite or absurdly large
    coordinates. Carries the iteration and the size of each force term
    at that point, so the cause is visible instead of the droplet just
    silently vanishing from the view (NaN/inf coordinates render as
    nothing)."""

    def __init__(self, iteration, reason, terms):
        self.iteration, self.reason, self.terms = iteration, reason, terms
        detail = ", ".join(f"{k}={v:.3g}" for k, v in terms.items())
        super().__init__(f"relaxation diverged at iteration {iteration}: {reason} "
                         f"[max force per term: {detail}]")


def relax(verts0, tris, boundary_idx, mask, solver_cfg, obstacle_cfg=None,
          start_iter=0, stop_iter=None, V_target=None, diag=None, anomaly_cfg=None, roughness=None, quasi=None):
    """Generator: yields (iteration, verts, area, volume, sphere_center,
    wetted_loop) for iterations start_iter+1 .. stop_iter (stop_iter
    defaults to solver_cfg.n_iterations, i.e. run to the end).
    `area` is the liquid/vapor surface area (facets wetting the sphere
    excluded). sphere_center is None whenever obstacle_cfg is None or
    disabled; wetted_loop is the current three-phase contact loop on the
    sphere (array of vertex indices, see wetting.py) or None when
    there is no contact. Also stops early if the gradient norm falls
    below grad_tol. Raises SolverDiverged instead of ever yielding
    non-finite coordinates.

    start_iter/stop_iter let main.py run the relaxation in segments
    (needed for remeshing, which changes the vertex and triangle arrays
    mid-run and therefore needs a fresh call to this generator)
    without resetting the overall schedule: the sphere's motion and the
    volume target are still evaluated against the FULL
    solver_cfg.n_iterations / the ORIGINAL starting volume -- pass
    V_target explicitly whenever start_iter > 0.

    `quasi` (optional quasistatic.QuasiStatic): event-driven sphere motion. The
    sphere stands at quasi.center() until the liquid has relaxed (quasi.observe
    reports "converged"; the iteration that converged is yielded with
    quasi.converged_now = True, the following iteration moves the sphere on) --
    instead of the position being a function of the iteration number. The
    generator returns when the path end has been converged (quasi.finished).

    `diag` (optional dict): after every iteration `diag["force"]` holds the
    force the liquid exerts on the sphere (3-vector, same units as the energy
    gradient), see the "force on the sphere" comment in the loop."""

    n_verts = verts0.shape[0]
    V0 = V_target if V_target is not None else volume_and_grad(verts0, tris)[0]
    stop_iter = solver_cfg.n_iterations if stop_iter is None else stop_iter

    # Closed mesh (see substrate.py): the flat base on the substrate is
    # part of `tris`, but it has no dynamics of its own -- its interior
    # vertices are re-embedded from the contact-line ring every
    # iteration -- so everything that defines the DOME's dynamics
    # (Sobolev smoothing, trust region, smoothing, wetted-patch
    # bookkeeping) is built from the dome triangles only, exactly as for
    # the open mesh.
    base_tri = base_triangle_mask(verts0, tris, boundary_idx)
    has_base = bool(base_tri.any())
    dome_tris = tris[~base_tri] if has_base else tris
    gamma = getattr(solver_cfg, "gamma_lv", 1.0)

    L = build_laplacian(n_verts, dome_tris)
    M = (sp.identity(n_verts, format="csr") + solver_cfg.sobolev_alpha * L).tocsc()
    solve = spla.factorized(M)

    kappa = np.cos(np.radians(solver_cfg.contact_angle_deg))
    use_obstacle = obstacle_cfg is not None and obstacle_cfg.enabled
    use_wetting = use_obstacle and getattr(obstacle_cfg, "wetting_enabled", False)
    kappa_sphere = (np.cos(np.radians(obstacle_cfg.wetting_contact_angle_deg))
                    if use_wetting else 0.0)
    n_iters = max(solver_cfg.n_iterations, 1)

    edges = unique_edges(dome_tris)
    e0, e1 = edges[:, 0], edges[:, 1]
    # Neighbor averaging operator (used to keep the interior of a wetted
    # patch, which has no surface energy of its own, from degenerating).
    adj = sp.coo_matrix((np.ones(2 * len(edges)),
                         (np.r_[e0, e1], np.r_[e1, e0])), shape=(n_verts, n_verts)).tocsr()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    max_step_frac = getattr(solver_cfg, "max_step_fraction", 0.0)
    smooth_every = getattr(solver_cfg, "tangential_smoothing_every", 0)
    smooth_strength = getattr(solver_cfg, "tangential_smoothing_strength", 0.3)
    adj_bin = (adj > 0).astype(float)
    # Fixed mesh topology for the whole relax() call -> precompute, once,
    # the grouping of edges by endpoint so the trust region's "shortest
    # incident edge per vertex" is a single reduceat instead of two
    # (slow) np.minimum.at scatter calls every iteration.
    _ends = np.concatenate([e0, e1])
    _eid = np.concatenate([np.arange(len(e0)), np.arange(len(e0))])
    _order = np.argsort(_ends, kind="stable")
    _ends_sorted, _eid_sorted = _ends[_order], _eid[_order]
    _starts = np.flatnonzero(np.r_[True, _ends_sorted[1:] != _ends_sorted[:-1]])
    _verts_with_edges = _ends_sorted[_starts]
    fixed_ring = np.zeros(n_verts, dtype=bool)
    fixed_ring[np.asarray(boundary_idx, dtype=int)] = True

    # Per-triangle interface energy weights: the free liquid/vapor dome
    # carries gamma_lv; the wetted substrate base carries the Young term
    # -gamma_lv*cos(theta) (E = gamma_lv*A_lv - cos(theta)*gamma_lv*A_sl).
    # Without a base mesh that Young term comes from the shoelace
    # formula on the ring instead (mathematically the same energy).
    tri_w0 = np.where(base_tri, -gamma * kappa, gamma)
    lv_mask = ~base_tri  # triangles that belong to the liquid/vapor surface
    if has_base:
        embedder = BaseEmbedder(tris, boundary_idx, base_tri, n_verts)
        base_vert = np.zeros(n_verts, dtype=bool)
        base_vert[np.unique(tris[base_tri])] = True  # base interior + ring
        base_int = np.zeros(n_verts, dtype=bool)
        base_int[embedder.interior] = True
        fixed_static = fixed_ring | base_int
    else:
        fixed_static = fixed_ring

    verts = verts0.copy()
    if has_base:
        embedder.apply(verts)
    if start_iter == 0:
        sphere_center = sphere_center_at(obstacle_cfg, 0.0) if use_obstacle else None
        A_start = area_and_grad(verts, tris)[2][lv_mask].sum()
        yield 0, verts.copy(), float(A_start), V0, sphere_center, None

    for i in range(start_iter + 1, stop_iter + 1):
        sphere_center = None
        wet = None
        wetted_loop = None
        tri_w = None

        if quasi is not None and quasi.converged_now:
            quasi.advance(verts)          # previous iteration was a recorded frame
            if quasi.finished:
                return
        if use_obstacle:
            sphere_center = (quasi.center() if quasi is not None
                             else sphere_center_at(obstacle_cfg, i / n_iters))

        if use_wetting:
            wet = compute_wetted_set(verts, tris, sphere_center, obstacle_cfg.radius,
                                     obstacle_cfg.wetting_snap_distance, edges)
            if wet.sum() >= obstacle_cfg.wetting_min_patch_vertices:
                inside = wet[tris].all(axis=1)
                if inside.any():
                    tri_w = np.where(inside, 0.0, 1.0)
                    wetted_loop = wetted_boundary_loop(wet, tris, verts, sphere_center)
            if tri_w is None:
                wet = None
            else:
                # Wrapping check: how far around the sphere does the
                # wetted patch reach? A genuine contact patch is a cap
                # (angular half-width well below 90 deg); vertices on
                # the far side of the sphere mean the surface has
                # closed around it.
                u = verts[wet] - sphere_center
                u /= np.linalg.norm(u, axis=1, keepdims=True)
                cdir = u.mean(axis=0)
                cn = np.linalg.norm(cdir)
                half_width = 180.0 if cn < 1e-9 else np.degrees(np.arccos(np.clip((u @ (cdir / cn)).min(), -1, 1)))
                if half_width > 115.0:
                    raise SolverDiverged(
                        i, f"the surface is wrapping around the sphere (contact patch spans "
                           f"{2 * half_width:.0f} deg). A single-sheet mesh cannot represent a sphere "
                           f"that has entered the liquid -- let the sphere touch the surface instead "
                           f"of passing through it", {"wetted_vertices": float(wet.sum())})

        w = tri_w0 if tri_w is None else tri_w0 * tri_w
        _, gA, tri_area = area_and_grad(verts, tris, w, base_tri if has_base else None)
        # `A` = geometric liquid/vapor area (dome, wetted sphere patch
        # excluded); the energy is gamma_lv * A (+ the wetting terms).
        A = float(np.sum(tri_area * (lv_mask if tri_w is None else lv_mask * tri_w)))
        V, gV = volume_and_grad(verts, tris)
        terms = {"area": np.abs(gA).max()}

        gE = gA + 2 * solver_cfg.volume_penalty * (V - V0) * gV
        terms["volume"] = np.abs(gV).max() * 2 * solver_cfg.volume_penalty * abs(V - V0)
        if not (0.5 * V0 < V < 1.5 * V0):
            raise SolverDiverged(
                i, f"volume {V:.3g} left the valid range around the target {V0:.3g} "
                   f"(the surface has folded or torn; the volume penalty would explode next)", terms)

        if abs(kappa) > 1e-12 and not has_base:
            # Wall energy at the flat substrate -- pulls the (already
            # free-sliding) contact line toward the target contact
            # angle instead of the neutral (90 deg) default. (With a base
            # mesh this term is carried by the base triangles' weight.)
            _, gW = substrate_wall_energy_and_grad(verts, boundary_idx, kappa)
            gE = gE + gamma * gW

        if anomaly_cfg is not None and getattr(anomaly_cfg, "enabled", False):
            # Circular wettability anomaly: extra Young term (kappa_a - kappa)
            # on the contact-line vertices only (anomaly.py); independent of
            # the sphere, so no contribution to the force on the sphere.
            _, gAn = anomaly_wall_energy_and_grad(verts, boundary_idx, anomaly_cfg, gamma, kappa)
            gE = gE + gAn
            terms["anomaly"] = np.abs(gAn).max()

        if roughness is not None:
            # Random roughness / chemical disorder (roughness.py): extra Young
            # term delta_kappa(x, y) on the wetted substrate area, evaluated as
            # an exact contact-line flux; acts on the contact-line vertices only
            # and is independent of the sphere (no force on the sphere).
            _, gRo = roughness.energy_and_grad(verts, boundary_idx, gamma)
            gE = gE + gRo
            terms["roughness"] = np.abs(gRo).max()

        if solver_cfg.substrate_stiffness > 0:
            _, gP = substrate_penetration_energy_and_grad(verts, solver_cfg.substrate_stiffness)
            gE = gE + gP
            terms["substrate"] = np.abs(gP).max()

        F_sphere = np.zeros(3)  # force of the liquid on the sphere (diag["force"])
        if use_obstacle:
            _, gS = sphere_contact_energy_and_grad(
                verts, sphere_center, obstacle_cfg.radius,
                obstacle_cfg.collision_stiffness, obstacle_cfg.adhesion_strength,
                obstacle_cfg.capture_radius,
            )
            if wet is not None:
                gS[wet] = 0.0  # wetted vertices are held ON the sphere by projection below
            gE = gE + gS
            terms["sphere"] = np.abs(gS).max()
            # Force on the sphere (virtual work, F = -dE/dc): the collision
            # penalty depends on (v - c), so F = +sum_i gS_i (Newton pair of
            # the push-out force -gS_i on the vertices).
            F_sphere += gS.sum(axis=0)

        if wetted_loop is not None:
            if abs(kappa_sphere) > 1e-12:
                _, gWet = sphere_wetting_energy_and_grad(
                    verts, wetted_loop, sphere_center, obstacle_cfg.radius, kappa_sphere)
                gE = gE + gamma * gWet
                terms["wetting"] = np.abs(gWet).max()
                # The Young term depends on the directions (v_i - c) only, so
                # dE/dc = -sum_i dE/dv_i: F = +gamma * sum_i gWet_i.
                F_sphere += gamma * gWet.sum(axis=0)
            else:
                # theta_Y = 90 deg -> kappa = cos(theta_Y) = 0: the wetting
                # energy E = -kappa*R^2*Omega and its gradient vanish
                # identically, so skip the (expensive, finite-difference)
                # evaluation. The loop itself is still needed for the
                # contact-line display and stays computed above.
                terms["wetting"] = 0.0

        if wet is not None:
            # The radial part of the force on a wetted vertex is the
            # sphere's reaction force (the constraint); only the
            # tangential part moves it. Removing it here keeps the
            # Sobolev smoothing from spreading it onto free neighbors.
            d = verts[wet] - sphere_center
            nrm = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
            rad = np.sum(gE[wet] * nrm, axis=1, keepdims=True) * nrm
            # Force on the sphere: the radial part of the energy gradient on a
            # wetted vertex is carried by the sphere (constraint reaction on
            # the vertex = +rad, so the liquid pushes on the sphere with
            # -rad). It contains the pressure on the wetted patch and the
            # radial pull of the free surface at the contact line.
            F_sphere -= rad.sum(axis=0)
            gE[wet] -= rad

        gE[boundary_idx, 2] = 0.0  # contact line stays on the substrate
        if has_base:
            gE[base_vert, 2] = 0.0  # ... and so does the whole base (wall constraint)

        if diag is not None:
            diag["force"] = F_sphere.copy()

        if np.linalg.norm(gE[mask]) < solver_cfg.grad_tol:
            if quasi is None:
                yield i, verts.copy(), A, V, sphere_center, wetted_loop
                return
            # quasi-static: the raw gradient has vanished, this step is converged
            quasi.observe(verts, F_sphere, 0.0, True, A, sphere_center, wetted_loop)
            yield i, verts.copy(), A, V, sphere_center, wetted_loop
            continue

        # Sobolev smoothing: (I + alpha*L)^-1 * gradient, componentwise
        g_smooth = solve(np.ascontiguousarray(gE))  # one call for all 3 columns
        g_smooth[boundary_idx, 2] = 0.0
        if has_base:
            g_smooth[base_int] = 0.0  # base interior is placed by the ring, not by forces

        step = solver_cfg.learning_rate * g_smooth
        if max_step_frac > 0:
            # Trust region: cap each vertex's move at a fraction of its
            # own shortest incident edge (see SolverConfig.max_step_fraction).
            el = row_norm(verts[e0] - verts[e1])
            shortest = np.full(n_verts, np.inf)
            shortest[_verts_with_edges] = np.minimum.reduceat(el[_eid_sorted], _starts)
            floor = 0.25 * np.median(el)  # a degenerate edge must not freeze its vertex
            cap = max_step_frac * np.maximum(shortest, floor)
            size = row_norm(step)
            step *= np.minimum(1.0, cap / np.maximum(size, 1e-300))[:, None]
        step_max = float(row_norm(step).max())

        verts = verts - step
        verts[boundary_idx, 2] = 0.0  # correct numerical drift

        if smooth_every > 0 and i % smooth_every == 0:
            # Mesh regularization: the area flow only defines the
            # NORMAL motion; its tangential part is a discretization
            # artifact that crowds vertices together (at the droplet's
            # apex, dozens end up within 1e-3 of each other), which
            # shrinks the trust-region cap there and stalls the
            # relaxation. Periodic tangential-only averaging keeps the
            # triangles evenly sized at negligible cost.
            fixed = fixed_static | wet if wet is not None else fixed_static
            verts = tangential_smooth_step(verts, tris, adj_bin, deg, fixed, smooth_strength)

        if has_base:
            embedder.apply(verts)  # base interior follows the (moved) contact-line ring

        if wet is not None:
            # The interior of the wetted patch has no surface energy, so
            # nothing keeps its triangles regular while the contact line
            # slides: relax them toward their neighbors' centroid (a
            # mesh-quality step, not physics), then put every wetted
            # vertex exactly onto the sphere.
            interior = wet & (np.asarray(adj @ wet.astype(float)) >= deg - 0.5)
            if interior.any():
                centroid = (adj @ verts) / np.maximum(deg, 1)[:, None]
                verts[interior] = 0.5 * verts[interior] + 0.5 * centroid[interior]
            d = verts[wet] - sphere_center
            verts[wet] = sphere_center + obstacle_cfg.radius * d / np.maximum(
                np.linalg.norm(d, axis=1, keepdims=True), 1e-12)

        if not np.all(np.isfinite(verts)):
            raise SolverDiverged(i, "non-finite coordinates", terms)
        if np.abs(verts).max() > 1e3:
            raise SolverDiverged(i, f"coordinates exploded (|x|max={np.abs(verts).max():.3g})", terms)

        if quasi is not None:
            status = quasi.observe(verts, F_sphere, step_max, False, A, sphere_center, wetted_loop)
            if status == "rollback":
                # step did not converge: back to the last converged state, smaller step
                verts = quasi.rollback()
                continue

        yield i, verts.copy(), A, V, sphere_center, wetted_loop
