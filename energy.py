"""
Energy functionals and their analytical gradients:
  - area (surface tension)
  - volume (for the penalty constraint)
  - mean curvature (derived from the area gradient, for coloring --
    see mathematical_background.pdf for the derivation)
  - graph Laplacian matrix (for the Sobolev preconditioning)
"""

import numpy as np
import scipy.sparse as sp

from substrate import base_triangle_mask


def cross3(a, b):
    """Row-wise cross product of two (n, 3) arrays. Same arithmetic as
    np.cross (so bit-identical results) but without its per-call
    overhead, which dominated the runtime for the small arrays used
    here (np.cross was called ~100k times per run)."""
    out = np.empty_like(a)
    out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return out


def row_norm(a):
    """Row-wise Euclidean norm of an (n, k) array (np.linalg.norm(a,
    axis=1) without its per-call overhead)."""
    return np.sqrt(np.einsum('ij,ij->i', a, a))


def scatter_add_rows(n, idx, values):
    """out[i] = sum of values[j] over all j with idx[j] == i, for an
    (m, 3) `values` -- the same result as np.add.at(zeros((n, 3)), idx,
    values) (identical summation order), several times faster."""
    out = np.empty((n, values.shape[1]))
    for d in range(values.shape[1]):
        out[:, d] = np.bincount(idx, weights=values[:, d], minlength=n)
    return out


def area_and_grad(verts, tris, weights=None, flat_down=None):
    """Surface area, its gradient, and the per-triangle areas.

    `weights` (optional, one number per triangle) scales each
    triangle's contribution to the returned total AND to the gradient;
    the returned per-triangle areas stay unweighted. solver.py uses
    weight 0 for facets lying in the wetted patch on the sphere: those
    are liquid/solid interface, not liquid/vapor surface, so they must
    not carry surface energy (see wetting.py).

    A (near-)degenerate triangle has zero area and an undefined normal;
    the division is guarded so that case yields a zero contribution
    instead of NaN -- one NaN here used to poison the whole mesh.

    `flat_down` (optional boolean mask over the triangles): these triangles
    lie on the substrate (the wetted base) and are counted with their SIGNED
    area with the normal fixed to (0, 0, -1). Their sum is then exactly the
    shoelace area of the contact-line ring (and so is its gradient) even if
    the base triangulation has folded (non-convex ring), where the usual
    |cross|/2 would count a folded triangle as positive area."""
    n = verts.shape[0]
    p1, p2, p3 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    cross = cross3(p2 - p1, p3 - p1)
    tri_area = 0.5 * row_norm(cross)
    N_hat = cross / (2 * np.maximum(tri_area, 1e-14))[:, None]
    if flat_down is not None and flat_down.any():
        N_hat[flat_down] = (0.0, 0.0, -1.0)
        tri_area = tri_area.copy()
        tri_area[flat_down] = 0.5 * cross[flat_down, 2] * -1.0

    grad = np.zeros_like(verts)
    for idx, a, b in [(tris[:, 0], p2, p3), (tris[:, 1], p3, p1), (tris[:, 2], p1, p2)]:
        contrib = 0.5 * cross3(a - b, N_hat)
        if weights is not None:
            contrib = contrib * weights[:, None]
        grad += scatter_add_rows(n, idx, contrib)

    total = tri_area.sum() if weights is None else float(np.sum(weights * tri_area))
    return total, grad, tri_area


def volume_and_grad(verts, tris):
    n = verts.shape[0]
    p1, p2, p3 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    c23 = cross3(p2, p3)
    V = np.sum(np.einsum('ij,ij->i', p1, c23)) / 6.0

    grad = np.zeros_like(verts)
    for idx, g in [(tris[:, 0], c23), (tris[:, 1], cross3(p3, p1)), (tris[:, 2], cross3(p1, p2))]:
        grad += scatter_add_rows(n, idx, g / 6.0)

    return V, grad


def center_of_mass(verts, tris):
    """Centre of mass of the enclosed liquid (uniform density), from the
    closed triangle mesh via the divergence theorem: sum over the signed
    tetrahedra (origin, p1, p2, p3) of V_t * (p1 + p2 + p3) / 4, divided by
    their total volume. The flat base lies in the plane z = 0 through the
    origin, so its tetrahedra have zero volume: the formula is exact for the
    closed mesh AND for the open dome alone. Returns (x, y, z)."""
    p1, p2, p3 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    vt = np.einsum('ij,ij->i', p1, cross3(p2, p3)) / 6.0
    V = vt.sum()
    if abs(V) < 1e-300:
        return np.full(3, np.nan)
    return (vt[:, None] * (p1 + p2 + p3) / 4.0).sum(axis=0) / V


def substrate_wall_energy_and_grad(verts, boundary_loop, kappa):
    """Wetting energy of the contact line against a flat substrate
    (z=0), from Young's equation:

        gamma_SV - gamma_SL = gamma_LV * cos(theta_Y)

    With the liquid/vapor surface tension normalized to 1 (as
    `area_and_grad` already does -- surface energy IS literally the
    area here), the substrate/vapor vs. substrate/liquid energy
    difference collapses into a single coefficient
    kappa = cos(theta_Y). Rather than needing an explicit substrate
    mesh, the standard trick (Stokes' theorem) rewrites the wetted-area
    contribution as a closed line integral around the contact line,
    which for a flat substrate is exactly the signed polygon area
    enclosed by the contact line's (x, y) projection (the "shoelace
    formula" -- the same divergence-theorem idea already used for
    `volume_and_grad`, just one dimension down):

        E_wall = -kappa * A_proj,   A_proj = 1/2 sum_i (x_i*y_{i+1} - x_{i+1}*y_i)

    kappa=0 (theta_Y=90 deg) means no wall energy at all: the contact
    line is then governed purely by the area/volume terms, which is
    exactly the model's previous (implicit) behavior. kappa>0 (theta_Y
    < 90 deg) rewards a larger wetted area -- the droplet spreads out.
    kappa<0 (theta_Y > 90 deg) rewards a smaller wetted area -- the
    droplet beads up and pulls the contact line inward.

    `boundary_loop` must be the contact-line vertex indices in cyclic
    (ring) order, e.g. the `boundary_idx` returned by `make_cone`."""
    idx = np.asarray(boundary_loop)
    x, y = verts[idx, 0], verts[idx, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    A_proj = 0.5 * np.sum(x * y_next - x_next * y)

    x_prev, y_prev = np.roll(x, 1), np.roll(y, 1)
    dA_dx = 0.5 * (y_next - y_prev)
    dA_dy = 0.5 * (x_prev - x_next)

    grad = np.zeros_like(verts)
    grad[idx, 0] = -kappa * dA_dx
    grad[idx, 1] = -kappa * dA_dy
    # No z-component: the substrate is flat, so this term never pulls
    # vertices out of the z=0 plane -- that constraint is enforced
    # separately (see solver.py), exactly like Evolver's CONSTRAINT
    # mechanism keeps the wall energy and the "stay on the wall"
    # projection as two independent pieces.

    return -kappa * A_proj, grad


def substrate_penetration_energy_and_grad(verts, stiffness):
    """Keeps the WHOLE surface (not just the contact line) from
    dipping below the substrate z=0 -- a one-sided quadratic penalty,
    same style as the sphere obstacle's non-penetration term. Only the
    contact-line ring is explicitly constrained to z=0 elsewhere in the
    solver; ordinarily nothing else ever approaches z=0 so this never
    activates, but a strong external force (e.g. the sphere obstacle
    pushing down on the surface) can otherwise fold part of the mesh
    underneath the substrate, which is unphysical."""
    z = verts[:, 2]
    below = z < 0.0
    grad = np.zeros_like(verts)
    grad[below, 2] = 2.0 * stiffness * z[below]
    return stiffness * np.sum(z[below] ** 2), grad


def mean_curvature(verts, tris, boundary_idx=None):
    """|H| per vertex, approximated from the area gradient:
    grad_A ~= 2*H*n*A_mixed  =>  H ~= |grad_A| / (2*A_mixed).

    This relation is only exact at interior vertices with a complete
    triangle fan ("1-ring"). Boundary vertices (the contact line) have
    an incomplete fan, so their raw area gradient picks up an extra
    contribution related to the geodesic curvature of the boundary
    curve itself, unrelated to the true surface curvature -- this
    shows up as a spuriously bright rim in the visualization. If
    `boundary_idx` is given, those vertices' displayed curvature is
    replaced by the average of their interior neighbors instead, so
    the coloring reflects the actual surface shape rather than this
    discretization artifact."""
    n_verts = verts.shape[0]
    # With a closed mesh (see substrate.py) only the free liquid/vapor
    # surface (the dome) has a curvature: the flat base is left out, and
    # its interior vertices get NaN (drawn in the plotter's NaN colour).
    base = base_triangle_mask(verts, tris, boundary_idx)
    if base.any():
        tris_all, tris = tris, tris[~base]
    _, gradA, tri_area = area_and_grad(verts, tris)
    A_mixed = np.zeros(n_verts)
    for col in range(3):
        np.add.at(A_mixed, tris[:, col], tri_area / 3.0)
    H = np.linalg.norm(gradA, axis=1) / (2 * np.maximum(A_mixed, 1e-12))

    if boundary_idx is not None and len(boundary_idx) > 0:
        H = _replace_boundary_curvature(H, tris, boundary_idx, n_verts)

    if base.any():
        used = np.zeros(n_verts, dtype=bool)
        used[tris.ravel()] = True
        H[~used] = np.nan

    return H


def _replace_boundary_curvature(H, tris, boundary_idx, n_verts):
    # Vectorized (no per-triangle Python loop): mean_curvature() runs on
    # every single rendered frame (every update_mesh() call, including
    # every frame the history slider's Play button steps through -- see
    # visualize.py), so this used to be the actual bottleneck behind
    # Play feeling like it nearly hangs the window, not anything in the
    # playback timing itself. The old version rebuilt a full
    # Python-list-of-sets adjacency graph over ALL triangles from
    # scratch on every call (~8ms at ~4000 triangles, purely in
    # interpreted Python) just to look up boundary vertices' neighbors.
    is_boundary = np.zeros(n_verts, dtype=bool)
    is_boundary[boundary_idx] = True

    # Every triangle edge, in both directions, as a flat (source,
    # destination) vertex-index array -- np.add.at then does the
    # per-vertex neighbor-sum/count that the Python loop did one
    # dict/set insertion at a time.
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    src, dst = edges[:, 0], edges[:, 1]

    # Only edges FROM a boundary vertex TO an interior one matter here
    # -- that's exactly what H_fixed averages over below.
    keep = is_boundary[src] & ~is_boundary[dst]
    src, dst = src[keep], dst[keep]

    sums = np.zeros(n_verts)
    counts = np.zeros(n_verts)
    np.add.at(sums, src, H[dst])
    np.add.at(counts, src, 1.0)

    H_fixed = H.copy()
    idx = np.asarray(boundary_idx)
    has_interior = counts[idx] > 0
    fixable = idx[has_interior]
    H_fixed[fixable] = sums[fixable] / counts[fixable]
    return H_fixed


def build_laplacian(n_verts, tris):
    """Simple graph Laplacian (unnormalized degree-adjacency matrix)
    over the mesh edges, used for the Sobolev preconditioning of the
    gradient."""
    edges = set()
    for t in tris:
        for a, b in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]:
            edges.add((min(a, b), max(a, b)))

    rows, cols, vals = [], [], []
    deg = np.zeros(n_verts)
    for a, b in edges:
        rows += [a, b]
        cols += [b, a]
        vals += [-1.0, -1.0]
        deg[a] += 1
        deg[b] += 1
    rows += list(range(n_verts))
    cols += list(range(n_verts))
    vals += list(deg)

    return sp.coo_matrix((vals, (rows, cols)), shape=(n_verts, n_verts)).tocsr()
