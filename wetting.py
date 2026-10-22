"""
Dynamic wetting/contact-line tracking on the sphere obstacle.

A patch of the droplet surface that is in contact with the sphere is
treated as "wetted": those facets are the liquid/SOLID interface, not
liquid/vapor surface any more, so (exactly like the flat substrate,
whose wetted base is not part of the mesh at all) they must not carry
the liquid/vapor surface energy -- see solver.py, where their area
weight is set to zero. What remains is the genuine three-phase contact
LINE: the boundary loop of the wetted patch. It gets the Young's
equation energy  E = -kappa * A_patch  (kappa = cos(theta_Y)), where
A_patch is the area on the sphere enclosed by the loop, the exact
spherical counterpart of the substrate's shoelace formula
(energy.py:substrate_wall_energy_and_grad).

Not modeled: contact-angle hysteresis (the wetted set is a pure
function of the current geometry, no memory), and pinch-off into a
separate droplet (needs a real topology change, see README).

Robustness notes (each fixes a concrete failure seen in testing):
  - Vertex directions are normalized onto the unit sphere before
    computing the patch area. The turning-angle formula assumes unit
    vectors; feeding it raw (gap != 0) vectors produced forces 10-20x
    larger than the surface forces and made the run diverge.
  - The loop is oriented explicitly (wetted side = enclosed side).
    Without that the energy depended on the arbitrary traversal
    direction: the same loop gave E=-1.35 one way and E=+0.79 the
    other, i.e. a randomly flipped force.
  - "Wetted" means |gap| < snap_distance (both sides of the sphere
    surface), so vertices buried deep inside the ball (still being
    pushed out by the collision term) don't count as contact.
"""

from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def unique_edges(tris):
    """(E, 2) array of the mesh's undirected edges (sorted vertex pair
    per row, rows in lexicographic order). Compute once per mesh topology
    and pass it to compute_wetted_set. One integer key per edge instead of
    np.unique(axis=0) (same result, several times faster)."""
    tris = np.asarray(tris)
    e = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    n = int(e.max()) + 1
    key = np.unique(e[:, 0].astype(np.int64) * n + e[:, 1])
    return np.column_stack([key // n, key % n]).astype(tris.dtype, copy=False)


def compute_wetted_set(verts, tris, center, radius, snap_distance, edges=None):
    """Boolean mask over vertices: True for vertices within
    `snap_distance` of the sphere surface on EITHER side
    (|‖v-center‖ - radius| < snap_distance), restricted to the single
    largest connected component (via mesh edges). One component only,
    because the boundary-loop extraction below assumes one simple
    loop."""
    gap = np.linalg.norm(verts - center, axis=1) - radius
    near = np.abs(gap) < snap_distance
    if not near.any():
        return near

    if edges is None:
        edges = unique_edges(tris)
    sel = near[edges[:, 0]] & near[edges[:, 1]]
    e = edges[sel]
    n = len(verts)
    graph = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    best = np.bincount(labels[near]).argmax()
    return near & (labels == best)


def wetted_boundary_loop(wetted_mask, tris, verts, center):
    """Ordered vertex-index loop (cyclic array) forming the boundary of
    the wetted patch, oriented counter-clockwise as seen from outside
    the sphere (so the ENCLOSED side is the wetted patch), or None if
    there is no single clean loop (no contact, several loops, a pinch
    point, or the patch touching the mesh's own boundary ring). Callers
    then simply skip the wetting energy for that iteration.

    A triangle counts as inside the patch only when all 3 vertices are
    wetted; the loop is the set of edges that belong to exactly one
    inside triangle."""
    inside = wetted_mask[tris].all(axis=1)
    if not inside.any():
        return None
    t = tris[inside]
    directed = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    n = len(wetted_mask)
    key = np.sort(directed, axis=1)
    key = key[:, 0].astype(np.int64) * n + key[:, 1]
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    boundary = directed[counts[inv] == 1]
    if len(boundary) < 3:
        return None

    succ = {}
    for a, b in boundary:
        if a in succ:
            return None  # two outgoing boundary edges: pinch point
        succ[int(a)] = int(b)
    start = int(boundary[0, 0])
    loop, cur = [start], succ[start]
    while cur != start:
        loop.append(cur)
        cur = succ.get(cur)
        if cur is None or len(loop) > len(succ):
            return None
    if len(loop) != len(succ):
        return None  # several disjoint loops

    loop = np.array(loop)
    u = verts[loop] - center
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    c = u.mean(axis=0)
    cn = np.linalg.norm(c)
    if cn < 0.2:
        return None  # loop wraps most of the sphere: "inside" ambiguous
    winding = np.sum(np.cross(u, np.roll(u, -1, axis=0)) @ (c / cn))
    return loop if winding > 0 else loop[::-1].copy()


def _turning_angles(u):
    """Signed turning angle at every vertex of a closed loop of unit
    vectors: the angle between incoming and outgoing edge, both
    projected into the tangent plane at that vertex."""
    v_prev, v_next = np.roll(u, 1, axis=0), np.roll(u, -1, axis=0)
    t_in = (u - v_prev)
    t_in = t_in - np.sum(t_in * u, axis=1, keepdims=True) * u
    t_in /= np.maximum(np.linalg.norm(t_in, axis=1, keepdims=True), 1e-12)
    t_out = (v_next - u)
    t_out = t_out - np.sum(t_out * u, axis=1, keepdims=True) * u
    t_out /= np.maximum(np.linalg.norm(t_out, axis=1, keepdims=True), 1e-12)
    return np.arctan2(np.sum(u * np.cross(t_in, t_out), axis=1),
                      np.sum(t_in * t_out, axis=1))


def patch_solid_angle(u):
    """Solid angle enclosed by the counter-clockwise loop of unit
    vectors `u`, via the discrete Gauss-Bonnet formula
    2*pi - sum(turning angles). This is the gauge-free form of the
    monopole (Wu-Yang) vector-potential line integral: only angle
    differences along the loop enter, so there is no singular direction
    to keep away from the patch. Exact for geodesic edges, converging
    for chords (checked against the spherical-cap area formula)."""
    return 2.0 * np.pi - _turning_angles(u).sum()


@lru_cache(maxsize=64)
def _stencil_colors(n):
    """Colors for the loop vertices such that two vertices of the same
    color are at least 3 apart (cyclically). A turning angle depends on
    3 consecutive vertices, so perturbing all vertices of one color at
    once changes each turning angle through at most ONE perturbed
    vertex -- which lets the finite-difference gradient below cover the
    whole loop in a handful of vectorized evaluations."""
    colors = -np.ones(n, dtype=int)
    for j in range(n):
        forbidden = {colors[(j + d) % n] for d in (-2, -1, 1, 2)}
        c = 0
        while c in forbidden:
            c += 1
        colors[j] = c
    return colors


def sphere_wetting_energy_and_grad(verts, loop_idx, center, radius, kappa_sphere):
    """E = -kappa_sphere * A_patch for the contact loop `loop_idx`
    (oriented as returned by wetted_boundary_loop), A_patch = radius^2
    times the enclosed solid angle; kappa_sphere = cos(theta_Y). The
    energy depends only on the loop vertices' DIRECTIONS as seen from
    the sphere center, so the force on them is purely tangential to the
    sphere (how far they sit off the surface is the collision term's
    business).

    The gradient is a central finite difference, evaluated for the
    whole loop at once via _stencil_colors (cost independent of the
    loop length up to vectorization), instead of a messy closed-form
    derivative of the turning-angle sum."""
    idx = np.asarray(loop_idx)
    n = len(idx)
    raw = verts[idx] - center
    u = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    A_patch = radius ** 2 * patch_solid_angle(u)
    E = -kappa_sphere * A_patch

    colors = _stencil_colors(n)
    eps = 1e-6 * radius
    dturn = np.zeros((n, 3))  # d(sum of turning angles) / d vertex
    for c in range(colors.max() + 1):
        sel = np.nonzero(colors == c)[0]
        for d in range(3):
            rp, rm = raw.copy(), raw.copy()
            rp[sel, d] += eps
            rm[sel, d] -= eps
            tp = _turning_angles(rp / np.linalg.norm(rp, axis=1, keepdims=True))
            tm = _turning_angles(rm / np.linalg.norm(rm, axis=1, keepdims=True))
            dt = (tp - tm) / (2 * eps)
            dturn[sel, d] = (dt + np.roll(dt, 1) + np.roll(dt, -1))[sel]

    grad = np.zeros_like(verts)
    grad[idx] = kappa_sphere * radius ** 2 * dturn
    return E, grad
