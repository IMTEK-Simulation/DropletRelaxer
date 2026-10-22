"""
The wetted BASE of the droplet: a flat triangle mesh on the substrate
(z = 0) that closes the liquid surface, so the fluid is enclosed by a
genuinely closed mesh -- the liquid/vapor "dome" plus the liquid/solid
"base" -- joined along the contact-line ring (`boundary_idx`).

Why it is a separate module: it only needs numpy/scipy and is imported
by energy.py (curvature colouring), mesh.py (start mesh), solver.py and
main.py, so it must not import any of them.

Conventions
  - The base triangles are appended AFTER the dome triangles in `tris`
    and use the ring vertices as their boundary; the base's interior
    vertices are appended after the dome's vertices. The dome's vertex
    and triangle indices therefore do not change when a base is added.
  - The base carries NO independent degrees of freedom: its interior
    vertices are always the harmonic ("Tutte") embedding of the current
    ring polygon, recomputed from the ring every iteration
    (`BaseEmbedder`). With a convex ring polygon such an embedding is
    guaranteed free of flipped triangles. Base vertices lie exactly on
    the wall constraint z = 0.
  - Base and dome are told apart purely TOPOLOGICALLY (`base_triangle_mask`):
    cutting the closed mesh along the ring edges leaves two connected
    pieces; the one containing the highest vertex is the dome, the other
    the base. That way remeshing (which renumbers vertices and splits
    triangles) needs no extra bookkeeping. A mesh without a base (ring
    edges belong to one triangle only) simply yields "no base".
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components
from wetting import unique_edges

_MASK_CACHE = {}


def _ring_edge_codes(boundary_idx, n_codes):
    b = np.asarray(boundary_idx, dtype=np.int64)
    a, c = b, np.roll(b, -1)
    lo, hi = np.minimum(a, c), np.maximum(a, c)
    return lo * n_codes + hi


def base_triangle_mask(verts, tris, boundary_idx):
    """Boolean mask over `tris`: True for triangles of the wetted base
    (see the module docstring). All False if the mesh has no base.
    The result is cached per topology (the mesh is queried on every
    rendered frame)."""
    tris = np.ascontiguousarray(tris)
    n_t = len(tris)
    if boundary_idx is None or len(boundary_idx) < 3 or n_t == 0:
        return np.zeros(n_t, dtype=bool)
    bidx = np.ascontiguousarray(np.asarray(boundary_idx, dtype=np.int64))
    key = (tris.shape, hash(tris.tobytes()), hash(bidx.tobytes()))
    hit = _MASK_CACHE.get(key)
    if hit is not None:
        return hit

    nv = int(tris.max()) + 1
    e = np.stack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=1).reshape(-1, 2)
    e = np.sort(e, axis=1).astype(np.int64)
    code = e[:, 0] * nv + e[:, 1]
    tri_of = np.repeat(np.arange(n_t), 3)
    order = np.argsort(code, kind="stable")
    cs, ts = code[order], tri_of[order]
    same = cs[:-1] == cs[1:]
    t1, t2, c1 = ts[:-1][same], ts[1:][same], cs[:-1][same]
    not_ring = ~np.isin(c1, _ring_edge_codes(bidx, nv))
    t1, t2 = t1[not_ring], t2[not_ring]
    graph = sp.coo_matrix((np.ones(len(t1)), (t1, t2)), shape=(n_t, n_t))
    _, labels = connected_components(graph, directed=False)

    if labels.max() == 0:
        mask = np.zeros(n_t, dtype=bool)  # a single piece: no base
    else:
        v_apex = int(np.argmax(verts[:, 2]))
        t_apex = int(np.nonzero((tris == v_apex).any(axis=1))[0][0])
        mask = labels != labels[t_apex]

    if len(_MASK_CACHE) > 64:
        _MASK_CACHE.clear()
    _MASK_CACHE[key] = mask
    return mask


def _add_base_canonical(verts, tris, boundary_idx, edge_len=None):
    """Closes an open dome mesh with a flat base triangulation.

    The base's boundary is the existing contact-line ring (same vertex
    indices, no duplicates). The CONNECTIVITY is built on a canonical
    circle: the ring vertices are placed on the unit circle at angles
    proportional to their arc length along the actual ring, the interior
    vertices are laid out as concentric hexagonal rings of 6k points
    (k = 1..n-1, plus the centre), and the canonical points are
    Delaunay-triangulated (a convex boundary, so every ring edge is
    guaranteed to be an edge of exactly one triangle -- unlike a
    Delaunay triangulation of the actual, possibly non-convex ring).
    The interior vertices are then placed on the real ring by the
    harmonic embedding (BaseEmbedder). Triangles are oriented with the
    normal pointing DOWN (outward from the liquid).

    The interior resolution follows the dome's typical edge length
    (`edge_len`, default: mean edge length of `tris`), capped by the
    ring's own vertex spacing, so a finely resolved contact line does
    not force a needlessly fine base.

    Returns (verts, tris, boundary_idx) with the base appended (the
    dome's indices are unchanged)."""
    from scipy.spatial import Delaunay

    ring = np.asarray(boundary_idx, dtype=int)
    ring_xy = verts[ring, :2]
    seg = np.hypot(*(np.roll(ring_xy, -1, axis=0) - ring_xy).T)
    total = float(seg.sum())
    orient = 1.0 if 0.5 * np.sum(ring_xy[:, 0] * np.roll(ring_xy[:, 1], -1)
                                 - np.roll(ring_xy[:, 0], -1) * ring_xy[:, 1]) > 0 else -1.0
    cum = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    ang = orient * 2 * np.pi * cum / total
    ring_c = np.column_stack([np.cos(ang), np.sin(ang)])

    if edge_len is None:
        e = unique_edges(tris)
        edge_len = float(np.linalg.norm(verts[e[:, 0]] - verts[e[:, 1]], axis=1).mean())
    r_eq = total / (2 * np.pi)
    n = int(np.clip(round(r_eq / edge_len), 2, max(2, round(len(ring) / 6))))

    inner = [np.zeros((1, 2))]
    for k in range(1, n):
        m = 6 * k
        a = 2 * np.pi * np.arange(m) / m
        inner.append(np.column_stack([k / n * np.cos(a), k / n * np.sin(a)]))
    inner = np.vstack(inner)
    pts = np.vstack([ring_c, inner])

    t = Delaunay(pts).simplices.copy()
    p0, p1, p2 = pts[t[:, 0]], pts[t[:, 1]], pts[t[:, 2]]
    cz = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    keep = np.abs(cz) > 1e-12
    t, cz = t[keep], cz[keep]
    # canonical circle has the ring's orientation -> clockwise seen from +z
    # (normal down) means cz*orient < 0
    flip = cz * orient > 0
    t[flip] = t[flip][:, [0, 2, 1]]

    n_dome = len(verts)
    mapping = np.empty(len(pts), dtype=int)
    mapping[:len(ring)] = ring
    mapping[len(ring):] = n_dome + np.arange(len(inner))
    base_tris = mapping[t]

    # Sanity: every ring edge must be an edge of exactly one base triangle.
    nn = n_dome + len(inner)
    be = np.sort(np.vstack([base_tris[:, [0, 1]], base_tris[:, [1, 2]], base_tris[:, [2, 0]]]), axis=1)
    keys, counts = np.unique(be[:, 0] * nn + be[:, 1], return_counts=True)
    re = np.sort(np.column_stack([ring, np.roll(ring, -1)]), axis=1)
    cnt = dict(zip(keys.tolist(), counts.tolist()))
    if any(cnt.get(int(k), 0) != 1 for k in (re[:, 0] * nn + re[:, 1])):
        raise ValueError("could not triangulate the substrate base: a ring edge is not "
                         "covered by exactly one base triangle")

    centre = ring_xy.mean(axis=0)
    new_verts = np.vstack([verts, np.column_stack([np.tile(centre, (len(inner), 1)),
                                                   np.zeros(len(inner))])])
    new_tris = np.vstack([tris, base_tris])
    emb = BaseEmbedder(new_tris, ring, base_triangle_mask(new_verts, new_tris, ring), len(new_verts))
    emb.apply(new_verts)

    # The embedding of a (nearly) convex ring cannot fold; check anyway.
    q0, q1, q2 = new_verts[base_tris[:, 0]], new_verts[base_tris[:, 1]], new_verts[base_tris[:, 2]]
    sa = (q1[:, 0] - q0[:, 0]) * (q2[:, 1] - q0[:, 1]) - (q1[:, 1] - q0[:, 1]) * (q2[:, 0] - q0[:, 0])
    if np.any(sa * orient >= 0):
        raise ValueError("substrate base mesh folded: the contact-line ring is too far from convex "
                         "for the harmonic embedding of the base")
    return new_verts, new_tris, np.asarray(boundary_idx)


def _ear_clip_base(verts, tris, boundary_idx):
    """Fallback for rings that are too far from convex for the harmonic
    embedding: triangulates the ring polygon itself by ear clipping (valid
    for ANY simple polygon), without interior vertices. Every ring edge
    belongs to exactly one base triangle; normals point down. The base is
    then just a closure of the mesh: the wall energy is evaluated with
    SIGNED areas (energy.area_and_grad, flat_down), i.e. it is exactly the
    shoelace area of the ring whatever the triangulation does later."""
    ring = np.asarray(boundary_idx, dtype=int)
    xy = verts[ring, :2]
    n = len(ring)
    area2 = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
    order = list(range(n)) if area2 > 0 else list(range(n))[::-1]   # counter-clockwise
    out = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    guard = 0
    while len(order) > 3:
        m = len(order)
        found = False
        for i in range(m):
            ia, ib, ic = order[i - 1], order[i], order[(i + 1) % m]
            a, b, c = xy[ia], xy[ib], xy[ic]
            if cross(a, b, c) <= 1e-12 * max(1.0, abs(area2)):
                continue                              # reflex / degenerate corner
            others = np.array([k for k in order if k not in (ia, ib, ic)], dtype=int)
            if len(others):
                q = xy[others]
                d1 = (b[0] - a[0]) * (q[:, 1] - a[1]) - (b[1] - a[1]) * (q[:, 0] - a[0])
                d2 = (c[0] - b[0]) * (q[:, 1] - b[1]) - (c[1] - b[1]) * (q[:, 0] - b[0])
                d3 = (a[0] - c[0]) * (q[:, 1] - c[1]) - (a[1] - c[1]) * (q[:, 0] - c[0])
                if np.any((d1 >= 0) & (d2 >= 0) & (d3 >= 0)):
                    continue                          # another vertex inside: not an ear
            out.append((ia, ic, ib))                  # ccw (a,b,c) reversed -> normal down
            order.pop(i)
            found = True
            break
        guard += 1
        if not found or guard > 10 * n:
            raise ValueError("could not triangulate the substrate base: the contact-line "
                             "ring is not a simple polygon (self-intersecting)")
    ia, ib, ic = order
    out.append((ia, ic, ib))
    base_tris = ring[np.array(out)]
    return verts.copy(), np.vstack([tris, base_tris]), np.asarray(boundary_idx)


def add_base_mesh(verts, tris, boundary_idx, edge_len=None):
    """Closes an open dome mesh with a flat base (see _add_base_canonical
    for the regular construction: hexagonal interior + harmonic embedding).
    If the contact-line ring is too far from convex for that embedding, the
    base falls back to an interior-free ear-clipping triangulation of the
    ring (_ear_clip_base), so a droplet that has moved sideways and deformed
    its contact line no longer stops the run."""
    try:
        return _add_base_canonical(verts, tris, boundary_idx, edge_len)
    except ValueError:
        return _ear_clip_base(verts, tris, boundary_idx)


def strip_base(verts, tris, boundary_idx):
    """Inverse of add_base_mesh: drops the base triangles and the base's
    interior vertices, returning the open dome (verts, tris,
    boundary_idx) with vertices renumbered compactly (dome vertices
    keep their relative order, so nothing changes when the base was
    appended last). Returns the input unchanged if there is no base.

    main.py strips the base before a remesh (refinement/collapse then
    act on the dome only, exactly as for an open mesh, and are not
    slowed down by triangles that carry no dynamics) and builds a fresh
    base afterwards -- the base is fully determined by the ring."""
    mask = base_triangle_mask(verts, tris, boundary_idx)
    if not mask.any():
        return verts, tris, np.asarray(boundary_idx)
    dome = tris[~mask]
    used = np.zeros(len(verts), dtype=bool)
    used[dome.ravel()] = True
    new_index = np.cumsum(used) - 1
    return (verts[used], new_index[dome],
            new_index[np.asarray(boundary_idx, dtype=int)])


class BaseEmbedder:
    """Places the base's interior vertices as the harmonic (Tutte)
    embedding of the current ring polygon: every interior vertex is the
    average of its base neighbours, ring vertices are the fixed boundary
    values, all on z = 0. The sparse factorization depends only on the
    topology, so build one per mesh topology and call `apply` every
    iteration (a single back-substitution)."""

    def __init__(self, tris, boundary_idx, base_mask, n_verts):
        self.interior = np.zeros(0, dtype=int)
        self._lu = None
        if not base_mask.any():
            return
        tb = tris[base_mask]
        in_base = np.unique(tb)
        is_ring = np.zeros(n_verts, dtype=bool)
        is_ring[np.asarray(boundary_idx, dtype=int)] = True
        interior = in_base[~is_ring[in_base]]
        if len(interior) == 0:
            return
        e = unique_edges(tb)
        A = sp.coo_matrix((np.ones(2 * len(e)),
                           (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                          shape=(n_verts, n_verts)).tocsr()
        deg = np.asarray(A.sum(axis=1)).ravel()
        A_int = A[interior]                      # rows of the interior vertices
        L_ii = sp.diags(deg[interior]) - A_int[:, interior]
        self._lu = spla.splu(L_ii.tocsc())
        self._A_int = A_int
        self._is_int = np.zeros(n_verts, dtype=bool)
        self._is_int[interior] = True
        self.interior = interior

    def apply(self, verts):
        """Sets the interior base vertices (x, y from the current ring,
        z = 0), in place; returns `verts`."""
        if self._lu is None:
            return verts
        xy = verts[:, :2].copy()
        xy[self._is_int] = 0.0
        rhs = self._A_int @ xy
        verts[self.interior, :2] = self._lu.solve(rhs)
        verts[self.interior, 2] = 0.0
        return verts


def embed_base(verts, tris, boundary_idx):
    """One-shot convenience wrapper (used by main.py after a remesh):
    re-embeds the base interior of `verts` in place. No-op without a
    base."""
    mask = base_triangle_mask(verts, tris, boundary_idx)
    if not mask.any():
        return verts
    return BaseEmbedder(tris, boundary_idx, mask, len(verts)).apply(verts)
