"""
Mesh generation: cone starting shape with a pinned contact line (a
circle at z=0), built as a refined hemisphere triangulation.
"""

import numpy as np

from energy import cross3, row_norm, scatter_add_rows
from wetting import unique_edges


def make_hemisphere(n_theta, n_phi, radius):
    thetas = np.linspace(0, np.pi / 2, n_theta + 1)
    verts = [np.array([0.0, 0.0, radius])]
    ring_start = [0]
    for i in range(1, n_theta + 1):
        ring_start.append(len(verts))
        th = thetas[i]
        for j in range(n_phi):
            ph = 2 * np.pi * j / n_phi
            verts.append(np.array([radius * np.sin(th) * np.cos(ph),
                                    radius * np.sin(th) * np.sin(ph),
                                    radius * np.cos(th)]))
    verts = np.array(verts)

    tris = []
    r1 = ring_start[1]
    for j in range(n_phi):
        tris.append([0, r1 + j, r1 + (j + 1) % n_phi])
    for i in range(1, n_theta):
        r_top, r_bot = ring_start[i], ring_start[i + 1]
        for j in range(n_phi):
            t0, t1 = r_top + j, r_top + (j + 1) % n_phi
            b0, b1 = r_bot + j, r_bot + (j + 1) % n_phi
            tris.append([t0, b0, b1])
            tris.append([t0, b1, t1])

    boundary_idx = np.arange(ring_start[n_theta], ring_start[n_theta] + n_phi)
    return verts, np.array(tris), boundary_idx


def make_cone(n_theta, n_phi, radius, start_height):
    """Non-optimal starting shape with the same contact line as the
    hemisphere, but significantly larger surface area -- serves as a
    visible test of the relaxation."""
    verts, tris, boundary_idx = make_hemisphere(n_theta, n_phi, radius)
    for i in range(len(verts)):
        rho = np.hypot(verts[i, 0], verts[i, 1])
        frac = 1.0 - rho / radius
        verts[i, 2] = frac * start_height
    return verts, tris, boundary_idx


def make_disk_cone(n_rings, radius, start_height):
    """Cone-shaped starting surface on a quasi-uniform disk mesh:
    concentric rings with 6k vertices on ring k (k = 1..n_rings), zipped
    together into near-equilateral triangles (valence 6 almost
    everywhere, including the tip). Unlike the polar (theta, phi) grid
    of `make_cone`, no fan of dozens of skinny triangles meets at the
    tip -- which matters because the tip is exactly where the sphere
    obstacle first touches the droplet, and a valence-28 pole both
    piles vertices up there under the area flow and makes local
    refinement add dozens of vertices per layer.

    Returns (verts, tris, boundary_idx) like make_cone; boundary_idx is
    the outer ring in counter-clockwise order, all at z = 0, and the
    triangles are oriented with the normal pointing up/outward."""
    n = int(n_rings)
    ring_start, pts = [0], [(0.0, 0.0)]
    for k in range(1, n + 1):
        ring_start.append(len(pts))
        m = 6 * k
        for j in range(m):
            a = 2 * np.pi * j / m
            pts.append((radius * k / n * np.cos(a), radius * k / n * np.sin(a)))
    xy = np.array(pts)

    tris = []
    for k in range(n):
        m_in = 1 if k == 0 else 6 * k
        m_out = 6 * (k + 1)
        s_in, s_out = ring_start[k], ring_start[k + 1]
        i = j = 0
        while j < m_out or (i < m_in and m_in > 1):
            a_in = 2 * np.pi * (i + 1) / m_in if i < m_in else np.inf
            a_out = 2 * np.pi * (j + 1) / m_out if j < m_out else np.inf
            if a_in < a_out and i < m_in:
                tris.append([s_in + i % m_in, s_out + j % m_out, s_in + (i + 1) % m_in])
                i += 1
            else:
                tris.append([s_in + i % m_in, s_out + j % m_out, s_out + (j + 1) % m_out])
                j += 1
    tris = np.array(tris)
    # Orient every triangle counter-clockwise seen from +z (normal up).
    p0, p1, p2 = xy[tris[:, 0]], xy[tris[:, 1]], xy[tris[:, 2]]
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    flip = cross < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]

    rho = np.hypot(xy[:, 0], xy[:, 1])
    verts = np.column_stack([xy, (1.0 - rho / radius) * start_height])
    boundary_idx = np.arange(ring_start[n], ring_start[n] + 6 * n)
    return verts, tris, boundary_idx


def _build_shape(mesh_cfg):
    """The mesh at its natural size (radius/start_height as given,
    before any MeshConfig.target_volume rescaling)."""
    if getattr(mesh_cfg, "mesh_type", "polar") == "disk":
        return make_disk_cone(mesh_cfg.n_rings, mesh_cfg.radius, mesh_cfg.start_height)
    return make_cone(mesh_cfg.n_theta, mesh_cfg.n_phi, mesh_cfg.radius, mesh_cfg.start_height)


def resolve_volume_scale(mesh_cfg):
    """The single length-scale factor `s` that `make_start_mesh` applies
    to reach `mesh_cfg.target_volume` (1.0, a no-op, if that's unset).
    Exposed on its own (not just used internally) so main.py can apply
    the SAME `s` to the other config values that must scale consistently
    with physical size for the solver to stay numerically well-behaved
    at any scale -- see the long comment on MeshConfig.target_volume."""
    if not getattr(mesh_cfg, "target_volume", None):
        return 1.0
    verts, tris, _ = _build_shape(mesh_cfg)
    from energy import volume_and_grad  # local import: avoids a mesh<->energy import cycle
    v_now = volume_and_grad(verts, tris)[0]
    return (mesh_cfg.target_volume / v_now) ** (1.0 / 3.0)


def make_start_mesh(mesh_cfg):
    """Starting cone according to MeshConfig.mesh_type, uniformly
    rescaled to MeshConfig.target_volume if that's set (see
    resolve_volume_scale/its docstring on MeshConfig)."""
    verts, tris, boundary_idx = _build_shape(mesh_cfg)
    s = resolve_volume_scale(mesh_cfg)
    if s != 1.0:
        verts = verts * s
    if getattr(mesh_cfg, "base_mesh_enabled", False):
        # Close the fluid with a flat base mesh on the substrate (see
        # substrate.py). Its volume contribution is exactly zero (all its
        # vertices have z = 0), so V and the dome's dynamics are unchanged.
        from substrate import add_base_mesh
        verts, tris, boundary_idx = add_base_mesh(verts, tris, boundary_idx)
    return verts, tris, boundary_idx


def build_dof_mask(n_verts, boundary_idx):
    """Degrees of freedom: all (x,y,z) are free except the z-component
    of the boundary vertices (the contact line stays on the substrate
    z=0)."""
    mask = np.ones((n_verts, 3), dtype=bool)
    mask[boundary_idx, 2] = False
    return mask


def subdivide_mesh(verts, tris, boundary_idx):
    """Uniform 1-to-4 triangle subdivision: every triangle is split
    into 4 by inserting a vertex at each edge's midpoint. This is a
    simple, always-safe form of remeshing -- because each unique edge
    gets exactly one midpoint, shared by both triangles that border
    it, there is never a "T-junction" (a mismatched edge where one
    side is subdivided and the neighbor isn't), so the result is
    always a valid, watertight-consistent open surface.

    It is GLOBAL refinement: the WHOLE mesh gets denser, not just a
    stretched region near an obstacle -- see README ("Sphere obstacle
    (prototype)") for why a real local/adaptive scheme (only refining
    where the mesh is actually distorted) is more involved and not
    implemented here.

    Returns (new_verts, new_tris, new_boundary_idx). The boundary ring
    is extended with the new midpoint on each of its edges, in the
    same cyclic order as the input `boundary_idx`, with height fixed
    back to exactly z=0 (its two parent vertices are already there, so
    this only guards against floating-point drift)."""
    verts_list = list(verts)
    edge_midpoint = {}

    def get_midpoint(a, b):
        key = (a, b) if a < b else (b, a)
        idx = edge_midpoint.get(key)
        if idx is None:
            idx = len(verts_list)
            verts_list.append(0.5 * (verts[a] + verts[b]))
            edge_midpoint[key] = idx
        return idx

    new_tris = []
    for i, j, k in tris:
        m_ij = get_midpoint(i, j)
        m_jk = get_midpoint(j, k)
        m_ki = get_midpoint(k, i)
        new_tris.append([i, m_ij, m_ki])
        new_tris.append([j, m_jk, m_ij])
        new_tris.append([k, m_ki, m_jk])
        new_tris.append([m_ij, m_jk, m_ki])

    new_verts = np.array(verts_list)

    new_boundary = []
    n_b = len(boundary_idx)
    for idx in range(n_b):
        a, b = boundary_idx[idx], boundary_idx[(idx + 1) % n_b]
        new_boundary.append(a)
        new_boundary.append(get_midpoint(a, b))
    new_boundary_idx = np.array(new_boundary)
    new_verts[new_boundary_idx, 2] = 0.0

    return new_verts, np.array(new_tris), new_boundary_idx


def collapse_short_edges(verts, tris, boundary_idx, min_len, protect_center=None, protect_radius=0.0,
                         protect_min_len=None):
    """Collapses edges shorter than `min_len` by merging their two
    endpoints into one (at their midpoint) -- the complement of
    subdivide_mesh's splitting. Keeps triangle count from growing
    without bound after repeated splits, by pruning resolution back
    down wherever it isn't (or is no longer) needed -- typically most
    of the mesh far from an active obstacle.

    `protect_center`/`protect_radius`, when given, exclude any edge
    with an endpoint within `protect_radius` of `protect_center` (e.g.
    the sphere obstacle's CURRENT position) from the `min_len`
    threshold above -- a plain length-based collapse would otherwise
    erode resolution everywhere, including right where an active
    obstacle still needs it. This is what turns "global split, global
    collapse" into an approximation of genuinely local/adaptive
    remeshing: split still adds resolution everywhere, but collapse
    only prunes it back OUTSIDE the protected region at the `min_len`
    threshold, so extra resolution persists exactly where it's
    currently needed.

    `protect_min_len`, when given, applies a SEPARATE (normally much
    smaller) threshold inside the protected region instead of excluding
    it outright: an edge there still collapses if it is shorter than
    `protect_min_len`. Without this, an edge that ends up pathologically
    tiny inside the protected zone -- e.g. the wetted patch's contact
    line getting pinched by competing forces as it moves -- can never
    be pruned again even after the obstacle has moved on, which stalls
    the solver there (the trust-region cap shrinks with the shortest
    incident edge) and needlessly bloats the triangle count. Leave it
    at the default (None) to fully protect the region as before.

    Conservative and safe by construction, at the cost of collapsing
    fewer edges per pass than a fully general implementation could:
      - Never touches an edge with either endpoint on the boundary
        ring (the contact line) -- its vertex set and cyclic order are
        left completely untouched, so nothing here has to special-case
        the boundary the way subdivide_mesh does.
      - Only collapses an edge (a,b) when the "link condition" holds:
        a and b's neighbor sets intersect in EXACTLY the two apex
        vertices of the (at most two) triangles already adjacent to
        that edge. If they share anything else, collapsing would pinch
        the surface into a non-manifold shape, so that edge is skipped.
      - Only commits a collapse if it doesn't flip or degenerate any
        triangle that keeps a vertex after the collapse (checked by
        comparing each affected triangle's normal direction before and
        after) -- otherwise skipped.
      - Once a vertex is touched by a committed collapse (as either
        endpoint or apex), it and every edge touching it are frozen for
        the rest of this pass, so one pass never cascades through a
        whole neighborhood -- call it repeatedly (e.g. once per
        remesh-check) for further pruning.

    Returns (new_verts, new_tris, new_boundary_idx) -- vertex indices
    are renumbered (removed vertices leave gaps), so boundary_idx is
    remapped to match even though the boundary vertices themselves are
    never removed."""
    n_verts = verts.shape[0]
    tris = np.asarray(tris)
    boundary_idx = np.asarray(boundary_idx)

    # --- Collapse candidates, vectorized. Kept in the order in which the
    # edges first appear when scanning the triangles (the same order a
    # dict filled triangle by triangle would have), and sorted below with
    # a STABLE sort by length, so ties are broken exactly as before.
    e_all = np.stack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=1).reshape(-1, 2)
    e_all = np.sort(e_all, axis=1)
    # unique rows via one integer key per edge (much faster than
    # np.unique(axis=0), same result: sorted by (a, b), first occurrence)
    _key = e_all[:, 0].astype(np.int64) * n_verts + e_all[:, 1]
    _, first = np.unique(_key, return_index=True)
    e_uniq = e_all[first]
    e_uniq = e_uniq[np.argsort(first, kind="stable")]
    is_bnd = np.zeros(n_verts, dtype=bool)
    is_bnd[boundary_idx] = True
    keep = ~(is_bnd[e_uniq[:, 0]] | is_bnd[e_uniq[:, 1]])
    e_uniq = e_uniq[keep]
    d_all = row_norm(verts[e_uniq[:, 0]] - verts[e_uniq[:, 1]])
    if protect_center is not None:
        prot = row_norm(verts - np.asarray(protect_center, dtype=float)) < protect_radius
        in_zone = prot[e_uniq[:, 0]] | prot[e_uniq[:, 1]]
    else:
        in_zone = np.zeros(len(e_uniq), dtype=bool)
    if protect_min_len is not None:
        candidate = np.where(in_zone, d_all < protect_min_len, d_all < min_len)
    else:
        candidate = np.where(in_zone, False, d_all < min_len)
    cand_edges = e_uniq[candidate]
    # The sort key uses the scalar np.linalg.norm (only for the few
    # candidates): on regular meshes many edges have (almost) equal
    # lengths, and a last-bit difference to the vectorized norm above
    # would reorder them and hence change which collapses win.
    cand_len = np.array([np.linalg.norm(verts[x] - verts[y]) for x, y in cand_edges])

    if len(cand_edges) == 0:
        # Nothing to collapse: same output as a full pass would give.
        return (np.array(verts, dtype=float), np.array(tris),
                np.array(boundary_idx, dtype=np.int64))

    order = np.argsort(cand_len, kind="stable")
    cand_sorted = [(int(x), int(y)) for x, y in cand_edges[order]]

    # --- Connectivity as CSR arrays built with numpy once; the per-vertex
    # neighbour / incident-triangle sets are only materialised lazily for the
    # (few) vertices a candidate edge actually touches, instead of filling a
    # Python set for every vertex of the mesh. The original (pre-pass) data
    # is what the checks need: vertices touched by a committed collapse are
    # frozen, so they are never looked at again.
    m = len(tris)
    tri_ids = np.repeat(np.arange(m), 3)
    vcol = tris.reshape(-1)
    o = np.argsort(vcol, kind="stable")          # per vertex: ascending triangle index
    vt_flat, vt_ptr = tri_ids[o], np.searchsorted(vcol[o], np.arange(n_verts + 1))
    # neighbours: both directions of every edge
    ea, eb = e_all[first, 0], e_all[first, 1]
    src = np.concatenate([ea, eb])
    dst = np.concatenate([eb, ea])
    o2 = np.argsort(src, kind="stable")
    nb_flat, nb_ptr = dst[o2], np.searchsorted(src[o2], np.arange(n_verts + 1))
    nb_cache = {}
    vt_cache = {}

    def neighbors_of(v):
        s_ = nb_cache.get(v)
        if s_ is None:
            s_ = set(nb_flat[nb_ptr[v]:nb_ptr[v + 1]].tolist())
            nb_cache[v] = s_
        return s_

    def vert_tris_of(v):
        s_ = vt_cache.get(v)
        if s_ is None:
            s_ = set(vt_flat[vt_ptr[v]:vt_ptr[v + 1]].tolist())
            vt_cache[v] = s_
        return s_

    P = np.array(verts, dtype=float)
    T = np.array(tris)                       # live triangles (rows are remapped in place)
    removed_vertex = np.zeros(n_verts, dtype=bool)
    removed_tri = np.zeros(m, dtype=bool)
    frozen = set()

    for (a, b) in cand_sorted:
        if a in frozen or b in frozen:
            continue
        common = (neighbors_of(a) & neighbors_of(b)) - {a, b}
        # triangles containing both a and b, ascending index (as the dict of
        # first-seen edges used to list them), minus those already removed
        adj = [ti for ti in sorted(vert_tris_of(a) & vert_tris_of(b)) if not removed_tri[ti]]
        apexes = set()
        ok = True
        for ti in adj:
            tri = T[ti]
            others = [int(v) for v in tri if v != a and v != b]
            if len(others) != 1:
                ok = False
                break
            apexes.add(others[0])
        if not ok or len(adj) != 2 or apexes != common:
            continue  # link condition fails -- would create a non-manifold vertex/edge

        new_pos = 0.5 * (P[a] + P[b])

        # Every triangle that will still exist after the collapse (i.e.
        # not one of the two being removed) must neither flip nor
        # degenerate. The triangles at a and b come from the per-vertex
        # incidence sets (no scan over the whole mesh); the check is
        # done for all of them at once.
        candidate_tris = [ti for ti in (vert_tris_of(a) | vert_tris_of(b))
                          if not removed_tri[ti] and ti not in adj]
        if candidate_tris:
            Tc = T[candidate_tris]
            T2 = np.where(Tc == b, a, Tc)
            degenerate = ((T2[:, 0] == T2[:, 1]) | (T2[:, 1] == T2[:, 2])
                          | (T2[:, 0] == T2[:, 2])).any()
            if degenerate:
                continue
            n_before = cross3(P[Tc[:, 1]] - P[Tc[:, 0]], P[Tc[:, 2]] - P[Tc[:, 0]])
            old_a = P[a].copy()
            P[a] = new_pos
            n_after = cross3(P[T2[:, 1]] - P[T2[:, 0]], P[T2[:, 2]] - P[T2[:, 0]])
            P[a] = old_a
            if (np.einsum('ij,ij->i', n_before, n_after) <= 0).any():
                continue

        # Commit: move a to the midpoint, remap b -> a everywhere,
        # drop the two now-degenerate triangles adjacent to (a,b).
        P[a] = new_pos
        removed_vertex[b] = True
        for ti in adj:
            removed_tri[ti] = True
        if candidate_tris:
            T[candidate_tris] = T2
            vert_tris_of(a).update(candidate_tris)

        frozen |= {a, b} | apexes | neighbors_of(a) | neighbors_of(b)

    kept_tris = T[~removed_tri]
    remap = np.cumsum(~removed_vertex) - 1       # new index of every kept vertex
    new_verts = P[~removed_vertex]
    new_tris = remap[kept_tris]
    new_boundary_idx = remap[boundary_idx]

    return new_verts, new_tris, new_boundary_idx


def tangential_smooth_step(verts, tris, adj, deg, fixed, strength=0.3):
    """One vectorized pass of TANGENTIAL-ONLY Laplacian smoothing: every
    non-fixed vertex is moved partway toward its neighbors' centroid, but
    only along the surface there -- the component of that offset along the
    vertex's own (area-weighted) normal is removed first. This equalizes
    triangle sizes and angles without noticeably changing the surface's
    shape, area or volume (unlike plain Laplacian smoothing, which would
    also flatten real curvature).

    `adj` is the sparse 0/1 vertex adjacency matrix, `deg` its row sums,
    `fixed` a boolean mask of vertices that must not move (contact line,
    vertices held on the sphere)."""
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    fn = cross3(p1 - p0, p2 - p0)
    vn = np.zeros_like(verts)
    for k in range(3):
        vn += scatter_add_rows(verts.shape[0], tris[:, k], fn)
    vn /= np.maximum(row_norm(vn)[:, None], 1e-300)
    offset = (adj @ verts) / np.maximum(deg, 1)[:, None] - verts
    offset -= np.sum(offset * vn, axis=1, keepdims=True) * vn
    offset[fixed] = 0.0
    return verts + strength * offset


def tangential_smooth(verts, tris, boundary_idx, strength=0.3, passes=1):
    """`tangential_smooth_step` applied `passes` times to every vertex
    except the contact-line ring. Run after each remesh event in main.py
    (a mesh-quality cleanup, not part of the physics); the solver also
    calls the single-pass version periodically while relaxing (see
    SolverConfig.tangential_smoothing_every)."""
    import scipy.sparse as sp
    n = verts.shape[0]
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    adj = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    adj = ((adj + adj.T) > 0).astype(float)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    fixed = np.zeros(n, dtype=bool)
    fixed[np.asarray(boundary_idx, dtype=int)] = True
    for _ in range(passes):
        verts = tangential_smooth_step(verts, tris, adj, deg, fixed, strength)
    return verts


def _all_edge_lengths(verts, tris):
    edges = unique_edges(tris)  # each undirected edge once
    return row_norm(verts[edges[:, 0]] - verts[edges[:, 1]])


def mean_edge_length(verts, tris):
    """Used as the reference scale for the remesh-trigger ratio in
    config.py (MeshConfig.remesh_trigger_ratio)."""
    return _all_edge_lengths(verts, tris).mean()


def max_edge_length(verts, tris):
    """Used as the live distortion metric that triggers a remesh (see
    subdivide_mesh) -- the longest edge currently in the mesh, checked
    against remesh_trigger_ratio times the ORIGINAL mean_edge_length."""
    return _all_edge_lengths(verts, tris).max()


def min_edge_length(verts, tris):
    """Informational only (used in the console log when a collapse
    pass runs) -- the single shortest edge currently in the mesh. NOT
    used as the collapse trigger: an "umbrella" mesh like this one's
    cone/hemisphere (a fan of triangles converging on one apex vertex)
    structurally always has a handful of very short "spoke" edges near
    the tip, regardless of any obstacle -- collapsing them is usually
    rejected by collapse_short_edges' own safety checks anyway (the
    apex is a high-valence vertex, so its neighboring edges tend to
    fail the link condition). Using the single minimum as a trigger
    would therefore fire immediately and permanently on that
    persistent, uncollapsible tip geometry alone. See
    short_edge_fraction for the metric actually used to trigger."""
    return _all_edge_lengths(verts, tris).min()


def short_edge_fraction(verts, tris, threshold):
    """Fraction of all edges shorter than `threshold` -- the metric
    used to trigger collapse_short_edges (see
    MeshConfig.remesh_min_edge_fraction). Unlike min_edge_length, a
    handful of persistently-short tip edges barely move this fraction,
    so it distinguishes "a genuinely over-refined region exists" from
    "the mesh has its usual few short spokes at the apex"."""
    lengths = _all_edge_lengths(verts, tris)
    return float(np.mean(lengths < threshold))


def refine_rivara(verts, tris, boundary_idx, max_len, fine_center=None, fine_radius=0.0,
                  fine_len=None, max_new_vertices=20000):
    """LOCAL refinement by Rivara longest-edge bisection -- the local
    counterpart of subdivide_mesh's global 1-to-4 split.

    Every triangle that has an edge longer than its target length gets
    refined. The target is `max_len` everywhere, or the smaller
    `fine_len` for edges whose midpoint lies within `fine_radius` of
    `fine_center` (e.g. the sphere obstacle's current position: extra
    resolution exactly where the contact is).

    How one triangle is refined (the classic algorithm): take its
    LONGEST edge e. If the triangle across e also has e as ITS longest
    edge (or e is a boundary edge), bisect both triangles at e's
    midpoint. Otherwise the neighbor's longest edge is a different,
    longer one -- so refine the neighbor first (recursively, with the
    same rule), which changes what lies across e, then try again. The
    propagation only ever walks towards longer edges, so it terminates,
    typically after one or two extra triangles, and the result is
    always conforming (no T-junctions) without touching the rest of the
    mesh. Bisecting at the midpoint of a straight edge is geometry-
    exact: area and enclosed volume are unchanged.

    Boundary handling: the contact-line ring (boundary_idx) is a
    boundary of the mesh; a bisected ring edge simply gets its midpoint
    inserted into the ring at the right place (the ring vertex list is
    returned in cyclic order again, longer).

    Returns (new_verts, new_tris, new_boundary_idx). Old vertex indices
    are unchanged (new vertices are appended), so per-vertex data can be
    carried over by extending it."""
    pos = [np.asarray(p, dtype=float) for p in verts]
    tri = {ti: tuple(int(x) for x in t) for ti, t in enumerate(tris)}
    next_tri = len(tri)
    edge_map = {}  # undirected edge -> set of triangle ids

    def key(a, b):
        return (a, b) if a < b else (b, a)

    def add_tri(t_id, t):
        tri[t_id] = t
        for k in range(3):
            edge_map.setdefault(key(t[k], t[(k + 1) % 3]), set()).add(t_id)

    def remove_tri(t_id):
        t = tri.pop(t_id)
        for k in range(3):
            e = key(t[k], t[(k + 1) % 3])
            edge_map[e].discard(t_id)
            if not edge_map[e]:
                del edge_map[e]

    for t_id in list(tri):
        t = tri.pop(t_id)
        add_tri(t_id, t)

    # Boundary ring as a successor map so that bisected ring edges can be
    # spliced in wherever they occur.
    ring = [int(b) for b in boundary_idx]
    ring_next = {ring[k]: ring[(k + 1) % len(ring)] for k in range(len(ring))}

    fine_center = None if fine_center is None else np.asarray(fine_center, dtype=float)

    def target_len(a, b):
        if fine_len is not None and fine_center is not None:
            mid = 0.5 * (pos[a] + pos[b])
            if np.linalg.norm(mid - fine_center) < fine_radius:
                return min(fine_len, max_len)
        return max_len

    def elen(a, b):
        return float(np.linalg.norm(pos[a] - pos[b]))

    def longest_edge(t):
        # Ties (equal lengths are common on regular meshes) are broken by
        # the vertex indices, giving all edges a strict total order. This
        # is essential: with arbitrary tie-breaking two neighbors can
        # each point at a different, equally long edge and the
        # propagation walks in a circle forever.
        best, best_rank = None, None
        for k in range(3):
            a, b = t[k], t[(k + 1) % 3]
            rank = (elen(a, b), key(a, b))
            if best_rank is None or rank > best_rank:
                best, best_rank = (a, b), rank
        return best, best_rank[0]

    def bisect(t_id, a, b, m):
        """Split triangle t_id at the midpoint m of its edge (a,b),
        preserving orientation."""
        nonlocal next_tri
        t = tri[t_id]
        k = next(k for k in range(3) if key(t[k], t[(k + 1) % 3]) == key(a, b))
        v0, v1, v2 = t[k], t[(k + 1) % 3], t[(k + 2) % 3]  # edge is (v0, v1), apex v2
        remove_tri(t_id)
        add_tri(t_id, (v0, m, v2))
        add_tri(next_tri, (m, v1, v2))
        next_tri += 1

    n_new = 0

    def refine_triangle(t_id):
        nonlocal n_new
        stack = [t_id]
        while stack:
            cur = stack[-1]
            if cur not in tri:
                stack.pop()
                continue
            (a, b), _ = longest_edge(tri[cur])
            e = key(a, b)
            others = [x for x in edge_map[e] if x != cur]
            if others:
                other = others[0]
                (oa, ob), _ = longest_edge(tri[other])
                if key(oa, ob) != e:
                    stack.append(other)  # neighbor must be refined first
                    continue
            # terminal edge (shared longest edge, or boundary): bisect
            m = len(pos)
            pos.append(0.5 * (pos[a] + pos[b]))
            n_new += 1
            for t2 in list(edge_map[e]):
                bisect(t2, a, b, m)
            if a in ring_next and ring_next[a] == b:
                ring_next[a], ring_next[m] = m, b
            elif b in ring_next and ring_next[b] == a:
                ring_next[b], ring_next[m] = m, a
            stack.pop()

    # Work list: repeatedly refine any triangle that still has an edge
    # above its target length. Each pass shrinks the excess, and the
    # cap keeps a pathological input from running away.
    def prescan(ids):
        """Vectorized, deliberately conservative version of the scalar
        `too_long` test below (a relative slack of 1e-9 so rounding can
        never hide a triangle): True for every triangle in `ids` that
        MIGHT have an edge above its target length. Existing vertices
        never move during refinement, so a triangle that is not flagged
        and has not been re-cut since is guaranteed not too long."""
        T = np.array([tri[i] for i in ids])
        Pm = np.array(pos)
        flag = np.zeros(len(ids), dtype=bool)
        for k in range(3):
            pa, pb = Pm[T[:, k]], Pm[T[:, (k + 1) % 3]]
            length = np.sqrt(np.einsum('ij,ij->i', pa - pb, pa - pb))
            target = np.full(len(ids), float(max_len))
            if fine_len is not None and fine_center is not None:
                mid = 0.5 * (pa + pb) - fine_center
                inside = np.sqrt(np.einsum('ij,ij->i', mid, mid)) < fine_radius
                target[inside] = min(fine_len, max_len)
            flag |= length > target * (1.0 - 1e-9)
        return dict(zip(ids, flag.tolist()))

    changed = True
    while changed and n_new < max_new_vertices:
        changed = False
        ids = list(tri)
        snap = {i: tri[i] for i in ids}
        maybe_long = prescan(ids) if ids else {}
        for t_id in ids:
            if t_id not in tri:
                continue
            t = tri[t_id]
            if t is snap[t_id] and not maybe_long[t_id]:
                continue  # untouched since the pre-scan and not too long
            too_long = any(elen(t[k], t[(k + 1) % 3]) > target_len(t[k], t[(k + 1) % 3])
                           for k in range(3))
            if too_long:
                refine_triangle(t_id)
                changed = True
                if n_new >= max_new_vertices:
                    break

    new_verts = np.array(pos)
    new_tris = np.array([tri[t] for t in sorted(tri)])
    new_boundary = [ring[0]]
    cur = ring_next[ring[0]]
    while cur != ring[0]:
        new_boundary.append(cur)
        cur = ring_next[cur]
    new_boundary = np.array(new_boundary)
    new_verts[new_boundary, 2] = 0.0
    return new_verts, new_tris, new_boundary


def long_edges_near(verts, tris, center, radius, max_len):
    """True if any edge whose midpoint lies within `radius` of `center`
    is longer than `max_len`. Used to trigger proactive refinement
    around the sphere obstacle before the surface there gets
    stretched."""
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    a, b = verts[e[:, 0]], verts[e[:, 1]]
    near = np.linalg.norm(0.5 * (a + b) - np.asarray(center), axis=1) < radius
    return bool(np.any(np.linalg.norm(a - b, axis=1)[near] > max_len))


def short_edges_near(verts, tris, center, radius, min_len):
    """The collapse-side counterpart to `long_edges_near`: True if any
    edge whose midpoint lies within `radius` of `center` is shorter
    than `min_len`. Local (Rivara) refinement targets a deliberately
    fine but bounded length near the sphere (MeshConfig.
    remesh_contact_len_ratio); this catches the case where an edge
    there has become pathologically shorter than that -- e.g. the
    wetted patch's contact line getting pinched by competing forces as
    it moves -- so main.py can trigger a collapse pass there even
    though the mesh as a WHOLE is nowhere near the global collapse
    trigger (remesh_min_edge_fraction), which local refinement, being
    local, normally never reaches."""
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    a, b = verts[e[:, 0]], verts[e[:, 1]]
    near = np.linalg.norm(0.5 * (a + b) - np.asarray(center), axis=1) < radius
    return bool(np.any(np.linalg.norm(a - b, axis=1)[near] < min_len))
