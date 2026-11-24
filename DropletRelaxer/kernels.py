"""Numba CPU kernels for the optional accelerated solver.

Keep fast-math disabled. Python adapters in _fast handle reference fallbacks
where exact NumPy or BLAS rounding matters to solver decisions.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def pairwise_sum_1d(a):
    n = len(a)
    if n == 0:
        return 0.0
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[i]
        return res
    if n <= 128:
        r0 = a[0]
        r1 = a[1]
        r2 = a[2]
        r3 = a[3]
        r4 = a[4]
        r5 = a[5]
        r6 = a[6]
        r7 = a[7]
        limit = n - (n % 8)
        for i in range(8, limit, 8):
            r0 += a[i]
            r1 += a[i + 1]
            r2 += a[i + 2]
            r3 += a[i + 3]
            r4 += a[i + 4]
            r5 += a[i + 5]
            r6 += a[i + 6]
            r7 += a[i + 7]
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        for i in range(limit, n):
            res += a[i]
        return res
    half = n // 2
    n2 = half - (half % 8)
    return pairwise_sum_1d(a[:n2]) + pairwise_sum_1d(a[n2:])


@njit(cache=True, fastmath=False)
def area_and_grad_kernel(verts, tris, has_weights, weights, has_flat_down, flat_down):
    n = verts.shape[0]
    m = tris.shape[0]
    tri_area = np.empty(m, dtype=np.float64)
    g0 = np.zeros((n, 3), dtype=np.float64)
    g1 = np.zeros((n, 3), dtype=np.float64)
    g2 = np.zeros((n, 3), dtype=np.float64)

    for t in range(m):
        i0 = tris[t, 0]
        i1 = tris[t, 1]
        i2 = tris[t, 2]
        p1x, p1y, p1z = verts[i0, 0], verts[i0, 1], verts[i0, 2]
        p2x, p2y, p2z = verts[i1, 0], verts[i1, 1], verts[i1, 2]
        p3x, p3y, p3z = verts[i2, 0], verts[i2, 1], verts[i2, 2]

        ux, uy, uz = p2x - p1x, p2y - p1y, p2z - p1z
        vx, vy, vz = p3x - p1x, p3y - p1y, p3z - p1z
        cx = uy * vz - uz * vy
        cy = uz * vx - ux * vz
        cz = ux * vy - uy * vx

        ta = 0.5 * np.sqrt((cx * cx + cz * cz) + cy * cy)
        denom = 2.0 * (ta if ta >= 1e-14 else 1e-14)
        nx = cx / denom
        ny = cy / denom
        nz = cz / denom
        if has_flat_down and flat_down[t]:
            nx, ny, nz = 0.0, 0.0, -1.0
            ta = 0.5 * cz * -1.0
        tri_area[t] = ta

        w = weights[t] if has_weights else 1.0

        dx, dy, dz = p2x - p3x, p2y - p3y, p2z - p3z
        c0x = 0.5 * (dy * nz - dz * ny)
        c0y = 0.5 * (dz * nx - dx * nz)
        c0z = 0.5 * (dx * ny - dy * nx)
        if has_weights:
            c0x *= w
            c0y *= w
            c0z *= w
        g0[i0, 0] += c0x
        g0[i0, 1] += c0y
        g0[i0, 2] += c0z

        dx, dy, dz = p3x - p1x, p3y - p1y, p3z - p1z
        c1x = 0.5 * (dy * nz - dz * ny)
        c1y = 0.5 * (dz * nx - dx * nz)
        c1z = 0.5 * (dx * ny - dy * nx)
        if has_weights:
            c1x *= w
            c1y *= w
            c1z *= w
        g1[i1, 0] += c1x
        g1[i1, 1] += c1y
        g1[i1, 2] += c1z

        dx, dy, dz = p1x - p2x, p1y - p2y, p1z - p2z
        c2x = 0.5 * (dy * nz - dz * ny)
        c2y = 0.5 * (dz * nx - dx * nz)
        c2z = 0.5 * (dx * ny - dy * nx)
        if has_weights:
            c2x *= w
            c2y *= w
            c2z *= w
        g2[i2, 0] += c2x
        g2[i2, 1] += c2y
        g2[i2, 2] += c2z

    grad = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        grad[i, 0] = (g0[i, 0] + g1[i, 0]) + g2[i, 0]
        grad[i, 1] = (g0[i, 1] + g1[i, 1]) + g2[i, 1]
        grad[i, 2] = (g0[i, 2] + g1[i, 2]) + g2[i, 2]
    if not has_weights:
        total = pairwise_sum_1d(tri_area)
    else:
        w_ta = np.empty(m, dtype=np.float64)
        for t in range(m):
            w_ta[t] = weights[t] * tri_area[t]
        total = pairwise_sum_1d(w_ta)
    return total, grad, tri_area


@njit(cache=True, fastmath=False)
def volume_and_grad_kernel(verts, tris):
    n = verts.shape[0]
    m = tris.shape[0]
    vt = np.empty(m, dtype=np.float64)
    g0 = np.zeros((n, 3), dtype=np.float64)
    g1 = np.zeros((n, 3), dtype=np.float64)
    g2 = np.zeros((n, 3), dtype=np.float64)

    for t in range(m):
        i0 = tris[t, 0]
        i1 = tris[t, 1]
        i2 = tris[t, 2]
        p1x, p1y, p1z = verts[i0, 0], verts[i0, 1], verts[i0, 2]
        p2x, p2y, p2z = verts[i1, 0], verts[i1, 1], verts[i1, 2]
        p3x, p3y, p3z = verts[i2, 0], verts[i2, 1], verts[i2, 2]

        c23x = p2y * p3z - p2z * p3y
        c23y = p2z * p3x - p2x * p3z
        c23z = p2x * p3y - p2y * p3x
        vt[t] = (p1x * c23x + p1z * c23z) + p1y * c23y

        g0[i0, 0] += c23x / 6.0
        g0[i0, 1] += c23y / 6.0
        g0[i0, 2] += c23z / 6.0

        c31x = p3y * p1z - p3z * p1y
        c31y = p3z * p1x - p3x * p1z
        c31z = p3x * p1y - p3y * p1x
        g1[i1, 0] += c31x / 6.0
        g1[i1, 1] += c31y / 6.0
        g1[i1, 2] += c31z / 6.0

        c12x = p1y * p2z - p1z * p2y
        c12y = p1z * p2x - p1x * p2z
        c12z = p1x * p2y - p1y * p2x
        g2[i2, 0] += c12x / 6.0
        g2[i2, 1] += c12y / 6.0
        g2[i2, 2] += c12z / 6.0

    grad = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        grad[i, 0] = (g0[i, 0] + g1[i, 0]) + g2[i, 0]
        grad[i, 1] = (g0[i, 1] + g1[i, 1]) + g2[i, 1]
        grad[i, 2] = (g0[i, 2] + g1[i, 2]) + g2[i, 2]
    return pairwise_sum_1d(vt) / 6.0, grad


@njit(cache=True, fastmath=False)
def substrate_wall_kernel(verts, idx, kappa):
    nb = len(idx)
    grad = np.zeros_like(verts)
    shoelace = np.empty(nb, dtype=np.float64)
    for k in range(nb):
        kp = (k - 1) if k > 0 else (nb - 1)
        kn = (k + 1) if (k + 1) < nb else 0
        vi = idx[k]
        vp = idx[kp]
        vn = idx[kn]
        xk, yk = verts[vi, 0], verts[vi, 1]
        xn, yn = verts[vn, 0], verts[vn, 1]
        xp, yp = verts[vp, 0], verts[vp, 1]
        shoelace[k] = xk * yn - xn * yk
        grad[vi, 0] = -kappa * 0.5 * (yn - yp)
        grad[vi, 1] = -kappa * 0.5 * (xp - xn)
    return -kappa * (0.5 * pairwise_sum_1d(shoelace)), grad


@njit(cache=True, fastmath=False)
def substrate_penetration_kernel(verts, stiffness):
    n = verts.shape[0]
    grad = np.zeros_like(verts)
    first = 0
    while first < n and not verts[first, 2] < 0.0:
        first += 1
    if first == n:
        return stiffness * 0.0, grad
    terms = np.empty(n - first, dtype=np.float64)
    count = 0
    for i in range(first, n):
        z = verts[i, 2]
        if z < 0.0:
            grad[i, 2] = 2.0 * stiffness * z
            terms[count] = z * z
            count += 1
    return stiffness * pairwise_sum_1d(terms[:count]), grad


@njit(cache=True, fastmath=False)
def sphere_contact_kernel(verts, center, radius, stiffness, adhesion, capture_radius):
    n = verts.shape[0]
    cx, cy, cz = center[0], center[1], center[2]
    grad = np.zeros_like(verts)
    # Match NumPy's reductions over the compact mask selections, not padded zeros.
    pen = np.empty(n, dtype=np.float64)
    adh = np.empty(n, dtype=np.float64)
    n_pen = n_adh = 0
    for i in range(n):
        dx = verts[i, 0] - cx
        dy = verts[i, 1] - cy
        dz = verts[i, 2] - cz
        dist = np.sqrt((dx * dx + dy * dy) + dz * dz)
        denom = dist if dist >= 1e-12 else 1e-12
        nx = dx / denom
        ny = dy / denom
        nz = dz / denom
        gap = dist - radius
        if gap < 0.0:
            fac = 2.0 * stiffness * gap
            # The native path adds into a positive-zero buffer (so -0 + 0 -> +0).
            grad[i, 0] = 0.0 + fac * nx
            grad[i, 1] = 0.0 + fac * ny
            grad[i, 2] = 0.0 + fac * nz
            pen[n_pen] = gap * gap
            n_pen += 1
        elif gap < capture_radius:
            fac = 2.0 * adhesion * gap
            grad[i, 0] = 0.0 + fac * nx
            grad[i, 1] = 0.0 + fac * ny
            grad[i, 2] = 0.0 + fac * nz
            adh[n_adh] = gap * gap
            n_adh += 1
    return stiffness * pairwise_sum_1d(pen[:n_pen]) + adhesion * pairwise_sum_1d(adh[:n_adh]), grad


@njit(cache=True, fastmath=False)
def unique_edges_kernel(tris):
    m = tris.shape[0]
    if m == 0:
        return np.empty((0, 2), dtype=np.int64)
    max_v = 0
    for t in range(m):
        for k in range(3):
            v = tris[t, k]
            if v > max_v:
                max_v = v
    n = np.int64(max_v + 1)
    keys = np.empty(3 * m, dtype=np.int64)
    for t in range(m):
        i0, i1, i2 = np.int64(tris[t, 0]), np.int64(tris[t, 1]), np.int64(tris[t, 2])
        a, b = (i0, i1) if i0 < i1 else (i1, i0)
        keys[3 * t] = a * n + b
        a, b = (i1, i2) if i1 < i2 else (i2, i1)
        keys[3 * t + 1] = a * n + b
        a, b = (i2, i0) if i2 < i0 else (i0, i2)
        keys[3 * t + 2] = a * n + b
    keys.sort()
    u_count = 1
    prev = keys[0]
    for i in range(1, 3 * m):
        cur = keys[i]
        if cur != prev:
            u_count += 1
            prev = cur
    out = np.empty((u_count, 2), dtype=np.int64)
    prev = keys[0]
    out[0, 0] = prev // n
    out[0, 1] = prev % n
    idx = 1
    for i in range(1, 3 * m):
        cur = keys[i]
        if cur != prev:
            out[idx, 0] = cur // n
            out[idx, 1] = cur % n
            idx += 1
            prev = cur
    return out


@njit(cache=True, fastmath=False)
def compute_wetted_set_kernel(verts, center, radius, snap_distance, edges):
    n = verts.shape[0]
    cx, cy, cz = center[0], center[1], center[2]
    near = np.zeros(n, dtype=np.bool_)
    any_near = False
    for i in range(n):
        dx = verts[i, 0] - cx
        dy = verts[i, 1] - cy
        dz = verts[i, 2] - cz
        gap = np.sqrt((dx * dx + dy * dy) + dz * dz) - radius
        if np.abs(gap) < snap_distance:
            near[i] = True
            any_near = True
    if not any_near:
        return near

    parent = np.arange(n, dtype=np.int64)
    ne = edges.shape[0]
    for k in range(ne):
        a = edges[k, 0]
        b = edges[k, 1]
        if near[a] and near[b]:
            ra = a
            while parent[ra] != ra:
                parent[ra] = parent[parent[ra]]
                ra = parent[ra]
            rb = b
            while parent[rb] != rb:
                parent[rb] = parent[parent[rb]]
                rb = parent[rb]
            if ra != rb:
                if ra < rb:
                    parent[rb] = ra
                else:
                    parent[ra] = rb

    counts = np.zeros(n, dtype=np.int64)
    best_root = -1
    best_cnt = -1
    for i in range(n):
        if near[i]:
            r = i
            while parent[r] != r:
                r = parent[r]
            parent[i] = r
            counts[r] += 1
    for i in range(n):
        if counts[i] > best_cnt:
            best_cnt = counts[i]
            best_root = i
    out = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if near[i] and parent[i] == best_root:
            out[i] = True
    return out


@njit(cache=True, fastmath=False, error_model="numpy")
def wetted_loop_orient_kernel(loop, verts, center):
    n_bnd = len(loop)
    if n_bnd == 0:
        return np.empty(0, dtype=np.int64)
    u = np.empty((n_bnd, 3), dtype=np.float64)
    cx = cy = cz = 0.0
    scx, scy, scz = center[0], center[1], center[2]
    for i in range(n_bnd):
        v = loop[i]
        dx = verts[v, 0] - scx
        dy = verts[v, 1] - scy
        dz = verts[v, 2] - scz
        nrm = np.sqrt((dx * dx + dy * dy) + dz * dz)
        ux, uy, uz = dx / nrm, dy / nrm, dz / nrm
        u[i, 0], u[i, 1], u[i, 2] = ux, uy, uz
        cx += ux
        cy += uy
        cz += uz
    cx /= n_bnd
    cy /= n_bnd
    cz /= n_bnd
    cn = np.sqrt((cx * cx + cy * cy) + cz * cz)
    # A positive-length loop always returns a result except near an ambiguous
    # decision, where the wrapper needs the original NumPy/BLAS rounding.
    slack = 1e-12 * n_bnd
    if not np.isfinite(cn) or abs(cn - 0.2) <= slack:
        return None
    if cn < 0.2:
        return np.empty(0, dtype=np.int64)
    cx /= cn
    cy /= cn
    cz /= cn
    winding = 0.0
    for i in range(n_bnd):
        inext = (i + 1) if (i + 1) < n_bnd else 0
        ux, uy, uz = u[i, 0], u[i, 1], u[i, 2]
        vx, vy, vz = u[inext, 0], u[inext, 1], u[inext, 2]
        wx = uy * vz - uz * vy
        wy = uz * vx - ux * vz
        wz = ux * vy - uy * vx
        winding += (wx * cx + wy * cy) + wz * cz
    if abs(winding) <= slack:
        return None
    if winding > 0.0:
        return loop.copy()
    return loop[::-1].copy()


@njit(cache=True, fastmath=False)
def wetted_boundary_unoriented_kernel(wetted_mask, tris):
    n = len(wetted_mask)
    m = tris.shape[0]
    n_inside = 0
    for t in range(m):
        if wetted_mask[tris[t, 0]] and wetted_mask[tris[t, 1]] and wetted_mask[tris[t, 2]]:
            n_inside += 1
    if n_inside == 0:
        return np.empty(0, dtype=np.int64)

    n_dir = 3 * n_inside
    dir_a = np.empty(n_dir, dtype=np.int64)
    dir_b = np.empty(n_dir, dtype=np.int64)
    keys = np.empty(n_dir, dtype=np.int64)
    nn = np.int64(n)
    idx = 0
    for t in range(m):
        i0, i1, i2 = np.int64(tris[t, 0]), np.int64(tris[t, 1]), np.int64(tris[t, 2])
        if wetted_mask[i0] and wetted_mask[i1] and wetted_mask[i2]:
            dir_a[idx] = i0
            dir_b[idx] = i1
            keys[idx] = (i0 * nn + i1) if i0 < i1 else (i1 * nn + i0)

            dir_a[n_inside + idx] = i1
            dir_b[n_inside + idx] = i2
            keys[n_inside + idx] = (i1 * nn + i2) if i1 < i2 else (i2 * nn + i1)

            dir_a[2 * n_inside + idx] = i2
            dir_b[2 * n_inside + idx] = i0
            keys[2 * n_inside + idx] = (i2 * nn + i0) if i2 < i0 else (i0 * nn + i2)
            idx += 1

    order = np.argsort(keys)
    is_bnd = np.zeros(n_dir, dtype=np.bool_)
    for k in range(n_dir):
        cur_k = keys[order[k]]
        same_prev = (k > 0) and (keys[order[k - 1]] == cur_k)
        same_next = (k + 1 < n_dir) and (keys[order[k + 1]] == cur_k)
        if not same_prev and not same_next:
            is_bnd[order[k]] = True

    succ = np.full(n, -1, dtype=np.int64)
    n_bnd = 0
    start = -1
    for e in range(n_dir):
        if is_bnd[e]:
            a = dir_a[e]
            b = dir_b[e]
            if succ[a] != -1:
                return np.empty(0, dtype=np.int64)
            succ[a] = b
            if start == -1:
                start = a
            n_bnd += 1
    if n_bnd < 3:
        return np.empty(0, dtype=np.int64)

    loop = np.empty(n_bnd, dtype=np.int64)
    loop[0] = start
    cur = succ[start]
    cnt = 1
    while cur != start:
        if cur == -1 or cnt >= n_bnd:
            return np.empty(0, dtype=np.int64)
        loop[cnt] = cur
        cnt += 1
        cur = succ[cur]
    if cnt != n_bnd:
        return np.empty(0, dtype=np.int64)
    return loop


@njit(cache=True, fastmath=False)
def check_verts_bounds_kernel(verts):
    n = verts.shape[0]
    max_abs = 0.0
    for i in range(n):
        for d in range(3):
            v = verts[i, d]
            if not np.isfinite(v):
                return False, np.nan
            av = -v if v < 0.0 else v
            if av > max_abs:
                max_abs = av
    return True, max_abs


@njit(inline="always")
def _vec_at(a, i):
    return a[i, 0], a[i, 1], a[i, 2]


@njit(inline="always", error_model="numpy")
def _projected_tangent(a, b, u):
    tx, ty, tz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    dot = (tx * u[0] + ty * u[1]) + tz * u[2]
    tx -= dot * u[0]
    ty -= dot * u[1]
    tz -= dot * u[2]
    norm = np.sqrt((tx * tx + ty * ty) + tz * tz)
    if norm < 1e-12:
        norm = 1e-12
    return tx / norm, ty / norm, tz / norm


@njit(inline="always")
def _turn_components(u, ti, to):
    cx = ti[1] * to[2] - ti[2] * to[1]
    cy = ti[2] * to[0] - ti[0] * to[2]
    cz = ti[0] * to[1] - ti[1] * to[0]
    y = (u[0] * cx + u[1] * cy) + u[2] * cz
    x = (ti[0] * to[0] + ti[1] * to[1]) + ti[2] * to[2]
    return y, x


@njit(cache=True, fastmath=False, error_model="numpy")
def turning_angle_components_kernel(u):
    """Compute the arguments to NumPy's arctan2 without temporary vectors."""
    n = len(u)
    y = np.empty(n, dtype=np.float64)
    x = np.empty(n, dtype=np.float64)
    for i in range(n):
        prev = _vec_at(u, i - 1 if i > 0 else n - 1)
        cur = _vec_at(u, i)
        nex = _vec_at(u, i + 1 if i + 1 < n else 0)
        ti = _projected_tangent(prev, cur, cur)
        to = _projected_tangent(cur, nex, cur)
        y[i], x[i] = _turn_components(cur, ti, to)
    return y, x


@njit(cache=True, fastmath=False, error_model="numpy")
def sphere_wetting_components_kernel(raw, colors, n_colors, eps):
    """Batch the unperturbed angles and the three angles each perturbation changes."""
    n = len(raw)
    u = np.empty_like(raw)
    incoming = np.empty_like(raw)
    outgoing = np.empty_like(raw)
    # Leave a padding column so NumPy calls arctan2 on rows of length n,
    # just like the vectorized reference, rather than fusing all the rows.
    y = np.zeros((1 + 6 * n_colors, n + 1))
    x = np.zeros_like(y)
    for i in range(n):
        rx, ry, rz = _vec_at(raw, i)
        norm = np.sqrt((rx * rx + ry * ry) + rz * rz)
        u[i, 0], u[i, 1], u[i, 2] = rx / norm, ry / norm, rz / norm
    for i in range(n):
        prev = _vec_at(u, i - 1 if i > 0 else n - 1)
        cur = _vec_at(u, i)
        nex = _vec_at(u, i + 1 if i + 1 < n else 0)
        ti = _projected_tangent(prev, cur, cur)
        to = _projected_tangent(cur, nex, cur)
        incoming[i, 0], incoming[i, 1], incoming[i, 2] = ti
        outgoing[i, 0], outgoing[i, 1], outgoing[i, 2] = to
        y[0, i], x[0, i] = _turn_components(cur, ti, to)
    for i in range(n):
        p = i - 1 if i > 0 else n - 1
        q = i + 1 if i + 1 < n else 0
        prev, nex = _vec_at(u, p), _vec_at(u, q)
        ti_prev, to_next = _vec_at(incoming, p), _vec_at(outgoing, q)
        rx, ry, rz = _vec_at(raw, i)
        for d in range(3):
            for sign in range(2):
                delta = eps if sign == 0 else -eps
                px = rx + delta if d == 0 else rx
                py = ry + delta if d == 1 else ry
                pz = rz + delta if d == 2 else rz
                norm = np.sqrt((px * px + py * py) + pz * pz)
                cur = (px / norm, py / norm, pz / norm)
                row = 1 + 6 * colors[i] + 2 * d + sign
                # Other vertices of this color are at least three apart,
                # so their three-angle writes never collide.
                to_prev = _projected_tangent(prev, cur, prev)
                y[row, p], x[row, p] = _turn_components(prev, ti_prev, to_prev)
                ti = _projected_tangent(prev, cur, cur)
                to = _projected_tangent(cur, nex, cur)
                y[row, i], x[row, i] = _turn_components(cur, ti, to)
                ti_next = _projected_tangent(cur, nex, nex)
                y[row, q], x[row, q] = _turn_components(nex, ti_next, to_next)
    return y[:, :n], x[:, :n]


@njit(cache=True, fastmath=False, error_model="numpy")
def sphere_wetting_gradient_kernel(angles, colors, idx, n_verts, span, scale):
    """Assemble the centered differences in the same order as NumPy."""
    n = len(idx)
    grad = np.zeros((n_verts, 3), dtype=np.float64)
    for i in range(n):
        p = i - 1 if i > 0 else n - 1
        q = i + 1 if i + 1 < n else 0
        for d in range(3):
            a = 1 + 6 * colors[i] + 2 * d
            b = a + 1
            here = (angles[a, i] - angles[b, i]) / span
            prev = (angles[a, p] - angles[b, p]) / span
            nex = (angles[a, q] - angles[b, q]) / span
            grad[idx[i], d] = scale * ((here + prev) + nex)
    return grad


@njit(cache=True, fastmath=False)
def roughness_wetted_integral_kernel(verts, loop, x0, dx, N, d_grid, F_grid, gx, gw):
    n = len(loop)
    P = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        v = loop[i]
        P[i, 0] = verts[v, 0]
        P[i, 1] = verts[v, 1]

    sa_terms = np.empty(n, dtype=np.float64)
    e = np.empty((n, 2), dtype=np.float64)
    ua = np.empty(n, dtype=np.float64)
    ub = np.empty(n, dtype=np.float64)
    va = np.empty(n, dtype=np.float64)
    vb = np.empty(n, dtype=np.float64)
    cnt_u = np.empty(n, dtype=np.int64)
    cnt_v = np.empty(n, dtype=np.int64)
    first_u = np.empty(n, dtype=np.float64)
    first_v = np.empty(n, dtype=np.float64)
    Ku = 0
    Kv = 0
    for m in range(n):
        mn = (m + 1) if (m + 1) < n else 0
        sa_terms[m] = P[m, 0] * P[mn, 1] - P[mn, 0] * P[m, 1]
        e[m, 0] = P[mn, 0] - P[m, 0]
        e[m, 1] = P[mn, 1] - P[m, 1]
        ua[m] = (P[m, 0] - x0) / dx
        ub[m] = (P[m, 0] + e[m, 0] - x0) / dx
        va[m] = (P[m, 1] - x0) / dx
        vb[m] = (P[m, 1] + e[m, 1] - x0) / dx

        lou = ua[m] if ua[m] < ub[m] else ub[m]
        hiu = ub[m] if ua[m] < ub[m] else ua[m]
        fu = np.floor(lou) + 1.0
        cu = int(np.ceil(hiu) - fu)
        if cu < 0:
            cu = 0
        first_u[m] = fu
        cnt_u[m] = cu
        if cu > Ku:
            Ku = cu

        lov = va[m] if va[m] < vb[m] else vb[m]
        hiv = vb[m] if va[m] < vb[m] else va[m]
        fv = np.floor(lov) + 1.0
        cv = int(np.ceil(hiv) - fv)
        if cv < 0:
            cv = 0
        first_v[m] = fv
        cnt_v[m] = cv
        if cv > Kv:
            Kv = cv

    signed_area = 0.5 * pairwise_sum_1d(sa_terms)
    orient = 1.0 if signed_area >= 0.0 else -1.0

    n_s = 2 + Ku + Kv
    n_p = n_s - 1
    n_q = 4 * n_p
    term_I = np.empty(n * n_q, dtype=np.float64)
    row_start = np.empty(n_q, dtype=np.float64)
    row_end = np.empty(n_q, dtype=np.float64)
    wk_start = np.empty(n, dtype=np.float64)
    wk_end = np.empty(n, dtype=np.float64)
    S = np.empty(n_s, dtype=np.float64)
    u_max = float(N) - 1e-9
    outside = False

    for m in range(n):
        S[0] = 0.0
        S[1] = 1.0
        du = ub[m] - ua[m]
        safe_u = 1.0 if du == 0.0 else du
        for k in range(Ku):
            if k < cnt_u[m]:
                sk = (first_u[m] + k - ua[m]) / safe_u
                if sk < 0.0:
                    sk = 0.0
                elif sk > 1.0:
                    sk = 1.0
                S[2 + k] = sk
            else:
                S[2 + k] = 1.0
        dv = vb[m] - va[m]
        safe_v = 1.0 if dv == 0.0 else dv
        for k in range(Kv):
            if k < cnt_v[m]:
                sk = (first_v[m] + k - va[m]) / safe_v
                if sk < 0.0:
                    sk = 0.0
                elif sk > 1.0:
                    sk = 1.0
                S[2 + Ku + k] = sk
            else:
                S[2 + Ku + k] = 1.0
        S.sort()

        ax, ay = P[m, 0], P[m, 1]
        ex, ey = e[m, 0], e[m, 1]
        base_idx = m * n_q
        for p in range(n_p):
            lo = S[p]
            wid = S[p + 1] - lo
            for q in range(4):
                col = 4 * p + q
                s = lo + wid * gx[q]
                ws = wid * gw[q]
                X = ax + s * ex
                Y = ay + s * ey

                u = (X - x0) / dx
                v = (Y - x0) / dx
                if u < 0.0 or u > N or v < 0.0 or v > N:
                    outside = True
                if u < 0.0:
                    u = 0.0
                elif u > u_max:
                    u = u_max
                if v < 0.0:
                    v = 0.0
                elif v > u_max:
                    v = u_max
                ii = int(np.floor(u))
                jj = int(np.floor(v))
                tu = u - ii
                wv = v - jj

                d00 = d_grid[jj, ii]
                d01 = d_grid[jj, ii + 1]
                d10 = d_grid[jj + 1, ii]
                d11 = d_grid[jj + 1, ii + 1]

                row0 = F_grid[jj, ii] + dx * (d00 * tu + 0.5 * (d01 - d00) * tu * tu)
                row1 = F_grid[jj + 1, ii] + dx * (d10 * tu + 0.5 * (d11 - d10) * tu * tu)
                pot = (1.0 - wv) * row0 + wv * row1
                term_I[base_idx + col] = ws * pot * ey

                r0 = d00 * (1.0 - tu) + d01 * tu
                r1 = d10 * (1.0 - tu) + d11 * tu
                kap = (1.0 - wv) * r0 + wv * r1

                row_start[col] = ws * (1.0 - s) * kap
                row_end[col] = ws * s * kap

        wk_start[m] = pairwise_sum_1d(row_start)
        wk_end[m] = pairwise_sum_1d(row_end)

    I = orient * pairwise_sum_1d(term_I)
    grad = np.zeros_like(verts)
    for m in range(n):
        mp = (m - 1) if m > 0 else (n - 1)
        nx_m = e[m, 1]
        ny_m = -e[m, 0]
        nx_p = e[mp, 1]
        ny_p = -e[mp, 0]
        v = loop[m]
        grad[v, 0] = orient * (wk_start[m] * nx_m + wk_end[mp] * nx_p)
        grad[v, 1] = orient * (wk_start[m] * ny_m + wk_end[mp] * ny_p)
    return I, grad, outside


@njit(cache=True, fastmath=False)
def tangential_smooth_normals_kernel(verts, tris):
    n = verts.shape[0]
    m = tris.shape[0]
    g0 = np.zeros((n, 3), dtype=np.float64)
    g1 = np.zeros((n, 3), dtype=np.float64)
    g2 = np.zeros((n, 3), dtype=np.float64)

    for t in range(m):
        i0, i1, i2 = tris[t, 0], tris[t, 1], tris[t, 2]
        ux = verts[i1, 0] - verts[i0, 0]
        uy = verts[i1, 1] - verts[i0, 1]
        uz = verts[i1, 2] - verts[i0, 2]
        vx = verts[i2, 0] - verts[i0, 0]
        vy = verts[i2, 1] - verts[i0, 1]
        vz = verts[i2, 2] - verts[i0, 2]
        fnx = uy * vz - uz * vy
        fny = uz * vx - ux * vz
        fnz = ux * vy - uy * vx

        g0[i0, 0] += fnx
        g0[i0, 1] += fny
        g0[i0, 2] += fnz
        g1[i1, 0] += fnx
        g1[i1, 1] += fny
        g1[i1, 2] += fnz
        g2[i2, 0] += fnx
        g2[i2, 1] += fny
        g2[i2, 2] += fnz

    vn = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        nx = (g0[i, 0] + g1[i, 0]) + g2[i, 0]
        ny = (g0[i, 1] + g1[i, 1]) + g2[i, 1]
        nz = (g0[i, 2] + g1[i, 2]) + g2[i, 2]
        nrm = np.sqrt((nx * nx + nz * nz) + ny * ny)
        if nrm < 1e-300:
            nrm = 1e-300
        vn[i, 0] = nx / nrm
        vn[i, 1] = ny / nrm
        vn[i, 2] = nz / nrm
    return vn


@njit(cache=True, fastmath=False)
def tangential_smooth_step_kernel(verts, tris, indptr, indices, data, deg, fixed, strength):
    vn = tangential_smooth_normals_kernel(verts, tris)
    n = verts.shape[0]
    out = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        if fixed[i]:
            # Vanilla still adds strength * 0.0 on fixed rows (signed zeros).
            fixed_offset = strength * 0.0
            out[i, 0] = verts[i, 0] + fixed_offset
            out[i, 1] = verts[i, 1] + fixed_offset
            out[i, 2] = verts[i, 2] + fixed_offset
            continue
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            col = indices[k]
            val = data[k]
            s0 += val * verts[col, 0]
            s1 += val * verts[col, 1]
            s2 += val * verts[col, 2]
        d_val = deg[i] if deg[i] >= 1.0 else 1.0
        ox = s0 / d_val - verts[i, 0]
        oy = s1 / d_val - verts[i, 1]
        oz = s2 / d_val - verts[i, 2]
        vnx, vny, vnz = vn[i, 0], vn[i, 1], vn[i, 2]
        dot = (ox * vnx + oy * vny) + oz * vnz
        ox -= dot * vnx
        oy -= dot * vny
        oz -= dot * vnz
        out[i, 0] = verts[i, 0] + strength * ox
        out[i, 1] = verts[i, 1] + strength * oy
        out[i, 2] = verts[i, 2] + strength * oz
    return out


@njit(cache=True, fastmath=False)
def disc_radii_kernel(a, d, cx, cy, rho0, h, tg):
    """Compute the radii and clipped profile coordinates at the Gauss nodes."""
    rho = np.empty((len(a), len(tg)), dtype=np.float64)
    u = np.empty_like(rho)
    span = 2.0 * h
    for i in range(len(a)):
        for g in range(len(tg)):
            rx = (a[i, 0] + tg[g] * d[i, 0]) - cx
            ry = (a[i, 1] + tg[g] * d[i, 1]) - cy
            rg = np.sqrt(rx * rx + ry * ry)
            rho[i, g] = rg
            u[i, g] = min(max((rg - rho0) / span, 0.0), 1.0)
    return rho, u


@njit(cache=True, fastmath=False, error_model="numpy")
def disc_integral_kernel(a, d, lengths, idx, n_verts, cx, cy, rho0, sp_coef, tg, wg, rho, u, profile):
    ne, ng = len(idx), len(tg)
    shoelace = np.empty(ne, dtype=np.float64)
    for i in range(ne):
        j = i + 1 if i + 1 < ne else 0
        shoelace[i] = a[i, 0] * a[j, 1] - a[j, 0] * a[i, 1]
    sg = 1.0 if (0.5 * pairwise_sum_1d(shoelace)) >= 0.0 else -1.0
    terms = np.empty(ne * ng, dtype=np.float64)
    grad = np.zeros((n_verts, 3), dtype=np.float64)
    end = np.empty((ne, 2), dtype=np.float64)
    nc = len(sp_coef)
    for i in range(ne):
        L = lengths[i]
        safe = max(L, 1e-300)
        nx = sg * (d[i, 1] / safe)
        ny = sg * -(d[i, 0] / safe)
        sx = sy = ex = ey = 0.0
        for g in range(ng):
            t_val = tg[g]
            w = wg[g] * L
            rx = (a[i, 0] + t_val * d[i, 0]) - cx
            ry = (a[i, 1] + t_val * d[i, 1]) - cy
            rg = rho[i, g]
            if rg > 1e-12:
                square = rg * rg
                if rg <= rho0:
                    pot = 0.5 * square
                else:
                    ug = u[i, g]
                    pot = sp_coef[nc - 1] + ug * 0.0
                    for j in range(nc - 2, -1, -1):
                        pot = sp_coef[j] + pot * ug
                fac = pot / square
            else:
                fac = 0.5
            terms[i * ng + g] = (fac * (rx * nx + ry * ny)) * w

            start_w = (profile[i, g] * (1.0 - t_val)) * w
            end_w = (profile[i, g] * t_val) * w
            sx += start_w * nx
            sy += start_w * ny
            ex += end_w * nx
            ey += end_w * ny
        grad[idx[i], 0] += sx
        grad[idx[i], 1] += sy
        end[i, 0], end[i, 1] = ex, ey
    # Native np.add.at uses two separate passes, starts before ends.
    for i in range(ne):
        j = i + 1 if i + 1 < ne else 0
        grad[idx[j], 0] += end[i, 0]
        grad[idx[j], 1] += end[i, 1]
    return pairwise_sum_1d(terms), grad


@njit(cache=True, fastmath=False)
def quickselect_median(a):
    n = len(a)
    k = n // 2
    lo = 0
    hi = n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < a[lo]:
            a[lo], a[mid] = a[mid], a[lo]
        if a[hi] < a[lo]:
            a[lo], a[hi] = a[hi], a[lo]
        if a[mid] < a[hi]:
            a[mid], a[hi] = a[hi], a[mid]
        pivot = a[hi]
        lt = lo
        gt = hi
        i = lo
        while i <= gt:
            v = a[i]
            if v < pivot:
                a[lt], a[i] = v, a[lt]
                lt += 1
                i += 1
            elif v > pivot:
                a[i], a[gt] = a[gt], v
                gt -= 1
            else:
                i += 1
        if k < lt:
            hi = lt - 1
        elif k > gt:
            lo = gt + 1
        else:
            break
    if (n & 1) == 1:
        return a[k]
    m_prev = a[0]
    for idx in range(1, k):
        if a[idx] > m_prev:
            m_prev = a[idx]
    return 0.5 * (m_prev + a[k])


@njit(cache=True, fastmath=False)
def trust_region_step_kernel(verts, g_smooth, lr, max_step_frac, e0, e1, boundary_idx, base_int_idx):
    n = verts.shape[0]
    ne = len(e0)
    for k in range(len(boundary_idx)):
        g_smooth[boundary_idx[k], 2] = 0.0
    for k in range(len(base_int_idx)):
        bi = base_int_idx[k]
        g_smooth[bi, 0] = 0.0
        g_smooth[bi, 1] = 0.0
        g_smooth[bi, 2] = 0.0
    step_max = 0.0
    # NumPy einsum's three-component reduction uses a different order for
    # interleaved rows than for the column-major output of the sparse solve.
    row_major = g_smooth.flags.c_contiguous

    if max_step_frac > 0.0 and ne > 0:
        el = np.empty(ne, dtype=np.float64)
        shortest = np.full(n, np.inf, dtype=np.float64)
        for k in range(ne):
            a = e0[k]
            b = e1[k]
            dx = verts[a, 0] - verts[b, 0]
            dy = verts[a, 1] - verts[b, 1]
            dz = verts[a, 2] - verts[b, 2]
            length = np.sqrt((dx * dx + dz * dz) + dy * dy)
            el[k] = length
            if length < shortest[a]:
                shortest[a] = length
            if length < shortest[b]:
                shortest[b] = length
        floor = 0.25 * quickselect_median(el)
        for i in range(n):
            sx = lr * g_smooth[i, 0]
            sy = lr * g_smooth[i, 1]
            sz = lr * g_smooth[i, 2]
            size = np.sqrt((sx * sx + sz * sz) + sy * sy if row_major else (sx * sx + sy * sy) + sz * sz)
            sh = shortest[i] if shortest[i] >= floor else floor
            cap = max_step_frac * sh
            denom = size if size >= 1e-300 else 1e-300
            ratio = cap / denom
            if ratio < 1.0:
                sx *= ratio
                sy *= ratio
                sz *= ratio
                size = np.sqrt((sx * sx + sz * sz) + sy * sy if row_major else (sx * sx + sy * sy) + sz * sz)
            if size > step_max:
                step_max = size
            verts[i, 0] -= sx
            verts[i, 1] -= sy
            verts[i, 2] -= sz
    else:
        for i in range(n):
            sx = lr * g_smooth[i, 0]
            sy = lr * g_smooth[i, 1]
            sz = lr * g_smooth[i, 2]
            size = np.sqrt((sx * sx + sz * sz) + sy * sy if row_major else (sx * sx + sy * sy) + sz * sz)
            if size > step_max:
                step_max = size
            verts[i, 0] -= sx
            verts[i, 1] -= sy
            verts[i, 2] -= sz

    for k in range(len(boundary_idx)):
        verts[boundary_idx[k], 2] = 0.0
    return step_max


@njit(cache=True, fastmath=False)
def base_rhs_kernel(indptr, indices, data, verts, is_int):
    n_int = len(indptr) - 1
    rhs = np.empty((n_int, 2), dtype=np.float64)
    for r in range(n_int):
        s0 = 0.0
        s1 = 0.0
        for k in range(indptr[r], indptr[r + 1]):
            col = indices[k]
            val = data[k]
            if is_int[col]:
                s0 += val * 0.0
                s1 += val * 0.0
            else:
                s0 += val * verts[col, 0]
                s1 += val * verts[col, 1]
        rhs[r, 0] = s0
        rhs[r, 1] = s1
    return rhs


@njit(cache=True, fastmath=False)
def wetted_radial_project_kernel(verts, gE, wet, sphere_center):
    n = verts.shape[0]
    cx, cy, cz = sphere_center[0], sphere_center[1], sphere_center[2]
    fx = fy = fz = 0.0
    for i in range(n):
        if wet[i]:
            dx = verts[i, 0] - cx
            dy = verts[i, 1] - cy
            dz = verts[i, 2] - cz
            dist = np.sqrt((dx * dx + dy * dy) + dz * dz)
            denom = dist if dist >= 1e-12 else 1e-12
            nx = dx / denom
            ny = dy / denom
            nz = dz / denom
            dot = (gE[i, 0] * nx + gE[i, 1] * ny) + gE[i, 2] * nz
            rx = dot * nx
            ry = dot * ny
            rz = dot * nz
            fx += rx
            fy += ry
            fz += rz
            gE[i, 0] -= rx
            gE[i, 1] -= ry
            gE[i, 2] -= rz
    return fx, fy, fz


@njit(cache=True, fastmath=False)
def wetted_snap_kernel(verts, wet, sphere_center, radius):
    n = verts.shape[0]
    cx, cy, cz = sphere_center[0], sphere_center[1], sphere_center[2]
    for i in range(n):
        if wet[i]:
            dx = verts[i, 0] - cx
            dy = verts[i, 1] - cy
            dz = verts[i, 2] - cz
            dist = np.sqrt((dx * dx + dy * dy) + dz * dz)
            denom = dist if dist >= 1e-12 else 1e-12
            verts[i, 0] = cx + radius * dx / denom
            verts[i, 1] = cy + radius * dy / denom
            verts[i, 2] = cz + radius * dz / denom


@njit(cache=True, fastmath=False)
def collapse_short_edges_kernel(verts, tris, boundary_idx, min_len,
                                has_prot, prot_center, prot_radius, has_prot_min, prot_min_len):
    n_verts = verts.shape[0]
    m = tris.shape[0]
    is_bnd = np.zeros(n_verts, dtype=np.bool_)
    for i in range(len(boundary_idx)):
        is_bnd[boundary_idx[i]] = True

    prot = np.zeros(n_verts, dtype=np.bool_)
    if has_prot:
        cx, cy, cz = prot_center[0], prot_center[1], prot_center[2]
        for i in range(n_verts):
            dx = verts[i, 0] - cx
            dy = verts[i, 1] - cy
            dz = verts[i, 2] - cz
            if np.sqrt((dx * dx + dz * dz) + dy * dy) < prot_radius:
                prot[i] = True

    cand_a = np.empty(3 * m, dtype=np.int64)
    cand_b = np.empty(3 * m, dtype=np.int64)
    n_raw = 0
    for t in range(m):
        for k in range(3):
            u = tris[t, k]
            v = tris[t, (k + 1) if (k + 1) < 3 else 0]
            a = u if u < v else v
            b = v if u < v else u
            if is_bnd[a] or is_bnd[b]:
                continue
            in_zone = (prot[a] or prot[b]) if has_prot else False
            if in_zone and not has_prot_min:
                continue
            thresh = prot_min_len if in_zone else min_len
            dx = verts[a, 0] - verts[b, 0]
            dy = verts[a, 1] - verts[b, 1]
            dz = verts[a, 2] - verts[b, 2]
            d = np.sqrt((dx * dx + dz * dz) + dy * dy)
            if d < thresh:
                cand_a[n_raw] = a
                cand_b[n_raw] = b
                n_raw += 1

    if n_raw == 0:
        return verts.copy(), tris.copy(), boundary_idx.copy()

    nn = np.int64(n_verts)
    keys = np.empty(n_raw, dtype=np.int64)
    for i in range(n_raw):
        keys[i] = cand_a[i] * nn + cand_b[i]
    order_k = np.argsort(keys, kind="mergesort")
    keep = np.zeros(n_raw, dtype=np.bool_)
    n_uniq = 0
    for i in range(n_raw):
        if i == 0 or keys[order_k[i]] != keys[order_k[i - 1]]:
            keep[order_k[i]] = True
            n_uniq += 1

    cand_edges = np.empty((n_uniq, 2), dtype=np.int64)
    cand_len = np.empty(n_uniq, dtype=np.float64)
    idx = 0
    for i in range(n_raw):
        if keep[i]:
            a = cand_a[i]
            b = cand_b[i]
            cand_edges[idx, 0] = a
            cand_edges[idx, 1] = b
            dx = verts[a, 0] - verts[b, 0]
            dy = verts[a, 1] - verts[b, 1]
            dz = verts[a, 2] - verts[b, 2]
            cand_len[idx] = np.sqrt((dx * dx + dy * dy) + dz * dz)
            idx += 1

    order = np.argsort(cand_len, kind="mergesort")

    vt_counts = np.zeros(n_verts, dtype=np.int64)
    for t in range(m):
        vt_counts[tris[t, 0]] += 1
        vt_counts[tris[t, 1]] += 1
        vt_counts[tris[t, 2]] += 1
    vt_ptr = np.empty(n_verts + 1, dtype=np.int64)
    vt_ptr[0] = 0
    for i in range(n_verts):
        vt_ptr[i + 1] = vt_ptr[i] + vt_counts[i]
        vt_counts[i] = vt_ptr[i]
    vt_flat = np.empty(3 * m, dtype=np.int64)
    for t in range(m):
        for k in range(3):
            v = tris[t, k]
            vt_flat[vt_counts[v]] = t
            vt_counts[v] += 1

    all_keys = np.empty(3 * m, dtype=np.int64)
    for t in range(m):
        for k in range(3):
            u = tris[t, k]
            v = tris[t, (k + 1) if (k + 1) < 3 else 0]
            a = u if u < v else v
            b = v if u < v else u
            all_keys[3 * t + k] = a * nn + b
    all_keys.sort()
    n_ue = 1
    for i in range(1, 3 * m):
        if all_keys[i] != all_keys[i - 1]:
            n_ue += 1
    nb_counts = np.zeros(n_verts, dtype=np.int64)
    for i in range(3 * m):
        if i == 0 or all_keys[i] != all_keys[i - 1]:
            kval = all_keys[i]
            a = kval // nn
            b = kval % nn
            nb_counts[a] += 1
            nb_counts[b] += 1
    nb_ptr = np.empty(n_verts + 1, dtype=np.int64)
    nb_ptr[0] = 0
    for i in range(n_verts):
        nb_ptr[i + 1] = nb_ptr[i] + nb_counts[i]
        nb_counts[i] = nb_ptr[i]
    nb_flat = np.empty(2 * n_ue, dtype=np.int64)
    for i in range(3 * m):
        if i == 0 or all_keys[i] != all_keys[i - 1]:
            kval = all_keys[i]
            a = kval // nn
            b = kval % nn
            nb_flat[nb_counts[a]] = b
            nb_counts[a] += 1
            nb_flat[nb_counts[b]] = a
            nb_counts[b] += 1

    P = verts.copy()
    T = tris.copy()
    removed_vertex = np.zeros(n_verts, dtype=np.bool_)
    removed_tri = np.zeros(m, dtype=np.bool_)
    frozen = np.zeros(n_verts, dtype=np.bool_)

    for oi in range(n_uniq):
        ci = order[oi]
        a = cand_edges[ci, 0]
        b = cand_edges[ci, 1]
        if frozen[a] or frozen[b]:
            continue

        na0, na1 = nb_ptr[a], nb_ptr[a + 1]
        nb0, nb1 = nb_ptr[b], nb_ptr[b + 1]
        n_common = 0
        c_v0 = -1
        c_v1 = -1
        for ia in range(na0, na1):
            va = nb_flat[ia]
            if va == a or va == b:
                continue
            for ib in range(nb0, nb1):
                if nb_flat[ib] == va:
                    if n_common == 0:
                        c_v0 = va
                    elif n_common == 1:
                        c_v1 = va
                    n_common += 1
                    break

        ta0, ta1 = vt_ptr[a], vt_ptr[a + 1]
        tb0, tb1 = vt_ptr[b], vt_ptr[b + 1]
        n_adj = 0
        adj0 = -1
        adj1 = -1
        ap0 = -1
        ap1 = -1
        ok = True
        for ia in range(ta0, ta1):
            ti = vt_flat[ia]
            if removed_tri[ti]:
                continue
            in_b = False
            for ib in range(tb0, tb1):
                if vt_flat[ib] == ti:
                    in_b = True
                    break
            if in_b:
                t0, t1, t2 = T[ti, 0], T[ti, 1], T[ti, 2]
                n_oth = 0
                oth = -1
                if t0 != a and t0 != b:
                    oth = t0
                    n_oth += 1
                if t1 != a and t1 != b:
                    oth = t1
                    n_oth += 1
                if t2 != a and t2 != b:
                    oth = t2
                    n_oth += 1
                if n_oth != 1:
                    ok = False
                    break
                if n_adj == 0:
                    adj0 = ti
                    ap0 = oth
                elif n_adj == 1:
                    adj1 = ti
                    ap1 = oth
                n_adj += 1

        if not ok or n_adj != 2 or n_common != 2 or ap0 == ap1:
            continue
        if not ((ap0 == c_v0 and ap1 == c_v1) or (ap0 == c_v1 and ap1 == c_v0)):
            continue

        npx = 0.5 * (P[a, 0] + P[b, 0])
        npy = 0.5 * (P[a, 1] + P[b, 1])
        npz = 0.5 * (P[a, 2] + P[b, 2])

        n_ct = 0
        ct = np.empty((ta1 - ta0) + (tb1 - tb0), dtype=np.int64)
        for ia in range(ta0, ta1):
            ti = vt_flat[ia]
            if not removed_tri[ti] and ti != adj0 and ti != adj1:
                ct[n_ct] = ti
                n_ct += 1
        for ib in range(tb0, tb1):
            ti = vt_flat[ib]
            if not removed_tri[ti] and ti != adj0 and ti != adj1:
                dup = False
                for k in range(n_ct):
                    if ct[k] == ti:
                        dup = True
                        break
                if not dup:
                    ct[n_ct] = ti
                    n_ct += 1

        valid = True
        for k in range(n_ct):
            ti = ct[k]
            v0, v1, v2 = T[ti, 0], T[ti, 1], T[ti, 2]
            w0 = a if v0 == b else v0
            w1 = a if v1 == b else v1
            w2 = a if v2 == b else v2
            if w0 == w1 or w1 == w2 or w0 == w2:
                valid = False
                break
            ux, uy, uz = P[v1, 0] - P[v0, 0], P[v1, 1] - P[v0, 1], P[v1, 2] - P[v0, 2]
            vx, vy, vz = P[v2, 0] - P[v0, 0], P[v2, 1] - P[v0, 1], P[v2, 2] - P[v0, 2]
            nbx = uy * vz - uz * vy
            nby = uz * vx - ux * vz
            nbz = ux * vy - uy * vx

            p0x = npx if w0 == a else P[w0, 0]
            p0y = npy if w0 == a else P[w0, 1]
            p0z = npz if w0 == a else P[w0, 2]
            p1x = npx if w1 == a else P[w1, 0]
            p1y = npy if w1 == a else P[w1, 1]
            p1z = npz if w1 == a else P[w1, 2]
            p2x = npx if w2 == a else P[w2, 0]
            p2y = npy if w2 == a else P[w2, 1]
            p2z = npz if w2 == a else P[w2, 2]

            ux2, uy2, uz2 = p1x - p0x, p1y - p0y, p1z - p0z
            vx2, vy2, vz2 = p2x - p0x, p2y - p0y, p2z - p0z
            nax = uy2 * vz2 - uz2 * vy2
            nay = uz2 * vx2 - ux2 * vz2
            naz = ux2 * vy2 - uy2 * vx2

            if ((nbx * nax + nbz * naz) + nby * nay) <= 0.0:
                valid = False
                break

        if not valid:
            continue

        P[a, 0], P[a, 1], P[a, 2] = npx, npy, npz
        removed_vertex[b] = True
        removed_tri[adj0] = True
        removed_tri[adj1] = True
        for k in range(n_ct):
            ti = ct[k]
            if T[ti, 0] == b:
                T[ti, 0] = a
            if T[ti, 1] == b:
                T[ti, 1] = a
            if T[ti, 2] == b:
                T[ti, 2] = a

        frozen[a] = True
        frozen[b] = True
        frozen[ap0] = True
        frozen[ap1] = True
        for ia in range(na0, na1):
            frozen[nb_flat[ia]] = True
        for ib in range(nb0, nb1):
            frozen[nb_flat[ib]] = True

    n_kv = 0
    remap = np.empty(n_verts, dtype=np.int64)
    for i in range(n_verts):
        if not removed_vertex[i]:
            remap[i] = n_kv
            n_kv += 1
        else:
            remap[i] = -1
    new_verts = np.empty((n_kv, 3), dtype=np.float64)
    idx = 0
    for i in range(n_verts):
        if not removed_vertex[i]:
            new_verts[idx, 0] = P[i, 0]
            new_verts[idx, 1] = P[i, 1]
            new_verts[idx, 2] = P[i, 2]
            idx += 1

    n_kt = 0
    for t in range(m):
        if not removed_tri[t]:
            n_kt += 1
    new_tris = np.empty((n_kt, 3), dtype=np.int64)
    idx = 0
    for t in range(m):
        if not removed_tri[t]:
            new_tris[idx, 0] = remap[T[t, 0]]
            new_tris[idx, 1] = remap[T[t, 1]]
            new_tris[idx, 2] = remap[T[t, 2]]
            idx += 1

    nb = len(boundary_idx)
    new_bnd = np.empty(nb, dtype=np.int64)
    for i in range(nb):
        new_bnd[i] = remap[boundary_idx[i]]

    return new_verts, new_tris, new_bnd


@njit(cache=True, fastmath=False)
def wetted_interior_indices_kernel(wet, indptr, indices, deg):
    n = len(wet)
    tmp = np.empty(n, dtype=np.int64)
    cnt = 0
    for i in range(n):
        if wet[i]:
            s = 0.0
            for k in range(indptr[i], indptr[i + 1]):
                if wet[indices[k]]:
                    s += 1.0
            if s >= deg[i] - 0.5:
                tmp[cnt] = i
                cnt += 1
    return tmp[:cnt].copy()


@njit(cache=True, fastmath=False)
def wetted_interior_smooth_kernel(verts, interior_idx, indptr, indices, deg):
    m = len(interior_idx)
    if m == 0:
        return
    new_pos = np.empty((m, 3), dtype=np.float64)
    for idx in range(m):
        i = interior_idx[idx]
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            col = indices[k]
            s0 += verts[col, 0]
            s1 += verts[col, 1]
            s2 += verts[col, 2]
        d_val = deg[i] if deg[i] >= 1.0 else 1.0
        new_pos[idx, 0] = 0.5 * verts[i, 0] + 0.5 * (s0 / d_val)
        new_pos[idx, 1] = 0.5 * verts[i, 1] + 0.5 * (s1 / d_val)
        new_pos[idx, 2] = 0.5 * verts[i, 2] + 0.5 * (s2 / d_val)
    for idx in range(m):
        i = interior_idx[idx]
        verts[i, 0] = new_pos[idx, 0]
        verts[i, 1] = new_pos[idx, 1]
        verts[i, 2] = new_pos[idx, 2]


@njit(cache=True, fastmath=False)
def lv_area_sum_kernel(tri_area, lv_mask, has_tri_w, tri_w):
    m = len(tri_area)
    arr = np.empty(m, dtype=np.float64)
    # Multiply in the same order as NumPy, even on masked-out triangles.
    if not has_tri_w:
        for t in range(m):
            active = 1.0 if lv_mask[t] else 0.0
            arr[t] = tri_area[t] * active
    else:
        for t in range(m):
            active = 1.0 if lv_mask[t] else 0.0
            arr[t] = tri_area[t] * (active * tri_w[t])
    # NumPy's sum also adds the positive-zero identity to the partial sum.
    return 0.0 + pairwise_sum_1d(arr)


@njit(cache=True, fastmath=False)
def check_ring_edges_covered_kernel(base_tris, ring, nn):
    m = base_tris.shape[0]
    n_be = 3 * m
    bkeys = np.empty(n_be, dtype=np.int64)
    nn64 = np.int64(nn)
    for t in range(m):
        i0, i1, i2 = np.int64(base_tris[t, 0]), np.int64(base_tris[t, 1]), np.int64(base_tris[t, 2])
        a, b = (i0, i1) if i0 < i1 else (i1, i0)
        bkeys[3 * t] = a * nn64 + b
        a, b = (i1, i2) if i1 < i2 else (i2, i1)
        bkeys[3 * t + 1] = a * nn64 + b
        a, b = (i2, i0) if i2 < i0 else (i0, i2)
        bkeys[3 * t + 2] = a * nn64 + b
    bkeys.sort()

    nr = len(ring)
    for i in range(nr):
        inext = (i + 1) if (i + 1) < nr else 0
        u = np.int64(ring[i])
        v = np.int64(ring[inext])
        a, b = (u, v) if u < v else (v, u)
        rk = a * nn64 + b
        lo = 0
        hi = n_be
        while lo < hi:
            mid = (lo + hi) // 2
            if bkeys[mid] < rk:
                lo = mid + 1
            else:
                hi = mid
        if lo >= n_be or bkeys[lo] != rk:
            return False
        if (lo + 1 < n_be) and bkeys[lo + 1] == rk:
            return False
    return True


@njit(cache=True, fastmath=False)
def edge_lengths_kernel(verts, edges):
    ne = edges.shape[0]
    out = np.empty(ne, dtype=np.float64)
    for k in range(ne):
        a = edges[k, 0]
        b = edges[k, 1]
        dx = verts[a, 0] - verts[b, 0]
        dy = verts[a, 1] - verts[b, 1]
        dz = verts[a, 2] - verts[b, 2]
        out[k] = np.sqrt((dx * dx + dz * dz) + dy * dy)
    return out


@njit(cache=True, fastmath=False)
def long_edges_near_kernel(verts, tris, center, radius, max_len):
    m = tris.shape[0]
    cx, cy, cz = center[0], center[1], center[2]
    for t in range(m):
        for k in range(3):
            a = tris[t, k]
            b = tris[t, (k + 1) if (k + 1) < 3 else 0]
            mx = 0.5 * (verts[a, 0] + verts[b, 0]) - cx
            my = 0.5 * (verts[a, 1] + verts[b, 1]) - cy
            mz = 0.5 * (verts[a, 2] + verts[b, 2]) - cz
            if np.sqrt((mx * mx + my * my) + mz * mz) < radius:
                dx = verts[a, 0] - verts[b, 0]
                dy = verts[a, 1] - verts[b, 1]
                dz = verts[a, 2] - verts[b, 2]
                if np.sqrt((dx * dx + dy * dy) + dz * dz) > max_len:
                    return True
    return False


@njit(cache=True, fastmath=False)
def short_edges_near_kernel(verts, tris, center, radius, min_len):
    m = tris.shape[0]
    cx, cy, cz = center[0], center[1], center[2]
    for t in range(m):
        for k in range(3):
            a = tris[t, k]
            b = tris[t, (k + 1) if (k + 1) < 3 else 0]
            mx = 0.5 * (verts[a, 0] + verts[b, 0]) - cx
            my = 0.5 * (verts[a, 1] + verts[b, 1]) - cy
            mz = 0.5 * (verts[a, 2] + verts[b, 2]) - cz
            if np.sqrt((mx * mx + my * my) + mz * mz) < radius:
                dx = verts[a, 0] - verts[b, 0]
                dy = verts[a, 1] - verts[b, 1]
                dz = verts[a, 2] - verts[b, 2]
                if np.sqrt((dx * dx + dy * dy) + dz * dz) < min_len:
                    return True
    return False


@njit(cache=True, fastmath=False)
def rivara_prescan_kernel(pos, T, max_len, has_fine, fine_center, fine_radius, fine_len):
    m = T.shape[0]
    flag = np.zeros(m, dtype=np.bool_)
    base_t = max_len * (1.0 - 1e-9)
    fine_t = (fine_len if fine_len < max_len else max_len) * (1.0 - 1e-9)
    cx = fine_center[0] if has_fine else 0.0
    cy = fine_center[1] if has_fine else 0.0
    cz = fine_center[2] if has_fine else 0.0

    for i in range(m):
        for k in range(3):
            a = T[i, k]
            b = T[i, (k + 1) if (k + 1) < 3 else 0]
            ax, ay, az = pos[a, 0], pos[a, 1], pos[a, 2]
            bx, by, bz = pos[b, 0], pos[b, 1], pos[b, 2]
            dx, dy, dz = ax - bx, ay - by, az - bz
            length = np.sqrt((dx * dx + dz * dz) + dy * dy)
            thresh = base_t
            if has_fine:
                mx = 0.5 * (ax + bx) - cx
                my = 0.5 * (ay + by) - cy
                mz = 0.5 * (az + bz) - cz
                if np.sqrt((mx * mx + mz * mz) + my * my) < fine_radius:
                    thresh = fine_t
            if length > thresh:
                flag[i] = True
                break
    return flag
