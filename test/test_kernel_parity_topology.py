"""Exact parity for topology and remeshing kernels.

The NumPy/SciPy oracles are adapted from the pre-Numba implementations in
wetting.py, substrate.py and mesh.py. They never call a kernel or switch a
backend flag. For collapse, small meshes specify the expected winning edge
(or lack of one); a NumPy helper only materializes that known collapse.
"""
from collections import Counter

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from DropletRelaxer import _fast
from kernel_parity_helpers import assert_bits

pytestmark = pytest.mark.skipif(not _fast.numba_available(), reason="requires Numba")
k = _fast.numba_kernels() if _fast.numba_available() else None

_EMPTY_TRIS = np.empty((0, 3), dtype=np.int64)
_EMPTY_LOOP = np.empty(0, dtype=np.int64)
_ORIGIN = np.zeros(3)


def _directed_edges(tris):
    return np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])


@pytest.mark.parametrize("tris", [
    pytest.param(_EMPTY_TRIS, id="empty"),
    pytest.param(np.array([[12, 8, 3], [8, 12, 2], [3, 8, 12], [2, 2, 12]]), id="duplicates-and-self-edge"),
    pytest.param(np.array([[2**31 - 1, 0, 2**30]], dtype=np.int64), id="sparse-ids-require-64-bit"),
])
def test_unique_edges_kernel(tris):
    expected = np.unique(np.sort(_directed_edges(tris), axis=1), axis=0).astype(np.int64)
    assert_bits(k.unique_edges_kernel(tris), expected)


@pytest.mark.parametrize("native", [False, True])
def test_public_unique_edges_preserves_empty_mesh_error(native, monkeypatch):
    from DropletRelaxer.wetting import unique_edges

    monkeypatch.setattr(_fast, "_native_requested", native)
    with pytest.raises(ValueError, match="zero-size array"):
        unique_edges(_EMPTY_TRIS)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("xs,supplied,expected", [
    pytest.param([], False, [], id="no-vertices"),
    pytest.param([2.0, 3.0], False, [False, False], id="outside-shell"),
    pytest.param([1.0, 2.0], False, None, id="nearby-vertex-needs-edges"),
    pytest.param([1.0, -1.0], True, [True, False], id="supplied-empty-edges"),
])
def test_public_wetted_set_preserves_empty_triangle_behavior(xs, supplied, expected, native, monkeypatch):
    from DropletRelaxer.wetting import compute_wetted_set

    monkeypatch.setattr(_fast, "_native_requested", native)
    verts = np.zeros((len(xs), 3))
    verts[:, 0] = xs
    args = (verts, _EMPTY_TRIS, np.zeros(3), 1.0, 0.25, np.empty((0, 2), dtype=np.int64) if supplied else None)
    if expected is None:
        with pytest.raises(ValueError, match="zero-size array"):
            compute_wetted_set(*args)
    else:
        assert_bits(compute_wetted_set(*args), np.array(expected, dtype=bool))


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_edge_helpers_preserve_native_dtype_and_thresholds(dtype, monkeypatch):
    from DropletRelaxer import mesh

    monkeypatch.setattr(_fast, "_native_requested", False)
    verts = np.array([[0, 0, 0], [1, 1, 0], [0, 10, 0]], dtype=dtype)
    tris = np.array([[0, 1, 2]], dtype=np.int64)
    center = np.array([0.5, 0.5, 0], dtype=dtype)
    # Only edge (0, 1) is near the center. Its native rounded length is
    # exactly the threshold, whereas rounding after promotion to float64
    # would make it strictly longer for both input types.
    threshold = float(dtype(np.sqrt(2)))
    assert threshold < np.sqrt(2)
    for measure in (mesh.min_edge_length, mesh.mean_edge_length, mesh.max_edge_length):
        with _fast.kernel_mode(native=True):
            expected = measure(verts, tris)
        assert np.asarray(expected).dtype == np.dtype(dtype)
        assert_bits(measure(verts, tris), expected)
    assert mesh.short_edge_fraction(verts, tris, threshold) == 0.0
    assert not mesh.long_edges_near(verts, tris, center, 0.125, threshold)
    assert not mesh.short_edges_near(verts, tris, center, 0.125, threshold)


def _wetted_set_reference(verts, center, radius, snap, edges):
    near = np.abs(np.linalg.norm(verts - center, axis=1) - radius) < snap
    if not near.any():
        return near
    a, b = edges[near[edges].all(axis=1)].T
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(verts), len(verts)))
    _, labels = connected_components(graph, directed=False)
    return near & (labels == np.bincount(labels[near]).argmax())


@pytest.mark.parametrize("radial,links,snap,selected", [
    pytest.param([], [], 0.25, [], id="empty"),
    pytest.param([0, 0.75, 1.25, 2], [[0, 1], [2, 3]], 0.25, [], id="strict-shell-excludes-both-equalities"),
    pytest.param([0.875, 1.125, 1, 0.75, 1.25, 1, 1], [[1, 0], [2, 1], [6, 5], [2, 3], [3, 5]],
                 0.25, [0, 1, 2], id="largest-component-on-both-sides-with-dry-bridge"),
    pytest.param([0, 1, 1, 1, 1], [[4, 3], [2, 1], [2, 1], [2, 2]],
                 0.25, [1, 2], id="component-tie-selects-smallest-index"),
    pytest.param([2, 1, 1, 1], [], 0.25, [1], id="all-isolated"),
    pytest.param([1, 1], [[0, 1]], 0.0, [], id="zero-snap"),
])
def test_compute_wetted_set_kernel(radial, links, snap, selected):
    center = np.array([2.0, -3.0, 4.0])
    verts = np.tile(center, (len(radial), 1))
    verts[:, 0] += radial
    edges = np.array(links, dtype=np.int64).reshape(-1, 2)
    expected = _wetted_set_reference(verts, center, 1.0, snap, edges)
    assert_bits(expected, np.isin(np.arange(len(verts)), selected))
    assert_bits(k.compute_wetted_set_kernel(verts, center, 1.0, snap, edges), expected)


@pytest.mark.parametrize("seed", [0, 3, 17, 12345])
def test_wetted_shell_one_ulp_either_side(seed):
    rng = np.random.default_rng(seed)
    center = rng.uniform(-2, 2, size=3)
    verts = center + rng.normal(size=(24, 3))
    edges = np.column_stack((np.arange(len(verts) - 1), np.arange(1, len(verts))))
    gaps = np.abs(np.linalg.norm(verts - center, axis=1) - 1.0)
    for idx, gap in enumerate(gaps):
        for side, snap in enumerate((np.nextafter(gap, 0.0), gap, np.nextafter(gap, np.inf))):
            expected = _wetted_set_reference(verts, center, 1.0, snap, edges)
            actual = k.compute_wetted_set_kernel(verts, center, 1.0, snap, edges)
            try:
                assert_bits(actual, expected)
            except AssertionError as err:
                raise AssertionError(f"seed={seed}, vertex={idx}, side={side}, snap={snap.hex()}") from err


def _unoriented_boundary_reference(mask, tris):
    directed = _directed_edges(tris[mask[tris].all(axis=1)])
    _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True, return_inverse=True)
    boundary = directed[counts[inverse.reshape(-1)] == 1]
    if len(boundary) < 3:
        return _EMPTY_LOOP
    successors = dict(boundary.tolist())
    if len(successors) != len(boundary):
        return _EMPTY_LOOP
    start = int(boundary[0, 0])
    loop = [start]
    for _ in range(len(boundary) - 1):
        nxt = successors.get(loop[-1])
        if nxt is None or nxt == start:
            return _EMPTY_LOOP
        loop.append(nxt)
    return np.array(loop, dtype=np.int64) if successors.get(loop[-1]) == start else _EMPTY_LOOP


def _orient_reference(loop, verts, center):
    if not len(loop):
        return _EMPTY_LOOP
    u = verts[loop] - center
    # The vintage vectorized computation propagates NaN for a zero radial
    # vector; its winding > 0 check then takes the reversal branch.
    with np.errstate(invalid="ignore"):
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        c = u.mean(axis=0)
        cn = np.linalg.norm(c)
        if cn < 0.2:
            return _EMPTY_LOOP
        winding = np.sum(np.cross(u, np.roll(u, -1, axis=0)) @ (c / cn))
    return loop if winding > 0 else loop[::-1]


_CAP_CENTER = np.array([2.0, -3.0, 4.0])
_CAP = _CAP_CENTER + np.array([[1, 0, 1], [0, 2, 2], [-0.5, 0, 0.5], [0, -4, 4], [2, 2, 1], [0, 0, 1]])
_EQUATOR = np.array([[1.0, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
_CENTER_VERTEX = np.array([[0.0, 0, 0], [1, 0, 1], [0, 1, 1]])


@pytest.mark.parametrize("verts,center,indices,answer", [
    pytest.param(_CAP, _CAP_CENTER, [], [], id="empty"),
    pytest.param(_CAP, _CAP_CENTER, [2, 3, 0, 1], [2, 3, 0, 1], id="counterclockwise-shifted-and-radially-scaled"),
    pytest.param(_CAP, _CAP_CENTER, [0, 3, 2, 1], [1, 2, 3, 0], id="clockwise-reverses-without-rerotating"),
    pytest.param(_EQUATOR, _ORIGIN, [0, 1, 2, 3], [], id="ambiguous-equator"),
    pytest.param(np.array([[1.0, 0, 1], [0, 0, 1], [-1, 0, 1]]), _ORIGIN,
                 [0, 1, 2], [2, 1, 0], id="zero-winding-reverses"),
    pytest.param(_CENTER_VERTEX, _ORIGIN, [0, 1, 2], [2, 1, 0], id="sphere-center-vertex"),
])
def test_wetted_loop_orient_kernel(verts, center, indices, answer):
    loop = np.array(indices, dtype=np.int64)
    expected = _orient_reference(loop, verts, center)
    assert_bits(expected, np.array(answer, dtype=np.int64))
    assert_bits(_fast.orient_wetted_loop(loop, verts, center), expected)


@pytest.fixture(params=[
    pytest.param((_EMPTY_TRIS, [True] * 6, []), id="empty-triangles"),
    pytest.param(([[2, 3, 0], [2, 0, 1]], [False] * 6, []), id="no-wet-faces"),
    pytest.param(([[2, 3, 0], [2, 0, 1], [0, 3, 4]], [True] * 4 + [False] * 2, [2, 3, 0, 1]),
                 id="first-boundary-edge-and-partially-wet-face"),
    pytest.param(([[2, 0, 3], [2, 1, 0]], [True] * 6, [2, 1, 0, 3]), id="clockwise-patch"),
    pytest.param(([[0, 1, 2], [0, 3, 4]], [True] * 6, []), id="pinch-point"),
    pytest.param(([[0, 1, 2], [3, 4, 5]], [True] * 6, []), id="disjoint-loops"),
    pytest.param(([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], [True] * 6, []), id="closed-patch"),
    pytest.param(([[0, 1, 2], [0, 3, 2]], [True] * 6, []), id="inconsistent-face-winding"),
])
def patch(request):
    tris, wet, unoriented = request.param
    return np.array(tris, dtype=np.int64), np.array(wet), np.array(unoriented, dtype=np.int64)


def test_wetted_boundary_unoriented_kernel(patch):
    tris, mask, answer = patch
    expected = _unoriented_boundary_reference(mask, tris)
    assert_bits(expected, answer)
    assert_bits(k.wetted_boundary_unoriented_kernel(mask, tris), expected)


def test_wetted_boundary_loop_kernel(patch):
    tris, mask, _ = patch
    expected = _orient_reference(_unoriented_boundary_reference(mask, tris), _CAP, _CAP_CENTER)
    assert_bits(_fast.wetted_boundary_loop_fast(mask, tris, _CAP, _CAP_CENTER), expected)


@pytest.mark.parametrize("verts,tris,answer", [
    pytest.param(_EQUATOR, [[0, 1, 2], [0, 2, 3]], [], id="ambiguous-equator"),
    pytest.param(_CENTER_VERTEX, [[0, 1, 2]], [2, 1, 0], id="sphere-center-vertex"),
])
def test_wetted_boundary_loop_kernel_degenerate_orientation(verts, tris, answer):
    tris = np.array(tris, dtype=np.int64)
    mask = np.ones(len(verts), dtype=bool)
    expected = _orient_reference(_unoriented_boundary_reference(mask, tris), verts, _ORIGIN)
    assert_bits(expected, np.array(answer, dtype=np.int64))
    assert_bits(_fast.wetted_boundary_loop_fast(mask, tris, verts, _ORIGIN), expected)


@pytest.mark.parametrize("faces,ring,answer", [
    pytest.param([[0, 1, 2], [0, 2, 3]], [0, 1, 2, 3], True, id="shared-interior-edge-ok"),
    pytest.param([[0, 1, 2], [0, 2, 3]], [3, 2, 1, 0], True, id="reversed-ring"),
    pytest.param([[0, 1, 2], [0, 2, 3]], [0, 1, 2], False, id="ring-edge-covered-twice"),
    pytest.param([[0, 1, 4], [1, 2, 4], [2, 3, 4]], [0, 1, 2, 3], False, id="closing-edge-missing"),
    pytest.param(_EMPTY_TRIS, [0, 1, 2], False, id="empty-base"),
    pytest.param(_EMPTY_TRIS, [], True, id="empty-ring"),
])
def test_check_ring_edges_covered_kernel(faces, ring, answer):
    tris = np.array(faces, dtype=np.int64)
    ring = np.array(ring, dtype=np.int64)
    counts = Counter(map(tuple, np.sort(_directed_edges(tris), axis=1)))
    expected = all(counts[tuple(sorted((a, b)))] == 1 for a, b in zip(ring, np.roll(ring, -1)))
    assert expected is answer
    assert k.check_ring_edges_covered_kernel(tris, ring, 5) is expected


@pytest.mark.parametrize("pairs", [
    pytest.param([], id="empty"),
    pytest.param([[0, 1], [1, 0], [0, 0], [0, 2], [2, 1]], id="order-reversal-and-zero-length"),
    pytest.param([[0, 3], [2, 3], [3, 1], [3, 0]], id="nontrivial-rounding"),
])
def test_edge_lengths_kernel(pairs):
    verts = np.array([[0, -0.0, 0], [3, 4, 12], [-0.7, 3.5, -1.1], [1.628052, 1.739799, 0.063828]])
    edges = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    delta = verts[edges[:, 0]] - verts[edges[:, 1]]
    expected = np.sqrt(np.einsum("ij,ij->i", delta, delta))  # vintage row_norm
    assert_bits(k.edge_lengths_kernel(verts, edges), expected)


_NEAR_VERTS = np.array([[-2.0, 0, 0], [2, 0, 0], [0, 3, 0]])
_NEAR_TRIS = np.array([[1, 2, 0]], dtype=np.int64)  # qualifying edge is the closing edge


@pytest.fixture(params=[
    pytest.param((_NEAR_TRIS, [0, 0, 0], 0.5, 3.5, 4.5, True, True), id="midpoint-inside-endpoints-outside"),
    pytest.param((_NEAR_TRIS, [0, 0, 0], 0.5, 4.0, 4.0, False, False), id="exact-length-threshold"),
    pytest.param((_NEAR_TRIS, [0, 0, 0], 0.5, np.nextafter(4.0, 0), np.nextafter(4.0, np.inf), True, True),
                 id="one-ulp-past-length-threshold"),
    pytest.param((_NEAR_TRIS, [0, 0, 1], 1.0, 3.5, 4.5, False, False), id="midpoint-on-radius"),
    pytest.param((_NEAR_TRIS, [0, 0, 1], np.nextafter(1.0, np.inf), 3.5, 4.5, True, True), id="just-inside-radius"),
    pytest.param((_NEAR_TRIS, [-2, 0, 0], 0.5, 3.5, 4.5, False, False), id="endpoint-near-midpoints-far"),
    pytest.param((_EMPTY_TRIS, [0, 0, 0], 1.0, 3.5, 4.5, False, False), id="empty"),
])
def near_case(request):
    tris, center, radius, max_len, min_len, long, short = request.param
    return tris, np.array(center, dtype=float), radius, max_len, min_len, long, short


def _lengths_near_reference(verts, tris, center, radius):
    a, b = verts[_directed_edges(tris)].transpose(1, 0, 2)
    near = np.linalg.norm(0.5 * (a + b) - center, axis=1) < radius
    return np.linalg.norm(a - b, axis=1)[near]


def test_long_edges_near_kernel(near_case):
    tris, center, radius, max_len, _, answer, _ = near_case
    expected = bool(np.any(_lengths_near_reference(_NEAR_VERTS, tris, center, radius) > max_len))
    assert expected is answer
    assert k.long_edges_near_kernel(_NEAR_VERTS, tris, center, radius, max_len) is expected


def test_short_edges_near_kernel(near_case):
    tris, center, radius, _, min_len, _, answer = near_case
    expected = bool(np.any(_lengths_near_reference(_NEAR_VERTS, tris, center, radius) < min_len))
    assert expected is answer
    assert k.short_edges_near_kernel(_NEAR_VERTS, tris, center, radius, min_len) is expected


def _rivara_reference(pos, tris, max_len, center, radius, fine_len):
    a, b = _directed_edges(tris).T
    delta = pos[a] - pos[b]
    length = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    target = np.full(len(a), max_len, dtype=float)
    if center is not None:
        midpoint = 0.5 * (pos[a] + pos[b]) - center
        inside = np.sqrt(np.einsum("ij,ij->i", midpoint, midpoint)) < radius
        target[inside] = min(fine_len, max_len)
    return (length > target * (1.0 - 1e-9)).reshape(3, len(tris)).any(axis=0)


_RIVARA_LIMIT = 2.0 * (1.0 - 1e-9)


@pytest.mark.parametrize("lengths,offsets,center,radius,fine_len,answer", [
    pytest.param([], [], None, 0.0, 1.0, [], id="empty"),
    pytest.param([0.5, 3.0, 1.5], [0, 5, -5], None, 0.0, 1.0, [False, True, False], id="mixed-global-flags"),
    pytest.param([_RIVARA_LIMIT, np.nextafter(_RIVARA_LIMIT, np.inf), 2.0], [0, 0, 0],
                 None, 0.0, 1.0, [False, True, True], id="slack-and-strict-threshold"),
    pytest.param([1.5, 1.5], [0, 5], [0.75, 0, 0], 0.25, 1.0, [True, False], id="local-only"),
    pytest.param([1.5], [0], [0.75, 0, 1], 1.0, 1.0, [False], id="fine-midpoint-on-radius"),
    pytest.param([1.5], [0], [0.75, 0, 1], np.nextafter(1.0, np.inf), 1.0, [True], id="fine-midpoint-inside-radius"),
    pytest.param([2.5], [0], [1.25, 0, 0], 0.25, 4.0, [True], id="fine-limit-capped-at-global"),
])
def test_rivara_prescan_kernel(lengths, offsets, center, radius, fine_len, answer):
    verts = np.array([(x, y + offset, 0.0) for length, offset in zip(lengths, offsets)
                      for x, y in [(0.0, 0.0), (length, 0.0), (0.5 * length, 0.25 * length)]]).reshape(-1, 3)
    tris = np.arange(len(verts), dtype=np.int64).reshape(-1, 3)[:, [1, 2, 0]].copy()
    center = None if center is None else np.array(center, dtype=float)
    expected = _rivara_reference(verts, tris, 2.0, center, radius, fine_len)
    assert_bits(expected, np.array(answer, dtype=bool))
    actual = k.rivara_prescan_kernel(verts, tris, 2.0, center is not None,
                                          _ORIGIN if center is None else center, radius, fine_len)
    assert_bits(actual, expected)


def _strip_mesh(xs):
    """Diamond-shaped disk with a chain of interior vertices under the apexes."""
    n = len(xs)
    top, bottom, left, right = range(n, n + 4)
    verts = np.array([(x, 0.0, 0.0) for x in xs] + [(0, 1, 0), (0, -1, 0), (-2, 0, 0), (2, 0, 0)])
    faces = [(i, i + 1, top) for i in range(n - 1)] + [(i + 1, i, bottom) for i in range(n - 1)]
    faces += [(0, top, left), (0, left, bottom), (n - 1, bottom, right), (n - 1, right, top)]
    return verts, np.array(faces, dtype=np.int64), np.array([right, top, left, bottom], dtype=np.int64)


def _known_collapse_reference(verts, tris, ring, edge):
    if edge is None:
        return verts, tris, ring
    a, b = edge
    incident = (tris == a).any(axis=1) & (tris == b).any(axis=1)
    assert np.count_nonzero(incident) == 2
    kept = np.arange(len(verts)) != b
    new_verts = verts.copy()
    new_verts[a] = 0.5 * (verts[a] + verts[b])
    remap = np.cumsum(kept, dtype=np.int64) - 1
    return new_verts[kept], remap[np.where(tris[~incident] == b, a, tris[~incident])], remap[ring]


def _assert_collapse(verts, tris, ring, min_len, winner, center=None, radius=0.0, local_min=None):
    originals = tuple(x.copy() for x in (verts, tris, ring))
    expected = _known_collapse_reference(*originals, winner)
    actual = k.collapse_short_edges_kernel(
        verts, tris, ring, min_len, center is not None, _ORIGIN if center is None else center,
        radius, local_min is not None, 0.0 if local_min is None else local_min,
    )
    for result, reference in zip(actual, expected):
        assert_bits(result, reference)
    for source, original in zip((verts, tris, ring), originals):
        assert_bits(source, original)
    assert_bits(actual[0][actual[2]], verts[ring])


@pytest.mark.parametrize("min_len,center,radius,local_min,winner", [
    pytest.param(0.5, None, 0.0, None, (0, 1), id="midpoint-and-ring-remap"),
    pytest.param(0.25, None, 0.0, None, None, id="strict-length-threshold"),
    pytest.param(np.nextafter(0.25, np.inf), None, 0.0, None, (0, 1), id="one-ulp-below-threshold"),
    pytest.param(0.5, [0.125, 0, 0.25], 0.25, None, (0, 1), id="endpoint-on-protection-radius"),
    pytest.param(0.5, [0.125, 0, 0.25], np.nextafter(0.25, np.inf), None, None, id="one-protected-endpoint"),
    pytest.param(0.5, [0.125, 0, 0], 0.125, 0.25, None, id="strict-protected-threshold"),
    pytest.param(0.125, [0.125, 0, 0], 0.125, 0.375, (0, 1), id="protected-threshold-overrides-global"),
])
def test_collapse_short_edges_kernel_thresholds(min_len, center, radius, local_min, winner):
    verts, tris, ring = _strip_mesh([-0.125, 0.125])
    center = None if center is None else np.array(center, dtype=float)
    _assert_collapse(verts, tris, ring, min_len, winner, center, radius, local_min)


def test_collapse_short_edges_kernel_empty():
    _assert_collapse(np.empty((0, 3)), _EMPTY_TRIS, _EMPTY_LOOP, 1.0, None)


def test_collapse_short_edges_kernel_boundary_endpoint():
    verts, tris, _ = _strip_mesh([-0.125, 0.125])
    # Removing a spoke face exposes vertex 0 on the ring. Edge (0, 1)
    # still has both incident faces, but it may not move the contact line.
    _assert_collapse(verts, tris[[0, 1, 3, 4, 5]], np.array([5, 2, 0, 4, 3], dtype=np.int64), 0.5, None)


def test_collapse_short_edges_kernel_link_condition():
    # A triangular bipyramid: the equatorial edge (0,1) has common
    # neighbor 2 as well as its two opposite apexes 3 and 4.
    verts = np.array([[-0.125, 0, 0], [0.125, 0, 0], [0, -1, 0], [0, -0.25, 1], [0, -0.25, -1]])
    tris = np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3], [1, 0, 4], [2, 1, 4], [0, 2, 4]], dtype=np.int64)
    _assert_collapse(verts, tris, _EMPTY_LOOP, 0.5, None)


@pytest.mark.parametrize("corner_x", [pytest.param(0.0625, id="flip"), pytest.param(0.0, id="degenerate")])
def test_collapse_short_edges_kernel_rejects_bad_surviving_face(corner_x):
    verts, tris, _ = _strip_mesh([-0.125, 0.125])
    verts = np.vstack([verts, [corner_x, 2, 0]])
    tris = np.vstack([tris[:2], [[0, 2, 6], [0, 6, 4]], tris[3:]])
    # The notch edge (top, new corner) crosses or meets the proposed midpoint.
    p = verts[tris]
    assert (np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 2] > 0).all()
    _assert_collapse(verts, tris, np.array([5, 2, 6, 4, 3], dtype=np.int64), 0.5, None)


@pytest.mark.parametrize("xs,reverse,winner", [
    pytest.param([-0.75, -0.5, 0.5, 0.75], False, (0, 1), id="equal-length-first-seen"),
    pytest.param([-0.75, -0.5, 0.5, 0.75], True, (2, 3), id="equal-length-reordered-faces"),
    pytest.param([-0.875, -0.5, 0.5, 0.75], False, (2, 3), id="shorter-edge-wins-despite-later-row"),
])
def test_collapse_short_edges_kernel_candidate_order_and_freeze(xs, reverse, winner):
    verts, tris, ring = _strip_mesh(xs)
    if reverse:
        tris[[0, 2]] = tris[[2, 0]]
    # The two candidate edges don't share an endpoint, but their endpoints
    # are neighbors. After accepting one, the other must be frozen out.
    _assert_collapse(verts, tris, ring, 0.5, winner)


@pytest.mark.parametrize("seed", [0, 3, 17, 12345])
def test_collapse_near_ties_with_shuffled_faces(seed, monkeypatch):
    from DropletRelaxer.mesh import collapse_short_edges

    rng = np.random.default_rng(seed)
    monkeypatch.setattr(_fast, "_native_requested", False)
    for direction in (0.0, 0.75, np.inf):
        # Start with two equally short edges; moving the fourth vertex by one
        # ULP makes the second edge slightly shorter or longer than the first.
        verts, tris, ring = _strip_mesh([-0.75, -0.5, 0.5, np.nextafter(0.75, direction)])
        tris = tris[rng.permutation(len(tris))]
        for threshold in (np.nextafter(0.25, 0.0), 0.25, np.nextafter(0.25, np.inf)):
            with _fast.kernel_mode(native=True):
                expected = collapse_short_edges(verts, tris, ring, threshold)
            actual = collapse_short_edges(verts, tris, ring, threshold)
            try:
                for got, want in zip(actual, expected, strict=True):
                    assert_bits(got, want)
            except AssertionError as err:
                raise AssertionError(f"seed={seed}, toward={direction}, threshold={threshold.hex()}") from err
