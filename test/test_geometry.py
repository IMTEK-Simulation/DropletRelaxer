"""Area, volume, centre of mass: analytic references and finite differences."""
import numpy as np
import pytest

from conftest import fd_gradient, rel_err
from DropletRelaxer.energy import area_and_grad, volume_and_grad, center_of_mass
from DropletRelaxer.mesh import make_disk_cone, make_hemisphere


def _random_dome(rng, n_rings=4):
    verts, tris, b = make_disk_cone(n_rings, 1.0, 0.8)
    verts = verts + 0.03 * rng.standard_normal(verts.shape)
    return verts, tris, b


def test_single_triangle_area_and_normal_gradient():
    v = np.array([[0.0, 0, 0], [2.0, 0, 0], [0, 3.0, 0]])
    t = np.array([[0, 1, 2]])
    A, g, a = area_and_grad(v, t)
    assert A == pytest.approx(3.0)
    # moving the apex straight up by dz must not change the area to first order
    assert np.allclose(g[:, 2], 0.0)
    # moving vertex 2 along +y increases the area: dA/dy2 = base/2 = 1
    assert g[2, 1] == pytest.approx(-1.0) or g[2, 1] == pytest.approx(1.0)
    assert abs(g[2, 1]) == pytest.approx(1.0)


def test_area_gradient_matches_finite_differences(rng):
    v, t, _ = _random_dome(rng)
    g = area_and_grad(v, t)[1]
    gfd = fd_gradient(lambda x: area_and_grad(x, t)[0], v, h=1e-6)
    assert rel_err(g, gfd) < 1e-7


def test_weighted_area_gradient_matches_finite_differences(rng):
    v, t, _ = _random_dome(rng)
    w = rng.uniform(0.0, 1.5, len(t))
    w[::5] = 0.0   # wetted facets carry no surface energy
    g = area_and_grad(v, t, weights=w)[1]
    gfd = fd_gradient(lambda x: area_and_grad(x, t, weights=w)[0], v, h=1e-6)
    assert rel_err(g, gfd) < 1e-7


def test_flat_down_triangles_have_signed_area():
    v = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
    ccw = np.array([[0, 1, 2]])      # counter-clockwise seen from +z: normal +z
    cw = np.array([[0, 2, 1]])
    fd = np.array([True])
    # base faces down (-z): a triangle with normal -z counts positive
    assert area_and_grad(v, cw, flat_down=fd)[0] == pytest.approx(0.5)
    assert area_and_grad(v, ccw, flat_down=fd)[0] == pytest.approx(-0.5)


def test_flat_down_gradient_matches_finite_differences(rng):
    v = np.array([[0.0, 0, 0], [1.0, 0.1, 0], [0.2, 1.0, 0], [-0.8, 0.5, 0]])
    t = np.array([[0, 2, 1], [0, 3, 2]])
    fd = np.array([True, True])
    g = area_and_grad(v, t, flat_down=fd)[1]
    gfd = fd_gradient(lambda x: area_and_grad(x, t, flat_down=fd)[0], v, h=1e-6,
                      idx=[k for k in range(12) if k % 3 != 2])
    assert rel_err(g[:, :2], gfd[:, :2]) < 1e-7


@pytest.mark.parametrize("n", [8, 16, 32])
def test_hemisphere_area_converges_to_2piR2(n):
    v, t, _ = make_hemisphere(n, 4 * n, 1.0)
    A = area_and_grad(v, t)[0]
    err = abs(A - 2 * np.pi) / (2 * np.pi)
    assert err < {8: 2e-2, 16: 6e-3, 32: 2e-3}[n]


@pytest.mark.parametrize("n", [8, 16, 32])
def test_hemisphere_volume_and_com(n):
    v, t, _ = make_hemisphere(n, 4 * n, 1.0)
    V, _ = volume_and_grad(v, t)
    tol = {8: 3e-2, 16: 1e-2, 32: 4e-3}[n]
    assert abs(V - 2 * np.pi / 3) / (2 * np.pi / 3) < tol
    c = center_of_mass(v, t)
    assert abs(c[2] - 3.0 / 8.0) < tol
    assert abs(c[0]) < 1e-9 and abs(c[1]) < 1e-9


def test_volume_gradient_matches_finite_differences(rng):
    v, t, _ = _random_dome(rng)
    g = volume_and_grad(v, t)[1]
    gfd = fd_gradient(lambda x: volume_and_grad(x, t)[0], v, h=1e-6)
    assert rel_err(g, gfd) < 1e-8


def test_scaling_laws(rng):
    v, t, _ = _random_dome(rng)
    s = 2.7
    A1, V1 = area_and_grad(v, t)[0], volume_and_grad(v, t)[0]
    A2, V2 = area_and_grad(s * v, t)[0], volume_and_grad(s * v, t)[0]
    assert A2 == pytest.approx(s ** 2 * A1, rel=1e-12)
    assert V2 == pytest.approx(s ** 3 * V1, rel=1e-12)
    c1, c2 = center_of_mass(v, t), center_of_mass(s * v, t)
    assert np.allclose(c2, s * c1, rtol=1e-10)


def test_energy_is_invariant_under_rigid_motion_in_the_plane(rng):
    v, t, _ = _random_dome(rng)
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    v2 = v @ R.T + np.array([3.0, -1.0, 0.0])
    assert area_and_grad(v2, t)[0] == pytest.approx(area_and_grad(v, t)[0], rel=1e-12)


def test_energy_does_not_depend_on_triangle_numbering(rng):
    v, t, _ = _random_dome(rng)
    perm = rng.permutation(len(v))
    inv = np.argsort(perm)
    v2, t2 = v[perm], inv[t]
    assert area_and_grad(v2, t2)[0] == pytest.approx(area_and_grad(v, t)[0], rel=1e-12)
    assert volume_and_grad(v2, t2)[0] == pytest.approx(volume_and_grad(v, t)[0], rel=1e-12)
