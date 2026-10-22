"""Flat wall: shoelace energy, closed mesh with base, penetration penalty."""
import numpy as np
import pytest

from conftest import fd_gradient, rel_err
from DropletRelaxer.energy import (area_and_grad, volume_and_grad, substrate_wall_energy_and_grad,
                                   substrate_penetration_energy_and_grad)
from DropletRelaxer.mesh import make_disk_cone
from DropletRelaxer.substrate import add_base_mesh, strip_base, base_triangle_mask


def _poly_area(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def test_shoelace_known_polygons():
    sq = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0.0]])
    E, g = substrate_wall_energy_and_grad(sq, np.arange(4), kappa=0.5)
    assert E == pytest.approx(-0.5 * 4.0)
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    circ = np.c_[np.cos(th), np.sin(th), np.zeros_like(th)]
    E, _ = substrate_wall_energy_and_grad(circ, np.arange(200), kappa=1.0)
    assert -E == pytest.approx(0.5 * 200 * np.sin(2 * np.pi / 200), rel=1e-12)
    # clockwise ordering flips the sign (signed area)
    E2, _ = substrate_wall_energy_and_grad(circ[::-1], np.arange(200), kappa=1.0)
    assert E2 == pytest.approx(-E)


def test_wall_gradient_matches_finite_differences_and_is_tangential(rng):
    th = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    r = 1 + 0.2 * rng.standard_normal(30)
    v = np.c_[r * np.cos(th), r * np.sin(th), np.zeros(30)]
    kappa = np.cos(np.radians(120))
    E, g = substrate_wall_energy_and_grad(v, np.arange(30), kappa)
    gfd = fd_gradient(lambda x: substrate_wall_energy_and_grad(x, np.arange(30), kappa)[0], v, h=1e-6)
    assert rel_err(g, gfd) < 1e-8
    assert np.all(g[:, 2] == 0.0)


def test_wall_energy_sign_follows_young_angle():
    th = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    small = np.c_[np.cos(th), np.sin(th), np.zeros(50)]
    big = 2 * small
    for angle, spreads in [(60.0, True), (120.0, False)]:
        k = np.cos(np.radians(angle))
        e_small = substrate_wall_energy_and_grad(small, np.arange(50), k)[0]
        e_big = substrate_wall_energy_and_grad(big, np.arange(50), k)[0]
        # hydrophilic (<90 deg): larger wetted area lowers the energy; hydrophobic: raises it
        assert (e_big < e_small) == spreads


def test_penetration_penalty_gradient_and_one_sidedness(rng):
    v = rng.standard_normal((20, 3))
    E, g = substrate_penetration_energy_and_grad(v, 80.0)
    gfd = fd_gradient(lambda x: substrate_penetration_energy_and_grad(x, 80.0)[0], v, h=1e-6)
    assert rel_err(g, gfd) < 1e-6
    above = np.abs(v)
    assert substrate_penetration_energy_and_grad(above, 80.0)[0] == 0.0
    assert E == pytest.approx(80.0 * np.sum(np.minimum(v[:, 2], 0.0) ** 2))


def test_base_closes_the_mesh_without_changing_volume(rng):
    v, t, b = make_disk_cone(5, 1.0, 0.8)
    v = v.copy()
    v[:, 2] = np.where(np.isin(np.arange(len(v)), b), 0.0, v[:, 2])
    v2, t2, b2 = add_base_mesh(v, t, b)
    mask = base_triangle_mask(v2, t2, b2)
    assert mask.any()
    assert volume_and_grad(v2, t2)[0] == pytest.approx(volume_and_grad(v, t)[0], rel=1e-12)
    # closed surface: every edge is shared by exactly two triangles
    e = np.sort(np.vstack([t2[:, [0, 1]], t2[:, [1, 2]], t2[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    assert np.all(cnt == 2)
    # strip_base is the inverse
    v3, t3, b3 = strip_base(v2, t2, b2)
    assert len(t3) == len(t) and np.allclose(v3, v)


def test_closed_mesh_energy_equals_open_mesh_energy(rng):
    """gamma*A_dome - cos(theta)*gamma*A_base (base counted signed) equals the
    open-mesh form gamma*A_dome + E_wall(shoelace)."""
    v, t, b = make_disk_cone(5, 1.0, 0.8)
    # non-circular, random ring to make the test non-trivial; keep it star-shaped
    ang = np.arctan2(v[b, 1], v[b, 0])
    r = np.hypot(v[b, 0], v[b, 1]) * (1 + 0.08 * np.sin(3 * ang))
    v = v.copy()
    v[b, 0], v[b, 1], v[b, 2] = r * np.cos(ang), r * np.sin(ang), 0.0
    gamma, theta = 0.88, np.radians(120.0)
    kappa = np.cos(theta)
    A_dome = area_and_grad(v, t)[0]
    E_open = gamma * A_dome + gamma * substrate_wall_energy_and_grad(v, b, kappa)[0]

    v2, t2, b2 = add_base_mesh(v, t, b)
    base = base_triangle_mask(v2, t2, b2)
    w = np.where(base, -gamma * kappa, gamma)
    E_closed = area_and_grad(v2, t2, weights=w, flat_down=base)[0]
    assert E_closed == pytest.approx(E_open, rel=1e-12, abs=1e-12)


def test_nonconvex_ring_falls_back_and_keeps_signed_area():
    n = 40
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1 + 0.45 * np.cos(5 * th)            # strongly non-convex star
    ring = np.c_[r * np.cos(th), r * np.sin(th), np.zeros(n)]
    # tiny open dome over that ring: apex + fan
    apex = np.array([[0, 0, 0.5]])
    v = np.vstack([ring, apex])
    t = np.array([[i, (i + 1) % n, n] for i in range(n)])
    b = np.arange(n)
    v2, t2, b2 = add_base_mesh(v, t, b)
    base = base_triangle_mask(v2, t2, b2)
    A_signed = area_and_grad(v2, t2, flat_down=base)[2][base].sum()
    assert A_signed == pytest.approx(_poly_area(ring[:, :2]), rel=1e-10)
