"""Topology setup is reused by a run, never across distinct meshes."""

import numpy as np
import pytest

from DropletRelaxer import driver, _fast, substrate
from DropletRelaxer.anomaly import disc_integral_and_grad
from DropletRelaxer.mesh import _make_start_mesh_with_topology, build_dof_mask
from DropletRelaxer.solver import relax
from kernel_parity_helpers import assert_bits
from test_driver import Recorder, short_cfg


@pytest.fixture
def fast_kernels(monkeypatch):
    if not _fast.numba_available():
        pytest.skip("requires Numba")
    monkeypatch.setattr(_fast, "_native_requested", False)


class Checkpoints(Recorder):
    def checkpoint(self, state):
        self.state = state


def test_driver_reuses_edges_and_base_setup_until_remesh(monkeypatch, fast_kernels):
    built = []
    segments = []
    edge_checks = []
    build_embedder = substrate.BaseEmbedder
    run_segment = driver.relax
    max_edge_length = driver.max_edge_length

    def build(*args):
        embedder = build_embedder(*args)
        built.append(embedder)
        return embedder

    def run(*args, _topology, **kwargs):
        segments.append(_topology)
        # The base builder already set up the factorization for this segment.
        assert _topology.embedder is built[-1]
        return run_segment(*args, _topology=_topology, **kwargs)

    def measure(verts, tris, *, edges=None):
        if edges is not None:
            assert edges is segments[-1].dome_edges
            edge_checks.append(edges)
        return max_edge_length(verts, tris, edges=edges)

    monkeypatch.setattr(substrate, "BaseEmbedder", build)
    monkeypatch.setattr(driver, "relax", run)
    monkeypatch.setattr(driver, "max_edge_length", measure)

    cfg = short_cfg()
    cfg.mesh.base_mesh_enabled = True
    result = driver.Driver(cfg).run()
    assert result["remeshes"] >= 2
    assert len(segments) == len(built) == result["remeshes"] + 1
    assert len({id(segment.dome_edges) for segment in segments}) == len(segments)
    assert len(edge_checks) > 1
    assert all(not segment.dome_edges.flags.writeable for segment in segments)


@pytest.mark.parametrize("change", ["triangles", "ring", "vertex count"])
def test_relax_rejects_setup_for_a_different_mesh(change, fast_kernels):
    cfg = short_cfg()
    verts, tris, boundary, topology = _make_start_mesh_with_topology(cfg.mesh, reuse_topology=True)
    if change == "triangles":
        tris = tris[::-1].copy()
    elif change == "ring":
        boundary = np.roll(boundary, 1)
    else:
        verts = np.vstack([verts, verts[-1]])
    mask = build_dof_mask(len(verts), boundary)
    with pytest.raises(ValueError, match="topology belongs to a different mesh"):
        next(relax(verts, tris, boundary, mask, cfg.solver, _topology=topology))


def test_restart_rebuilds_setup_and_matches_uninterrupted_run(monkeypatch, fast_kernels):
    cfg = short_cfg()
    cfg.mesh.base_mesh_enabled = True
    full = driver.Driver(cfg).run()
    stop = Checkpoints(stop_at=315)
    partial = driver.Driver(cfg, sinks=[stop]).run()
    assert 0 < partial["remeshes"] < full["remeshes"]
    assert stop.state["iteration"] == partial["iteration"] == 315

    rebuild = _fast.MeshTopology
    restarted_topologies = []

    def track_rebuild(verts, tris, boundary):
        topology = rebuild(verts, tris, boundary)
        restarted_topologies.append(topology)
        return topology

    monkeypatch.setattr(_fast, "MeshTopology", track_rebuild)
    resumed = driver.Driver(cfg, restart=stop.state).run()
    restored = [topology for topology in restarted_topologies
                if np.array_equal(topology.tris, stop.state["tris"])]
    assert len(restored) == 1
    assert restored[0].tris is not stop.state["tris"]
    assert_bits(restored[0].boundary_idx, stop.state["boundary_idx"])
    for key in ("verts", "tris", "boundary_idx", "energy", "area", "volume", "iteration", "remeshes"):
        assert_bits(resumed[key], full[key])


@pytest.mark.parametrize("closed", [False, True])
def test_native_never_uses_topology_cache(monkeypatch, closed):
    def unexpected_topology(*args, **kwargs):
        pytest.fail("native run constructed a fast topology cache")

    segment = driver.relax

    def run(*args, _topology, **kwargs):
        assert _topology is None
        return segment(*args, _topology=_topology, **kwargs)

    max_edge_length = driver.max_edge_length

    def measure(verts, tris, *, edges=None):
        assert edges is None
        return max_edge_length(verts, tris)

    monkeypatch.setattr(_fast, "_native_requested", True)
    monkeypatch.setattr(_fast, "MeshTopology", unexpected_topology)
    monkeypatch.setattr(_fast, "base_rhs_fast", unexpected_topology)
    monkeypatch.setattr(_fast, "numba_kernels", unexpected_topology)
    monkeypatch.setattr(driver, "relax", run)
    monkeypatch.setattr(driver, "max_edge_length", measure)

    cfg = short_cfg()
    cfg.mesh.base_mesh_enabled = closed
    assert driver.Driver(cfg).run()["remeshes"] >= 2


def test_native_anomaly_does_not_use_fast_caches(monkeypatch):
    leggauss = np.polynomial.legendre.leggauss
    orders = []

    def quadrature(order):
        orders.append(order)
        return leggauss(order)

    def unexpected_cache(*args, **kwargs):
        pytest.fail("native anomaly used the fast setup cache")

    monkeypatch.setattr(_fast, "_native_requested", True)
    monkeypatch.setattr(_fast, "numba_kernels", unexpected_cache)
    monkeypatch.setattr(np.polynomial.legendre, "leggauss", quadrature)
    verts = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
    args = (verts, np.arange(3), (0.3, 0.4), 0.5, 0.1, 16)
    disc_integral_and_grad(*args)
    disc_integral_and_grad(*args)
    assert orders == [16, 16]
