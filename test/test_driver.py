"""The driver and its sinks: frames, events, stop requests, and that nothing a
sink does changes the numerics."""
import subprocess
import sys

import numpy as np
import pytest

from DropletRelaxer.config import Config
from DropletRelaxer.driver import Driver
from DropletRelaxer.io import NullSink


def short_cfg(n_iterations=430):
    """Sessile drop that remeshes early (global split at 300, collapse at 360)."""
    cfg = Config()
    cfg.obstacle.enabled = False
    cfg.roughness.enabled = False
    cfg.quasi.enabled = False
    cfg.solver.contact_angle_deg = 120.0
    cfg.solver.n_iterations = n_iterations
    cfg.mesh.remesh_split_mode = "global"
    cfg.mesh.remesh_trigger_ratio = 1.3
    cfg.mesh.remesh_min_edge_ratio = 0.6
    cfg.output.every = 100
    return cfg


class Recorder(NullSink):
    def __init__(self, stop_at=None, mutate=False):
        self.stop_at, self.mutate = stop_at, mutate
        self.frames, self.events, self.ticks, self.status, self.run = [], [], 0, None, None

    def start(self, run):
        self.run = run

    def frame(self, f):
        self.frames.append(f)

    def event(self, e):
        self.events.append(e)

    def tick(self, i, progress, eta):
        self.ticks += 1
        assert 0.0 <= progress <= 1.0
        if self.stop_at is not None and i >= self.stop_at:
            self.run.stop.request("test")

    def finish(self, status):
        self.status = status


def test_solver_imports_without_pyvista():
    code = ("import sys\n"
            "for m in ('pyvista', 'vtk', 'vtkmodules'): sys.modules[m] = None\n"
            "import DropletRelaxer.driver, DropletRelaxer.console, DropletRelaxer.io\n")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_sink_does_not_change_result():
    plain = Driver(short_cfg()).run()
    rec = Recorder()
    with_sink = Driver(short_cfg(), sinks=[rec]).run()
    assert np.array_equal(plain["verts"], with_sink["verts"])
    assert plain["energy"] == with_sink["energy"] and plain["remeshes"] == with_sink["remeshes"]
    assert rec.ticks > 0 and rec.status.reason == "finished"


def test_frames_and_events():
    rec = Recorder()
    res = Driver(short_cfg(), sinks=[rec]).run()
    its = [f.iteration for f in rec.frames]
    assert its[0] == 0 and its[-1] == res["iteration"] == 430   # final state recorded
    assert all(i % 100 == 0 or i % 60 == 0 for i in its[:-1])      # cadence or remesh checks
    assert its == sorted(set(its))
    remeshes = [e for e in rec.events if e.kind == "remesh"]
    assert len(remeshes) == res["remeshes"] >= 2
    assert [e.data["epoch"] for e in remeshes] == list(range(1, len(remeshes) + 1))
    # each frame is consistent with its topology, arrays are read-only copies
    for f in rec.frames:
        assert f.tris.max() < len(f.verts) and len(f.curvature) == len(f.verts)
        assert not f.verts.flags.writeable and not f.tris.flags.writeable
        assert np.isnan(f.gap) and f.sphere_center is None
    with pytest.raises(ValueError):
        rec.frames[0].verts[0, 0] = 1.0
    assert rec.frames[-1].epoch == res["remeshes"]


def test_stop_request_ends_run_cleanly():
    rec = Recorder(stop_at=150)
    res = Driver(short_cfg(), sinks=[rec]).run()
    assert res["iteration"] == 150
    assert rec.status.reason == "stopped" and rec.status.message == "test"
    assert rec.frames[-1].iteration == 150    # the final state is recorded
