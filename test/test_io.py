"""Run file: round trip, VTKHDF bookkeeping, restart, CLI."""
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest

from DropletRelaxer.driver import Driver
from DropletRelaxer.io import NullSink
from DropletRelaxer.io.reader import RunFile
from DropletRelaxer.io.writer import H5Writer
from DropletRelaxer.config import ConfigError, dumps, loads
from test_driver import Recorder, short_cfg


def assert_frames_equal(a, b):
    for name in ("index", "iteration", "x", "epoch", "energy", "area", "volume", "gap", "step_iters", "converged"):
        va, vb = getattr(a, name), getattr(b, name)
        assert va == vb or (np.isnan(va) and np.isnan(vb)), name
    for name in ("verts", "tris", "boundary_idx", "force", "com", "com_offset_xy", "curvature"):
        assert np.array_equal(getattr(a, name), getattr(b, name), equal_nan=True), name
    for name in ("wetted_loop", "sphere_center"):
        va, vb = getattr(a, name), getattr(b, name)
        assert (va is None and vb is None) or np.array_equal(va, vb), name


def cfg_with_checkpoints():
    cfg = short_cfg()
    cfg.output.checkpoint_every = 100
    return cfg


def test_round_trip(tmp_path):
    path = tmp_path / "run.h5"
    rec = Recorder()
    Driver(cfg_with_checkpoints(), sinks=[rec, H5Writer(str(path), viewer_table={"colormap": "viridis"})]).run()
    with RunFile(str(path)) as rf:
        assert len(rf) == len(rec.frames) and rf.status == "finished"
        for a, b in zip(rec.frames, rf.frames()):
            assert_frames_equal(a, b)
        s = rf.series()
        assert np.array_equal(s["energy"], [f.energy for f in rec.frames])
        assert s["force"].shape == (len(rec.frames), 3)
        assert rf.viewer_table() == {"colormap": "viridis"}
        cfg, _ = rf.config()
        assert cfg == cfg_with_checkpoints()           # the config as given, not the scaled one
        info = rf.run_info()
        assert info.mesh_scale == rec.run.mesh_scale and info.config == rec.run.config
        ev = rf.events()
        assert [e["kind"] for e in ev] == ["remesh"] * len([e for e in rec.events if e.kind == "remesh"])
        assert rf.checkpoint()["iteration"] == 430
        assert "frames" in rf.describe()
    # VTKHDF bookkeeping: per-step offsets are consistent with the counts
    with h5py.File(path) as f:
        st, v = f["VTKHDF/Steps"], f["VTKHDF"]
        n = st.attrs["NSteps"]
        assert np.array_equal(st["PointOffsets"][:], np.r_[0, np.cumsum(v["NumberOfPoints"][:])[:-1]])
        cells = v["Polygons/NumberOfCells"][:]
        assert np.array_equal(st["CellOffsets"][:, 2], np.r_[0, np.cumsum(cells)[:-1]])
        assert v["Polygons/Offsets"].shape[0] == cells.sum() + n
        assert v["FieldData/sphere_center"].shape == (n, 3)


class Kill(NullSink):
    """Aborts the run like a wall-time kill (no final frame or checkpoint)."""

    def __init__(self, at):
        self.at = at

    def tick(self, i, progress, eta):
        if i >= self.at:
            raise KeyboardInterrupt


def test_restart_after_kill_reproduces_uninterrupted_run(tmp_path):
    rec_full = Recorder()
    full = Driver(cfg_with_checkpoints(), sinks=[rec_full]).run()

    path = str(tmp_path / "run.h5")
    writer = H5Writer(path)
    with pytest.raises(KeyboardInterrupt):
        Driver(cfg_with_checkpoints(), sinks=[writer, Kill(370)]).run()
    writer.f.close()                              # what the OS does to a killed process
    with RunFile(path) as rf:
        assert rf.status == "running"
        ck = rf.checkpoint()
        cfg, table = rf.config()
        assert ck["iteration"] == 200 and len(rf) > ck["scalars"]["n_frames"]   # frames past the checkpoint
    res = Driver(cfg, sinks=[H5Writer(path, append=True)], restart=ck).run()
    assert np.array_equal(res["verts"], full["verts"]) and res["energy"] == full["energy"]
    assert res["remeshes"] == full["remeshes"]
    with RunFile(path) as rf:
        assert rf.status == "finished" and len(rf) == len(rec_full.frames)
        for a, b in zip(rec_full.frames, rf.frames()):
            assert_frames_equal(a, b)
        assert [e["iteration"] for e in rf.events()] == [e.iteration for e in rec_full.events if e.kind == "remesh"]


def test_restart_with_more_iterations_continues(tmp_path):
    path = str(tmp_path / "run.h5")
    Driver(cfg_with_checkpoints(), sinks=[H5Writer(path)]).run()
    longer = cfg_with_checkpoints()
    longer.solver.n_iterations = 600
    full = Driver(longer).run()
    with RunFile(path) as rf:
        cfg, _ = rf.config(["solver.n_iterations=600"])
        ck = rf.checkpoint()
        with pytest.raises(ConfigError, match="cannot change at a restart"):
            rf.config(["solver.contact_angle_deg=30"])
    res = Driver(cfg, sinks=[H5Writer(path, append=True)], restart=ck).run()
    # The sphere-free schedule does not depend on n_iterations, so the
    # continued run equals one that was given 600 iterations from the start.
    assert np.array_equal(res["verts"], full["verts"])


def test_quasistatic_state_round_trip():
    from types import SimpleNamespace

    from DropletRelaxer.quasistatic import QuasiStatic
    q = SimpleNamespace(step=0.1, min_step_factor=0.125, adaptive=True, min_iter_per_step=1, max_iter_per_step=5,
                        check_every=1, tol_energy=1e-6, tol_force=1e-6, tol_step=1e-6)
    o = SimpleNamespace(center_start=(0.0, 0.0, 1.0), center_end=(1.0, 0.0, 1.0))
    a = QuasiStatic(q, o, 1.0, 1.0)
    a.energy_fn = lambda *args: 1.0
    v = np.zeros((4, 3))
    a.observe(v, np.ones(3), 0.0, True, 1.0, None, None)
    a.advance(v + 1.0)
    a.observe(v, np.ones(3) * 2, 0.5, False, 1.0, None, None)
    import json
    state, ck = a.get_state()
    state = json.loads(json.dumps(state))
    b = QuasiStatic(q, o, 1.0, 1.0)
    b.set_state(state, ck)
    for k in QuasiStatic._STATE:
        assert getattr(a, k) == getattr(b, k), k
    assert np.array_equal(a.prev_F, b.prev_F) and a.ckpt[0] == b.ckpt[0] and np.array_equal(a.ckpt[1], b.ckpt[1])


def test_cli_run_info_restart(tmp_path):
    scen = tmp_path / "s.toml"
    cfg = short_cfg()
    cfg.solver.n_iterations = 200
    scen.write_text(dumps(cfg))
    out = str(tmp_path / "s.h5")
    cli = [sys.executable, "-m", "DropletRelaxer.cli"]
    subprocess.run(cli + ["run", str(scen), "-o", out, "-q"], check=True, capture_output=True)
    again = subprocess.run(cli + ["run", str(scen), "-o", out, "-q"], capture_output=True, text=True)
    assert again.returncode != 0 and "exists" in again.stderr
    info = subprocess.run(cli + ["info", out], check=True, capture_output=True, text=True).stdout
    assert "status: finished" in info and "checkpoint at iteration 200" in info
    subprocess.run(cli + ["restart", out, "-q", "--set", "solver.n_iterations=300"], check=True, capture_output=True)
    with RunFile(out) as rf:
        assert rf.series()["iteration"][-1] == 300
        assert loads(rf.config_text())[0].solver.n_iterations == 200    # the stored scenario is unchanged
    assert os.path.getsize(out) > 0


def test_follow_reads_a_file_while_it_is_written(tmp_path):
    """SWMR: a reader sees complete frames appear while another process writes."""
    import time
    scen = tmp_path / "s.toml"
    cfg = short_cfg()
    cfg.solver.n_iterations = 3000
    cfg.output.every = 50
    scen.write_text(dumps(cfg))
    out = tmp_path / "s.h5"
    proc = subprocess.Popen([sys.executable, "-m", "DropletRelaxer.cli", "run", str(scen), "-o", str(out), "-q"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    seen, rf, t0 = [], None, time.time()
    try:
        while time.time() - t0 < 120:
            if rf is None:
                try:
                    rf = RunFile(str(out), follow=True)
                except (OSError, KeyError):
                    time.sleep(0.05)
                    continue
            rf.refresh()
            n = len(rf)
            if n:
                f = rf.frame(n - 1)                   # the newest complete frame is fully readable
                assert len(f.curvature) == len(f.verts)
            if not seen or n != seen[-1]:
                seen.append(n)
            if rf.status != "running":
                break
            time.sleep(0.02)
    finally:
        proc.wait(timeout=120)
        if rf is not None:
            rf.close()
    assert proc.returncode == 0, proc.stderr.read()
    with RunFile(str(out)) as final:
        assert seen[-1] == len(final) and final.status == "finished"
    assert len(seen) > 2                              # watched the frame count grow
