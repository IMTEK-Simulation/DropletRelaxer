"""The live window as a frame sink, off screen."""
import os
import sys

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

# the solver's test helpers (short_cfg, Recorder) are shared with the viewer tests
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test"))

from DropletRelaxer.driver import Driver  # noqa: E402
from DropletViewer.config import ViewerConfig  # noqa: E402
from DropletViewer.history import Series  # noqa: E402
from DropletViewer.live import LiveWindow  # noqa: E402
from test_driver import Recorder, short_cfg  # noqa: E402


def test_live_window_does_not_change_result():
    rec = Recorder()
    plain = Driver(short_cfg(), sinks=[rec]).run()
    win = LiveWindow(ViewerConfig(off_screen=True))
    live = Driver(short_cfg(), sinks=[win]).run()
    assert np.array_equal(plain["verts"], live["verts"])
    assert len(win.frames) == len(rec.frames)
    assert win.scene.epoch == win.frames[-1].epoch


def test_scene_replays_frames_across_epochs():
    win = LiveWindow(ViewerConfig(off_screen=True))
    Driver(short_cfg(), sinks=[win]).run()
    # a fresh scene shows every recorded frame, back and forth across remeshes
    from DropletViewer.scene import Scene
    sc = Scene(win.run, ViewerConfig(off_screen=True))
    for f in win.frames + win.frames[::-1]:
        sc.show_frame(f)
        assert sc.mesh.n_points == len(f.verts)
        assert np.allclose(sc.mesh["curvature"], f.curvature, equal_nan=True)
    img = sc.plotter.screenshot(return_img=True)
    assert img.std() > 0
    sc.plotter.close()


def test_series_arrays():
    win = LiveWindow(ViewerConfig(off_screen=True))
    Driver(short_cfg(), sinks=[win]).run()
    s = win.series.arrays()
    assert s["force"].shape == (len(win.frames), 3)
    assert np.all(np.isnan(s["sphere_center"]))
    assert list(s["iteration"]) == [f.iteration for f in win.frames]
    assert isinstance(win.series, Series)


def test_viewer_config_documented_and_strict():
    from dataclasses import fields

    from DropletRelaxer.config import ConfigError, field_docs
    from DropletViewer.config import ViewerConfig, from_table
    docs = field_docs(ViewerConfig)
    assert all(docs.get(f.name) for f in fields(ViewerConfig))
    v = from_table({"curvature_clim": [0, 2]}, ["colormap=viridis"])
    assert v.curvature_clim == (0.0, 2.0) and v.colormap == "viridis"
    with pytest.raises(ConfigError, match="did you mean"):
        from_table({"colormp": "x"})


def test_droplet_view_live_off_screen_writes_and_replays(tmp_path):
    import subprocess

    from DropletRelaxer.config import dumps
    from DropletRelaxer.io.reader import RunFile
    cfg = short_cfg()
    scen = tmp_path / "s.toml"
    scen.write_text(dumps(cfg, viewer={"off_screen": True}))
    out = tmp_path / "run.h5"
    subprocess.run([sys.executable, "-m", "DropletViewer.cli", "live", str(scen), "-o", str(out)],
                   check=True, capture_output=True)
    with RunFile(str(out)) as rf:
        assert rf.status == "finished" and len(rf) > 2
    video = tmp_path / "run.mp4"
    subprocess.run([sys.executable, "-m", "DropletViewer.cli", str(out), "--video", str(video)],
                   check=True, capture_output=True)
    assert video.stat().st_size > 0
