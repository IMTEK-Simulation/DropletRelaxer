"""The three viewer modes: replay a run file (optionally following a run in
progress), export a video from a run file, and run a scenario live."""

import os
import time

from DropletRelaxer.io.reader import RunFile

from .config import from_table
from .history import make_history_slider, series_from_file
from .interaction import enable_right_button_pan
from .scene import Scene


def _viewer_config(table, overrides, off_screen=None):
    vcfg = from_table(table, overrides)
    if off_screen is not None:
        from dataclasses import replace
        vcfg = replace(vcfg, off_screen=off_screen)
    return vcfg


def _show_latest(scene, rf):
    """Last frame and all chart curves of `rf` on screen."""
    s = series_from_file(rf)
    scene.set_series(list(s.x), list(s.energy), list(s.force))
    scene.show_frame(rf.frame(-1))


def replay(path, overrides=(), follow=False, poll_interval_ms=500):
    """Window on a run file. With `follow`, the window keeps up with a run that
    is still being written (the file is re-read every `poll_interval_ms`) and
    gets the history slider once the run has ended."""
    rf = RunFile(path, follow=follow)
    run = rf.run_info()
    scene = Scene(run, _viewer_config(rf.viewer_table(), overrides))
    save_dir = os.path.dirname(os.path.abspath(path))
    state = {"n": 0, "slider": False}

    def attach_slider():
        if len(rf) > 1 and not state["slider"]:
            make_history_slider(scene, rf.frames(), series_from_file(rf), save_dir=save_dir)
            state["slider"] = True

    if len(rf):
        _show_latest(scene, rf)
        state["n"] = len(rf)
    scene.open_window(title=f"Droplet Relaxer -- {os.path.basename(path)}")
    plotter = scene.plotter

    if follow and rf.status == "running":
        def poll(_step):
            if state["slider"]:
                return
            rf.refresh()
            n = len(rf)
            if n > state["n"]:
                _show_latest(scene, rf)
                state["n"] = n
                scene.update()
            if rf.status != "running":
                print(f"Run ended ({rf.status}).")
                attach_slider()
                scene.update()
        plotter.iren.add_timer_event(max_steps=10_000_000, duration=poll_interval_ms, callback=poll)
        print(f"Following {path} ({len(rf)} frames so far) -- close the window to stop watching.")
    else:
        attach_slider()
    if scene.vcfg.pan_with_right_button:
        enable_right_button_pan(plotter)
    plotter.show()
    rf.close()


def export_video(path, out, overrides=()):
    """Renders the frames of a run file into a video (off screen)."""
    from .video import record_video, write_frame_table
    with RunFile(path) as rf:
        if not len(rf):
            raise SystemExit(f"{path} has no frames")
        scene = Scene(rf.run_info(), _viewer_config(rf.viewer_table(), overrides, off_screen=True))
        series = series_from_file(rf)
        s = series.arrays()
        scene.set_series(list(s["x"]), list(s["energy"]), list(s["force"]))

        def show_index(k):
            scene.show_frame(rf.frame(k))
            scene.mark(s["x"][k], s["energy"][k], s["force"][k])

        written = record_video(scene, out, len(rf), show_index)
        if rf.run_info().quasi_static:
            write_frame_table(os.path.splitext(written)[0] + "_frames.csv", series)
        scene.plotter.close()
    return written


def live(config_path, overrides=(), output=None, force=False):
    """Runs a scenario with the live window (solver in this process). The run
    is always written to a file (default runs/<timestamp>/run.vtkhdf); after the
    run the window replays it."""
    import sys

    from DropletRelaxer.cli import install_signal_handlers
    from DropletRelaxer.config import load, loads
    from DropletRelaxer.console import ConsoleSink
    from DropletRelaxer.driver import Driver
    from DropletRelaxer.io.frame import StopRequest
    from DropletRelaxer.io.writer import H5Writer

    from .live import LiveWindow

    cfg, table = load(config_path, overrides) if config_path else loads("", overrides)
    vcfg = from_table(table.get("viewer"))
    if output is None:
        output = os.path.join("runs", time.strftime("%Y%m%d_%H%M%S"), "run.vtkhdf")
    if os.path.exists(output) and not force:
        raise SystemExit(f"{output} exists; use --force to overwrite it")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    print(f"Writing the run to {output}")
    stop = StopRequest()
    install_signal_handlers(stop)      # Ctrl-C ends the run cleanly, like the Stop button
    sinks = [ConsoleSink(),
             H5Writer(output, viewer_table=table.get("viewer"), command_line=sys.argv),
             LiveWindow(vcfg, save_dir=os.path.dirname(os.path.abspath(output)), run_file=output)]
    return Driver(cfg, sinks=sinks, stop=stop).run()
