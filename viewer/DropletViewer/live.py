"""In-process live window: a frame sink for DropletRelaxer.driver.Driver.

The solver runs synchronously on the main thread (VTK must own it; no
threads). `frame()` redraws the scene, `tick()` (called every iteration)
lets the window handle its events at most every `event_interval` seconds, so
it can be rotated or closed between frames, and refreshes the progress bar
once a second. Closing the window or pressing Stop sets the run's
StopRequest; the driver then ends the run cleanly. Nothing else flows back
into the numerics, so a run gives identical results with or without a window.
"""

import time

from DropletRelaxer.console import format_duration
from DropletRelaxer.io.frame import NullSink

from .config import ViewerConfig
from .history import Series, make_history_slider
from .scene import ProgressBar, Scene


def _progress_text(frac, elapsed, eta):
    """Text inside the window's progress bar."""
    eta_s = "..." if eta is None else "~" + format_duration(eta)
    return f"Progress {100 * frac:.0f} %   |   elapsed {format_duration(elapsed)}   |   ETA {eta_s}"


class LiveWindow(NullSink):
    """Live PyVista window. After the run it stays open with the history
    slider, Play and Save (video into `save_dir`), unless it was closed or is
    off screen."""

    def __init__(self, vcfg=None, save_dir=None, event_interval=0.1, progress_interval=1.0):
        self.vcfg = vcfg if vcfg is not None else ViewerConfig()
        self.save_dir = save_dir
        self.event_interval, self.progress_interval = event_interval, progress_interval
        self.scene = None
        self.closed = False

    # -- sink interface ---------------------------------------------------

    def start(self, run):
        self.run = run
        self.scene = scene = Scene(run, self.vcfg)
        self.frames, self.series = [], Series.empty()
        self.t0 = self.t_events = self.t_progress = time.perf_counter()
        scene.open_window()
        self.bar = ProgressBar(scene.plotter)
        plotter = scene.plotter
        self._stop_button = plotter.add_checkbox_button_widget(
            self._stop_pressed, value=False, position=(10.0, 10.0), size=28,
            color_on="#d62728", color_off="lightgray")
        self._stop_text = plotter.add_text("Stop", position=(44, 14), font_size=10, color="black")
        if not scene.off_screen:
            plotter.iren.add_observer("ExitEvent", self._window_closed)

    def frame(self, f):
        if self.closed:
            return
        self.frames.append(f)
        self.series.append(f)
        self.scene.show_frame(f)
        self.scene.set_series(self.series.x, self.series.energy, self.series.force)
        self.scene.update()

    def event(self, e):
        if self.closed or e.kind != "remesh":
            return
        d = e.data
        self.scene.show_mesh(d["epoch"], d["verts"], d["tris"], d["boundary_idx"],
                             d["sphere_center"], d["wetted_loop"])
        self.scene.update()

    def tick(self, i, progress, eta):
        if self.closed or self.scene.off_screen:
            return
        now = time.perf_counter()
        if now - self.t_progress >= self.progress_interval:
            # Refreshing the bar redraws the (unchanged) scene and lets the
            # window handle its events; the mesh itself is only redrawn per frame.
            self.t_progress = self.t_events = now
            self.bar.update(progress, _progress_text(progress, now - self.t0, eta))
            self.scene.update()
        elif now - self.t_events >= self.event_interval:
            self.t_events = now
            self.scene.process_events()
        if self._window_gone():
            self._window_closed()

    def finish(self, status):
        if self.scene is None:
            return
        plotter = self.scene.plotter
        if self.closed or self._window_gone():
            return
        self.bar.hide()   # the history slider takes its place
        plotter.clear_button_widgets()
        plotter.remove_actor(self._stop_text, render=False)
        if self.scene.off_screen:
            # No display attached (automated test/server run) -- nothing to
            # keep open, so close and release resources right away.
            plotter.close()
            return
        if len(self.frames) > 1:
            make_history_slider(self.scene, self.frames, self.series, save_dir=self.save_dir)
        print("Relaxation finished. The window stays open -- "
              "rotate/zoom freely, close it (or press 'q') when done.")
        if self.vcfg.pan_with_right_button:
            # Re-assert (idempotent; see enable_right_button_pan's docstring).
            from .interaction import enable_right_button_pan
            enable_right_button_pan(plotter)
        plotter.show()

    # -- window state -------------------------------------------------------

    def _stop_pressed(self, is_on):
        if is_on:
            self.run.stop.request("Stop button")

    def _window_closed(self, *_args):
        if not self.closed:
            self.closed = True
            self.run.stop.request("window closed")

    def _window_gone(self):
        p = self.scene.plotter
        return bool(getattr(p, "_closed", False)) or p.render_window is None
