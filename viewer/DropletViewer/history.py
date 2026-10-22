"""History slider with Play and Save buttons: scrubs through the recorded
frames of a run after it has finished (or of a run file)."""

import os

import numpy as np

from .video import record_video, write_frame_table


class Series:
    """Per-frame scalars of a run, one entry per recorded frame: what the
    charts plot and the slider needs without loading any mesh. Built
    incrementally from Frames (live window) or read from a run file."""

    KEYS = ("iteration", "x", "energy", "volume", "force", "sphere_center", "step_iters", "converged")

    def __init__(self, **arrays):
        for k in self.KEYS:
            setattr(self, k, arrays.get(k, []))

    @classmethod
    def empty(cls):
        return cls(**{k: [] for k in cls.KEYS})

    def append(self, f):
        self.iteration.append(f.iteration)
        self.x.append(f.x)
        self.energy.append(f.energy)
        self.volume.append(f.volume)
        self.force.append(np.asarray(f.force))
        self.sphere_center.append(np.full(3, np.nan) if f.sphere_center is None else np.asarray(f.sphere_center))
        self.step_iters.append(f.step_iters)
        self.converged.append(f.converged)

    def __len__(self):
        return len(self.x)

    def arrays(self):
        """The series as numpy arrays (force and sphere_center (N, 3))."""
        return {k: np.asarray(getattr(self, k)).reshape((len(self), 3) if k in ("force", "sphere_center")
                                                         else (len(self),)) for k in self.KEYS}


def series_from_file(rf):
    """Series of a run file (io.reader.RunFile)."""
    s = rf.series()
    return Series(**{k: s[k] for k in Series.KEYS})


def make_history_slider(scene, frames, series, save_dir=None):
    """Adds the slider widget -- call this once, after the relaxation has
    finished -- that scrubs through the recorded frames, so you can look at
    any frame again instead of only the final shape. Also adds a Play button
    that steps through the same frames on a timer, and (with `save_dir`) a
    Save button that records the replay as a video into that folder.

    `frames` is a sequence of io.frame.Frame (a list, or a lazy run-file
    reader); `series` the matching Series. The slider value is the frame's x
    (tip x-coordinate in quasi-static mode, iteration otherwise) and snaps to
    the nearest recorded frame; if x does not vary (a path without x motion)
    the iteration number is used instead. Frames of a different topology epoch
    swap in a new actor, exactly like a live remesh."""
    plotter, vcfg = scene.plotter, scene.vcfg
    s = series.arrays()
    x_arr, x_label = s["x"].astype(float), scene.x_label
    if np.ptp(x_arr) < 1e-9:
        x_arr = s["iteration"].astype(float)
        x_label = "Iteration"
    n = len(x_arr)
    state = {"idx": n - 1, "playing": False}

    def callback(value):
        idx = int(np.argmin(np.abs(x_arr - value)))   # nearest recorded frame
        scene.show_frame(frames[idx])
        # The curves stay the full, finished run; only the markers move.
        scene.mark(s["x"][idx], s["energy"][idx], s["force"][idx])
        # Keep the play cursor in sync with whatever frame is now shown,
        # whether we got here from a manual drag or from _advance_play
        # below, so pressing Play always resumes from HERE.
        state["idx"] = idx
        try:
            plotter.update()
        except RuntimeError:
            pass  # widget construction fires the callback once before
            # the render window interactor is fully initialized in some
            # PyVista/VTK versions -- harmless, the next real interaction
            # (or the widget's own render call) picks the update up.

    lo, hi = float(x_arr.min()), float(x_arr.max())
    if hi - lo < 1e-9:
        hi = lo + 1.0
    slider = plotter.add_slider_widget(
        callback,
        rng=[lo, hi],
        value=float(x_arr[-1]),
        title=x_label,
        pointa=(0.22, 0.05), pointb=(0.78, 0.05),
        style="modern",
    )

    def show_index(idx):
        value = float(x_arr[idx])
        slider.GetRepresentation().SetValue(value)
        callback(value)

    # Play button: one long-lived repeating timer (started once, right here)
    # that no-ops on every tick while paused, rather than creating/destroying a
    # VTK timer on every Play/Pause click -- PyVista's timer wrapper doesn't
    # expose a clean "cancel" for that. See ViewerConfig.playback_max_frames:
    # advance by more than one recorded frame per tick when there are a lot
    # of them, so one full playback pass does a bounded amount of work.
    play_stride = max(1, n // max(vcfg.playback_max_frames, 1))

    def _advance_play(_step):
        if not state["playing"]:
            return
        idx = min(state["idx"] + play_stride, n - 1)
        if idx <= state["idx"]:
            # Reached the end: stop and visually untoggle the button.
            state["playing"] = False
            play_button.GetRepresentation().SetState(0)
            plotter.render()
            return
        show_index(idx)

    plotter.iren.add_timer_event(max_steps=10_000_000, duration=vcfg.playback_interval_ms,
                                 callback=_advance_play)

    def _toggle_play(is_on):
        if is_on:
            # Replaying after having reached the end: restart from the beginning.
            if state["idx"] >= n - 1:
                state["idx"] = -1
            state["playing"] = True
        else:
            state["playing"] = False

    play_button = plotter.add_checkbox_button_widget(
        _toggle_play, value=False, position=(10.0, 10.0), size=28,
        color_on="#2ca02c", color_off="lightgray",
    )
    plotter.add_text("Play", position=(44, 14), font_size=10, color="black")

    def save(folder):
        """Records the replay into `folder`/run.mp4 (+ frames.csv in
        quasi-static mode); returns the video path."""
        state["playing"] = False
        play_button.GetRepresentation().SetState(0)
        os.makedirs(folder, exist_ok=True)
        if scene.run.quasi_static:
            write_frame_table(os.path.join(folder, "frames.csv"), series)
        restore = state["idx"]
        try:
            return record_video(scene, os.path.join(folder, "run.mp4"), n, show_index)
        finally:
            show_index(restore)

    if save_dir is not None:
        # "Save" button next to Play: a push-style checkbox that springs back to "off".
        def _save_pressed(is_on):
            if not is_on:
                return
            save_button.GetRepresentation().SetState(0)  # springs back at once (so it is not "on" in the video)
            try:
                print("Recording video (the window shows the replay) ...")
                path = save(save_dir)
                print(f"Video saved to {os.path.abspath(path)}")
            except Exception as err:  # never kill the viewer over a failed save
                print(f"Save failed: {err}")
            finally:
                save_button.GetRepresentation().SetState(0)
                try:
                    plotter.render()
                except Exception:
                    pass

        save_button = plotter.add_checkbox_button_widget(
            _save_pressed, value=False, position=(100.0, 10.0), size=28,
            color_on="#1f77b4", color_off="lightgray",
        )
        plotter.add_text("Save", position=(134, 14), font_size=10, color="black")

    return save
