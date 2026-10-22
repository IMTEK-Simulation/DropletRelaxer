"""Console output as a sink: log lines and (optionally) a one-line progress bar."""

import sys
import time

from .io.frame import NullSink


def format_duration(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m{sec % 60:02d}s" if sec >= 3600 else \
        f"{sec // 60}m{sec % 60:02d}s"


class ConsoleSink(NullSink):
    """Prints the driver's log events. With `progress=True` it also keeps a
    progress line (fraction, iteration, elapsed, ETA, remeshes) at the bottom,
    redrawn at most every `interval` seconds; log lines are printed above it."""

    def __init__(self, progress=False, width=30, interval=0.25, stream=None):
        self.progress, self.width, self.interval = progress, width, interval
        self.stream = stream if stream is not None else sys.stdout
        self.t0 = time.perf_counter()
        self.t_last = 0.0
        self.remeshes = 0
        self._bar_shown = False
        self._state = (0, 0.0, None)

    def _clear_bar(self):
        if self._bar_shown:
            self.stream.write("\r" + " " * (self.width + 80) + "\r")
            self._bar_shown = False

    def _draw_bar(self):
        i, frac, eta = self._state
        el = time.perf_counter() - self.t0
        done = int(round(frac * self.width))
        eta_s = format_duration(eta) if eta is not None else "--"
        self.stream.write(f"\r[{'#' * done}{'-' * (self.width - done)}] {100 * frac:5.1f}%  "
                          f"iter {i}  elapsed {format_duration(el)}  ETA {eta_s}  remeshes {self.remeshes}   ")
        self.stream.flush()
        self._bar_shown = True

    def start(self, run):
        self.t0 = time.perf_counter()

    def event(self, e):
        if e.kind == "remesh":
            self.remeshes += 1
        elif e.kind == "log":
            self._clear_bar()
            print(e.message, file=self.stream, flush=True)
            if self.progress and self.t_last > 0:
                self._draw_bar()

    def tick(self, i, progress, eta):
        if not self.progress:
            return
        self._state = (i, progress, eta)
        now = time.perf_counter()
        if now - self.t_last < self.interval:
            return
        self.t_last = now
        self._draw_bar()

    def finish(self, status):
        if self.progress and self._bar_shown:
            self._draw_bar()
            self.stream.write("\n")
            self.stream.flush()
            self._bar_shown = False
