"""The two 2D charts docked at the top of the window: surface free energy and
the force of the liquid on the sphere, plus the diamond markers on them."""

import numpy as np
import pyvista as pv


def chart_axis(config, total_iters):
    """(x_range, x_label) of the charts for a run: the planned sphere path's
    x (tip x-coordinate) in quasi-static mode, the iteration number otherwise."""
    if config.quasi.enabled and config.obstacle.enabled:
        x0c, x1c = float(config.obstacle.center_start[0]), float(config.obstacle.center_end[0])
        x_range = ((x0c - 1.0, x0c + 1.0) if x1c == x0c
                   else (x0c - 0.03 * (x1c - x0c), x1c + 0.03 * (x1c - x0c)))
        return x_range, "Tip x-coordinate"
    return (0.0, float(max(total_iters, 1))), "Iteration"


# Colours of the axes tripod (pyvista's add_axes_at_origin): x tomato,
# y sea green, z blue. Reused for the gap (y), tip x-coordinate (x) and
# the force components.
_COL_X, _COL_Y, _COL_Z = "#ff6347", "#2e8b57", "#0000ff"
_RGB_X = (1.0, 99 / 255, 71 / 255)


class _Marker:
    """Diamond marker on a Chart2D curve. `update(x, y)` moves it to the point
    (x, y) -- the newest data point of its curve during the run, the scrubbed
    frame in the history slider. Instead of editing the point data of one
    long-lived scatter plot in place, every update replaces the (single-point)
    plot by a fresh one: that does not depend on how a given VTK/pyvista
    version propagates in-place table changes to a chart that is already
    displayed. `right_axis=True` puts the marker on the chart's right-hand y
    axis (plot corner 1), like the curve it belongs to."""

    def __init__(self, chart, color, right_axis=False, size=12):
        self.chart, self.color, self.right_axis, self.size = chart, color, right_axis, size
        self.plot = None
        self._aux = {}      # further markers (key -> _Marker) that follow the same frame
        self.update([0.0], [0.0])

    def update(self, x, y):
        if self.plot is not None:
            try:
                self.chart.remove_plot(self.plot)
            except Exception:
                pass
        self.plot = self.chart.scatter([float(x[-1])], [float(y[-1])], color=self.color, size=self.size)
        self.plot.marker_style = "d"
        if self.right_axis:
            self.chart.SetPlotCorner(self.plot, 1)


def make_energy_chart(plotter, x_range, x_label="Tip x-coordinate"):
    """Two 2D line-chart overlays docked at the top of the render window,
    side by side: the surface free energy (left) and the force of the liquid on
    the sphere, components Fx/Fy/Fz in the colours of the axes tripod (right).
    Both share the x-axis, the tip x-coordinate (sphere centre x) in
    quasi-static mode (one point per converged sphere step) or the iteration
    number in the old iteration-driven mode. The x-axis is fixed to `x_range`
    from the start (the planned sphere path), so the curve's reach across the
    plot shows how far the run has got. Diamond markers (`_Marker`) mark the
    newest data point of every curve (during the run) or the frame shown by
    the history slider. Returns (chart, line, marker, dchart, dline, dmarker);
    call update_energy_chart(...) to append data."""
    x_range = [float(x_range[0]), float(x_range[1])]
    chart = pv.Chart2D(size=(0.42, 0.33), loc=(0.07, 0.65))
    chart.background_color = (1.0, 1.0, 1.0, 0.55)
    chart.border_color = (0.3, 0.3, 0.3)
    chart.x_label = x_label
    chart.y_label = "Surface free energy"
    chart.x_range = x_range
    line = chart.line([x_range[0]], [0], color="#7b3294", width=3)
    marker = _Marker(chart, "#1b1b1b")
    plotter.add_chart(chart)

    dchart = pv.Chart2D(size=(0.42, 0.33), loc=(0.53, 0.65))
    dchart.background_color = (1.0, 1.0, 1.0, 0.55)
    dchart.border_color = (0.3, 0.3, 0.3)
    dchart.x_label = x_label
    dchart.y_label = "Capillary force"
    dchart.x_range = x_range
    dchart._line_x = dchart.line([x_range[0]], [0], color=_COL_X, width=2, label="Fx")
    dchart._line_y = dchart.line([x_range[0]], [0], color=_COL_Y, width=2, label="Fy")
    dline = dchart.line([x_range[0]], [0], color=_COL_Z, width=2, label="Fz")
    dchart.legend_visible = True
    _leg = dchart.GetLegend()
    _leg.GetBrush().SetColor(255, 255, 255, 0)
    dmarker = _Marker(dchart, _COL_Z)
    dmarker._aux = {"fx": _Marker(dchart, _COL_X), "fy": _Marker(dchart, _COL_Y)}
    plotter.add_chart(dchart)
    return chart, line, marker, dchart, dline, dmarker


def update_energy_chart(chart, line, xs, energies, marker=None, dchart=None, dline=None,
                         dmarker=None, forces=None):
    """Push the accumulated frames onto the two charts: `xs` (tip x-coordinate
    or iteration, one value per recorded frame), `energies` (surface free
    energy) and `forces` (force of the liquid on the sphere, (N, 3), solver.py
    `diag["force"]`). The x-ranges are fixed in make_energy_chart; only the
    y-ranges follow the data. The newest point of every curve gets a diamond
    marker. In quasi-static mode every frame is an equilibrium, so the
    values are plotted as they are (no averaging)."""
    line.update(xs, energies)
    y_min, y_max = min(energies), max(energies)
    pad = 0.05 * (y_max - y_min if y_max > y_min else max(abs(y_max), 1e-6))
    chart.y_range = [y_min - pad, y_max + pad]
    if marker is not None:
        marker.update([xs[-1]], [energies[-1]])

    if dchart is not None and forces is not None:
        fa = np.asarray(forces, dtype=float).reshape(-1, 3)
        dchart._line_x.update(xs, fa[:, 0])
        dchart._line_y.update(xs, fa[:, 1])
        dline.update(xs, fa[:, 2])
        lo, hi = float(fa.min()), float(fa.max())
        dpad = 0.05 * (hi - lo if hi > lo else max(abs(hi), 1e-6))
        dchart.y_range = [lo - dpad, hi + dpad]
        if dmarker is not None:
            dmarker.update([xs[-1]], [fa[-1, 2]])
            if "fx" in dmarker._aux:
                dmarker._aux["fx"].update([xs[-1]], [fa[-1, 0]])
                dmarker._aux["fy"].update([xs[-1]], [fa[-1, 1]])
