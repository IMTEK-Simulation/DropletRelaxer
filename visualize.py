"""
PyVista-based live visualization: shows the surface colored by mean
curvature, with a live-updating energy (area) chart overlaid at the
top of the window, and updates both while the relaxation is running
(see live_pyvista_demo.py for the original test).
"""

import math
import os

import numpy as np
import pyvista as pv
import vtk

from energy import mean_curvature

_SCALAR_BAR_TITLE = "mean curvature |H|"


def _gradient_dedup(x_arr, areas_arr):
    """np.gradient(areas, x) needs a strictly increasing `x` -- but a
    repeated x means a zero spacing, which makes np.gradient divide by
    zero and return inf/nan for that point, and a single nan/inf in the
    y-range silently blanks the ENTIRE chart's y-axis (no tick labels,
    no visible curve at all), not just that one point.

    The one caller of this (the sphere-path-length derivative -- see
    _sphere_path_length and update_energy_chart) can have `x` repeat for
    a LONG stretch of consecutive frames whenever the obstacle sphere
    simply isn't moving yet (e.g. every frame during
    ObstacleConfig.hold_atStart): those frames genuinely have no
    "per unit distance dragged" force to report, since no distance was
    covered between them. Computes the gradient on the de-duplicated,
    strictly-increasing subsequence (one point per DISTINCT x, keeping
    the first frame at each), then fills every ORIGINAL frame that is
    NOT itself the start of a new distinct x with exactly 0 rather than
    interpolating a neighboring transition's value across the whole
    stretch -- interpolating there used to broadcast one single (and
    often very large, since it then divides the area change accumulated
    over the WHOLE stationary stretch by the tiny first real step) value
    across every stationary frame, which is what made the chart's
    y-scale start pinned to a huge, physically-meaningless plateau
    instead of settling near 0 until genuine contact/dragging began."""
    x_arr = np.asarray(x_arr, dtype=float)
    areas_arr = np.asarray(areas_arr, dtype=float)
    n = len(x_arr)
    if n < 2:
        return np.zeros(n)
    # `x_arr` (the sphere's cumulative path length) is non-decreasing by
    # construction, so a run of equal consecutive values is a stretch
    # where the sphere didn't move between those recorded frames. Anchor
    # each distinct x value to its LAST occurrence, not its first: the
    # first occurrence of x=0, for instance, is the very start of the
    # run, while the last occurrence is the frame right before the
    # sphere starts moving -- using that one keeps the gradient's
    # boundary estimate a genuinely LOCAL one (one render_every step
    # away from where real motion begins) instead of spanning the
    # entire stationary stretch (during which the area can still change
    # a lot for reasons that have nothing to do with the sphere, e.g.
    # the droplet's own initial equilibration during
    # ObstacleConfig.hold_atStart) and dividing that by the tiny first
    # real step in x -- which is what used to produce a single very
    # large, physically-meaningless spike right where motion begins.
    keep_last = np.concatenate((np.diff(x_arr) > 0, [True]))
    if keep_last.sum() < 2:
        return np.zeros(n)
    x_u, areas_u = x_arr[keep_last], areas_arr[keep_last]
    deriv_u = np.gradient(areas_u, x_u)
    deriv = np.zeros(n)
    deriv[keep_last] = deriv_u
    return deriv


def _trailing_mean(values, window):
    """Trailing moving average: entry i is the mean of the last `window`
    entries up to and including i (fewer at the very start of the
    series, where fewer than `window` values exist yet). `window <= 1`
    returns the input unchanged. Used to smooth the capillary-force
    chart -- the raw -dE/ds is a finite difference of a relaxing
    (never perfectly converged) energy and therefore noisy from frame
    to frame; see VisConfig.force_average_window."""
    v = np.asarray(values, dtype=float)
    if window is None or window <= 1 or len(v) == 0:
        return v
    c = np.concatenate(([0.0], np.cumsum(v)))
    idx = np.arange(1, len(v) + 1)
    lo = np.maximum(idx - int(window), 0)
    return (c[idx] - c[lo]) / (idx - lo)


def _sphere_path_length(centers):
    """Cumulative Euclidean distance the obstacle sphere's center has
    traveled, one running total per entry of `centers` (a list of
    (x, y, z) tuples/arrays, or None for a frame with no obstacle/no
    recorded center -- e.g. ObstacleConfig.enabled=False, or the very
    first frame before any solver step has run). A None entry, or two
    identical consecutive centers (the sphere sitting still during
    ObstacleConfig.hold_atStart's hold period), simply adds 0 -- the
    running total only advances while the sphere is actually moving.
    Always non-decreasing by construction, which is exactly what
    _gradient_dedup needs as its "x" coordinate."""
    s = np.zeros(len(centers))
    last = None
    for i, c in enumerate(centers):
        if i > 0:
            s[i] = s[i - 1]
        if c is not None:
            if last is not None:
                s[i] += float(np.linalg.norm(np.asarray(c, dtype=float) - last))
            last = np.asarray(c, dtype=float)
    return s


def _capillary_force(centers, areas, window=1):
    """Vertical capillary force on the sphere, F_z = -dE/dz, per frame.

    The energy is differentiated against the sphere's cumulative PATH
    LENGTH s (always increasing, so it is well-defined even where z
    itself repeats or reverses -- during the hold and at the turning
    point). That gives dE/ds, the rate along the direction of motion. To
    turn it into the z-component it needs the sign of the motion: with
    the sphere going down dz/ds = -1, going up +1, so

        F_z = -dE/dz = -(dE/ds) * (dz/ds)

    (exact for a purely vertical path, the default; for other paths it
    is the z-part of the unit tangent times the force along the path).
    Positive = the liquid pushes the sphere up, negative = pulls it
    down. dz/ds is taken on the same de-duplicated subsequence as dE/ds
    (see _gradient_dedup) and is not meaningful at the single turning-point frame
    (path direction undefined; the value there comes out near 0).

    Caveat: E differences between recorded frames also contain the
    droplet's own relaxation (it is not in equilibrium at every frame),
    so this is only approximately a force; `window` = trailing moving
    average over that many frames (VisConfig.force_average_window)."""
    areas = np.asarray(areas, dtype=float)
    if centers is None:
        return np.zeros(len(areas))
    s = _sphere_path_length(centers)
    z = _center_z(centers)
    dE_ds = _gradient_dedup(s, areas)
    dz_ds = np.zeros(len(s))
    keep_last = np.concatenate((np.diff(s) > 0, [True]))
    if keep_last.sum() >= 2:
        dz_ds[keep_last] = np.gradient(z[keep_last], s[keep_last])
    return _trailing_mean(-dE_ds * dz_ds, window)


def tris_to_pyvista_faces(tris):
    n = tris.shape[0]
    faces = np.hstack([np.full((n, 1), 3), tris]).astype(np.int64)
    return faces.ravel()


def _add_droplet_mesh(plotter, verts, tris, boundary_idx, vis_cfg):
    faces = tris_to_pyvista_faces(tris)
    mesh = pv.PolyData(verts.copy(), faces)
    mesh["curvature"] = mean_curvature(verts, tris, boundary_idx)
    # See VisConfig.edge_visible_max_triangles: past a certain triangle
    # count the black wireframe can cover nearly every pixel and the
    # surface reads as solid black -- auto-hide it once that dense.
    show_edges = vis_cfg.show_edges and len(tris) <= vis_cfg.edge_visible_max_triangles
    actor = plotter.add_mesh(
        mesh,
        scalars="curvature",
        cmap=vis_cfg.colormap,
        clim=list(vis_cfg.curvature_clim) if vis_cfg.curvature_clim else None,
        show_edges=show_edges,
        edge_color="black",
        line_width=0.3,
        nan_color="lightgray",  # the flat substrate base (no curvature) -- see substrate.py
        scalar_bar_args={
            "title": _SCALAR_BAR_TITLE,
            # Vertical, docked to the right edge, but kept clear of the
            # chart overlays above it (the energy chart occupies y in
            # roughly [0.70, 0.98], and the derivative chart right below
            # it [0.57, 0.69] -- see make_energy_chart's loc/size) by
            # capping its own height well below both.
            "vertical": True,
            "position_x": 0.88,
            "position_y": 0.05,
            "width": 0.08,
            "height": 0.50,
        },
    )
    return mesh, actor


def make_plotter(verts0, tris, boundary_idx, vis_cfg):
    # Disable multisampling (MSAA) BEFORE the render window is created.
    # Some GLX setups -- notably a multi-monitor desktop combined into a
    # single X11 screen at an unusual resolution -- offer no multisample
    # FBConfig at all, and VTK's GLX config search then fails outright
    # with "Could not find a decent config"/"...visual" (it doesn't fall
    # back to a non-multisample config on its own), which is followed by
    # a crash instead of a clean error. This is a one-line, harmless fix
    # (only loses anti-aliasing on the triangle edges) -- see README
    # Troubleshooting, "Could not find a decent config" despite a working
    # GLX driver.
    vtk.vtkOpenGLRenderWindow.SetGlobalMaximumNumberOfMultiSamples(0)
    plotter = pv.Plotter(off_screen=vis_cfg.off_screen, window_size=vis_cfg.window_size)
    mesh, actor = _add_droplet_mesh(plotter, verts0, tris, boundary_idx, vis_cfg)

    # The eye/focal-point coordinates below were tuned for the ORIGINAL
    # dimensionless scale (droplet radius ~1, substrate_size 3.5). They
    # are absolute world coordinates, so at a very different physical
    # scale (see MeshConfig.target_volume) they'd leave the camera
    # sitting far too close to -- or even inside -- the (now much
    # bigger) droplet: an extreme close-up on a couple of triangles,
    # which is what "the whole droplet renders solid black" usually
    # turns out to be (the triangle edges fill the entire close-up view)
    # rather than genuinely too many/too-small triangles. Scale the
    # whole camera rig against substrate_size, the same reference used
    # for the axes tripod and its labels above.
    cam_scale = vis_cfg.substrate_size / 3.5
    plotter.camera_position = [
        (3 * cam_scale, -3 * cam_scale, 3 * cam_scale),
        (0, 0, 1 * cam_scale),
        (0, 0, 1),
    ]
    # add_axes() (the default) draws a small orientation-only triad as a
    # fixed screen-corner widget, not actually placed in the 3D scene.
    # add_axes_at_origin() instead adds a real vtkAxesActor IN the scene
    # -- placed here at a CORNER of the substrate grid (see
    # make_substrate_mesh, which spans +-substrate_size/2 in x and y,
    # centered at the world origin) rather than at the origin itself, so
    # it sits clear of the droplet instead of right underneath it. The
    # corner is picked to sit roughly under the default camera position
    # above -- (3, -3, 3) looks from +x/-y -- so it stays in view without
    # extra rotating.
    axes_actor = plotter.add_axes_at_origin()
    # IMPORTANT: vtkAxesActor silently ignores plain SetPosition() --
    # confirmed by direct testing (GetPosition() reports the new value,
    # but the rendered pixels don't change at all; a previous version of
    # this code used SetPosition and never actually moved anything,
    # which is why the tripod kept appearing to sit inside/behind the
    # droplet regardless of which corner was picked). SetUserTransform()
    # with a plain translation does move it.
    #
    # The actual GRID corner (+-0.5*substrate_size in x and y) is also
    # NOT usable here: with this camera, world-to-display projection
    # puts it off the edge of the window for 3 of the 4 corners, and
    # ON TOP of the droplet's own screen position for the 4th (checked
    # numerically -- at the default scale, (-0.5, +0.5)*substrate_size
    # projects to display (400, 431) in an 800x800 window, i.e. dead
    # center) -- the frustum this camera rig uses is considerably
    # tighter than the substrate grid it's framing.
    #
    # (-0.25, -0.25) (the fully symmetric corner) was the first position
    # checked and used for a while, but there isn't much further room
    # along that exact diagonal: (-0.30, -0.30) already projects
    # off-screen (checked numerically). The window's aspect and this
    # camera's specific framing leave much more slack along x than
    # along y at this same camera angle, though (checked numerically by
    # scanning both axes independently) -- so moving further out means
    # moving mostly along -x with only a small additional -y offset,
    # not further along the symmetric diagonal. (-0.40, -0.10) was
    # checked the same way as the original position -- both on the
    # initial cone and again on a heavily sphere-pressed, spread-out
    # mid-run shape -- to land clearly on-screen (with margin) and
    # clear of the droplet's own on-screen footprint at every stage,
    # while sitting noticeably further from the droplet than
    # (-0.25, -0.25) did (Euclidean distance from the origin ~1.17x
    # larger).
    fx, fy = -0.40 * vis_cfg.substrate_size, -0.10 * vis_cfg.substrate_size
    axes_transform = vtk.vtkTransform()
    ox, oy, oz = getattr(vis_cfg, "axes_offset", (0.0, 0.0, 0.0))   # VisConfig.axes_offset
    axes_transform.Translate(fx + ox * vis_cfg.substrate_size, fy + oy * vis_cfg.substrate_size,
                             oz * vis_cfg.substrate_size)
    axes_actor.SetUserTransform(axes_transform)
    # Its default arm length (1, 1, 1) is an absolute world-space size,
    # not relative to the scene -- on a very different physical scale
    # (see MeshConfig.target_volume) it would be either invisibly tiny
    # or absurdly huge. Scale it against the substrate grid instead
    # (deliberately modest -- a corner marker, not a scene-dominating
    # prop competing with the droplet itself).
    arm = 0.12 * vis_cfg.substrate_size
    axes_actor.SetTotalLength(arm, arm, arm)
    # Shrink the x/y/z label size to a quarter of vtkAxesActor's
    # default (halved once, then halved again) -- its captions auto-size
    # their text to fill a 2D box (in normalized viewport units) rather
    # than exposing a direct font-size setter that reliably sticks, so
    # shrink that box instead of the text property.
    for _axis in ("X", "Y", "Z"):
        _cap = getattr(axes_actor, f"Get{_axis}AxisCaptionActor2D")()
        _cap.SetWidth(_cap.GetWidth() * 0.25)
        _cap.SetHeight(_cap.GetHeight() * 0.25)

    if vis_cfg.save_gif:
        plotter.open_gif(vis_cfg.save_gif)

    # Live volume readout: the volume penalty (Eq. penalty in the math
    # document) only ever pulls TOWARDS the target volume, it doesn't
    # enforce it exactly (see solver.py/SolverConfig.volume_penalty),
    # so the actual current volume is worth watching directly rather
    # than trusting it "must" be constant. A CornerAnnotation (cheap:
    # `set_text` just replaces the string, no new actor/re-layout) in
    # the bottom-right corner, clear of the slider (which only spans
    # x in [0.22, 0.78]) and the Play button (bottom-LEFT) below it.
    # The three readouts (COM, COM xy-offset, volume) sit together at the
    # bottom left, stacked above the Play/Save buttons (pixel positions).
    volume_text = plotter.add_text("V = --", position=(10, 48),
                                    font_size=10, color="black")

    # Centre of mass of the liquid, bottom-left (a pixel position slightly
    # above the Play/Save buttons of the history slider, which sit at the very
    # bottom left, so the two never overlap).
    plotter._com_text = plotter.add_text("COM = --", position=(10, 86),
                                         font_size=10, color="black")
    plotter._com_dev_text = plotter.add_text("COM offset to sphere (xy) = --", position=(10, 67),
                                             font_size=10, color="black")

    return plotter, mesh, actor, volume_text


def update_volume_text(volume_text, V, V0):
    """Refreshes the bottom-left volume readout (third line of the info block) (see make_plotter) with
    the CURRENT volume and its deviation from the target -- both as an
    absolute number and as a percentage, since a "small" absolute
    deviation can still be a large fraction of a small target volume or
    vice versa. Deliberately shown every render, not just at the end,
    so a deviation that creeps up over the course of a run (e.g. while
    the sphere obstacle is actively pressing into the surface, see
    "Sphere obstacle" in the README) is visible as it happens rather
    than only in the final printout."""
    dev = V - V0
    pct = 100.0 * dev / V0 if V0 != 0 else 0.0
    volume_text.SetInput(f"V = {V:.5g}  (target {V0:.5g}, {dev:+.4g} / {pct:+.3f}%)")


def com_xy_offset_to_sphere(com, sphere_center=None):
    """Offset of the centre of mass from the sphere centre in the xy plane,
    returned as the vector (dx, dy) = COM_xy - sphere_xy. Without a sphere
    (None) the reference is the axis x = y = 0."""
    c = np.asarray(com, dtype=float)
    ref = np.zeros(2) if sphere_center is None else np.asarray(sphere_center, dtype=float)[:2]
    return c[:2] - ref


def com_xy_deviation(com):
    """Distance of the centre of mass from the centred position in the xy
    plane, sqrt(x^2 + y^2): the centred position is the axis x = y = 0 (the
    start droplet's centre, also the default sphere axis). Accepts one point
    (3,) or an array (N, 3)."""
    c = np.asarray(com, dtype=float)
    return np.hypot(c[..., 0], c[..., 1])


def update_com_text(plotter, com, sphere_center=None):
    """Refreshes the bottom-left centre-of-mass readout (see make_plotter)."""
    txt = getattr(plotter, "_com_text", None)
    if txt is not None and com is not None:
        txt.SetInput(f"COM = ({com[0]:+.3f}, {com[1]:+.3f}, {com[2]:.3f})")
        dev = getattr(plotter, "_com_dev_text", None)
        if dev is not None:
            d = com_xy_offset_to_sphere(com, sphere_center)
            dev.SetInput(f"COM offset to sphere (xy) = {np.hypot(*d):.4g}  (dx {d[0]:+.3g}, dy {d[1]:+.3g})")


def update_mesh(mesh, verts, tris, boundary_idx, curvature=None):
    """`curvature`, when given, is used as-is instead of recomputing it
    from `verts`/`tris` -- an optional fast path for the history
    slider/Play button (see make_history_slider), which replays frames
    whose curvature was already computed once during the live run
    (main.py caches it into `history`); recomputing the identical
    values on every scrub/Play tick was pure waste and, at a few
    thousand triangles, was itself a measurable share of what made
    Play feel sluggish. A live run always passes None here (its verts
    are new every frame, so there is nothing to reuse)."""
    mesh.points = verts
    mesh["curvature"] = curvature if curvature is not None else mean_curvature(verts, tris, boundary_idx)


def replace_mesh(plotter, actor, verts, tris, boundary_idx, vis_cfg):
    """Swaps in a fresh PolyData with a DIFFERENT vertex/triangle count
    (used after mesh.py:subdivide_mesh remeshes the droplet) -- unlike
    update_mesh, `mesh.points = ...` only works in place when the point
    count is unchanged, so a differently-sized mesh needs a new actor
    instead. Removes the old actor and its scalar bar, then adds a
    fresh one with the same appearance settings. Returns (new_mesh,
    new_actor); replace your stored references with these."""
    plotter.remove_actor(actor, render=False)
    try:
        plotter.remove_scalar_bar(_SCALAR_BAR_TITLE)
    except (KeyError, ValueError):
        pass  # no scalar bar was tracked under this title -- fine
    return _add_droplet_mesh(plotter, verts, tris, boundary_idx, vis_cfg)


def make_substrate_mesh(plotter, size, resolution):
    """A translucent grid at z=0 representing the substrate: mostly so
    the droplet's motion (e.g. while being dragged by the sphere
    obstacle) is visible against a fixed reference frame, rather than
    only via the small axes triad in the corner. Purely decorative --
    never updated, since the substrate itself doesn't move."""
    # Sits a hair BELOW z=0 so it does not z-fight with the droplet's flat
    # base mesh (which lies exactly on z=0 -- see substrate.py).
    plane = pv.Plane(center=(0.0, 0.0, -1e-3 * size), direction=(0.0, 0.0, 1.0),
                      i_size=size, j_size=size,
                      i_resolution=resolution, j_resolution=resolution)
    plotter.add_mesh(plane, style="wireframe", color="gray", opacity=0.35, line_width=1)
    plotter.add_mesh(plane, style="surface", color="lightgray", opacity=0.06,
                      lighting=False)
    return plane


def make_anomaly_disc(plotter, anomaly_cfg, wall_angle_deg, size):
    """Translucent disc (+ outline) on the substrate marking the circular
    wettability anomaly (anomaly.py): blue = more hydrophilic than the
    surrounding wall, violet = more hydrophobic. Sits a hair ABOVE z = 0 so it
    is visible through the (translucent) droplet and over its flat base."""
    cx, cy = anomaly_cfg.center_xy
    r = anomaly_cfg.radius
    z = 2e-3 * size
    color = "deepskyblue" if anomaly_cfg.contact_angle_deg < wall_angle_deg else "mediumorchid"
    disc = pv.Disc(center=(cx, cy, z), inner=0.0, outer=r, normal=(0, 0, 1), r_res=1, c_res=96)
    plotter.add_mesh(disc, color=color, opacity=0.35, lighting=False)
    t = np.linspace(0.0, 2 * np.pi, 97)
    ring = pv.lines_from_points(np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t),
                                                 np.full_like(t, z)]))
    plotter.add_mesh(ring, color=color, line_width=2, lighting=False)
    return disc


def make_roughness_surface(plotter, field, size, opacity=0.75, cmap="coolwarm", resolution=240):
    """Colours the substrate by the random wall of roughness.py: the local
    contact angle theta_eff = arccos(kappa_eff(x, y)) in degrees, kappa_eff =
    kappa_Y + delta_kappa (the field the contact line actually feels, incl.
    clipping to [-1, 1]). A plane just above z = 0 (below the anomaly disc),
    side `size` (limited to the field box), with a VERTICAL colorbar on the
    LEFT edge of the window. Returns the plane mesh."""
    half = 0.5 * min(float(size), field.L) * 0.999
    plane = pv.Plane(center=(0.0, 0.0, 1.5e-3 * size), direction=(0.0, 0.0, 1.0),
                     i_size=2 * half, j_size=2 * half,
                     i_resolution=resolution, j_resolution=resolution)
    p = plane.points
    kap = np.clip(field.kappa0 + field.delta_kappa(p[:, 0], p[:, 1]), -1.0, 1.0)
    theta = np.degrees(np.arccos(kap))
    plane.point_data["theta_eff"] = theta
    th0 = math.degrees(math.acos(np.clip(field.kappa0, -1.0, 1.0)))
    dev = max(float(np.abs(theta - th0).max()), 1e-6)
    plotter.add_mesh(
        plane, scalars="theta_eff", cmap=cmap, clim=[th0 - dev, th0 + dev],
        opacity=opacity, lighting=False, show_edges=False, nan_color="lightgray",
        scalar_bar_args={
            "title": "theta_eff\n[deg]",
            "vertical": True,
            "position_x": 0.015,
            "position_y": 0.17,
            "width": 0.06,
            "height": 0.40,      # like the right colorbar; above the info text, below the charts
            "fmt": "%.0f",
            "title_font_size": 12,
            "label_font_size": 12,
        },
    )
    return plane


class ProgressBar:
    """Horizontal progress bar in the window, at the place where the history
    slider appears after the run: grey bar that fills green from left to right
    and shows "Progress 37 %  |  ETA ..." (text set by the caller). Built from
    2D actors in normalized viewport coordinates; `update(frac, text)` changes
    the green part and the text, `hide()` removes it (before the slider is
    created)."""

    X0, X1, Y0, Y1 = 0.22, 0.78, 0.026, 0.058

    def __init__(self, plotter):
        self.plotter = plotter
        self.bg = self._rect(self.X0, self.X1, (0.55, 0.55, 0.55))
        self.fg = self._rect(self.X0, self.X0, (0.18, 0.72, 0.25))
        self.txt = vtk.vtkTextActor()
        self.txt.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.txt.SetPosition(0.5 * (self.X0 + self.X1), 0.5 * (self.Y0 + self.Y1))
        tp = self.txt.GetTextProperty()
        tp.SetJustificationToCentered()
        tp.SetVerticalJustificationToCentered()
        tp.SetFontSize(14)
        tp.SetColor(0.0, 0.0, 0.0)
        tp.BoldOn()
        for a in (self.bg[0], self.fg[0], self.txt):
            plotter.add_actor(a, reset_camera=False, render=False)
        self.update(0.0, "Progress 0 %")

    def _rect(self, x0, x1, color):
        pts = vtk.vtkPoints()
        for p in ((x0, self.Y0), (x1, self.Y0), (x1, self.Y1), (x0, self.Y1)):
            pts.InsertNextPoint(p[0], p[1], 0.0)
        quad = vtk.vtkCellArray()
        quad.InsertNextCell(4)
        for k in range(4):
            quad.InsertCellPoint(k)
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetPolys(quad)
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToNormalizedViewport()
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(poly)
        mapper.SetTransformCoordinate(coord)
        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        return actor, pts, poly

    def update(self, frac, text):
        frac = min(max(float(frac), 0.0), 1.0)
        x1 = self.X0 + frac * (self.X1 - self.X0)
        _, pts, poly = self.fg
        pts.SetPoint(1, x1, self.Y0, 0.0)
        pts.SetPoint(2, x1, self.Y1, 0.0)
        pts.Modified()
        poly.Modified()
        self.txt.SetInput(text)

    def render(self):
        if getattr(self.plotter, "off_screen", False):
            return
        try:
            self.plotter.update()
        except Exception:
            pass

    def hide(self):
        for a in (self.bg[0], self.fg[0], self.txt):
            self.plotter.remove_actor(a, render=False)


class _RightDragPanStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Right-click-drag pans (translates) the camera instead of
    zooming it (the vtkInteractorStyleTrackballCamera default). Left-
    click-drag still rotates and the scroll wheel still zooms -- only
    the right button's action changes. Handy since the default pan
    button (middle-click) has no convenient equivalent on most
    trackpads.

    Implemented by overriding OnRightButtonDown/OnRightButtonUp
    directly. An earlier version used AddObserver("RightButtonPressEvent",
    ...) instead, on the theory that a direct override wasn't reliably
    dispatched -- that was wrong and was the actual bug: VTK's own
    vtkInteractorStyleTrackballCamera.OnRightButtonDown() override does
    NOT invoke the "RightButtonPressEvent" VTK event at all (only the
    common base class's default implementation does, which the
    trackball-camera subclass replaces rather than calls), so that
    observer silently never fired and every right-click fell straight
    through to the default Dolly/zoom -- confirmed by directly calling
    OnRightButtonDown() on both versions and checking GetState()
    (VTKIS_DOLLY vs. the intended VTKIS_PAN). The direct override below
    does not call the inherited OnRightButtonDown()/Up() (doing so
    would also run the default StartDolly()/EndDolly(), fighting our
    pan) and is the standard, verified-working pattern."""

    def OnRightButtonDown(self):
        # Same preamble as VTK's own OnRightButtonDown: the renderer under
        # the cursor must be made the current one, otherwise Pan() has no
        # camera to move and silently does nothing (this was the "right
        # drag no longer works" bug once the window contained several
        # renderers/charts).
        x, y = self.GetInteractor().GetEventPosition()
        self.FindPokedRenderer(x, y)
        if self.GetCurrentRenderer() is None:
            return
        self.StartPan()

    def OnRightButtonUp(self):
        self.EndPan()


def enable_right_button_pan(plotter):
    """Call once, after the interactive window is open (i.e. after
    plotter.show(...)), to remap the right mouse button as described
    in _RightDragPanStyle.

    Sets it through PyVista's OWN tracked `plotter.iren.style` property
    rather than calling `.interactor.SetInteractorStyle(...)` directly.
    The latter looks equivalent and works at first, but every
    `plotter.show()` call -- including the SECOND one after the history
    slider is added at the end of the run -- internally calls
    `iren.update_style()`, which reapplies whatever PyVista itself has
    tracked as the current style, silently reverting a style that was
    only ever set on the raw interactor. Setting `iren.style` instead
    makes PyVista track OUR style, so it survives every later
    `update_style()`/`show()` call.

    That alone turned out not to be enough in practice: right-drag pan
    was reported broken again, both mid-run and after the slider
    appears. PyVista has MORE than one path that can reassign the
    interactor's style out from under a plain one-time assignment --
    e.g. double-clicking a chart (the energy chart docked at the top of
    this window covers a good chunk of the screen) toggles a
    Context/chart-interaction style via `_set_context_style`, and other
    widgets can have similar effects -- and chasing down every possible
    trigger individually is fragile. Instead this installs a self-
    healing 'RenderEvent' observer that re-applies our style on every
    single render if something else has swapped it out, which is cheap
    (one identity check per render) and correct regardless of WHAT
    caused the swap."""
    style = getattr(plotter, "_right_drag_pan_style", None)
    if style is None:
        style = _RightDragPanStyle()
        plotter._right_drag_pan_style = style

        def _reassert_style(*_args):
            if plotter.iren.interactor.GetInteractorStyle() is not style:
                plotter.iren.style = style

        plotter.iren.add_observer("RenderEvent", _reassert_style)

        # The Python overrides OnRightButtonDown/Up above are NOT called by
        # VTK's C++ event dispatch (measured: after a real right-button press
        # the style was in VTKIS_DOLLY, not PAN -- the C++ implementation of
        # vtkInteractorStyleTrackballCamera runs instead). So the press/release
        # are intercepted on the interactor with a higher priority than the
        # style's own handler, start/stop the pan there, and abort the event so
        # the default dolly never starts.
        inter = plotter.iren.interactor

        def _make(handler):
            holder = {}

            def cb(_obj, _event):
                if handler():
                    holder["cmd"].SetAbortFlag(1)
            tag = inter.AddObserver(handler.event, cb, 10.0)
            holder["cmd"] = inter.GetCommand(tag)
            return tag

        def _press():
            cur = plotter.iren.interactor.GetInteractorStyle()
            x, y = inter.GetEventPosition()
            cur.FindPokedRenderer(x, y)
            if cur.GetCurrentRenderer() is None:
                return False
            cur.StartPan()
            return True

        def _release():
            cur = plotter.iren.interactor.GetInteractorStyle()
            if cur.GetState() == 2:  # VTKIS_PAN
                cur.EndPan()
                return True
            return False

        _press.event = "RightButtonPressEvent"
        _release.event = "RightButtonReleaseEvent"
        plotter._right_drag_tags = (_make(_press), _make(_release))

    plotter.iren.style = style


def make_obstacle_mesh(plotter, obstacle_cfg):
    """Adds the translucent sphere obstacle to the scene, built at the
    origin so it can be cheaply re-centered every frame by translating
    its points (see update_obstacle_mesh). Returns the PyVista mesh, or
    None if there is no obstacle to show."""
    if obstacle_cfg is None or not obstacle_cfg.enabled:
        return None

    sphere = pv.Sphere(radius=obstacle_cfg.radius, center=(0.0, 0.0, 0.0),
                        theta_resolution=24, phi_resolution=24)
    plotter.add_mesh(sphere, color="steelblue", opacity=0.45, smooth_shading=True,
                      specular=0.6, specular_power=20)
    return sphere


def update_wetting_outline(plotter, current_actor, loop_points):
    """Draws (or redraws) the dynamically-tracked three-phase contact
    line on the sphere (see wetting.py) as a closed bright polyline --
    the visual answer to "is there now a real contact line": whenever
    a wetted patch currently exists, its boundary loop is drawn
    explicitly as its own object, separate from the curvature
    coloring. `loop_points` is the current loop's vertex positions in
    cyclic order, or None when there is currently no contact patch.

    The number of loop vertices changes every frame as the patch grows
    or shrinks, so (like replace_mesh for the main surface) this
    can't just reassign points on an existing PolyData -- the previous
    frame's line actor is removed and a fresh one added. Returns the
    new actor (or None); callers should store it and pass it back in
    as `current_actor` next time."""
    if current_actor is not None:
        plotter.remove_actor(current_actor, render=False)
    if loop_points is None or len(loop_points) < 3:
        return None
    closed = np.vstack([loop_points, loop_points[:1]])
    line_mesh = pv.lines_from_points(closed)
    return plotter.add_mesh(line_mesh, color="lime", line_width=5,
                             render_lines_as_tubes=True, lighting=False, render=False)


def update_ring_outline(plotter, current_actor, ring_points):
    """Draws (or redraws) the substrate contact line -- the ring where
    the droplet meets the flat wall -- as a closed cyan polyline, the
    counterpart of the lime three-phase line on the sphere
    (update_wetting_outline). `ring_points` are the ring vertices'
    positions in cyclic order (`verts[boundary_idx]`). Like the sphere's
    line it is removed and re-added each call (the ring gets more
    vertices whenever a remesh splits a ring edge). Returns the new
    actor; pass it back in as `current_actor` next time."""
    if current_actor is not None:
        plotter.remove_actor(current_actor, render=False)
    if ring_points is None or len(ring_points) < 3:
        return None
    closed = np.vstack([ring_points, ring_points[:1]])
    line_mesh = pv.lines_from_points(closed)
    return plotter.add_mesh(line_mesh, color="cyan", line_width=5,
                             render_lines_as_tubes=True, lighting=False, render=False)


def update_obstacle_mesh(sphere_mesh, center, base_points):
    """Moves the sphere obstacle to `center` by translating its
    (cached, origin-centered) `base_points`. No-op if there is no
    obstacle or `center` is None (obstacle disabled)."""
    if sphere_mesh is None or center is None:
        return
    sphere_mesh.points = base_points + np.asarray(center)


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


_GAP_OFFSET = 0.0


def set_gap_offset(radius):
    """The "Gap" chart shows z_center - radius, i.e. the height of the sphere's
    lowest point above the substrate (main.py passes ObstacleConfig.radius).
    Forces/derivatives are unaffected by this constant offset."""
    global _GAP_OFFSET
    _GAP_OFFSET = float(radius)


def _center_z(centers):
    """Gap per frame = z of the sphere center minus the sphere radius (see
    set_gap_offset); frames without a recorded
    center (None) take the nearest known value (first known one is
    back-filled), so the curve has no gaps."""
    z = [None if c is None else float(c[2]) - _GAP_OFFSET for c in centers]
    known = [v for v in z if v is not None]
    if not known:
        return np.zeros(len(z))
    last = known[0]
    for i, v in enumerate(z):
        if v is None:
            z[i] = last
        else:
            last = v
    return np.array(z)


def _padded_range(a, min_span=2.0, pad_frac=0.05):
    """Axis range for a curve: its min/max plus 5 % padding, but at least
    `min_span` wide (a constant curve, e.g. the gap of a purely lateral
    path, would otherwise get a range of ~1e-6 and unreadable tick labels)."""
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi - lo < min_span:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5 * min_span, mid + 0.5 * min_span
    pad = pad_frac * (hi - lo)
    return [lo - pad, hi + pad]


def _center_x(centers):
    """Sphere-centre x per frame ("tip x-coordinate"); None frames take the
    nearest known value, as in _center_z."""
    x = [None if c is None else float(c[0]) for c in centers]
    known = [t for t in x if t is not None]
    if not known:
        return np.zeros(len(x))
    last = known[0]
    for i, t in enumerate(x):
        if t is None:
            x[i] = last
        else:
            last = t
    return np.array(x)


def make_history_slider(plotter, history, topologies, mesh, actor, obstacle_mesh,
                         obstacle_base_points, vis_cfg, initial_wetting_actor=None,
                         marker=None, dmarker=None, volume_text=None, V0=None,
                         initial_ring_actor=None, save_callback=None, x_label="Tip x-coordinate"):
    """Adds a slider widget -- call this once, after the relaxation has
    finished -- that scrubs through the recorded frames, so you can look
    at any frame again instead of only the final shape. Also
    adds a Play button next to it (see the bottom of this function) that
    steps through the same recorded frames on a timer, for a fairly
    smooth "replay" instead of manual dragging.

    `history` is a list of dicts (see main.py), one per recorded frame,
    each with keys "x"/"iteration"/"verts"/"epoch"/"sphere_center"/....
    In quasi-static mode a frame is an equilibrated sphere step. The slider
    value is the frame's "x": the TIP x-coordinate (sphere centre x), or the
    iteration number in the old iteration-driven mode; it snaps to the nearest
    recorded frame. If x does not vary (a path without x motion) the
    iteration number is used instead.
    `topologies` is the list of (tris, boundary_idx) pairs indexed by
    "epoch" -- a new entry is appended each time the mesh gets
    remeshed, since a remesh changes the vertex/triangle count and
    PyVista can't reassign `.points` in place across a mismatched count
    (see replace_mesh); scrubbing across such a boundary therefore
    swaps in a new actor exactly like a live remesh event does."""
    x_arr = np.array([f["x"] for f in history], dtype=float)
    if np.ptp(x_arr) < 1e-9:
        x_arr = np.array([f["iteration"] for f in history], dtype=float)
        x_label = "Iteration"
    f_all = np.array([f.get("force", np.zeros(3)) for f in history], dtype=float)
    fz_arr, fx_arr, fy_arr = f_all[:, 2], f_all[:, 0], f_all[:, 1]
    # `initial_wetting_actor`: the contact-line actor (if any) still in the
    # scene from the live run that just finished. Without it, this state
    # dict starts believing no such actor exists, so the FIRST time the
    # slider is dragged to a frame with no (or a different) contact patch,
    # update_wetting_outline() has nothing to remove -- the live run's
    # final contact line then lingers in the scene forever alongside
    # whatever the slider draws next.
    state = {"mesh": mesh, "actor": actor, "epoch": history[-1]["epoch"],
             "wetting_actor": initial_wetting_actor, "ring_actor": initial_ring_actor,
             "idx": len(history) - 1,
             "playing": False}

    def callback(value):
        idx = int(np.argmin(np.abs(x_arr - value)))   # nearest recorded frame
        frame = history[idx]
        tris, boundary_idx = topologies[frame["epoch"]]

        if frame["epoch"] != state["epoch"]:
            # A remesh boundary: needs a full new actor either way (see
            # replace_mesh's docstring), so this doesn't go through the
            # cached-curvature fast path below -- it's the rare case
            # (a handful of times per run), not the one that made Play
            # sluggish.
            new_mesh, new_actor = replace_mesh(plotter, state["actor"], frame["verts"],
                                                tris, boundary_idx, vis_cfg)
            state["mesh"], state["actor"], state["epoch"] = new_mesh, new_actor, frame["epoch"]
        else:
            # Reuse the curvature already computed for this exact frame
            # during the live run (main.py caches it into `history`)
            # instead of recomputing mean_curvature from scratch on
            # every single scrub/Play tick -- this is the fast path
            # that actually matters for Play responsiveness.
            update_mesh(state["mesh"], frame["verts"], tris, boundary_idx,
                        curvature=frame.get("curvature"))

        update_obstacle_mesh(obstacle_mesh, frame["sphere_center"], obstacle_base_points)
        wetted_loop = frame.get("wetted_loop")
        loop_points = frame["verts"][wetted_loop] if wetted_loop is not None else None
        state["wetting_actor"] = update_wetting_outline(plotter, state["wetting_actor"], loop_points)
        state["ring_actor"] = update_ring_outline(plotter, state["ring_actor"],
                                                  frame["verts"][boundary_idx])
        xv = frame["x"]
        if marker is not None and "area" in frame:
            # Move the diamond marker to the (x, energy) point of whatever
            # frame is now shown -- the curves themselves stay the full,
            # finished run; only the markers move.
            marker.update([xv], [frame["area"]])
        if dmarker is not None:
            dmarker.update([xv], [fz_arr[idx]])
            if "fx" in dmarker._aux:
                dmarker._aux["fx"].update([xv], [fx_arr[idx]])
                dmarker._aux["fy"].update([xv], [fy_arr[idx]])
        if volume_text is not None and "volume" in frame:
            # Same idea as the diamond markers above: the volume readout
            # should reflect whatever frame is CURRENTLY shown while
            # scrubbing/Play, not just the live run's latest value.
            update_volume_text(volume_text, frame["volume"], V0)
        if "com" in frame:
            update_com_text(plotter, frame["com"], frame.get("sphere_center"))
        # Keep the play cursor in sync with whatever frame is now shown,
        # whether we got here from a manual drag or from _advance_play
        # below, so pressing Play always resumes from HERE rather than
        # wherever it last left off (and a manual drag while playing
        # doesn't get immediately overridden by the next timer tick).
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

    # Play button: steps through the recorded history on a timer instead
    # of manual dragging. Implemented as a single long-lived repeating
    # timer (started once, right here) that just no-ops on every tick
    # while paused, rather than creating/destroying a VTK timer on every
    # Play/Pause click -- PyVista's timer wrapper doesn't expose a clean
    # "cancel" for that, and gating a cheap identity/flag check per tick
    # is simpler and just as responsive.
    # See VisConfig.playback_max_frames: advance by more than one
    # recorded frame per tick when there are a lot of them, so one full
    # playback pass does a bounded amount of work (each played frame can
    # cost a real curvature recompute and, at a remesh boundary, a full
    # mesh/actor rebuild) instead of scaling with however densely the
    # run happened to be recorded.
    play_stride = max(1, len(history) // max(vis_cfg.playback_max_frames, 1))

    def _advance_play(_step):
        if not state["playing"]:
            return
        idx = min(state["idx"] + play_stride, len(history) - 1)
        if idx <= state["idx"]:
            # Reached the end: stop and visually untoggle the button
            # rather than leaving it showing "playing" with nothing left
            # to advance.
            state["playing"] = False
            play_button.GetRepresentation().SetState(0)
            plotter.render()
            return
        value = float(x_arr[idx])
        slider.GetRepresentation().SetValue(value)
        callback(value)

    plotter.iren.add_timer_event(max_steps=10_000_000, duration=vis_cfg.playback_interval_ms,
                                  callback=_advance_play)

    def _record_video(path):
        """Renders the recorded frames one after another into a video file
        (the window content incl. charts and the info block), then restores
        the frame that was shown. Returns the written file path."""
        import imageio
        state["playing"] = False
        play_button.GetRepresentation().SetState(0)
        stride = max(1, -(-len(history) // max(int(vis_cfg.video_max_frames), 1)))
        idxs = list(range(0, len(history), stride))
        if idxs[-1] != len(history) - 1:
            idxs.append(len(history) - 1)
        restore = state["idx"]
        try:
            writer = imageio.get_writer(path, fps=vis_cfg.video_fps, codec="libx264",
                                        quality=8, macro_block_size=2)
        except Exception as err:
            path = os.path.splitext(path)[0] + ".gif"
            print(f"No ffmpeg video writer ({err}); writing {os.path.basename(path)} instead")
            writer = imageio.get_writer(path, fps=vis_cfg.video_fps)
        try:
            for n, k in enumerate(idxs):
                value = float(x_arr[k])
                slider.GetRepresentation().SetValue(value)
                callback(value)
                plotter.render()
                writer.append_data(plotter.screenshot(return_img=True))
                if n % max(1, len(idxs) // 10) == 0:
                    print(f"  video: {100 * n // len(idxs)} %")
        finally:
            writer.close()
            value = float(x_arr[restore])
            slider.GetRepresentation().SetValue(value)
            callback(value)
        return path

    def _toggle_play(is_on):
        if is_on:
            # Replaying after having already reached the end: restart
            # from the beginning instead of doing nothing (idx is
            # already the last frame, so +1 would immediately stop).
            if state["idx"] >= len(history) - 1:
                state["idx"] = -1
            state["playing"] = True
        else:
            state["playing"] = False

    play_button = plotter.add_checkbox_button_widget(
        _toggle_play, value=False, position=(10.0, 10.0), size=28,
        color_on="#2ca02c", color_off="lightgray",
    )
    plotter.add_text("Play", position=(44, 14), font_size=10, color="black")

    if save_callback is not None:
        # "Save" button next to Play: a push-style checkbox -- the callback
        # (main.py:save_run_folder) asks for a location ("save as" dialog)
        # and writes the folder; the button then springs back to "off".
        def _save_pressed(is_on):
            if not is_on:
                return
            save_button.GetRepresentation().SetState(0)  # springs back at once (so it is not "on" in the video)
            try:
                save_callback(record_video=_record_video)
            except Exception as err:  # never kill the viewer over a failed save
                print(f"Save failed: {err}")
            finally:
                rep = save_button.GetRepresentation()
                rep.SetState(0)
                try:
                    plotter.render()
                except Exception:
                    pass

        save_button = plotter.add_checkbox_button_widget(
            _save_pressed, value=False, position=(100.0, 10.0), size=28,
            color_on="#1f77b4", color_off="lightgray",
        )
        plotter.add_text("Save", position=(134, 14), font_size=10, color="black")


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
