"""
The 3D scene: the droplet surface coloured by mean curvature, substrate grid,
roughness/anomaly overlays, the sphere, both contact lines, the info texts and
the progress bar, plus the two charts (charts.py). `Scene` bundles all actors
and shows a Frame (io.frame) -- from the live driver or from a run file.
"""

import math

import numpy as np
import pyvista as pv
import vtk

from DropletRelaxer.energy import mean_curvature

from .charts import chart_axis, make_energy_chart, update_energy_chart
from .config import scaled
from .interaction import enable_right_button_pan

_SCALAR_BAR_TITLE = "mean curvature |H|"


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


def update_com_text(plotter, com, d):
    """Refreshes the bottom-left centre-of-mass readout (see make_plotter);
    `d` is the xy offset COM - sphere centre (analysis.com_xy_offset_to_sphere)."""
    txt = getattr(plotter, "_com_text", None)
    if txt is not None and com is not None:
        txt.SetInput(f"COM = ({com[0]:+.3f}, {com[1]:+.3f}, {com[2]:.3f})")
        dev = getattr(plotter, "_com_dev_text", None)
        if dev is not None:
            dev.SetInput(f"COM offset to sphere (xy) = {np.hypot(*d):.4g}  (dx {d[0]:+.3g}, dy {d[1]:+.3g})")


def update_mesh(mesh, verts, tris, boundary_idx, curvature=None):
    """`curvature`, when given, is used as-is instead of recomputing it
    from `verts`/`tris`: every Frame carries the curvature the driver
    computed, so neither the live window nor the history slider/Play button
    recomputes it (at a few thousand triangles that recompute was a
    measurable share of what made Play feel sluggish)."""
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


class Scene:
    """All actors of the window for one run (`run`: io.frame.RunInfo, from
    the driver or a run file). `show_frame(frame)` puts a Frame on screen,
    `show_mesh(...)` an unrecorded state (right after a remesh), and
    `set_series(...)` the chart curves."""

    def __init__(self, run, vcfg, off_screen=None):
        self.run = run
        self.vcfg = vcfg = scaled(vcfg, run.mesh_scale)
        if off_screen is not None and off_screen != vcfg.off_screen:
            from dataclasses import replace
            self.vcfg = vcfg = replace(vcfg, off_screen=off_screen)
        cfg = run.config
        plotter, mesh, actor, volume_text = make_plotter(run.verts, run.tris, run.boundary_idx, vcfg)
        self.plotter, self.mesh, self.actor, self.volume_text = plotter, mesh, actor, volume_text
        self.epoch = 0
        self.n_points = len(run.verts)
        if vcfg.show_substrate:
            make_substrate_mesh(plotter, vcfg.substrate_size, vcfg.substrate_grid_resolution)
        if run.roughness_field is not None and vcfg.show_roughness:
            make_roughness_surface(plotter, run.roughness_field, vcfg.substrate_size,
                                   vcfg.roughness_opacity, vcfg.roughness_colormap)
        if cfg.anomaly.enabled:
            make_anomaly_disc(plotter, cfg.anomaly, cfg.solver.contact_angle_deg, vcfg.substrate_size)
        self.x_range, self.x_label = chart_axis(cfg, run.total_iters)
        (self.chart, self.line, self.marker,
         self.dchart, self.dline, self.dmarker) = make_energy_chart(plotter, self.x_range, self.x_label)
        self.obstacle_mesh = make_obstacle_mesh(plotter, cfg.obstacle)
        self.obstacle_base_points = (self.obstacle_mesh.points.copy()
                                     if self.obstacle_mesh is not None else None)
        if self.obstacle_mesh is not None:
            # the sphere mesh is built at the origin; put it at its start position right away
            # (frames -- and with them update_obstacle_mesh -- only come after the first equilibration)
            from DropletRelaxer.obstacle import sphere_center_at
            update_obstacle_mesh(self.obstacle_mesh, sphere_center_at(cfg.obstacle, 0.0),
                                 self.obstacle_base_points)
        self.wetting_actor = None
        self.ring_actor = update_ring_outline(plotter, None, run.verts[run.boundary_idx])
        update_volume_text(volume_text, run.V0, run.V0)
        self.bar = None

    @property
    def off_screen(self):
        return self.vcfg.off_screen

    def open_window(self, title="Droplet Relaxer"):
        """Opens the real interactive window (only meaningful with a display).
        auto_close=False and interactive_update=True let the Python code keep
        running while the window stays open."""
        if self.off_screen:
            return
        self.plotter.show(title=title, interactive_update=True, auto_close=False)
        if self.vcfg.pan_with_right_button:
            enable_right_button_pan(self.plotter)

    def _set_mesh(self, epoch, verts, tris, boundary_idx, curvature=None):
        if epoch != self.epoch or len(verts) != self.n_points:
            # A remesh boundary: a different vertex/triangle count needs a new
            # actor (see replace_mesh).
            self.mesh, self.actor = replace_mesh(self.plotter, self.actor, verts, tris, boundary_idx, self.vcfg)
            self.epoch, self.n_points = epoch, len(verts)
            if curvature is not None:
                self.mesh["curvature"] = curvature
        else:
            update_mesh(self.mesh, verts, tris, boundary_idx, curvature=curvature)

    def _set_lines(self, verts, boundary_idx, sphere_center, wetted_loop):
        update_obstacle_mesh(self.obstacle_mesh, sphere_center, self.obstacle_base_points)
        loop_points = verts[wetted_loop] if wetted_loop is not None else None
        self.wetting_actor = update_wetting_outline(self.plotter, self.wetting_actor, loop_points)
        self.ring_actor = update_ring_outline(self.plotter, self.ring_actor, verts[boundary_idx])

    def show_frame(self, f):
        """Mesh, sphere, contact lines and readouts of the Frame `f`."""
        self._set_mesh(f.epoch, f.verts, f.tris, f.boundary_idx, f.curvature)
        self._set_lines(f.verts, f.boundary_idx, f.sphere_center, f.wetted_loop)
        update_com_text(self.plotter, f.com, f.com_offset_xy)
        update_volume_text(self.volume_text, f.volume, self.run.V0)

    def show_mesh(self, epoch, verts, tris, boundary_idx, sphere_center=None, wetted_loop=None):
        """An unrecorded state, e.g. the new mesh right after a remesh: the
        sphere and BOTH contact lines must follow it, otherwise they stay at
        the old frame (ring at the substrate, no sphere line) while the bridge
        forms."""
        self._set_mesh(epoch, verts, tris, boundary_idx)
        if sphere_center is not None:
            self._set_lines(verts, boundary_idx, sphere_center, wetted_loop)
        else:
            self.ring_actor = update_ring_outline(self.plotter, self.ring_actor, verts[boundary_idx])

    def set_series(self, xs, energies, forces):
        """Full chart curves (one value per recorded frame); markers on the last point."""
        if len(xs):
            update_energy_chart(self.chart, self.line, xs, energies, marker=self.marker, dchart=self.dchart,
                                dline=self.dline, dmarker=self.dmarker, forces=forces)

    def mark(self, x, energy, force):
        """Moves the chart markers to one frame (history slider)."""
        self.marker.update([x], [energy])
        self.dmarker.update([x], [force[2]])
        self.dmarker._aux["fx"].update([x], [force[0]])
        self.dmarker._aux["fy"].update([x], [force[1]])

    def update(self):
        """Redraw and handle pending window events (no-op off screen)."""
        if self.off_screen:
            return
        try:
            self.plotter.update()
        except Exception:
            pass

    def process_events(self):
        """Handle pending window events (rotate, close) without redrawing."""
        if self.off_screen:
            return
        try:
            self.plotter.iren.process_events()
        except Exception:
            pass
