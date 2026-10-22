"""Display settings of the viewer (the [viewer] table of a run's TOML file)."""

from dataclasses import dataclass, replace
from typing import Tuple


@dataclass
class ViewerConfig:
    """Display settings of the viewer ([viewer] table of the run's TOML file).
    Lengths are in the units of the unscaled start mesh; the viewer multiplies
    them by the run's mesh_scale (MeshConfig.target_volume), like the solver
    does with its own lengths."""

    window_size: Tuple[int, int] = (800, 800)

    colormap: str = "inferno"
    # Any Matplotlib/VTK colormap works, e.g. "viridis", "coolwarm".

    curvature_clim: Tuple[float, float] = (0.0, 4.0)
    # Fixed color scale for the mean curvature, so colors don't keep
    # rescaling during the animation. Set to None for automatic
    # per-frame scaling.

    show_edges: bool = True
    # Overlay the triangle edges as a thin wireframe.

    edge_visible_max_triangles: int = 8000
    # Above this triangle count, the edge wireframe is automatically
    # turned off for that mesh regardless of `show_edges`, and turned
    # back on if a later remesh event drops back below it. Once
    # triangles get numerous/small enough that neighboring edges are
    # only a pixel or two apart on screen (easily reached after a few
    # splits, e.g. under an active sphere obstacle -- see "Rivara local
    # refinement" and "Contact-patch collapse" in the README), the black
    # wireframe can cover nearly every pixel and the surface reads as
    # solid black instead of showing the curvature coloring underneath.

    off_screen: bool = False
    # Set to True ONLY for automated tests/servers without a display (the
    # window is rendered into memory and closed at the end). For normal local
    # use: leave as False. If you hit OpenGL/VTK errors on Linux, see the
    # "Troubleshooting" section in README.md before changing this.

    video_max_frames: int = 600
    # The "Save" button (after the run) and `droplet-view --video` record the
    # replay (all recorded frames incl. charts, as shown in the window) as an
    # mp4 (needs imageio-ffmpeg; without it a .gif is written). At most this
    # many frames are rendered into the video (the recorded
    # frames are subsampled evenly). Recording takes roughly as long as
    # playing that many frames by hand.

    video_fps: int = 30
    # Frame rate of the saved video.

    show_substrate: bool = True
    # Draw a translucent grid at z=0 representing the substrate, so the
    # droplet's motion (e.g. while being dragged around by the sphere
    # obstacle) reads clearly against a fixed reference frame, rather
    # than only being visible via the small axes triad in the corner.

    substrate_size: float = 3.5
    # Side length of the substrate grid, centered at the origin.

    substrate_grid_resolution: int = 14
    # Number of grid cells per side of the substrate.

    show_roughness: bool = True
    # With RoughnessConfig.enabled: draw the random wall (local contact angle
    # theta_eff = arccos(kappa_eff) of the substrate) as a coloured plane just
    # above z = 0 (size = substrate_size), with a vertical colorbar on the left.

    roughness_opacity: float = 0.75
    # Opacity of that plane (0 = invisible, 1 = opaque).

    roughness_colormap: str = "coolwarm"
    # Colormap of the roughness plane (and its colorbar).

    pan_with_right_button: bool = True
    # Remap the right mouse button to pan/translate the camera. The
    # default PyVista/VTK binding uses the right button to zoom
    # (dolly) and puts pan on the middle button, which many
    # trackpads/mice don't have a convenient equivalent for. With this
    # on: left-drag rotates, right-drag pans, and the scroll wheel
    # still zooms.

    axes_offset: Tuple[float, float, float] = (0.0, -0.25, 0.01)
    # Extra displacement of the x/y/z axes tripod (fractions of
    # substrate_size, added to its base position): negative y, positive z.

    playback_interval_ms: int = 90
    # How often (in milliseconds) the history slider's Play button
    # advances to the next PLAYED frame (see playback_max_frames below
    # for what "played" means) -- 90 ms is ~11 fps. This is playback speed,
    # not simulation speed: a smaller value plays the recorded history
    # back faster, not the relaxation itself (which has already
    # finished by the time you can press Play).
    #
    # Was 40 ms (~25 fps) originally, which could make Play feel like
    # it nearly hangs the window -- most of that turned out to be a
    # genuine performance bug in mean_curvature's boundary-vertex fixup
    # (energy.py:_replace_boundary_curvature rebuilt a full Python-loop
    # adjacency graph from scratch on every single frame; fixed to a
    # vectorized version, ~5x faster), but even with that fixed, redoing
    # real work (a curvature recompute, and at a remesh-boundary frame a
    # full mesh/actor rebuild -- see visualize.py:replace_mesh) 25x a
    # second is still more than is comfortable on modest hardware. 90 ms
    # leaves real headroom without looking obviously choppy.

    playback_max_frames: int = 120
    # Play doesn't necessarily step through every SINGLE recorded frame
    # -- with output.every=10 over a few thousand iterations plus extra
    # frames at every remesh event, that can be several hundred, each
    # potentially costing a mesh/actor rebuild. Instead it subsamples
    # down to at most this many evenly-spaced frames for one full
    # playback pass, which bounds the total work regardless of how
    # densely the run was recorded (recording density is set by
    # output.every/remesh frequency for a good history slider scrub
    # resolution, which is a different concern from playback smoothness
    # here). Raise this (or lower playback_interval_ms) for a smoother
    # -- but heavier -- playback; the history SLIDER itself is
    # unaffected and can still land on any individual recorded frame.


def scaled(vcfg, mesh_scale):
    """Copy of `vcfg` with the lengths in physical units: substrate_size
    scales with the mesh, the curvature colour limits (1/length) inversely."""
    if mesh_scale == 1.0:
        return vcfg
    clim = vcfg.curvature_clim
    if clim:
        clim = (clim[0] / mesh_scale, clim[1] / mesh_scale)
    return replace(vcfg, substrate_size=vcfg.substrate_size * mesh_scale, curvature_clim=clim)
