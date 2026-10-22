"""
Central configuration for the Droplet Relaxer.

All parameters you would typically want to tune are collected HERE --
the rest of the program only reads from these three dataclasses.
Simply edit this file and run `python main.py` again.
"""

import copy
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class MeshConfig:
    """Geometry of the initial mesh (a cone as a non-optimal starting
    shape). The boundary ring starts at z=0, but note that only its
    height is pinned -- the contact line is otherwise free to slide
    in x/y during relaxation (see SolverConfig.contact_angle_deg)."""

    mesh_type: str = "disk"

    base_mesh_enabled: bool = True
    # True: the fluid is enclosed by a CLOSED mesh -- besides the free
    # liquid/vapor surface (the "dome") there is a flat triangle mesh on
    # the substrate (z = 0) filling the wetted base, sharing the contact-
    # line ring with the dome (see substrate.py). Its vertices lie on the
    # wall constraint z = 0 and follow the ring automatically (harmonic
    # embedding), and its area is the liquid/solid area A_sl used in the
    # Young energy  E = gamma_lv*A_lv - cos(theta)*gamma_lv*A_sl.  False:
    # the old open dome with the wall energy from the shoelace formula
    # (mathematically the same energy, no base mesh).
    # "disk": quasi-uniform concentric-ring mesh, valence 6 everywhere
    # (recommended; see mesh.py:make_disk_cone). "polar": the original
    # latitude/longitude grid (n_theta x n_phi), whose tip is a fan of
    # n_phi skinny triangles -- vertices pile up there under the area
    # flow and the sphere obstacle meets exactly that weak spot.

    n_rings: int = 11
    # "disk" mesh: number of rings between tip and boundary (the
    # boundary ring gets 6*n_rings vertices; total 6*n_rings^2 triangles).

    n_theta: int = 14
    # Number of "latitude rings" between the tip and the boundary.
    # Higher = finer mesh radially, but more compute cost (~quadratic).

    n_phi: int = 28
    # Number of points per ring (azimuthal resolution).
    # Higher = rounder shape, but more compute cost.

    radius: float = 1.0
    # Radius of the fixed contact line (= cone/droplet base).

    start_height: float = 0.8
    # Height of the cone tip at the start. Larger = more strongly
    # deformed (further from the optimum) starting state, takes
    # longer to converge.
    #
    # Lowered further from 1.1 (itself halved from the original 2.2)
    # so the droplet starts out completely clear of the sphere obstacle
    # at its held start position, rather than just touching it. Note
    # that `target_volume` rescales the mesh to hit an exact volume, so
    # this doesn't simply scale the tip height proportionally -- a
    # shorter cone at the same base radius has less natural volume, so
    # resolve_volume_scale picks a LARGER scale factor to compensate.
    # At the default physical scale (see target_volume), start_height
    # =0.8 gives a scale factor s=53.81 (vs. s=48.39 at the old 1.1),
    # so a scaled tip height of 0.8*53.81 =~ 43.0 -- BELOW the sphere's
    # static resting bottom at its held start position
    # (center_start[2] - radius = 150 - 100 = 50, see ObstacleConfig),
    # unlike the old 1.1 (scaled tip ~53.2, just past that same 50).
    # So with this value there is no contact at all yet during
    # ObstacleConfig.hold_atStart (the sphere held motionless at
    # center_start for the first share of the run): the droplet gets
    # to equilibrate completely undisturbed before the sphere starts
    # moving toward center_mid=(0,0,120) (bottom at 20, well past the
    # droplet), where real contact begins.

    target_volume: Optional[float] = 130000.0
    # Overall PHYSICAL SIZE of the droplet, as its exact starting
    # volume -- an alternative to hand-tuning `radius`/`start_height`.
    # When set, the mesh built from radius/start_height above (which
    # still controls its SHAPE/aspect ratio) is uniformly rescaled so
    # its volume equals this value exactly (mesh.py:resolve_volume_scale
    # computes the one length-scale factor `s` this takes; make_start_mesh
    # applies it). None (default) uses the mesh at its natural size
    # (radius/start_height as given, no rescaling).
    #
    # IMPORTANT: rescaling length by `s` does NOT leave the solver's
    # other ABSOLUTE-valued (not ratio-based) constants dynamically
    # equivalent on their own -- ObstacleConfig.wetting_snap_distance/
    # capture_radius, SolverConfig.volume_penalty (and the viewer's
    # substrate_size) are all expressed in the SAME physical length units
    # as this volume/radius/the obstacle's position, not as fractions of
    # the mesh. config.apply_volume_scale rescales them automatically by the
    # right power of `s` (length^1 for the first two, length^-4 for
    # volume_penalty -- see its docstring) whenever target_volume differs
    # from the mesh's natural size, so a run at any physical scale (e.g.
    # micrometers: droplet volume 130000, sphere radius 100) behaves the
    # same as the tuned default (both in stability and in how many
    # iterations/remeshes it takes) without hand-retuning those by hand.
    # ObstacleConfig.collision_stiffness/adhesion_strength/
    # SolverConfig.substrate_stiffness need NOT be rescaled (their
    # gradients are already scale-invariant -- also explained there), and
    # everything under "remesh_*" is already expressed as a RATIO of the
    # mesh's own edge length, so it auto-scales with no change needed.

    remesh_enabled: bool = True
    # Whenever the mesh has stretched too much (its longest edge
    # exceeds remesh_trigger_ratio times the ORIGINAL mean edge
    # length -- typically because the sphere obstacle dragged part of
    # the surface with it), it is uniformly refined: every triangle
    # split into 4 (see mesh.py:subdivide_mesh). This is GLOBAL
    # refinement -- the whole mesh gets denser, not just the
    # distorted region -- so triangle count can grow quickly; see
    # remesh_max_count. Set to False to go back to a fixed mesh.

    remesh_check_every: int = 60
    # How often (in iterations) to check the distortion metric above
    # and possibly trigger a remesh.

    remesh_trigger_ratio: float = 1.8
    # Remesh once the longest edge in the mesh exceeds this many times
    # the ORIGINAL mesh's mean edge length.

    remesh_max_count: int = 2
    # Hard cap on how many times the mesh may be refined (split) in one
    # run. Each refinement quadruples the triangle count (1-to-4
    # split), so even 2 refinements already means 16x the original
    # triangle count (and a per-iteration cost increased by roughly the
    # same factor, since the solver's cost scales with mesh size) --
    # raise this only if you also have the compute budget for it. 3
    # refinements (64x) turns a default ~3s run into more like a minute
    # or two.

    remesh_split_mode: str = "local"
    # How a split is carried out once the trigger above fires:
    #   "local"  = Rivara longest-edge bisection (mesh.py:refine_rivara):
    #              only the too-long triangles (plus the few neighbors
    #              needed to stay conforming) are bisected, everything
    #              else is left alone. Far cheaper in triangle count and
    #              solver time than "global", and it can refine
    #              specifically around the sphere (see below).
    #   "global" = the old uniform 1-to-4 split of every triangle
    #              (mesh.py:subdivide_mesh), capped by remesh_max_count.

    remesh_local_target_ratio: float = 1.0
    # Local mode: bisect until no edge is longer than this multiple of
    # the ORIGINAL mean edge length (away from the sphere).

    remesh_contact_len_ratio: float = 0.7
    # Local mode: within ~2 sphere radii of the sphere's current
    # position, refine down to this multiple of the original mean edge
    # length instead -- extra resolution exactly where the contact
    # patch and the contact line are. Refinement there also triggers
    # proactively (as soon as the sphere gets near a coarse region),
    # not only after the surface has already been stretched.

    remesh_max_triangles: int = 60000
    # Local mode: stop refining once the mesh has this many triangles
    # (safety cap against runaway growth).

    remesh_min_edge_ratio: float = 0.35
    # The complement of remesh_trigger_ratio: edges shorter than this
    # fraction of the ORIGINAL mean edge length are candidates for
    # collapsing (mesh.py:collapse_short_edges). This is what keeps
    # splitting from being a one-way ratchet: a region that no longer
    # needs the extra resolution a global split gave it (e.g. once the
    # sphere obstacle has moved on) gets thinned back out again,
    # concentrating resolution where it's actually still needed. Set to
    # 0 to disable collapsing (split-only, the original behavior).

    remesh_min_edge_fraction: float = 0.30
    # A collapse pass only TRIGGERS once at least this fraction of all
    # edges are shorter than remesh_min_edge_ratio -- not the instant a
    # single short edge appears. This mesh's cone/hemisphere is an
    # "umbrella" triangulation (a fan converging on one apex vertex),
    # which always has a handful of naturally short "spoke" edges near
    # the tip; triggering off the single shortest edge would fire
    # immediately and permanently on that alone (collapse_short_edges
    # usually can't remove those particular edges anyway -- the apex is
    # a high-valence vertex, so its neighbors tend to fail the
    # manifold-safety check). This fraction-based trigger only fires
    # once a genuinely large chunk of the mesh has gotten small (e.g.
    # after a split pass over-refined a region that has since relaxed).
    # A collapse pass also only ever runs after at least one split has
    # happened -- there's nothing to prune back on the original mesh.


@dataclass
class SolverConfig:
    """Optimization method: Sobolev-preconditioned gradient descent
    (see the method comparison) minimizes area at fixed volume
    (penalty formulation)."""

    learning_rate: float = 0.25
    # Step size per iteration. Too large -> the mesh can "overshoot"
    # or degenerate; too small -> slow convergence.

    sobolev_alpha: float = 3.0
    # Strength of the Laplacian smoothing of the gradient (Sobolev
    # preconditioning). 0 = plain (unpreconditioned) gradient descent.
    # Larger = smoother but still global motions; see the method
    # comparison for the effect on convergence speed.

    volume_penalty: float = 50.0
    # Weight k in E(x) = Area(x) + k * (Volume(x) - V0)^2.
    # Larger = stronger penalty on volume deviation (more accurate
    # volume conservation), but too large a value can make the
    # optimization numerically unstable.

    n_iterations: int = 1000000
    # Maximum total number of iterations (safety limit). In quasi-static mode
    # (QuasiStaticConfig) the run ends when the sphere has reached center_end;
    # n_iterations only stops a run that does not get there. In the old
    # iteration-driven mode (quasi.enabled = False) it is the run length.

    grad_tol: float = 1e-6
    # Stopping criterion: stop once the gradient norm falls below this
    # value (in practice rarely reached, see the method comparison --
    # the process is usually bounded by n_iterations instead).

    gamma_lv: float = 0.88
    # Liquid/vapor surface tension gamma_lv (energy per area, in the
    # model's own units). All interface energies scale with it: the
    # free surface carries gamma_lv*A_lv, the wetted substrate base
    # -gamma_lv*cos(theta)*A_sl, the wetted sphere patch
    # -gamma_lv*cos(theta_sphere)*A_patch. The equilibrium SHAPE does not
    # depend on it (only ratios enter), but the numerical penalty terms
    # (volume_penalty, collision_stiffness, substrate_stiffness) keep
    # their absolute values, so a smaller gamma_lv makes them relatively
    # stronger. The plotted "Surface free energy" is the total
    # E = gamma_lv*A_lv - cos(theta)*gamma_lv*A_sl - cos(theta_s)*gamma_lv*A_patch
    # (main.py:surface_free_energy); the capillary force follows from it.

    contact_angle_deg: float = 90.0
    # Target (Young's) equilibrium contact angle between the droplet
    # and the flat substrate, in degrees. The contact line is always
    # free to slide in x/y (only its height z=0 is fixed); this angle
    # controls where it settles:
    #   90  = neutral wetting -- no wall energy at all (kappa=0), the
    #         contact line is governed purely by the area/volume terms.
    #   <90 = more wetting -- the droplet is pulled to spread out
    #         (larger contact radius).
    #   >90 = less wetting -- the droplet beads up and pulls the
    #         contact line inward (smaller contact radius).
    # See energy.py:substrate_wall_energy_and_grad for the derivation
    # (Young's equation via a line-integral "wall energy").

    substrate_stiffness: float = 80.0
    # Penalty weight keeping the whole surface from dipping below the
    # substrate z=0 (not just the contact line, which is constrained
    # separately). Only matters when something pushes part of the mesh
    # down hard enough to fold it under the table -- e.g. the sphere
    # obstacle. See energy.py:substrate_penetration_energy_and_grad.
    # Keep this roughly the same order of magnitude as volume_penalty
    # and ObstacleConfig.collision_stiffness -- much larger values can
    # make the different penalty terms fight each other and destabilize
    # the (explicit) gradient step.

    max_step_fraction: float = 0.3
    # Trust region for the explicit step: no vertex may move further in
    # ONE iteration than this fraction of its own shortest incident
    # edge (excess is scaled down, direction kept). Penalty terms
    # (collision, adhesion, volume, wetting) produce forces that do not
    # shrink with the mesh size, while triangle edges do -- so on a
    # refined mesh a single step could otherwise carry a vertex several
    # edge lengths, flip triangles, and blow the whole mesh up (the
    # "droplet vanishes / overflow warnings" failure). On a healthy,
    # well-resolved run this cap is never reached, so it costs nothing.
    # 0 disables it (not recommended).

    tangential_smoothing_every: int = 10
    tangential_smoothing_strength: float = 0.3
    # Every this many iterations, move each free vertex 30 % of the way
    # toward its neighbors' centroid, tangentially only (mesh.py:
    # tangential_smooth_step). Area minimization only determines the
    # NORMAL motion of the surface; the tangential part of the discrete
    # flow is a numerical artifact that piles vertices up (without this,
    # ~57 vertices end up within 1e-3 of the apex), which starves the
    # trust region there and makes the apex relax very slowly. 0 = off.


@dataclass
class ObstacleConfig:
    """A movable rigid sphere the droplet can be dragged/deformed
    around (see obstacle.py). This is a geometric collision + adhesion
    obstacle with an exact Young-angle wetting term (wetting.py).
    Droplet and sphere never separate topologically (no pinch-off)."""

    enabled: bool = True
    # False disables the obstacle entirely (no sphere drawn, no extra
    # energy term -- behaves exactly like before this feature existed).

    radius: float = 100.0
    # Sphere radius. Together with MeshConfig.target_volume, the default
    # scenario is now a physically-scaled example (e.g. micrometers: a
    # 100 um bead pressed into a ~130000 um^3 droplet) rather than the
    # earlier dimensionless (radius ~1) toy scale -- see the long
    # comment on MeshConfig.target_volume for how main.py keeps the
    # solver's OTHER absolute-valued constants consistent at this scale.

    center_start: Tuple[float, float, float] = (0.0, 0.0, 125.0)
    center_end: Tuple[float, float, float] = (50.0, 0.0, 125.0)
    # Straight sphere path from center_start to center_end (physical units,
    # NOT rescaled with the mesh). With QuasiStaticConfig.enabled (default) the
    # sphere moves in steps along this line and only moves on once the liquid
    # bridge has relaxed (see QuasiStaticConfig); the plots then use the tip
    # x-coordinate (sphere centre x) as their x-axis. Start == end: one
    # equilibrium at a fixed position. With quasi-static mode off the sphere
    # moves linearly over solver.n_iterations iterations instead.
    # Pick positions that touch the droplet's SURFACE (a single-sheet surface
    # mesh cannot represent a sphere that enters the liquid; with wetting that
    # ends in the "wrapping around the sphere" stop from solver.py). With
    # radius 100 the sphere's bottom point is at center[2] - radius = 25 for
    # the default z = 125, i.e. it dips into the ~40-tall droplet.

    collision_stiffness: float = 80.0
    # How hard vertices are pushed back out once the sphere overlaps
    # them (quadratic penalty, same style as SolverConfig.volume_penalty
    # -- keep it roughly the same order of magnitude as that and
    # SolverConfig.substrate_stiffness). Too small -> the mesh visibly
    # sinks into the sphere; too large -> the different penalty terms
    # fight each other (SolverConfig.max_step_fraction keeps this from
    # blowing up, but it then converges slowly).

    adhesion_strength: float = 0.0
    # How strongly nearby surface points are pulled onto the sphere
    # once within capture_radius. This is a numerical "capture" aid
    # that gets the surface into contact; with wetting_enabled the
    # actual contact physics (energy, angle) is handled by the wetted
    # patch below, and adhesion only acts on the not-yet-wetted rest.
    # DEFAULT 0 (off). Adhesion is a per-vertex point force that does
    # not shrink with the mesh size, while the surface-tension force per
    # vertex scales with the triangle area (~h^2); on the fine mesh
    # after a remesh, adhesion was measured at 10-18x the surface-
    # tension force on the two rows next to the contact line. It drags
    # a whole band of the sheet onto the sphere (92-94 % of all vertices
    # ended up "wetted"), distorts the rows next to the contact line and
    # thereby falsifies the contact angle at the sphere. With 0 the
    # contact still forms (collision + snap-distance capture, see
    # wetting_snap_distance); the wetted share was 20-47 % instead.
    # If you switch it on, keep it small (a value of 3.0 used to end in
    # the "wrapping" stop).

    capture_radius: float = 0.25
    # Distance from the sphere surface within which adhesion acts.

    wetting_enabled: bool = True
    # True: the part of the surface in contact with the sphere becomes
    # a genuine wetted patch (wetting.py). Its facets are treated as
    # liquid/SOLID interface -- they are held on the sphere and carry
    # no liquid/vapor surface energy (weight 0 in the area term), just
    # like the flat substrate's wetted base is not part of the surface
    # at all. The patch's boundary loop is the explicit three-phase
    # contact line and gets the Young's-equation energy
    # -cos(theta_Y) * (wetted area on the sphere). Patch membership is
    # recomputed from the current geometry every iteration (no memory /
    # hysteresis -- see wetting.py). False = collision + adhesion only.

    wetting_snap_distance: float = 0.01
    # A vertex is in contact ("wetted") when it is within this distance
    # of the sphere surface, on either side. Wetted vertices are then
    # projected exactly onto the sphere. NOTE: like the other absolute
    # lengths, main.py multiplies it by the mesh scale (x53.8 for the
    # default volume), so 0.01 means ~0.54 in the working units.
    # IMPORTANT: keep the scaled value well below the dome edge length
    # (here ~2-3.5): with the former default 0.06 (-> 3.2, i.e. MORE than
    # one edge) the first row of dome vertices next to the contact line was
    # re-captured onto the sphere in every iteration, which pinned the line:
    # for theta_sphere = 120 deg the static angle stayed at ~100 deg instead
    # of 120 and the line could hardly recede (hysteresis). With 0.01 the
    # static angle is 119-120 deg. Too small -> contact struggles to
    # establish and can flicker (0.003 also works in the tests).

    wetting_contact_angle_deg: float = 90.0
    # Target (Young's) contact angle for the sphere, same convention as
    # SolverConfig.contact_angle_deg for the substrate: 90 = neutral (no
    # wetting energy, kappa_sphere=0), <90 = wetting (the patch is
    # rewarded for growing), >90 = non-wetting (rewarded for
    # shrinking). Independent from the substrate's contact angle since
    # a real system can have different surface chemistry on the two
    # materials.

    wetting_min_patch_vertices: int = 6
    # Ignore contact (treat as not wetted) unless at least this many
    # vertices are wetted: a loop from a handful of vertices is too
    # coarse a discretization of a contact line.


@dataclass
class OutputConfig:
    """What the driver hands to its sinks (HDF5 file, live window, console)."""

    every: int = 200
    # Only for the iteration-driven mode (QuasiStaticConfig.enabled = False or
    # no sphere): record a frame every how many iterations (plus one at every
    # remesh). In quasi-static mode a frame is recorded exactly when a sphere
    # step has converged, independent of this value. (Was VisConfig.render_every.)

    checkpoint_every: int = 5000
    # Overwrite the restart checkpoint in the output file every this many
    # iterations (and at the end of the run). 0 = only at the end.


@dataclass
class AnomalyConfig:
    """Circular wettability anomaly on the substrate (see anomaly.py): inside
    the disc the wall has another contact angle than elsewhere
    (`SolverConfig.contact_angle_deg`). All lengths are in the same
    physical units as the sphere positions (ObstacleConfig.center_*), NOT
    rescaled with the mesh."""
    enabled: bool = False
    center_xy: tuple = (80.0, 0.0)
    # Centre of the disc in the substrate plane (x, y).
    radius: float = 20.0
    # Radius of the disc (centre of the transition zone).
    contact_angle_deg: float = 120.0
    # Young contact angle on the disc: < 90 = more hydrophilic (the droplet
    # is drawn onto it), > 90 = more hydrophobic (the droplet avoids it).
    transition_fraction: float = 0.1
    # Total width of the smooth transition zone as a fraction of the radius
    # (0.1 -> from 0.95 r to 1.05 r; quintic smoothstep, C2). It should not
    # be much smaller than the contact-line edge length (printed at the
    # start of a run), otherwise the contact line cannot resolve it.


@dataclass
class RoughnessConfig:
    """Random substrate roughness / chemical disorder (see roughness.py and the
    README section "Random substrate roughness"). Adds a quenched random
    wettability field delta_kappa(x, y) to the wall energy
    E = -gamma_lv * Int_wetted delta_kappa dA (contact-line vertices only).
    Lengths are in the same physical units as the sphere positions and the
    anomaly (NOT rescaled with the mesh)."""
    enabled: bool = True
    # Master switch (extra option, off by default).
    mode: str = "chemical"
    # "topography": Gaussian height field h(x, y); local Wenzel factor
    #   r = sqrt(1 + |grad h|^2), kappa_eff = clip(r * cos(theta_Y), -1, 1).
    #   NO effect at theta_Y = 90 deg (SolverConfig.contact_angle_deg).
    # "chemical":   kappa = cos(theta_Y) + delta_kappa, Gaussian delta_kappa.
    seed: int = 0
    # Random seed (fixed -> reproducible realisation; vary it for statistics).

    # --- amplitude --------------------------------------------------------
    rms_slope: float = 0.3
    # topography: rms of |grad h| (dimensionless). Keep <~ 0.5.
    rms_height: Optional[float] = None
    # topography: rms height (a length); if not None it overrides rms_slope.
    rms_kappa: float = 0.2
    # chemical: rms of delta_kappa. Keep <~ (1 - |cos theta_Y|) / 3, otherwise
    # kappa is clipped to [-1, 1] (the printed summary reports the fraction).

    # --- spectrum / geometry ----------------------------------------------
    box_size: float = 400.0
    # Side L of the field box, centred on the origin (periodic when generated).
    # The contact line must stay inside [-L/2, L/2]^2 (warning otherwise).
    q_cutoff: Optional[float] = None
    # Explicit upper wavenumber q_c; None -> from the initial contact-line edge
    # length l: q_c = 2 pi / (wavelengths_per_edge * l).
    wavelengths_per_edge: float = 4.0
    # lambda_c = wavelengths_per_edge * l. 4 keeps even the squared slope below
    # the Nyquist limit of the contact line; >= 2 is the hard limit.
    points_per_wavelength: int = 8
    # Grid points per lambda_c of the field grid.
    max_grid: int = 4096
    # Upper limit of the grid size (N x N).
    normalize_sample: bool = True
    # True: this realisation has exactly the requested rms; False: the rms is
    # only correct on average over seeds.


@dataclass
class QuasiStaticConfig:
    """Event-driven (quasi-static) sphere motion, see quasistatic.py: the sphere
    stands still until the liquid bridge has relaxed, a frame (energy, force,
    shape) is recorded, then it advances by `step` and the relaxation continues
    from the relaxed state (warm start). Needs ObstacleConfig.enabled; the sphere
    path is ObstacleConfig.center_start -> center_end. The plots/slider use the
    tip x-coordinate (sphere centre x). Lengths in physical units."""
    enabled: bool = True
    # False: the old iteration-driven motion (sphere position = iteration /
    # n_iterations, frames every OutputConfig.every iterations).

    step: float = 0.1
    # Path length per sphere step. Should be clearly smaller than the shortest
    # wavelength of the substrate field (RoughnessConfig: lambda_c = 4 * contact
    # line edge length, ~20 here) and than the anomaly width.

    adaptive: bool = False
    # False: fixed step; a step that does not converge within max_iter_per_step
    # is accepted anyway and marked "not converged" (the frame is recorded).
    # True: such a step is rolled back to the last converged state and retried
    # with half the step (down to step * min_step_factor); after a step that
    # converged quickly (< max_iter_per_step / 3 iterations) the step doubles
    # again (up to `step`). Resolves metastable jumps finer, but the frames are
    # then unevenly spaced in x.

    min_step_factor: float = 0.125
    # adaptive only: smallest step = step * min_step_factor.

    min_iter_per_step: int = 200
    # A step is never declared converged before this many iterations.

    max_iter_per_step: int = 20000
    # Iteration budget of one step (see `adaptive` for what happens beyond it).

    check_every: int = 100
    # Convergence is tested every this many iterations of a step by comparing
    # with the state of the previous test.

    tol_energy: float = 2e-6
    # |E - E_prev| / (gamma_lv * V0^(2/3)) below this ...
    tol_force: float = 1e-5
    # ... and max |F - F_prev| / (gamma_lv * V0^(1/3)) over the force components ...
    tol_step: float = 1e-4
    # ... and the largest vertex move of one iteration within the last window,
    # relative to V0^(1/3), all below their tolerance => converged. (E_prev,
    # F_prev: values at the previous test.) The scales make the tolerances
    # independent of the physical size of the droplet.


@dataclass
class Config:
    mesh: MeshConfig = field(default_factory=MeshConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    obstacle: ObstacleConfig = field(default_factory=ObstacleConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    roughness: RoughnessConfig = field(default_factory=RoughnessConfig)
    quasi: QuasiStaticConfig = field(default_factory=QuasiStaticConfig)
    # Optional random substrate roughness / chemical disorder (roughness.py);
    # disabled by default.
    roughness_field: object = field(default=None, repr=False, compare=False)
    # Runtime object (RoughnessField) built by main() from `roughness`; not a setting.


def apply_volume_scale(cfg, mesh_scale):
    """Copy of `cfg` with every value that is expressed in absolute physical
    units rescaled by the length factor `mesh_scale` (see
    MeshConfig.target_volume and mesh.resolve_volume_scale), so a run behaves
    the same (same stability, roughly the same iteration and remesh counts) at
    whatever physical scale target_volume asks for.

    volume_penalty gets s^-4: its gradient contribution scales as length^5,
    one power of length short of the area gradient's length^1 -- see solver.py's
    gE -- so a length^-4 coefficient keeps that term the same size relative to
    the others. collision_stiffness/adhesion_strength/substrate_stiffness are
    NOT touched: their gradient (2*k*gap*n_hat) is already length^1, matching
    the area gradient with a dimensionless k.

    Display values (substrate size, curvature colour limits) belong to the
    viewer, which applies the same factor itself (RunInfo.mesh_scale)."""
    cfg = copy.deepcopy(cfg)
    if mesh_scale != 1.0:
        cfg.solver.volume_penalty /= mesh_scale ** 4
        cfg.obstacle.wetting_snap_distance *= mesh_scale
        cfg.obstacle.capture_radius *= mesh_scale
    return cfg


# The instance actually used by main.py. Either edit this directly, or
# build your own Config() with different values in main.py.
CONFIG = Config()
