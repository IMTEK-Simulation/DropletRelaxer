# Droplet Relaxer

A minimal but fully functional recreation of the core principle behind
Ken Brakke's Surface Evolver: a triangle mesh (a droplet with a free
contact line on a flat substrate, at a configurable contact angle) is
relaxed to its area-minimal shape at fixed volume, via
Sobolev-preconditioned gradient descent, with live 3D visualization
(colored by mean curvature). A movable rigid-sphere obstacle can
optionally be dragged through the droplet to see it deform around it.

## Quick start (recommended)

```bash
./run.sh
```

On first run, this script automatically creates its own virtual
environment (`venv/`), installs the dependencies into it, and then
starts `main.py` -- all in one command, without you having to activate
a venv yourself. On every subsequent run, the install step is skipped
and the program starts directly. If the script isn't executable yet:
run `chmod +x run.sh` once.

An interactive 3D window will open, with a live surface-energy chart
docked at the top (so you can see the relaxation progress at a glance,
not just the final shape), a translucent grid at z=0 marking the
substrate, and the droplet's surface colored by mean curvature. While
the optimization runs: left-drag rotates, right-drag pans/translates
the camera, and the scroll wheel zooms. The window stays open after
the relaxation finishes -- keep exploring the final shape, and close
it (or press 'q' inside it) whenever you're done. The console prints
the starting and final area, volume deviation, and contact-line radius
at the end. By default a translucent sphere is also shown, sweeping
through the droplet over the course of the run -- see "Sphere obstacle
(prototype)" below to tune or disable it.

A small axes tripod sits off to the side of the droplet, at world
position `(-0.40*substrate_size, -0.10*substrate_size, 0)` (moved
further out than the earlier symmetric `-0.25/-0.25` corner, which
can't go further along that diagonal without leaving the window; there
is more room along -x), for orientation, with its x/y/z captions at a
quarter of the default VTK size so they don't dominate the scene. (That position was picked empirically,
not analytically -- with this camera setup, the mathematical grid
corner `(+-0.5*substrate_size, +-0.5*substrate_size)` turns out to
project either off-screen entirely or onto the droplet's own on-screen
position for 3 of its 4 sign combinations; `-0.25/-0.25` was checked to
land clearly on-screen and clear of the droplet both on the initial
cone and on a heavily sphere-pressed, spread-out shape. Also note it's
placed via `SetUserTransform`, not `SetPosition` -- `vtkAxesActor`
silently ignores the latter, which is the reason an earlier version of
this tripod kept rendering at the world origin, overlapping the
droplet, regardless of what position it was "set" to.) The
mean-curvature colorbar is vertical, docked to the right edge below the
two charts described next.

Two chart overlays are docked at the top, side by side: the surface free
energy ("Surface free energy", left) and ("Capillary force", right) the **force vector
F = -dE/dc on the sphere** (c = sphere centre) as three lines: F_x, F_y, F_z
(positive F_z = the liquid pushes the sphere up). It works for any motion
direction of the sphere (vertical, lateral, oblique).

How it is computed (`solver.py`, accumulated in every relaxation iteration,
`diag["force"]`): by virtual work / force balance on the sphere, the force is the sum of
(i) the collision (non-penetration) gradient on the vertices pushed by the
sphere, (ii) the Young/wetting term on the wetted sphere patch (depends on v - c),
and (iii) the radial reaction on wetted vertices (liquid pressure on the patch
plus the radial pull of the contact line), i.e. the part of the surface-energy
gradient normal to the sphere that the constraint absorbs. No finite
differences in time or path length are involved, so the sign problem of
moving up/down and the pollution by the liquid's own relaxation of the old
energy-difference method are gone. Check: in static equilibrium F_z = -58.9
from the direct force vs. -59.95 from dE_tot/dz of equilibrium states (~2 %).
Both x-axes are the **tip x-coordinate** (sphere centre x) with fixed limits (the
planned path), one point per converged sphere step (see "Quasi-static sphere
motion"); the old iteration-driven mode (`QuasiStaticConfig.enabled = False`) plots
over the iteration number. Every value is an equilibrium value and plotted as it is
(no averaging); the y-ranges follow the data. The old
energy-path estimate -(dE/ds)(dz/ds) is kept only as a cross-check
(`force_from_energy.npy`, `force_parallel.npy` = F projected on the motion direction).

**Lateral force / "friction" -- important caveat.** On a uniform substrate a
droplet is free to translate, so at equilibrium E does not depend on the lateral
position of the sphere and F_x, F_y -> 0. The lateral force seen during a
moving run is a *pseudo-dynamic lag/restoring force* of the gradient-flow
relaxation: the liquid lags behind the moving sphere, and how much depends on
the artificial mobility (learning rate, Sobolev alpha, trust region) and on the
speed; it vanishes as the speed -> 0. It is therefore NOT a physical viscous
friction and contains no contact-angle-hysteresis pinning (the model has none).
Use it for qualitative/comparative statements at fixed settings, not as an
absolute friction coefficient. Transients right at first contact can be very large
(spikes), which is why the y-range uses percentiles.

The **current fluid volume** is shown live at the bottom right
(`V = ..., target, absolute and % deviation`), and follows the history
slider/Play. The volume is only held by a penalty
(`SolverConfig.volume_penalty`), so it is *not* exactly constant: a
deviation of about 1-5 % appears, mostly while the sphere presses in.

A small
diamond marker on each curve tracks the newest point (the frame shown) -- while just watching the live run it sits at the
advancing tip of the line, and it's most useful together with the
history slider below: scrubbing to an earlier frame moves both
diamonds back along their curves so you can read off exactly which
energy value (and drag cost) that frame corresponds to.

## Manual installation (alternative)

If you'd rather manage the environment yourself instead of using
`run.sh`:

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Adjusting parameters

**All** tunable parameters are collected in `config.py`, in these
groups:

- `MeshConfig` -- mesh resolution and starting shape (cone geometry)
- `SolverConfig` -- optimization method (step size, preconditioning,
  volume weighting, iteration count, contact angle)
- `ObstacleConfig` -- the optional movable sphere obstacle (see below)
- `AnomalyConfig` -- circular wettability anomaly on the substrate (see below)
- `RoughnessConfig` -- random substrate roughness / chemical disorder (see below; enabled in the shipped defaults, mode "chemical", anomaly disabled)
- `VisConfig` -- visualization (live on/off, render frequency, color
  scale, GIF export)

Just edit the values in `config.py` and run `python main.py` (or
`./run.sh`) again. Every parameter has a comment explaining what it
does and which direction changes push the behavior.

Typical first experiments:

- Increase/decrease `SolverConfig.learning_rate` to see at what point
  the optimization becomes unstable, or how much it slows down.
- Set `SolverConfig.sobolev_alpha = 0.0` to switch to plain
  (unpreconditioned) gradient descent and see the difference in
  convergence speed directly.
- Increase `MeshConfig.n_rings` (or `n_theta` / `n_phi` for `mesh_type="polar"`) for a finer mesh (more
  compute cost, but a smoother shape).
- Change `SolverConfig.contact_angle_deg` away from 90 and watch the
  contact line grow or shrink (see "Free contact line" below).
- Set `ObstacleConfig.enabled = False` to go back to a plain droplet
  with no obstacle.
- Set `MeshConfig.remesh_enabled = False` to see the raw, undistorted
  mesh distortion the obstacle causes (compare against the default
  `True`) -- see "Is this real contact? Remeshing" below.
- Set `VisConfig.pan_with_right_button = False` to get the default
  PyVista mouse mapping back (right-drag zooms, middle-drag pans).
- Set `VisConfig.save_gif = "relaxation.gif"` to additionally save an
  animation.

## Project structure

```
run.sh          - startup script (automatic venv, see Quick start above)
pyproject.toml  - package metadata (DropletRelaxer), pytest settings
DropletRelaxer/ - the solver package:
config.py       - all parameters (see above)
mesh.py         - mesh generation (disk/polar cone), remeshing (global split, Rivara local split, collapse), tangential smoothing
energy.py       - area/volume/curvature/wall energies and their analytical gradients
obstacle.py     - the movable sphere obstacle: path, collision, weak adhesion
wetting.py      - wetted patch, contact line and its Young-angle energy on the sphere
export.py       - "Save" button: writes the run folder (config, npy files)
substrate.py    - closed-mesh base: liquid/solid disk on z = 0 (build, strip, harmonic re-embedding)
roughness.py    - optional random substrate roughness / chemical disorder (band-limited Gaussian field, exact contact-line energy and gradient)
quasistatic.py  - event-driven sphere stepping: convergence test, step advance, adaptive rollback, ETA
solver.py       - Sobolev-preconditioned gradient descent (generator)
visualize.py    - PyVista live visualization (mesh, substrate grid, sphere, energy chart, camera controls)
analysis.py     - per-frame post-processing (surface free energy)
main.py         - entry point, wires everything together (incl. the remesh loop)
test/           - test suite (pytest): formula checks and equilibrium/force tests, see "Tests" below
tools/reference_run.py - records the reference runs of test/test_reference.py
docs/mathematical_background.pdf  - theory (Part I), implementation (Part II), Evolver comparison and outlook (Part III)
docs/mathematical_background.tex  - LaTeX source for the above
docs/mathematical_background.txt  - same LaTeX source as a plain .txt file
```

## Tests

The suite lives in `test/` and is separate from the program: a normal run does **not** execute it.

```
pip install -e ".[test]"   # installs the package and pytest
pytest                     # fast tests, ~10 s
pytest -m slow             # slow tests, ~6 min
pytest -m ""               # everything
pytest test/test_force.py::test_force_equals_minus_energy_derivative   # a single slow test, by node id
```

**Fast tests** (plain `pytest`; formulas on small meshes, each against an independent reference):
- area/volume/centre of mass: analytic values (triangle, hemisphere with convergence in the resolution), scaling laws, invariance under rigid motion and re-numbering;
- every analytic gradient (area with weights and signed base, volume, shoelace wall energy, substrate and sphere penalties, anomaly polygon flux, sphere wetting energy) against central finite differences;
- solid angle by turning angles against the spherical-cap formula; anomaly flux against a brute-force grid integral, roughness field (spectrum, flux integral, gradient), polygon inside/outside the disc, smootherstep properties (C2); closed mesh (base) against the open-mesh energy; non-convex ring fallback;
- graph Laplacian (symmetric, zero row sums, PSD), Sobolev solve as low-pass filter;
- subdivision and Rivara refinement keep area and volume to rounding; collapse and tangential smoothing change them by less than 1-2 %;
- sphere path (straight line, clamped) and the quasi-static logic: convergence test, step advance and path end, adaptive rollback/growth, reset after remesh, solver stepping the sphere only after convergence.

**Slow tests** (`pytest -m slow`; real relaxations on a fixed ~600-vertex mesh):
- static droplet = spherical cap with the Young angle (60, 90, 120 deg): contact radius within 1.5 % of the analytic cap for the actual volume (measured 0.1-0.6 %), all vertices on one sphere (<2 %), fitted contact angle within 1.5 deg, mean curvature = 1/R;
- the same equilibrium after a global remesh in the middle of the run;
- Sobolev alpha, learning rate and trust region change the path but not the result (contact radius spread < 0.3 %, energy < 0.03 %); energy decreases monotonically (up to the tiny rises of the tangential smoothing); the volume deficit of the penalty is proportional to 1/k (ratio 2.0 for k and 2k);
- force on the sphere against the central difference of the re-equilibrated total energy (one global refinement, z0 +- 2): agreement 0.04 % at a wetting angle of 90 deg and 4-6 % at 120 deg (tolerances 3 % and 10 %); lateral force on a homogeneous wall below 0.2 % of F_z;
- reference runs (`test/test_reference.py`, recorded by `tools/reference_run.py`): a sessile drop with global split + collapse and a short quasi-static sphere run with local remeshing reproduce the recorded final energy, area, volume, contact radius, remesh count and sphere force (relative 1e-10). With `DROPLET_REFERENCE_HASH=1` the SHA-256 of the final mesh is compared too (bit identity; meaningful only on the machine that recorded the reference).

Checked by mutation: deliberately breaking a formula (wrong factor in the area/volume gradient, sign of the shoelace gradient, solid angle, anomaly sign, Laplacian, sphere penalty, path speed in the fast suite; removed radial reaction, flipped sign of the Young force, flipped wall-angle sign in the slow suite) makes at least one test fail.

**Not covered (known limits):** the lateral force with a wettability anomaly is not tested against -dE*/dX: in the tried setup the energy of the re-equilibrated states was still changing with the number of iterations (slow lateral mode), so the numerical derivative was not reliable. The model itself (no hysteresis, bilateral wetting constraint) is not validated against an independent code or an axisymmetric capillary-bridge solution either.

## Free contact line and the contact angle

The contact line (where the droplet meets the substrate) is always
free to slide in x/y -- only its height is pinned to the substrate
plane, z=0. With no additional energy term, this free boundary
naturally settles at a 90-degree angle to the substrate (that is what
"no wall energy" means physically: minimizing area at fixed volume
with a boundary confined to a plane, but otherwise free, always meets
that plane orthogonally). That neutral case is
`SolverConfig.contact_angle_deg = 90.0`, the default.

Setting it away from 90 adds a wall energy term derived from Young's
equation (`gamma_SV - gamma_SL = gamma_LV * cos(theta_Y)`). Values below
90 deg pull the contact line outward (more wetting, the droplet spreads);
values above 90 deg pull it inward (less wetting, the droplet beads up).
This is an exact energy, not an approximation -- unlike the sphere obstacle
below.

**Closed mesh with a substrate base (`substrate.py`).** The fluid is a closed
surface: the free dome (liquid/vapor) plus a flat base lying on the substrate
(liquid/solid, z = 0), joined along the contact-line ring (drawn as a **cyan**
tube; the contact line on the sphere is lime). The base is *not* a set of
independent degrees of freedom: its interior vertices are re-placed every
iteration as the harmonic (Tutte) embedding of the ring, which is fold-free
for a convex ring, and the base has no z-motion (wall constraint). The total
energy is

    E = gamma_lv * A_lv  -  cos(theta) * gamma_lv * A_sl      (gamma_lv = SolverConfig.gamma_lv = 0.88)

implemented as per-triangle weights (dome `gamma_lv`, base `-gamma_lv*cos(theta)`,
sphere-wetted dome triangles 0). On a fixed mesh this reproduces the earlier
open-mesh + shoelace-integral wall energy to ~1e-13 (checked for 60/90/120 deg),
so the physics is unchanged; the base only makes the fluid explicitly enclosed
(and `A_sl` directly available). The dome dynamics (Sobolev Laplacian, trust
region, smoothing, remesh triggers) use dome triangles only. Before each remesh
the base is stripped, the open dome is remeshed, and the base is rebuilt from
the ring. The chart's **Surface free energy** is this total energy (dome + wetted base +
wetted sphere patch, `main.py:surface_free_energy`), and the **Capillary force**
is the direct force vector F = -dE/dc described above (computed from the same energy gradients, not by differentiating E along the path). `MeshConfig.base_mesh_enabled = False` restores the open mesh.
If the ring becomes too non-convex for the harmonic embedding (e.g. after a long
sideways drift), `add_base_mesh` falls back to an interior-free ear-clipping
triangulation of the ring (any simple polygon); the wall energy uses SIGNED base
areas (exactly the shoelace area), so a folded base cannot corrupt the physics.
Previously this case aborted the run with "substrate base mesh folded".
Cost: about +30 % run time (see the timings below).

## Circular wettability anomaly on the substrate (`AnomalyConfig`)

`CONFIG.anomaly` (`config.py`, module `anomaly.py`) gives a circular region of the
substrate another Young contact angle: `enabled`, `center_xy` (x, y), `radius`,
`contact_angle_deg` (< wall angle = more hydrophilic: the droplet is drawn onto it;
> wall angle = more hydrophobic: avoided) and `transition_fraction` (default 0.1:
the total width of the smooth transition zone is 10 % of the radius, i.e.
0.95 r ... 1.05 r). Lengths are in the physical units of the sphere positions
(not rescaled with the mesh). The disc is drawn on the substrate (blue =
hydrophilic, violet = hydrophobic).

Model: the wall energy is -gamma_lv * cos(theta_w(x, y)) * A_sl, i.e. an integral
of the local wettability over the wetted region. The homogeneous wall angle
(`SolverConfig.contact_angle_deg`) keeps being handled by the base triangles; the
anomaly adds only the difference,
E_anom = -gamma_lv * (cos theta_a - cos theta_0) * Integral_wetted s(x, y) dA,
with s = 1 inside, 0 outside and a quintic smoothstep (C2) in the transition zone.
The integral is evaluated EXACTLY for the polygon of the contact line as a boundary
flux (divergence theorem with a radial field, closed-form polynomial, Gauss
quadrature along each edge), so no base mesh or area sampling is needed; the
gradient is the exact shape derivative of the same integral (checked against finite
differences, ~1e-9; the integral against a brute-force grid, ~1e-5 relative). It
acts on the contact-line vertices only and is independent of the sphere (no direct
contribution to the force on the sphere); it is included in "Surface free energy".

Things to know: (1) A patch that lies completely under the wetted area has no
effect at all -- the energy only counts the part of the wetted region that
changes, i.e. what the contact line sweeps over. (2) The transition should not be
much narrower than the contact-line edge length (printed at the start; a note
appears if it is); the contact line then only resolves it coarsely. (3) There is no
contact-angle hysteresis in the model: pinning at the patch edge comes only from the
energy barrier. (4) Expected behaviour: a hydrophilic patch gives an energy well
(the droplet is drawn onto it; moving the droplet away costs about
gamma*(cos theta_a - cos theta_0)*A_covered), a hydrophobic patch an energy hill. The
lateral force on the sphere when dragging the droplet over it is quantitatively
meaningful for slow motion (compare with -dE/dX); for fast motion the
pseudo-dynamic lag force described above is added. Test: without the sphere, a
hydrophilic patch at the contact line moves the droplet toward it (COM x +8 vs. 0),
a hydrophobic one slightly away.

## Random substrate roughness (`RoughnessConfig`, optional)

`CONFIG.roughness` (module `roughness.py`, disabled by default; set `enabled = True`)
adds a random, quenched wettability field to the substrate. Young: `gamma_sv - gamma_sl
= gamma_lv * kappa`, `kappa = cos(theta_Y)`; wall energy `E = -gamma_lv * Int_wetted kappa dA`.
Only the difference to the homogeneous wall, `delta_kappa = kappa_eff - kappa_Y`, is added
(the homogeneous part stays in the base triangles):

    E_rough = -gamma_lv * Int_wetted delta_kappa(x, y) dA

Two modes (`mode`):

- `"topography"`: Gaussian height field `h(x, y)` with a band-limited white spectrum
  (plateau for `q_L <= |q| <= q_c`, hard cutoff, no `q = 0`); local Wenzel factor
  `r = sqrt(1 + |grad h|^2)`, `kappa_eff = clip(r * kappa_Y, -1, 1)`. Amplitude: `rms_slope`
  (or `rms_height`). No effect at `theta_Y = 90 deg` (the default wall angle!), so set
  `SolverConfig.contact_angle_deg` away from 90 deg.
- `"chemical"`: `kappa = kappa_Y + delta_kappa`, Gaussian band-limited `delta_kappa` with rms `rms_kappa`.

The integral over the wetted polygon is evaluated exactly for the bilinear interpolant of
`delta_kappa` as a contact-line flux (divergence theorem); the gradient is the exact shape
derivative and acts on the contact-line vertices (x, y) only. It does not depend on the
sphere, so it adds no force to the sphere's force output, but it does change the equilibrium
contact line, and hence the force indirectly. The cutoff follows the initial contact-line edge
length `l` (printed at the start): `lambda_c = wavelengths_per_edge * l` (default 4 l,
`q_c = pi / (2 l)`), so even the squared slope stays below the Nyquist limit of the contact
line. The field lives in a box `box_size` (physical units, centred on 0; default 400) that the
contact line must not leave (warning otherwise). `normalize_sample=True` gives this realisation
exactly the requested rms; `seed` makes it reproducible.

Visualisation: with `VisConfig.show_roughness = True` (default) the substrate is drawn as a
coloured plane showing the local contact angle `theta_eff = arccos(kappa_eff)` of the random
wall, with a vertical colorbar on the LEFT of the window (`roughness_opacity`,
`roughness_colormap`; the colour range is symmetric around the wall angle).

Limits: local Wenzel model only (no geometric pinning à la Joanny-de Gennes); the random
field creates metastable states (contact-angle-hysteresis-like pinning), so use long runs and
several seeds; keep `rms_slope <~ 0.5` and, for `"chemical"`, `rms_kappa <~ (1-|kappa_Y|)/3`
(otherwise the clipping fraction in the printed summary is > 0). Roughness is not remeshed with
the mesh: the field is fixed at the initial edge length; finer contact-line edges are fine, coarser
ones trigger the resolution warning. Tests: `tests/test_roughness.py` (cutoff, normalisation,
plateau, gradient vs finite differences, brute-force integral, orientation invariance, clipping).

## Physical scale (`MeshConfig.target_volume`)

The default scenario is now expressed at a real physical scale rather
than the earlier dimensionless (droplet radius ~1) toy scale: a droplet
of volume 130000 (e.g. cubic micrometers) with a sphere of radius 100
pressed into it. Setting `target_volume` rescales the mesh built from
`radius`/`start_height` (which still control its shape/aspect ratio) to
have EXACTLY that volume.

This matters beyond just the geometry: most of the solver's constants
are expressed as ratios of the mesh's own size (everything under
`remesh_*`, `max_step_fraction`, ...) and need no change at all -- but a
few (`SolverConfig.volume_penalty`, `ObstacleConfig.
wetting_snap_distance`/`capture_radius`, `VisConfig.substrate_size`) are
absolute physical lengths (or, for `volume_penalty`, an inverse
length^4), tuned for the old ~1-unit scale. `main.py` rescales all of
them automatically by the right power of the same length-scale factor
`target_volume` implies, so a run behaves the same (same stability, same
rough iteration/remesh counts) at any physical scale without hand-
retuning. `collision_stiffness`/`adhesion_strength`/`substrate_stiffness`
are deliberately NOT touched -- their gradient is already scale-
invariant. `VisConfig.curvature_clim` IS touched, but in the opposite
direction (divided by the scale factor, not multiplied): mean curvature
is an inverse length, so it shrinks as the mesh grows. Skipping this one
is exactly what caused an earlier "the fluid renders completely solid
black" bug at this default scale (mesh ~38x bigger than the unscaled
toy size) -- the actual curvature values dropped to a small fraction of
a `curvature_clim` still tuned for the old scale, so almost the entire
surface clamped to the low end of the `"inferno"` colormap, which is
black. It looked like a mesh-density/edge-rendering issue (and is easy
to mistake for one), but had nothing to do with triangle size. See the
long comment on `MeshConfig.target_volume` for the dimensional analysis
behind each choice.

**Performance note:** while the mesh's own triangle SIZE scales
correctly, a much larger sphere relative to the droplet (this default:
radius 100 vs. a droplet that settles to about 40) can still change
performance a lot, independently of the volume scaling itself, because
`ObstacleConfig.radius` sets the ABSOLUTE size of the "near the sphere"
refinement zone (`2 * radius`): a big sphere makes that zone cover a much
larger fraction of the whole mesh, so local (Rivara) mode ends up
re-refining almost everywhere, repeatedly, instead of a small patch --
a reversal of the usual local-mode advantage (see "Rivara local
refinement"), though the contact-patch collapse fix below narrows the
gap considerably. On the current default scenario (radius 100,
`center_start`/`center_mid`/`center_end` = 150/100/150 (the earlier default path used for these measurements; the current defaults in `config.py` are the quasi-static lateral pass (0,0,125) -> (20,0,125), the wettability anomaly disabled and the roughness enabled) (`hold_atStart`/`mid_fraction` were removed with the quasi-static motion; then with mid z = 100 the sphere bottom touches the substrate and the droplet is squeezed to ~0 thickness on the axis, volume ~94 %; use 120 for a gentler press),
`MeshConfig.start_height` = 0.8), local mode completes the full 2500
iterations in ~127 s (41 remesh events) and global mode in ~32 s (10
events); both finish cleanly with no divergence, and re-running either
mode reproduces the identical sequence of areas/volumes/remesh events
(see "Is the computation deterministic?" below). (An earlier version of
this scenario, at the old `start_height` = 2.2, had global mode diverge
partway through instead of finishing -- the taller starting cone made
for a rockier first few hundred iterations, which global mode's much
coarser, all-at-once splitting handled worse than local mode's small,
targeted ones. Lowering start_height -- see MeshConfig.start_height's
comment -- fixed that as a side effect, on top of its main purpose of
giving the droplet real equilibration time before the sphere gets
involved.) Global mode is faster here purely because it ends up with
fewer total triangles (it coarsens the entire mesh whenever possible,
not just the fine trail behind the sphere); for a sphere comparable to
or larger than the droplet, try it if `"local"`'s triangle count is
more than you need.

## Quasi-static sphere motion (`QuasiStaticConfig`)

The sphere does not move on a schedule any more: it stands still until the liquid
bridge has relaxed, a **frame** is recorded, then it advances by `step` along the
straight path `ObstacleConfig.center_start -> center_end` (default x = 0 ... 20, z = 125)
and the relaxation continues from the relaxed state (warm start). Code: `quasistatic.py`
(state and decisions), `solver.relax(..., quasi=...)` (the loop), `main.py` (frames).

**Convergence** (tested every `check_every` iterations of a step, never before
`min_iter_per_step`): all three of
`|E - E_prev| / (gamma_lv V0^(2/3)) < tol_energy`,
`max |F - F_prev| / (gamma_lv V0^(1/3)) < tol_force` and the largest vertex move of one
iteration within the window `/ V0^(1/3) < tol_step` (E_prev, F_prev: values at the
previous test; F = force of the liquid on the sphere; the scales make the tolerances
independent of the physical size). The raw gradient criterion `SolverConfig.grad_tol`
also counts. At equilibrium the force is F = -dE*/dc (envelope argument), so the
recorded energy and force are consistent -- unlike during a fast motion. Measured with
the shipped defaults (roughness on, `step = 1`, full run with remeshing, 4 frames x = 0 ... 3): the first
equilibration needs ~16000 iterations, each further step ~8000-8500 (~8 ms per iteration with ~3200 dome triangles, i.e. ~70 s per step);
the whole path x = 0 ... 20 (21 frames) is therefore ~180000 iterations, roughly 25 minutes. A
smaller `step` or tighter tolerances cost proportionally more. In the first test the force had settled to 3e-6 of its scale
while the energy was still creeping down by 1e-6 per 100 iterations (the slow
contact-line mode), which is why the energy tolerance is looser than the force one.

**Fixed vs. adaptive step (`adaptive`).** Fixed (default): every step has the same length,
the frames are evenly spaced in x and runs are directly comparable; a step that does not
converge within `max_iter_per_step` is accepted and marked "NOT CONVERGED" (console
line, `frames.csv`). It can step over a narrow feature of the substrate field and waste
time where everything is smooth. Adaptive: such a step is rolled back to the last
converged state and retried with half the step (down to `step * min_step_factor`);
after a step that converged in less than a third of the budget the step doubles again
(up to `step`). It resolves metastable jumps finer and saves time on smooth stretches,
but the frames are unevenly spaced and the result depends on the tolerances. Choose
`step` clearly smaller than the substrate field's shortest wavelength (lambda_c ~ 20)
and the anomaly width.

**Frames and display.** The window (mesh, sphere, contact lines, charts, history) is
only updated when a step has converged; in between only a progress bar is refreshed
about once per second (so the window stays responsive). Nothing is recorded for the
unrelaxed starting cone. The progress bar sits where the history slider appears after the
run: grey, filling green from left to right, text "Progress 37 % | elapsed ... | ETA ~...".
The progress is the fraction of the path covered; the ETA extrapolates the wall time per
unit path length of the recent steps (exponential average) and is only a rough guide,
because steps near the substrate features take more iterations. In `end_only` mode the
terminal shows the same information as a one-line bar. After the run the slider scrubs
over the tip x-coordinate (snapping to the nearest frame; Play steps through the frames
at constant speed in frame index), and a table of the frames (tip position, iterations
of the step, converged flag, energy, force) is printed and saved as `frames.csv` by the
Save button.

**Remeshing** works as before; the comparison values and the rollback state are reset
after a remesh. **Limits:** path dependence (forward and backward passes can follow
different branches), the energy of a jump between metastable states is not a
physical time evolution, and without a sphere (or with `enabled = False`) the old
iteration-driven mode (`render_every`, sphere position = iteration / `n_iterations`) is used.

## Sphere obstacle (prototype)

`ObstacleConfig` adds a rigid sphere that moves in a straight line
from `center_start` to `center_end` over the course of the relaxation
(set both equal for a static sphere). The droplet deforms around it
via two smooth penalty terms in `obstacle.py`:

- **collision** (`collision_stiffness`): pushes any vertex that ends
  up inside the sphere back out.
- **adhesion** (`adhesion_strength`, active within `capture_radius` of
  the surface): pulls nearby vertices onto the sphere. **Off by default
  (0)**: measured on the refined mesh, it was 10-18x the surface-tension
  force on the two rows next to the contact line, dragged 92-94 % of all
  vertices onto the sphere (20-47 % without it), distorted the rows at the
  contact line and thereby falsified the contact angle. Contact forms
  without it (collision + `wetting_snap_distance` capture).

**Limitations to be aware of** (the actual contact physics -- wetted
patch, contact line, Young angle -- is in "Capillary bridge" below; these two
penalty terms only provide collision and a weak pull-in):

- The droplet is one single-sheet mesh. It can deform around the sphere
  and hug it over a cap, but a sphere that goes *into* the liquid cannot be
  represented (see "Why can the droplet vanish?"), and there is no pinch-off
  into separate blobs.
- Keep `adhesion_strength` at 0 (default) or small. A strong pull drags a whole
  neighborhood of the sheet onto the sphere, where the vertices are snapped
  and pile up in a thin band; the run then ends in the "wrapping" stop.
- Volume conservation loosens somewhat while the obstacle is actively
  pushing on the surface (the collision/adhesion forces compete with
  the volume penalty). If you need tighter conservation, try raising
  `SolverConfig.volume_penalty` a bit, or lowering
  `ObstacleConfig.collision_stiffness`/`adhesion_strength`.
- A coarse mesh shows visible faceting/streaking in the curvature
  color right around the contact dimple, simply because there aren't
  enough vertices there to resolve it smoothly -- increase
  `MeshConfig.n_rings` for a cleaner-looking result (at proportionally
  higher compute cost), or see "Is this real contact? Remeshing" below.
- All three penalty weights (`SolverConfig.volume_penalty`,
  `SolverConfig.substrate_stiffness`, and
  `ObstacleConfig.collision_stiffness`) interact through the same
  explicit gradient step, so pushing any one of them much higher than
  the others tends to cause visible oscillation or outright numerical
  blow-up rather than a stiffer constraint. If you hit that, either
  bring the weights back into the same rough order of magnitude, or
  lower `SolverConfig.learning_rate`.

`SolverConfig.substrate_stiffness` is a related but separate
safeguard: it keeps the *whole* surface (not just the contact ring)
above the substrate plane, since the obstacle can otherwise push part
of the mesh down hard enough to fold it underneath the table.

## Is this real contact? Remeshing

**No, not in the rigorous sense** -- worth being precise about, since
it affects what the visible distortion at the contact point means.
There is no actual collision *detection* (no triangle-vs-sphere
intersection query, no tracked contact patch); each vertex
independently feels the smooth penalty force from
`obstacle.py:sphere_contact_energy_and_grad` based only on its own
distance to the sphere center, every gradient step, same as any other
energy term. That is a reasonable, common approximation (it is, in
essence, an implicit contact formulation), and it is exactly why the
droplet visibly sticks to and wraps around the sphere -- but it's a
per-vertex force field, not a resolved geometric contact.

That matters for the distortion you saw: with the mesh's ORIGINAL
resolution (`MeshConfig.n_rings`), a handful of vertices carry
the entire collision response for the whole contact patch, so their
triangles get stretched far beyond their neutral shape -- that's the
faceted/streaky look. **Adaptive remeshing is implemented** to address
this, as two complementary passes:

- **Split** (`mesh.py:subdivide_mesh`): once the mesh's longest edge
  exceeds `remesh_trigger_ratio` times the original mean edge length,
  every triangle is split into 4 (GLOBAL -- the whole mesh gets finer,
  not just the contact patch), capped at `remesh_max_count` times
  (16x the original triangle count by default).
- **Collapse** (`mesh.py:collapse_short_edges`): once too large a
  fraction of edges (`remesh_min_edge_fraction`) have become shorter
  than `remesh_min_edge_ratio` times the original mean, short edges
  away from the obstacle are merged back down -- pruning the extra
  resolution a split added everywhere it *isn't* needed anymore. It
  never touches edges within `2.5x ObstacleConfig.radius` of the
  sphere's CURRENT position, so resolution stays high exactly where
  the obstacle is still active. Together with the (global) split, this
  gets you something close to local/adaptive resolution without
  implementing a fully local scheme: split adds detail everywhere,
  collapse takes most of it back except near the obstacle. A run
  typically settles at a stable resolution once collapsing stops
  finding anything safe left to remove (logged as "no safe collapses
  found" -- this reliably happens at the mesh's apex, a structurally
  short-edged point that's protected from collapsing by the same
  manifold-safety checks that make it safe everywhere else).

**Two honest limitations remain:**

- Collapse is spatially local (protected near the obstacle), but split
  is still global -- there's no cheap way to split ONLY the contact
  patch without risking T-junctions (mismatched edges where one
  triangle is refined and its neighbor isn't); see "Next steps".
- Neither operation can fix a region that already **folded or
  self-intersected** under a strong enough hit -- both are
  topology-preserving (split only ever subdivides, collapse only ever
  merges within its manifold-safety checks), they don't untangle
  anything. If you still see a small tangled knot right at the contact
  point (most visible with a stiff/fast collision), that's what's
  happening: increase `MeshConfig.n_rings` (more starting
  resolution before any deformation happens), lower
  `ObstacleConfig.collision_stiffness`, or slow the sphere down
  (`center_start`/`center_end` closer together, or spread over more of
  `SolverConfig.n_iterations`) to reduce how hard any single region
  gets hit.

A genuinely rigorous version of this would need real collision
detection (mesh-vs-sphere intersection, not per-vertex distance) -- see
"Next steps" below.

**Local splitting (Rivara refinement) is implemented** -- see the section
"Rivara local refinement" further down. Background: a *global* split
(`subdivide_mesh`) cuts every triangle into 4 whether it needs it or not,
because that is trivially T-junction-free. Cutting only the stretched
triangles creates hanging nodes wherever a refined triangle borders an
unrefined one; Rivara's longest-edge bisection avoids them by also refining
the neighbor across the bisected edge (propagating a step or two), so the
mesh stays conforming without touching the rest.

**A second source of the same visible distortion, independent of
triangle SIZE:** splitting and collapsing only ever act on edge
*length*, but the sphere shears the surface under it as it moves,
which stretches triangles into thin, obtuse slivers even when their
edges are still within the normal length range -- length-based
remeshing alone doesn't catch that. To address this specifically, a
**tangential mesh smoothing** pass now runs once after every
remesh event (`mesh.py:tangential_smooth`; the solver additionally runs the
same pass every 10 iterations, see "Mesh quality"): each interior vertex is
nudged partway toward its neighbors' centroid, but *only* in the
direction tangent to the surface there (the component of that pull
along the vertex's own normal is subtracted out first). That
straightens out sheared/skinny triangles and equalizes triangle
angles without materially changing the surface's actual shape, area,
or volume -- unlike plain Laplacian smoothing, which would also flatten
out genuine curvature. It's a cheap, purely cosmetic/mesh-quality
cleanup step (not part of the physics), so it only runs at remesh
events, not every iteration.

## Rivara local refinement (`remesh_split_mode = "local"`, default)

`mesh.refine_rivara` splits only where needed: a triangle with an edge above
its target length gets its **longest edge bisected**; if the neighbor across
that edge has a different, longer longest edge, the neighbor is refined first
(propagation only walks toward longer edges, so it terminates -- ties are
broken by vertex indices for a strict edge order). The result is conforming
(no T-junctions), geometry-exact (area and volume unchanged) and local. Target
lengths: `remesh_local_target_ratio * L0` everywhere, and the finer
`remesh_contact_len_ratio * L0` within two sphere radii of the sphere; it also
triggers proactively when the sphere approaches coarse triangles.
`"global"` keeps the old uniform 1-to-4 split.

Effort: about 120 lines, no new dependencies. Measured on the default
experiment (2500 iterations, off-screen, wetting angle 90 deg, adhesion off,
this machine; live charts on):

| mode   | time before / after run-time optimization | final triangles | remeshes |
|--------|-------------------------------------------|-----------------|----------|
| local  | ~34 s / ~15 s (~23-30 s with the closed base) | ~3 600 dome + ~900 base | ~41 |
| global | ~27 s / ~9 s (~20 s with the closed base)     | ~3 100 dome + base      | 10       |

(The older table that used to be here -- ~16 s, ~2 750 triangles -- predates
the sphere path/hold/start-height changes and no longer applies.)

**Run-time profile and optimizations.** Profiling the default local run
showed ~49 % of the time in remeshing (`collapse_short_edges` ~30 %,
`refine_rivara` ~19 %) and ~41 % in the solver (of which the wetting-energy
finite-difference gradient was ~15 %). What was changed, all without
changing what is computed:

- `solver.py`: the wetting energy/gradient is skipped when
  `kappa_sphere = cos(theta_Y) = 0` (sphere contact angle 90 deg, the
  default) -- it is identically zero then. **Only at 90 deg**: for any other
  angle the finite-difference gradient is still evaluated (a closed-form
  gradient would be the next step). The three coordinate columns are solved
  with one Sobolev-solve call; the trust region's "shortest incident edge"
  uses a precomputed `reduceat` instead of two `np.minimum.at` calls.
- `energy.py`/`mesh.py`: `np.cross`/`np.linalg.norm`/`np.add.at` on small
  arrays (~100k calls per run) replaced by equivalent `cross3`/`row_norm`/
  `bincount` versions; `_all_edge_lengths` is vectorized.
- `mesh.py:collapse_short_edges`: candidates are found vectorized, the
  per-vertex triangle incidence is maintained instead of scanning all
  triangles for each candidate, and the flip test of one collapse is done
  in one vectorized step. `refine_rivara` pre-scans all triangles for
  too-long edges in one vectorized (conservative) pass and only runs the
  scalar test on flagged or already re-cut triangles.

Checked against the previous code: on a fixed mesh (remeshing off) area and
volume agree to ~1e-16 relative and vertices to ~1e-13 (90 deg; ~1e-9 at 60
deg); `refine_rivara`/`collapse_short_edges` return bit-identical meshes for
identical input; global mode reproduces the previous result exactly. **Local
mode with remeshing is not reproducible to the last bit across such
changes**: the start mesh is very symmetric (many edges of equal length), so
which of several equal-length edges is collapsed first depends on rounding
noise. Perturbing the *unmodified* old code's start mesh by 1e-12 already
changes the number of remeshes (35 instead of 36) and the final area by
~0.01 %. Physically the results are equivalent (final area 9341-9343,
contact radius ~31.9).

Local mode wins on triangle count at the contact (finer, better-shaped
triangles exactly where the sphere is) at a comparable, sometimes even
shorter, run time -- see "Contact-patch collapse" below for what changed
this from an earlier, much less favorable measurement (~12 200 local vs.
~6 200 global triangles, local *slower*). Global mode still ends up with
fewer triangles overall because it coarsens the ENTIRE mesh whenever
possible, not just the fine trail behind the sphere. With
`remesh_contact_len_ratio = 1.0` (coarser contact target) local mode is
smaller still. Where local mode's advantage grows is bigger, longer runs,
because a global split multiplies *all* triangles by 4 every time
(`remesh_max_triangles` caps the local mode instead).

## Contact-patch collapse: keeping contact triangles from shrinking forever

Splitting alone only ever adds resolution; something has to prune it back,
or the mesh only grows. `collapse_short_edges` has always excluded a
`protect_radius` around the sphere from collapsing at all, on the
assumption that triangles there are deliberately fine and should stay
that way. That assumption doesn't hold: as the wetted patch's contact
line moves, competing forces (the wetting energy pulling it one way, the
liquid/vapor area term resisting) can pinch some of its edges far below
the deliberate contact-refinement target -- down to a fraction of a
percent of the mesh's overall scale -- and the blanket protection meant
those edges could never be pruned again, even long after the sphere had
moved on. That both wasted triangles (they piled up over the whole run)
and hurt performance: the trust-region step cap (`max_step_fraction`)
shrinks with the shortest incident edge, so a pathologically tiny edge
effectively freezes its vertex.

The fix: `collapse_short_edges` takes an additional `protect_min_len` --
inside the protected radius, an edge still collapses if it's shorter than
this SEPARATE (much smaller) threshold, currently 40% of the contact
target length. This is checked after every local split (not only when a
split ISN'T needed, which -- since local mode's split trigger fires
almost continuously while the sphere is active -- would otherwise rarely
get a turn); local mode also gets its own dedicated trigger
(`mesh.short_edges_near`) for the case where the mesh-wide short-edge
fraction never crosses the global collapse threshold even though a few
edges right at the contact have (which is the usual case, since local
refinement, being local, rarely moves that global fraction). Each pass
is repeated (up to 4x) because a single collapse pass freezes every
vertex it touches to avoid cascading through a whole neighborhood at
once, which otherwise leaves some collapsible edges untouched simply
because a neighbor was already frozen that round.

Effect on the default experiment: local mode's final triangle count
roughly halved again after adding the repeat passes (from ~6 000 to
~2 750), and the smallest edge in the wetted patch went from ~1e-4
(essentially degenerate) to consistently within a factor of a few of the
intended target. A few structurally locked slivers can still remain --
`collapse_short_edges` is deliberately conservative (it never collapses
an edge if doing so would flip a triangle or create a non-manifold
vertex, see its docstring), so a small number of very short edges can be
topologically un-collapsible without a bigger operation like an edge
flip -- but this is no longer the common case.

## Mesh quality: disk mesh and tangential smoothing

- `MeshConfig.mesh_type = "disk"` (default) builds the cone on concentric
  rings with 6k vertices on ring k: valence 6 nearly everywhere. The old polar
  grid (`"polar"`) ends in a fan of `n_phi` skinny triangles at the tip. The
  area flow piles ~60 vertices within 1e-3 of that tip, which starved the
  trust region (the apex relaxed extremely slowly) and, worse, is exactly
  where the sphere first touches: refinement there added dozens of vertices
  per layer.
- `SolverConfig.tangential_smoothing_every` (10) / `_strength` (0.3): area
  minimization only fixes the *normal* motion; the tangential part of the
  discrete flow is an artifact. A cheap tangential-only averaging pass every
  10 iterations keeps triangles evenly sized. The apex now reaches
  equilibrium in ~500 instead of >2500 iterations (top z: 1.53 at it 250 vs.
  1.90 before).

## Why can the droplet vanish?

The window draws whatever coordinates the solver returns. NaN, inf or huge
values render as *nothing* -- the droplet "disappears" and numpy prints
`RuntimeWarning: overflow / invalid value` (in `energy.py` the division by a
zero triangle area, in `np.zeros_like`/`dot` the already-huge numbers). Those
warnings are symptoms; the crash happened earlier. Typical chains:

1. **A force term that does not shrink with the mesh.** Collision, adhesion,
   volume and wetting forces are independent of the edge length. On a refined
   mesh one explicit step can carry a vertex several edge lengths, triangles
   flip, areas go through zero, the volume penalty explodes.
   *Guard:* `SolverConfig.max_step_fraction` (trust region: no vertex moves
   more than 30 % of its shortest incident edge per iteration).
2. **Degenerate triangles** (zero area -> division by zero in the area
   gradient). *Guard:* area is clamped (`1e-14`).
3. **A buggy wetting energy** (was the cause of the reported crash): it
   depended on the loop orientation and used non-normalized vectors, so
   forces were 10-20x too large. Fixed (see "Capillary bridge").
4. **The sphere goes through the liquid.** A single-sheet mesh cannot
   represent a sphere that enters the droplet; the sheet is wrapped and
   stretched until it folds. *Guard:* if the wetted patch reaches around the
   sphere by more than 115 deg half-width the run stops with an explanation.
   Use a path that touches the surface (the default). A hydrophilic sphere
   (theta_Y < 90 deg) pressed deep into the liquid gets wetted beyond its
   equator (contact line at 180 deg - theta_Y for a flat interface), which
   also hits this limit.
5. **Adhesion too strong / sphere meets the unrelaxed cone tip.** The pull
   drags the spike around and snaps a band of vertices onto the sphere.
   Keep `adhesion_strength` at its default 0 (or small) and the quasi-static mode equilibrates the droplet at `center_start` before the sphere moves.
6. **Volume out of range** (volume outside 0.5-1.5 x target: the surface has
   folded or torn).

Since this version the solver never yields non-finite coordinates: it raises
`SolverDiverged` with the iteration, the reason and the maximum force of each
term, main.py prints it, and the window keeps showing the **last valid
state** instead of an empty scene.

## Is the computation deterministic?

Yes: for the same config, the same run (off-screen or live) produces
bit-for-bit identical areas, volumes, contact radii, and remesh
events every time -- verified by re-running the full default scenario
(`remesh_split_mode = "global"`, 2500 iterations) twice and diffing
the console output; the only difference between the two runs is the
wall-clock timestamp in an unrelated VTK log line, not anything the
solver computed. Nothing in the physics loop draws from a random
source (no `random`/`numpy.random`, no seed anywhere in the project),
and there's no threading or multiprocessing in the solver or the
mesh operations that could make results depend on scheduling. The two
places that might look suspicious at a glance turn out not to matter:
- The sparse linear solve behind the Sobolev preconditioning
  (`scipy.sparse.linalg.factorized`, Section "Sobolev Preconditioning"
  in the math document) is a **direct** sparse LU factorization
  (SuperLU), not an iterative or randomized method -- it always
  produces the same result for the same matrix.
- A few places build a Python `set` (e.g. `energy.py:build_laplacian`'s
  edge collection). Python's hash randomization (`PYTHONHASHSEED`)
  varies string/object hashing between runs, but every `set` in this
  project holds plain **integers**, whose hash is the integer itself
  (`hash(n) == n` in CPython) -- so iteration order there is fixed
  regardless of `PYTHONHASHSEED`, and isn't a source of nondeterminism
  either.

What genuinely does change the result, deterministically: any change
to the config, the installed package versions (NumPy/SciPy/VTK point
releases can shift floating-point rounding at the last bit), or the
platform's underlying BLAS/LAPACK -- none of which is randomness, just
a different well-defined computation.

## Background

The method was determined, in a preceding comparison (plain gradient
descent vs. Sobolev gradient descent vs. L-BFGS vs. Newton-CG), to be
the best trade-off between implementation effort and convergence
speed. The curvature color is derived directly from the area gradient
(which is needed for the optimization step anyway), via the
relationship

    grad_A(v_i)  ~=  2 * H(v_i) * n(v_i) * A_mixed(v_i)

This relation only holds exactly for interior vertices with a complete
triangle fan. Boundary vertices (the contact line) have an incomplete
fan, so their raw curvature estimate is spuriously inflated (picking
up a contribution related to the boundary curve's own geodesic
curvature, not the true surface curvature) -- in early versions of
this project this showed up as an artificially bright rim in the
visualization. `energy.py:mean_curvature` now replaces each boundary
vertex's displayed value with the average of its interior neighbors to
correct for this (see `_replace_boundary_curvature`).

**Why color by mean curvature specifically, and what it tells you:**
by the Young--Laplace law, a liquid surface at true equilibrium (no
gravity, uniform surface tension) has *constant* mean curvature
everywhere -- the pressure jump across the surface, `deltaP = 2*gamma*H`,
has to be uniform, or the surface would still have a net force pushing
it to move. So a perfectly flat, uniform curvature color is a direct,
visual readout of "how converged is this shape" -- much more
informative than watching the outline alone, since two visually
similar blobs can be at very different distances from equilibrium.
Bright/hot regions (high |H|) are exactly where the surface is most
sharply bent and therefore where the AREA GRADIENT -- the thing the
optimizer actually descends -- is largest: since `grad_A ~ 2*H*n*A_mixed`
(the same relation used for the coloring), the curvature map doubles
as a preview of where the mesh will move most in the next few steps.
That's also why remeshing distortion and curvature hot spots line up:
wherever the surface is forced into a sharp bend (like the still-not-
fully-relaxed cone tip, or the sphere obstacle's contact dimple), both
the curvature and the local mesh distortion spike together.

## Capillary bridge: genuine contact on the sphere

Beyond the soft collision/adhesion penalty (`obstacle.py`), the droplet forms
a **real, dynamically tracked wetting contact** with the sphere, with its own
Young's-equation energy (`wetting.py`, on by default,
`ObstacleConfig.wetting_enabled`).

**Default experiment.** The sphere starts at `center_start` = (0, 0, 125), the droplet
equilibrates there, and the sphere then moves in steps along the x-axis to `center_end` = (20, 0, 125)
(quasi-static, see above); it wets the apex and drags the contact line over the
substrate. Choose positions that
*touch the surface* -- see "Why can the droplet vanish?".

**Wetted patch and contact line.** Every iteration, vertices within
`wetting_snap_distance` of the sphere are candidates; the largest connected
group is the wetted patch, and its boundary -- an ordered loop of vertices --
is the three-phase contact line (drawn in green; scrubs with the history
slider). Wetted vertices are held exactly on the sphere by projection.

**The energy, and a bug that was fixed.** The wetted area on the sphere
enclosed by the loop is `R^2 * Omega`, where the solid angle `Omega` is
`2*pi - sum(turning angles)` of the loop (discrete Gauss-Bonnet / holonomy;
the gauge-free equivalent of the Wu-Yang monopole potential -- no global
vector potential is needed, only one that is valid on the patch). The
energy is `E = -kappa * R^2 * Omega`, `kappa = cos(theta_Y)`
(`wetting_contact_angle_deg`).

An earlier version *also* kept charging liquid/vapor surface energy for the
mesh triangles lying on the sphere. That is double counting: those facets
are no longer liquid/vapor interface, they are solid/liquid contact. The
effective coefficient was `kappa - 1` instead of `kappa`, so the contact
angle came out >= 90 deg regardless of the setting. Now every triangle whose
three vertices are all wetted gets **area weight 0** ("energetic opening").
The triangles stay in the mesh (volume closure, reversibility when the
sphere leaves), but energetically the mesh is open there and the contact
line is an explicit boundary. Measured contact angles (sphere pressed into a
droplet, equilibrium):

| set theta_Y | measured (new) | measured (old) |
|-------------|----------------|----------------|
| 60 deg      | 60.1 deg       | 116.8 deg      |
| 90 deg      | 89.5 deg       | 156.2 deg      |
| 120 deg     | 117.9 deg      | unstable       |

**Does the mesh literally open a hole?** Not topologically. Energetically it
does (zero weight on wetted facets + explicit contact-line boundary), which
is what changes the physics. A literal annulus (second boundary ring, like
Surface Evolver) would additionally need: a ring constraint that follows the
moving sphere, volume closure through the hole (via
`int u dOmega = 1/2 oint u x du`), ring splitting/merging, a release
criterion for closing the hole again (depinning, hysteresis: advancing vs.
receding angle), and pinch-off topology changes. The same reasoning applies
to the wall at z = 0: there the contact line is already an explicit boundary
ring (the mesh is closed there by the flat base, and its Young energy is the exact
-cos(theta)*gamma_lv*A_sl term, equal to the former shoelace integral), so nothing is double counted.

**Contact-line mobility and the contact angle (measured, root cause found).**
For a sphere angle of 120 deg the static contact angle at the sphere was only
~100 deg (wall: ~117 deg; 90 deg was fine, 89.5). Not a speed effect: halving
the sphere speed changes the radii by < 1 %, and holding the sphere still does
not move the angle. The cause was `wetting_snap_distance`: main.py scales it
with the mesh (x53.8), so the old default 0.06 became ~3.2 -- MORE than one
edge length (~2.3). The first row of dome vertices next to the contact line
was therefore re-captured onto the sphere in every iteration: the line could not
recede and the Young force (which was correct: ~1.02 per loop vertex measured vs.
1.02 expected) was balanced by the capture instead of by the dome tension. With
the new default 0.01 (~0.54 scaled) the static sphere angle is 119-120 deg
(0.003 gives the same), the wall angle 118 deg (the 2 deg remainder is the
first-row measurement bias), and 90 deg is unchanged. Keep the scaled snap
distance well below the edge length.

What remains: (i) the wetted vertex set is still held by a *bilateral* constraint
(vertices are projected onto the sphere and never detach on their own; without
remeshing the set stays constant while the line slides inward by tangential
vertex motion, with remeshing it shrinks), a unilateral release criterion would
be the physical replacement; (ii) at 120 deg the wall ring is still ~4 % larger on
the way up than on the way down at the same sphere height, independent of the
sphere speed (2500 vs 5000 iterations) -- candidates are the soft volume penalty
(volume differs by path) and mesh effects; not yet isolated.

**Numerical (non-physical) ingredients:** Sobolev preconditioning and the
trust region (change the path, not the minimum); periodic tangential smoothing
(30 % every 10 iterations, not derived from the energy); remeshing; the soft
volume penalty (volume drifts 2-3 % from V0 during contact); soft collision and
substrate-penetration penalties (absolute stiffnesses, not scaled with
`gamma_lv`); the snap distance and minimum patch size for wetting; the
wrapping guard; the 10-point averaging of the plotted force; the pseudo-time of
the gradient descent (no inertia, viscosity or contact-line friction, so
"speed" is only the schedule of the sphere); adhesion (off by default).

**Known simplifications:**
- No contact-angle hysteresis: membership is geometric (with a sticky
  projection), no advancing/receding distinction.
- The gradient of the wetting energy is a finite-difference gradient
  (vectorized with stencil coloring: cost ~ 6 evaluations, independent of
  the loop length). Verified against a brute-force finite difference.
- Directions are normalized onto the unit sphere before the turning angles
  are computed, so the energy is orientation-independent (an earlier
  version was not, which produced forces 10-20x too large and was one cause
  of vanishing droplets).
- Only one connected contact patch is tracked (the largest).
- Facets inside the wetted patch are kept regular by a neighbor-averaging
  step (mesh quality, not physics).

## History slider

After the relaxation finishes (`live=True`, `off_screen=False`), a
slider appears at the bottom of the window (in the place of the progress bar, labelled
"Tip x-coordinate") that scrubs through the whole run -- drag it to revisit any earlier frame (every
frame that was recorded -- in quasi-static mode one per converged sphere step -- is kept
in memory; disable with `VisConfig.enable_history_slider = False` to
save that memory on a very long/high-resolution run instead). Its
numeric readout is the tip x-coordinate (the iteration number in the old
iteration-driven mode), snapped to the nearest recorded frame. It also correctly replays across a remesh
event (the mesh's vertex/triangle count is swapped in and out as
needed while you scrub), so you can rewind from the final, remeshed
shape all the way back to the original coarse cone. It also carries
both diamond markers (energy and its derivative) along with it, so the
frame under the slider and the points on both charts always agree.

A **Play** button sits at the bottom-left of the window, next to the
slider: click it to step through the recorded history automatically
(`VisConfig.playback_interval_ms`, 90 ms/~11 fps by default) instead of
dragging by hand, for a fairly smooth "replay" of the whole run. Click
again to pause; it also stops (and un-highlights) on its own once it
reaches the end, and clicking Play again after that restarts from the
beginning. Most played frames are a cheap in-place point update, but a
frame that lands exactly on a remesh event swaps in a new mesh/actor
instead (see `replace_mesh`) and is noticeably more expensive, so very
fast playback can still show a brief hitch right at those frames.

**Performance.** Play used to make the window feel like it nearly
hangs, especially on a long, densely-recorded run. Three separate
things were tightened up:
- `energy.py:_replace_boundary_curvature` (the fixup that replaces
  each contact-line vertex's spuriously large curvature with the
  average of its interior neighbors, see "Background" below) used to
  rebuild a full Python-loop adjacency graph over the WHOLE mesh from
  scratch on every single rendered frame -- including every played
  frame, since it runs inside `mean_curvature`, called by
  `update_mesh` on each Play tick. It's now vectorized with
  `numpy.add.at` instead of a per-triangle Python loop (verified
  numerically identical, ~5x faster at a few thousand triangles).
- Each frame's curvature is now also cached once, right when it's
  first computed during the live run, and stored alongside that
  frame's mesh/sphere-position data. Scrubbing or playing back through
  history reuses that cached array instead of recomputing curvature
  for a mesh that hasn't actually changed (only remesh-boundary frames
  still need a fresh recompute, since those get a whole new mesh/actor
  anyway).
- Play no longer necessarily visits every single recorded frame:
  `VisConfig.playback_max_frames` (120 by default) caps one full
  playback pass to at most that many evenly-spaced frames, subsampling
  the recorded history if it's denser than that. This bounds the total
  work of pressing Play to roughly the same amount regardless of how
  long or densely the run was recorded, instead of scaling with
  however many frames happened to accumulate.

Right-drag pan keeps working once the slider appears, and while it's
being used -- both were reported broken at one point. The "after the
slider appears" half traced back to PyVista's `plotter.show()`
re-applying its own tracked interactor style internally (via
`iren.update_style()`) every time it's called, silently overriding a
style that had only been set directly on the raw VTK interactor
(`.interactor.SetInteractorStyle(...)`); setting it through PyVista's
own tracked `plotter.iren.style` property instead fixed that specific
path, but apparently wasn't the whole story -- something else (most
likely PyVista's automatic interactor-style swap for interacting with
an on-screen chart, since the energy/derivative charts cover a good
chunk of the window) could still knock the style out from under a
one-time assignment. Rather than chase down every possible trigger
individually, `visualize.py` now also installs a small self-healing
`RenderEvent` observer that re-applies the pan style on every single
render if something else has swapped it out -- a cheap identity check
per frame, correct regardless of the specific cause.

**The full mathematical derivation** -- from the continuous variational
problem, through the explicit derivation of the discrete area/volume
gradients, to the Sobolev preconditioning, with a table mapping each
equation to its corresponding line of code -- is in
`docs/mathematical_background.pdf` (or compile
`docs/mathematical_background.txt` yourself with a LaTeX toolchain;
see below).

## Remeshing: frequent vs. rare (measured), and the parameters

Default scenario, local mode, 2500 iterations, off-screen (times measured with
three runs in parallel, so only the ranking is reliable, not the absolute values):

| setting | remeshes | time | final dome triangles | triangle quality median / 5th pct / worst | final A_lv |
|---|---|---|---|---|---|
| remeshing off | 0 | ~8 s | 726 | 0.70 / 0.40 / 0.30 | 9339 |
| rare (`remesh_check_every` = 300) | 8 | ~28 s | 3946 | 0.92 / 0.51 / **0.025** | 9371 |
| default (60) | 41 | ~30-37 s | 3678 | 0.96 / 0.77 / 0.25 | 9372 |
| frequent (15) | 159 | ~48 s | 3500 | 0.95 / 0.75 / 0.45 | 9392 |

(quality = 4*sqrt(3)*area / sum of squared edge lengths; 1 = equilateral.)

**Frequent:** the mesh never gets far from the desired edge length, so there are no
degenerate triangles (best worst-case quality), the contact line is resolved
early, and the contact angle estimate is cleaner. Costs: every remesh strips and
rebuilds the base, re-factorizes the Sobolev operator, and each event perturbs
the shape slightly (vertex insertion/collapse and tangential smoothing; the
following iterations relax it). Here ~60 % more run time than the default. Local remeshing is also not
bit-reproducible (tie-breaking).

**Rare:** cheaper and fewer perturbations, but the mesh is stretched or
squashed for hundreds of iterations between events: in the table the worst
triangle has quality 0.025 (a sliver), the first rows next to the sphere get
distorted (the effect you saw as triangles "pulled" up to the contact line),
and the contact patch is resolved too coarsely to follow a moving contact line.
Too rare can end in a folded mesh/`SolverDiverged`.

**Off:** fastest (8 s) but only 726 triangles; fine for the free droplet, too
coarse once the sphere deforms it.

Parameters (`MeshConfig`, all lengths are ratios of the ORIGINAL mean edge L0 and
therefore scale automatically):

- `remesh_enabled` -- master switch.
- `remesh_check_every` (60) -- how often (iterations) the triggers below are
  evaluated; the main "frequent vs. rare" knob. Between checks the mesh is
  never changed.
- `remesh_trigger_ratio` (1.8) -- split when the longest dome edge exceeds this
  times L0. NOTE: in the current default scenario it has no effect (identical
  results for 1.2 and 3.0), because the sphere is near the surface from the
  start and the proactive contact trigger below fires anyway; it matters for
  scenarios without a nearby sphere.
- `remesh_split_mode` -- `"local"` (Rivara longest-edge bisection of only the
  too-long triangles plus neighbours; cheap, can refine around the sphere) or
  `"global"` (uniform 1-to-4 split of every triangle, capped by
  `remesh_max_count`; triangle count x4 per split).
- `remesh_local_target_ratio` (1.0) -- local mode: bisect until no edge is longer
  than this times L0 (away from the sphere).
- `remesh_contact_len_ratio` (0.7) -- local mode: target edge length within ~2
  sphere radii of the sphere (finer where the contact patch/line are); also
  triggers a split proactively when a coarse region approaches the sphere.
- `remesh_max_triangles` (60000) / `remesh_max_count` (2) -- safety caps for
  local / global mode.
- `remesh_min_edge_ratio` (0.35) -- edges shorter than this times L0 are
  collapse candidates (undoes over-refinement; 0 = never collapse).
- `remesh_min_edge_fraction` (0.30) -- a global collapse pass is only triggered
  when at least this fraction of all edges is short (the apex "spokes" are
  always short and must not trigger it). Local mode additionally has a
  contact-only collapse trigger for tiny edges at the contact patch.
- Related (`SolverConfig`): `tangential_smoothing_every`/`_strength` -- periodic
  tangential smoothing between remeshes (mesh quality without changing the
  shape); `max_step_fraction` -- trust region that keeps vertices from
  overrunning their neighbours between remeshes.

Practical rule: use the default; go to 30 when you see distorted triangles at
the contact line (costs time), to 120-300 only for quick previews.

## Centre of mass, saving a run, Numba

- **Centre of mass** (`energy.py:center_of_mass`): the liquid's centroid from the
  closed mesh via the divergence theorem (signed tetrahedra with the origin; the
  base lies in z = 0 and contributes nothing, so it is exact for the closed mesh).
  Shown bottom-left as `COM = (x, y, z)`, with the xy offset of the COM relative to the sphere centre `COM offset to sphere (xy) = d (dx, dy)` below it, d = sqrt(dx^2+dy^2), (dx, dy) = COM_xy - sphere_xy (updated live and while scrubbing/Play).
  Check: the start cone gives z = 10.72 vs. h/4 = 10.76 for an ideal cone.
- **Save button** (next to Play, available after the run): opens a "Save as"
  dialog (tkinter; falls back to zenity/kdialog, or to `./runs/<timestamp>` if no
  dialog is installed -- `sudo apt install python3-tk` for the dialog) and writes
  a folder with `config.py` (copy), `config_effective.json` (values actually used,
  incl. the automatic mesh-scale adjustments), and `.npy` files `iteration`,
  `gap` (z of the sphere centre minus the sphere radius = height of the sphere's lowest point above the substrate, as plotted in the "Gap" chart), `energy`
  (Surface free energy), `force` (N x 3, columns x, y, z; as plotted), `force_raw` (N x 3, unaveraged), `force_parallel` (N, projection on the motion direction), `force_from_energy` (N, old estimate), `com_offset` (N, distance d of the COM from the sphere centre in the xy-plane) and `com_offset_xy` (N x 2, (dx, dy)) and
  `sphere_center` (N x 3), and a video `run.mp4` of the replay (window content incl. charts, info block and markers; `VisConfig.save_video`, `video_max_frames` = 600, `video_fps` = 30; needs `imageio` + `imageio-ffmpeg`, installed via requirements.txt -- re-run `./run.sh`; falls back to `run.gif` without ffmpeg; recording takes about as long as playing that many frames). One entry per recorded frame (one per converged sphere step; with `QuasiStaticConfig.enabled = False` every `render_every` iterations and at remesh events); plus `frames.csv` (quasi-static mode). Code: `export.py`.
- **Chart x-axis:** fixed to the planned sphere path (tip x-coordinate); the diamond marker of each curve sits on the newest data point (the force chart's y-range always includes it). One diamond per curve (energy; Fx/Fy/Fz), in the colour of its curve.
- **Window only at the end (`VisConfig.live_mode = "end_only"`, default `"live"`):** the run is computed without any window; the same frames are recorded (quasi-static: one per converged step), and when the run is finished the window opens with the full charts, the final shape and the history slider/Play/Save. With software OpenGL the live window costs about as much as the computation itself (6000 iterations: 126 s live vs. 62 s without window; on a GPU the difference is smaller). Needs `live=True` and a display; with `off_screen=True` it behaves like `live=False`. In this mode the terminal shows a one-line progress bar (fraction, iteration, elapsed time, ETA, number of remeshings) instead of the `[... collapse ...]` lines.
- **Rendering cost:** the contact-line overlays used to trigger one extra full render each per displayed frame (pyvista `add_mesh` renders by default); they now use `render=False`. Measured with software OpenGL (no GPU): 887 -> 419 renders, live run 120 s -> 75 s. Further levers: `VisConfig.render_every` (50 instead of 10 -> 68 s in the same test), `window_size`, `save_video`/`video_max_frames`.
- **Speed-ups of the computation (bit-identical results):** (1) `mesh.py:collapse_short_edges` no longer fills Python sets for every vertex/edge/triangle of the mesh; connectivity is built once as numpy CSR arrays and per-vertex sets are created lazily only for the few vertices a candidate edge touches (a pass takes ~0.6 s instead of ~3.5 s per 168 recorded calls in local mode, 0.6 s instead of 1.9 s per 47 calls in global mode; outputs identical on all 215 recorded real calls). (2) The undirected-edge lists (`wetting.unique_edges`, edge lengths, substrate base) use one integer key per edge instead of `np.unique(axis=0)` (3-4x faster, identical output). Whole default run (2500 iterations, old default path): 25 s -> 19-21 s with identical final area, volume and contact radius (9371.78373, 127728.97620, 32.75489) and the same 41 remeshes. Measured but NOT worth it (reverted): skipping the wetting computation (on the default path the sphere is in contact with a ~450-vertex patch in ~85 % of the iterations, so there is nothing to skip) and a sorted-key variant of the boundary-loop extraction (no gain).
- **Numba (tested, not enabled):** the vectorized area and volume kernels are
  ~9x faster as compiled loops (area+gradient 0.83 -> 0.09 ms, volume+gradient
  0.60 -> 0.06 ms on a 4000-triangle mesh, results identical to 1e-15), but they
  are only ~15 % of the run time: the whole default run went from ~25 s to
  ~21-23 s (10-15 %), JIT compile ~1-1.5 s (cached after the first run). The
  rest is Python-structure code (`collapse_short_edges` ~21 %, contact-loop
  extraction ~7 %, SuperLU, sorting) that would need a larger rewrite;
  a full Numba port might reach ~1.4-1.5x in total. Costs: a heavy dependency
  (numba/llvmlite), cache files, slightly different round-off (and thus a
  different chaotic remesh trajectory).

## Troubleshooting

### OpenGL/VTK window errors on Linux (native desktop)

If `python main.py` (or `./run.sh`) prints errors such as
`Could not find a decent config`, `Could not find a decent visual`,
or crashes right after startup with a segmentation fault, this is a
VTK/OpenGL issue, not a bug in this project. VTK tries a real
GPU-accelerated GLX context first, then falls back to EGL, then to a
software renderer (OSMesa) -- and crashes if that last fallback
library is missing instead of failing gracefully.

Run these three diagnostic commands and check the output:

```bash
echo $XDG_SESSION_TYPE
sudo apt install -y mesa-utils libosmesa6
glxinfo | grep "OpenGL renderer"
```

- If `$XDG_SESSION_TYPE` prints `wayland`, that is very likely the
  cause: VTK's GLX window needs an X11 session. Switch to an
  "... on Xorg" session at your login screen (usually via the gear/
  settings icon) and try again.
- The `apt install` command installs the missing `libosmesa6` software
  fallback so the program at least runs (via software rendering)
  even if GPU acceleration isn't working.
- `glxinfo | grep "OpenGL renderer"` shows whether your system has a
  working OpenGL driver at all, outside of Python. If this doesn't
  return a real renderer name, the issue is at the graphics-driver
  level (e.g. a missing proprietary NVIDIA driver), independent of
  this project.

**If you have Conda installed and your shell prompt shows `(base)` or
another environment name**, that is very likely the actual cause, even
if the checks above look fine (X11 session, a real GPU renderer, and
`libosmesa6` installed). Conda sets `LD_LIBRARY_PATH` to point at its
own bundled libraries (its own `libstdc++`, X11 libraries, etc.),
which can shadow the system's Mesa/OpenGL libraries for any process
started from that shell -- including a Python process running inside
an unrelated venv, since it still inherits the shell's environment
variables. `run.sh` already detects this automatically (it unsets
`LD_LIBRARY_PATH` for its own process only, your shell/Conda
environment is left untouched) and prints a note when it does so. If
you're running `python main.py` directly instead of `./run.sh`, either
run `conda deactivate` first, or launch with
`env -u LD_LIBRARY_PATH python main.py`.

**Confirmed fix for the `(base)` case:** the venv must not be built on Conda's
Python (it then loads Conda's libraries even with `LD_LIBRARY_PATH` unset).
`run.sh` now uses `/usr/bin/python3` whenever a Conda environment is active
(override: `DROPLET_PYTHON=/path/to/python3 ./run.sh`). If an old venv was built
on Conda's Python, delete it once: `rm -rf venv`.

**After a reboot / system update it suddenly stopped working?** The most
common reasons are (a) the login came up as a Wayland session instead of
Xorg, (b) a kernel or driver update left the GPU driver unloaded (then
even `glxinfo -B` fails), or (c) `libosmesa6` is missing so VTK's last
fallback crashes. Quickest test: `DROPLET_SOFTWARE_GL=1 ./run.sh` forces
Mesa's software renderer (slower, but independent of the GPU driver).
`run.sh` also prints a hint if it detects Wayland, a missing `DISPLAY`,
or a failing `glxinfo`.

**All of the above look fine (X11, a real GPU renderer via `glxinfo -B`,
`glxgears` opens and runs hardware-accelerated) and it STILL fails, even
with `DROPLET_SOFTWARE_GL=1` and across different VTK versions?** The
likely cause is then **no multisample (anti-aliasing) GLX FBConfig being
offered at all** -- some setups don't have one (a multi-monitor desktop
combined into a single, unusually large X11 screen is one known trigger),
and VTK's default GLX config search asks for multisampling and, unlike a
plain GLX demo like `glxgears`, does not fall back to a config without it
-- it just fails outright with "Could not find a decent
config"/"...visual", then crashes instead of a clean error. `visualize.py`
now disables multisampling globally before the render window is created
(`vtk.vtkOpenGLRenderWindow.SetGlobalMaximumNumberOfMultiSamples(0)`,
harmless -- you only lose anti-aliasing on the triangle edges), which
fixes this. If you're on an older copy of this project without that line,
either pull the update or add it yourself at the top of
`visualize.py:make_plotter`.

To isolate this from the project entirely and confirm the diagnosis:

```bash
source venv/bin/activate
python -c "import vtk; print(vtk.vtkVersion.GetVTKVersion())"
python -c "import vtk; vtk.vtkRenderWindow().Render()"                          # fails
python -c "import vtk; vtk.vtkOpenGLRenderWindow.SetGlobalMaximumNumberOfMultiSamples(0); vtk.vtkRenderWindow().Render()"  # should work
```

`requirements.txt` used to pin `vtk<9.5` as a precaution against this
crash. With multisampling disabled the pin is no longer needed, so it only
requires `vtk>=9.3` (for `vtkHDFReader`). If a new VTK release still fails
to open a window, `rm -rf venv && ./run.sh` forces a clean reinstall
(`run.sh` only reinstalls when `requirements.txt` is newer than
`venv/.requirements_installed`), and an upper bound can be added back.

### Compiling the math document yourself

`docs/mathematical_background.txt` is the exact same LaTeX source as
the `.tex` file, just with a `.txt` extension so it can be downloaded
and opened as plain text. To compile it into a PDF yourself:

```bash
cp docs/mathematical_background.txt mathematical_background.tex
pdflatex mathematical_background.tex
pdflatex mathematical_background.tex   # run twice for the table of
                                        # contents and cross-references
```

## Next steps (not yet implemented)

- A literal topological opening at the sphere (annulus with a second boundary
  ring), including release/depinning criteria -- see "Does the mesh literally
  open a hole?"
- Edge flips (equiangulation) and collapse of the fine trail behind the
  sphere in local mode
- Real collision detection against the sphere (mesh-vs-sphere intersection
  instead of the per-vertex distance penalty)
- Contact-angle hysteresis for the sphere wetting patch
- An analytical gradient for the sphere wetting energy
- Topology change (pinch-off into a detached droplet)
- Gravity as an additional energy term
