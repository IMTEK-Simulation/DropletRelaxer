# Restructuring plan: solver / file format / viewer / docs

Status: proposal (2026-10-22), decisions of §7 folded in. The VTK pin bump is done. Nothing below is implemented yet except the git
link to `git@github.com:IMTEK-Simulation/DropletRelaxer.git` (local `main`
tracks `origin/main`, which holds only the MIT `LICENSE`; all code is untracked).

Goals: (1) Quarto documentation that stays in sync with the code, rendered to
PDF in CI; (2) split the monolith into a headless solver that runs on a
cluster and a separate viewer, connected only by a file; (3) a package layout
and build system in IMTEK house style (NuMPI, muGrid); (4) leave the door
open for later CUDA + HIP kernels without building any of it now.

## 1. Where the coupling is today

- `main.py` imports `visualize` at module level, so the solver cannot even be
  imported without PyVista/VTK. This is also why four test modules fail in an
  environment without PyVista.
- The solver core is already clean: `solver.py`, `mesh.py`, `energy.py`,
  `substrate.py`, `wetting.py`, `obstacle.py`, `anomaly.py`, `roughness.py`,
  `quasistatic.py` never read `VisConfig`. Only `main.py` does.
- `main.py` mixes three jobs: the remesh/restart loop around `relax()`
  (numerics), the live PyVista context (viewer), and history/export bookkeeping.
- `visualize.py` contains physics post-processing that belongs to the solver:
  `_capillary_force`, `_sphere_path_length`, `_gradient_dedup`,
  `com_xy_offset_to_sphere`. Those values are also what `export.py` saves.
- `VisConfig.render_every` doubles as the frame-recording cadence. It becomes
  an output setting of the solver.
- Configuration exists only as an edited Python file (`config.py`). That does
  not work for job arrays on a cluster.

## 2. Target architecture

```
  config.toml ──► droplet-relax (solver, headless) ──► run.h5 (VTKHDF + /DropletRelaxer)
                                                          │
                     ParaView / any VTK ◄─────────────────┤
                     droplet-view (PyVista viewer) ◄──────┘  (replay, follow live, video)
```

- **Solver**: config in, HDF5 file out. Depends on numpy, scipy, h5py only.
  Writes frames, scalar time series and a restart checkpoint into one file.
- **Viewer**: three modes, one rendering path.
  - `droplet-view run.h5` replays a finished run.
  - `droplet-view run.h5 --follow` watches a file the solver is still writing,
    e.g. on a cluster (HDF5 SWMR mode).
  - `droplet-view live config.toml` runs the solver **in the same process**
    with a live window (see below).

  The history slider works on the file in all three modes. "Save" turns into
  video export from the file. The viewer imports from the solver only the
  driver (for `live`) and `DropletRelaxer.io`. The solver never imports the viewer.
- **Analysis** (forces, centre of mass, gap, COM offset, curvature) moves into
  the solver and is written per frame, so the viewer only plots stored numbers.

### In-process live window (decided: keep it)

**Mechanism: frame sinks.** The driver (today's remesh/restart loop in
`main.py`) knows nothing about outputs. It gets a list of sinks:

```python
class FrameSink(Protocol):
    def start(self, run: RunInfo) -> None: ...   # config, substrate, obstacle, start mesh
    def frame(self, f: Frame) -> None: ...       # every output.every its / each quasi-static step
    def event(self, e: Event) -> None: ...       # remesh, step rollback, divergence
    def tick(self, i: int) -> None: ...          # every iteration; must be cheap
    def finish(self, status: RunStatus) -> None: ...
```

`Frame` is a frozen dataclass in `DropletRelaxer.io`. It holds the iteration,
read-only `verts`, `tris` + topology epoch, `boundary_idx`, `wetted_loop`,
`sphere_center`, the per-frame scalars and the point data. The HDF5 writer is a
sink. The reader yields the same `Frame` objects. The viewer's scene code therefore
takes one input type, whether frames come from the live driver or from a file.

| Command | Package | Sinks |
|---|---|---|
| `droplet-relax run cfg.toml -o run.h5` | DropletRelaxer | `H5Writer`, `ConsoleProgress` |
| `droplet-view live cfg.toml [-o run.h5]` | DropletViewer | `H5Writer`, `LiveWindow` |

In `live` mode the file is always written. The default is
`runs/<timestamp>/run.h5`, which matches today's Save folder. When the run
ends, the window stays open and switches to replay mode on that file, so the
history slider, Play and Save use exactly the code of `droplet-view run.h5`.
This removes the in-memory `history` list, which today grows with run length.

**Threading: none, as today.** The solver runs synchronously on the main
thread. `LiveWindow.frame()` redraws. `LiveWindow.tick()` calls
`plotter.update()` at most every ~100 ms of wall time, so the window can be
rotated or closed between frames. Today's window cannot do that: it only
processes events every `render_every` iterations. The wall-time throttle does
not affect the numerics, because the only thing that flows back to the solver
is the stop flag (next paragraph). Rejected alternatives:
- Solver in a worker thread, GUI on the main thread. This breaks the
  no-threading invariant. VTK must own the main thread. Because of the GIL, it
  gains little.
- Spawn `droplet-relax` as a subprocess and `--follow` its file. This gives a
  responsive window, but it is just the file mode, which exists anyway.

**Stop flag.** Closing the window or pressing Stop sets a `StopRequest`. The
driver checks it once per iteration. It then finishes the iteration, writes a
final frame and the checkpoint, and returns. The same flag is set by a SIGTERM
handler in `droplet-relax`. SLURM sends SIGTERM before a wall-time kill, so a
cluster job ends with a valid checkpoint.

**Determinism.** Sinks receive read-only arrays (`setflags(write=False)`).
The driver computes everything the window shows (energy, force, COM,
curvature) whether or not a window is attached. Test: a run with `[H5Writer]`
and a run with `[H5Writer, LiveWindow(off_screen=True)]` give identical
datasets (timestamps excepted).

**Configuration.** `VisConfig` moves to `DropletViewer.config.ViewerConfig`
and is read from a `[viewer]` table. That table may sit in the same TOML file.
The solver loader skips it but copies it verbatim into
`/DropletRelaxer/config_toml`, so a later replay uses the same colour limits.
Mapping of today's fields:
- `render_every` → solver `output.every`. The live window draws every frame
  that is written.
- `live_mode = "end_only"` is dropped. It equals `droplet-relax run` followed by
  `droplet-view run.h5`.
- `save_gif` → `droplet-view --video`.
- `off_screen` stays a viewer option and is used by the tests.

**Does the architecture become more complex?** A little, and in one place:

- Added compared to a file-only plan:
  - the `Frame`/`FrameSink` protocol (~50 lines; the writer needs an interface like this anyway);
  - a `LiveWindow` sink (~150 lines, mostly calls into `scene.py`);
  - the `droplet-view live` command;
  - the stop flag (which the cluster needs anyway).
- Not added: no second rendering path, no in-memory history, no threads.
- Compared to today it is simpler. In `main.py` the plotter calls are
  interleaved with the remesh loop: a `live_ctx` dict is threaded through
  `_run_until_remesh_or_done`, with `headless` branches and history-dict
  assembly. All of that leaves the solver.
- The one real cost: the scene code must handle a topology change on any
  frame. `replace_mesh` already does that.

## 3. Interop file format: HDF5 using the VTKHDF layout

### Recommendation

Use **HDF5 with the VTKHDF file layout** (type `PolyData`, temporal, format
version 2.x). VTKHDF is VTK's own HDF5-based format. It is the established
geometry format closest to what the solver produces:

- triangle surface meshes in 3D (`PolyData/Polygons`);
- time series in one file, where **the number of points and cells may change
  from step to step**. This is required because remeshing changes the topology;
- read natively by ParaView and by VTK ≥ 9.3 (`vtkHDFReader`), so by PyVista
  too. `requirements.txt` now requires `vtk>=9.3` (the old `<9.5` pin is gone);
- written with plain `h5py` (no VTK on the cluster). It is plain HDF5 with
  chunking and gzip compression.

Extra solver-specific data goes into a sibling group, `/DropletRelaxer`, which
VTK readers ignore.

### Alternatives considered

| Format | Why not |
|---|---|
| NetCDF-3 | No groups, no compression, and no ragged arrays. A changing vertex count needs hand-rolled offset indexing. 2 GiB per-variable limit in the classic format. |
| NetCDF-4 + UGRID conventions | UGRID targets horizontal 2D/layered geoscience meshes and assumes one fixed mesh over time. A surface embedded in 3D whose topology changes per frame is outside its model. |
| XDMF + HDF5 | ParaView reads it and it can change topology over time. But it needs an XML sidecar next to the data, and its VTK reader support is weaker than VTKHDF's. |
| VTU/PVD series | One file per frame, so thousands of small files per run on a cluster file system. |

### Proposed layout (`format_version = 1`)

```
/VTKHDF                         attrs: Version=[2,x], Type="PolyData"
  NumberOfPoints   [n_parts]    # one part per frame
  Points           [ΣN, 3]  f8
  Polygons/
    NumberOfCells           [n_parts]
    NumberOfConnectivityIds [n_parts]
    Offsets                 [Σ(M+1)] i8
    Connectivity            [Σ3M]    i8
  (Vertices, Lines, Strips: empty, but present as the spec requires)
  PointData/
    mean_curvature    [ΣN]   f8
    contact_line      [ΣN]   u1   # 1 on the solid/liquid/vapor ring
    wetted_sphere     [ΣN]   u1
  CellData/
    base              [ΣM]   u1   # 1 = liquid/solid base triangle on z=0
  FieldData/                      # one row per frame
    iteration, energy, area_lv, volume, gap, sphere_center[3],
    force[3], force_raw[3], force_parallel, com[3], com_offset_xy[2]
  Steps/                          attrs: NSteps
    Values            [n_steps]   # iteration number, or sphere path coordinate
    PartOffsets, NumberOfParts, PointOffsets, CellOffsets,
    ConnectivityIdOffsets, PointDataOffsets/*, CellDataOffsets/*,
    FieldDataOffsets/*, FieldDataSizes/*

/DropletRelaxer                 attrs: format_version, code_version (git describe),
                                       created, hostname, command line, mesh_scale
  config_toml       string      # config as given (after --set); used by restart
  config_effective  string      # JSON, after volume scaling; what actually ran
  events/           remesh (iteration, kind, n_vert before/after),
                    quasi-static step accepted/rolled back, divergence
  substrate/        anomaly parameters; roughness field sampled on a grid
                    (+ seed and spectrum parameters, so it is reproducible)
  obstacle/         radius, planned path
  checkpoint/       latest full state for restart: verts, tris, boundary_idx,
                    iteration, quasi-static controller state, V_target
```

Write rules: all datasets are chunked and resizable, a frame is appended and
flushed atomically (data first, then the `Steps` counters), and the file is
opened in SWMR mode so `--follow` readers see consistent frames. The checkpoint
group is overwritten in place every `output.checkpoint_every` iterations.
That lets a run killed by a wall-time limit restart with
`droplet-relax restart run.h5`.

To verify early: whether the installed VTK version reads per-step
`FieldData` from temporal VTKHDF (added in VTKHDF 2.x; confirm the minimum VTK
release). The viewer must not depend on it, so it reads `FieldData` via h5py
in any case. ParaView users simply see the geometry and point/cell arrays.

A short spec page (`docs/file-format.qmd`) is generated from one Python
schema module, so writer, reader, tests and docs share a single definition.

## 4. Repository layout and build system

House style, as observed:

- **NuMPI** is pure Python, built with `hatchling` + `hatch-vcs` (version from git tags into `_version.py`). It has a top-level package directory, `test/`, flake8 via pre-commit, and GitHub workflows `tests.yml`, `flake8.yml`, `publish.yml`.
- **muGrid** is C++ with Python bindings, built with `scikit-build-core` + CMake + `setuptools-scm`. It has `MUGRID_ENABLE_CUDA` / `MUGRID_ENABLE_HIP` CMake options, C++ in `src/`, Python in `language_bindings/python/`, and docs built in a separate `documentation.yml` workflow.

Recommendation: **start in NuMPI style (pure Python, hatchling + hatch-vcs),
and keep `src/` free for a future muGrid-style C++/CUDA/HIP library**. One
repository, two distributions:

```
DropletRelaxer/
├── LICENSE  README.md  CHANGELOG.md  CLAUDE.md  CONTRIBUTING.md
├── pyproject.toml                 # distribution "DropletRelaxer" (solver)
├── .pre-commit-config.yaml  .gitignore  .gitattributes
├── .github/workflows/
│   ├── tests.yml                  # pytest matrix (Python 3.12-3.14), fast tests; slow tests nightly/on label
│   ├── flake8.yml                 # or pre-commit run --all-files
│   ├── documentation.yml          # Quarto → PDF (see §5)
│   └── publish.yml                # PyPI on tags
├── DropletRelaxer/                # solver package: no VTK/PyVista import anywhere
│   ├── __init__.py  _version.py (generated)
│   ├── config.py                  # dataclasses + load/dump TOML; no VisConfig
│   ├── geometry/                  # start mesh, remeshing (split/collapse/Rivara), smoothing, substrate base
│   ├── energies/                  # area/volume, wall, anomaly, roughness, sphere contact, wetting
│   ├── optimizer/                 # relax() generator, Sobolev preconditioner, quasi-static controller
│   ├── driver.py                  # remesh/restart loop now in main.py; emits Frames to FrameSinks, honours StopRequest
│   ├── analysis.py                # force, COM, gap (moved out of visualize.py)
│   ├── io/                        # frame.py (Frame, FrameSink), schema.py, writer.py, reader.py (h5py only)
│   └── cli.py                     # droplet-relax run|restart|info
├── viewer/                        # second distribution "DropletViewer"
│   ├── pyproject.toml             # deps: DropletRelaxer, pyvista, vtk>=9.3, imageio(-ffmpeg)
│   └── DropletViewer/
│       ├── config.py              # ViewerConfig ([viewer] TOML table), ex-VisConfig
│       ├── scene.py  charts.py  history.py  interaction.py  video.py
│       ├── live.py                # LiveWindow frame sink
│       └── cli.py                 # droplet-view run.h5 [--follow] [--video out.mp4] | live cfg.toml
├── docs/                          # Quarto project (§5)
├── examples/                      # *.toml: default scenario, capillary bridge, roughness sweep, SLURM job-array script
└── test/                          # solver tests (NuMPI uses test/); viewer tests in viewer/test/
```

Why two distributions instead of one package with a `[viz]` extra: when the
solver later gains a compiled CUDA/HIP backend, only the solver's
`pyproject.toml` switches its build backend to `scikit-build-core`. The viewer
stays a pure-Python wheel that installs on any laptop. The viewer depends only
on `DropletRelaxer.io`, which must stay free of compiled imports.

Python: `requires-python = ">=3.12"` for both distributions, tested on
3.12–3.14. VTK has no upper bound. The GLX crash that motivated the old `<9.5`
pin is a missing multisample config, which `make_plotter` works around.
`tests.yml` guards against new VTK regressions with an off-screen render test
of the viewer (under `xvfb-run`) on the latest VTK.

### Configuration in TOML

A scenario is **one TOML file**. It has the solver tables `[mesh]`, `[solver]`,
`[obstacle]`, `[anomaly]`, `[roughness]`, `[quasi]` and `[output]`, plus an
optional `[viewer]` table. Keys that are left out take the dataclass defaults,
so `examples/default.toml` may be nearly empty.

```toml
[solver]
contact_angle_deg = 120.0

[obstacle]
center_end = [50.0, 0.0, 125.0]

[output]
every = 200            # frame cadence (was vis.render_every)
checkpoint_every = 5000

[viewer]               # ignored by droplet-relax, read by droplet-view
curvature_clim = [0.0, 4.0]
```

Loader rules (`DropletRelaxer.config.load(path, overrides)`):

- **Unknown tables or keys are an error**, with a "did you mean" hint. A typo
  in a job-array sweep must not silently run the default. The solver loader
  checks `[viewer]` for TOML syntax only. `DropletViewer` validates its
  contents.
- **Types are checked against the annotations.** TOML arrays become tuples for
  `Tuple[...]` fields. Integers are accepted for floats.
- **`None` does not exist in TOML.** Optional fields with default `None`
  (`roughness.rms_height`, `roughness.q_cutoff`) are `None` when the key is
  absent. `mesh.target_volume` defaults to a number. Setting it to `0` means
  "no rescaling" (today: `None`). `vis.save_gif` disappears (`--video`).
- **`--set section.key=value`** on the command line parses `value` as a TOML
  value and is applied after the file. Example:
  `droplet-relax run base.toml -o run_$SLURM_ARRAY_TASK_ID.h5 --set solver.contact_angle_deg=$ANGLE`.
- **Documentation per field.** The `#` comment that follows each field becomes
  an attribute docstring, i.e. a string literal directly below the field.
  Code readers see the same text as today. The Quarto parameter table
  extracts it with `ast`. A test fails if a field has no docstring.
- **Runtime fields leave the config.** `Config.roughness_field` (a
  `RoughnessField` object) becomes driver state, built from `[roughness]`
  and passed to sinks via `RunInfo`. `Config.selftest` goes away (§7).

**Volume scaling in one place.** Today `main()` rescales `volume_penalty`,
`wetting_snap_distance`, `capture_radius`, `substrate_size` and
`curvature_clim` after `resolve_volume_scale`, and `tests/runhelpers.scaled_setup`
duplicates part of that. This becomes
`DropletRelaxer.config.apply_volume_scale(cfg, s) -> Config`. It is a pure
function returning a new config, used by the driver and the tests alike.
`substrate_size` and `curvature_clim` are viewer fields, so the solver stores
`mesh_scale` in `/DropletRelaxer` attrs. `DropletViewer` applies it to its own
config. The file keeps two copies of the config:
- `config_toml`: the file as given, after `--set`, used for restart and to reproduce the run;
- `config_effective`: after scaling, JSON as today, for reading off what actually ran.

**Restart** reads `config_toml` from the file. Overrides at restart are limited
to `solver.n_iterations` and `output.*`. Any other change is a new run.

### Tests: plain pytest

There is no wrapper script. `run_tests.sh`, `--selftest`/`Config.selftest`
and `main._run_selftest` are removed.

```bash
pip install -e ".[test]" -e "viewer[test]"   # pytest is in the `test` extra
pytest                    # fast tests (seconds), solver + viewer
pytest -m slow            # real relaxations (minutes)
pytest -m ""              # everything
pytest test/test_force.py::test_force_equals_minus_energy_derivative   # runs even though it is slow
```

- Configuration moves from `pytest.ini` to `[tool.pytest.ini_options]` in
  `pyproject.toml`:
  - `testpaths = ["test", "viewer/test"]`;
  - `--strict-markers`;
  - one marker, `slow`.
- The `fast` marker goes away. "Fast" simply means "not slow".
- Slow tests are no longer deselected through `addopts = -m "not slow"`.
  Instead, `pytest_collection_modifyitems` in `test/conftest.py` deselects
  them only if no `-m` is given **and** the test was not named explicitly by
  node id on the command line (~15 lines). This removes the gotcha noted in
  CLAUDE.md, where a slow test selected by node id was silently skipped.
- `test/conftest.py` no longer edits `sys.path`. Tests import
  `DropletRelaxer.…` from the editable install.
- Viewer tests live in `viewer/test/` and render off-screen. They are skipped
  through `pytest.importorskip("pyvista")`, so `pytest` in a solver-only
  environment (cluster) passes.
- CI (`tests.yml`):
  - `pytest` on 3.12–3.14 for every push and PR;
  - viewer tests under `xvfb-run` with the latest VTK;
  - `pytest -m slow` nightly and on PRs labelled `slow`.

### Reference runs (regression guard for the migration)

The refactoring standard ("identical final area/volume/contact radius") needs
fixed scenarios that run in under a minute each. `n_iterations = 10⁶` with
quasi-static stepping is far too long. Two scenarios, both chosen to cover
remeshing:

1. **Sessile drop.** No sphere, roughness off, θ = 120°, remeshing on. Run
   long enough to trigger at least one split and one collapse.
2. **Sphere drag.** The default scenario with quasi-static stepping, cut off
   after ~5 accepted sphere steps. This covers wetting, the contact remesh
   triggers and the force output.

For each, `tools/reference_run.py` records into `test/reference/<name>.json`:
- final energy, area, volume, contact radius;
- remesh count, iteration count;
- the force on the sphere;
- a SHA-256 of the final `verts`/`tris` bytes.

`test/test_reference.py` (slow) compares the scalars with `rtol=1e-10`. It
compares the hash only when an environment variable says the machine is the
one that recorded the reference. The reason: SuperLU and BLAS results can
differ in the last bits across CPUs and builds, so bit identity is a
same-machine check, not a CI check. The JSON is regenerated deliberately,
in its own commit, whenever the numerics change on purpose.

### Keeping later GPU support (CUDA + HIP) easy

This plan does not build any GPU code. It avoids choices that would make it
harder later:

- **Array-in, array-out kernels.** Every energy term keeps the signature
  `(arrays…) -> (value, gradient)` on flat contiguous arrays: verts `(N,3) f8`,
  tris `(M,3)` int. No Python sets, dicts or objects inside the per-iteration path.
  These map one to one onto C++/CUDA/HIP kernels or onto CuPy, which supports
  both CUDA and ROCm.
- **Topology on the host, iterations on the device.** Remeshing is rare and
  irregular, so it stays on the CPU. The existing design already restarts
  `relax()` after each remesh, which is exactly where a host-device transfer
  would go.
- **Linear solve behind an interface.** The Sobolev step currently uses SuperLU
  (`scipy.sparse.linalg.factorized`). Wrap it as `Preconditioner.solve(rhs)`
  so a GPU build can substitute cuDSS/rocSOLVER or a preconditioned CG.
- **Output decoupled from the iteration.** The writer receives frames every
  `output.every` iterations, so a device backend needs no host sync per step.
- **Determinism.** README treats bit-identical results as the refactoring
  standard. GPU reductions are generally not bit-reproducible, so regression
  tests should compare against references with tolerances, and keep
  bit-identity as a CPU-only check.
- **Layout.** When the C++ library arrives, it goes in `src/` with a root
  `CMakeLists.txt` and `DROPLETRELAXER_ENABLE_CUDA` / `DROPLETRELAXER_ENABLE_HIP`
  options, mirroring muGrid. The Python package directory does not move.
- **Expectations.** Current meshes have ~600 to a few thousand vertices. At that size
  kernel-launch latency dominates, and job-array parallelism over parameters
  (CPU) will beat a GPU. GPU pays off at roughly 10⁵ vertices or more.

## 5. Documentation in Quarto

- Convert `docs/mathematical_background.tex` with pandoc
  (`pandoc -f latex -t markdown`) into a Quarto book. The book has three
  parts: Theory, Implementation, Comparison/Outlook, as in the current
  document. Then fix by hand:
  - **TikZ figure:** redraw it as SVG, or keep it as a `{=latex}` raw block if PDF is the only target.
  - **`algorithmic` block:** use the `quarto-ext/pseudocode` extension.
  - **`thebibliography`:** move it to `references.bib`.
  - **Custom macros `\R`, `\grad`, `\dif`:** define them once in a `_macros.tex` include, plus the MathJax config.
- Drop the `.txt` duplicate and the committed PDF. The PDF becomes a CI artifact.
- Move the long design log from `README.md` (remeshing measurements,
  determinism, troubleshooting, Numba evaluation) into an "Implementation
  notes" part of the book. README shrinks to: what it is, install, quick
  start, link to the docs.
- **Keeping docs current**:
  1. CI renders the PDF on every pull request, so a broken document fails the PR.
  2. Executable `{python}` chunks generate the parameter reference table from
     the config dataclasses and the file-format page from `io/schema.py`, so
     these never drift.
  3. The "Mapping from equations to code" section references functions by
     dotted name, and a small test checks that every referenced name imports.
  4. CLAUDE.md and the PR template state that a change to numerics must
     update the corresponding section.

Workflow sketch (`.github/workflows/documentation.yml`):

```yaml
name: Documentation
on:
  push: { branches: [main], tags: ['*'] }
  pull_request:
concurrency: { group: documentation-${{ github.ref }}, cancel-in-progress: true }
jobs:
  pdf:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }          # hatch-vcs needs tags
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - run: pip install -e .[docs]       # jupyter, pyyaml for executable chunks
      - uses: quarto-dev/quarto-actions/setup@v2
        with: { tinytex: true }
      - run: quarto render docs --to pdf
      - uses: actions/upload-artifact@v4
        with: { name: DropletRelaxer-docs, path: docs/_book/*.pdf }
      - if: startsWith(github.ref, 'refs/tags/')
        uses: softprops/action-gh-release@v2
        with: { files: docs/_book/*.pdf }
```

Pin action majors to whatever muGrid currently uses when implementing. An
HTML build to GitHub Pages can be added to the same workflow later.

## 6. Migration sequence

Each step keeps the test suite green and is one or a few commits.

1. **Baseline commit.** Add `.gitignore` (`__pycache__/`, `.pytest_cache/`,
   `venv/`, `runs/`, `*.h5`, `docs/_book/`), commit the current code as-is.
   Then add `tools/reference_run.py`, record the two reference runs, and add
   `test/test_reference.py` (see "Reference runs").
2. **Package skeleton.** Move modules with `git mv` into `DropletRelaxer/`,
   add `pyproject.toml` (hatchling + hatch-vcs, `test` extra), pre-commit,
   `tests.yml`. Rename `tests/` → `test/`. Move the pytest settings into
   `pyproject.toml` and add the node-id-aware slow deselection. Drop the
   `sys.path` hack, `run_tests.sh` and the selftest. Update README and CLAUDE.md
   commands. The reference runs reproduce bit-identically.
3. **Decouple.** Move post-processing from `visualize.py` into
   `analysis.py`. Introduce `Frame`/`FrameSink`/`StopRequest`. Turn the loop in
   `main.py` into `driver.py`, which emits frames to sinks. Wrap the existing
   `visualize.py` code as a first `LiveWindow` sink. `main.py` then shrinks to
   "build driver + LiveWindow, run". This keeps the live window working through
   the whole migration. Remove all viewer imports from the solver path. Tests:
   - import the solver with `pyvista` blocked;
   - a run with and without an attached sink gives bit-identical results.
4. **Config + CLI.** TOML loader with strict key checking and `--set`.
   Convert the field comments to attribute docstrings. Add
   `apply_volume_scale` and use it in the driver and in `test/runhelpers.py`.
   Add `droplet-relax run` and the examples directory. Split `VisConfig` off
   into the future viewer config. `main.py` reads `examples/default.toml`
   instead of `CONFIG`. Reference run still bit-identical.
5. **File format.** Schema, writer, reader, restart. Tests: round trip,
   changing topology across remeshes, restart reproduces an uninterrupted
   run, the file opens in `vtkHDFReader`.
6. **Viewer.** Move `LiveWindow` and the scene code into `DropletViewer`. Port the
   history slider to read the file. Replace `export.py` with video export
   (`video.py`) plus the quasi-static frame table. After a `live` run, switch
   to replay on the file just written. Add `--follow` (SWMR) and `--video`. Add
   time-throttled event pumping in `tick()` and the Stop button.
   `run.sh` becomes `droplet-view live examples/default.toml`.
7. **Docs.** Quarto conversion, README slimming, `documentation.yml`,
   generated parameter and file-format pages.
8. **CI and release.** Slow tests as a nightly job, `publish.yml`, first tag.

## 7. Decisions

Settled (2026-10-22):

- Package names: `DropletRelaxer` (solver) and `DropletViewer` (viewer).
  These are also the distribution names on PyPI.
- The in-process live window is kept. It is built as a frame sink on the same
  driver and file format (§2, "In-process live window").
- Minimum Python version: 3.12. `tomllib` comes from the standard library, so
  there is no `tomli` backport.

Settled (round 2):

- VTK: no upper bound, `vtk>=9.3` (done in `requirements.txt`, README, CLAUDE.md).
- `droplet-view live` without `-o` keeps its output in `runs/<timestamp>/run.h5`.
- The `.npy` export is dropped. Save exports video (+ the frame table for
  quasi-static runs). `export.save_run`, `config_effective.json` as a loose file
  and the `ask_save_path` dialog go away. Everything else is in the `.h5`.
- One TOML file per scenario, with an optional `[viewer]` table.

Settled (round 3):

- Selftest is dropped (`--selftest`, `Config.selftest`, `_run_selftest`).
- `run_tests.sh` is dropped. Tests run with plain `pytest` (§4, "Tests").

Settled (round 4):

- `run.sh` and `requirements.txt` are deleted. `droplet-view` warns at startup
  if `DISPLAY` is unset or the session is Wayland. It has a `--software-gl`
  flag that sets the Mesa variables before VTK is imported. The Conda advice
  moves to the troubleshooting docs.
