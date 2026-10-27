# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small Surface-Evolver-style simulator: a triangle mesh of a liquid droplet on a flat substrate is relaxed to minimal surface free energy at fixed volume (Sobolev-preconditioned gradient descent), with an optional rigid sphere driven through it, wettability anomaly / roughness on the substrate, and adaptive remeshing.

One repository, two distributions:
- **`DropletRelaxer/`** (root `pyproject.toml`): the headless solver. Depends on numpy, scipy, h5py only and must never import PyVista/VTK (`test/test_driver.py::test_solver_imports_without_pyvista`).
- **`viewer/`** (`viewer/pyproject.toml`, package `DropletViewer`): PyVista viewer: replay of run files, `--follow`, `--video`, and `droplet-view live` (solver in-process with a live window). Depends on DropletRelaxer, never the other way round.

`docs/` is a Quarto book (theory, implementation, usage, generated parameter and file-format references, design log in `docs/appendix/notes.qmd`). `RESTRUCTURING_PLAN.md` records why the code is laid out this way.

## Commands

```bash
pip install -e ".[test]" -e "viewer[test]"
pytest                        # fast tests, solver + viewer (~20 s); viewer tests skip without pyvista
pytest -m slow                # real relaxations incl. reference runs (~6 min)
pytest -m ""                  # everything
pytest test/test_force.py::test_force_equals_minus_energy_derivative   # a slow test named by node id runs
DROPLET_REFERENCE_HASH=1 pytest -m slow test/test_reference.py      # also bit identity (same machine only)
python tools/reference_run.py # re-record test/reference/*.json -- only when numerics change on purpose, own commit
pre-commit run --all-files    # flake8 (.flake8, max line 120)

droplet-relax run examples/sessile.toml -o run.vtkhdf [--set solver.contact_angle_deg=120]
droplet-relax restart run.vtkhdf | info run.vtkhdf | show-config [scenario.toml]
droplet-view run.vtkhdf [--follow | --video out.mp4]
droplet-view live examples/default.toml

pip install -e ".[docs]" -e viewer && quarto render docs --to pdf    # QUARTO_PYTHON=<venv python> if needed
```

Slow tests are deselected by `test/conftest.py` unless `-m` is given or the test is named by node id.

## Configuration

Parameters are dataclasses in `DropletRelaxer/config.py` (`mesh`, `solver`, `obstacle`, `output`, `anomaly`, `roughness`, `quasi`); the viewer's display settings are `DropletViewer/config.py:ViewerConfig` (`[viewer]` table). Scenarios are TOML files loaded strictly (unknown keys/wrong types are errors). **Every field has a docstring directly below it** (attribute docstring); `test/test_config.py` fails otherwise, and the docs' parameter reference is generated from them. Keep that convention when adding parameters. TOML has no `None`: optional fields default to `None` when absent.

## Architecture

- **`driver.Driver`** runs the loop: builds/scales the start mesh, drives one `solver.relax()` generator until the iteration budget ends or a remesh trigger fires ("split", "collapse", "contact_collapse"; checked every `mesh.remesh_check_every`), remeshes, starts a new `relax()` (rebuilding the sparse factorization). Each remesh renumbers vertices and starts a new topology epoch. Loop state that survives a remesh lives on the instance so checkpoints capture it.
- **Sinks** (`io/frame.py`): the driver emits read-only `Frame`s (energy, force, COM, gap, curvature all computed by the driver via `analysis.py`), `Event`s, `tick`s and checkpoint dicts to `FrameSink`s: `io.writer.H5Writer`, `console.ConsoleSink`, `DropletViewer.live.LiveWindow`. Only a `StopRequest` flows back (Stop button, window closed, SIGTERM/Ctrl-C in the CLIs).
- **`solver.relax()`** yields `(i, verts, area, volume, sphere_center, wetted_loop)` per iteration; Sobolev preconditioner via `scipy.sparse.linalg.factorized`; raises `SolverDiverged`. Quasi-static sphere stepping via `quasistatic.QuasiStatic` (its `get_state`/`set_state` are part of checkpoints).
- **Energy terms** each return value and analytic gradient: `energy.py`, `anomaly.py`, `roughness.py`, `obstacle.py`, `wetting.py`. Every new gradient gets a central-finite-difference test (`fd_gradient` in `test/numhelpers.py`).
- **Closed mesh**: dome + flat base on z = 0 (`substrate.py`), rebuilt from the contact-line ring after each remesh, excluded from the liquid/vapor area (`base_triangle_mask`). Contact-line vertices are constrained to the substrate via `mesh.build_dof_mask`.
- **Physical scaling**: `mesh.resolve_volume_scale()` + `config.apply_volume_scale()` (used by the driver and `test/runhelpers.scaled_setup`); the viewer scales its own lengths by `RunInfo.mesh_scale`.
- **Run file** (`io/schema.py` is the single definition): VTKHDF temporal PolyData under `/VTKHDF` (ParaView-readable; verified with VTK 9.3 and 9.7) + `/DropletRelaxer` (config, ragged rings, events, checkpoint slots). Written in SWMR mode with HDF5 1.10 format (`writer.LIBVER`) so VTK's bundled HDF5 can read it; readers count frames by `Steps/Values` (written last). Changing the layout means bumping `schema.FORMAT_VERSION`.

## Invariants worth preserving

- Runs are bit-for-bit deterministic: no randomness in the physics loop (the roughness field is drawn once from `RoughnessConfig.seed`), no threading, every Python `set` holds plain integers. Attaching sinks (window, writer) must not change results; a restart from a checkpoint must equal the uninterrupted run. Tests check all three. Refactors must reproduce the reference runs (`test/test_reference.py`).
- A change to the numerics must update the corresponding chapter in `docs/` (the equation-to-code table in `docs/implementation/mapping.qmd` uses dotted names that `test/test_docs.py` imports).
- `vtk` has no upper bound. The GLX "Could not find a decent config" crash seen with 9.7.1 is handled by disabling multisampling in `DropletViewer/scene.py:make_plotter`; keep that call.
