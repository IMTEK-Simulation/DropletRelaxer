# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small Surface-Evolver-style simulator: a triangle mesh of a liquid droplet on a flat substrate is relaxed to minimal surface free energy at fixed volume (Sobolev-preconditioned gradient descent), with an optional rigid sphere driven through it, wettability anomaly / roughness on the substrate, adaptive remeshing, and a PyVista live view. The solver modules form the package `DropletRelaxer/` (flat, no subpackages); `main.py`, `visualize.py` and `export.py` still sit in the repo root during the restructuring (see `RESTRUCTURING_PLAN.md`). Tests import the installed package (`pip install -e ".[test]"`).

`README.md` is long and is the primary design log (measurements, rationale for parameters, troubleshooting). `docs/mathematical_background.tex` has the derivations; `.txt` is an identical copy of the `.tex` and must be kept in sync if either is edited.

## Commands

```bash
./run.sh                      # creates ./venv, installs requirements.txt once, runs main.py (args forwarded)
python main.py                # run directly in an existing env

pip install -e ".[test]"      # package + pytest
pytest                        # fast tests (~5 s)
pytest -m slow                # real relaxations (~6 min), incl. reference runs
pytest -m ""                  # everything
pytest -k anomaly
pytest test/test_force.py::test_force_equals_minus_energy_derivative   # a slow test named by node id runs
DROPLET_REFERENCE_HASH=1 pytest -m slow test/test_reference.py      # also bit identity (same machine only)
python tools/reference_run.py # re-record test/reference/*.json (only when numerics change on purpose)
pre-commit run --all-files    # flake8 (.flake8, max line 120)
```

Slow tests are deselected by `test/conftest.py` unless `-m` is given or the test is named by node id.

There is no CLI for parameters: all settings live in `config.py` (`CONFIG = Config()`, dataclass groups `mesh`, `solver`, `obstacle`, `vis`, `anomaly`, `roughness`, `quasi`). Every field carries a comment explaining its effect; keep that convention when adding parameters. For headless runs set `vis.live = False` or `vis.off_screen = True`.

`vtk` has no upper bound. The GLX "Could not find a decent config" crash seen with 9.7.1 is handled by disabling multisampling in `visualize.py:make_plotter` (README Troubleshooting); keep that call.

## Architecture

- **`main.py`** orchestrates everything. It builds the start mesh, then loops: `_run_until_remesh_or_done()` drives a single `solver.relax()` generator until either the iteration budget ends or a remesh trigger fires ("split", "collapse", "contact_collapse"; checked every `mesh.remesh_check_every`). On a trigger it remeshes (`mesh.py`) and starts a **new** `relax()` generator, which rebuilds the sparse Laplacian factorization. Each remesh renumbers vertices and starts a new history "epoch".
- **`solver.relax()`** is a generator yielding `(i, verts, area, volume, sphere_center, wetted_loop)` per iteration. It sums gradients of all energy terms, applies the Sobolev preconditioner (direct SuperLU via `scipy.sparse.linalg.factorized`), and raises `SolverDiverged` on non-finite/huge coordinates. It also optionally steps the sphere when the `QuasiStatic` controller (`quasistatic.py`) reports convergence.
- **Energy terms** each live in a function returning value and analytic gradient: `energy.py` (area, volume penalty, substrate wall/shoelace energy, penetration, Laplacian, centre of mass), `anomaly.py`, `roughness.py`, `obstacle.py` (sphere contact/adhesion, sphere path), `wetting.py` (wetted patch and Young-angle energy on the sphere). Every new gradient is expected to get a central-finite-difference test (see `fd_gradient` in `test/conftest.py`).
- **Closed mesh**: the droplet is a dome plus a flat base on z = 0 (`substrate.py`). The base is stripped and rebuilt from the contact-line ring after each remesh, never triggers a remesh, and is excluded from the liquid/vapor area (`base_triangle_mask`). Contact-line vertices are constrained to the substrate via the DOF mask from `mesh.build_dof_mask`.
- **Physical scaling**: `mesh.resolve_volume_scale()` rescales the mesh to `mesh.target_volume`; `main()` then rescales dependent config values (`volume_penalty /= s**4`, snap distance, capture radius, substrate size, inverse for `curvature_clim`). `test/runhelpers.scaled_setup()` duplicates this logic, so changes to the scaling in `main()` must be mirrored there.
- **`visualize.py`** (PyVista window, charts, history slider, Play/Save) and **`export.py`** (Save button: run folder with config copy, `config_effective.json`, `.npy` series, video) are output-only.

## Invariants worth preserving

- Runs are bit-for-bit deterministic: no randomness in the physics loop (the roughness field is drawn once from `RoughnessConfig.seed`), no threading, and every Python `set` holds plain integers. Keep it that way; README documents performance changes as "identical final area/volume/contact radius", which is the standard for refactors.
- Slow tests run on a fixed mesh without remeshing via `test/runhelpers.py` (`static_cfg`, `run_solver`, `cap_prediction`) and compare against analytic spherical caps and energy derivatives.
