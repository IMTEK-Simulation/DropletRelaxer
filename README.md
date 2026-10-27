# DropletRelaxer

A small Surface-Evolver-style simulator. A triangle mesh of a liquid droplet on
a flat substrate is relaxed to minimal surface free energy at fixed volume
(Sobolev-preconditioned gradient descent), optionally with

- a rigid sphere pressed into the droplet and moved along a path in
  quasi-static steps, with wetting of the sphere (capillary bridge) and the
  force of the liquid on the sphere,
- a circular wettability anomaly or random roughness / chemical disorder on
  the substrate,
- adaptive remeshing (Rivara refinement, edge collapse) as the shape changes.

The **solver** (`DropletRelaxer`) is headless and depends only on NumPy, SciPy
and h5py, so it runs on a cluster. It writes one HDF5 file per run in the
VTKHDF layout, which ParaView opens directly. The **viewer**
(`DropletViewer`, PyVista) replays run files, follows runs in progress, renders
videos, or runs the solver in-process with a live window.

## Installation

Python ≥ 3.12.

```bash
git clone https://github.com/IMTEK-Simulation/DropletRelaxer.git
cd DropletRelaxer
python -m venv venv && source venv/bin/activate
pip install -e .            # solver
pip install -e viewer       # viewer (desktop only)
```

## Quick start

```bash
droplet-view live examples/default.toml                    # solver + live window
droplet-relax run examples/sessile.toml -o sessile.vtkhdf  # headless
droplet-view sessile.vtkhdf                                # replay with history slider, Play, Save (video)
droplet-view sessile.vtkhdf --video sessile.mp4            # video without a window
droplet-relax show-config                                  # every parameter, documented
```

Scenarios are TOML files that set only what differs from the defaults; any
value can be overridden with `--set section.key=value`, which makes parameter
sweeps on a cluster a job array (`examples/slurm_contact_angle_sweep.sh`).
`droplet-relax restart run.vtkhdf` continues a run from its checkpoint,
bit-identically to an uninterrupted run.

## Documentation

The documentation is a Quarto book in `docs/`: the mathematical background
(theory, discretization, numerics), the implementation, a comparison with the
Surface Evolver, usage, the configuration reference, the run-file format,
troubleshooting and the design log. The CI builds it as a PDF on every push
(artifact `DropletRelaxer-docs` of the *Documentation* workflow; attached to
releases). To build it locally:

```bash
pip install -e ".[docs]" -e viewer
quarto render docs --to pdf      # docs/_book/DropletRelaxer.pdf
```

## Tests

```bash
pip install -e ".[test]" -e "viewer[test]"
pytest              # fast tests (seconds)
pytest -m slow      # real relaxations, reference runs (minutes)
```

## Repository layout

```
DropletRelaxer/    solver package (driver, energies, mesh, solver, config, io, cli)
viewer/            DropletViewer distribution (scene, live window, history slider, video, cli)
examples/          scenario TOML files, SLURM job-array template
docs/              Quarto book
test/              solver tests (viewer tests in viewer/test/)
tools/             reference_run.py: records the reference runs of test/test_reference.py
```

## License

MIT, see [LICENSE](LICENSE).
