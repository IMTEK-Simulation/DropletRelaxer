# Changelog

## Unreleased

Restructuring into a headless solver and a separate viewer (see
`RESTRUCTURING_PLAN.md`). Numerics unchanged: the reference runs reproduce
bit-identically.

- Two distributions: `DropletRelaxer` (solver: numpy, scipy, h5py) and
  `DropletViewer` (PyVista viewer), Python >= 3.12.
- Scenarios as TOML files with strict checking and `--set` overrides; every
  parameter documented in its docstring (`droplet-relax show-config`).
- `droplet-relax run|restart|info|show-config`; clean stop on SIGTERM/Ctrl-C
  with a checkpoint; bit-identical restarts.
- Run file: HDF5 in the VTKHDF layout (opens in ParaView), written in SWMR
  mode so `droplet-view --follow` can watch a running job.
- `droplet-view run.vtkhdf [--follow | --video out.mp4]` and `droplet-view live
  scenario.toml` (solver in-process with the live window, Stop button).
- The Save button records the replay video; the `.npy` export is gone (all
  data is in the run file).
- Documentation as a Quarto book with generated parameter and file-format
  references, built to PDF in CI.
- `run.sh`, `run_tests.sh`, `main.py`, `requirements.txt` and the selftest are
  removed; tests run with plain `pytest`. VTK is no longer capped below 9.5.
