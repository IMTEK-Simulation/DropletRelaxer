"""Saving a finished run ("Save" button next to Play, see visualize.py).

A run is saved as a FOLDER containing
    config.py              the config file the run was started with (copy)
    config_effective.json  the parameters actually used (after the automatic
                           mesh-scale adjustments of main.py)
    iteration.npy          iteration number of every recorded frame, shape (N,)
    gap.npy                gap = z of the sphere centre minus the sphere radius
                           (height of the sphere's lowest point above the
                           substrate), as plotted in the "Gap" chart (NaN if
                           there is no sphere), shape (N,)
    energy.npy             surface free energy, as plotted, shape (N,)
    force.npy              force of the liquid on the sphere, vector (Fx, Fy, Fz), as
                           plotted, shape (N, 3). From the force balance in the solver
                           (pressure + surface tension carried by the wetted patch,
                           Young term, collision), i.e. F = -dE/d(sphere centre)
    force_raw.npy          the same without averaging, shape (N, 3)
    force_parallel.npy     force component along the sphere's direction of motion
                           (friction / drag measure; NaN while it does not move), (N,)
    force_from_energy.npy  older estimate -dE/ds along the path from the energy
                           curve (cross-check; polluted by the liquid's own
                           relaxation), shape (N,)
    frames.csv             (quasi-static mode) one row per equilibrium frame: tip position x,y,z,
                           iteration, iterations used by the step, converged flag (1/0),
                           energy, Fx, Fy, Fz
    com_offset.npy         offset of the liquid's centre of mass from the sphere
                           centre in the xy plane, |COM_xy - sphere_xy| (from the
                           axis x = y = 0 in frames without a sphere), shape (N,)
    com_offset_xy.npy      the same as vector (dx, dy), shape (N, 2)
    sphere_center.npy      sphere centre (x, y, z), NaN without sphere, (N, 3)
    README.txt             what the files contain
"""

import dataclasses
import json
import os
import shutil
import subprocess
import time

import numpy as np


def ask_save_path(default_name, initial_dir=None):
    """"Save as" dialog; returns the chosen path (a folder with this name is
    created) or None if cancelled. Tries tkinter, then zenity/kdialog; if no
    dialog is available at all, returns a default path ./runs/<name> and says
    so on the console (so the button still works on a bare system)."""
    initial_dir = initial_dir or os.getcwd()
    try:
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        path = filedialog.asksaveasfilename(
            title="Save run as (a folder with this name will be created)",
            initialdir=initial_dir, initialfile=default_name)
        root.destroy()
        return path or None
    except Exception:
        pass
    start = os.path.join(initial_dir, default_name)
    for cmd in (["zenity", "--file-selection", "--save", "--confirm-overwrite",
                 "--title=Save run as (folder name)", f"--filename={start}"],
                ["kdialog", "--getsavefilename", start]):
        if shutil.which(cmd[0]):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True)
                return out.stdout.strip() or None
            except Exception:
                continue
    path = os.path.join(initial_dir, "runs", default_name)
    print("No file dialog available (install python3-tk or zenity); "
          f"saving to {path}")
    return path


def _trailing_mean(values, window):
    v = np.asarray(values, dtype=float)
    if window is None or window <= 1 or len(v) == 0:
        return v
    c = np.concatenate((np.zeros((1,) + v.shape[1:]), np.cumsum(v, axis=0)))
    idx = np.arange(1, len(v) + 1)
    lo = np.maximum(idx - int(window), 0)
    shape = (-1,) + (1,) * (v.ndim - 1)
    return (c[idx] - c[lo]) / (idx - lo).reshape(shape)


FRAME_KEYS = ["x", "y", "z", "iteration", "step_iters", "converged", "energy", "Fx", "Fy", "Fz"]
FRAME_HEADER = ",".join(FRAME_KEYS)


def format_frame_table(rows):
    """Console table of the recorded frames (one per sphere step in
    quasi-static mode): tip position, iterations used by the step, whether it
    converged, surface free energy and the force of the liquid on the sphere."""
    if not rows:
        return ""
    lines = ["Equilibrium frames (tip position, force of the liquid on the sphere):",
             f"{'x':>8}{'y':>8}{'z':>8}{'step its':>10}{'conv':>6}{'energy':>14}{'Fx':>11}{'Fy':>11}{'Fz':>11}"]
    for r in rows:
        lines.append(f"{r['x']:>8.2f}{r['y']:>8.2f}{r['z']:>8.2f}{r['step_iters']:>10}"
                     f"{'yes' if r['converged'] else 'NO':>6}{r['energy']:>14.4f}"
                     f"{r['Fx']:>11.4f}{r['Fy']:>11.4f}{r['Fz']:>11.4f}")
    return "\n".join(lines)


def save_run(folder, cfg, iterations, energy, centers, com, config_source=None,
             force_fn=None, avg_window=1, gap_offset=0.0, forces=None, frame_rows=None):
    """Writes the folder described in the module docstring. `centers` is the
    per-frame list of sphere centres (tuple or None), `com` the per-frame
    centre of mass, `forces` the per-frame force of the liquid on the sphere
    (3-vectors, solver.py `diag["force"]`). `force_fn(centers, energy, window)`
    is the older energy-derivative force (visualize._capillary_force), kept as
    a cross-check (force_from_energy.npy)."""
    if os.path.exists(folder) and not os.path.isdir(folder):
        folder = folder + "_run"
    os.makedirs(folder, exist_ok=True)

    it = np.asarray(iterations, dtype=float)
    en = np.asarray(energy, dtype=float)
    sc = np.array([[np.nan] * 3 if c is None else list(c) for c in centers], dtype=float)
    # frames recorded before the sphere position was known (e.g. frame 0)
    # take the nearest known position, as the plotted Gap/tip-x curves do
    ok = ~np.isnan(sc).any(axis=1)
    if ok.any() and not ok.all():
        idx = np.maximum.accumulate(np.where(ok, np.arange(len(sc)), -1))
        idx = np.where(idx < 0, np.argmax(ok), idx)
        sc = sc[idx]
    com = np.asarray(com, dtype=float).reshape(len(it), 3)
    ref = np.where(np.isnan(sc[:, :2]), 0.0, sc[:, :2])          # no sphere -> axis x = y = 0
    off = com[:, :2] - ref
    out = {
        "iteration.npy": it,
        "gap.npy": sc[:, 2] - gap_offset,
        "energy.npy": en,
        "com_offset.npy": np.hypot(off[:, 0], off[:, 1]),
        "com_offset_xy.npy": off,
        "sphere_center.npy": sc,
    }
    if forces is not None:
        f_raw = np.asarray(forces, dtype=float).reshape(len(it), 3)
        f_avg = _trailing_mean(f_raw, avg_window)
        out["force.npy"] = f_avg
        out["force_raw.npy"] = f_raw
        # component along the instantaneous direction of motion of the sphere
        # (friction/drag measure; NaN while the sphere does not move)
        c = np.where(np.isnan(sc), 0.0, sc)
        vel = np.gradient(c, axis=0) if len(c) > 1 else np.zeros_like(c)
        speed = np.linalg.norm(vel, axis=1)
        t_hat = np.where(speed[:, None] > 1e-9, vel / np.maximum(speed, 1e-300)[:, None], np.nan)
        out["force_parallel.npy"] = np.sum(f_avg * t_hat, axis=1)
    if force_fn is not None:
        out["force_from_energy.npy"] = np.asarray(force_fn(centers, energy, avg_window), dtype=float)
    for name, arr in out.items():
        np.save(os.path.join(folder, name), arr)

    if frame_rows:
        with open(os.path.join(folder, "frames.csv"), "w") as f:
            f.write(FRAME_HEADER + "\n")
            for r in frame_rows:
                f.write(",".join(str(int(r[k]) if k == "converged" else r[k]) for k in FRAME_KEYS) + "\n")

    if config_source and os.path.isfile(config_source):
        shutil.copy(config_source, os.path.join(folder, "config.py"))
    with open(os.path.join(folder, "config_effective.json"), "w") as f:
        _d = {fl.name: getattr(cfg, fl.name) for fl in dataclasses.fields(cfg)
              if fl.name != "roughness_field"}      # runtime object, not a setting
        json.dump({k: dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v
                   for k, v in _d.items()}, f, indent=2, default=str)
    with open(os.path.join(folder, "README.txt"), "w") as f:
        f.write(__doc__ + f"\nSaved {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return folder
