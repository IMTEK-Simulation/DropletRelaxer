"""HDF5 run-file reader (h5py only); layout in schema.py.

`RunFile` gives the frames back as the same `Frame` objects the driver emits
(lazily, one at a time), the per-frame scalars as arrays, the RunInfo, the
events and the restart checkpoint. With `follow=True` the file is opened in
SWMR mode and `refresh()` picks up frames appended by a running solver.
"""

import json

import h5py
import numpy as np

from ..config import ConfigError, apply_volume_scale, loads, parse_override
from .frame import Frame, RunInfo, readonly
from .schema import FIELD_ARRAYS, GROUP, POLYGONS, RAGGED

# a restart may only change how long the run goes and what is written
RESTART_OVERRIDABLE = ("solver.n_iterations", "output.")


class RunFile:
    def __init__(self, path, follow=False):
        self.path = path
        self.f = h5py.File(path, "r", swmr=follow) if follow else h5py.File(path, "r")
        self.follow = follow
        self._run_info = None
        self._n_follow = 0
        self._status_follow = "running"
        self.refresh()

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- metadata --------------------------------------------------------------

    @property
    def attrs(self):
        return self.f[GROUP].attrs

    @property
    def status(self):
        """"running", "finished", "stopped" or "diverged" ("running" also
        for a run that was killed). In follow mode: as of the last refresh()."""
        if self.follow:
            return self._status_follow
        return self.f[GROUP]["status"][0].decode().strip()

    @property
    def end_iteration(self):
        return int(self.f[GROUP]["end_iteration"][0])

    def refresh(self):
        """Follow mode: make frames appended since the last call visible.
        Steps/Values is refreshed FIRST and fixes the frame count; the data of
        those frames was written before it, so refreshing everything else
        afterwards shows at least that much."""
        if not self.follow:
            return
        # Order matters: the writer sets the status after the last frame, so a
        # status read FIRST that says "finished" guarantees that the frame
        # count read next includes every frame.
        status = self.f[GROUP]["status"]
        status.refresh()
        self._status_follow = status[0].decode().strip()
        values = self.f["VTKHDF/Steps/Values"]
        values.refresh()
        n = int(values.shape[0])
        for g in (self.f["VTKHDF"], self.f[GROUP]):
            g.visititems(lambda _n, obj: obj.refresh() if isinstance(obj, h5py.Dataset) else None)
        self._n_follow = n

    def __len__(self):
        """Number of complete frames. The writer appends Steps/Values and then
        NSteps after all other data of a frame; a following (SWMR) reader can
        only refresh datasets, so it counts Values."""
        if self.follow:
            return self._n_follow
        return int(self.f["VTKHDF/Steps"].attrs["NSteps"])

    def config_text(self):
        t = self.f[GROUP]["config_toml"][()]
        return t.decode() if isinstance(t, bytes) else t

    def config(self, restart_overrides=()):
        """(Config as given, table) from the stored TOML. `restart_overrides`
        ('section.key=value') may only change solver.n_iterations and output.*."""
        for o in restart_overrides:
            key = ".".join(parse_override(o)[0])
            if not (key == RESTART_OVERRIDABLE[0] or key.startswith(RESTART_OVERRIDABLE[1])):
                raise ConfigError(f"'{key}' cannot change at a restart (only solver.n_iterations and output.*); "
                                  "start a new run instead")
        return loads(self.config_text(), restart_overrides)

    def run_info(self):
        """RunInfo of the run (effective config, start mesh, roughness field)."""
        if self._run_info is None:
            cfg, _ = self.config()
            a = self.attrs
            mesh_scale = float(a["mesh_scale"])
            eff = apply_volume_scale(cfg, mesh_scale)
            s = self.f[GROUP]["start"]
            rf = None
            edge = float(a["roughness_edge"])
            if eff.roughness.enabled and np.isfinite(edge):
                import math

                from ..roughness import make_roughness_field
                rf = make_roughness_field(eff.roughness, edge, math.cos(math.radians(eff.solver.contact_angle_deg)))
            self._run_info = RunInfo(
                config=eff, mesh_scale=mesh_scale, V0=float(a["V0"]), A0=float(a["A0"]),
                verts=readonly(s["verts"][:]), tris=readonly(s["tris"][:]), boundary_idx=readonly(s["boundary_idx"][:]),
                total_iters=int(a["total_iters"]), quasi_static=bool(a["quasi_static"]),
                roughness_field=rf, roughness_edge=edge, config_unscaled=cfg)
        return self._run_info

    def viewer_table(self):
        return self.config()[1].get("viewer")

    # -- frames ------------------------------------------------------------------

    def series(self):
        """{name: array} of the per-frame scalars (FieldData) plus epoch,
        step_iters and converged, for all complete frames."""
        n = len(self)
        fd, fr = self.f["VTKHDF/FieldData"], self.f[GROUP]["frames"]
        out = {a.name: fd[a.name][:n] for a in FIELD_ARRAYS}
        for name in ("epoch", "step_iters", "converged"):
            out[name] = fr[name][:n]
        out["converged"] = out["converged"].astype(bool)
        return out

    def frame(self, k):
        """Frame k (negative k counts from the end)."""
        n = len(self)
        if k < 0:
            k += n
        if not 0 <= k < n:
            raise IndexError(f"frame {k} out of range ({n} frames)")
        v, st, d = self.f["VTKHDF"], self.f["VTKHDF/Steps"], self.f[GROUP]
        p0 = int(st["PointOffsets"][k])
        N = int(v["NumberOfPoints"][k])
        k0 = int(st["ConnectivityIdOffsets"][k, POLYGONS])
        M = int(v["Polygons/NumberOfCells"][k])
        verts = v["Points"][p0:p0 + N]
        tris = v["Polygons/Connectivity"][k0:k0 + 3 * M].reshape(M, 3)
        curvature = v["PointData/mean_curvature"][p0:p0 + N]
        fd = {a.name: v["FieldData"][a.name][k] for a in FIELD_ARRAYS}
        fr = d["frames"]
        ragged = {}
        for name, off in RAGGED.items():
            a, b = int(fr[off][k]), int(fr[off][k + 1])
            ragged[name] = fr[name][a:b]
        sc = fd["sphere_center"]
        return Frame(
            index=k, iteration=int(fd["iteration"]), x=float(fd["x"]), epoch=int(fr["epoch"][k]),
            verts=readonly(verts), tris=readonly(tris, np.int64), boundary_idx=readonly(ragged["boundary_idx"]),
            wetted_loop=readonly(ragged["wetted_loop"]) if len(ragged["wetted_loop"]) else None,
            sphere_center=None if np.isnan(sc).all() else readonly(sc),
            energy=float(fd["energy"]), area=float(fd["area_lv"]), volume=float(fd["volume"]),
            force=readonly(fd["force"]), com=readonly(fd["com"]), com_offset_xy=readonly(fd["com_offset_xy"]),
            gap=float(fd["gap"]), curvature=readonly(curvature),
            step_iters=int(fr["step_iters"][k]), converged=bool(fr["converged"][k]),
        )

    def frames(self):
        """Lazy sequence of all complete frames."""
        return FrameSequence(self)

    # -- events and checkpoint -----------------------------------------------------

    def events(self):
        ev = self.f[GROUP]["events"]
        out = []
        for i, kind, data in zip(ev["iteration"][:], ev["kind"][:], ev["data"][:]):
            data = json.loads(data.decode() if isinstance(data, bytes) else data)
            out.append({"iteration": int(i), "kind": kind.decode(), **data})
        return out

    def checkpoint(self):
        """The latest restart checkpoint as a dict for Driver(restart=...), or None."""
        ck = self.f[GROUP]["checkpoint"]
        cur = int(ck.attrs["current"])
        if cur < 0:
            return None
        g = ck[f"slot{cur}"]
        meta = json.loads(bytes(g["state"][:]).decode())
        return {
            "iteration": int(meta.pop("iteration")),
            "verts": g["verts"][:], "tris": g["tris"][:], "boundary_idx": g["boundary_idx"][:],
            "quasi_ckpt_verts": g["quasi_ckpt_verts"][:] if meta.pop("has_quasi_ckpt_verts") else None,
            "scalars": meta,
        }

    def describe(self):
        """Human-readable summary (droplet-relax info)."""
        a = self.attrs
        n = len(self)
        lines = [f"{self.path}: DropletRelaxer run file (format {int(a['format_version'])}, "
                 f"written by DropletRelaxer {a['code_version']})",
                 f"  created {a['created']} on {a['hostname']}",
                 f"  command: {a['command_line']}" if a["command_line"] else "  command: (API)",
                 f"  status: {self.status}, last iteration {self.end_iteration}",
                 f"  {n} frames" + (" (quasi-static: one per sphere step)" if a["quasi_static"] else "")]
        if n:
            s = self.series()
            lines.append(f"  iterations {int(s['iteration'][0])} .. {int(s['iteration'][-1])}, "
                         f"{int(s['epoch'][-1])} remeshes, energy {s['energy'][0]:.6g} -> {s['energy'][-1]:.6g}")
        ck = self.checkpoint()
        lines.append(f"  checkpoint at iteration {ck['iteration']}" if ck else "  no checkpoint")
        return "\n".join(lines)


class FrameSequence:
    """Sequence view of a RunFile's frames: len() and [k], read on demand."""

    def __init__(self, rf):
        self.rf = rf

    def __len__(self):
        return len(self.rf)

    def __getitem__(self, k):
        return self.rf.frame(k)

    def __iter__(self):
        for k in range(len(self)):
            yield self.rf.frame(k)
