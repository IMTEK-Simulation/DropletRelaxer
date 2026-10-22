"""HDF5 run-file writer (a frame sink); layout in schema.py.

All datasets are created chunked and resizable when the run starts; then the
file switches to SWMR mode, so `droplet-view --follow` can read it while it
grows. A frame is appended data first; `Steps/Values` and then the `Steps/NSteps`
attribute, which readers use as the frame count, are updated last, followed
by a flush. The
restart checkpoint alternates between two slots, and the `current` attribute
is switched only after a slot is complete, so a run killed while writing a
checkpoint still has the previous one.
"""

import json
import os
import socket
import time

import h5py
import numpy as np

from .. import __version__
from ..config import dumps, to_table
from ..substrate import base_triangle_mask
from .frame import NullSink
from .schema import (ARRAYS, CELL_ARRAYS, FIELD_ARRAYS, FORMAT_VERSION, GROUP, POINT_ARRAYS, POLYGONS, RAGGED,
                     TOPOLOGIES, VTKHDF_VERSION, field_values)

# SWMR needs at least the HDF5 1.10 file format; capping it there keeps the
# file readable by older HDF5 libraries, e.g. the one bundled with VTK.
LIBVER = ("v110", "v110")
_CHUNK = 4096
_COMPRESSION = dict(compression="gzip", compression_opts=4, shuffle=True)


def _ascii(s):
    """Fixed-length ASCII string attribute value (what VTK's reader expects)."""
    return np.bytes_(s.encode("ascii"))


def _create(group, name, shape_tail=(), dtype="f8", chunk=_CHUNK, compress=True):
    return group.create_dataset(name, shape=(0,) + tuple(shape_tail), maxshape=(None,) + tuple(shape_tail),
                                dtype=dtype, chunks=(chunk,) + tuple(shape_tail), **(_COMPRESSION if compress else {}))


def _append(ds, values):
    values = np.asarray(values, dtype=ds.dtype)
    n = ds.shape[0]
    ds.resize(n + len(values), axis=0)
    if len(values):
        ds[n:] = values
    return n


def _truncate(ds, n):
    if ds.shape[0] > n:
        ds.resize(n, axis=0)


class H5Writer(NullSink):
    """Writes a run into `path`. `viewer_table`: the scenario's [viewer] table
    (stored with the config). `append=True` continues a run file after a
    restart: frames and events after the checkpoint are dropped first."""

    def __init__(self, path, viewer_table=None, command_line=None, append=False):
        self.path = path
        self.viewer_table = viewer_table
        self.command_line = " ".join(command_line) if command_line else ""
        self.append = append
        self.f = None

    # -- layout ------------------------------------------------------------

    def _create_layout(self, run):
        f = self.f
        v = f.create_group("VTKHDF")
        v.attrs["Version"] = np.array(VTKHDF_VERSION, dtype=np.int64)
        v.attrs.create("Type", _ascii("PolyData"))
        _create(v, "NumberOfPoints", dtype="i8", chunk=256)
        _create(v, "Points", (3,), "f8")
        for topo in TOPOLOGIES:
            g = v.create_group(topo)
            _create(g, "NumberOfCells", dtype="i8", chunk=256)
            _create(g, "NumberOfConnectivityIds", dtype="i8", chunk=256)
            _create(g, "Offsets", dtype="i8")
            _create(g, "Connectivity", dtype="i8")
        pd, cd, fd = v.create_group("PointData"), v.create_group("CellData"), v.create_group("FieldData")
        for a in POINT_ARRAYS:
            _create(pd, a.name, dtype=a.dtype)
        for a in CELL_ARRAYS:
            _create(cd, a.name, dtype=a.dtype)
        for a in FIELD_ARRAYS:
            _create(fd, a.name, (a.components,) if a.components > 1 else (), a.dtype, chunk=256)
        st = v.create_group("Steps")
        st.attrs["NSteps"] = np.int64(0)
        for name in ("Values",):
            _create(st, name, dtype="f8", chunk=256, compress=False)
        for name in ("PartOffsets", "NumberOfParts", "PointOffsets"):
            _create(st, name, dtype="i8", chunk=256, compress=False)
        _create(st, "CellOffsets", (len(TOPOLOGIES),), "i8", chunk=256, compress=False)
        _create(st, "ConnectivityIdOffsets", (len(TOPOLOGIES),), "i8", chunk=256, compress=False)
        for grp, arrays in (("PointDataOffsets", POINT_ARRAYS), ("CellDataOffsets", CELL_ARRAYS),
                            ("FieldDataOffsets", FIELD_ARRAYS)):
            g = st.create_group(grp)
            for a in arrays:
                _create(g, a.name, dtype="i8", chunk=256, compress=False)
        g = st.create_group("FieldDataSizes")
        for a in FIELD_ARRAYS:
            _create(g, a.name, (2,), "i8", chunk=256, compress=False)

        d = f.create_group(GROUP)
        cfg = run.config
        d.attrs["format_version"] = np.int64(FORMAT_VERSION)
        d.attrs["code_version"] = __version__
        d.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        d.attrs["hostname"] = socket.gethostname()
        d.attrs["command_line"] = self.command_line
        d.attrs["mesh_scale"] = float(run.mesh_scale)
        d.attrs["V0"] = float(run.V0)
        d.attrs["A0"] = float(run.A0)
        d.attrs["quasi_static"] = bool(run.quasi_static)
        d.attrs["total_iters"] = np.int64(run.total_iters)
        # status and end iteration are small datasets, not attributes: a SWMR
        # reader can refresh datasets, not attributes
        d.create_dataset("status", data=np.array([b"running"], dtype="S16"))
        d.create_dataset("end_iteration", data=np.array([-1], dtype=np.int64))
        rf = run.roughness_field
        d.attrs["roughness_edge"] = float(run.roughness_edge)
        # RunInfo.config is the effective (volume-scaled) config; the file stores
        # the scenario as given, which is what a restart reads.
        given = run.config_unscaled if run.config_unscaled is not None else run.config
        d.create_dataset("config_toml", data=dumps(given, viewer=self.viewer_table))
        d.create_dataset("config_effective", data=json.dumps(to_table(cfg), indent=1))
        s = d.create_group("start")
        s.create_dataset("verts", data=np.asarray(run.verts))
        s.create_dataset("tris", data=np.asarray(run.tris))
        s.create_dataset("boundary_idx", data=np.asarray(run.boundary_idx))
        fr = d.create_group("frames")
        for a in (x for x in ARRAYS if x.location == "frames"):
            _create(fr, a.name, dtype=a.dtype, chunk=256, compress=False)
        for name, off in RAGGED.items():
            _create(fr, name, dtype="i8")
            _append(fr[off], [0])          # ragged offsets have one more entry than frames
        ev = d.create_group("events")
        _create(ev, "iteration", dtype="i8", chunk=64, compress=False)
        _create(ev, "kind", dtype="S16", chunk=64, compress=False)
        _create(ev, "data", dtype=h5py.string_dtype(), chunk=64, compress=False)
        sub = d.create_group("substrate")
        if rf is not None:
            stride = max(1, -(-rf.N // 512))
            k = sub.create_dataset("roughness_kappa_eff", data=np.asarray(rf.kappa_eff)[::stride, ::stride],
                                   **_COMPRESSION)
            k.attrs["x0"], k.attrs["dx"] = float(rf.x0), float(rf.dx * stride)
            k.attrs["description"] = ("effective cos(contact angle) kappa_eff(x, y) on the grid "
                                      "x = x0 + i*dx, y = x0 + j*dx (array index [i, j])")
        ck = d.create_group("checkpoint")
        ck.attrs["current"] = np.int64(-1)
        for slot in ("slot0", "slot1"):
            g = ck.create_group(slot)
            _create(g, "verts", (3,), "f8", compress=False)
            _create(g, "tris", (3,), "i8", compress=False)
            _create(g, "boundary_idx", dtype="i8", compress=False)
            _create(g, "quasi_ckpt_verts", (3,), "f8", compress=False)
            _create(g, "state", dtype="u1", compress=False)    # JSON bytes

    # -- truncation for restarts --------------------------------------------

    def _truncate_to(self, n):
        """Drops frames n, n+1, ... (and events after the last kept frame)."""
        v, d = self.f["VTKHDF"], self.f[GROUP]
        st = v["Steps"]
        nsteps = int(st.attrs["NSteps"])
        if n >= nsteps:
            return
        n_points = int(st["PointOffsets"][n])
        cells = int(st["CellOffsets"][n, POLYGONS])
        conn = int(st["ConnectivityIdOffsets"][n, POLYGONS])
        st.attrs["NSteps"] = np.int64(n)
        for name in ("Values", "PartOffsets", "NumberOfParts", "PointOffsets", "CellOffsets", "ConnectivityIdOffsets"):
            _truncate(st[name], n)
        for grp in ("PointDataOffsets", "CellDataOffsets", "FieldDataOffsets", "FieldDataSizes"):
            for ds in st[grp].values():
                _truncate(ds, n)
        _truncate(v["NumberOfPoints"], n)
        _truncate(v["Points"], n_points)
        for topo in TOPOLOGIES:
            g = v[topo]
            _truncate(g["NumberOfCells"], n)
            _truncate(g["NumberOfConnectivityIds"], n)
        g = v["Polygons"]
        _truncate(g["Offsets"], cells + n)
        _truncate(g["Connectivity"], conn)
        for topo in ("Vertices", "Lines", "Strips"):
            _truncate(v[topo]["Offsets"], n)
        for a in POINT_ARRAYS:
            _truncate(v["PointData"][a.name], n_points)
        for a in CELL_ARRAYS:
            _truncate(v["CellData"][a.name], cells)
        for a in FIELD_ARRAYS:
            _truncate(v["FieldData"][a.name], n)
        fr = d["frames"]
        for name, off in RAGGED.items():
            _truncate(fr[name], int(fr[off][n]))
            _truncate(fr[off], n + 1)
        for name in ("epoch", "step_iters", "converged"):
            _truncate(fr[name], n)

    def _truncate_events_after(self, iteration):
        ev = self.f[GROUP]["events"]
        its = ev["iteration"][:]
        keep = int(np.searchsorted(its, iteration, side="right"))
        for ds in ev.values():
            _truncate(ds, keep)

    # -- sink interface -------------------------------------------------------

    def start(self, run):
        if self.append:
            self.f = h5py.File(self.path, "r+", libver=LIBVER)
            if run.restart_frames is not None:
                self._truncate_to(run.restart_frames)
                self._truncate_events_after(run.start_iteration)
            d = self.f[GROUP]
            d["status"][0] = b"running"
            d["end_iteration"][0] = -1
        else:
            if os.path.exists(self.path):
                os.remove(self.path)
            self.f = h5py.File(self.path, "w", libver=LIBVER)
            self._create_layout(run)
        self.f.flush()
        self.f.swmr_mode = True

    def frame(self, fr):
        v, d = self.f["VTKHDF"], self.f[GROUP]
        st = v["Steps"]
        n = int(st.attrs["NSteps"])
        N, M = len(fr.verts), len(fr.tris)

        p0 = _append(v["Points"], fr.verts)
        _append(v["NumberOfPoints"], [N])
        poly = v["Polygons"]
        c0 = poly["Offsets"].shape[0] - n          # cells written so far (each part has M+1 offsets)
        k0 = _append(poly["Connectivity"], np.asarray(fr.tris, dtype=np.int64).ravel())
        _append(poly["Offsets"], np.arange(M + 1, dtype=np.int64) * 3)
        _append(poly["NumberOfCells"], [M])
        _append(poly["NumberOfConnectivityIds"], [3 * M])
        for topo in ("Vertices", "Lines", "Strips"):
            g = v[topo]
            _append(g["NumberOfCells"], [0])
            _append(g["NumberOfConnectivityIds"], [0])
            _append(g["Offsets"], [0])

        contact = np.zeros(N, np.uint8)
        contact[fr.boundary_idx] = 1
        sphere_line = np.zeros(N, np.uint8)
        if fr.wetted_loop is not None:
            sphere_line[fr.wetted_loop] = 1
        point_values = {"mean_curvature": fr.curvature, "contact_line": contact, "sphere_contact_line": sphere_line}
        for a in POINT_ARRAYS:
            _append(v["PointData"][a.name], point_values[a.name])
            _append(st["PointDataOffsets"][a.name], [p0])
        base = base_triangle_mask(fr.verts, fr.tris, fr.boundary_idx).astype(np.uint8)
        for a in CELL_ARRAYS:
            _append(v["CellData"][a.name], {"base": base}[a.name])
            _append(st["CellDataOffsets"][a.name], [c0])
        fv = field_values(fr)
        for a in FIELD_ARRAYS:
            val = np.asarray(fv[a.name], dtype=a.dtype).reshape((1, a.components) if a.components > 1 else (1,))
            _append(v["FieldData"][a.name], val)
            _append(st["FieldDataOffsets"][a.name], [n])
            _append(st["FieldDataSizes"][a.name], [[a.components, 1]])   # [components, tuples]

        frs = d["frames"]
        _append(frs["epoch"], [fr.epoch])
        _append(frs["step_iters"], [fr.step_iters])
        _append(frs["converged"], [int(fr.converged)])
        for name, off in RAGGED.items():
            vals = getattr(fr, name)
            vals = np.zeros(0, np.int64) if vals is None else np.asarray(vals, np.int64)
            _append(frs[name], vals)
            _append(frs[off], [frs[name].shape[0]])

        _append(st["PartOffsets"], [n])
        _append(st["NumberOfParts"], [1])
        _append(st["PointOffsets"], [p0])
        offs = np.zeros(len(TOPOLOGIES), np.int64)
        offs[POLYGONS] = c0
        _append(st["CellOffsets"], [offs])
        coffs = np.zeros(len(TOPOLOGIES), np.int64)
        coffs[POLYGONS] = k0
        _append(st["ConnectivityIdOffsets"], [coffs])
        # last: Values (followed by NSteps) completes the frame; readers count
        # frames by them, so a half-written frame is never visible
        _append(st["Values"], [fr.x if self._quasi else float(fr.iteration)])
        st.attrs["NSteps"] = np.int64(n + 1)
        self.f.flush()

    @property
    def _quasi(self):
        return bool(self.f[GROUP].attrs["quasi_static"])

    def event(self, e):
        if e.kind not in ("remesh", "diverged"):
            return
        ev = self.f[GROUP]["events"]
        data = {k: (int(x) if isinstance(x, (int, np.integer)) else x) for k, x in e.data.items()
                if isinstance(x, (int, float, str, np.integer))}
        if e.message:
            data["message"] = e.message
        _append(ev["iteration"], [e.iteration])
        _append(ev["kind"], [e.kind.encode()])
        _append(ev["data"], [json.dumps(data)])
        self.f.flush()

    def checkpoint(self, state):
        ck = self.f[GROUP]["checkpoint"]
        cur = int(ck.attrs["current"])
        slot = 1 if cur == 0 else 0
        g = ck[f"slot{slot}"]
        for name in ("verts", "tris", "boundary_idx", "quasi_ckpt_verts"):
            val = state.get(name)
            val = np.zeros((0, 3) if name != "boundary_idx" else (0,)) if val is None else np.asarray(val)
            _truncate(g[name], 0)
            _append(g[name], val)
        meta = dict(state["scalars"], iteration=state["iteration"],
                    has_quasi_ckpt_verts=state.get("quasi_ckpt_verts") is not None)
        raw = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        _truncate(g["state"], 0)
        _append(g["state"], raw)
        self.f.flush()
        ck.attrs["current"] = np.int64(slot)       # switch only after the slot is complete
        self.f.flush()

    def finish(self, status):
        d = self.f[GROUP]
        d["status"][0] = status.reason.encode()[:16]
        d["end_iteration"][0] = status.iteration
        self.f.flush()
        self.f.close()
        self.f = None
