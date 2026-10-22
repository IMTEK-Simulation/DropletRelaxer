"""The run file: HDF5 in the VTKHDF layout (temporal PolyData) plus a
/DropletRelaxer group. This module is the single definition of its contents;
writer.py, reader.py, the tests and the documentation page are built from it.

    /VTKHDF                     temporal PolyData, one part per frame; read by
                                ParaView and VTK >= 9.3 (vtkHDFReader)
      NumberOfPoints, Points, Polygons/{NumberOfCells, NumberOfConnectivityIds,
      Offsets, Connectivity}, Vertices/Lines/Strips (empty), PointData/*,
      CellData/*, FieldData/* (one tuple per frame), Steps/*
    /DropletRelaxer             attrs: format_version, code_version, created,
                                hostname, command_line, mesh_scale, V0, A0,
                                quasi_static, total_iters, roughness_edge
      config_toml               the scenario as given (after --set), complete
      config_effective          JSON, after volume scaling: what actually ran
      start/                    start mesh: verts, tris, boundary_idx
      frames/                   per-frame data VTK has no place for (ragged
                                contact-line rings, epochs, quasi-static info)
      events/                   remesh / divergence events
      substrate/                sampled roughness field (kappa_eff), if any
      checkpoint/               two alternating slots for restart; attr
                                `current` names the valid one (-1: none)
"""

from dataclasses import dataclass

import numpy as np

FORMAT_VERSION = 1
VTKHDF_VERSION = (2, 2)
GROUP = "DropletRelaxer"


@dataclass(frozen=True)
class Array:
    """One per-frame array. `location`: "field" (VTKHDF FieldData, one tuple
    per frame), "point" (PointData, one value per vertex), "cell" (CellData,
    one value per triangle) or "frames" (/DropletRelaxer/frames, one value
    per frame)."""

    name: str
    location: str
    dtype: str
    components: int
    description: str


ARRAYS = [
    # FieldData: one tuple per frame, plotted by the viewer
    Array("iteration", "field", "i8", 1, "iteration number of the frame"),
    Array("x", "field", "f8", 1, "chart abscissa: sphere centre x (quasi-static) or iteration"),
    Array("energy", "field", "f8", 1, "surface free energy (gamma_lv A_lv - cos(theta) gamma_lv A_sl + ...)"),
    Array("area_lv", "field", "f8", 1, "liquid/vapor area"),
    Array("volume", "field", "f8", 1, "liquid volume"),
    Array("gap", "field", "f8", 1, "height of the sphere's lowest point above the substrate (NaN: no sphere)"),
    Array("sphere_center", "field", "f8", 3, "sphere centre (NaN: no sphere)"),
    Array("force", "field", "f8", 3, "force of the liquid on the sphere (from the force balance in the solver)"),
    Array("com", "field", "f8", 3, "centre of mass of the liquid"),
    Array("com_offset_xy", "field", "f8", 2, "COM_xy - sphere_xy (axis x = y = 0 without sphere)"),
    # PointData
    Array("mean_curvature", "point", "f8", 1, "mean curvature per vertex (NaN on the flat base)"),
    Array("contact_line", "point", "u1", 1, "1 on the solid/liquid/vapor contact line on the substrate"),
    Array("sphere_contact_line", "point", "u1", 1, "1 on the three-phase contact line on the sphere"),
    # CellData
    Array("base", "cell", "u1", 1, "1 = liquid/solid base triangle on z = 0"),
    # /DropletRelaxer/frames
    Array("epoch", "frames", "i8", 1, "topology epoch (increases with every remesh)"),
    Array("step_iters", "frames", "i8", 1, "quasi-static: iterations of this step (-1 otherwise)"),
    Array("converged", "frames", "u1", 1, "quasi-static: step converged within max_iter_per_step"),
    Array("boundary_offsets", "frames", "i8", 1,
          "start of this frame's ring in boundary_idx (ragged, one more entry than frames)"),
    Array("loop_offsets", "frames", "i8", 1,
          "start of this frame's sphere loop in wetted_loop (ragged, one more entry than frames)"),
]

FIELD_ARRAYS = [a for a in ARRAYS if a.location == "field"]
POINT_ARRAYS = [a for a in ARRAYS if a.location == "point"]
CELL_ARRAYS = [a for a in ARRAYS if a.location == "cell"]

# ragged per-frame vertex lists in /DropletRelaxer/frames (indices into the frame's points)
RAGGED = {"boundary_idx": "boundary_offsets", "wetted_loop": "loop_offsets"}

TOPOLOGIES = ("Vertices", "Lines", "Polygons", "Strips")   # order of CellOffsets columns
POLYGONS = 2

EVENT_KINDS = ("remesh", "diverged")


def field_values(f):
    """{FieldData name: value} of a Frame."""
    nan3 = np.full(3, np.nan)
    return {
        "iteration": f.iteration, "x": f.x, "energy": f.energy, "area_lv": f.area, "volume": f.volume,
        "gap": f.gap, "sphere_center": nan3 if f.sphere_center is None else f.sphere_center,
        "force": f.force, "com": f.com, "com_offset_xy": f.com_offset_xy,
    }


def markdown_table():
    """The per-frame arrays as a Markdown table (for the documentation)."""
    where = {"field": "/VTKHDF/FieldData", "point": "/VTKHDF/PointData", "cell": "/VTKHDF/CellData",
             "frames": f"/{GROUP}/frames"}
    rows = ["| Name | Location | Type | Components | Description |", "|---|---|---|---|---|"]
    for a in ARRAYS:
        rows.append(f"| `{a.name}` | `{where[a.location]}` | `{a.dtype}` | {a.components} | {a.description} |")
    return "\n".join(rows)
