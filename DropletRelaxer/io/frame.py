"""The data that flows from the driver to its outputs.

The driver (driver.py) knows nothing about files or windows. It hands a
`RunInfo` to every sink at the start, then one `Frame` per recorded state,
`Event`s (remesh, divergence, log lines) as they happen, a cheap `tick` every
iteration, and finally a `RunStatus`. The HDF5 writer, the console output and
the viewer's live window are all sinks; the file reader yields the same
`Frame` objects, so a viewer has one input type whatever the source.

Sinks must not change what they receive: the arrays of a Frame are read-only,
and the only thing that may flow back to the solver is a `StopRequest`.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np


def readonly(a, dtype=None):
    """Read-only copy of `a` (None stays None)."""
    if a is None:
        return None
    a = np.array(a, dtype=dtype, copy=True)
    a.setflags(write=False)
    return a


class StopRequest:
    """Set by a sink (window closed, Stop button) or a signal handler; the
    driver checks it once per iteration and ends the run cleanly."""

    def __init__(self):
        self.requested = False
        self.reason = None

    def request(self, reason="stop requested"):
        if not self.requested:
            self.requested, self.reason = True, reason


@dataclass(frozen=True)
class Frame:
    """One recorded state. In quasi-static mode a frame is an equilibrated
    sphere step; otherwise every `output.every` iterations and at remeshes."""

    index: int                      # frame number, 0-based
    iteration: int
    x: float                        # chart abscissa: sphere centre x (quasi-static) or iteration
    epoch: int                      # topology epoch; increases with every remesh
    verts: np.ndarray               # (N, 3) float64
    tris: np.ndarray                # (M, 3) int64
    boundary_idx: np.ndarray        # contact-line ring on the substrate, cyclic order
    wetted_loop: Optional[np.ndarray]   # three-phase loop on the sphere (vertex indices) or None
    sphere_center: Optional[np.ndarray]  # (3,) or None without a sphere
    energy: float                   # surface free energy (analysis.surface_free_energy)
    area: float                     # liquid/vapor area
    volume: float
    force: np.ndarray               # (3,) force of the liquid on the sphere
    com: np.ndarray                 # (3,) centre of mass of the liquid
    com_offset_xy: np.ndarray       # (2,) COM_xy - sphere_xy (axis x = y = 0 without sphere)
    gap: float                      # sphere bottom above the substrate (NaN without sphere)
    curvature: np.ndarray           # (N,) mean curvature per vertex (NaN on the flat base)
    step_iters: int = -1            # quasi-static: iterations used by this step (-1 otherwise)
    converged: bool = True          # quasi-static: step converged within max_iter_per_step


@dataclass(frozen=True)
class Event:
    """Something that happened between frames.

    kind: "remesh" (data: remesh kind, n_verts_before/after, n_tris_before/after,
          and the new mesh: verts, tris, boundary_idx, wetted_loop, sphere_center),
          "diverged" (message), "log" (message: a console line)."""

    kind: str
    iteration: int
    message: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class RunInfo:
    """Static information about a run, handed to every sink at the start."""

    config: Any                     # effective DropletRelaxer.config.Config (after volume scaling)
    mesh_scale: float               # length factor applied by mesh.target_volume
    V0: float                       # target volume
    A0: float                       # initial liquid/vapor area
    verts: np.ndarray               # start mesh
    tris: np.ndarray
    boundary_idx: np.ndarray
    total_iters: int
    quasi_static: bool              # frames are equilibrated sphere steps
    roughness_field: Any = None     # roughness.RoughnessField or None
    stop: StopRequest = field(default_factory=StopRequest)
    start_iteration: int = 0        # > 0 when restarted from a checkpoint


@dataclass
class RunStatus:
    """How a run ended."""

    reason: str                     # "finished", "diverged", "stopped"
    iteration: int
    message: str = ""


class FrameSink(Protocol):
    def start(self, run: RunInfo) -> None: ...

    def frame(self, f: Frame) -> None: ...

    def event(self, e: Event) -> None: ...

    def tick(self, i: int, progress: float, eta: Optional[float]) -> None: ...

    def finish(self, status: RunStatus) -> None: ...


class NullSink:
    """Base class with no-op methods; subclass and override what you need."""

    def start(self, run):
        pass

    def frame(self, f):
        pass

    def event(self, e):
        pass

    def tick(self, i, progress, eta):
        pass

    def finish(self, status):
        pass
