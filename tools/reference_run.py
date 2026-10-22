"""Records the reference runs used by the regression test (test/test_reference.py).

    python tools/reference_run.py            # rewrite all reference JSON files
    python tools/reference_run.py sessile    # only one scenario

Each scenario is a short, deterministic, headless run that exercises
remeshing (and, for "sphere", the quasi-static sphere stepping, wetting and the
force on the sphere). The JSON file holds the final scalars and a SHA-256 of the
final mesh bytes; regenerate it deliberately, in its own commit, whenever the
numerics change on purpose.
"""
import copy
import hashlib
import json
import os
import sys

import numpy as np

import DropletRelaxer.config as _config
from DropletRelaxer.console import ConsoleSink
from DropletRelaxer.driver import Driver

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REFERENCE_DIR = os.path.join(ROOT, "test", "reference")


def _sessile(cfg):
    cfg.obstacle.enabled = False
    cfg.roughness.enabled = False
    cfg.quasi.enabled = False
    cfg.solver.contact_angle_deg = 120.0
    cfg.solver.n_iterations = 6000
    # make both remesh kinds fire early: a global split, then a collapse
    cfg.mesh.remesh_split_mode = "global"
    cfg.mesh.remesh_trigger_ratio = 1.3
    cfg.mesh.remesh_min_edge_ratio = 0.6


def _sphere(cfg):
    cfg.obstacle.center_end = (0.2, 0.0, 125.0)   # 2 quasi-static steps of 0.1
    cfg.solver.n_iterations = 200000


SCENARIOS = {"sessile": _sessile, "sphere": _sphere}


def run(name, sinks=()):
    cfg = copy.deepcopy(_config.CONFIG)
    SCENARIOS[name](cfg)
    res = Driver(cfg, sinks=sinks).run()
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(res["verts"], dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(res["tris"], dtype=np.int64).tobytes())
    out = {k: res[k] for k in ("iteration", "energy", "area", "volume", "contact_radius",
                               "remeshes", "force", "quasi_frames")}
    out["n_verts"] = int(len(res["verts"]))
    out["n_tris"] = int(len(res["tris"]))
    out["sha256"] = h.hexdigest()
    return out


def main(names):
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    for name in names or SCENARIOS:
        out = run(name, sinks=[ConsoleSink()])
        with open(os.path.join(REFERENCE_DIR, f"{name}.json"), "w") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
