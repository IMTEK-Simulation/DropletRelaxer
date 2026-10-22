"""Regression guard: the reference scenarios of tools/reference_run.py must
reproduce the recorded final state (test/reference/*.json).

Scalars are compared with a tight relative tolerance. The SHA-256 of the final
mesh is compared only when DROPLET_REFERENCE_HASH=1 is set, i.e. on the machine
that recorded the reference: SuperLU/BLAS results can differ in the last bits
across CPUs and builds, so bit identity is a same-machine check."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import reference_run  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("name", sorted(reference_run.SCENARIOS))
def test_reference_run(name):
    with open(os.path.join(reference_run.REFERENCE_DIR, f"{name}.json")) as f:
        ref = json.load(f)
    out = reference_run.run(name)
    for key in ("iteration", "remeshes", "quasi_frames", "n_verts", "n_tris"):
        assert out[key] == ref[key], key
    for key in ("energy", "area", "volume", "contact_radius"):
        np.testing.assert_allclose(out[key], ref[key], rtol=1e-10, err_msg=key)
    if ref["force"] is not None:
        np.testing.assert_allclose(out["force"], ref["force"], rtol=1e-8, atol=1e-10, err_msg="force")
    if os.environ.get("DROPLET_REFERENCE_HASH") == "1":
        assert out["sha256"] == ref["sha256"]


def test_quasistatic_restart_is_bit_identical(tmp_path):
    """The sphere scenario killed mid-run and restarted from its last
    checkpoint ends in exactly the state of the uninterrupted run (quasi-static
    controller state, remesh bookkeeping and mesh all restored)."""
    from DropletRelaxer.config import Config
    from DropletRelaxer.driver import Driver
    from DropletRelaxer.io import NullSink
    from DropletRelaxer.io.reader import RunFile
    from DropletRelaxer.io.writer import H5Writer

    class Kill(NullSink):
        def tick(self, i, progress, eta):
            if i >= 17000:
                raise KeyboardInterrupt

    cfg = Config()
    reference_run.SCENARIOS["sphere"](cfg)
    cfg.output.checkpoint_every = 1000
    path = str(tmp_path / "sphere.h5")
    writer = H5Writer(path)
    with pytest.raises(KeyboardInterrupt):
        Driver(cfg, sinks=[writer, Kill()]).run()
    writer.f.close()
    with RunFile(path) as rf:
        ck = rf.checkpoint()
        cfg2, _ = rf.config()
    assert 0 < ck["iteration"] < 17000
    res = Driver(cfg2, sinks=[H5Writer(path, append=True)], restart=ck).run()
    with open(os.path.join(reference_run.REFERENCE_DIR, "sphere.json")) as f:
        ref = json.load(f)
    np.testing.assert_allclose(res["energy"], ref["energy"], rtol=1e-10)
    assert res["iteration"] == ref["iteration"] and res["quasi_frames"] == ref["quasi_frames"]
    if os.environ.get("DROPLET_REFERENCE_HASH") == "1":
        import hashlib
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(res["verts"], dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(res["tris"], dtype=np.int64).tobytes())
        assert h.hexdigest() == ref["sha256"]
