"""Regression guard: the reference scenarios of tools/reference_run.py must
reproduce the recorded final state (tests/reference/*.json).

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
