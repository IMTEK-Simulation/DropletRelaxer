"""Shared helpers for the test suite.

Fast tests (everything not marked `slow`; run by plain `pytest`) check single
formulas against independent references (analytic values, finite differences,
brute-force integrals). Slow tests (marker `slow`) run real relaxations and
compare equilibria and forces with analytic geometry / numerical energy
derivatives. Run them with `pytest -m slow`, everything with `pytest -m ""`,
or a single slow test by naming its node id.
"""
from pathlib import Path

import numpy as np
import pytest


def _requested_node_ids(config):
    """(path, name) for every `path::name` argument on the command line, with
    `path` resolved against the directory pytest was started from."""
    out = []
    for a in config.args:
        path, sep, name = str(a).partition("::")
        if sep:
            out.append(((config.invocation_params.dir / path).resolve(), name))
    return out


def _mark_expression_given(config):
    return any(a == "-m" or (a.startswith("-m") and not a.startswith("--"))
               for a in config.invocation_params.args)


def pytest_collection_modifyitems(config, items):
    """Slow tests run only on request: with any -m expression (`-m slow`, or
    `-m ""` for everything), or when the test is named by its node id."""
    if _mark_expression_given(config):
        return
    requested = _requested_node_ids(config)
    keep, drop = [], []
    for it in items:
        name = it.nodeid.split("::", 1)[1]
        named = any(Path(it.path).resolve() == p and (name == n or name.startswith(n + "[")
                                                       or name.startswith(n + "::"))
                    for p, n in requested)
        (drop if "slow" in it.keywords and not named else keep).append(it)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def fd_gradient(f, x, h=1e-6, idx=None):
    """Central finite-difference gradient of the scalar function f(x) for
    array x (all entries, or only the flat indices `idx`)."""
    x = np.array(x, dtype=float)
    flat = x.reshape(-1)
    n = flat.size
    idx = range(n) if idx is None else idx
    g = np.zeros(n)
    for k in idx:
        old = flat[k]
        flat[k] = old + h
        fp = f(x.copy())
        flat[k] = old - h
        fm = f(x.copy())
        flat[k] = old
        g[k] = (fp - fm) / (2 * h)
    return g.reshape(x.shape)


def rel_err(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)
