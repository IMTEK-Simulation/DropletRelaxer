"""Shared helpers for the test suite.

Fast tests (marker `fast`, default) check single formulas against independent
references (analytic values, finite differences, brute-force integrals).
Slow tests (marker `slow`) run real relaxations and compare equilibria and
forces with analytic geometry / numerical energy derivatives.
"""
import copy
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_collection_modifyitems(config, items):
    # every test that is not marked slow counts as fast
    for it in items:
        if "slow" not in it.keywords:
            it.add_marker(pytest.mark.fast)


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
