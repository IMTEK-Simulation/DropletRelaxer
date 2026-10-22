"""Numerical helpers for the tests: central finite differences, relative error."""
import numpy as np


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
