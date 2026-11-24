"""Helpers for strict kernel parity tests (including signs of floating-point zeros)."""

import numpy as np


def assert_bits(actual, expected):
    """Check shape, dtype, and bytes in logical C order without a tolerance."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    # np.void compares the full item representation, not its interpreted value.
    typ = np.dtype(f"V{actual.dtype.itemsize}")
    got = actual.ravel().view(typ)
    want = expected.ravel().view(typ)
    off = np.flatnonzero(got != want)
    if len(off):
        k = int(off[0])
        where = np.unravel_index(k, actual.shape) if actual.ndim else ()
        raise AssertionError(
            f"{len(off)}/{got.size} elements differ in bits; first at {where}: "
            f"{actual[where]!r} ({got[k].tobytes().hex()}) != "
            f"{expected[where]!r} ({want[k].tobytes().hex()})"
        )
