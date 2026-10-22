"""
Random roughness field on the substrate (optional wall-energy term).

Physics
-------
Young:        gamma_sv - gamma_sl = gamma_lv * kappa,   kappa = cos(theta_Y)
Wall energy:  E_wall = -gamma_lv * Integral_wetted kappa(x, y) dA

Two ways to make kappa(x, y) random (RoughnessConfig.mode):

  "topography"  A Gaussian random HEIGHT field h(x, y) with a band-limited
                white spectrum (plateau for q_L <= |q| <= q_c, zero above).
                The local Wenzel factor r = sqrt(1 + |grad h|^2) is the ratio of
                true to projected solid area. Per projected area the wall energy
                is multiplied by r, so
                    kappa_eff(x, y) = clip(r(x, y) * kappa_Y, -1, 1),
                    cos(theta_eff)  = r * cos(theta_Y)        (Wenzel, local).
                NOTE: at theta_Y = 90 deg (kappa_Y = 0) this has no effect.

  "chemical"    kappa itself fluctuates: kappa = kappa_Y + delta_kappa with a
                Gaussian band-limited delta_kappa of rms `rms_kappa`
                (equivalently the surface-energy difference gamma_sv-gamma_sl
                fluctuates by gamma_lv * delta_kappa).

Only the DIFFERENCE to the homogeneous wall, delta_kappa = kappa_eff - kappa_Y,
is added here (the homogeneous part stays in the base triangles / wall term):

    E_rough = -gamma_lv * Integral_wetted delta_kappa(x, y) dA

The integral over the wetted polygon (contact line) is evaluated EXACTLY for the
bilinear interpolant of delta_kappa on the grid, as a contact-line flux
(divergence theorem with the potential F(x, y) = Integral_{x0}^{x} delta_kappa dx'),
and the gradient is the shape derivative  dI/dr_i = Sum_edges Integral delta_kappa *
phi_i * n dl   (acts on the contact-line vertices only, in x and y).

Cutoff: the field only enters through the contact line, so the contact-line edge
length `l` sets the resolution. Default  lambda_c = 4 l, q_c = 2 pi / (4 l) = pi / (2 l).
Then even the squared slope (r - 1 ~ |grad h|^2/2, bandwidth 2 q_c) stays below the
Nyquist limit pi / l of the contact line.
"""
from __future__ import annotations

import math
import warnings
import numpy as np


_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(4)
_GAUSS_X = 0.5 * (_GAUSS_X + 1.0)
_GAUSS_W = 0.5 * _GAUSS_W


class RoughnessField:
    """delta_kappa(x, y) on a grid + contact-line energy/gradient."""

    def __init__(self, cfg, kappa0, q_L, q_c, L, N, h, dkappa, kappa_eff, info):
        self.cfg = cfg
        self.kappa0 = kappa0
        self.q_L, self.q_c, self.L, self.N = q_L, q_c, L, N
        self.dx = L / N
        self.x0 = -0.5 * L
        self.h = h                      # height field (topography) or delta_kappa (chemical)
        self.kappa_eff = kappa_eff      # (N, N), for plotting
        self.info = info
        # periodic wrap to N+1 nodes: array index [j (y), i (x)]
        d = np.pad(dkappa, ((0, 1), (0, 1)), mode="wrap")
        self._d = d
        F = np.zeros_like(d)
        F[:, 1:] = np.cumsum(0.5 * self.dx * (d[:, :-1] + d[:, 1:]), axis=1)
        self._F = F
        self._warned_outside = False

    # ------------------------------------------------------------------ eval
    def _locate(self, x, y):
        u = (x - self.x0) / self.dx
        v = (y - self.x0) / self.dx
        outside = (u < 0) | (u > self.N) | (v < 0) | (v > self.N)
        if outside.any() and not self._warned_outside:
            warnings.warn("roughness: contact line left the field box "
                          f"[-{self.L/2:g}, {self.L/2:g}]^2 -- increase RoughnessConfig.box_size")
            self._warned_outside = True
        u = np.clip(u, 0.0, self.N - 1e-9)
        v = np.clip(v, 0.0, self.N - 1e-9)
        i = np.floor(u).astype(int)
        j = np.floor(v).astype(int)
        return i, j, u - i, v - j

    def delta_kappa(self, x, y):
        """Bilinear delta_kappa at points (arrays)."""
        i, j, t, w = self._locate(x, y)
        d = self._d
        r0 = d[j, i] * (1 - t) + d[j, i + 1] * t
        r1 = d[j + 1, i] * (1 - t) + d[j + 1, i + 1] * t
        return (1 - w) * r0 + w * r1

    def _potential(self, x, y):
        """F(x, y) = Integral_{x0}^{x} delta_kappa(x', y) dx' (exact for the bilinear field)."""
        i, j, t, w = self._locate(x, y)
        d, F, dx = self._d, self._F, self.dx

        def row(jj):
            return F[jj, i] + dx * (d[jj, i] * t + 0.5 * (d[jj, i + 1] - d[jj, i]) * t * t)

        return (1 - w) * row(j) + w * row(j + 1)

    # ------------------------------------------------------------ line integral
    def _quadrature(self, P):
        """Split every edge at the grid lines it crosses; 4-point Gauss on each piece.

        Inside one piece the potential F(x(s), y(s)) is a cubic polynomial in s (quadratic
        in x times linear in y), so 4-point Gauss is exact: energy and gradient are exact
        for the bilinear field, which makes the gradient the exact derivative of the energy.
        """
        a = P
        e = np.roll(P, -1, axis=0) - P
        n = len(P)
        ua, ub = (a[:, 0] - self.x0) / self.dx, (a[:, 0] + e[:, 0] - self.x0) / self.dx
        va, vb = (a[:, 1] - self.x0) / self.dx, (a[:, 1] + e[:, 1] - self.x0) / self.dx

        def crossings(c0, c1):
            lo, hi = np.minimum(c0, c1), np.maximum(c0, c1)
            first = np.floor(lo) + 1.0
            cnt = np.maximum(0, np.ceil(hi) - first).astype(int)
            K = int(cnt.max()) if n else 0
            out = np.ones((n, K))
            d = c1 - c0
            safe = np.where(d == 0.0, 1.0, d)
            for k in range(K):
                sk = (first + k - c0) / safe
                out[:, k] = np.where(k < cnt, sk, 1.0)
            return out

        S = np.concatenate([np.zeros((n, 1)), np.ones((n, 1)),
                            crossings(ua, ub), crossings(va, vb)], axis=1)
        S = np.sort(np.clip(S, 0.0, 1.0), axis=1)
        lo, wid = S[:, :-1], S[:, 1:] - S[:, :-1]                      # (n, P)
        s = (lo[:, :, None] + wid[:, :, None] * _GAUSS_X[None, None, :]).reshape(n, -1)
        ws = (wid[:, :, None] * _GAUSS_W[None, None, :]).reshape(n, -1)
        pts = a[:, None, :] + s[:, :, None] * e[:, None, :]
        return a, e, s, ws, pts

    def wetted_integral_and_grad(self, verts, loop):
        """I = Integral_wetted delta_kappa dA and dI/dverts (n_verts, 3; xy only)."""
        loop = np.asarray(loop)
        P = verts[loop, :2]
        Pn = np.roll(P, -1, axis=0)
        signed_area = 0.5 * np.sum(P[:, 0] * Pn[:, 1] - Pn[:, 0] * P[:, 1])
        orient = 1.0 if signed_area >= 0 else -1.0

        a, e, s, ws, pts = self._quadrature(P)
        X, Y = pts[..., 0], pts[..., 1]
        I = orient * np.sum(ws * self._potential(X, Y) * e[:, 1:2])    # oint F dy

        kap = self.delta_kappa(X, Y)
        nvec = np.stack([e[:, 1], -e[:, 0]], axis=1)                   # n * dl for CCW
        wk_start = np.sum(ws * (1.0 - s) * kap, axis=1)                # hat function of the start vertex
        wk_end = np.sum(ws * s * kap, axis=1)                          # hat function of the end vertex
        gI = orient * (wk_start[:, None] * nvec + np.roll(wk_end[:, None] * nvec, 1, axis=0))
        grad = np.zeros_like(verts)
        grad[loop, 0] = gI[:, 0]
        grad[loop, 1] = gI[:, 1]
        return I, grad

    def energy_and_grad(self, verts, loop, gamma_lv=1.0):
        """E_rough = -gamma_lv * Integral_wetted delta_kappa dA, and its gradient."""
        I, gI = self.wetted_integral_and_grad(verts, loop)
        return -gamma_lv * I, -gamma_lv * gI

    # ------------------------------------------------------------------ helpers
    def check_resolution(self, verts, loop):
        """Message on the contact-line resolution (warns if lambda_c < 2 l)."""
        P = verts[np.asarray(loop), :2]
        l = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
        lam_c = 2 * math.pi / self.q_c
        msg = (f"contact-line edge length: mean {l.mean():.4g}, max {l.max():.4g}; "
               f"lambda_c = {lam_c:.4g} = {lam_c / l.max():.2f} x l_max")
        if lam_c < 2.0 * l.max():
            msg += "  -> WARNING: field not resolved by the contact line (need lambda_c >= 2 l)"
        return msg

    def describe(self):
        i = self.info
        lines = [
            f"mode            : {self.cfg.mode}",
            f"box L / grid    : {self.L:g} / {self.N}^2  (dx = {self.dx:.4g})",
            f"q_L, q_c        : {self.q_L:.4g}, {self.q_c:.4g}   (lambda_c = {2*math.pi/self.q_c:.4g})",
            f"sample rms      : {i['rms_primary']:.4g}",
        ]
        if self.cfg.mode == "topography":
            lines += [
                f"rms height, slope: {i['rms_h']:.4g}, {i['rms_slope']:.4g}",
                f"C0 (height PSD) : {i['C0']:.4g}   [sigma_h^2 = C0 (q_c^2 - q_L^2)/(4 pi)]",
                f"<r>, max r      : {i['r_mean']:.4f}, {i['r_max']:.4f}",
                f"mean theta_eff  : {math.degrees(math.acos(np.clip(self.kappa_eff.mean(), -1, 1))):.2f} deg "
                f"(theta_Y = {math.degrees(math.acos(self.kappa0)):.2f} deg)",
            ]
        lines.append(f"clipped fraction: {i['clipped']:.2%}")
        return "\n".join(lines)


def _count_modes(N, dx, q_lo, q_hi):
    q = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    QX, QY = np.meshgrid(q, q)
    Q = np.hypot(QX, QY)
    m = (Q >= q_lo) & (Q <= q_hi)
    return m.sum(), (Q[m] ** 2).sum()


def make_roughness_field(cfg, edge_length: float, kappa0: float) -> RoughnessField:
    """Build the field.

    cfg         : config.RoughnessConfig (all settings live in config.py)

    edge_length : typical contact-line edge length (physical units) -> sets q_c
    kappa0      : cos(theta_Y) of the homogeneous wall (SolverConfig.contact_angle_deg)
    """
    L = float(cfg.box_size)
    q_L = 2.0 * math.pi / L
    q_c = float(cfg.q_cutoff) if cfg.q_cutoff else 2.0 * math.pi / (cfg.wavelengths_per_edge * edge_length)
    if q_c < 2.0 * q_L:
        raise ValueError(f"cutoff q_c={q_c:.4g} too close to q_L={q_L:.4g}: increase box_size "
                         "or decrease the edge length")
    dx_target = (2.0 * math.pi / q_c) / cfg.points_per_wavelength
    N = 2 ** int(math.ceil(math.log2(L / dx_target)))
    if N > cfg.max_grid:
        warnings.warn(f"roughness grid {N}^2 capped to {cfg.max_grid}^2 (fewer points per wavelength)")
        N = cfg.max_grid
    dx = L / N

    rng = np.random.default_rng(cfg.seed)
    noise = rng.standard_normal((N, N))                       # [y, x]
    Fh = np.fft.rfft2(noise)
    QX = 2 * np.pi * np.fft.rfftfreq(N, d=dx)[None, :]
    QY = 2 * np.pi * np.fft.fftfreq(N, d=dx)[:, None]
    Q = np.hypot(QX, QY)
    mask = (Q >= 0.999 * q_L) & (Q <= q_c)                    # plateau, hard cutoff, no q = 0
    Fh = Fh * mask
    f = np.fft.irfft2(Fh, s=(N, N))
    gx = np.fft.irfft2(1j * QX * Fh, s=(N, N))
    gy = np.fft.irfft2(1j * QY * Fh, s=(N, N))
    n_modes, sum_q2 = _count_modes(N, dx, 0.999 * q_L, q_c)

    info = {}
    if cfg.mode == "topography":
        if cfg.rms_height is not None:
            slope_t = cfg.rms_height * math.sqrt(0.5 * (q_c ** 2 + q_L ** 2))
        else:
            slope_t = cfg.rms_slope
        s_samp = math.sqrt(np.mean(gx ** 2 + gy ** 2))
        s_exp = math.sqrt(sum_q2 / N ** 2)
        scale = slope_t / (s_samp if cfg.normalize_sample else s_exp)
        h, gx, gy = f * scale, gx * scale, gy * scale
        r = np.sqrt(1.0 + gx ** 2 + gy ** 2)
        raw = kappa0 * r
        kappa_eff = np.clip(raw, -1.0, 1.0)
        sig_h = float(h.std())
        info.update(
            rms_primary=sig_h, rms_h=sig_h, rms_slope=float(math.sqrt(np.mean(gx ** 2 + gy ** 2))),
            C0=4 * math.pi * sig_h ** 2 / (q_c ** 2 - q_L ** 2),
            r_mean=float(r.mean()), r_max=float(r.max()),
            clipped=float(np.mean(raw != kappa_eff)),
        )
        primary = h
    elif cfg.mode == "chemical":
        s_samp = float(f.std())
        s_exp = math.sqrt(n_modes) / N
        scale = cfg.rms_kappa / (s_samp if cfg.normalize_sample else s_exp)
        primary = f * scale
        raw = kappa0 + primary
        kappa_eff = np.clip(raw, -1.0, 1.0)
        info.update(rms_primary=float(primary.std()), clipped=float(np.mean(raw != kappa_eff)))
    else:
        raise ValueError("RoughnessConfig.mode must be 'topography' or 'chemical'")

    dkappa = kappa_eff - kappa0
    return RoughnessField(cfg, kappa0, q_L, q_c, L, N, primary, dkappa, kappa_eff, info)
