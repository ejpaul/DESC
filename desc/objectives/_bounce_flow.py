"""Prompt trapped-particle loss from the transfer operator of the bounce-averaged flow.

The trapped motion of a pitch class (mirror field B_c) is reduced to the (s, α) mirror
section.  To first order in the drift the one-bounce displacement is the quadrature of
the guiding-centre drift along the unperturbed field line between the mirror points,

    D(s, α) = 2 ∫_{-π/2}^{π/2} (ds/dθ, dα/dθ) dθ,   ζ = c + Δ sin θ,   B(c ± Δ) = B_c,

with ds/dt = v_d·∇s, dα/dt = v_d·∇α and dζ/dt = v_∥ B^ζ/B.  D is the velocity of an
integrable flow on the section whose orbits are the level sets of the longitudinal
invariant; its time-one map M is integrated with RK4 substeps of a bicubic interpolant
of D on a grid of the section.  The map is discretised as a Markov transition operator K
(quadratic B-spline weights of the landing points, C^1 in M, wall as an absorbing ghost
row) and the loss of the class is the exponential-clock hitting probability of the wall,

    (I − e^{−ν} K) u = e^{−ν} w_wall,     J_c = ∫ μ_c u / ∫ μ_c,

where ν is the clock rate per bounce (mean horizon 1/ν bounces) and μ_c = W_c S ds dα is
the birth measure of the class on the section (W_c the loss-cone moment of the birth
distribution, S the radial birth profile).  The objective returns the contribution of
each class to the prompt loss fraction of the whole distribution,

    f_c = NFP ω_c ∫ μ_c u_c / ∫ S dV,

with ω_c the quadrature weight of the class in B_c, so that ``loss_function="sum"``
gives the total loss fraction.

The chain equilibrium → tables → D → M is differentiated by JAX (the mirror points enter
through the implicit-function derivative of B(ζ) = B_c); the operator solve is a host
callback with a custom VJP whose backward pass is one adjoint solve, so the objective
supports ``deriv_mode="rev"`` only.

References
----------
Campaign record: Princeton Dropbox data/August_2026/20260826_squid_driftsurf_flux_fit,
methods note "A loss functional from the bounce map" (E. J. Paul), §3 and §6.
"""

from functools import partial

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from interpax import interp2d
from numpy.polynomial.legendre import leggauss

from desc.backend import jax, jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.utils import errorif, setdefault

from .objective_funs import _Objective, collect_docs

_MU0 = 4e-7 * np.pi
_ALPHA_M = 6.644657230e-27
_ALPHA_Q = 2 * 1.602176634e-19
_E_ALPHA = 3.52e6 * 1.602176634e-19

TABLE_KEYS = ["B", "Bz", "A_s", "A_t", "A_z", "C_s", "C_t", "C_z", "Jz"]


# --------------------------------------------------------------------------
# geometry tables on the (s, θ_PEST, ζ) grid, from the equilibrium
# --------------------------------------------------------------------------


def _regrid_theta_pest(F, thP, thp_g):
    """Resample F(ρ, θ, ζ) to a uniform θ_PEST grid, column by column.

    θ_PEST = θ + λ is increasing in θ when 1 + λ_θ > 0, so each (ρ, ζ) column is a
    monotone 1-D interpolation, made periodic by extension.  Differentiable in both
    the abscissae (λ) and the values.
    """

    def col(x, y):
        xe = jnp.concatenate([x - 2 * jnp.pi, x, x + 2 * jnp.pi])
        ye = jnp.concatenate([y, y, y])
        return jnp.interp(thp_g, xe, ye)

    # F, thP: (nr, nt, nz) -> vmap over ρ and ζ
    f = jax.vmap(jax.vmap(col, in_axes=(1, 1), out_axes=1), in_axes=(0, 0))
    return f(thP, F)


def tables_from_data(data, grid, rho, nfp, v2, nthp):
    """Class-independent guiding-centre tables on (s, θ_PEST, ζ) from DESC data.

    v_d·∇X = w A_X + (v² − w) C_X with w = v_∥², for X ∈ {s, θ_PEST, ζ}:
    A_X = (m/qB) (b×κ)·∇X, C_X = (m/2qB²) (b×∇|B|)·∇X, κ the field-line curvature.
    Jz = |√g| / (2ρ(1+λ_θ)) so that d³x = Jz ds dα dζ.
    """

    def r(k):
        return grid.meshgrid_reshape(data[k], "rtz")

    def rv(k):
        return jnp.stack(
            [grid.meshgrid_reshape(data[k][:, i], "rtz") for i in range(3)], -1
        )

    B = r("|B|")
    Bz = r("B^zeta")
    b = rv("b")
    gB = rv("grad(|B|)")
    e_r, e_t, e_z = rv("e^rho"), rv("e^theta"), rv("e^zeta")
    lam, l_r, l_t, l_z = r("lambda"), r("lambda_r"), r("lambda_t"), r("lambda_z")
    kap = rv("kappa")
    sg = r("sqrt(g)")
    theta = grid.meshgrid_reshape(grid.nodes[:, 1], "rtz")
    zeta = grid.compress(grid.nodes[:, 2], surface_label="zeta")
    R = rho[:, None, None]
    grad_s = 2 * R[..., None] * e_r
    grad_tP = l_r[..., None] * e_r + (1 + l_t)[..., None] * e_t + l_z[..., None] * e_z
    bxk = jnp.cross(b, kap)
    bxg = jnp.cross(b, gB)
    cA = _ALPHA_M / (_ALPHA_Q * B)
    cC = _ALPHA_M / (2 * _ALPHA_Q * B**2)
    raw = {
        "B": B,
        "Bz": Bz,
        "Jz": jnp.abs(sg) / (2 * R * (1 + l_t)),
        "A_s": cA * jnp.sum(bxk * grad_s, -1),
        "A_t": cA * jnp.sum(bxk * grad_tP, -1),
        "A_z": cA * jnp.sum(bxk * e_z, -1),
        "C_s": cC * jnp.sum(bxg * grad_s, -1),
        "C_t": cC * jnp.sum(bxg * grad_tP, -1),
        "C_z": cC * jnp.sum(bxg * e_z, -1),
    }
    thP = theta + lam
    thp_g = jnp.linspace(0, 2 * jnp.pi, nthp, endpoint=False)
    tb = {k: _regrid_theta_pest(v, thP, thp_g) for k, v in raw.items()}
    iota = grid.compress(data["iota"])
    iota_r = grid.compress(data["iota_r"])
    tb.update(
        s=rho**2,
        thp=thp_g,
        z=zeta,
        iota=iota,
        iota_s=iota_r / (2 * rho),
        T=2 * np.pi / nfp,
        v2=v2,
    )
    return tb


# --------------------------------------------------------------------------
# sampling the tables along frozen field lines
# --------------------------------------------------------------------------


def _interp3(field, s_g, thp_g, z_g, T, s, thp, z):
    """Trilinear interpolation, periodic in θ_PEST (2π) and ζ (T)."""
    ns, nthp, nz = field.shape
    i1 = jnp.clip(jnp.searchsorted(s_g, s), 1, ns - 1)
    i0 = i1 - 1
    ws = jnp.clip((s - s_g[i0]) / (s_g[i1] - s_g[i0]), 0.0, 1.0)
    dt = thp_g[1] - thp_g[0]
    ft = (thp % (2 * jnp.pi)) / dt
    j0 = jnp.floor(ft).astype(int) % nthp
    j1 = (j0 + 1) % nthp
    wt = ft - jnp.floor(ft)
    dz = z_g[1] - z_g[0]
    fz = (z % T) / dz
    k0 = jnp.floor(fz).astype(int) % nz
    k1 = (k0 + 1) % nz
    wz = fz - jnp.floor(fz)
    c00 = field[i0, j0, k0] * (1 - wz) + field[i0, j0, k1] * wz
    c01 = field[i0, j1, k0] * (1 - wz) + field[i0, j1, k1] * wz
    c10 = field[i1, j0, k0] * (1 - wz) + field[i1, j0, k1] * wz
    c11 = field[i1, j1, k0] * (1 - wz) + field[i1, j1, k1] * wz
    return (c00 * (1 - wt) + c01 * wt) * (1 - ws) + (c10 * (1 - wt) + c11 * wt) * ws


def sample(tb, s, alpha, zeta, keys=TABLE_KEYS):
    """Tables at (s, α, ζ) on the field line θ_PEST = α + ι(s) ζ."""
    iota = jnp.interp(s, tb["s"], tb["iota"])
    iota_s = jnp.interp(s, tb["s"], tb["iota_s"])
    thp = alpha + iota * zeta
    out = {
        k: _interp3(tb[k], tb["s"], tb["thp"], tb["z"], tb["T"], s, thp, zeta)
        for k in keys
    }
    out["iota"] = iota
    out["iota_s"] = iota_s
    return out


def drift_integrands(tb, s, alpha, zeta, Bc, h, th, floor=1e-4):
    """(ds/dθ, dα/dθ, dW/dθ) along the arc ζ = c + h sin θ (regular at the tips)."""
    fx = sample(tb, s, alpha, zeta)
    w = tb["v2"] * jnp.maximum(1.0 - fx["B"] / Bc, floor)
    vd_s = w * fx["A_s"] + (tb["v2"] - w) * fx["C_s"]
    vd_t = w * fx["A_t"] + (tb["v2"] - w) * fx["C_t"]
    vd_z = w * fx["A_z"] + (tb["v2"] - w) * fx["C_z"]
    vd_a = vd_t - fx["iota"] * vd_z - fx["iota_s"] * zeta * vd_s
    cth = jnp.cos(th)
    vpar = jnp.sqrt(w) * jnp.sign(cth)
    dth_dt = vpar * fx["Bz"] / fx["B"] / (h * cth)
    # loss-cone birth moment: Jz B / (2 B_c^2 sqrt(1 - B/B_c)) dζ, dζ = h cos θ dθ
    f = jnp.maximum(1.0 - fx["B"] / Bc, 1e-8)
    dW = fx["Jz"] * fx["B"] / (2 * Bc**2) / jnp.sqrt(f) * h * cth
    return vd_s / dth_dt, vd_a / dth_dt, dW


# --------------------------------------------------------------------------
# mirror points: scan for the well-0 tips (no gradient), then Newton (implicit gradient)
# --------------------------------------------------------------------------


def well0_tips_guess(tb, Bc, s, alpha, nscan=192):
    """Tips of the well containing the |B| minimum of the reference field period.

    Marches outward from the minimum to the first crossings of B_c on a dense ζ scan
    over three periods; returns (zL, zR, valid, dζ).  A class whose mirror points fall
    more than one period from the minimum is not counted.  Not differentiated.
    """
    T = tb["T"]
    z = jnp.linspace(-T, 2 * T, 3 * nscan, endpoint=False)
    B = jax.vmap(lambda zz: sample(tb, s, alpha, zz, keys=["B"])["B"])(z)
    mid = (z >= 0) & (z < T)
    im = jnp.argmin(jnp.where(mid, B, jnp.inf))
    k = jnp.arange(z.size)
    above = B >= Bc
    right = above & (k > im)
    left = above & (k < im)
    hasR = jnp.any(right)
    hasL = jnp.any(left)
    jR = jnp.argmax(right)  # first index above B_c to the right
    jL = z.size - 1 - jnp.argmax(left[::-1])  # last index above B_c to the left
    dz = z[1] - z[0]

    def cross(j0, j1):
        B0, B1 = B[j0], B[j1]
        f = (Bc - B0) / jnp.where(B1 == B0, 1.0, B1 - B0)
        return z[j0] + f * (z[j1] - z[j0])

    zR = cross(jR - 1, jR)
    zL = cross(jL + 1, jL)
    # the class exists in the main well only if both mirror points lie within one
    # field period of the minimum; wells spanning several periods (barely trapped,
    # transitioning orbits) are outside the single-well model and carry no measure
    valid = hasR & hasL & (B[im] < Bc) & (zR - z[im] <= T) & (z[im] - zL <= T)
    return zL, zR, valid, dz


def _safe(d):
    return jnp.where(jnp.abs(d) > 1e-6, d, jnp.where(d >= 0, 1e-6, -1e-6))


def displacement_fn(nq=32, newton=8, nscan=192):
    """disp(tb, Bc, sa) -> (D (2,), W): one-bounce displacement of the flow and the
    loss-cone moment of the class at the section point sa = (s, α)."""
    xg, wg = leggauss(nq)
    thj = jnp.asarray(xg * np.pi / 2)
    wj = jnp.asarray(wg * np.pi / 2)

    def disp(tb, Bc, sa):
        s0, al0 = sa[0], sa[1]
        zL0, zR0, valid, dzs = well0_tips_guess(tb, Bc, s0, al0, nscan)
        zL0, zR0 = jax.lax.stop_gradient(zL0), jax.lax.stop_gradient(zR0)
        Bof = lambda zz: sample(tb, s0, al0, zz, keys=["B"])["B"]  # noqa: E731

        def tip(z0):
            def body(z, _):
                dB = (Bof(z + 1e-3) - Bof(z - 1e-3)) / 2e-3
                return (
                    z - jnp.clip((Bof(z) - Bc) / _safe(dB), -0.5 * dzs, 0.5 * dzs),
                    None,
                )

            z, _ = jax.lax.scan(body, z0, None, length=newton)
            z = jax.lax.stop_gradient(z)
            # value = converged root; derivative = implicit-function derivative
            return z - (Bof(z) - Bc) / _safe(jax.grad(Bof)(z))

        zL, zR = tip(zL0), tip(zR0)
        c, h = 0.5 * (zL + zR), 0.5 * (zR - zL)
        zLs, zRs, hs = (jax.lax.stop_gradient(v) for v in (zL, zR, h))
        ok = (
            valid
            & (hs > 1e-4)
            & (jnp.abs(Bof(zLs) - Bc) < 1e-3)
            & (jnp.abs(Bof(zRs) - Bc) < 1e-3)
        )
        # keep the unselected branch finite so the VJP of jnp.where stays NaN-free
        c = jnp.where(ok, c, 0.0)
        h = jnp.where(ok, h, 1.0)
        zj = c + h * jnp.sin(thj)
        fs, fa, fw = jax.vmap(
            lambda z, th: drift_integrands(tb, s0, al0, z, Bc, h, th)
        )(zj, thj)
        D = 2.0 * jnp.stack([wj @ fs, wj @ fa])
        W = wj @ fw
        return jnp.where(ok, D, 0.0), jnp.where(ok, W, 0.0)

    return disp


def displacement_grid(tb, Bc, P, disp, chunk=4000):
    """D and W at the section points P (n, 2); chunked and rematerialised for memory."""
    n = len(P)
    pad = (-n) % chunk
    Pp = jnp.vstack([P, jnp.tile(P[:1], (pad, 1))]).reshape(-1, chunk, 2)
    f = jax.checkpoint(jax.vmap(lambda sa: disp(tb, Bc, sa)))
    D, W = jax.lax.map(f, Pp)
    return D.reshape(-1, 2)[:n], W.reshape(-1)[:n]


def flow_map(D_grid, s_ax, a_ax, P, nsub=8, k=1, s_loss=None):
    """Time-k map of dx/dτ = D(x) (k bounces) with a bicubic interpolant of D (periodic
    in α), integrated by RK4 with ``nsub`` substeps per bounce.  With ``s_loss`` given,
    paths that touch the wall within the k bounces are reported at their maximum s."""

    def Dfun(x):
        s = jnp.clip(x[:, 0], s_ax[0], s_ax[-1])
        a = x[:, 1] % (2 * jnp.pi)
        ds = interp2d(
            s, a, s_ax, a_ax, D_grid[..., 0], method="cubic", period=(None, 2 * np.pi)
        )
        da = interp2d(
            s, a, s_ax, a_ax, D_grid[..., 1], method="cubic", period=(None, 2 * np.pi)
        )
        return jnp.stack([ds, da], -1)

    h = 1.0 / nsub
    ds = s_ax[1] - s_ax[0]
    s_hi = (
        s_ax[-1] + 3 * ds if s_loss is None else jnp.maximum(s_loss, s_ax[-1]) + 3 * ds
    )

    def step(carry, _):
        x, smax = carry
        k1 = Dfun(x)
        k2 = Dfun(x + 0.5 * h * k1)
        k3 = Dfun(x + 0.5 * h * k2)
        k4 = Dfun(x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        # keep the state bounded: s just beyond the wall is lost anyway, and α mod 2π
        # (both keep the tangent map bounded for large drifts)
        x = jnp.stack([jnp.clip(x[:, 0], s_ax[0], s_hi), x[:, 1] % (2 * jnp.pi)], -1)
        return (x, jnp.maximum(smax, x[:, 0])), None

    (x, smax), _ = jax.lax.scan(step, (P, P[:, 0]), None, length=int(nsub * k))
    if s_loss is None:
        return x
    # a path that reached the wall during the k bounces is absorbed: near the wall the
    # row coordinate is the running maximum of s rather than the end point (smooth blend
    # over two cells so the operator stays continuous in the flow)
    t = jnp.clip((smax - (s_loss - 3 * ds)) / (2 * ds), 0.0, 1.0)
    sig = t * t * (3 - 2 * t)
    s_used = x[:, 0] + (smax - x[:, 0]) * sig
    return jnp.stack([s_used, x[:, 1]], -1)


# --------------------------------------------------------------------------
# transfer operator (host side) with a custom VJP
# --------------------------------------------------------------------------


def _bspline2_w(tau):
    return np.stack([0.5 * (0.5 - tau) ** 2, 0.75 - tau**2, 0.5 * (0.5 + tau) ** 2], 0)


def _row_coord(sp_, s_ax, s_loss):
    ns = len(s_ax)
    s_ext = np.append(s_ax, s_loss)
    fs = np.interp(np.clip(sp_, s_ax[0], s_loss), s_ext, np.arange(ns + 1))
    i0 = np.minimum(np.floor(np.clip(fs, 0, ns)).astype(int), ns - 1)
    return fs, 1.0 / (s_ext[i0 + 1] - s_ext[i0])


def _build_K(MP, s_ax, a_ax, s_loss):
    """Sparse transition matrix (quadratic B-spline weights) and wall weights."""
    ns, na = MP.shape[:2]
    N = ns * na
    da = a_ax[1] - a_ax[0]
    sp_ = MP[..., 0].ravel()
    ap_ = MP[..., 1].ravel()
    bad = ~np.isfinite(sp_) | ~np.isfinite(ap_)
    idx = np.arange(N)
    sp_ = np.where(bad, np.repeat(s_ax, na), sp_)
    ap_ = np.where(bad, np.tile(a_ax, ns), ap_)
    fs, _ = _row_coord(sp_, s_ax, s_loss)
    fa = (ap_ % (2 * np.pi)) / da
    ms = np.rint(fs).astype(int)
    ts = fs - ms
    ma = np.rint(fa).astype(int)
    ta = fa - ma
    ma = ma % na
    Ws, Wa = _bspline2_w(ts), _bspline2_w(ta)
    w_wall = np.zeros(N)
    rows, cols, vals = [], [], []
    for a_ in range(3):
        ii = np.clip(ms + a_ - 1, 0, ns + 1)
        for b_ in range(3):
            jj = (ma + b_ - 1) % na
            wt = Ws[a_] * Wa[b_]
            onwall = ii >= ns
            np.add.at(w_wall, idx[onwall], wt[onwall])
            rows.append(idx[~onwall])
            cols.append((ii * na + jj)[~onwall])
            vals.append(wt[~onwall])
    K = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N, N),
    )
    K.sum_duplicates()
    return K, w_wall


def _grad_U(lam, q, MP, s_ax, a_ax, u, s_loss):
    """d(λ·q·U(M))/dM(P): λ q ∇U at the landing points, U the C^1 interpolant of u
    with u = 1 on the wall ghost rows; zero beyond the wall and at the inner clamp."""
    ns, na = u.shape
    da = a_ax[1] - a_ax[0]
    sp_ = MP[..., 0]
    ap_ = MP[..., 1] % (2 * np.pi)
    fs, dfs = _row_coord(sp_, s_ax, s_loss)
    fa = ap_ / da
    U = np.vstack([u, np.ones((2, na))])
    ms = np.rint(fs).astype(int)
    ts = fs - ms
    ma = np.rint(fa).astype(int)
    ta = fa - ma
    ma = ma % na
    Ws, Wa = _bspline2_w(ts), _bspline2_w(ta)
    dWs = np.stack([-(0.5 - ts), -2 * ts, (0.5 + ts)], 0)
    dWa = np.stack([-(0.5 - ta), -2 * ta, (0.5 + ta)], 0)
    dUds = np.zeros_like(sp_)
    dUda = np.zeros_like(sp_)
    for a_ in range(3):
        ii = np.clip(ms + a_ - 1, 0, ns + 1)
        for b_ in range(3):
            Uab = U[ii, (ma + b_ - 1) % na]
            dUds += dWs[a_] * Wa[b_] * Uab
            dUda += Ws[a_] * dWa[b_] * Uab
    g = (lam * q)[..., None] * np.stack([dUds * dfs, dUda / da], -1)
    g[sp_ >= s_loss] = 0.0
    g[..., 0][sp_ <= s_ax[0]] = 0.0
    return g


def _resolvent_host(MP, mu, s_ax, a_ax, s_loss, nu):
    """Host solve: numerator ∫ μ u of the exponential-clock hitting probability and the
    adjoint λ for the cotangent.  ``nu`` is the clock rate per operator step."""
    MP = np.asarray(MP, float)
    mu = np.asarray(mu, float)
    ns, na = mu.shape
    K, w_wall = _build_K(MP, s_ax, a_ax, s_loss)
    q = np.exp(-nu)
    lu = spla.splu((sp.identity(ns * na, format="csc") - q * K).tocsc())
    u = lu.solve(q * w_wall)
    lam = lu.solve(mu.ravel(), trans="T")
    num = float(mu.ravel() @ u)
    return np.array(num), u.reshape(ns, na), lam.reshape(ns, na), np.array(q)


def make_resolvent_op(s_ax, a_ax, s_loss, nu):
    """jax-callable numerator(MP, mu) = ∫ μ u with a custom VJP (adjoint solve)."""
    ns, na = len(s_ax), len(a_ax)
    shapes = (
        jax.ShapeDtypeStruct((), jnp.float64),
        jax.ShapeDtypeStruct((ns, na), jnp.float64),
        jax.ShapeDtypeStruct((ns, na), jnp.float64),
        jax.ShapeDtypeStruct((), jnp.float64),
    )
    host = partial(_resolvent_host, s_ax=s_ax, a_ax=a_ax, s_loss=s_loss, nu=nu)

    @jax.custom_vjp
    def numerator(MP, mu):
        return jax.pure_callback(host, shapes, MP, mu)[0]

    def fwd(MP, mu):
        num, u, lam, q = jax.pure_callback(host, shapes, MP, mu)
        return num, (MP, u, lam, q)

    def bwd(res, ct):
        MP, u, lam, q = res
        g = jax.pure_callback(
            lambda MP_, lam_, q_, u_: _grad_U(
                np.asarray(lam_),
                float(q_),
                np.asarray(MP_),
                s_ax,
                a_ax,
                np.asarray(u_),
                s_loss,
            ),
            jax.ShapeDtypeStruct((ns, na, 2), jnp.float64),
            MP,
            lam,
            q,
            u,
        )
        return ct * g, ct * u

    numerator.defvjp(fwd, bwd)
    return numerator


# --------------------------------------------------------------------------
# the objective
# --------------------------------------------------------------------------


def birth_profile(kind, s, T0_keV=11.5):
    """Radial birth-rate profile S(s): 'uniform' or 'reactivity' (n ∝ 1−s⁵, T ∝ 1−s)."""
    if kind == "uniform":
        return jnp.ones_like(s)
    T = jnp.maximum(T0_keV * (1 - s), 1e-10)
    return (1 - s**5) ** 2 * T ** (-2.0 / 3.0) * jnp.exp(-19.94 * T ** (-1.0 / 3.0))


class BounceFlowLoss(_Objective):
    """Prompt collisionless loss of trapped alphas from the bounce-averaged drift flow.

    For each pitch class B_c the one-bounce displacement of the bounce-averaged flow is
    computed on a grid of the mirror section, its time-one map is discretised as a
    transition operator and the exponential-clock hitting probability of the wall is
    integrated against the birth measure of the class.  Returns the contribution of
    each class to the prompt loss fraction; ``loss_function="sum"`` gives the total.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to optimize.
    bcrit : ndarray
        Mirror fields B_c of the pitch classes [T].  Keep the classes away from the
        trapped-passing boundary (B_c within ~10% of the maximum |B|): there the
        bounce time and the loss-cone moment diverge and the single-well model does
        not apply.
    bcrit_weights : ndarray, optional
        Quadrature weights in B_c (default: trapezoid on ``bcrit``).
    s_loss : float
        Loss surface in normalized toroidal flux (wall of the operator).
    nu : float
        Exponential-clock rate per bounce; 1/nu is the mean horizon in bounces.
    k : int
        Bounces composed per operator step (the flow is integrated over k bounces
        before it is discretised).  Larger k reduces the grid coarse-graining of the
        operator at no cost in map evaluations; the clock rate per step is k nu.
    rho : ndarray, optional
        Flux surfaces of the geometry tables (default 48 surfaces, s ∈ [0.02, 0.98]).
    M_tab, N_tab : int
        Poloidal / toroidal resolution of the table grid (2M+1 by 2N+1 points).
    ns, na : int
        Section grid in s and α.
    num_quad, nsub, newton, nscan : int
        Bounce quadrature nodes, RK4 substeps per bounce of the flow map, Newton steps
        for the mirror points, ζ samples per period of the well scan.
    birth : {"reactivity", "uniform"}
        Radial birth profile.
    Ekin, mass, charge : float
        Particle kinetic energy [J], mass [kg] and charge [C]; default a 3.52 MeV alpha.
    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _coordinates = ""
    _units = "~"
    _print_value_fmt = "Bounce-flow loss fraction: "
    _static_attrs = _Objective._static_attrs + [
        "_hyper",
        "_per_class",
        "_disp",
        "_op",
        "_keys",
        "_v2",
        "_rho",
    ]

    def __init__(
        self,
        eq,
        *,
        bcrit,
        bcrit_weights=None,
        s_loss=0.98,
        nu=1.0 / 350.0,
        k=10,
        rho=None,
        M_tab=64,
        N_tab=32,
        ns=200,
        na=300,
        num_quad=32,
        nsub=4,
        newton=8,
        nscan=192,
        birth="reactivity",
        Ekin=_E_ALPHA,
        mass=_ALPHA_M,
        charge=_ALPHA_Q,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        deriv_mode="rev",
        name="bounce-flow loss",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        errorif(
            deriv_mode not in ("rev", "auto"),
            msg="BounceFlowLoss supports deriv_mode='rev' only.",
        )
        bcrit = np.atleast_1d(np.asarray(bcrit, float))
        if bcrit_weights is None:
            if bcrit.size == 1:
                bcrit_weights = np.ones(1)
            else:
                d = np.diff(bcrit)
                bcrit_weights = np.concatenate(
                    [[d[0] / 2], (d[:-1] + d[1:]) / 2, [d[-1] / 2]]
                )
        self._hyper = {
            "bcrit": tuple(float(b) for b in bcrit),
            "bcrit_weights": tuple(float(w) for w in bcrit_weights),
            "s_loss": float(s_loss),
            "nu": float(nu),
            "k": int(k),
            "M_tab": int(M_tab),
            "N_tab": int(N_tab),
            "ns": int(ns),
            "na": int(na),
            "num_quad": int(num_quad),
            "nsub": int(nsub),
            "newton": int(newton),
            "nscan": int(nscan),
            "birth": birth,
            "Ekin": float(Ekin),
            "mass": float(mass),
            "charge": float(charge),
        }
        self._rho = rho
        self._per_class = False
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode="rev",
            name=name,
            jac_chunk_size=jac_chunk_size,
        )

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays: table grid, transforms, section grid, quadratures."""
        eq = self.things[0]
        hp = self._hyper
        rho = np.asarray(setdefault(self._rho, np.sqrt(np.linspace(0.02, 0.98, 48))))
        grid = LinearGrid(rho=rho, M=hp["M_tab"], N=hp["N_tab"], NFP=eq.NFP, sym=False)
        errorif(not grid.can_fft2, msg="table grid must support FFT transforms")
        self._keys = [
            "|B|",
            "B^zeta",
            "b",
            "grad(|B|)",
            "e^rho",
            "e^theta",
            "e^zeta",
            "lambda",
            "lambda_r",
            "lambda_t",
            "lambda_z",
            "kappa",
            "iota",
            "iota_r",
            "sqrt(g)",
        ]
        # denominator ∫ S dV on a radial grid
        rr = np.linspace(rho[0], rho[-1], 96)
        gV = LinearGrid(rho=rr, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
        s_ax = np.linspace(rho[0] ** 2 + 0.005, hp["s_loss"] - 0.005, hp["ns"])
        a_ax = np.linspace(0, 2 * np.pi, hp["na"], endpoint=False)
        S, A = np.meshgrid(s_ax, a_ax, indexing="ij")
        self._constants = {
            "quad_weights": 1.0,
            "rho": jnp.asarray(rho),
            "grid": grid,
            "transforms": get_transforms(self._keys, eq, grid),
            "profiles": get_profiles(self._keys, eq, grid),
            "gridV": gV,
            "transformsV": get_transforms(["V_r(r)"], eq, gV),
            "profilesV": get_profiles(["V_r(r)"], eq, gV),
            "rr": jnp.asarray(rr),
            "s_ax": jnp.asarray(s_ax),
            "a_ax": jnp.asarray(a_ax),
            "P": jnp.asarray(np.column_stack([S.ravel(), A.ravel()])),
            "S_sec": birth_profile(hp["birth"], jnp.asarray(s_ax)),
            "cell": float((s_ax[1] - s_ax[0]) * (a_ax[1] - a_ax[0])),
            "bcrit": jnp.asarray(hp["bcrit"]),
            "bcrit_weights": jnp.asarray(hp["bcrit_weights"]),
        }
        self._v2 = 2 * hp["Ekin"] / hp["mass"]
        self._disp = displacement_fn(
            nq=hp["num_quad"], newton=hp["newton"], nscan=hp["nscan"]
        )
        self._op = make_resolvent_op(s_ax, a_ax, hp["s_loss"], hp["nu"] * hp["k"])
        self._dim_f = len(hp["bcrit"])
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Per-class contributions f_c to the prompt loss fraction (sum = total).

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, e.g.
            ``Equilibrium.params_dict``.
        constants : dict
            Dictionary of constant data, e.g. transforms, profiles etc.
            Defaults to ``self.constants``. (Deprecated)

        Returns
        -------
        f : ndarray
            Contribution of each pitch class to the prompt loss fraction.

        """
        constants = self._get_deprecated_constants(constants)
        eq = self.things[0]
        hp = self._hyper
        data = compute_fun(
            eq, self._keys, params, constants["transforms"], constants["profiles"]
        )
        tb = tables_from_data(
            data,
            constants["grid"],
            constants["rho"],
            eq.NFP,
            self._v2,
            2 * hp["M_tab"] + 1,
        )
        # global scale of the drift coefficients: m/q enters A, C (alpha by default)
        scale = (hp["mass"] / _ALPHA_M) * (_ALPHA_Q / hp["charge"])
        for k in ["A_s", "A_t", "A_z", "C_s", "C_t", "C_z"]:
            tb[k] = tb[k] * scale
        dV = compute_fun(
            eq, ["V_r(r)"], params, constants["transformsV"], constants["profilesV"]
        )
        Vr = constants["gridV"].compress(dV["V_r(r)"])
        rr = constants["rr"]
        denom = jnp.trapezoid(birth_profile(hp["birth"], rr**2) * Vr, rr)
        s_ax, a_ax, P = constants["s_ax"], constants["a_ax"], constants["P"]
        ns, na = hp["ns"], hp["na"]

        def one_class(Bc):
            D, W = displacement_grid(tb, Bc, P, self._disp)
            MP = flow_map(
                D.reshape(ns, na, 2),
                s_ax,
                a_ax,
                P,
                nsub=hp["nsub"],
                k=hp["k"],
                s_loss=hp["s_loss"],
            ).reshape(ns, na, 2)
            mu = W.reshape(ns, na) * constants["S_sec"][:, None] * constants["cell"]
            return self._op(MP, mu), jnp.sum(mu)

        num, tot = jax.lax.map(one_class, constants["bcrit"])
        if self._per_class:
            return num / tot
        return eq.NFP * constants["bcrit_weights"] * num / denom

    def class_loss(self, params):
        """Loss fraction of each class, ∫ μ_c u_c / ∫ μ_c (diagnostic)."""
        self._per_class = True
        try:
            return self.compute(params)
        finally:
            self._per_class = False
