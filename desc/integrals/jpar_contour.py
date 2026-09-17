"""Main-well J∥ contour diagnostic for energetic-particle losses.

Builds the single-valued second adiabatic invariant on the main magnetic well
(the well containing the global |B| minimum along each field-line segment) and
measures how close J = const contours come to connecting a seed to the plasma
boundary (minimax J-margin).

This module is a **diagnostic only** — not a differentiable objective.

Well labelling
--------------
J∥ on the drift plane is only single-valued if every (ρ, α) refers to the *same*
well family. Selecting the deepest well per (ρ, α) over several field periods
(``well_select="deepest"``) stitches the map from wells whose field-line labels
differ by k·ι(ρ)·2π/NFP; the seams act as spurious J barriers in the contour
search. The default ``well_select="period"`` anchors the main well to the well
containing the |B| minimum of one fixed field period (well-0 family), which is
a consistent labelling: the resulting map is the well-0 map composed with a
smooth α-shift, so its contour connectivity is that of the drift orbits.
Margins measured on the stitched map are dominated by the seams (order
ΔJ_char), so a ``dJ_crit`` calibrated against it does not transfer: on a
consistently labelled map the margins of most seeds are a small fraction of
ΔJ_char and ``dJ_crit`` should be chosen at the grid tolerance,
of order 0.03–0.1 ΔJ_char.


The v1 flood-fill / absolute-tolerance path is retained for regression tests and
legacy diagnostics; new work should use ``minimax_margin`` / ``diagnose_margin``.

References
----------
Paul, Bhattacharjee et al. (reactor-scale QS study) on drift-convective orbits;
Nemov et al., Phys. Plasmas 15, 052501 (2008) for the topological Γ_c criterion.
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import NamedTuple

import numpy as np

from desc.backend import jnp
from desc.grid import LinearGrid
from desc.integrals.bounce_integral import Bounce2D
from desc.integrals.quad_utils import chebgauss2


def jpar_integrand(data, B, pitch):
    """Energy-normalized J∥ density: ∫ √|1 − λ B| dℓ.

    Bounce2D multiplies by |e_ζ| and quadrature weights outside this callable,
    converting the ζ integral into an arc-length integral.
    """
    return jnp.sqrt(jnp.abs(1.0 - pitch * B))


class JMainResult(NamedTuple):
    """Main-well J∥ on a (ρ, α, λ) grid."""

    J: np.ndarray
    """Shape (n_rho, n_alpha, n_pitch). Main-well J∥; NaN where ``mask`` is False."""

    mask: np.ndarray
    """Shape (n_rho, n_alpha, n_pitch). True where a main well exists."""

    rho: np.ndarray
    """Shape (n_rho,). Radial labels used for the Bounce2D grid."""

    alpha: np.ndarray
    """Shape (n_alpha,). Field-line labels α ∈ [0, 2π)."""

    pitch_inv: np.ndarray
    """Shape (n_rho, n_pitch) or (n_pitch,). Sampled 1/λ values."""

    pitch_inv_weight: np.ndarray | None
    """Pitch quadrature weights in 1/λ (same shape as ``pitch_inv``), or None."""

    main_well_index: np.ndarray
    """Shape (n_rho, n_alpha, n_pitch). Well index selected as the main well."""

    B_bot: np.ndarray
    """Shape (n_rho, n_alpha, n_pitch). |B| at the main-well bottom."""

    delta_zeta: np.ndarray
    """Shape (n_rho, n_alpha, n_pitch). Main-well bounce span ``z2 − z1``."""

    zeta1: np.ndarray
    """Shape (n_rho, n_alpha, n_pitch). Main-well lower bounce ζ (NaN if no well)."""

    zeta2: np.ndarray
    """Shape (n_rho, n_alpha, n_pitch). Main-well upper bounce ζ (NaN if no well)."""

    num_transit: int
    num_well: int

    well_select: str = "period"
    """Well labelling used: ``"period"`` (well containing the |B| minimum of the
    reference field period) or ``"deepest"`` (deepest well per (ρ, α))."""

    ref_period: int | None = None
    """Index of the reference field period (``well_select="period"``)."""

    zeta_min: np.ndarray | None = None
    """Shape (n_rho, n_alpha). ζ of the |B| minimum in the reference period."""


def fieldline_B_argmin(bounce, period, n_sub=8):
    """ζ of the |B| minimum of one field period along each field line.

    Parameters
    ----------
    bounce : Bounce2D
        Bounce object built with ``spline=True``.
    period : int
        Field-period index in ``[0, num_transit * NFP)``.
    n_sub : int
        Sub-samples per spline interval used to locate the minimum.

    Returns
    -------
    zeta_min, B_min : ndarray
        Shape (n_rho, n_alpha). Location and value of the minimum.
    """
    c = np.asarray(bounce._c["B(z)"])  # (n_rho, n_alpha, n_knot - 1, 4), power basis
    knots = np.asarray(bounce._c["knots"])
    if c.ndim != 4:
        raise ValueError("fieldline_B_argmin requires Bounce2D(spline=True)")
    T = 2 * np.pi / bounce._NFP
    lo, hi = period * T, (period + 1) * T
    k = np.flatnonzero((knots[:-1] >= lo - 1e-12) & (knots[:-1] < hi - 1e-12))
    if k.size == 0:
        raise ValueError(f"period {period} lies outside the field-line domain")
    x = np.linspace(0.0, 1.0, n_sub, endpoint=False)
    dz = (knots[k + 1] - knots[k])[:, None] * x[None, :]  # (n_int, n_sub)
    ck = c[:, :, k, :]  # (n_rho, n_alpha, n_int, 4)
    B = ((ck[..., 0:1] * dz + ck[..., 1:2]) * dz + ck[..., 2:3]) * dz + ck[..., 3:4]
    B = B.reshape(B.shape[0], B.shape[1], -1)
    z = (knots[k][:, None] + dz).reshape(-1)
    i = np.argmin(B, axis=-1)
    return z[i], np.take_along_axis(B, i[..., None], axis=-1)[..., 0]


def select_period_well(J_all, B_bot_all, z1, z2, zeta_min):
    """Pick the well that contains ``zeta_min`` (the reference-period minimum).

    Parameters
    ----------
    J_all, B_bot_all, z1, z2 : array_like
        Shape (..., n_well), as in ``select_main_well``.
    zeta_min : array_like
        Broadcastable to the leading axes of ``J_all``: ζ of the |B| minimum of
        the reference field period on each field line.

    Returns
    -------
    Same as ``select_main_well``. The mask is False where no well contains the
    reference minimum (the class does not exist in the main well of that line).
    """
    J_all = np.asarray(J_all)
    B_bot_all = np.asarray(B_bot_all)
    z1 = np.asarray(z1)
    z2 = np.asarray(z2)
    zm = np.asarray(zeta_min, dtype=float)[..., None]
    valid = z1 < z2
    contains = valid & (z1 <= zm) & (zm <= z2)
    has = np.any(contains, axis=-1)
    main_idx = np.argmax(contains, axis=-1)  # first containing well; 0 if none
    gather = np.expand_dims(main_idx, axis=-1)
    J_main = np.take_along_axis(J_all, gather, axis=-1)[..., 0]
    B_bot = np.take_along_axis(B_bot_all, gather, axis=-1)[..., 0]
    z1_m = np.take_along_axis(z1, gather, axis=-1)[..., 0]
    z2_m = np.take_along_axis(z2, gather, axis=-1)[..., 0]
    mask = has & np.isfinite(J_main) & (J_main > 0)
    J_out = np.where(mask, J_main, np.nan)
    B_out = np.where(mask, B_bot, np.nan)
    delta_zeta = np.where(mask, z2_m - z1_m, np.nan)
    zeta1 = np.where(mask, z1_m, np.nan)
    zeta2 = np.where(mask, z2_m, np.nan)
    return J_out, mask, main_idx, B_out, delta_zeta, zeta1, zeta2


def select_main_well(J_all, B_bot_all, z1, z2):
    """Pick the well whose interior contains the global |B| minimum.

    Parameters
    ----------
    J_all, B_bot_all, z1, z2 : array_like
        Shape (..., n_well). Bounce integrals, |B| at each well bottom, and
        bounce-point ζ coordinates from ``Bounce2D.points``.

    Returns
    -------
    J_main, mask, main_idx, B_bot, delta_zeta, zeta1, zeta2 : ndarray
        Main-well J, existence mask, well index, |B| at that well bottom,
        bounce span ``z2 − z1``, and the bounce ζ coordinates themselves.
        Shapes match ``J_all`` with the well axis removed.
    """
    J_all = np.asarray(J_all)
    B_bot_all = np.asarray(B_bot_all)
    z1 = np.asarray(z1)
    z2 = np.asarray(z2)

    valid = z1 < z2
    B_masked = np.where(valid, B_bot_all, np.inf)
    # All-invalid rows → argmin picks 0; mask will be False.
    main_idx = np.argmin(B_masked, axis=-1)
    gather = np.expand_dims(main_idx, axis=-1)
    J_main = np.take_along_axis(J_all, gather, axis=-1)[..., 0]
    B_bot = np.take_along_axis(B_bot_all, gather, axis=-1)[..., 0]
    mask = np.take_along_axis(valid, gather, axis=-1)[..., 0]
    z1_m = np.take_along_axis(z1, gather, axis=-1)[..., 0]
    z2_m = np.take_along_axis(z2, gather, axis=-1)[..., 0]
    # Drop spurious zeros from padded wells (z1=z2=0 → integrate returns 0).
    mask = mask & np.isfinite(J_main) & (J_main > 0)
    J_out = np.where(mask, J_main, np.nan)
    B_out = np.where(mask, B_bot, np.nan)
    delta_zeta = np.where(mask, z2_m - z1_m, np.nan)
    zeta1 = np.where(mask, z1_m, np.nan)
    zeta2 = np.where(mask, z2_m, np.nan)
    return J_out, mask, main_idx, B_out, delta_zeta, zeta1, zeta2


def build_j_main(
    eq,
    rho=None,
    alpha=None,
    *,
    pitch_inv=None,
    num_pitch=45,
    num_transit=2,
    num_well=None,
    num_quad=32,
    X=32,
    Y=32,
    Y_B=None,
    pitch_batch_size=None,
    nufft_eps=1e-6,
    spline=True,
    check_points=False,
    well_select="period",
    ref_period=None,
):
    """Build main-well J∥(ρ, α, λ) with Bounce2D.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to analyze.
    rho, alpha : array_like, optional
        Radial and field-line grids. Defaults: 24 ρ ∈ (0, 1], 64 α ∈ [0, 2π).
    pitch_inv : array_like, optional
        Custom 1/λ samples. Shape (n_pitch,) broadcasts over ρ, or
        (n_rho, n_pitch). If None, uses ``Bounce2D.get_pitch_inv_quad`` on
        each surface's [min_tz |B|, max_tz |B|].
    num_pitch : int
        Used only when ``pitch_inv`` is None.
    num_transit : int
        Toroidal transits followed per field line. Prefer a small value (~1–2)
        when α is an explicit grid coordinate; large values create many wells
        that are hard to label coherently across the grid.
    num_well : int, optional
        Max wells retained per (ρ, α, λ). Default from ``num_well_rule``.
    num_quad, X, Y, Y_B, nufft_eps, spline
        Bounce2D quadrature / Fourier–Chebyshev resolution knobs.
    pitch_batch_size : int, optional
        Integrate this many pitches at a time (memory control).
    check_points : bool
        If True, run ``Bounce2D.check_points`` (no plots) on the first pitch
        batch as a sanity check.
    well_select : {"period", "deepest"}
        ``"period"`` (default): the main well is the well containing the |B|
        minimum of the field period ``ref_period`` on each line (well-0 family;
        consistent labelling across the drift plane, see module docstring).
        Requires ``spline=True``. ``"deepest"``: the deepest well per (ρ, α)
        over all transits (legacy; stitches wells and creates spurious barriers).
    ref_period : int, optional
        Field-period index for ``well_select="period"``. Default: the middle
        period of the domain, ``(num_transit * NFP) // 2``.

    Returns
    -------
    JMainResult
    """
    if well_select not in ("period", "deepest"):
        raise ValueError(
            f"well_select must be 'period' or 'deepest', got {well_select!r}"
        )
    if well_select == "period" and not spline:
        raise ValueError("well_select='period' requires spline=True")
    rho = np.asarray(
        rho if rho is not None else np.linspace(0.1, 1.0, 24), dtype=float
    )
    alpha = np.asarray(
        alpha if alpha is not None else np.linspace(0.0, 2 * np.pi, 64, endpoint=False),
        dtype=float,
    )

    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
    data = eq.compute(
        Bounce2D.required_names + ["min_tz |B|", "max_tz |B|"],
        grid=grid,
    )
    angle = Bounce2D.angle(eq, X=X, Y=Y, rho=rho)
    quad = chebgauss2(num_quad)

    bounce_kwargs = {
        "Y_B": Y_B,
        "alpha": jnp.asarray(alpha),
        "num_transit": num_transit,
        "quad": quad,
        "nufft_eps": nufft_eps,
        "spline": spline,
    }
    # Drop None Y_B so Bounce2D applies Y_B_rule.
    if Y_B is None:
        bounce_kwargs.pop("Y_B")

    bounce = Bounce2D(grid, data, angle, **bounce_kwargs)

    zeta_min = None
    if well_select == "period":
        if ref_period is None:
            ref_period = (num_transit * eq.NFP) // 2
        zeta_min, _ = fieldline_B_argmin(bounce, ref_period)

    min_B = np.asarray(grid.compress(data["min_tz |B|"]))
    max_B = np.asarray(grid.compress(data["max_tz |B|"]))

    pitch_weight = None
    if pitch_inv is None:
        pitch_inv, pitch_weight = Bounce2D.get_pitch_inv_quad(
            min_B, max_B, num_pitch, simp=True
        )
        pitch_inv = np.asarray(pitch_inv)
        pitch_weight = np.asarray(pitch_weight)
    else:
        pitch_inv = np.asarray(pitch_inv, dtype=float)
        if pitch_inv.ndim == 1:
            pitch_inv = np.broadcast_to(pitch_inv, (rho.size, pitch_inv.shape[0])).copy()
        elif pitch_inv.shape[0] != rho.size:
            raise ValueError(
                f"pitch_inv has shape {pitch_inv.shape}; expected "
                f"(n_pitch,) or (n_rho={rho.size}, n_pitch)."
            )

    n_pitch = pitch_inv.shape[-1]
    if pitch_batch_size is None:
        pitch_batch_size = n_pitch
    pitch_batch_size = max(1, int(pitch_batch_size))

    B_reshape = Bounce2D.reshape(grid, data["|B|"])

    J_chunks = []
    mask_chunks = []
    idx_chunks = []
    Bbot_chunks = []
    dz_chunks = []
    z1_chunks = []
    z2_chunks = []
    resolved_num_well = num_well

    for i0 in range(0, n_pitch, pitch_batch_size):
        i1 = min(i0 + pitch_batch_size, n_pitch)
        p_batch = jnp.asarray(pitch_inv[:, i0:i1])
        points = bounce.points(p_batch, num_well=num_well)
        if check_points and i0 == 0:
            bounce.check_points(points, p_batch, plot=False)

        J_all = bounce.integrate(
            jpar_integrand,
            p_batch,
            points=points,
            num_well=num_well,
            nufft_eps=nufft_eps,
        )
        B_bot_all = bounce.interp_to_argmin(
            B_reshape, points, nufft_eps=nufft_eps
        )
        z1, z2 = points
        if resolved_num_well is None:
            resolved_num_well = int(np.asarray(z1).shape[-1])

        if well_select == "period":
            J_m, msk, midx, Bb, dz, z1m, z2m = select_period_well(
                J_all, B_bot_all, z1, z2, zeta_min[:, :, None]
            )
        else:
            J_m, msk, midx, Bb, dz, z1m, z2m = select_main_well(
                J_all, B_bot_all, z1, z2
            )
        J_chunks.append(J_m)
        mask_chunks.append(msk)
        idx_chunks.append(midx)
        Bbot_chunks.append(Bb)
        dz_chunks.append(dz)
        z1_chunks.append(z1m)
        z2_chunks.append(z2m)

    return JMainResult(
        J=np.concatenate(J_chunks, axis=-1),
        mask=np.concatenate(mask_chunks, axis=-1),
        rho=rho,
        alpha=alpha,
        pitch_inv=pitch_inv,
        pitch_inv_weight=pitch_weight,
        main_well_index=np.concatenate(idx_chunks, axis=-1),
        B_bot=np.concatenate(Bbot_chunks, axis=-1),
        delta_zeta=np.concatenate(dz_chunks, axis=-1),
        zeta1=np.concatenate(z1_chunks, axis=-1),
        zeta2=np.concatenate(z2_chunks, axis=-1),
        num_transit=num_transit,
        num_well=int(resolved_num_well),
        well_select=well_select,
        ref_period=ref_period,
        zeta_min=zeta_min,
    )


def hard_range(J_alpha_slice, mask):
    """Exact achievable range of J over α at one flux surface.

    Parameters
    ----------
    J_alpha_slice, mask : array_like
        1D arrays over α (optionally already sliced in pitch).

    Returns
    -------
    j_min, j_max : float
        Achievable range. If no valid points, returns (-inf, +inf) so the
        surface cannot pinch any contour (no main well present).
    """
    valid = np.asarray(J_alpha_slice)[np.asarray(mask, dtype=bool)]
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return -np.inf, np.inf
    return float(valid.min()), float(valid.max())


def persistence_check(J, mask, s_idx0, alpha_idx0, s_grid=None):
    """Necessary condition: is J0 achievable at every surface out to the wall?

    Checking every intermediate surface matters: a pinch at intermediate s*
    blocks any continuous path even when both endpoints look fine.

    Parameters
    ----------
    J, mask : array_like
        Shape (n_s, n_alpha). Single pitch slice.
    s_idx0, alpha_idx0 : int
        Seed indices.
    s_grid : array_like, optional
        Radial coordinates for reporting ``pinch_s`` (e.g. ρ or s=ρ²).

    Returns
    -------
    dict
        Keys: connected, pinch_s, pinch_idx, J0, range_at_pinch.
    """
    J = np.asarray(J)
    mask = np.asarray(mask, dtype=bool)
    if not mask[s_idx0, alpha_idx0]:
        return {
            "connected": False,
            "pinch_s": None if s_grid is None else np.nan,
            "pinch_idx": int(s_idx0),
            "J0": np.nan,
            "range_at_pinch": None,
            "reason": "seed has no main well",
        }

    J0 = float(J[s_idx0, alpha_idx0])
    for i_s in range(s_idx0, J.shape[0]):
        m_s, M_s = hard_range(J[i_s], mask[i_s])
        if not (m_s <= J0 <= M_s):
            pinch_s = None if s_grid is None else float(np.asarray(s_grid)[i_s])
            return {
                "connected": False,
                "pinch_s": pinch_s,
                "pinch_idx": int(i_s),
                "J0": J0,
                "range_at_pinch": (m_s, M_s),
                "reason": "range pinch",
            }
    return {
        "connected": True,
        "pinch_s": None,
        "pinch_idx": None,
        "J0": J0,
        "range_at_pinch": None,
        "reason": None,
    }


def reaches_wall_flood_fill(J, mask, s_idx0, alpha_idx0, tol, wall_idx=None):
    """Exact discrete connectivity of the seed's J = J0 level-set branch.

    Neighborhood: ±1 in s (non-periodic) and ±1 in α (periodic).

    Parameters
    ----------
    J, mask : array_like
        Shape (n_s, n_alpha).
    s_idx0, alpha_idx0 : int
        Seed.
    tol : float
        Level-set band half-width: points with |J − J0| ≤ tol (and mask) are
        admissible. Start near ``0.5 * |∇J| * grid_spacing`` and sweep.
    wall_idx : int, optional
        Radial index treated as the wall / limiter. Default: outermost surface.

    Returns
    -------
    reached_wall : bool
    visited : ndarray of bool, shape (n_s, n_alpha)
    """
    J = np.asarray(J)
    mask = np.asarray(mask, dtype=bool)
    n_s, n_a = J.shape
    if wall_idx is None:
        wall_idx = n_s - 1

    if not mask[s_idx0, alpha_idx0]:
        visited = np.zeros_like(mask, dtype=bool)
        return False, visited

    J0 = float(J[s_idx0, alpha_idx0])
    valid = mask & np.isfinite(J) & (np.abs(J - J0) <= tol)
    if not valid[s_idx0, alpha_idx0]:
        # Seed itself outside band (tol=0 edge case) — still allow the seed.
        valid = valid.copy()
        valid[s_idx0, alpha_idx0] = True

    visited = np.zeros_like(valid, dtype=bool)
    stack = deque([(int(s_idx0), int(alpha_idx0))])
    visited[s_idx0, alpha_idx0] = True
    reached_wall = False

    while stack:
        s_i, a_i = stack.pop()
        if s_i >= wall_idx:
            reached_wall = True
        for ds in (-1, 1):
            s_j = s_i + ds
            if 0 <= s_j < n_s and valid[s_j, a_i] and not visited[s_j, a_i]:
                visited[s_j, a_i] = True
                stack.append((s_j, a_i))
        for da in (-1, 1):
            a_j = (a_i + da) % n_a
            if valid[s_i, a_j] and not visited[s_i, a_j]:
                visited[s_i, a_j] = True
                stack.append((s_i, a_j))

    return reached_wall, visited


def default_tol(J, mask, s_idx0, alpha_idx0, fraction=0.5):
    """Heuristic level-set half-width from local finite differences.

    ``tol ≈ fraction * mean(|Δ_s J|, |Δ_α J|)`` at the seed, using only
    masked neighbors. Falls back to a small relative scale if the gradient
    vanishes.

    .. note::
        This scales as ``O(h)`` under grid refinement. Prefer ``minimax_margin``.
    """
    J = np.asarray(J)
    mask = np.asarray(mask, dtype=bool)
    if not mask[s_idx0, alpha_idx0]:
        return np.nan
    J0 = float(J[s_idx0, alpha_idx0])
    n_s, n_a = J.shape
    deltas = []
    for ds, da in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        s_j = s_idx0 + ds
        a_j = (alpha_idx0 + da) % n_a
        if 0 <= s_j < n_s and mask[s_j, a_j] and np.isfinite(J[s_j, a_j]):
            deltas.append(abs(float(J[s_j, a_j]) - J0))
    if not deltas:
        return max(1e-12, 1e-3 * abs(J0))
    return float(fraction * np.mean(deltas))


def _neighbor_offsets(eight=False):
    """4-neighbor (default) or 8-neighbor stencil offsets in (Δs, Δα)."""
    if eight:
        return (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        )
    return ((-1, 0), (1, 0), (0, -1), (0, 1))


def dilate_mask(J, mask, n_iter=1):
    """Expand the trapping mask by nearest-neighbor fill of J.

    Each iteration assigns unmasked cells that touch a masked neighbor the mean
    J of those neighbors and marks them masked. This is a cheap sub-cell-ish
    treatment of a ragged λ|B|=1 boundary; prefer a true contour of λ|B|=1 when
    available.
    """
    J = np.array(J, dtype=float, copy=True)
    mask = np.array(mask, dtype=bool, copy=True)
    n_s, n_a = J.shape
    offs = _neighbor_offsets(eight=False)
    for _ in range(int(n_iter)):
        add_s, add_a, add_J = [], [], []
        for s_i in range(n_s):
            for a_i in range(n_a):
                if mask[s_i, a_i]:
                    continue
                vals = []
                for ds, da in offs:
                    s_j = s_i + ds
                    a_j = (a_i + da) % n_a
                    if 0 <= s_j < n_s and mask[s_j, a_j] and np.isfinite(J[s_j, a_j]):
                        vals.append(J[s_j, a_j])
                if vals:
                    add_s.append(s_i)
                    add_a.append(a_i)
                    add_J.append(float(np.mean(vals)))
        if not add_s:
            break
        for s_i, a_i, jv in zip(add_s, add_a, add_J):
            J[s_i, a_i] = jv
            mask[s_i, a_i] = True
    return J, mask


def minimax_margin(
    J,
    mask,
    s_idx0,
    alpha_idx0,
    wall_idx=None,
    *,
    eight_neighbor=False,
):
    """Bottleneck-path margin: min over paths (seed→wall) of max |J − J₀|.

    Returns ``+inf`` if no admissible (masked) path exists — the seed's main-well
    domain does not reach the wall (detrap / mask barrier). That is a different
    channel from a large finite margin.

    Parameters
    ----------
    J, mask : array_like
        Shape (n_s, n_alpha). α periodic, s not.
    s_idx0, alpha_idx0 : int
        Seed indices.
    wall_idx : int, optional
        Radial index treated as the wall. Default: outermost surface.
    eight_neighbor : bool
        If True, use an 8-neighbor stencil (convergence check vs 4-neighbor).

    Returns
    -------
    float
        ΔJ_margin. 0 iff a discrete J=J₀ level set connects seed to wall.
    """
    J = np.asarray(J)
    mask = np.asarray(mask, dtype=bool)
    n_s, n_a = J.shape
    if wall_idx is None:
        wall_idx = n_s - 1
    if not (0 <= s_idx0 < n_s and 0 <= alpha_idx0 < n_a):
        return np.inf
    if not mask[s_idx0, alpha_idx0] or not np.isfinite(J[s_idx0, alpha_idx0]):
        return np.inf

    J0 = float(J[s_idx0, alpha_idx0])
    cost = np.where(mask & np.isfinite(J), np.abs(J - J0), np.inf)
    best = np.full((n_s, n_a), np.inf)
    best[s_idx0, alpha_idx0] = float(cost[s_idx0, alpha_idx0])
    pq = [(best[s_idx0, alpha_idx0], int(s_idx0), int(alpha_idx0))]
    done = np.zeros((n_s, n_a), dtype=bool)
    offs = _neighbor_offsets(eight=eight_neighbor)

    while pq:
        c, s_i, a_i = heapq.heappop(pq)
        if done[s_i, a_i]:
            continue
        done[s_i, a_i] = True
        if s_i >= wall_idx:
            return float(c)
        for ds, da in offs:
            s_j = s_i + ds
            a_j = (a_i + da) % n_a
            if not (0 <= s_j < n_s):
                continue
            cj = cost[s_j, a_j]
            if not np.isfinite(cj):
                continue
            cand = c if c >= cj else cj
            if cand < best[s_j, a_j]:
                best[s_j, a_j] = cand
                heapq.heappush(pq, (cand, s_j, a_j))
    return np.inf


def barely_trapped_crit(nfp, factor=2.0):
    """Critical main-well bounce span |Δζ| = ``factor * π / nfp``."""
    return float(factor) * np.pi / float(nfp)


def is_barely_trapped(delta_zeta, nfp, factor=2.0):
    """True where the main-well bounce span exceeds the barely-trapped cut.

    Particles / seeds with ``|Δζ| > 2 π / nfp`` (default) are treated as
    barely trapped. Non-finite Δζ (no main well) are not flagged here.
    """
    dz = np.asarray(delta_zeta, float)
    crit = barely_trapped_crit(nfp, factor=factor)
    return np.isfinite(dz) & (dz > crit)


def diagnose_margin(
    J,
    mask,
    seeds,
    lambdas,
    *,
    wall_idx=None,
    eight_neighbor=False,
    mask_dilate=0,
    delta_zeta=None,
    nfp=None,
    barely_factor=2.0,
):
    """Compute ΔJ_margin for each (seed, λ).

    Parameters
    ----------
    J, mask : array_like
        Shape (n_s, n_alpha, n_pitch).
    seeds : sequence of (s_idx, alpha_idx)
    lambdas : array_like
        Pitch values for reporting (length n_pitch).
    wall_idx : int, optional
    eight_neighbor : bool
    mask_dilate : int
        Pass to ``dilate_mask`` per pitch slice before path search.
    delta_zeta : array_like, optional
        Shape (n_s, n_alpha, n_pitch). Main-well bounce span. If given with
        ``nfp``, seeds with ``Δζ > barely_factor · π / nfp`` are flagged
        ``"barely trapped"`` and skipped (margin left at +inf).
    nfp : int, optional
        Number of field periods (required with ``delta_zeta``).
    barely_factor : float
        Multiplier in the barely-trapped cut. Default 2.0 (|Δζ| > 2π/nfp).

    Returns
    -------
    margins : ndarray, shape (n_seed, n_pitch)
        Finite margin, ``+inf`` if no path / no main well at seed.
    flags : ndarray of str, shape (n_seed, n_pitch)
        ``"ok"``, ``"no main well"``, ``"barely trapped"``, or
        ``"masked barrier"`` (inf after dilate).
    """
    J = np.asarray(J)
    mask = np.asarray(mask, dtype=bool)
    lambdas = np.asarray(lambdas, dtype=float)
    seeds = list(seeds)
    n_seed = len(seeds)
    n_pitch = J.shape[-1]
    margins = np.full((n_seed, n_pitch), np.inf)
    flags = np.empty((n_seed, n_pitch), dtype=object)

    use_bt = delta_zeta is not None
    if use_bt:
        if nfp is None:
            raise ValueError("nfp is required when delta_zeta is provided")
        delta_zeta = np.asarray(delta_zeta, float)
        if delta_zeta.shape != J.shape:
            raise ValueError(
                f"delta_zeta shape {delta_zeta.shape} != J shape {J.shape}"
            )
        bt_crit = barely_trapped_crit(nfp, factor=barely_factor)

    for i_lam in range(n_pitch):
        J_lam = J[..., i_lam]
        m_lam = mask[..., i_lam]
        if mask_dilate > 0:
            J_lam, m_lam = dilate_mask(J_lam, m_lam, n_iter=mask_dilate)
        for i_seed, (s_idx0, a_idx0) in enumerate(seeds):
            if not mask[s_idx0, a_idx0, i_lam]:
                # Original (pre-dilate) seed must have a main well.
                flags[i_seed, i_lam] = "no main well"
                continue
            if use_bt and delta_zeta[s_idx0, a_idx0, i_lam] > bt_crit:
                flags[i_seed, i_lam] = "barely trapped"
                continue
            m = minimax_margin(
                J_lam,
                m_lam,
                s_idx0,
                a_idx0,
                wall_idx=wall_idx,
                eight_neighbor=eight_neighbor,
            )
            margins[i_seed, i_lam] = m
            if np.isfinite(m):
                flags[i_seed, i_lam] = "ok"
            else:
                flags[i_seed, i_lam] = "masked barrier"
    return margins, flags


def characteristic_delta_j(J, mask, axis_alpha=1):
    """Per-pitch characteristic |ΔJ| from α-range on each surface, then median.

    Used to nondimensionalize margins: ``ΔJ_margin / ΔJ_char``.

    Returns
    -------
    dJ_char : ndarray, shape (n_pitch,)
    """
    J = np.asarray(J)
    mask = np.asarray(mask, dtype=bool)
    n_s, _, n_pitch = J.shape
    out = np.full(n_pitch, np.nan)
    for i_p in range(n_pitch):
        spans = []
        for i_s in range(n_s):
            m = mask[i_s, :, i_p]
            if not np.any(m):
                continue
            vals = J[i_s, m, i_p]
            vals = vals[np.isfinite(vals)]
            if vals.size >= 2:
                spans.append(float(vals.max() - vals.min()))
            elif vals.size == 1:
                spans.append(0.0)
        if spans:
            out[i_p] = float(np.median(spans))
    # Fallback: global finite-J scale
    bad = ~np.isfinite(out) | (out <= 0)
    if np.any(bad):
        finite = J[mask & np.isfinite(J)]
        fallback = float(np.nanmedian(np.abs(finite))) if finite.size else 1.0
        fallback = max(fallback, 1e-12)
        out = np.where(bad, fallback, out)
    return out


def p_loss_from_margin(margins, dJ_crit, *, soft=False):
    """Map ΔJ_margin → loss probability.

    Sharp: ``p = 1`` if ``margin < dJ_crit`` when ``dJ_crit > 0``; if
    ``dJ_crit == 0``, ``p = 1`` iff ``margin == 0`` (exact discrete level set).
    Soft: ``p = clip(1 - margin/dJ_crit, 0, 1)``.
    Infinite margins (mask barrier) → 0 under this main-well contour channel.
    """
    margins = np.asarray(margins, dtype=float)
    dJ_crit = np.asarray(dJ_crit, dtype=float)
    finite = np.isfinite(margins)
    if dJ_crit.ndim == 1 and margins.ndim == 2:
        scale = dJ_crit[None, :]
    else:
        scale = dJ_crit
    if soft:
        scale_safe = np.maximum(scale, 1e-30)
        return np.where(finite, np.clip(1.0 - margins / scale_safe, 0.0, 1.0), 0.0)
    # sharp
    zero_thresh = np.asarray(scale) <= 0
    hit = np.where(
        zero_thresh,
        finite & (margins <= 0.0),
        finite & (margins < scale),
    )
    return hit.astype(float)


def weighted_fraction(values, *, w_seed=None, w_lambda=None, eligible=None):
    """Weighted mean of ``values`` over (seed, λ), optional eligibility mask."""
    values = np.asarray(values, dtype=float)
    n_seed, n_pitch = values.shape
    if w_seed is None:
        w_seed = np.ones(n_seed)
    else:
        w_seed = np.asarray(w_seed, dtype=float)
    if w_lambda is None:
        w_lambda = np.ones(n_pitch)
    w_lambda = np.asarray(w_lambda, dtype=float)
    if w_lambda.ndim == 1:
        W = np.outer(w_seed, w_lambda)
    else:
        W = w_seed[:, None] * w_lambda
    if eligible is None:
        eligible = np.ones(values.shape, dtype=bool)
    else:
        eligible = np.asarray(eligible, dtype=bool)
    W = np.where(eligible, W, 0.0)
    denom = W.sum()
    if denom <= 0:
        return np.nan
    return float(np.sum(np.where(eligible, values * W, 0.0)) / denom)


def fusion_birth_radial_weight(s):
    """Unnormalized fusion birth radial weight ∝ n_D n_T 〈σv〉 (Bader-style).

    Matches the campaign firm3d birth profile (n∝1−s⁵, T∝1−s). Does not include
    the Boozer Jacobian; multiply by surface Jacobian separately if available.
    """
    s = np.asarray(s, dtype=float)
    n = 1.0 - s**5
    T = 17.5 * (1.0 - s)
    sigmav = np.zeros_like(T)
    pos = T > 0
    Tp = T[pos]
    sigmav[pos] = Tp ** (-2.0 / 3.0) * np.exp(-19.94 * Tp ** (-1.0 / 3.0))
    return n * n * sigmav


def map_physical_seeds(rho_grid, alpha_grid, rho_seeds, alpha_seeds):
    """Map physical (ρ, α) quadrature nodes to nearest grid indices."""
    rho_grid = np.asarray(rho_grid, dtype=float)
    alpha_grid = np.asarray(alpha_grid, dtype=float)
    seeds = []
    for rho0 in np.asarray(rho_seeds, dtype=float).ravel():
        i_s = int(np.argmin(np.abs(rho_grid - rho0)))
        for a0 in np.asarray(alpha_seeds, dtype=float).ravel():
            d = np.abs(alpha_grid - a0)
            d = np.minimum(d, 2 * np.pi - d)
            i_a = int(np.argmin(d))
            seeds.append((i_s, i_a))
    return seeds


def deterministic_quadrature(
    rho_grid,
    alpha_grid,
    *,
    n_rho_quad=12,
    n_alpha_quad=16,
    rho_min=0.15,
    rho_max=0.90,
    pitch_inv=None,
    pitch_inv_weight=None,
    pitch_weight="gammac",
    B_birth=None,
):
    """Fixed physical (ρ, α) quadrature + pitch weights.

    Seed locations are independent of the J-grid resolution so an h-scan does
    not change the integration measure — only the path metric.

    Parameters
    ----------
    pitch_weight : {"gammac", "firm3d"}
        ``gammac``: Γ_c ∫…dλ weights from ``pitch_inv_weight / pitch_inv²``.
        ``firm3d``: uniform in ``v∥/v`` as in firm3d
        ``initialize_velocity_uniform`` (requires ``B_birth``).
    B_birth : array_like, optional
        Birth |B| for each seed, shape ``(n_seed,)``. Required for
        ``pitch_weight="firm3d"``.

    Returns
    -------
    seeds : list of (i_s, i_a)
    w_seed : ndarray (n_seed,)
    w_lambda : ndarray (n_pitch,) or (n_seed, n_pitch)
    rho_seeds, alpha_seeds : 1d arrays (physical nodes)
    """
    rho_seeds = np.linspace(rho_min, rho_max, int(n_rho_quad))
    alpha_seeds = np.linspace(0.0, 2 * np.pi, int(n_alpha_quad), endpoint=False)
    seeds = map_physical_seeds(rho_grid, alpha_grid, rho_seeds, alpha_seeds)

    # Cell weights: birth(s) × ρ Δρ Δα  (ρ factor ~ area element in cylindrical proxy)
    dr = (rho_max - rho_min) / max(n_rho_quad - 1, 1)
    da = 2 * np.pi / n_alpha_quad
    w_r = fusion_birth_radial_weight(rho_seeds**2) * rho_seeds * dr
    w_seed = np.repeat(w_r, n_alpha_quad) * da

    mode = str(pitch_weight).lower()
    if pitch_inv is None:
        w_lambda = None
    elif mode == "firm3d":
        if B_birth is None:
            raise ValueError("B_birth is required for pitch_weight='firm3d'")
        w_lambda = firm3d_seed_pitch_weights(seeds, pitch_inv, B_birth)
    elif mode == "gammac":
        if pitch_inv_weight is None:
            w_lambda = None
        else:
            pitch_inv = np.asarray(pitch_inv, dtype=float)
            pitch_inv_weight = np.asarray(pitch_inv_weight, dtype=float)
            w = gammac_pitch_weights(pitch_inv, pitch_inv_weight)
            if w.ndim == 2:
                w_lambda = w[w.shape[0] // 2]
            else:
                w_lambda = w
    else:
        raise ValueError(
            f"pitch_weight must be 'gammac' or 'firm3d', got {pitch_weight!r}"
        )
    return seeds, w_seed, w_lambda, rho_seeds, alpha_seeds


def diagnose(
    J,
    mask,
    seeds,
    lambdas,
    *,
    s_grid=None,
    wall_idx=None,
    tol=None,
    tol_fraction=0.5,
):
    """Run persistence filter then flood-fill for each (seed, λ).

    Parameters
    ----------
    J, mask : array_like
        Shape (n_s, n_alpha, n_pitch).
    seeds : sequence of (s_idx, alpha_idx)
        Seed locations in grid indices.
    lambdas : array_like
        Pitch values λ corresponding to the last axis of ``J`` (for reporting).
    s_grid : array_like, optional
        Radial coordinates for pinch reporting.
    wall_idx : int, optional
        Wall radial index for flood fill.
    tol : float or array_like, optional
        Level-set half-width. Scalar, per-pitch (n_pitch,), or full
        (n_seed, n_pitch). If None, ``default_tol`` is used per seed/pitch.
    tol_fraction : float
        Passed to ``default_tol`` when ``tol`` is None.

    Returns
    -------
    report : list of dict
    verdicts : ndarray of str, shape (n_seed, n_pitch)
        One of ``"blocked"``, ``"loss cone"``, ``"confined (disjoint branch)"``,
        ``"no main well"``.
    """
    J = np.asarray(J)
    mask = np.asarray(mask, dtype=bool)
    lambdas = np.asarray(lambdas, dtype=float)
    seeds = list(seeds)
    n_seed = len(seeds)
    n_pitch = J.shape[-1]

    if tol is None:
        tol_arr = np.full((n_seed, n_pitch), np.nan)
        auto_tol = True
    else:
        tol_arr = np.broadcast_to(np.asarray(tol, dtype=float), (n_seed, n_pitch)).copy()
        auto_tol = False

    verdicts = np.empty((n_seed, n_pitch), dtype=object)
    report = []

    for i_lam, lam in enumerate(lambdas):
        J_lam = J[..., i_lam]
        m_lam = mask[..., i_lam]
        for i_seed, (s_idx0, a_idx0) in enumerate(seeds):
            if not m_lam[s_idx0, a_idx0]:
                verdict = "no main well"
                entry = {
                    "lambda": float(lam),
                    "seed": (int(s_idx0), int(a_idx0)),
                    "verdict": verdict,
                    "tol": None,
                    "persistence": None,
                }
                verdicts[i_seed, i_lam] = verdict
                report.append(entry)
                continue

            fast = persistence_check(J_lam, m_lam, s_idx0, a_idx0, s_grid)
            if not fast["connected"]:
                verdict = "blocked"
                used_tol = None
            else:
                if auto_tol:
                    used_tol = default_tol(
                        J_lam, m_lam, s_idx0, a_idx0, fraction=tol_fraction
                    )
                    tol_arr[i_seed, i_lam] = used_tol
                else:
                    used_tol = float(tol_arr[i_seed, i_lam])
                reaches, _ = reaches_wall_flood_fill(
                    J_lam, m_lam, s_idx0, a_idx0, used_tol, wall_idx
                )
                verdict = "loss cone" if reaches else "confined (disjoint branch)"

            verdicts[i_seed, i_lam] = verdict
            report.append(
                {
                    "lambda": float(lam),
                    "seed": (int(s_idx0), int(a_idx0)),
                    "verdict": verdict,
                    "tol": used_tol,
                    "persistence": fast,
                }
            )

    return report, verdicts


def gammac_pitch_weights(pitch_inv, pitch_inv_weight):
    """Pitch weights matching Γ_c's ∫ … dλ measure on a 1/λ quadrature.

    Γ_c sums ``fun(pitch_inv) * pitch_inv_weight / pitch_inv**2`` (see
    ``desc.compute._fast_ion._Gamma_c``).
    """
    pitch_inv = np.asarray(pitch_inv, dtype=float)
    pitch_inv_weight = np.asarray(pitch_inv_weight, dtype=float)
    return pitch_inv_weight / pitch_inv**2


def firm3d_vpar_pitch_weights(pitch_inv, B_birth):
    """Pitch weights matching firm3d's uniform-``v∥`` birth measure.

    firm3d ``initialize_velocity_uniform`` draws ``v∥ ∼ Uniform[-v₀, v₀]`` at
    fixed energy, i.e. ``ξ = v∥/v ∼ Uniform[-1, 1]``. With
    ``pitch_inv = B_crit = B_birth / (1 - ξ²)`` (DESC ``1/λ``), a diagnostic
    that depends only on ``|ξ|`` has average

        ⟨f⟩ = ∫₀¹ f(ξ) dξ

    (folding ±ξ against the Uniform[-1,1] density). This returns Voronoi cell
    widths of the nodes in ξ-space on ``[0, ξ_max]``, shape-broadcast from
    ``pitch_inv`` and ``B_birth``.
    """
    p = np.asarray(pitch_inv, dtype=float)
    B = np.asarray(B_birth, dtype=float)
    if B.ndim == p.ndim:
        B_b = B
    else:
        B_b = np.broadcast_to(B[..., None], p.shape)
    xi2 = 1.0 - B_b / p
    valid = np.isfinite(p) & np.isfinite(B_b) & (p > B_b) & (xi2 > 0.0)
    xi = np.full(p.shape, np.nan)
    xi[valid] = np.sqrt(xi2[valid])

    w = np.zeros(p.shape, dtype=float)
    # Voronoi widths along the last (pitch) axis
    flat_leading = int(np.prod(p.shape[:-1], dtype=int)) if p.ndim > 1 else 1
    xi_r = xi.reshape(flat_leading, p.shape[-1])
    w_r = w.reshape(flat_leading, p.shape[-1])
    for i in range(flat_leading):
        xrow = xi_r[i]
        ok = np.isfinite(xrow)
        n_ok = int(ok.sum())
        if n_ok == 0:
            continue
        idx = np.where(ok)[0]
        xs = xrow[idx]
        order = np.argsort(xs)
        idx = idx[order]
        xs = xs[order]
        if n_ok == 1:
            # Full trapped interval mass on the single node.
            w_r[i, idx[0]] = float(xs[0]) if xs[0] > 0 else 1.0
            continue
        edges = np.empty(n_ok + 1)
        edges[0] = 0.0
        edges[-1] = xs[-1]
        edges[1:-1] = 0.5 * (xs[:-1] + xs[1:])
        w_r[i, idx] = np.diff(edges)
    return w


def firm3d_seed_pitch_weights(seeds, pitch_inv, B_birth):
    """Per-seed firm3d ``v∥`` weights, shape ``(n_seed, n_pitch)``.

    ``pitch_inv`` may be ``(n_pitch,)`` or ``(n_rho, n_pitch)``; in the latter
    case row ``seeds[i][0]`` is used for seed ``i``.
    """
    pitch_inv = np.asarray(pitch_inv, dtype=float)
    B_birth = np.asarray(B_birth, dtype=float).ravel()
    seeds = list(seeds)
    n_seed = len(seeds)
    if B_birth.size != n_seed:
        raise ValueError(
            f"B_birth has length {B_birth.size}, expected n_seed={n_seed}"
        )
    if pitch_inv.ndim == 1:
        p = np.broadcast_to(pitch_inv, (n_seed, pitch_inv.shape[0])).copy()
    elif pitch_inv.ndim == 2:
        p = np.stack([pitch_inv[int(s[0])] for s in seeds], axis=0)
    else:
        raise ValueError(f"pitch_inv ndim={pitch_inv.ndim}, expected 1 or 2")
    return firm3d_vpar_pitch_weights(p, B_birth)


def loss_cone_fraction(verdicts, *, w_seed=None, w_lambda=None):
    """Weighted fraction of (seed, λ) samples with verdict ``loss cone``.

    Parameters
    ----------
    verdicts : array_like of str
        Shape (n_seed, n_pitch).
    w_seed : array_like, optional
        Shape (n_seed,). Default: uniform.
    w_lambda : array_like, optional
        Shape (n_pitch,) or (n_seed, n_pitch). Default: uniform over pitches
        that are not ``no main well`` for that seed.

    Returns
    -------
    float
        Weighted loss-cone fraction in [0, 1].
    """
    verdicts = np.asarray(verdicts, dtype=object)
    n_seed, n_pitch = verdicts.shape
    if w_seed is None:
        w_seed = np.ones(n_seed)
    else:
        w_seed = np.asarray(w_seed, dtype=float)
    if w_lambda is None:
        w_lambda = np.ones(n_pitch)
    w_lambda = np.asarray(w_lambda, dtype=float)
    if w_lambda.ndim == 1:
        W = np.outer(w_seed, w_lambda)
    else:
        W = w_seed[:, None] * w_lambda

    eligible = verdicts != "no main well"
    W = np.where(eligible, W, 0.0)
    denom = W.sum()
    if denom <= 0:
        return np.nan
    numer = W[verdicts == "loss cone"].sum()
    return float(numer / denom)


def lambdas_from_pitch_inv(pitch_inv, rho_index=None):
    """Convert pitch_inv grid to λ values for reporting.

    If ``pitch_inv`` is 2D (n_rho, n_pitch), take the row ``rho_index``
    (default: mid-radius) so a single λ vector labels the pitch axis.
    """
    pitch_inv = np.asarray(pitch_inv, dtype=float)
    if pitch_inv.ndim == 1:
        return 1.0 / pitch_inv
    if rho_index is None:
        rho_index = pitch_inv.shape[0] // 2
    return 1.0 / pitch_inv[rho_index]


def default_seeds(n_rho, n_alpha, rho_indices=None, n_alpha_seeds=8):
    """Build a modest (ρ, α) seed set for surface coverage.

    Multiple α seeds are mandatory: fields are not α-symmetric (see GammaC
    docstring on rational-surface averages).
    """
    if rho_indices is None:
        # Prefer core–midradius seeds; avoid axis (ρ=0) and optionally wall.
        candidates = [i for i in range(n_rho) if i < max(1, n_rho - 1)]
        if len(candidates) >= 3:
            rho_indices = [
                candidates[0],
                candidates[len(candidates) // 2],
                candidates[-1],
            ]
        else:
            rho_indices = candidates or [0]
    alpha_indices = np.linspace(0, n_alpha, n_alpha_seeds, endpoint=False, dtype=int)
    return [(int(i_s), int(i_a)) for i_s in rho_indices for i_a in alpha_indices]
