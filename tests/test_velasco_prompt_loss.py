"""Tests for the Velasco et al. prompt loss models Gamma_delta and Gamma_alpha."""

import numpy as np
import pytest

from desc.compute._fast_ion import _alpha_loss_cone
from desc.examples import get
from desc.grid import LinearGrid
from desc.integrals.bounce_integral import Bounce2D

GAMMA_TH = 0.2


def _brute_force_loss_cone(gamma_c_star, poloidal_drift, gamma_th):
    """Velasco et al. equation 25, looped in Python over the alpha grid."""
    g = np.asarray(gamma_c_star)
    d = np.asarray(poloidal_drift)
    n_r, n_a, n_p, n_w = g.shape
    lost = np.zeros(g.shape, bool)
    for i_r in range(n_r):
        for i_p in range(n_p):
            for i_w in range(n_w):
                col = g[i_r, :, i_p, i_w]
                marked = np.abs(col) > gamma_th
                if not marked.any():
                    continue
                for i_a in range(n_a):
                    step = -1 if d[i_r, i_a, i_p, i_w] < 0 else 1
                    for k in range(n_a):
                        j = (i_a + step * k) % n_a
                        if marked[j]:
                            lost[i_r, i_a, i_p, i_w] = col[j] > gamma_th
                            break
    return lost


@pytest.mark.unit
@pytest.mark.parametrize("scale", [0.05, 0.5, 1.0])
def test_alpha_loss_cone_vs_brute_force(scale):
    """The vectorized alpha march matches a direct Python implementation.

    The scale sets how much of the alpha grid is superbanana, covering the
    empty, sparse, and dense limits.
    """
    rng = np.random.default_rng(0)
    shape = (2, 16, 3, 4)
    for _ in range(5):
        g = np.tanh(scale * rng.standard_normal(shape))
        d = rng.standard_normal(shape)
        np.testing.assert_array_equal(
            np.asarray(_alpha_loss_cone(g, d, GAMMA_TH)),
            _brute_force_loss_cone(g, d, GAMMA_TH),
        )


@pytest.mark.unit
def test_alpha_loss_cone_precession_direction():
    """A single superbanana pair fills the arc traversed toward alpha_out."""
    n_a = 12
    g = np.zeros((1, n_a, 1, 1))
    g[0, 2, 0, 0] = -1.0  # alpha_in
    g[0, 7, 0, 0] = +1.0  # alpha_out

    # Precessing toward increasing alpha, the loss cone is the arc that runs
    # from just past alpha_in up to and including alpha_out. alpha_in itself
    # reaches its own inward superbanana first and stays confined.
    lost = np.asarray(_alpha_loss_cone(g, np.ones_like(g), GAMMA_TH))[0, :, 0, 0]
    np.testing.assert_array_equal(np.where(lost)[0], np.arange(3, 8))

    # Reversing the precession reverses the arc, which wraps through alpha = 0.
    lost = np.asarray(_alpha_loss_cone(g, -np.ones_like(g), GAMMA_TH))[0, :, 0, 0]
    np.testing.assert_array_equal(
        np.where(lost)[0], np.array([0, 1, 7, 8, 9, 10, 11])
    )


@pytest.mark.unit
def test_alpha_loss_cone_no_superbanana():
    """No location exceeds the threshold, so nothing is classified as lost."""
    g = np.full((1, 8, 1, 1), 0.1)
    assert not np.asarray(_alpha_loss_cone(g, np.ones_like(g), GAMMA_TH)).any()


@pytest.mark.regression
@pytest.mark.parametrize("main_well", [False, True])
def test_velasco_models_normalization(main_well):
    """Both models span exactly zero to f_trapped as gamma_th is swept.

    With ``gamma_th = -inf`` every trapped orbit is classified unconfined, so
    the phase space average must reproduce Velasco equation 24,
    f_trapped = <sqrt(1 - B / B_max)>, which is evaluated in closed form
    without bounce integrals. This pins the normalization of both models.

    Keeping only the deepest well drops the trapped particles held by ripple
    wells, so there the bound is the main well trapped fraction rather than the
    full one, and only the upper bound of the comparison survives.

    The bound approaches f_trapped from below as ``num_transit`` grows, because
    a well that straddles either end of the finite field line is discarded. That
    truncation decays like one over the field line length, so on W7-X the
    residual is 25 % for a single transit and 7 % for four, which sets the
    tolerance used here.
    """
    eq = get("W7-X")
    rho = np.linspace(0.3, 0.9, 3)
    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
    kwargs = dict(
        angle=Bounce2D.angle(eq, X=32, Y=32, rho=rho, tol=1e-10),
        alpha=np.linspace(0, 2 * np.pi, 24, endpoint=False),
        num_transit=4,
        num_well=80,
        num_pitch=48,
        num_quad=32,
        Y_B=grid.num_zeta * grid.NFP,
        nufft_eps=1e-10,
        main_well=main_well,
    )
    names = ["Gamma_alpha", "Gamma_delta"]

    f_trapped = grid.compress(
        eq.compute("f_trapped (Velasco)", grid)["f_trapped (Velasco)"]
    )
    everything = eq.compute(names, grid, gamma_th=-np.inf, **kwargs)
    nothing = eq.compute(names, grid, gamma_th=np.inf, **kwargs)
    nominal = eq.compute(names, grid, gamma_th=GAMMA_TH, **kwargs)

    # Classifying every trapped orbit as lost recovers the trapped fraction.
    bound = grid.compress(everything["Gamma_delta"])
    for name in names:
        np.testing.assert_allclose(
            grid.compress(everything[name]), bound, rtol=1e-12,
            err_msg=f"{name} disagrees with Gamma_delta when nothing is confined",
        )
        np.testing.assert_allclose(grid.compress(nothing[name]), 0, atol=1e-14)
    assert (bound < f_trapped).all()
    if not main_well:
        np.testing.assert_allclose(bound, f_trapped, rtol=1e-1)

    g_a = grid.compress(nominal["Gamma_alpha"])
    g_d = grid.compress(nominal["Gamma_delta"])
    # Model II classifies a subset of what model I classifies.
    assert (g_a >= 0).all()
    assert (g_a <= g_d + 1e-12).all()
    assert (g_d <= bound + 1e-12).all()
