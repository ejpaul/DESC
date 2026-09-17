"""Tests for the bounce-averaged-flow loss objective."""

import numpy as np
import pytest

from desc.examples import get
from desc.objectives import BounceFlowLoss, ObjectiveFunction

_eV = 1.602176634e-19


def _small(eq, bcrit, **kw):
    kwargs = dict(
        rho=np.sqrt(np.linspace(0.05, 0.9, 6)),
        M_tab=8,
        N_tab=2,
        ns=24,
        na=32,
        num_quad=8,
        nscan=48,
        k=10,
    )
    kwargs.update(kw)
    return BounceFlowLoss(eq, bcrit=bcrit, **kwargs)


@pytest.mark.unit
def test_bounce_flow_axisymmetric_no_loss():
    """In an axisymmetric field the bounce-averaged radial drift vanishes.

    The exact class loss is zero; what remains is the grid coarse-graining of the
    operator, which composing k bounces per step drives to the noise floor.
    """
    eq = get("DSHAPE")
    eq.change_resolution(4, 4, 0, 8, 8, 0)
    data = eq.compute(["min_tz |B|", "max_tz |B|"])
    Bc = 0.5 * (float(data["min_tz |B|"].min()) + float(data["max_tz |B|"].max()))
    obj = _small(eq, [Bc], N_tab=2, k=25)
    obj.build(verbose=0)
    loss = np.asarray(obj.class_loss(eq.params_dict))
    assert np.all(np.isfinite(loss))
    assert loss[0] < 1e-3


@pytest.mark.unit
def test_bounce_flow_build_and_shape():
    """Objective builds through ObjectiveFunction with reverse-mode derivatives."""
    eq = get("HELIOTRON")
    eq.change_resolution(3, 3, 3, 6, 6, 6)
    data = eq.compute(["min_tz |B|", "max_tz |B|"])
    lo, hi = float(data["min_tz |B|"].max()), float(data["max_tz |B|"].min())
    bcrit = np.linspace(lo + 0.3 * (hi - lo), hi - 0.3 * (hi - lo), 2)
    obj = _small(eq, bcrit, Ekin=3.52e4 * _eV)
    of = ObjectiveFunction(obj)
    of.build(verbose=0)
    assert of._deriv_mode == "blocked"
    f = np.asarray(of.compute_scaled_error(of.x()))
    assert f.shape == (2,)
    assert np.all(np.isfinite(f)) and np.all(f >= 0)


@pytest.mark.slow
def test_bounce_flow_gradient_vs_finite_difference():
    """Reverse-mode gradient through tables, quadrature, flow map and operator."""
    eq = get("HELIOTRON")
    eq.change_resolution(4, 4, 4, 8, 8, 8)
    data = eq.compute(["min_tz |B|", "max_tz |B|"])
    Bc = 0.5 * (float(data["min_tz |B|"].max()) + float(data["max_tz |B|"].min()))
    obj = _small(
        eq,
        [Bc],
        Ekin=3.52e4 * _eV,
        rho=np.sqrt(np.linspace(0.05, 0.9, 8)),
        M_tab=12,
        N_tab=6,
        ns=32,
        na=40,
        num_quad=12,
        nscan=64,
    )
    of = ObjectiveFunction(obj)
    of.build(verbose=0)
    x = of.x()
    g = np.asarray(of.grad(x))
    assert np.all(np.isfinite(g))
    rng = np.random.default_rng(0)
    v = rng.normal(size=x.shape)
    v /= np.linalg.norm(v)
    eps = 1e-5
    fd = (
        float(of.compute_scalar(x + eps * v)) - float(of.compute_scalar(x - eps * v))
    ) / (2 * eps)
    np.testing.assert_allclose(fd, g @ v, rtol=5e-2)
