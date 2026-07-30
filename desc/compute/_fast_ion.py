"""Compute functions for fast ion confinement."""

from functools import partial

from desc.backend import jit, jnp

from ..batching import batch_map
from ..integrals.bounce_integral import Bounce2D
from ..integrals.surface_integral import surface_averages
from ..utils import cross, dot, safediv
from ._neoclassical import _bounce_doc, _bounce_static_argnames
from .data_index import register_compute_fun

# We rewrite equivalents of Nemov et al.'s expressions (21, 22) to resolve
# the indeterminate form of the limit and use single-valued maps of physical
# coordinates. This avoids the computational issues of multivalued maps.
# The derivative (∂/∂ψ)|ϑ,ϕ belongs to flux coordinates which satisfy
# α = ϑ − χ(ψ) ϕ where α is the poloidal label of ψ,α Clebsch coordinates.
# Choosing χ = ι implies ϑ, ϕ are PEST angles.
# ∂G/∂((λB₀)⁻¹) =     λ²B₀  ∫ dℓ (1 − λ|B|/2) / √(1 − λ|B|) ∂|B|/∂ψ / |B|
# ∂V/∂((λB₀)⁻¹) = 3/2 λ²B₀  ∫ dℓ √(1 − λ|B|) R / |B|
# ∂g/∂((λB₀)⁻¹) =     λ²B₀² ∫ dℓ (1 − λ|B|/2) / √(1 − λ|B|) |∇ψ| κ_g / |B|
# K ≝ R dψ/dρ
# tan(π/2 γ_c) =
#              ∫ dℓ (1 − λ|B|/2) / √(1 − λ|B|) |∇ψ| κ_g / |B|
#              ----------------------------------------------
# (|∇ρ| ‖e_α|ρ,ϕ‖)ᵢ ∫ dℓ [ (1 − λ|B|/2)/√(1 − λ|B|) ∂|B|/∂ρ + √(1 − λ|B|) K ] / |B|


def _v_tau(data, B, pitch):
    # Note v τ = 4λ⁻²B₀⁻¹ ∂I/∂((λB₀)⁻¹) where v is the particle velocity,
    # τ is the bounce time, and I is defined in Nemov et al. eq. 36.
    return safediv(2.0, jnp.sqrt(jnp.abs(1 - pitch * B)))


def _drift1(data, B, pitch):
    return (
        safediv(1 - 0.5 * pitch * B, jnp.sqrt(jnp.abs(1 - pitch * B)))
        * data["|grad(psi)|*kappa_g"]
        / B
    )


def _drift2(data, B, pitch):
    return (
        safediv(1 - 0.5 * pitch * B, jnp.sqrt(jnp.abs(1 - pitch * B)))
        * data["|B|_r|v,p"]
        + jnp.sqrt(jnp.abs(1 - pitch * B)) * data["K"]
    ) / B


@register_compute_fun(
    name="Gamma_c",
    label=(
        # Γ_c = π/(8√2) ∫ dλ 〈 ∑ⱼ [v τ γ_c²]ⱼ 〉
        "\\Gamma_c = \\frac{\\pi}{8 \\sqrt{2}} "
        "\\int d\\lambda \\langle \\sum_j (v \\tau \\gamma_c^2)_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Fast ion confinement proxy (scalar)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "min_tz |B|",
        "max_tz |B|",
        "B^phi",
        "B^phi_r|v,p",
        "|B|_r|v,p",
        "b",
        "grad(phi)",
        "grad(psi)",
        "|grad(psi)|",
        "|grad(rho)|",
        "|e_alpha|r,p|",
        "kappa_g",
        "iota_r",
        "V_psi",
    ]
    + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    **_bounce_doc,
)
@partial(jit, static_argnames=_bounce_static_argnames)
def _Gamma_c(params, transforms, profiles, data, **kwargs):
    """Fast ion confinement proxy as defined by Nemov et al.

    Notes
    -----
    A much more performant version is available at https://github.com/unalmis/DESC.
    The reference 2 below refers to that implementation.

    [1] Poloidal motion of trapped particle orbits in real-space coordinates.
        V. V. Nemov, S. V. Kasilov, W. Kernbichler, G. O. Leitold.
        Phys. Plasmas 1 May 2008; 15 (5): 052501.
        https://doi.org/10.1063/1.2912456.
        Equation 61.

    [2] Spectrally accurate, reverse-mode differentiable bounce-averaging algorithm
        and its applications. Kaya Unalmis et al. Journal of Plasma Physics.

    A 3D stellarator magnetic field admits ripple wells that lead to enhanced
    radial drift of trapped particles. The energetic particle confinement
    metric γ_c quantifies whether the contours of the second adiabatic invariant
    close on the flux surfaces. In the limit where the poloidal drift velocity
    majorizes the radial drift velocity, the contours lie parallel to flux
    surfaces. The optimization metric Γ_c averages γ_c² over the distribution
    of trapped particles on each flux surface.

    The radial electric field has a negligible effect, since fast particles
    have high energy with collisionless orbits, so it is assumed to be zero.
    """
    # noqa: unused dependency
    data["Gamma_c"] = _gamma_c_Nemov_model(data, transforms["grid"], kwargs)
    return data


def _gamma_c_Nemov_model(data, grid, kwargs, per_pitch=False):
    """Nemov's Gamma_c, optionally left resolved in the pitch coordinate.

    Parameters
    ----------
    per_pitch : bool
        If ``True``, stop before the quadrature over λ and return the density
        dΓ_c/dλ with shape (num rho, num pitch) on the nodes given by
        ``Bounce2D.get_pitch_inv_quad``, rather than the surface quantity Γ_c
        expanded onto ``grid``. Contracting the density with the λ weights
        ``pitch_inv weight / pitch_inv²`` recovers Γ_c exactly, since that
        contraction is the only step this skips.

    """
    (
        angle,
        Y_B,
        alpha,
        num_transit,
        num_well,
        num_pitch,
        pitch_batch_size,
        surf_batch_size,
        nufft_eps,
        spline,
        quad,
        vander,
    ) = Bounce2D._defaults(-2, grid, **kwargs)

    def Gamma_c(data):
        bounce = Bounce2D(
            grid,
            data,
            data["angle"],
            Y_B,
            alpha,
            num_transit,
            quad,
            nufft_eps=nufft_eps,
            is_fourier=True,
            spline=spline,
            vander=vander,
        )

        def fun(pitch_inv):
            points = bounce.points(pitch_inv, num_well)
            v_tau, drift1, drift2 = bounce.integrate(
                [_v_tau, _drift1, _drift2],
                pitch_inv,
                data,
                ["|grad(psi)|*kappa_g", "|B|_r|v,p", "K"],
                points,
                nufft_eps=nufft_eps,
                is_fourier=True,
            )
            # This is γ_c π/2.
            gamma_c = jnp.arctan(
                safediv(
                    drift1,
                    drift2
                    * bounce.interp_to_argmin(
                        data["|grad(rho)|*|e_alpha|r,p|"],
                        points,
                        nufft_eps=nufft_eps,
                        is_fourier=True,
                    ),
                )
            )
            return (v_tau * gamma_c**2).sum(-1).mean(-2)

        density = batch_map(fun, data["pitch_inv"], pitch_batch_size)
        if per_pitch:
            return density
        return jnp.sum(
            density * data["pitch_inv weight"] / data["pitch_inv"] ** 2, axis=-1
        )

    # It is assumed the grid is sufficiently dense to reconstruct |B|,
    # so anything smoother than |B| may be captured accurately as a single
    # Fourier series rather than transforming each component. Last term in K
    # behaves as ∂log(|B|²/(R₀B₀B^ϕ))/∂ρ |B| where R₀B₀ is a constant with
    # units Tesla meters. Smoothness is determined by positive lower bound of
    # log argument, and hence behaves as ∂log(|B|/B₀)/∂ρ |B| = ∂|B|/∂ρ.
    fun_data = {
        "|grad(psi)|*kappa_g": data["|grad(psi)|"] * data["kappa_g"],
        "|grad(rho)|*|e_alpha|r,p|": data["|grad(rho)|"] * data["|e_alpha|r,p|"],
        "|B|_r|v,p": data["|B|_r|v,p"],
        "K": data["iota_r"]
        * dot(cross(data["grad(psi)"], data["b"]), data["grad(phi)"])
        - (2 * data["|B|_r|v,p"] - data["|B|"] * data["B^phi_r|v,p"] / data["B^phi"]),
    }
    out = Bounce2D.batch(
        Gamma_c,
        fun_data,
        data,
        angle,
        grid,
        num_pitch,
        surf_batch_size,
        expand_out=not per_pitch,
    )
    V_psi = grid.compress(data["V_psi"])[:, jnp.newaxis] if per_pitch else data["V_psi"]
    return out / V_psi / (num_transit * 2**0.5)


def _radial_drift(data, B, pitch):
    return safediv(
        data["cvdrift0"] * (1 - 0.5 * pitch * B), jnp.sqrt(jnp.abs(1 - pitch * B))
    )


def _poloidal_drift(data, B, pitch):
    return safediv(
        (data["gbdrift (periodic)"] + data["gbdrift (secular)/phi"] * data["zeta"])
        * (1 - 0.5 * pitch * B),
        jnp.sqrt(jnp.abs(1 - pitch * B)),
    )


@register_compute_fun(
    name="gamma_c",
    label="\\sum_{w} \\gamma_c(\\rho, \\alpha, \\lambda, w)",
    units="~",
    units_long="None",
    description="Fast ion confinement proxy",
    dim=2,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=[
        "min_tz |B|",
        "max_tz |B|",
        "B^phi",
        "B^phi_r|v,p",
        "|B|_r|v,p",
        "b",
        "grad(phi)",
        "grad(psi)",
        "|grad(psi)|",
        "|grad(rho)|",
        "|e_alpha|r,p|",
        "kappa_g",
        "iota_r",
    ]
    + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    **_bounce_doc,
)
@partial(jit, static_argnames=_bounce_static_argnames)
def _little_gamma_c_Nemov(params, transforms, profiles, data, **kwargs):
    """Fast ion confinement proxy as defined by Nemov et al.

    Returns
    -------
    ∑_w γ_c(ρ, α, λ, w) where w indexes a well.
        Shape (num rho, num alpha, num pitch).

    """
    # noqa: unused dependency
    grid = transforms["grid"]
    (
        angle,
        Y_B,
        alpha,
        num_transit,
        num_well,
        num_pitch,
        _,
        _,
        nufft_eps,
        spline,
        quad,
        vander,
    ) = Bounce2D._defaults(-2, grid, **kwargs)

    def gamma_c0(data):
        bounce = Bounce2D(
            grid,
            data,
            data["angle"],
            Y_B,
            alpha,
            num_transit,
            quad,
            nufft_eps=nufft_eps,
            is_fourier=True,
            spline=spline,
            vander=vander,
        )

        points = bounce.points(data["pitch_inv"], num_well)
        drift1, drift2 = bounce.integrate(
            [_drift1, _drift2],
            data["pitch_inv"],
            data,
            ["|grad(psi)|*kappa_g", "|B|_r|v,p", "K"],
            points,
            nufft_eps=nufft_eps,
            is_fourier=True,
            low_ram=True,
        )
        return (2 / jnp.pi) * jnp.arctan(
            safediv(
                drift1,
                drift2
                * bounce.interp_to_argmin(
                    data["|grad(rho)|*|e_alpha|r,p|"],
                    points,
                    nufft_eps=nufft_eps,
                    is_fourier=True,
                ),
            )
        ).sum(-1)

    fun_data = {
        "|grad(psi)|*kappa_g": data["|grad(psi)|"] * data["kappa_g"],
        "|grad(rho)|*|e_alpha|r,p|": data["|grad(rho)|"] * data["|e_alpha|r,p|"],
        "|B|_r|v,p": data["|B|_r|v,p"],
        "K": data["iota_r"]
        * dot(cross(data["grad(psi)"], data["b"]), data["grad(phi)"])
        - (2 * data["|B|_r|v,p"] - data["|B|"] * data["B^phi_r|v,p"] / data["B^phi"]),
    }
    data["gamma_c"] = Bounce2D.batch(
        gamma_c0, fun_data, data, angle, grid, num_pitch, 1
    )
    return data


@register_compute_fun(
    name="Gamma_c Velasco",
    label=(
        # Γ_c = π/(8√2) ∫ dλ 〈 ∑ⱼ [v τ γ_c²]ⱼ 〉
        "\\Gamma_c = \\frac{\\pi}{8 \\sqrt{2}} "
        "\\int d\\lambda \\langle \\sum_j (v \\tau \\gamma_c^2)_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Fast ion confinement proxy (scalar) "
    "as defined by Velasco et al. (doi:10.1088/1741-4326/ac2994)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "min_tz |B|",
        "max_tz |B|",
        "cvdrift0",
        "gbdrift (periodic)",
        "gbdrift (secular)/phi",
        "V_psi",
    ]
    + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    **_bounce_doc,
)
@partial(jit, static_argnames=_bounce_static_argnames)
def _Gamma_c_Velasco(params, transforms, profiles, data, **kwargs):
    """Fast ion confinement proxy as defined by Velasco et al.

    Notes
    -----
    A much more performant version is available at https://github.com/unalmis/DESC.
    The reference 2 below refers to that implementation.

    [1] A model for the fast evaluation of prompt losses of energetic ions in
        stellarators. Equation 16.
        J.L. Velasco et al. 2021 Nucl. Fusion 61 116059.
        https://doi.org/10.1088/1741-4326/ac2994.

    [2] Spectrally accurate, reverse-mode differentiable bounce-averaging algorithm
        and its applications. Kaya Unalmis et al. Journal of Plasma Physics.

    """
    # noqa: unused dependency
    grid = transforms["grid"]
    (
        angle,
        Y_B,
        alpha,
        num_transit,
        num_well,
        num_pitch,
        pitch_batch_size,
        surf_batch_size,
        nufft_eps,
        spline,
        quad,
        vander,
    ) = Bounce2D._defaults(-1, grid, **kwargs)

    def Gamma_c(data):
        bounce = Bounce2D(
            grid,
            data,
            data["angle"],
            Y_B,
            alpha,
            num_transit,
            quad,
            nufft_eps=nufft_eps,
            is_fourier=True,
            spline=spline,
            vander=vander,
        )

        def fun(pitch_inv):
            v_tau, radial_drift, poloidal_drift = bounce.integrate(
                [_v_tau, _radial_drift, _poloidal_drift],
                pitch_inv,
                data,
                ["cvdrift0", "gbdrift (periodic)", "gbdrift (secular)/phi"],
                num_well=num_well,
                nufft_eps=nufft_eps,
                is_fourier=True,
            )
            # This is γ_c π/2.
            gamma_c = jnp.arctan(safediv(radial_drift, poloidal_drift))
            return (v_tau * gamma_c**2).sum(-1).mean(-2)

        return jnp.sum(
            batch_map(fun, data["pitch_inv"], pitch_batch_size)
            * data["pitch_inv weight"]
            / data["pitch_inv"] ** 2,
            axis=-1,
        )

    data["Gamma_c Velasco"] = (
        Bounce2D.batch(
            Gamma_c,
            {
                "cvdrift0": data["cvdrift0"],
                "gbdrift (periodic)": data["gbdrift (periodic)"],
                "gbdrift (secular)/phi": data["gbdrift (secular)/phi"],
            },
            data,
            angle,
            grid,
            num_pitch,
            surf_batch_size,
            expand_out=True,
        )
        / data["V_psi"]
        / (num_transit * 2**0.5)
    )
    return data


################################################################################
# Velasco et al. prompt loss models.
# J.L. Velasco et al 2021 Nucl. Fusion 61 116059.
# https://doi.org/10.1088/1741-4326/ac2994
################################################################################

# Γ_c weights (γ_c π/2)² by v τ and divides by V_psi √2 num_transit to obtain
# equation 16, which equals π/(4√2) 〈∫dλ B (1−λB)^(−1/2) (γ_c^*)²〉. Replacing
# the weight by a classifier C therefore requires the factor below to obtain
# Velasco's normalization ½ 〈∫dλ B (1−λB)^(−1/2) C〉, for which C ≡ 1 recovers
# f_trapped (equation 24). Setting ``gamma_th=-inf`` in ``Gamma_delta`` makes
# C ≡ 1 and is the recommended check of this normalization.
_VELASCO_NORM = jnp.pi / 2**0.5

_gamma_th_doc = """float :
    Threshold γ_th on γ_c^* above which a superbanana is declared.
    Velasco et al. use ``0.2``, which corresponds to a bounce averaged
    trajectory that traverses a distance 1 in s while precessing π in α.
    Varying the equivalent drift ratio between π/2 and 2π does not
    significantly change the predicted loss fractions. Default is ``0.2``.
    """
_main_well_doc = """bool :
    Whether to keep only the deepest well of each field line, the one holding
    the smallest |B|. Velasco et al. discard ripple wells on the grounds that
    their limited angular extent lets a particle precess out of the well before
    it drifts appreciably in radius, so that they drive stochastic rather than
    prompt losses. Setting this to ``True`` reproduces that assumption and makes
    γ_c^* a function of α and λ alone. Setting it to ``False`` sums over every
    well, which is the convention of ``Gamma_c`` and makes the models bounded by
    the full ``f_trapped (Velasco)``. Default is ``False``.
    """
_velasco_doc = {
    **_bounce_doc,
    "gamma_th": _gamma_th_doc,
    "main_well": _main_well_doc,
}
_velasco_static_argnames = _bounce_static_argnames + ("gamma_th", "main_well")


def _gamma_c_star(radial_drift, poloidal_drift):
    """Velasco et al. equation 14.

    γ_c^* = 2/π arctan( 〈𝐯_M⋅∇s〉 / |〈𝐯_M⋅∇α〉| ).

    This differs from Nemov's γ_c (equation 15) only in that the denominator is
    unsigned, so that (γ_c^*)² = γ_c² while the sign of γ_c^* is the sign of the
    bounce averaged radial drift. That sign is what distinguishes the inward
    (α_in, γ_c^* < −γ_th) from the outward (α_out, γ_c^* > γ_th) edge of a
    superbanana, and hence is what the models of Velasco et al. sections 4.1 and
    4.2 classify on.
    """
    return (2 / jnp.pi) * jnp.arctan(safediv(radial_drift, jnp.abs(poloidal_drift)))


def _velasco_drifts(bounce, data, pitch_inv, num_well, nufft_eps, main_well):
    """Return v τ, γ_c^*, and the bounce averaged tangential drift."""
    points = bounce.points(pitch_inv, num_well)
    v_tau, radial_drift, poloidal_drift = bounce.integrate(
        [_v_tau, _radial_drift, _poloidal_drift],
        pitch_inv,
        data,
        ["cvdrift0", "gbdrift (periodic)", "gbdrift (secular)/phi"],
        points=points,
        nufft_eps=nufft_eps,
        is_fourier=True,
    )
    if main_well:
        z1, z2 = points
        exists = z1 < z2
        # Wells that were not detected are padded with z1 = z2 = 0; sending
        # their depth to infinity keeps argmin from selecting them.
        deepest = jnp.argmin(
            jnp.where(
                exists,
                bounce.interp_to_argmin(
                    data["|B|"], points, nufft_eps=nufft_eps, is_fourier=True
                ),
                jnp.inf,
            ),
            axis=-1,
        )[..., None]
        v_tau = jnp.take_along_axis(v_tau, deepest, axis=-1) * jnp.take_along_axis(
            exists, deepest, axis=-1
        )
        radial_drift = jnp.take_along_axis(radial_drift, deepest, axis=-1)
        poloidal_drift = jnp.take_along_axis(poloidal_drift, deepest, axis=-1)
    return v_tau, _gamma_c_star(radial_drift, poloidal_drift), poloidal_drift


def _superbanana_exists(gamma_c_star, gamma_th):
    """Heaviside of Velasco et al. equation 22.

    ``H(max(γ_c^*(α|λ)) − γ_th)`` where the maximum is taken over the field line
    label α and over the wells of each field line. The result depends only on
    (ρ, λ), so it is returned with the α and well axes kept as singletons.
    """
    return jnp.max(gamma_c_star, axis=(-3, -1), keepdims=True) > gamma_th


def _alpha_loss_cone(gamma_c_star, poloidal_drift, gamma_th):
    """Product of Heavisides in Velasco et al. equation 25.

    A trapped particle precesses monotonically in α in the direction of
    〈𝐯_M⋅∇α〉 at fixed λ. It is lost when it reaches an outward superbanana at
    α_out (γ_c^* > γ_th) and is retained when it first reaches an inward one at
    α_in (γ_c^* < −γ_th), so the loss cone is the set of α that meet α_out
    before α_in when marching along the precession direction. Marching over the
    whole periodic α grid generalizes equation 25 to any number of superbanana
    pairs and handles the periodicity of α automatically.

    Parameters
    ----------
    gamma_c_star, poloidal_drift : jnp.ndarray
        Shape (num rho, num alpha, num pitch, num well). The α axis must be a
        sorted grid that covers [0, 2π) so that neighboring indices are
        neighboring field lines.

    Returns
    -------
    jnp.ndarray
        Boolean array shaped like ``gamma_c_star``.

    """
    num_alpha = gamma_c_star.shape[-3]
    # Move α last so the march is a gather along the final axis.
    outward = jnp.moveaxis(gamma_c_star > gamma_th, -3, -1)
    marked = jnp.moveaxis(jnp.abs(gamma_c_star) > gamma_th, -3, -1)
    # Particles with no tangential precession never leave α; treating them as
    # co-precessing only affects a measure zero set of the α grid.
    step = jnp.where(jnp.moveaxis(poloidal_drift, -3, -1) < 0, -1, 1)

    index = jnp.arange(num_alpha)
    # visit[..., i, k] is the α index reached after k precession steps from i.
    visit = (index[:, None] + step[..., None] * index) % num_alpha
    shape = visit.shape
    hit = jnp.take_along_axis(
        jnp.broadcast_to(marked[..., None, :], shape), visit, axis=-1
    )
    exits = jnp.take_along_axis(
        jnp.broadcast_to(outward[..., None, :], shape), visit, axis=-1
    )
    # argmax returns the first ``True``; guard the case of no superbanana at all,
    # for which argmax would spuriously select the starting point.
    first = jnp.argmax(hit, axis=-1)[..., None]
    lost = jnp.take_along_axis(exits, first, axis=-1)[..., 0] & jnp.any(
        hit, axis=(-2, -1), keepdims=True
    )[..., 0]
    return jnp.moveaxis(lost, -1, -3)


def _velasco_model(classifier, data, grid, kwargs, per_pitch=False):
    """Phase space average of a 0/1 orbit classifier, normalized as Velasco.

    Returns ½ 〈∫dλ B (1−λB)^(−1/2) C 〉 where C is the classifier, so that the
    result lies between 0 and ``f_trapped (Velasco)``. With ``main_well=True``
    the upper bound is instead the trapped fraction held by the deepest well of
    each field line, which is obtained by evaluating ``Gamma_delta`` with
    ``gamma_th=-inf``.

    Parameters
    ----------
    per_pitch : bool
        If ``True``, stop before the quadrature over λ and return the density
        dΓ/dλ with shape (num rho, num pitch) on the nodes given by
        ``Bounce2D.get_pitch_inv_quad``, rather than the surface quantity Γ
        expanded onto ``grid``. Contracting the density with the λ weights
        ``pitch_inv weight / pitch_inv²`` recovers Γ exactly, since that
        contraction is the only step this skips. Intended for diagnostics that
        resolve which pitch angles a model declares lost.

    """
    (
        angle,
        Y_B,
        alpha,
        num_transit,
        num_well,
        num_pitch,
        pitch_batch_size,
        surf_batch_size,
        nufft_eps,
        spline,
        quad,
        vander,
    ) = Bounce2D._defaults(-1, grid, **kwargs)
    gamma_th = kwargs.get("gamma_th", 0.2)
    main_well = kwargs.get("main_well", False)

    def Gamma(data):
        bounce = Bounce2D(
            grid,
            data,
            data["angle"],
            Y_B,
            alpha,
            num_transit,
            quad,
            nufft_eps=nufft_eps,
            is_fourier=True,
            spline=spline,
            vander=vander,
        )

        def fun(pitch_inv):
            v_tau, gamma_c_star, poloidal_drift = _velasco_drifts(
                bounce, data, pitch_inv, num_well, nufft_eps, main_well
            )
            unconfined = classifier(gamma_c_star, poloidal_drift, gamma_th)
            return (v_tau * unconfined).sum(-1).mean(-2)

        density = batch_map(fun, data["pitch_inv"], pitch_batch_size)
        if per_pitch:
            return density
        return jnp.sum(
            density * data["pitch_inv weight"] / data["pitch_inv"] ** 2, axis=-1
        )

    out = Bounce2D.batch(
        Gamma,
        {
            "cvdrift0": data["cvdrift0"],
            "gbdrift (periodic)": data["gbdrift (periodic)"],
            "gbdrift (secular)/phi": data["gbdrift (secular)/phi"],
        },
        data,
        angle,
        grid,
        num_pitch,
        surf_batch_size,
        expand_out=not per_pitch,
    )
    V_psi = grid.compress(data["V_psi"])[:, jnp.newaxis] if per_pitch else data["V_psi"]
    return out * _VELASCO_NORM / V_psi / (num_transit * 2**0.5)


_velasco_data = [
    "min_tz |B|",
    "max_tz |B|",
    "cvdrift0",
    "gbdrift (periodic)",
    "gbdrift (secular)/phi",
    "V_psi",
] + Bounce2D.required_names


@register_compute_fun(
    name="f_trapped (Velasco)",
    label="f_{\\mathrm{trapped}} = \\langle \\sqrt{1 - B / B_{\\max}} \\rangle",
    units="~",
    units_long="None",
    description="Fraction of trapped particles bounding Gamma_alpha and "
    "Gamma_delta, as defined by Velasco et al. "
    "(doi:10.1088/1741-4326/ac2994)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["|B|", "max_tz |B|", "sqrt(g)"],
    resolution_requirement="tz",
)
def _f_trapped_Velasco(params, transforms, profiles, data, **kwargs):
    """Fraction of trapped particles, Velasco et al. equation 24.

    This is the closed form of ½ 〈∫ dλ B (1−λB)^(−1/2)〉 over the trapped
    domain λ ∈ [B_max⁻¹, B⁻¹]. It is the upper bound of both ``Gamma_delta``
    and ``Gamma_alpha``, which classify a subset of the trapped particles as
    promptly lost.

    Note this is a different quantity from ``trapped fraction``, the effective
    trapped particle fraction of neoclassical bootstrap theory.
    """
    data["f_trapped (Velasco)"] = surface_averages(
        transforms["grid"],
        jnp.sqrt(jnp.abs(1 - data["|B|"] / data["max_tz |B|"])),
        sqrt_g=data["sqrt(g)"],
    )
    return data


@register_compute_fun(
    name="Gamma_delta",
    label=(
        # Γ_δ = ½ 〈∫ dλ B (1−λB)^(−1/2) H(max(γ_c^*(α|λ)) − γ_th) 〉
        "\\Gamma_{\\delta} = \\frac{1}{2} \\left\\langle \\int d\\lambda "
        "\\frac{B}{\\sqrt{1 - \\lambda B}} H\\left("
        "\\max(\\gamma_c^*(\\alpha \\vert \\lambda)) - \\gamma_{\\mathrm{th}}"
        "\\right) \\right\\rangle"
    ),
    units="~",
    units_long="None",
    description="Prompt loss fraction of energetic ions from superbanana "
    "existence (Velasco et al. model I, doi:10.1088/1741-4326/ac2994)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=_velasco_data,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    **_velasco_doc,
)
@partial(jit, static_argnames=_velasco_static_argnames)
def _Gamma_delta(params, transforms, profiles, data, **kwargs):
    """Prompt loss model I of Velasco et al., equation 22.

    Notes
    -----
    [1] A model for the fast evaluation of prompt losses of energetic ions in
        stellarators. Equation 22.
        J.L. Velasco et al. 2021 Nucl. Fusion 61 116059.
        https://doi.org/10.1088/1741-4326/ac2994.

    Every particle whose pitch angle admits a superbanana somewhere on the flux
    surface is counted as lost. Unlike ``Gamma_c``, which is a proxy for how far
    the contours of J deviate from the flux surface, this is a classification of
    orbits into confined and unconfined and is therefore a direct estimate of
    the loss fraction, bounded above by ``f_trapped (Velasco)``.

    Ignoring where on the surface the superbanana sits makes this model
    pessimistic; ``Gamma_alpha`` refines it with the α loss cone and is the
    better predictor. Reference [1] section 5.4 reports that Γ_δ generally
    overestimates the losses found by full orbit simulations.

    Warnings
    --------
    The Heaviside classification is not differentiable, so this is a diagnostic
    rather than an optimization objective.

    """
    data["Gamma_delta"] = _velasco_model(
        lambda g, d, th: _superbanana_exists(g, th), data, transforms["grid"], kwargs
    )
    return data


@register_compute_fun(
    name="Gamma_alpha",
    label=(
        # Γ_α = ½ 〈∫ dλ B (1−λB)^(−1/2) H((α_out−α)〈v_M⋅∇α〉) H((α−α_in)〈v_M⋅∇α〉) 〉
        "\\Gamma_{\\alpha} = \\frac{1}{2} \\left\\langle \\int d\\lambda "
        "\\frac{B}{\\sqrt{1 - \\lambda B}} "
        "H\\left((\\alpha_{\\mathrm{out}} - \\alpha) "
        "\\overline{\\mathbf{v}_M \\cdot \\nabla \\alpha}\\right) "
        "H\\left((\\alpha - \\alpha_{\\mathrm{in}}) "
        "\\overline{\\mathbf{v}_M \\cdot \\nabla \\alpha}\\right) "
        "\\right\\rangle"
    ),
    units="~",
    units_long="None",
    description="Prompt loss fraction of energetic ions from the alpha loss "
    "cone (Velasco et al. model II, doi:10.1088/1741-4326/ac2994)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=_velasco_data,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    **_velasco_doc,
)
@partial(jit, static_argnames=_velasco_static_argnames)
def _Gamma_alpha(params, transforms, profiles, data, **kwargs):
    """Prompt loss model II of Velasco et al., equation 25.

    Notes
    -----
    [1] A model for the fast evaluation of prompt losses of energetic ions in
        stellarators. Equation 25.
        J.L. Velasco et al. 2021 Nucl. Fusion 61 116059.
        https://doi.org/10.1088/1741-4326/ac2994.

    Refines ``Gamma_delta`` by asking not only whether a superbanana exists at a
    given pitch angle but whether the particle precesses into it. Only the
    trapped particles born between an inward superbanana at α_in and the next
    outward one at α_out, in the direction of their tangential magnetic drift,
    escape; the rest stay on closed contours of J. Reference [1] validates
    Γ_α against ASCOT as a quantitative prediction of the prompt loss fraction,
    f_pl = Γ_α (equation 27).

    Because the α loss cone is resolved on the grid of field line labels, the
    keyword ``alpha`` must be a sorted grid covering [0, 2π), e.g.
    ``np.linspace(0, 2 * np.pi, 64, endpoint=False)``, rather than the single
    field line that is the default of the other bounce averaged metrics. With
    an explicit α grid, use ``num_transit=1`` so the surface is covered once.

    Warnings
    --------
    The Heaviside classification is not differentiable, so this is a diagnostic
    rather than an optimization objective.

    Particles are assumed to remain in the same well index while precessing in
    α. This is exact under ``main_well=True``, the single well per field line
    limit that reference [1] assumes, and degrades with ``main_well=False``
    where ripple wells appear and disappear across neighboring field lines.

    """
    data["Gamma_alpha"] = _velasco_model(
        _alpha_loss_cone, data, transforms["grid"], kwargs
    )
    return data
