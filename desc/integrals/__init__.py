"""Classes for function integration."""

from ._interp_utils import nufft1d2r, nufft2d2r
from .bounce_integral import Bounce1D, Bounce2D
from .jpar_contour import (
    JMainResult,
    barely_trapped_crit,
    build_j_main,
    characteristic_delta_j,
    default_seeds,
    default_tol,
    deterministic_quadrature,
    diagnose,
    diagnose_margin,
    dilate_mask,
    firm3d_seed_pitch_weights,
    firm3d_vpar_pitch_weights,
    fusion_birth_radial_weight,
    gammac_pitch_weights,
    hard_range,
    is_barely_trapped,
    jpar_integrand,
    lambdas_from_pitch_inv,
    loss_cone_fraction,
    map_physical_seeds,
    minimax_margin,
    p_loss_from_margin,
    persistence_check,
    reaches_wall_flood_fill,
    select_main_well,
    weighted_fraction,
)
from .singularities import (
    DFTInterpolator,
    FFTInterpolator,
    compute_B_plasma,
    singular_integral,
    virtual_casing_biot_savart,
)
from .surface_integral import (
    line_integrals,
    surface_averages,
    surface_averages_map,
    surface_integrals,
    surface_integrals_map,
    surface_integrals_transform,
    surface_max,
    surface_min,
    surface_variance,
)
