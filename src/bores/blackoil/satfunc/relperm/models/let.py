"""LET (Lomeland-Ebeltoft-Thomas) relative permeability model and table."""

import typing

import attrs
import numba
import numpy as np

from bores.blackoil.satfunc.relperm.mixing_rules import (
    MixingRule,
    baker_linear_rule,
    get_mixing_rule,
)
from bores.blackoil.satfunc.relperm.models.derivatives import (
    WettabilitySystem,
    apply_oil_wet_mixing_rule,
    apply_water_wet_mixing_rule,
    broadcast_saturations,
    finalize_mixed_wet_derivatives,
    finalize_oil_wet_derivatives,
    finalize_water_wet_derivatives,
)
from bores.blackoil.satfunc.relperm.tables import (
    MinimumRelPerm,
    RelativePermeabilityTable,
    clamp_relperm,
    resolve_min_relperm,
    show_invalid_saturation,
)
from bores.constants import c
from bores.errors import ValidationError
from bores.serde.base import Serializable
from bores.types import (
    FluidPhase,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    RelativePermeabilities,
    RelativePermeabilityDerivatives,
    Wettability,
)

__all__ = [
    "LETParameters",
    "LETThreePhaseRelPermTable",
    "compute_let_relative_permeabilities",
    "compute_let_relative_permeability_derivatives",
]


@attrs.frozen
class LETParameters(Serializable):
    """
    LET curve-shape parameters for a single relative permeability curve.

    The LET correlation computes relative permeability from normalized
    saturation S* as:

        kr = kr_max * S*^L / (S*^L + E * (1 - S*)^T)

    where L, E, and T control different regions of the curve:

    - **L** (low-end): Controls curvature at low normalized saturation.
      Higher values delay the onset of flow (the curve stays near zero longer
      before rising). Analogous to a Corey exponent for the lower end.

    - **E** (elevation): Controls the overall position/elevation of the curve
      between the endpoints. Higher values push the curve downward (lower kr
      at intermediate saturations). E = 1 gives a curve similar to a simple
      power law. E < 1 raises the curve; E > 1 suppresses it.

    - **T** (top-end): Controls curvature at high normalized saturation.
      Higher values make the curve flatten earlier as it approaches kr_max
      (the curve reaches its plateau sooner). Analogous to a Corey exponent
      for the upper end.

    All three parameters must be positive. Typical ranges are L in [0.5, 5],
    E in [0.1, 10], and T in [0.5, 5].
    """

    L: Number = 2.0
    """Low-end shape parameter. Controls curvature near zero normalized saturation."""
    E: Number = 1.0
    """Elevation parameter. Controls overall curve height at intermediate saturations."""
    T: Number = 2.0
    """Top-end shape parameter. Controls curvature near maximum normalized saturation."""

    def __attrs_post_init__(self) -> None:
        if self.L <= 0:
            raise ValidationError(f"LET parameter `L` must be positive, got {self.L}")
        if self.E <= 0:
            raise ValidationError(f"LET parameter `E` must be positive, got {self.E}")
        if self.T <= 0:
            raise ValidationError(f"LET parameter `T` must be positive, got {self.T}")


@numba.njit(cache=True)
def _let_relperm(
    normalized_saturation: NumberArray[NDimension],
    L: Number,
    E: Number,
    T: Number,
) -> NumberArray[NDimension]:
    """
    Core LET relative permeability formula (without endpoint scaling).

    Computes: S*^L / (S*^L + E * (1 - S*)^T)

    Returns 0 when S* = 0, and 1 when S* = 1. For intermediate values the
    curve shape is governed by L, E, and T.

    :param normalized_saturation: Effective (normalized) saturation, clipped to [0, 1].
    :param L: Low-end shape parameter (> 0).
    :param E: Elevation parameter (> 0).
    :param T: Top-end shape parameter (> 0).
    :return: Relative permeability value(s) in [0, 1].
    """
    s = normalized_saturation
    numerator = s**L
    denominator = numerator + E * (1.0 - s) ** T
    # Safe division: denominator is zero only when both s^L = 0 and (1-s)^T = 0,
    # which requires s = 0 and s = 1 simultaneously (impossible). For s = 0,
    # numerator = 0 and denominator = E > 0, so result = 0. For s = 1,
    # numerator = 1 and denominator = 1, so result = 1. No special handling needed
    # for well-formed inputs, but protect against floating-point edge cases.
    return np.where(denominator > 0.0, numerator / denominator, 0.0)  # type: ignore[return-value]


def compute_let_relative_permeabilities(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    water_L: Number,
    water_E: Number,
    water_T: Number,
    oil_water_L: Number,
    oil_water_E: Number,
    oil_water_T: Number,
    gas_oil_L: Number,
    gas_oil_E: Number,
    gas_oil_T: Number,
    gas_L: Number,
    gas_E: Number,
    gas_T: Number,
    maximum_water_relperm: Number = 1.0,
    maximum_oil_relperm: Number = 1.0,
    maximum_gas_relperm: Number = 1.0,
    wettability: Wettability = Wettability.WATER_WET,
    mixed_wet_water_fraction: Number = 0.5,
    mixing_rule: MixingRule = baker_linear_rule,
    saturation_epsilon: Number = 1e-6,
    minimum_mobile_pore_space: Number = 1e-9,
    minimum_water_relperm: Number | None = None,
    minimum_oil_relperm: Number | None = None,
    minimum_gas_relperm: Number | None = None,
) -> tuple[NumberOrArray[NDimension], NumberOrArray[NDimension], NumberOrArray[NDimension]]:
    """
    Compute three-phase relative permeabilities using the LET correlation.

    The LET (Lomeland-Ebeltoft-Thomas) model uses three curve-shape parameters
    (L, E, T) per phase-pair, providing more flexibility than the single Corey
    exponent for fitting laboratory data. Each two-phase kr curve is computed
    independently from normalized saturation, then the three-phase oil kr is
    obtained through a mixing rule.

    Supports both scalar and array inputs for saturations.

    :param water_saturation: Water saturation (fraction, 0 to 1).
    :param oil_saturation: Oil saturation (fraction, 0 to 1).
    :param gas_saturation: Gas saturation (fraction, 0 to 1).
    :param irreducible_water_saturation: Irreducible water saturation (swc).
    :param residual_oil_saturation_water: Residual oil saturation to waterflood (sorw).
    :param residual_oil_saturation_gas: Residual oil saturation to gas flood (sorg).
    :param residual_gas_saturation: Residual gas saturation (sgr).
    :param water_L: Water LET `L` parameter.
    :param water_E: Water LET `E` parameter.
    :param water_T: Water LET `T` parameter.
    :param oil_water_L: Oil (water-oil system) LET `L` parameter.
    :param oil_water_E: Oil (water-oil system) LET `E` parameter.
    :param oil_water_T: Oil (water-oil system) LET `T` parameter.
    :param gas_oil_L: Oil (gas-oil system) LET `L` parameter.
    :param gas_oil_E: Oil (gas-oil system) LET `E` parameter.
    :param gas_oil_T: Oil (gas-oil system) LET `T` parameter.
    :param gas_L: Gas LET `L` parameter.
    :param gas_E: Gas LET `E` parameter.
    :param gas_T: Gas LET `T` parameter.
    :param maximum_water_relperm: Endpoint relative permeability for water (krw_max).
    :param maximum_oil_relperm: Endpoint relative permeability for oil (kro_max).
    :param maximum_gas_relperm: Endpoint relative permeability for gas (krg_max).
    :param wettability: Wettability type (water-wet or oil-wet).
    :param mixed_wet_water_fraction: Fraction of water-wet behavior in mixed-wet case (0 to 1).
    :param mixing_rule: Three-phase mixing rule for oil relative permeability.
    :param minimum_water_relperm: Resolved minimum min_value for water kr (`None` = no min_value).
    :param minimum_oil_relperm: Resolved minimum min_value for oil kr (`None` = no min_value).
    :param minimum_gas_relperm: Resolved minimum min_value for gas kr (`None` = no min_value).
    :return: (krw, kro, krg) tuple of relative permeabilities.
    """
    sw = np.atleast_1d(water_saturation)
    so = np.atleast_1d(oil_saturation)
    sg = np.atleast_1d(gas_saturation)
    is_scalar = (
        np.isscalar(water_saturation)
        and np.isscalar(oil_saturation)
        and np.isscalar(gas_saturation)
        and np.isscalar(irreducible_water_saturation)
        and np.isscalar(residual_oil_saturation_water)
        and np.isscalar(residual_oil_saturation_gas)
        and np.isscalar(residual_gas_saturation)
    )

    sw, so, sg = np.broadcast_arrays(sw, so, sg)
    if np.any((sw < 0) | (sw > 1) | (so < 0) | (so > 1) | (sg < 0) | (sg > 1)):
        raise ValidationError(
            f"Saturations must be between 0 and 1. Sw: {show_invalid_saturation(sw)}, So: {show_invalid_saturation(so)}, Sg: {show_invalid_saturation(sg)}"
        )

    # Normalize saturations if they do not sum to 1
    total_saturation = sw + so + sg
    needs_norm = (np.abs(total_saturation - 1.0) > saturation_epsilon) & (total_saturation > 0.0)
    if np.any(needs_norm):
        sw = np.where(needs_norm, sw / total_saturation, sw)
        so = np.where(needs_norm, so / total_saturation, so)
        sg = np.where(needs_norm, sg / total_saturation, sg)

    swc = irreducible_water_saturation
    sorw = residual_oil_saturation_water
    sorg = residual_oil_saturation_gas
    sgr = residual_gas_saturation

    if wettability == Wettability.WATER_WET:
        # Water kr (wetting phase)
        movable_water_range = 1.0 - swc - sorw  # type: ignore[operator]
        sw_star = np.where(
            movable_water_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - swc) / movable_water_range, 0.0, 1.0),
        )
        krw = maximum_water_relperm * _let_relperm(sw_star, water_L, water_E, water_T)

        # Gas kr (non-wetting phase)
        movable_gas_range = 1.0 - swc - sgr - sorg  # type: ignore[operator]
        sg_star = np.where(
            movable_gas_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - sgr) / movable_gas_range, 0.0, 1.0),
        )
        krg = maximum_gas_relperm * _let_relperm(sg_star, gas_L, gas_E, gas_T)

        # Oil kr (intermediate phase, three-phase mixing)
        movable_oil_water_range = 1.0 - swc - sorw  # type: ignore[operator]
        so_star_w = np.where(
            movable_oil_water_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((1.0 - sw - sorw) / movable_oil_water_range, 0.0, 1.0),  # type: ignore[operator]
        )
        kro_w = _let_relperm(so_star_w, oil_water_L, oil_water_E, oil_water_T)

        movable_gas_oil_range = 1.0 - swc - sorg - sgr  # type: ignore[operator]
        so_star_g = np.where(
            movable_gas_oil_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((1.0 - swc - sg - sorg) / movable_gas_oil_range, 0.0, 1.0),  # type: ignore[operator]
        )
        kro_g = _let_relperm(so_star_g, gas_oil_L, gas_oil_E, gas_oil_T)

        kro_mixed = mixing_rule(
            kro_w=kro_w,
            kro_g=kro_g,
            krw=krw,
            krg=krg,
            kr_max=1.0,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            irreducible_water_saturation=swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=sorg,  # type: ignore[arg-type]
        )
        kro = maximum_oil_relperm * np.clip(kro_mixed, 0.0, 1.0)

    elif wettability == Wettability.OIL_WET:
        # Oil is wetting, water becomes intermediate
        movable_oil_range = 1.0 - sorw - sorg  # type: ignore[operator]
        max_residual = np.minimum(sorw, sorg)
        so_star = np.where(
            movable_oil_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((so - max_residual) / movable_oil_range, 0.0, 1.0),
        )
        kro = maximum_oil_relperm * _let_relperm(so_star, oil_water_L, oil_water_E, oil_water_T)

        movable_gas_range = 1.0 - sgr - swc  # type: ignore[operator]
        sg_star = np.where(
            movable_gas_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - sgr) / movable_gas_range, 0.0, 1.0),
        )
        krg = maximum_gas_relperm * _let_relperm(sg_star, gas_L, gas_E, gas_T)

        movable_water_range_ow = 1.0 - swc - sorw  # type: ignore[operator]
        sw_star_ow = np.where(
            movable_water_range_ow <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - swc) / movable_water_range_ow, 0.0, 1.0),
        )
        krw_ow = _let_relperm(sw_star_ow, water_L, water_E, water_T)

        movable_water_range_gw = 1.0 - swc - sgr  # type: ignore[operator]
        sw_star_gw = np.where(
            movable_water_range_gw <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - swc) / movable_water_range_gw, 0.0, 1.0),
        )
        krw_gw = _let_relperm(sw_star_gw, water_L, water_E, water_T)

        krw_mixed = mixing_rule(
            kro_w=krw_ow,
            kro_g=krw_gw,
            krw=kro,
            krg=krg,
            kr_max=1.0,
            water_saturation=so,
            oil_saturation=sw,
            gas_saturation=sg,
            irreducible_water_saturation=np.minimum(sorw, sorg),
            residual_oil_saturation_water=swc,  # type: ignore[arg-type]
            residual_oil_saturation_gas=swc,  # type: ignore[arg-type]
        )
        krw = maximum_water_relperm * np.clip(krw_mixed, 0.0, 1.0)

    elif wettability == Wettability.MIXED_WET:
        # Water-wet sub-system
        movable_water_range = 1.0 - swc - sorw  # type: ignore[operator]
        sw_star_ww = np.where(
            movable_water_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - swc) / movable_water_range, 0.0, 1.0),
        )
        krw_ww = maximum_water_relperm * _let_relperm(sw_star_ww, water_L, water_E, water_T)

        movable_gas_range = 1.0 - swc - sgr - sorg  # type: ignore[operator]
        sg_star_ww = np.where(
            movable_gas_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - sgr) / movable_gas_range, 0.0, 1.0),
        )
        krg_ww = maximum_gas_relperm * _let_relperm(sg_star_ww, gas_L, gas_E, gas_T)

        movable_oil_water_range = 1.0 - swc - sorw  # type: ignore[operator]
        so_star_w_ww = np.where(
            movable_oil_water_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((1.0 - sw - sorw) / movable_oil_water_range, 0.0, 1.0),  # type: ignore[operator]
        )
        kro_w_ww = _let_relperm(so_star_w_ww, oil_water_L, oil_water_E, oil_water_T)

        movable_gas_oil_range = 1.0 - swc - sorg - sgr  # type: ignore[operator]
        so_star_g_ww = np.where(
            movable_gas_oil_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((1.0 - swc - sg - sorg) / movable_gas_oil_range, 0.0, 1.0),  # type: ignore[operator]
        )
        kro_g_ww = _let_relperm(so_star_g_ww, gas_oil_L, gas_oil_E, gas_oil_T)

        kro_ww = maximum_oil_relperm * np.clip(
            mixing_rule(
                kro_w=kro_w_ww,
                kro_g=kro_g_ww,
                krw=krw_ww,
                krg=krg_ww,
                kr_max=1.0,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                irreducible_water_saturation=swc,  # type: ignore[arg-type]
                residual_oil_saturation_water=sorw,  # type: ignore[arg-type]
                residual_oil_saturation_gas=sorg,  # type: ignore[arg-type]
            ),
            0.0,
            1.0,
        )

        # Oil-wet sub-system
        movable_oil_range_ow = 1.0 - sorw - sorg  # type: ignore[operator]
        max_residual_ow = np.minimum(sorw, sorg)
        so_star_ow = np.where(
            movable_oil_range_ow <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((so - max_residual_ow) / movable_oil_range_ow, 0.0, 1.0),
        )
        kro_ow = maximum_oil_relperm * _let_relperm(
            so_star_ow, oil_water_L, oil_water_E, oil_water_T
        )

        movable_gas_range_ow = 1.0 - sgr - swc  # type: ignore[operator]
        sg_star_ow = np.where(
            movable_gas_range_ow <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - sgr) / movable_gas_range_ow, 0.0, 1.0),
        )
        krg_ow = maximum_gas_relperm * _let_relperm(sg_star_ow, gas_L, gas_E, gas_T)

        movable_water_range_ow = 1.0 - swc - sorw  # type: ignore[operator]
        sw_star_ow = np.where(
            movable_water_range_ow <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - swc) / movable_water_range_ow, 0.0, 1.0),
        )
        krw_ow_proxy = _let_relperm(sw_star_ow, water_L, water_E, water_T)

        movable_water_range_gw = 1.0 - swc - sgr  # type: ignore[operator]
        sw_star_gw = np.where(
            movable_water_range_gw <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - swc) / movable_water_range_gw, 0.0, 1.0),
        )
        krw_gw_proxy = _let_relperm(sw_star_gw, water_L, water_E, water_T)

        krw_ow = maximum_water_relperm * np.clip(
            mixing_rule(
                kro_w=krw_ow_proxy,
                kro_g=krw_gw_proxy,
                krw=kro_ow,
                krg=krg_ow,
                kr_max=1.0,
                water_saturation=so,
                oil_saturation=sw,
                gas_saturation=sg,
                irreducible_water_saturation=np.minimum(sorw, sorg),
                residual_oil_saturation_water=swc,  # type: ignore[arg-type]
                residual_oil_saturation_gas=swc,  # type: ignore[arg-type]
            ),
            0.0,
            1.0,
        )

        # Blend
        krw = mixed_wet_water_fraction * krw_ww + (1.0 - mixed_wet_water_fraction) * krw_ow
        kro = mixed_wet_water_fraction * kro_ww + (1.0 - mixed_wet_water_fraction) * kro_ow
        krg = mixed_wet_water_fraction * krg_ww + (1.0 - mixed_wet_water_fraction) * krg_ow

    else:
        raise ValidationError(f"Wettability {wettability!r} not implemented.")

    krw = np.clip(krw, 0.0, 1.0)
    kro = np.clip(kro, 0.0, 1.0)
    krg = np.clip(krg, 0.0, 1.0)
    krw = clamp_relperm(krw, minimum_water_relperm)  # type: ignore[arg-type]
    kro = clamp_relperm(kro, minimum_oil_relperm)  # type: ignore[arg-type]
    krg = clamp_relperm(krg, minimum_gas_relperm)  # type: ignore[arg-type]
    if is_scalar:
        krw = krw.item()  # type: ignore
        kro = kro.item()  # type: ignore
        krg = krg.item()  # type: ignore
    return krw, kro, krg  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_let_curve_slope_wrt_normalized_saturation(
    normalized_saturation: NumberArray[NDimension],
    L: Number,
    E: Number,
    T: Number,
    kr_max: Number,
) -> NumberArray[NDimension]:
    """
    Analytical derivative of the LET relative permeability curve with
    respect to normalized (effective) saturation.

    The LET curve is:

    ``
    kr = kr_max * S*^L / (S*^L + E * (1-S*)^T)
    ``

    Applying the quotient rule and simplifying:

    ``
    dkr / dS* = kr_max * E * S*^(L-1) * (1-S*)^(T-1)
                * [L*(1-S*) + T*S*] / (S*^L + E*(1-S*)^T)^2
    ``

    The result is zero when the normalized saturation is exactly 0 or 1
    (boundary conditions).

    :param normalized_saturation: Effective saturation in [0, 1], clamped
        internally away from 0 and 1 to avoid power-law singularities.
    :param L: LET low-end curvature parameter (positive).
    :param E: LET elevation parameter (positive).
    :param T: LET high-end curvature parameter (positive).
    :param kr_max: Endpoint relative permeability.
    :return: Derivative array with the same shape as `normalized_saturation`.
    """
    s = np.clip(normalized_saturation, 1e-15, 1.0 - 1e-15)
    denominator = s**L + E * (1.0 - s) ** T
    safe_denominator = np.where(denominator > 1e-30, denominator, 1e-30)
    numerator = E * (s ** (L - 1.0)) * ((1.0 - s) ** (T - 1.0)) * (L * (1.0 - s) + T * s)
    slope = kr_max * numerator / (safe_denominator**2)
    slope = np.where(normalized_saturation <= 0.0, 0.0, slope)
    slope = np.where(normalized_saturation >= 1.0, 0.0, slope)
    return slope  # type: ignore[return-value]


def compute_let_phase_curve(
    *,
    saturation: NumberOrArray[NDimension],
    lower_saturation: NumberOrArray[NDimension],
    mobile_range: NumberOrArray[NDimension],
    parameters: LETParameters,
    maximum_relperm: Number,
    minimum_mobile_pore_space: Number,
    zeros: NumberArray[NDimension],
) -> tuple[NumberArray[NDimension], NumberArray[NDimension]]:
    """
    LET relative permeability of one phase and its slope with respect to the
    saturation of that phase.

    The relative permeability is `maximum_relperm` times the LET curve of the
    saturation scaled between `lower_saturation` and the top of the mobile range.
    The slope is zero where the mobile range is smaller than
    `minimum_mobile_pore_space`.

    :param saturation: Saturation of the phase.
    :param lower_saturation: Saturation below which the phase is immobile.
    :param mobile_range: Width of the saturation interval over which the phase is mobile.
    :param parameters: LET shape parameters of the phase.
    :param maximum_relperm: Endpoint relative permeability of the phase.
    :param minimum_mobile_pore_space: Smallest mobile range treated as valid.
    :param zeros: Array of zeros with the shape of the saturations.
    :return: Tuple of the relative permeability and its slope, both before the minimum
        relative permeability is applied.
    """
    valid = mobile_range > minimum_mobile_pore_space
    effective_saturation = np.clip(
        (saturation - lower_saturation) / np.where(valid, mobile_range, 1.0),
        0.0,
        1.0,
    )
    relperm = maximum_relperm * _let_relperm(
        normalized_saturation=effective_saturation,
        L=parameters.L,
        E=parameters.E,
        T=parameters.T,
    )
    slope = np.where(
        valid,
        compute_let_curve_slope_wrt_normalized_saturation(
            normalized_saturation=effective_saturation,
            L=parameters.L,
            E=parameters.E,
            T=parameters.T,
            kr_max=maximum_relperm,
        )
        / mobile_range,
        zeros,
    )
    return relperm, slope  # type: ignore[return-value]


def compute_let_declining_curve(
    *,
    remaining_saturation: NumberOrArray[NDimension],
    mobile_range: NumberOrArray[NDimension],
    parameters: LETParameters,
    minimum_mobile_pore_space: Number,
    zeros: NumberArray[NDimension],
) -> tuple[NumberArray[NDimension], NumberArray[NDimension]]:
    """
    Unit-endpoint LET relative permeability of oil in a two-phase system, and its
    slope with respect to the saturation of the displacing phase.

    Oil relative permeability falls as the displacing phase advances, so the slope is
    negative. It is zero where the mobile range is smaller than
    `minimum_mobile_pore_space` and where the oil saturation lies outside its mobile
    interval.

    :param remaining_saturation: Oil saturation above its residual value, as a function
        of the displacing phase saturation of the two-phase system.
    :param mobile_range: Width of the saturation interval over which oil is mobile.
    :param parameters: LET shape parameters of oil in the two-phase system.
    :param minimum_mobile_pore_space: Smallest mobile range treated as valid.
    :param zeros: Array of zeros with the shape of the saturations.
    :return: Tuple of the oil relative permeability and its slope with respect to the
        displacing phase saturation.
    """
    valid = mobile_range > minimum_mobile_pore_space
    unclipped_effective_saturation = remaining_saturation / np.where(valid, mobile_range, 1.0)
    effective_saturation = np.clip(unclipped_effective_saturation, 0.0, 1.0)
    relperm = _let_relperm(
        normalized_saturation=effective_saturation,
        L=parameters.L,
        E=parameters.E,
        T=parameters.T,
    )
    slope = np.where(
        valid & (unclipped_effective_saturation > 0.0) & (unclipped_effective_saturation < 1.0),
        -compute_let_curve_slope_wrt_normalized_saturation(
            normalized_saturation=effective_saturation,
            L=parameters.L,
            E=parameters.E,
            T=parameters.T,
            kr_max=1.0,
        )
        / mobile_range,
        zeros,
    )
    return relperm, slope  # type: ignore[return-value]


def compute_let_water_wet_system(
    *,
    sw: NumberArray[NDimension],
    so: NumberArray[NDimension],
    sg: NumberArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    water: LETParameters,
    oil_water: LETParameters,
    gas_oil: LETParameters,
    gas: LETParameters,
    maximum_water_relperm: Number,
    maximum_oil_relperm: Number,
    maximum_gas_relperm: Number,
    mixing_rule: MixingRule,
    minimum_mobile_pore_space: Number,
    finite_difference_epsilon: Number,
    zeros: NumberArray[NDimension],
) -> WettabilitySystem:
    """
    LET relative permeabilities and derivatives of a water-wet system.

    Water and gas follow LET curves of their own saturation. Oil follows the
    three-phase mixing rule applied to the two-phase oil curves of the oil-water and
    gas-oil systems, and its derivatives are completed by the chain rule through the
    mixing rule.

    :param sw: Water saturation array.
    :param so: Oil saturation array.
    :param sg: Gas saturation array.
    :param irreducible_water_saturation: Irreducible water saturation.
    :param residual_oil_saturation_water: Residual oil saturation to water flooding.
    :param residual_oil_saturation_gas: Residual oil saturation to gas flooding.
    :param residual_gas_saturation: Residual gas saturation.
    :param water: LET parameters for water.
    :param oil_water: LET parameters for oil in the oil-water system.
    :param gas_oil: LET parameters for oil in the gas-oil system.
    :param gas: LET parameters for gas.
    :param maximum_water_relperm: Endpoint relative permeability of water.
    :param maximum_oil_relperm: Endpoint relative permeability of oil.
    :param maximum_gas_relperm: Endpoint relative permeability of gas.
    :param mixing_rule: Three-phase oil relative permeability mixing rule.
    :param minimum_mobile_pore_space: Smallest mobile range treated as valid.
    :param finite_difference_epsilon: Step used when the mixing rule has no analytical derivative.
    :param zeros: Array of zeros with the shape of the saturations.
    :return: The water-wet `WettabilitySystem`.
    """
    swc = irreducible_water_saturation
    sorw = residual_oil_saturation_water
    sorg = residual_oil_saturation_gas
    sgr = residual_gas_saturation

    krw, dkrw_dsw = compute_let_phase_curve(
        saturation=sw,
        lower_saturation=swc,
        mobile_range=1.0 - swc - sorw,  # type: ignore[operator]
        parameters=water,
        maximum_relperm=maximum_water_relperm,
        minimum_mobile_pore_space=minimum_mobile_pore_space,
        zeros=zeros,
    )
    krg, dkrg_dsg = compute_let_phase_curve(
        saturation=sg,
        lower_saturation=sgr,
        mobile_range=1.0 - swc - sgr - sorg,  # type: ignore[operator]
        parameters=gas,
        maximum_relperm=maximum_gas_relperm,
        minimum_mobile_pore_space=minimum_mobile_pore_space,
        zeros=zeros,
    )
    # Two-phase oil inputs with unit endpoint. Oil in the oil-water system depends on
    # Sw only (So = 1 - Sw - sorw above residual), and oil in the gas-oil system on Sg only.
    kro_w, dkro_w_dsw = compute_let_declining_curve(
        remaining_saturation=1.0 - sw - sorw,  # type: ignore[operator]
        mobile_range=1.0 - swc - sorw,  # type: ignore[operator]
        parameters=oil_water,
        minimum_mobile_pore_space=minimum_mobile_pore_space,
        zeros=zeros,
    )
    kro_g, dkro_g_dsg = compute_let_declining_curve(
        remaining_saturation=1.0 - swc - sg - sorg,  # type: ignore[operator]
        mobile_range=1.0 - swc - sorg - sgr,  # type: ignore[operator]
        parameters=gas_oil,
        minimum_mobile_pore_space=minimum_mobile_pore_space,
        zeros=zeros,
    )

    mixed_oil, partials = apply_water_wet_mixing_rule(
        mixing_rule=mixing_rule,
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg,
        water_saturation=sw,
        oil_saturation=so,
        gas_saturation=sg,
        irreducible_water_saturation=swc,
        residual_oil_saturation_water=sorw,
        residual_oil_saturation_gas=sorg,
        finite_difference_epsilon=finite_difference_epsilon,
    )
    kro = maximum_oil_relperm * mixed_oil
    dkro_dsw = maximum_oil_relperm * (
        partials["dkro_dkro_w"] * dkro_w_dsw
        + partials["dkro_dkrw"] * dkrw_dsw
        + partials["dkro_dsw_explicit"]
    )
    dkro_dso = maximum_oil_relperm * partials["dkro_dso_explicit"]
    dkro_dsg = maximum_oil_relperm * (
        partials["dkro_dkro_g"] * dkro_g_dsg
        + partials["dkro_dkrg"] * dkrg_dsg
        + partials["dkro_dsg_explicit"]
    )
    return WettabilitySystem(
        krw=krw,
        kro=kro,
        krg=krg,
        dkrw_dsw=dkrw_dsw,
        dkrw_dso=zeros.copy(),
        dkrw_dsg=zeros.copy(),
        dkro_dsw=dkro_dsw,
        dkro_dso=dkro_dso,
        dkro_dsg=dkro_dsg,
        dkrg_dsg=dkrg_dsg,
    )


def compute_let_oil_wet_system(
    *,
    sw: NumberArray[NDimension],
    so: NumberArray[NDimension],
    sg: NumberArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    water: LETParameters,
    oil_water: LETParameters,
    gas: LETParameters,
    maximum_water_relperm: Number,
    maximum_oil_relperm: Number,
    maximum_gas_relperm: Number,
    mixing_rule: MixingRule,
    minimum_mobile_pore_space: Number,
    finite_difference_epsilon: Number,
    zeros: NumberArray[NDimension],
) -> WettabilitySystem:
    """
    LET relative permeabilities and derivatives of an oil-wet system.

    Oil is the wetting phase, so oil follows the oil-water LET curve of oil saturation
    and gas a LET curve of gas saturation. Water is the intermediate phase, so it
    follows the three-phase mixing rule applied to the two-phase water curves, and its
    derivatives are completed by the chain rule through the rule.

    :param sw: Water saturation array.
    :param so: Oil saturation array.
    :param sg: Gas saturation array.
    :param irreducible_water_saturation: Irreducible water saturation.
    :param residual_oil_saturation_water: Residual oil saturation to water flooding.
    :param residual_oil_saturation_gas: Residual oil saturation to gas flooding.
    :param residual_gas_saturation: Residual gas saturation.
    :param water: LET parameters for water.
    :param oil_water: LET parameters for oil in the oil-water system.
    :param gas: LET parameters for gas.
    :param maximum_water_relperm: Endpoint relative permeability of water.
    :param maximum_oil_relperm: Endpoint relative permeability of oil.
    :param maximum_gas_relperm: Endpoint relative permeability of gas.
    :param mixing_rule: Three-phase mixing rule.
    :param minimum_mobile_pore_space: Smallest mobile range treated as valid.
    :param finite_difference_epsilon: Step used when the mixing rule has no analytical derivative.
    :param zeros: Array of zeros with the shape of the saturations.
    :return: The oil-wet `WettabilitySystem`.
    """
    swc = irreducible_water_saturation
    sorw = residual_oil_saturation_water
    sorg = residual_oil_saturation_gas
    sgr = residual_gas_saturation

    kro, dkro_dso = compute_let_phase_curve(
        saturation=so,
        lower_saturation=np.minimum(sorw, sorg),
        mobile_range=1.0 - sorw - sorg,  # type: ignore[operator]
        parameters=oil_water,
        maximum_relperm=maximum_oil_relperm,
        minimum_mobile_pore_space=minimum_mobile_pore_space,
        zeros=zeros,
    )
    krg, dkrg_dsg = compute_let_phase_curve(
        saturation=sg,
        lower_saturation=sgr,
        mobile_range=1.0 - sgr - swc,  # type: ignore[operator]
        parameters=gas,
        maximum_relperm=maximum_gas_relperm,
        minimum_mobile_pore_space=minimum_mobile_pore_space,
        zeros=zeros,
    )
    # Two-phase water inputs with unit endpoint. Both depend on Sw only.
    krw_oil_water, dkrw_oil_water_dsw = compute_let_phase_curve(
        saturation=sw,
        lower_saturation=swc,
        mobile_range=1.0 - swc - sorw,  # type: ignore[operator]
        parameters=water,
        maximum_relperm=1.0,
        minimum_mobile_pore_space=minimum_mobile_pore_space,
        zeros=zeros,
    )
    krw_gas_water, dkrw_gas_water_dsw = compute_let_phase_curve(
        saturation=sw,
        lower_saturation=swc,
        mobile_range=1.0 - swc - sgr,  # type: ignore[operator]
        parameters=water,
        maximum_relperm=1.0,
        minimum_mobile_pore_space=minimum_mobile_pore_space,
        zeros=zeros,
    )

    mixed_water, partials = apply_oil_wet_mixing_rule(
        mixing_rule=mixing_rule,
        krw_oil_water=krw_oil_water,
        krw_gas_water=krw_gas_water,
        kro=kro,
        krg=krg,
        water_saturation=sw,
        oil_saturation=so,
        gas_saturation=sg,
        irreducible_water_saturation=swc,
        residual_oil_saturation_water=sorw,
        residual_oil_saturation_gas=sorg,
        finite_difference_epsilon=finite_difference_epsilon,
    )
    krw = maximum_water_relperm * mixed_water
    dkrw_dsw = maximum_water_relperm * (
        partials["dkro_dkro_w"] * dkrw_oil_water_dsw
        + partials["dkro_dkro_g"] * dkrw_gas_water_dsw
        + partials["dkro_dso_explicit"]
    )
    dkrw_dso = maximum_water_relperm * (
        partials["dkro_dkrw"] * dkro_dso + partials["dkro_dsw_explicit"]
    )
    dkrw_dsg = maximum_water_relperm * (
        partials["dkro_dkrg"] * dkrg_dsg + partials["dkro_dsg_explicit"]
    )
    return WettabilitySystem(
        krw=krw,
        kro=kro,
        krg=krg,
        dkrw_dsw=dkrw_dsw,
        dkrw_dso=dkrw_dso,
        dkrw_dsg=dkrw_dsg,
        dkro_dsw=zeros.copy(),
        dkro_dso=dkro_dso,
        dkro_dsg=zeros.copy(),
        dkrg_dsg=dkrg_dsg,
    )


def compute_let_relative_permeability_derivatives(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    water: LETParameters,
    oil_water: LETParameters,
    gas_oil: LETParameters,
    gas: LETParameters,
    minimum_mobile_pore_space: Number,
    finite_difference_epsilon: Number,
    maximum_water_relperm: Number = 1.0,
    maximum_oil_relperm: Number = 1.0,
    maximum_gas_relperm: Number = 1.0,
    wettability: Wettability = Wettability.WATER_WET,
    mixed_wet_water_fraction: Number = 0.5,
    mixing_rule: MixingRule = baker_linear_rule,
    minimum_water_relperm: Number | None = None,
    minimum_oil_relperm: Number | None = None,
    minimum_gas_relperm: Number | None = None,
) -> RelativePermeabilityDerivatives:
    """
    Compute all nine partial derivatives of the three-phase relative permeabilities
    with respect to water, oil and gas saturation using the LET model.

    The two-phase LET derivatives are computed analytically through the effective
    saturation, and the three-phase derivative of the intermediate phase is completed
    by the chain rule through the mixing rule.

    Wherever the minimum relative permeability is active the corresponding derivative
    is smoothly reduced to zero, which keeps the Jacobian consistent with the floored
    relative permeability values.

    Supports scalar and array inputs.

    :param water_saturation: Water saturation (fraction, 0 to 1).
    :param oil_saturation: Oil saturation (fraction, 0 to 1).
    :param gas_saturation: Gas saturation (fraction, 0 to 1).
    :param irreducible_water_saturation: Irreducible water saturation.
    :param residual_oil_saturation_water: Residual oil saturation to water flooding.
    :param residual_oil_saturation_gas: Residual oil saturation to gas flooding.
    :param residual_gas_saturation: Residual gas saturation.
    :param water: LET parameters for water.
    :param oil_water: LET parameters for oil in the oil-water system.
    :param gas_oil: LET parameters for oil in the gas-oil system.
    :param gas: LET parameters for gas.
    :param minimum_mobile_pore_space: Smallest mobile range treated as valid.
    :param finite_difference_epsilon: Step used when the mixing rule has no analytical derivative.
    :param maximum_water_relperm: Endpoint relative permeability of water.
    :param maximum_oil_relperm: Endpoint relative permeability of oil.
    :param maximum_gas_relperm: Endpoint relative permeability of gas.
    :param wettability: Wettability of the rock (water-wet, oil-wet or mixed-wet).
    :param mixed_wet_water_fraction: Fraction of the pore space that is water-wet in a
        mixed-wet rock (0 to 1).
    :param mixing_rule: Three-phase oil relative permeability mixing rule.
    :param minimum_water_relperm: Resolved minimum water relative permeability, or `None`.
    :param minimum_oil_relperm: Resolved minimum oil relative permeability, or `None`.
    :param minimum_gas_relperm: Resolved minimum gas relative permeability, or `None`.
    :return: `RelativePermeabilityDerivatives` dictionary.
    """
    sw, so, sg, zeros, is_scalar = broadcast_saturations(
        water_saturation,
        oil_saturation,
        gas_saturation,
        irreducible_water_saturation,
        residual_oil_saturation_water,
        residual_oil_saturation_gas,
        residual_gas_saturation,
    )

    def water_wet_system() -> WettabilitySystem:
        return compute_let_water_wet_system(
            sw=sw,
            so=so,
            sg=sg,
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
            residual_gas_saturation=residual_gas_saturation,
            water=water,
            oil_water=oil_water,
            gas_oil=gas_oil,
            gas=gas,
            maximum_water_relperm=maximum_water_relperm,
            maximum_oil_relperm=maximum_oil_relperm,
            maximum_gas_relperm=maximum_gas_relperm,
            mixing_rule=mixing_rule,
            minimum_mobile_pore_space=minimum_mobile_pore_space,
            finite_difference_epsilon=finite_difference_epsilon,
            zeros=zeros,
        )

    def oil_wet_system() -> WettabilitySystem:
        return compute_let_oil_wet_system(
            sw=sw,
            so=so,
            sg=sg,
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
            residual_gas_saturation=residual_gas_saturation,
            water=water,
            oil_water=oil_water,
            gas=gas,
            maximum_water_relperm=maximum_water_relperm,
            maximum_oil_relperm=maximum_oil_relperm,
            maximum_gas_relperm=maximum_gas_relperm,
            mixing_rule=mixing_rule,
            minimum_mobile_pore_space=minimum_mobile_pore_space,
            finite_difference_epsilon=finite_difference_epsilon,
            zeros=zeros,
        )

    if wettability == Wettability.OIL_WET:
        return finalize_oil_wet_derivatives(
            system=oil_wet_system(),
            minimum_water_relperm=minimum_water_relperm,
            minimum_oil_relperm=minimum_oil_relperm,
            minimum_gas_relperm=minimum_gas_relperm,
            zeros=zeros,
            is_scalar=is_scalar,
        )
    if wettability == Wettability.MIXED_WET:
        return finalize_mixed_wet_derivatives(
            water_wet=water_wet_system(),
            oil_wet=oil_wet_system(),
            water_wet_fraction=mixed_wet_water_fraction,
            minimum_water_relperm=minimum_water_relperm,
            minimum_oil_relperm=minimum_oil_relperm,
            minimum_gas_relperm=minimum_gas_relperm,
            zeros=zeros,
            is_scalar=is_scalar,
        )
    return finalize_water_wet_derivatives(
        system=water_wet_system(),
        minimum_water_relperm=minimum_water_relperm,
        minimum_oil_relperm=minimum_oil_relperm,
        minimum_gas_relperm=minimum_gas_relperm,
        zeros=zeros,
        is_scalar=is_scalar,
    )


@attrs.frozen(slots=True)
class LETThreePhaseRelPermTable(RelativePermeabilityTable):
    """
    Implements the LET (Lomeland-Ebeltoft-Thomas) three-phase relative permeability model.

    Uses the LET correlation for two-phase relative permeability curves and a
    configurable mixing rule for three-phase oil relative permeability. The LET
    model provides more curve-fitting flexibility than Brooks-Corey by using
    three shape parameters (L, E, T) per phase-pair instead of a single Corey
    exponent.

    Each phase-pair is described by a `LETParameters` instance that groups the
    L, E, and T values:

    - `water`: Parameters for the water kr curve (wetting phase in water-wet).
    - `oil_water`: Parameters for oil kr in the oil-water two-phase system.
    - `gas_oil`: Parameters for oil kr in the gas-oil two-phase system.
    - `gas`: Parameters for the gas kr curve (non-wetting phase).

    Supports water-wet and oil-wet wettability assumptions. Supports both
    scalar and array inputs for saturations (`supports_vector=True`).

    **Minimum relperm min_values** (`minimum_water_relperm`, `minimum_oil_relperm`,
    `minimum_gas_relperm`): same semantics as `BrooksCoreyRelPermTable`.
    """

    irreducible_water_saturation: Number | None = None
    """(Default) Irreducible water saturation (swc)."""

    residual_oil_saturation_water: Number | None = None
    """(Default) Residual oil saturation after water flood (sorw)."""

    residual_oil_saturation_gas: Number | None = None
    """(Default) Residual oil saturation after gas flood (sorg)."""

    residual_gas_saturation: Number | None = None
    """(Default) Residual gas saturation (sgr)."""

    water: LETParameters = LETParameters()
    """LET parameters for the water relative permeability curve."""

    oil_water: LETParameters = LETParameters()
    """LET parameters for oil relative permeability in the water-oil system."""

    gas_oil: LETParameters = LETParameters()
    """LET parameters for oil relative permeability in the gas-oil system."""

    gas: LETParameters = LETParameters()
    """LET parameters for the gas relative permeability curve."""

    maximum_water_relperm: Number = 1.0
    """Endpoint (maximum) relative permeability for water."""

    maximum_oil_relperm: Number = 1.0
    """Endpoint (maximum) relative permeability for oil."""

    maximum_gas_relperm: Number = 1.0
    """Endpoint (maximum) relative permeability for gas."""

    wettability: Wettability = Wettability.WATER_WET
    """Wettability type (water-wet or oil-wet)."""

    mixed_wet_water_fraction: Number = 0.5
    """Fraction of pore space that is water-wet in mixed-wet systems (0-1)."""

    mixing_rule: MixingRule | str = baker_linear_rule
    """
    Mixing rule function or name to compute oil relative permeability in
    three-phase system. Accepts a function or a registered name string.
    """

    minimum_water_relperm: MinimumRelPerm = "auto"
    """
    Minimum `min_value` for the water relative permeability.

    `"auto"` - dtype-aware `min_value`; `None` - no `min_value`; `Number` - explicit value.
    """

    minimum_oil_relperm: MinimumRelPerm = "auto"
    """
    Minimum `min_value` for the oil relative permeability.

    `"auto"` - dtype-aware `min_value`; `None` - no `min_value`; `Number` - explicit value.
    """

    minimum_gas_relperm: MinimumRelPerm = "auto"
    """
    Minimum `min_value` for the gas relative permeability.

    `"auto"` - dtype-aware `min_value`; `None` - no `min_value`; `Number` - explicit value.
    """

    supports_vector: bool = attrs.field(init=False, repr=False, default=True)
    """Flag indicating support for array/vector inputs."""

    def __attrs_post_init__(self) -> None:
        mixing_rule = self.mixing_rule
        if isinstance(mixing_rule, str):
            object.__setattr__(self, "mixing_rule", get_mixing_rule(mixing_rule))

        resolve_min_relperm(self.minimum_water_relperm)
        resolve_min_relperm(self.minimum_oil_relperm)
        resolve_min_relperm(self.minimum_gas_relperm)

    def get_oil_water_wetting_phase(self) -> FluidPhase:
        wettability = self.wettability
        if wettability == Wettability.WATER_WET:
            return FluidPhase.WATER
        elif wettability == Wettability.OIL_WET:
            return FluidPhase.OIL
        elif self.mixed_wet_water_fraction >= 0.5:
            return FluidPhase.WATER
        return FluidPhase.OIL

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        return FluidPhase.OIL

    def get_oil_relperm_endpoint(self) -> Number:
        return self.maximum_oil_relperm

    def get_water_relperm_endpoint(self) -> Number:
        return self.maximum_water_relperm

    def get_gas_relperm_endpoint(self) -> Number:
        return self.maximum_gas_relperm

    def get_connate_water_saturation(self) -> Number:
        """
        This model doesn't distinguish connate from irreducible water
        saturation (a single `irreducible_water_saturation` parameter
        covers both).
        """
        return self.irreducible_water_saturation or 0.0

    def get_residual_oil_saturation_water(self) -> Number:
        return self.residual_oil_saturation_water or 0.0

    def get_residual_oil_saturation_gas(self) -> Number:
        return self.residual_oil_saturation_gas or 0.0

    def get_residual_gas_saturation(self) -> Number:
        return self.residual_gas_saturation or 0.0

    def evaluate(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        irreducible_water_saturation: NumberOrArray[NDimension] | None = None,
        residual_oil_saturation_water: NumberOrArray[NDimension] | None = None,
        residual_oil_saturation_gas: NumberOrArray[NDimension] | None = None,
        residual_gas_saturation: NumberOrArray[NDimension] | None = None,
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for water, oil, and gas using the
        LET correlation.

        Supports both scalar and array inputs for saturations.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param irreducible_water_saturation: Optional override for swc.
        :param residual_oil_saturation_water: Optional override for sorw.
        :param residual_oil_saturation_gas: Optional override for sorg.
        :param residual_gas_saturation: Optional override for sgr.
        :return: `RelativePermeabilities` dictionary.
        """
        sorw = (
            residual_oil_saturation_water
            if residual_oil_saturation_water is not None
            else self.residual_oil_saturation_water
        )
        sorg = (
            residual_oil_saturation_gas
            if residual_oil_saturation_gas is not None
            else self.residual_oil_saturation_gas
        )
        sgr = (
            residual_gas_saturation
            if residual_gas_saturation is not None
            else self.residual_gas_saturation
        )
        swc = (
            irreducible_water_saturation
            if irreducible_water_saturation is not None
            else self.irreducible_water_saturation
        )
        params_missing = []
        if swc is None:
            params_missing.append("swc")
        if sorw is None:
            params_missing.append("sorw")
        if sorg is None:
            params_missing.append("sorg")
        if sgr is None:
            params_missing.append("sgr")
        if params_missing:
            raise ValidationError(
                f"Residual saturations must be provided either as arguments or set in the model instance. "
                f"Missing: {', '.join(params_missing)}"
            )

        krw, kro, krg = compute_let_relative_permeabilities(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=sorg,  # type: ignore[arg-type]
            residual_gas_saturation=sgr,  # type: ignore[arg-type]
            water_L=self.water.L,
            water_E=self.water.E,
            water_T=self.water.T,
            oil_water_L=self.oil_water.L,
            oil_water_E=self.oil_water.E,
            oil_water_T=self.oil_water.T,
            gas_oil_L=self.gas_oil.L,
            gas_oil_E=self.gas_oil.E,
            gas_oil_T=self.gas_oil.T,
            gas_L=self.gas.L,
            gas_E=self.gas.E,
            gas_T=self.gas.T,
            maximum_water_relperm=self.get_water_relperm_endpoint(),
            maximum_oil_relperm=self.get_oil_relperm_endpoint(),
            maximum_gas_relperm=self.get_gas_relperm_endpoint(),
            wettability=self.wettability,
            mixed_wet_water_fraction=self.mixed_wet_water_fraction,
            mixing_rule=typing.cast(MixingRule, self.mixing_rule),
            saturation_epsilon=c.SATURATION_EPSILON,
            minimum_mobile_pore_space=c.MINIMUM_MOBILE_PORE_SPACE,
            minimum_water_relperm=resolve_min_relperm(self.minimum_water_relperm),
            minimum_oil_relperm=resolve_min_relperm(self.minimum_oil_relperm),
            minimum_gas_relperm=resolve_min_relperm(self.minimum_gas_relperm),
        )
        return RelativePermeabilities(water=krw, oil=kro, gas=krg)  # type: ignore[typeddict-item]

    def derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        irreducible_water_saturation: NumberOrArray[NDimension] | None = None,
        residual_oil_saturation_water: NumberOrArray[NDimension] | None = None,
        residual_oil_saturation_gas: NumberOrArray[NDimension] | None = None,
        residual_gas_saturation: NumberOrArray[NDimension] | None = None,
        **kwargs: typing.Any,
    ) -> RelativePermeabilityDerivatives:
        """
        Compute all nine partial derivatives of the three-phase relative
        permeabilities with respect to water saturation, oil saturation, and
        gas saturation using the LET model.

        Returns a dictionary containing:

        ``
        (dkrw/dsw, dkrw/dso, dkrw/dsg,
        dkro/dsw, dkro/dso, dkro/dsg,
        dkrg/dsw, dkrg/dso, dkrg/dsg)
        ``

        For the water-wet case all LET curve derivatives are computed
        analytically via the closed-form quotient-rule formula (see
        `compute_let_curve_slope_wrt_normalized_saturation`).
        The chain rule propagates these through the effective saturation
        normalisation to give derivatives with respect to physical saturation.
        The three-phase oil relative permeability derivative is then completed
        by the extended chain rule through the mixing rule (including the
        `dkro/dkrw` and `dkro/dkrg` terms for rules like Stone II that
        use the actual two-phase water/gas kr values).

        Wherever a minimum relperm min_value is active (raw kr ≤ min_value), the
        corresponding derivative is zeroed out, keeping the Jacobian consistent
        with the min_value kr value and preventing MBE.

        :param water_saturation: Water saturation (fraction, 0 to 1).
        :param oil_saturation: Oil saturation (fraction, 0 to 1).
        :param gas_saturation: Gas saturation (fraction, 0 to 1).
        :param irreducible_water_saturation: Optional override for the
            irreducible (connate) water saturation. Uses the model default
            when not provided.
        :param residual_oil_saturation_water: Optional override for the residual
            oil saturation to water flooding. Uses the model default when not
            provided.
        :param residual_oil_saturation_gas: Optional override for the residual
            oil saturation to gas flooding. Uses the model default when not
            provided.
        :param residual_gas_saturation: Optional override for the residual gas
            saturation. Uses the model default when not provided.
        :return: `RelativePermeabilityDerivatives` dictionary.
        """
        swc = (
            irreducible_water_saturation
            if irreducible_water_saturation is not None
            else self.irreducible_water_saturation
        )
        sorw = (
            residual_oil_saturation_water
            if residual_oil_saturation_water is not None
            else self.residual_oil_saturation_water
        )
        sorg = (
            residual_oil_saturation_gas
            if residual_oil_saturation_gas is not None
            else self.residual_oil_saturation_gas
        )
        sgr = (
            residual_gas_saturation
            if residual_gas_saturation is not None
            else self.residual_gas_saturation
        )

        params_missing = []
        if swc is None:
            params_missing.append("swc")
        if sorw is None:
            params_missing.append("sorw")
        if sorg is None:
            params_missing.append("sorg")
        if sgr is None:
            params_missing.append("sgr")
        if params_missing:
            raise ValidationError(
                f"Residual saturations must be provided either as arguments or set in the model instance. "
                f"Missing: {', '.join(params_missing)}"
            )

        return compute_let_relative_permeability_derivatives(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=sorg,  # type: ignore[arg-type]
            residual_gas_saturation=sgr,  # type: ignore[arg-type]
            water=self.water,
            oil_water=self.oil_water,
            gas_oil=self.gas_oil,
            gas=self.gas,
            minimum_mobile_pore_space=c.MINIMUM_MOBILE_PORE_SPACE,
            finite_difference_epsilon=c.FINITE_DIFFERENCE_EPSILON,
            maximum_water_relperm=self.get_water_relperm_endpoint(),
            maximum_oil_relperm=self.get_oil_relperm_endpoint(),
            maximum_gas_relperm=self.get_gas_relperm_endpoint(),
            wettability=self.wettability,
            mixed_wet_water_fraction=self.mixed_wet_water_fraction,
            mixing_rule=typing.cast(MixingRule, self.mixing_rule),  # type: ignore[arg-type]
            minimum_water_relperm=resolve_min_relperm(self.minimum_water_relperm),
            minimum_oil_relperm=resolve_min_relperm(self.minimum_oil_relperm),
            minimum_gas_relperm=resolve_min_relperm(self.minimum_gas_relperm),
        )
