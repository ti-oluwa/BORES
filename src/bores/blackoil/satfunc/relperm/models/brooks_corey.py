"""Brooks-Corey relative permeability model and table."""

import typing

import attrs
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
    "BrooksCoreyRelPermTable",
    "compute_brooks_corey_relative_permeabilities",
    "compute_brooks_corey_relative_permeability_derivatives",
]


def compute_brooks_corey_relative_permeabilities(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    water_exponent: Number,
    oil_exponent: Number,
    gas_exponent: Number,
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
    Computes relative permeability for water, oil, and gas in a three-phase system.
    Supports water-wet and oil-wet wettability assumptions.

    Uses Corey-type models for krw, krg, and Stone I rule for kro.

    Supports both scalar and array inputs for saturations.

    :param water_saturation: Current water saturation (fraction, between 0 and 1) - scalar or array.
    :param oil_saturation: Current oil saturation (fraction, between 0 and 1) - scalar or array.
    :param gas_saturation: Current gas saturation (fraction, between 0 and 1) - scalar or array.
    :param irreducible_water_saturation: Irreducible water saturation (swc).
    :param residual_oil_saturation_water: Residual oil saturation after water flood (sorw).
    :param residual_oil_saturation_gas: Residual oil saturation after gas flood (sorg).
    :param residual_gas_saturation: Residual gas saturation (sgr).
    :param water_exponent: Corey exponent for water relative permeability.
    :param oil_exponent: Corey exponent for oil relative permeability (affects Stone I blending).
    :param gas_exponent: Corey exponent for gas relative permeability.
    :param wettability: Wettability type (water-wet or oil-wet).
    :param mixed_wet_water_fraction: Fraction of pore space considered water-wet in mixed-wet systems (0 to 1).
    :param mixing_rule: Mixing rule function for three-phase oil relative permeability.
    :param saturation_epsilon: Tolerance for checking if saturations sum to 1.
    :param minimum_mobile_pore_space: Minimum mobile pore space to avoid division by zero in effective saturation calculations.
    :param minimum_water_relperm: Resolved minimum min_value for water kr (`None` = no min_value).
    :param minimum_oil_relperm: Resolved minimum min_value for oil kr (`None` = no min_value).
    :param minimum_gas_relperm: Resolved minimum min_value for gas kr (`None` = no min_value).
    :return: Tuple of (water_relative_permeability, oil_relative_permeability, gas_relative_permeability)
    """
    # Convert to arrays for vectorized operations
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

    # Broadcast all arrays to same shape
    sw, so, sg = np.broadcast_arrays(sw, so, sg)

    # Validate saturations
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

    if wettability == Wettability.WATER_WET:
        # 1. Water relperm (wetting phase)
        movable_water_range = (
            1.0 - irreducible_water_saturation - residual_oil_saturation_water  # type: ignore[operator]
        )
        effective_water_saturation = np.where(
            movable_water_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - irreducible_water_saturation) / movable_water_range, 0.0, 1.0),
        )
        krw = maximum_water_relperm * effective_water_saturation**water_exponent

        # 2. Gas relperm (nonwetting)
        movable_gas_range = (  # type: ignore[operator]
            1.0
            - irreducible_water_saturation  # type: ignore[operator]
            - residual_gas_saturation  # type: ignore[operator]
            - residual_oil_saturation_gas
        )
        effective_gas_saturation = np.where(
            movable_gas_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - residual_gas_saturation) / movable_gas_range, 0.0, 1.0),
        )
        krg = maximum_gas_relperm * effective_gas_saturation**gas_exponent

        # 3. Oil relperm (intermediate phase) - mixing rule blending
        # Make sure to apply oil curvature to the two-phase oil kr inputs before mixing,
        # not to the mixed output. (1-krw/krw_max) and (1-krg/krg_max) are the two-phase
        # oil kr approximations; so we shape them with `oil_exponent` before blending.
        kro_w_shaped = (1.0 - krw / maximum_water_relperm) ** oil_exponent
        kro_g_shaped = (1.0 - krg / maximum_gas_relperm) ** oil_exponent
        kro = maximum_oil_relperm * np.clip(
            mixing_rule(
                kro_w=kro_w_shaped,
                kro_g=kro_g_shaped,
                krw=krw,
                krg=krg,
                kr_max=1.0,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                irreducible_water_saturation=irreducible_water_saturation,  # type: ignore[arg-type]
                residual_oil_saturation_water=residual_oil_saturation_water,  # type: ignore[arg-type]
                residual_oil_saturation_gas=residual_oil_saturation_gas,  # type: ignore[arg-type]
            ),
            0.0,
            1.0,
        )

    elif wettability == Wettability.OIL_WET:
        # Oil is wetting, water becomes intermediate
        # 1. Oil relperm (wetting phase)
        movable_oil_range = (
            1.0 - residual_oil_saturation_water - residual_oil_saturation_gas  # type: ignore[operator]
        )
        max_residual = np.minimum(residual_oil_saturation_water, residual_oil_saturation_gas)
        effective_oil_saturation = np.where(
            movable_oil_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((so - max_residual) / movable_oil_range, 0.0, 1.0),
        )
        kro = maximum_oil_relperm * effective_oil_saturation**oil_exponent

        # 2. Gas relperm (nonwetting phase)
        movable_gas_range = 1.0 - residual_gas_saturation - irreducible_water_saturation  # type: ignore[operator]
        effective_gas_saturation = np.where(
            movable_gas_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - residual_gas_saturation) / movable_gas_range, 0.0, 1.0),
        )
        krg = maximum_gas_relperm * effective_gas_saturation**gas_exponent

        # 3. Water relperm (intermediate phase, use mixing rule style blending)
        kro_proxy_shaped = (1.0 - kro / maximum_oil_relperm) ** water_exponent
        krg_proxy_shaped = (1.0 - krg / maximum_gas_relperm) ** water_exponent
        krw = maximum_water_relperm * np.clip(
            mixing_rule(  # type: ignore[assignment]
                kro_w=kro_proxy_shaped,  # treat oil as wetting
                kro_g=krg_proxy_shaped,  # treat gas as nonwetting
                krw=kro,
                krg=krg,
                kr_max=1.0,
                water_saturation=so,
                oil_saturation=sw,
                gas_saturation=sg,
                irreducible_water_saturation=np.minimum(
                    residual_oil_saturation_water, residual_oil_saturation_gas
                ),
                residual_oil_saturation_water=irreducible_water_saturation,  # type: ignore[arg-type]
                residual_oil_saturation_gas=irreducible_water_saturation,  # type: ignore[arg-type]
            ),
            0.0,
            1.0,
        )

    elif wettability == Wettability.MIXED_WET:
        # Mixed-wet: interpolate between water-wet and oil-wet using mixed_wet_water_fraction.
        # Water-wet contribution
        movable_water_range_ww = (
            1.0 - irreducible_water_saturation - residual_oil_saturation_water  # type: ignore[operator]
        )
        effective_water_saturation_ww = np.where(
            movable_water_range_ww <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - irreducible_water_saturation) / movable_water_range_ww, 0.0, 1.0),
        )
        krw_ww = maximum_water_relperm * effective_water_saturation_ww**water_exponent

        movable_gas_range_ww = (  # type: ignore[operator]
            1.0
            - irreducible_water_saturation  # type: ignore[operator]
            - residual_gas_saturation  # type: ignore[operator]
            - residual_oil_saturation_gas
        )
        effective_gas_saturation_ww = np.where(
            movable_gas_range_ww <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - residual_gas_saturation) / movable_gas_range_ww, 0.0, 1.0),
        )
        krg_ww = maximum_gas_relperm * effective_gas_saturation_ww**gas_exponent
        kro_w_ww = (1.0 - krw_ww / maximum_water_relperm) ** oil_exponent
        kro_g_ww = (1.0 - krg_ww / maximum_gas_relperm) ** oil_exponent
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
                irreducible_water_saturation=irreducible_water_saturation,  # type: ignore[arg-type]
                residual_oil_saturation_water=residual_oil_saturation_water,  # type: ignore[arg-type]
                residual_oil_saturation_gas=residual_oil_saturation_gas,  # type: ignore[arg-type]
            ),
            0.0,
            1.0,
        )

        # Oil-wet contribution
        movable_oil_range_ow = (
            1.0 - residual_oil_saturation_water - residual_oil_saturation_gas  # type: ignore[operator]
        )
        max_residual_ow = np.minimum(residual_oil_saturation_water, residual_oil_saturation_gas)
        effective_oil_saturation_ow = np.where(
            movable_oil_range_ow <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((so - max_residual_ow) / movable_oil_range_ow, 0.0, 1.0),
        )
        kro_ow = maximum_oil_relperm * effective_oil_saturation_ow**oil_exponent

        movable_gas_range_ow = (
            1.0 - residual_gas_saturation - irreducible_water_saturation  # type: ignore[operator]
        )
        effective_gas_saturation_ow = np.where(
            movable_gas_range_ow <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - residual_gas_saturation) / movable_gas_range_ow, 0.0, 1.0),
        )
        krg_ow = maximum_gas_relperm * effective_gas_saturation_ow**gas_exponent
        kro_proxy_ow = (1.0 - kro_ow / maximum_oil_relperm) ** water_exponent
        krg_proxy_ow = (1.0 - krg_ow / maximum_gas_relperm) ** water_exponent
        krw_ow = maximum_water_relperm * np.clip(
            mixing_rule(
                kro_w=kro_proxy_ow,
                kro_g=krg_proxy_ow,
                krw=kro_ow,
                krg=krg_ow,
                kr_max=1.0,
                water_saturation=so,
                oil_saturation=sw,
                gas_saturation=sg,
                irreducible_water_saturation=np.minimum(
                    residual_oil_saturation_water, residual_oil_saturation_gas
                ),
                residual_oil_saturation_water=irreducible_water_saturation,  # type: ignore[arg-type]
                residual_oil_saturation_gas=irreducible_water_saturation,  # type: ignore[arg-type]
            ),
            0.0,
            1.0,
        )

        # Weighted blend
        krw = mixed_wet_water_fraction * krw_ww + (1.0 - mixed_wet_water_fraction) * krw_ow
        kro = mixed_wet_water_fraction * kro_ww + (1.0 - mixed_wet_water_fraction) * kro_ow
        krg = mixed_wet_water_fraction * krg_ww + (1.0 - mixed_wet_water_fraction) * krg_ow

    else:
        raise ValidationError(f"Wettability {wettability!r} not implemented.")

    # Clip all results to [0, 1] and apply per-phase minimum
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


def compute_brooks_corey_phase_curve(
    *,
    saturation: NumberOrArray[NDimension],
    lower_saturation: NumberOrArray[NDimension],
    mobile_range: NumberOrArray[NDimension],
    maximum_relperm: Number,
    exponent: Number,
    minimum_mobile_pore_space: Number,
    zeros: NumberArray[NDimension],
) -> tuple[NumberArray[NDimension], NumberArray[NDimension]]:
    """
    Corey power-law relative permeability of one phase and its slope with respect to
    the saturation of that phase.

    The relative permeability is `maximum_relperm * Se ** exponent`, where `Se` is the
    saturation scaled between `lower_saturation` and the top of the mobile range. The
    slope is zero where the mobile range is smaller than `minimum_mobile_pore_space`
    and outside the interval where the phase is mobile.

    :param saturation: Saturation of the phase.
    :param lower_saturation: Saturation below which the phase is immobile.
    :param mobile_range: Width of the saturation interval over which the phase is mobile.
    :param maximum_relperm: Endpoint relative permeability of the phase.
    :param exponent: Corey exponent of the phase.
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
    relperm = maximum_relperm * effective_saturation**exponent
    slope = maximum_relperm * np.where(
        valid & (effective_saturation > 0.0),
        exponent * (effective_saturation ** max(exponent - 1.0, 0.0)) / mobile_range,
        zeros,
    )
    return relperm, slope  # type: ignore[return-value]


def compute_brooks_corey_complement_curve(
    *,
    relperm: NumberOrArray[NDimension],
    maximum_relperm: Number,
    relperm_slope: NumberOrArray[NDimension],
    exponent: Number,
    zeros: NumberArray[NDimension],
) -> tuple[NumberArray[NDimension], NumberArray[NDimension]]:
    """
    Corey-shaped complement `(1 - kr / kr_max) ** exponent` of a phase relative
    permeability, and its slope with respect to the saturation that drives `kr`.

    This is the two-phase relative permeability of the displaced phase that the
    three-phase mixing rule takes as input.

    :param relperm: Relative permeability of the displacing phase.
    :param maximum_relperm: Endpoint relative permeability of the displacing phase.
    :param relperm_slope: Slope of `relperm` with respect to its own saturation.
    :param exponent: Corey exponent used for the complement.
    :param zeros: Array of zeros with the shape of the saturations.
    :return: Tuple of the complement and its slope with respect to the saturation that
        drives `relperm`.
    """
    one_minus_relperm = np.clip(1.0 - relperm / maximum_relperm, 0.0, None)
    complement = one_minus_relperm**exponent
    slope = np.where(
        one_minus_relperm > 0.0,
        exponent
        * (one_minus_relperm ** max(exponent - 1.0, 0.0))
        * (-relperm_slope / maximum_relperm),
        zeros,
    )
    return complement, slope  # type: ignore[return-value]


def compute_brooks_corey_water_wet_system(
    *,
    sw: NumberArray[NDimension],
    so: NumberArray[NDimension],
    sg: NumberArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    water_exponent: Number,
    oil_exponent: Number,
    gas_exponent: Number,
    maximum_water_relperm: Number,
    maximum_oil_relperm: Number,
    maximum_gas_relperm: Number,
    mixing_rule: MixingRule,
    minimum_mobile_pore_space: Number,
    finite_difference_epsilon: Number,
    zeros: NumberArray[NDimension],
) -> WettabilitySystem:
    """
    Brooks-Corey relative permeabilities and derivatives of a water-wet system.

    Water and gas follow Corey power laws of their own saturation. Oil follows the
    three-phase mixing rule applied to Corey-shaped two-phase oil curves, and its
    derivatives are completed by the chain rule through the mixing rule, including
    the terms for rules such as Stone II that depend on the water and gas relative
    permeabilities.

    :param sw: Water saturation array.
    :param so: Oil saturation array.
    :param sg: Gas saturation array.
    :param irreducible_water_saturation: Irreducible water saturation.
    :param residual_oil_saturation_water: Residual oil saturation to water flooding.
    :param residual_oil_saturation_gas: Residual oil saturation to gas flooding.
    :param residual_gas_saturation: Residual gas saturation.
    :param water_exponent: Corey exponent for water.
    :param oil_exponent: Corey exponent for oil.
    :param gas_exponent: Corey exponent for gas.
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

    krw, dkrw_dsw = compute_brooks_corey_phase_curve(
        saturation=sw,
        lower_saturation=swc,
        mobile_range=1.0 - swc - sorw,  # type: ignore[operator]
        maximum_relperm=maximum_water_relperm,
        exponent=water_exponent,
        minimum_mobile_pore_space=minimum_mobile_pore_space,
        zeros=zeros,
    )
    krg, dkrg_dsg = compute_brooks_corey_phase_curve(
        saturation=sg,
        lower_saturation=sgr,
        mobile_range=1.0 - swc - sgr - sorg,  # type: ignore[operator]
        maximum_relperm=maximum_gas_relperm,
        exponent=gas_exponent,
        minimum_mobile_pore_space=minimum_mobile_pore_space,
        zeros=zeros,
    )
    kro_w, dkro_w_dsw = compute_brooks_corey_complement_curve(
        relperm=krw,
        maximum_relperm=maximum_water_relperm,
        relperm_slope=dkrw_dsw,
        exponent=oil_exponent,
        zeros=zeros,
    )
    kro_g, dkro_g_dsg = compute_brooks_corey_complement_curve(
        relperm=krg,
        maximum_relperm=maximum_gas_relperm,
        relperm_slope=dkrg_dsg,
        exponent=oil_exponent,
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


def compute_brooks_corey_oil_wet_system(
    *,
    sw: NumberArray[NDimension],
    so: NumberArray[NDimension],
    sg: NumberArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    water_exponent: Number,
    oil_exponent: Number,
    gas_exponent: Number,
    maximum_water_relperm: Number,
    maximum_oil_relperm: Number,
    maximum_gas_relperm: Number,
    mixing_rule: MixingRule,
    minimum_mobile_pore_space: Number,
    finite_difference_epsilon: Number,
    zeros: NumberArray[NDimension],
) -> WettabilitySystem:
    """
    Brooks-Corey relative permeabilities and derivatives of an oil-wet system.

    Oil is the wetting phase, so oil follows a Corey power law of oil saturation and
    gas a Corey power law of gas saturation. Water is the intermediate phase, so it
    follows the three-phase mixing rule applied to Corey-shaped two-phase water
    curves, and its derivatives are completed by the chain rule through the rule.

    :param sw: Water saturation array.
    :param so: Oil saturation array.
    :param sg: Gas saturation array.
    :param irreducible_water_saturation: Irreducible water saturation.
    :param residual_oil_saturation_water: Residual oil saturation to water flooding.
    :param residual_oil_saturation_gas: Residual oil saturation to gas flooding.
    :param residual_gas_saturation: Residual gas saturation.
    :param water_exponent: Corey exponent for water.
    :param oil_exponent: Corey exponent for oil.
    :param gas_exponent: Corey exponent for gas.
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

    kro, dkro_dso = compute_brooks_corey_phase_curve(
        saturation=so,
        lower_saturation=np.minimum(sorw, sorg),
        mobile_range=1.0 - sorw - sorg,  # type: ignore[operator]
        maximum_relperm=maximum_oil_relperm,
        exponent=oil_exponent,
        minimum_mobile_pore_space=minimum_mobile_pore_space,
        zeros=zeros,
    )
    krg, dkrg_dsg = compute_brooks_corey_phase_curve(
        saturation=sg,
        lower_saturation=sgr,
        mobile_range=1.0 - sgr - swc,  # type: ignore[operator]
        maximum_relperm=maximum_gas_relperm,
        exponent=gas_exponent,
        minimum_mobile_pore_space=minimum_mobile_pore_space,
        zeros=zeros,
    )
    krw_oil_water, dkrw_oil_water_dso = compute_brooks_corey_complement_curve(
        relperm=kro,
        maximum_relperm=maximum_oil_relperm,
        relperm_slope=dkro_dso,
        exponent=water_exponent,
        zeros=zeros,
    )
    krw_gas_water, dkrw_gas_water_dsg = compute_brooks_corey_complement_curve(
        relperm=krg,
        maximum_relperm=maximum_gas_relperm,
        relperm_slope=dkrg_dsg,
        exponent=water_exponent,
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
    dkrw_dsw = maximum_water_relperm * partials["dkro_dso_explicit"]
    dkrw_dso = maximum_water_relperm * (
        partials["dkro_dkro_w"] * dkrw_oil_water_dso
        + partials["dkro_dkrw"] * dkro_dso
        + partials["dkro_dsw_explicit"]
    )
    dkrw_dsg = maximum_water_relperm * (
        partials["dkro_dkro_g"] * dkrw_gas_water_dsg
        + partials["dkro_dkrg"] * dkrg_dsg
        + partials["dkro_dsg_explicit"]
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


def compute_brooks_corey_relative_permeability_derivatives(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    water_exponent: Number,
    oil_exponent: Number,
    gas_exponent: Number,
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
    with respect to water, oil and gas saturation using the Brooks-Corey model.

    The two-phase Corey power-law derivatives are computed analytically through the
    effective saturation, and the three-phase derivative of the intermediate phase is
    completed by the chain rule through the mixing rule.

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
    :param water_exponent: Corey exponent for water.
    :param oil_exponent: Corey exponent for oil.
    :param gas_exponent: Corey exponent for gas.
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

    def evaluate_system(
        compute_system: typing.Callable[..., WettabilitySystem],
    ) -> WettabilitySystem:
        return compute_system(
            sw=sw,
            so=so,
            sg=sg,
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
            residual_gas_saturation=residual_gas_saturation,
            water_exponent=water_exponent,
            oil_exponent=oil_exponent,
            gas_exponent=gas_exponent,
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
            system=evaluate_system(compute_brooks_corey_oil_wet_system),
            minimum_water_relperm=minimum_water_relperm,
            minimum_oil_relperm=minimum_oil_relperm,
            minimum_gas_relperm=minimum_gas_relperm,
            zeros=zeros,
            is_scalar=is_scalar,
        )
    if wettability == Wettability.MIXED_WET:
        return finalize_mixed_wet_derivatives(
            water_wet=evaluate_system(compute_brooks_corey_water_wet_system),
            oil_wet=evaluate_system(compute_brooks_corey_oil_wet_system),
            water_wet_fraction=mixed_wet_water_fraction,
            minimum_water_relperm=minimum_water_relperm,
            minimum_oil_relperm=minimum_oil_relperm,
            minimum_gas_relperm=minimum_gas_relperm,
            zeros=zeros,
            is_scalar=is_scalar,
        )
    return finalize_water_wet_derivatives(
        system=evaluate_system(compute_brooks_corey_water_wet_system),
        minimum_water_relperm=minimum_water_relperm,
        minimum_oil_relperm=minimum_oil_relperm,
        minimum_gas_relperm=minimum_gas_relperm,
        zeros=zeros,
        is_scalar=is_scalar,
    )


@attrs.frozen(slots=True)
class BrooksCoreyRelPermTable(RelativePermeabilityTable):
    """
    Implements the Brooks-Corey-type three-phase relative permeability model.

    Supports water-wet and oil-wet wettability assumptions.

    **Minimum relperm min_values** (`minimum_water_relperm`, `minimum_oil_relperm`, `minimum_gas_relperm`):

    `"auto"` - `max(4 * machine_epsilon, 1e-8)` (dtype-aware, same as
    the CMG IMEX/GEM minimum mobility approach). `None` - no min_value
    (default, kr can reach zero exactly). `Number` - explicit user-set
    value.

    The min_value is applied to the computed kr value and the derivative is
    zeroed out in the min_value region, so the Jacobian is always consistent
    with the kr value (no MBE from mismatched kr/derivative pairs).
    """

    irreducible_water_saturation: Number | None = None
    """(Default) Irreducible water saturation (swc)."""

    residual_oil_saturation_water: Number | None = None
    """(Default) Residual oil saturation after water flood (sorw)."""

    residual_oil_saturation_gas: Number | None = None
    """(Default) Residual oil saturation after gas flood (sorg)."""

    residual_gas_saturation: Number | None = None
    """(Default) Residual gas saturation (sgr)."""

    water_exponent: Number = 2.0
    """
    Corey exponent for water relative permeability.

    Higher values make the curve steeper. Meaning slower krw increase with saturation.
    """

    oil_exponent: Number = 2.0
    """
    Corey exponent for oil relative permeability.

    Higher values make the curve steeper. Meaning slower kro increase with saturation.
    """

    gas_exponent: Number = 2.0
    """
    Corey exponent for gas relative permeability. Higher values make the curve steeper.

    Meaning slower krg increase with saturation.
    """

    maximum_water_relperm: Number = 1.0
    """Endpoint (maximum) relative permeability for water (krw at residual oil)."""

    maximum_oil_relperm: Number = 1.0
    """Endpoint (maximum) relative permeability for oil (krocw)."""

    maximum_gas_relperm: Number = 1.0
    """Endpoint (maximum) relative permeability for gas (krg at connate water + residual oil)."""

    wettability: Wettability = Wettability.WATER_WET
    """Wettability type (water-wet or oil-wet)."""

    mixed_wet_water_fraction: Number = 0.5
    """Fraction of pore space that is water-wet in mixed-wet systems (0-1)."""

    mixing_rule: MixingRule | str = baker_linear_rule
    """
    Mixing rule function or name to compute oil relative permeability in three-phase system.

    The function should take the following parameters in order:
    - kro_w: Oil relative permeability from oil-water table
    - kro_g: Oil relative permeability from oil-gas table
    - krw: Water relative permeability from oil-water table
    - krg: Gas relative permeability from gas-oil table
    - kr_max: Oil relative permeability at connate water
    - Sw: Water saturation
    - So: Oil saturation
    - Sg: Gas saturation
    and return the mixed oil relative permeability.
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
        covers both), matching how the rest of this codebase already
        treats the two as coincident by default.
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
        Compute relative permeabilities for water, oil, and gas.

        Supports both scalar and array inputs for saturations.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param irreducible_water_saturation: Optional override for irreducible water saturation.
        :param residual_oil_saturation_water: Optional override for residual oil saturation after water flood.
        :param residual_oil_saturation_gas: Optional override for residual oil saturation after gas flood.
        :param residual_gas_saturation: Optional override for residual gas saturation.
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

        krw, kro, krg = compute_brooks_corey_relative_permeabilities(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=sorg,  # type: ignore[arg-type]
            residual_gas_saturation=sgr,  # type: ignore[arg-type]
            water_exponent=self.water_exponent,
            oil_exponent=self.oil_exponent,
            gas_exponent=self.gas_exponent,
            maximum_water_relperm=self.get_water_relperm_endpoint(),
            maximum_oil_relperm=self.get_oil_relperm_endpoint(),
            maximum_gas_relperm=self.get_gas_relperm_endpoint(),
            wettability=self.wettability,
            mixed_wet_water_fraction=self.mixed_wet_water_fraction,
            mixing_rule=typing.cast(MixingRule, self.mixing_rule),  # type: ignore[arg-type]
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
        gas saturation using the Brooks-Corey model.

        Returns a dictionary containing:

        ``
        (dkrw/dsw, dkrw/dso, dkrw/dsg,
        dkro/dsw, dkro/dso, dkro/dsg,
        dkrg/dsw, dkrg/dso, dkrg/dsg)
        ``

        For the water-wet case all two-phase Corey power-law derivatives are
        computed analytically via the chain rule through effective saturation.
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

        return compute_brooks_corey_relative_permeability_derivatives(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=sorg,  # type: ignore[arg-type]
            residual_gas_saturation=sgr,  # type: ignore[arg-type]
            water_exponent=self.water_exponent,
            oil_exponent=self.oil_exponent,
            gas_exponent=self.gas_exponent,
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
