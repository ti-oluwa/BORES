"""LET (Lomeland-Ebeltoft-Thomas) relative permeability model and table."""

import typing

import attrs
import numba
import numpy as np

from bores.blackoil.satfunc.relperm.mixing_rules import (
    MixingRule,
    baker_linear_rule,
    get_builtin_rule_id,
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
    validate_and_normalize_saturations,
)
from bores.blackoil.satfunc.relperm.models.kernels import (
    RelPermCell,
    blend_cells,
    clip_unit,
    finalize_mixed_wet_cell,
    finalize_oil_wet_cell,
    finalize_water_wet_cell,
    floor_relperm,
    normalize_cell_saturations,
    prepare_relperm_inputs,
    rule_derivatives,
    rule_value,
    to_kernel_minimum,
)
from bores.blackoil.satfunc.relperm.models.validation import (
    validate_analytical_relperm_parameters,
)
from bores.blackoil.satfunc.relperm.tables import (
    MinimumRelPerm,
    RelativePermeabilityTable,
    clamp_relperm,
    resolve_min_relperm,
    show_invalid_saturation,
)
from bores.blackoil.satfunc.utils import element_at, get_floating_dtype
from bores.constants import c
from bores.errors import ValidationError
from bores.types import (
    FluidPhase,
    Integer,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
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
class LETParameters:
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


@numba.njit(cache=True, inline="always", error_model="numpy")
def compute_let_curve_cell(
    normalized_saturation: Number, L: Number, E: Number, T: Number
) -> Number:
    """LET relative permeability curve `S**L / (S**L + E * (1 - S)**T)` without endpoint scaling."""
    numerator = normalized_saturation**L
    denominator = numerator + E * (1.0 - normalized_saturation) ** T
    if denominator > 0.0:
        return numerator / denominator
    return 0.0


@numba.njit(cache=True, inline="always", error_model="numpy")
def compute_let_curve_slope_cell(
    normalized_saturation: Number, L: Number, E: Number, T: Number, kr_max: Number
) -> Number:
    """
    Derivative of the LET curve scaled by `kr_max` with respect to the normalized saturation.

    The slope is zero at normalized saturations of 0 and 1.
    """
    if normalized_saturation <= 0.0 or normalized_saturation >= 1.0:
        return 0.0
    s = normalized_saturation
    if s < 1e-15:
        s = 1e-15
    elif s > 1.0 - 1e-15:
        s = 1.0 - 1e-15
    denominator = s**L + E * (1.0 - s) ** T
    safe_denominator = denominator if denominator > 1e-30 else 1e-30
    numerator = E * (s ** (L - 1.0)) * ((1.0 - s) ** (T - 1.0)) * (L * (1.0 - s) + T * s)
    return kr_max * numerator / (safe_denominator**2)


@numba.njit(cache=True, inline="always", error_model="numpy")
def compute_let_effective_saturation(
    numerator: Number, mobile_range: Number, minimum_mobile_pore_space: Number
) -> Number:
    """
    Saturation scaled to the range 0 to 1, or zero when the mobile range is smaller than
    `minimum_mobile_pore_space`.
    """
    if mobile_range <= minimum_mobile_pore_space:
        return 0.0
    return clip_unit(numerator / mobile_range)


@numba.njit(cache=True, error_model="numpy")
def compute_let_phase(
    numerator: Number,
    mobile_range: Number,
    parameters: tuple[Number, Number, Number],
    maximum_relperm: Number,
    minimum_mobile_pore_space: Number,
) -> tuple[Number, Number]:
    """
    LET relative permeability of one phase and its slope with respect to the saturation of that
    phase.

    :param numerator: Saturation of the phase above the lower end of its mobile range.
    :param mobile_range: Width of the saturation interval over which the phase is mobile.
    :param parameters: LET parameters `(L, E, T)` of the phase.
    """
    L, E, T = parameters
    effective = compute_let_effective_saturation(
        numerator, mobile_range, minimum_mobile_pore_space
    )
    relperm = maximum_relperm * compute_let_curve_cell(effective, L, E, T)
    if mobile_range > minimum_mobile_pore_space:
        slope = compute_let_curve_slope_cell(effective, L, E, T, maximum_relperm) / mobile_range
    else:
        slope = 0.0
    return relperm, slope


@numba.njit(cache=True, error_model="numpy")
def compute_let_declining(
    remaining_saturation: Number,
    mobile_range: Number,
    parameters: tuple[Number, Number, Number],
    minimum_mobile_pore_space: Number,
) -> tuple[Number, Number]:
    """
    Unit-endpoint LET relative permeability of oil in a two-phase system, and its slope with
    respect to the saturation of the displacing phase (negative, since oil relative
    permeability falls as the displacing phase advances).
    """
    L, E, T = parameters
    effective = compute_let_effective_saturation(
        remaining_saturation, mobile_range, minimum_mobile_pore_space
    )
    relperm = compute_let_curve_cell(effective, L, E, T)
    slope = 0.0
    if mobile_range > minimum_mobile_pore_space:
        unclipped = remaining_saturation / mobile_range
        if unclipped > 0.0 and unclipped < 1.0:
            slope = -compute_let_curve_slope_cell(effective, L, E, T, 1.0) / mobile_range
    return relperm, slope


@numba.njit(cache=True, error_model="numpy")
def compute_let_water_wet_values(
    sw: Number,
    so: Number,
    sg: Number,
    swc: Number,
    sorw: Number,
    sorg: Number,
    sgr: Number,
    water: tuple[Number, Number, Number],
    oil_water: tuple[Number, Number, Number],
    gas_oil: tuple[Number, Number, Number],
    gas: tuple[Number, Number, Number],
    maximum_water_relperm: Number,
    maximum_oil_relperm: Number,
    maximum_gas_relperm: Number,
    rule_id: Integer,
    minimum_mobile_pore_space: Number,
    rule_minimum_mobile_pore_space: Number,
) -> tuple[Number, Number, Number]:
    """LET relative permeabilities of a water-wet cell, before limiting and flooring."""
    wL, wE, wT = water
    gL, gE, gT = gas
    owL, owE, owT = oil_water
    goL, goE, goT = gas_oil
    krw = maximum_water_relperm * compute_let_curve_cell(
        compute_let_effective_saturation(sw - swc, 1.0 - swc - sorw, minimum_mobile_pore_space),
        wL,
        wE,
        wT,
    )
    krg = maximum_gas_relperm * compute_let_curve_cell(
        compute_let_effective_saturation(
            sg - sgr, 1.0 - swc - sgr - sorg, minimum_mobile_pore_space
        ),
        gL,
        gE,
        gT,
    )
    kro_w = compute_let_curve_cell(
        compute_let_effective_saturation(
            1.0 - sw - sorw, 1.0 - swc - sorw, minimum_mobile_pore_space
        ),
        owL,
        owE,
        owT,
    )
    kro_g = compute_let_curve_cell(
        compute_let_effective_saturation(
            1.0 - swc - sg - sorg, 1.0 - swc - sorg - sgr, minimum_mobile_pore_space
        ),
        goL,
        goE,
        goT,
    )
    kro = maximum_oil_relperm * rule_value(
        rule_id,
        kro_w,
        kro_g,
        krw,
        krg,
        1.0,
        sw,
        so,
        sg,
        swc,
        sorw,
        sorg,
        rule_minimum_mobile_pore_space,
    )
    return krw, kro, krg


@numba.njit(cache=True, error_model="numpy")
def compute_let_oil_wet_values(
    sw: Number,
    so: Number,
    sg: Number,
    swc: Number,
    sorw: Number,
    sorg: Number,
    sgr: Number,
    water: tuple[Number, Number, Number],
    oil_water: tuple[Number, Number, Number],
    gas_oil: tuple[Number, Number, Number],
    gas: tuple[Number, Number, Number],
    maximum_water_relperm: Number,
    maximum_oil_relperm: Number,
    maximum_gas_relperm: Number,
    rule_id: Integer,
    minimum_mobile_pore_space: Number,
    rule_minimum_mobile_pore_space: Number,
) -> tuple[Number, Number, Number]:
    """LET relative permeabilities of an oil-wet cell, before limiting and flooring."""
    wL, wE, wT = water
    gL, gE, gT = gas
    owL, owE, owT = oil_water
    lowest_residual_oil = min(sorw, sorg)
    kro = maximum_oil_relperm * compute_let_curve_cell(
        compute_let_effective_saturation(
            so - lowest_residual_oil, 1.0 - sorw - sorg, minimum_mobile_pore_space
        ),
        owL,
        owE,
        owT,
    )
    krg = maximum_gas_relperm * compute_let_curve_cell(
        compute_let_effective_saturation(sg - sgr, 1.0 - sgr - swc, minimum_mobile_pore_space),
        gL,
        gE,
        gT,
    )
    krw_oil_water = compute_let_curve_cell(
        compute_let_effective_saturation(sw - swc, 1.0 - swc - sorw, minimum_mobile_pore_space),
        wL,
        wE,
        wT,
    )
    krw_gas_water = compute_let_curve_cell(
        compute_let_effective_saturation(sw - swc, 1.0 - swc - sgr, minimum_mobile_pore_space),
        wL,
        wE,
        wT,
    )
    krw = maximum_water_relperm * rule_value(
        rule_id,
        krw_oil_water,
        krw_gas_water,
        kro,
        krg,
        1.0,
        so,
        sw,
        sg,
        lowest_residual_oil,
        swc,
        swc,
        rule_minimum_mobile_pore_space,
    )
    return krw, kro, krg


@numba.njit(cache=True, error_model="numpy")
def compute_let_water_wet_cell(
    sw: Number,
    so: Number,
    sg: Number,
    swc: Number,
    sorw: Number,
    sorg: Number,
    sgr: Number,
    water: tuple[Number, Number, Number],
    oil_water: tuple[Number, Number, Number],
    gas_oil: tuple[Number, Number, Number],
    gas: tuple[Number, Number, Number],
    maximum_water_relperm: Number,
    maximum_oil_relperm: Number,
    maximum_gas_relperm: Number,
    rule_id: Integer,
    minimum_mobile_pore_space: Number,
    rule_minimum_mobile_pore_space: Number,
) -> RelPermCell:
    """
    LET relative permeabilities and derivatives of a water-wet cell.

    Water and gas follow LET curves of their own saturation. Oil follows the three-phase mixing
    rule applied to the two-phase oil curves of the oil-water and gas-oil systems, and its
    derivatives are completed by the chain rule through the mixing rule.
    """
    krw, dkrw_dsw = compute_let_phase(
        sw - swc, 1.0 - swc - sorw, water, maximum_water_relperm, minimum_mobile_pore_space
    )
    krg, dkrg_dsg = compute_let_phase(
        sg - sgr, 1.0 - swc - sgr - sorg, gas, maximum_gas_relperm, minimum_mobile_pore_space
    )
    kro_w, dkro_w_dsw = compute_let_declining(
        1.0 - sw - sorw, 1.0 - swc - sorw, oil_water, minimum_mobile_pore_space
    )
    kro_g, dkro_g_dsg = compute_let_declining(
        1.0 - swc - sg - sorg, 1.0 - swc - sorg - sgr, gas_oil, minimum_mobile_pore_space
    )
    d_kro_w, d_kro_g, d_krw, d_krg, d_sw, d_so, d_sg = rule_derivatives(
        rule_id,
        kro_w,
        kro_g,
        krw,
        krg,
        1.0,
        sw,
        so,
        sg,
        swc,
        sorw,
        sorg,
        rule_minimum_mobile_pore_space,
    )
    kro = maximum_oil_relperm * rule_value(
        rule_id,
        kro_w,
        kro_g,
        krw,
        krg,
        1.0,
        sw,
        so,
        sg,
        swc,
        sorw,
        sorg,
        rule_minimum_mobile_pore_space,
    )
    return RelPermCell(
        krw,
        kro,
        krg,
        dkrw_dsw,
        0.0,
        0.0,
        maximum_oil_relperm * (d_kro_w * dkro_w_dsw + d_krw * dkrw_dsw + d_sw),
        maximum_oil_relperm * d_so,
        maximum_oil_relperm * (d_kro_g * dkro_g_dsg + d_krg * dkrg_dsg + d_sg),
        dkrg_dsg,
    )


@numba.njit(cache=True, error_model="numpy")
def compute_let_oil_wet_cell(
    sw: Number,
    so: Number,
    sg: Number,
    swc: Number,
    sorw: Number,
    sorg: Number,
    sgr: Number,
    water: tuple[Number, Number, Number],
    oil_water: tuple[Number, Number, Number],
    gas_oil: tuple[Number, Number, Number],
    gas: tuple[Number, Number, Number],
    maximum_water_relperm: Number,
    maximum_oil_relperm: Number,
    maximum_gas_relperm: Number,
    rule_id: Integer,
    minimum_mobile_pore_space: Number,
    rule_minimum_mobile_pore_space: Number,
) -> RelPermCell:
    """
    LET relative permeabilities and derivatives of an oil-wet cell.

    Oil is the wetting phase, so oil follows the oil-water LET curve of oil saturation and gas a
    LET curve of gas saturation. Water is the intermediate phase, so it follows the three-phase
    mixing rule applied to the two-phase water curves, and its derivatives are completed by the
    chain rule through the rule. The rule is evaluated with the roles of water and oil exchanged.
    """
    lowest_residual_oil = min(sorw, sorg)
    kro, dkro_dso = compute_let_phase(
        so - lowest_residual_oil,
        1.0 - sorw - sorg,
        oil_water,
        maximum_oil_relperm,
        minimum_mobile_pore_space,
    )
    krg, dkrg_dsg = compute_let_phase(
        sg - sgr, 1.0 - sgr - swc, gas, maximum_gas_relperm, minimum_mobile_pore_space
    )
    krw_oil_water, dkrw_oil_water_dsw = compute_let_phase(
        sw - swc, 1.0 - swc - sorw, water, 1.0, minimum_mobile_pore_space
    )
    krw_gas_water, dkrw_gas_water_dsw = compute_let_phase(
        sw - swc, 1.0 - swc - sgr, water, 1.0, minimum_mobile_pore_space
    )
    # The rule is evaluated with water and oil exchanged, so its "water saturation" slot is oil
    # saturation and its "oil saturation" slot is water saturation.
    d_krw_oil_water, d_krw_gas_water, d_kro, d_krg, d_slot_water, d_slot_oil, d_slot_gas = (
        rule_derivatives(
            rule_id,
            krw_oil_water,
            krw_gas_water,
            kro,
            krg,
            1.0,
            so,
            sw,
            sg,
            lowest_residual_oil,
            swc,
            swc,
            rule_minimum_mobile_pore_space,
        )
    )
    krw = maximum_water_relperm * rule_value(
        rule_id,
        krw_oil_water,
        krw_gas_water,
        kro,
        krg,
        1.0,
        so,
        sw,
        sg,
        lowest_residual_oil,
        swc,
        swc,
        rule_minimum_mobile_pore_space,
    )
    return RelPermCell(
        krw,
        kro,
        krg,
        maximum_water_relperm
        * (
            d_krw_oil_water * dkrw_oil_water_dsw
            + d_krw_gas_water * dkrw_gas_water_dsw
            + d_slot_oil
        ),
        maximum_water_relperm * (d_kro * dkro_dso + d_slot_water),
        maximum_water_relperm * (d_krg * dkrg_dsg + d_slot_gas),
        0.0,
        dkro_dso,
        0.0,
        dkrg_dsg,
    )


@numba.njit(cache=True, error_model="numpy")
def compute_let_values_cell(
    sw: Number,
    so: Number,
    sg: Number,
    swc: Number,
    sorw: Number,
    sorg: Number,
    sgr: Number,
    water: tuple[Number, Number, Number],
    oil_water: tuple[Number, Number, Number],
    gas_oil: tuple[Number, Number, Number],
    gas: tuple[Number, Number, Number],
    maximum_water_relperm: Number,
    maximum_oil_relperm: Number,
    maximum_gas_relperm: Number,
    wettability: Wettability,
    mixed_wet_water_fraction: Number,
    rule_id: Integer,
    saturation_epsilon: Number,
    minimum_mobile_pore_space: Number,
    rule_minimum_mobile_pore_space: Number,
    minimum_water_relperm: Number,
    minimum_oil_relperm: Number,
    minimum_gas_relperm: Number,
) -> tuple[Number, Number, Number]:
    """
    LET relative permeabilities of one cell.

    The saturations are scaled to sum to 1 when they differ from 1 by more than
    `saturation_epsilon`, and the results are limited to the range 0 to 1 and floored by the
    minimum relative permeabilities. A minimum of `NO_MINIMUM` means no floor.

    :return: Tuple of water, oil and gas relative permeability.
    """
    sw, so, sg = normalize_cell_saturations(sw, so, sg, saturation_epsilon)
    if wettability == Wettability.WATER_WET:
        krw, kro, krg = compute_let_water_wet_values(
            sw,
            so,
            sg,
            swc,
            sorw,
            sorg,
            sgr,
            water,
            oil_water,
            gas_oil,
            gas,
            maximum_water_relperm,
            maximum_oil_relperm,
            maximum_gas_relperm,
            rule_id,
            minimum_mobile_pore_space,
            rule_minimum_mobile_pore_space,
        )
    elif wettability == Wettability.OIL_WET:
        krw, kro, krg = compute_let_oil_wet_values(
            sw,
            so,
            sg,
            swc,
            sorw,
            sorg,
            sgr,
            water,
            oil_water,
            gas_oil,
            gas,
            maximum_water_relperm,
            maximum_oil_relperm,
            maximum_gas_relperm,
            rule_id,
            minimum_mobile_pore_space,
            rule_minimum_mobile_pore_space,
        )
    else:
        krw_ww, kro_ww, krg_ww = compute_let_water_wet_values(
            sw,
            so,
            sg,
            swc,
            sorw,
            sorg,
            sgr,
            water,
            oil_water,
            gas_oil,
            gas,
            maximum_water_relperm,
            maximum_oil_relperm,
            maximum_gas_relperm,
            rule_id,
            minimum_mobile_pore_space,
            rule_minimum_mobile_pore_space,
        )
        krw_ow, kro_ow, krg_ow = compute_let_oil_wet_values(
            sw,
            so,
            sg,
            swc,
            sorw,
            sorg,
            sgr,
            water,
            oil_water,
            gas_oil,
            gas,
            maximum_water_relperm,
            maximum_oil_relperm,
            maximum_gas_relperm,
            rule_id,
            minimum_mobile_pore_space,
            rule_minimum_mobile_pore_space,
        )
        fraction = mixed_wet_water_fraction
        krw = fraction * krw_ww + (1.0 - fraction) * krw_ow
        kro = fraction * kro_ww + (1.0 - fraction) * kro_ow
        krg = fraction * krg_ww + (1.0 - fraction) * krg_ow
    return (
        floor_relperm(clip_unit(krw), minimum_water_relperm),
        floor_relperm(clip_unit(kro), minimum_oil_relperm),
        floor_relperm(clip_unit(krg), minimum_gas_relperm),
    )


@numba.njit(cache=True, error_model="numpy")
def compute_let_derivatives_cell(
    sw: Number,
    so: Number,
    sg: Number,
    swc: Number,
    sorw: Number,
    sorg: Number,
    sgr: Number,
    water: tuple[Number, Number, Number],
    oil_water: tuple[Number, Number, Number],
    gas_oil: tuple[Number, Number, Number],
    gas: tuple[Number, Number, Number],
    maximum_water_relperm: Number,
    maximum_oil_relperm: Number,
    maximum_gas_relperm: Number,
    wettability: Wettability,
    mixed_wet_water_fraction: Number,
    rule_id: Integer,
    saturation_epsilon: Number,
    minimum_mobile_pore_space: Number,
    rule_minimum_mobile_pore_space: Number,
    minimum_water_relperm: Number,
    minimum_oil_relperm: Number,
    minimum_gas_relperm: Number,
) -> tuple[Number, Number, Number, Number, Number, Number, Number, Number, Number]:
    """
    LET partial derivatives of the relative permeabilities of one cell.

    The saturations are scaled in the same way as in `compute_let_values_cell`, so the
    derivatives belong to the same point as the values. A minimum of `NO_MINIMUM` means no floor.

    :return: Tuple of `dkrw_dsw`, `dkro_dsw`, `dkrg_dsw`, `dkrw_dso`, `dkro_dso`, `dkrg_dso`,
        `dkrw_dsg`, `dkro_dsg` and `dkrg_dsg`.
    """
    sw, so, sg = normalize_cell_saturations(sw, so, sg, saturation_epsilon)
    if wettability == Wettability.WATER_WET:
        return finalize_water_wet_cell(
            compute_let_water_wet_cell(
                sw,
                so,
                sg,
                swc,
                sorw,
                sorg,
                sgr,
                water,
                oil_water,
                gas_oil,
                gas,
                maximum_water_relperm,
                maximum_oil_relperm,
                maximum_gas_relperm,
                rule_id,
                minimum_mobile_pore_space,
                rule_minimum_mobile_pore_space,
            ),
            minimum_water_relperm,
            minimum_oil_relperm,
            minimum_gas_relperm,
        )
    if wettability == Wettability.OIL_WET:
        return finalize_oil_wet_cell(
            compute_let_oil_wet_cell(
                sw,
                so,
                sg,
                swc,
                sorw,
                sorg,
                sgr,
                water,
                oil_water,
                gas_oil,
                gas,
                maximum_water_relperm,
                maximum_oil_relperm,
                maximum_gas_relperm,
                rule_id,
                minimum_mobile_pore_space,
                rule_minimum_mobile_pore_space,
            ),
            minimum_water_relperm,
            minimum_oil_relperm,
            minimum_gas_relperm,
        )
    return finalize_mixed_wet_cell(
        blend_cells(
            compute_let_water_wet_cell(
                sw,
                so,
                sg,
                swc,
                sorw,
                sorg,
                sgr,
                water,
                oil_water,
                gas_oil,
                gas,
                maximum_water_relperm,
                maximum_oil_relperm,
                maximum_gas_relperm,
                rule_id,
                minimum_mobile_pore_space,
                rule_minimum_mobile_pore_space,
            ),
            compute_let_oil_wet_cell(
                sw,
                so,
                sg,
                swc,
                sorw,
                sorg,
                sgr,
                water,
                oil_water,
                gas_oil,
                gas,
                maximum_water_relperm,
                maximum_oil_relperm,
                maximum_gas_relperm,
                rule_id,
                minimum_mobile_pore_space,
                rule_minimum_mobile_pore_space,
            ),
            mixed_wet_water_fraction,
        ),
        minimum_water_relperm,
        minimum_oil_relperm,
        minimum_gas_relperm,
    )


@numba.njit(cache=True, parallel=True, error_model="numpy")
def compute_let_values_loop(
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    irreducible_water_saturation: NumberArray[OneDimension],
    residual_oil_saturation_water: NumberArray[OneDimension],
    residual_oil_saturation_gas: NumberArray[OneDimension],
    residual_gas_saturation: NumberArray[OneDimension],
    water: tuple[Number, Number, Number],
    oil_water: tuple[Number, Number, Number],
    gas_oil: tuple[Number, Number, Number],
    gas: tuple[Number, Number, Number],
    maximum_water_relperm: Number,
    maximum_oil_relperm: Number,
    maximum_gas_relperm: Number,
    wettability: Wettability,
    mixed_wet_water_fraction: Number,
    rule_id: Integer,
    saturation_epsilon: Number,
    minimum_mobile_pore_space: Number,
    rule_minimum_mobile_pore_space: Number,
    minimum_water_relperm: Number,
    minimum_oil_relperm: Number,
    minimum_gas_relperm: Number,
    size: Integer,
    water_relperm: NumberArray[OneDimension],
    oil_relperm: NumberArray[OneDimension],
    gas_relperm: NumberArray[OneDimension],
) -> None:
    """
    Compute the LET relative permeabilities of `size` cells in parallel and write them
    to the output arrays.

    Per-cell inputs are flat arrays, or a single value shared by every cell.
    """
    for i in numba.prange(size):
        krw, kro, krg = compute_let_values_cell(
            np.float64(element_at(water_saturation, i)),
            np.float64(element_at(oil_saturation, i)),
            np.float64(element_at(gas_saturation, i)),
            np.float64(element_at(irreducible_water_saturation, i)),
            np.float64(element_at(residual_oil_saturation_water, i)),
            np.float64(element_at(residual_oil_saturation_gas, i)),
            np.float64(element_at(residual_gas_saturation, i)),
            water,
            oil_water,
            gas_oil,
            gas,
            maximum_water_relperm,
            maximum_oil_relperm,
            maximum_gas_relperm,
            wettability,
            mixed_wet_water_fraction,
            rule_id,
            saturation_epsilon,
            minimum_mobile_pore_space,
            rule_minimum_mobile_pore_space,
            minimum_water_relperm,
            minimum_oil_relperm,
            minimum_gas_relperm,
        )
        water_relperm[i] = krw
        oil_relperm[i] = kro
        gas_relperm[i] = krg


@numba.njit(cache=True, parallel=True, error_model="numpy")
def compute_let_derivatives_loop(
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    irreducible_water_saturation: NumberArray[OneDimension],
    residual_oil_saturation_water: NumberArray[OneDimension],
    residual_oil_saturation_gas: NumberArray[OneDimension],
    residual_gas_saturation: NumberArray[OneDimension],
    water: tuple[Number, Number, Number],
    oil_water: tuple[Number, Number, Number],
    gas_oil: tuple[Number, Number, Number],
    gas: tuple[Number, Number, Number],
    maximum_water_relperm: Number,
    maximum_oil_relperm: Number,
    maximum_gas_relperm: Number,
    wettability: Wettability,
    mixed_wet_water_fraction: Number,
    rule_id: Integer,
    saturation_epsilon: Number,
    minimum_mobile_pore_space: Number,
    rule_minimum_mobile_pore_space: Number,
    minimum_water_relperm: Number,
    minimum_oil_relperm: Number,
    minimum_gas_relperm: Number,
    size: Integer,
    dkrw_dsw: NumberArray[OneDimension],
    dkro_dsw: NumberArray[OneDimension],
    dkrg_dsw: NumberArray[OneDimension],
    dkrw_dso: NumberArray[OneDimension],
    dkro_dso: NumberArray[OneDimension],
    dkrg_dso: NumberArray[OneDimension],
    dkrw_dsg: NumberArray[OneDimension],
    dkro_dsg: NumberArray[OneDimension],
    dkrg_dsg: NumberArray[OneDimension],
) -> None:
    """
    Compute the LET relative permeability derivatives of `size` cells in parallel and
    write them to the output arrays.

    Per-cell inputs are flat arrays, or a single value shared by every cell.
    """
    for i in numba.prange(size):
        d = compute_let_derivatives_cell(
            np.float64(element_at(water_saturation, i)),
            np.float64(element_at(oil_saturation, i)),
            np.float64(element_at(gas_saturation, i)),
            np.float64(element_at(irreducible_water_saturation, i)),
            np.float64(element_at(residual_oil_saturation_water, i)),
            np.float64(element_at(residual_oil_saturation_gas, i)),
            np.float64(element_at(residual_gas_saturation, i)),
            water,
            oil_water,
            gas_oil,
            gas,
            maximum_water_relperm,
            maximum_oil_relperm,
            maximum_gas_relperm,
            wettability,
            mixed_wet_water_fraction,
            rule_id,
            saturation_epsilon,
            minimum_mobile_pore_space,
            rule_minimum_mobile_pore_space,
            minimum_water_relperm,
            minimum_oil_relperm,
            minimum_gas_relperm,
        )
        dkrw_dsw[i] = d[0]
        dkro_dsw[i] = d[1]
        dkrg_dsw[i] = d[2]
        dkrw_dso[i] = d[3]
        dkro_dso[i] = d[4]
        dkrg_dso[i] = d[5]
        dkrw_dsg[i] = d[6]
        dkro_dsg[i] = d[7]
        dkrg_dsg[i] = d[8]


def to_kernel_parameters(
    parameters: tuple[Number, Number, Number],
) -> tuple[Number, Number, Number]:
    """
    LET parameters `(L, E, T)` as the tuple of floats the compiled kernels take.

    :param parameters: LET parameters of one phase.
    :return: The parameters as a tuple of three floats.
    """
    shape_low, shape_elevation, shape_top = parameters
    return float(shape_low), float(shape_elevation), float(shape_top)


def compute_let_relative_permeabilities_compiled(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    water: tuple[Number, Number, Number],
    oil_water: tuple[Number, Number, Number],
    gas_oil: tuple[Number, Number, Number],
    gas: tuple[Number, Number, Number],
    rule_id: Integer,
    saturation_epsilon: Number,
    minimum_mobile_pore_space: Number,
    maximum_water_relperm: Number = 1.0,
    maximum_oil_relperm: Number = 1.0,
    maximum_gas_relperm: Number = 1.0,
    wettability: Wettability = Wettability.WATER_WET,
    mixed_wet_water_fraction: Number = 0.5,
    minimum_water_relperm: Number | None = None,
    minimum_oil_relperm: Number | None = None,
    minimum_gas_relperm: Number | None = None,
) -> tuple[NumberOrArray[NDimension], NumberOrArray[NDimension], NumberOrArray[NDimension]]:
    """
    LET relative permeabilities computed by the compiled parallel loop, for the
    built-in mixing rule identified by `rule_id`.

    The parameters match `compute_let_relative_permeabilities`, with the mixing rule replaced by
    its identifier from `get_builtin_rule_id` and each phase's LET parameters given as a tuple
    `(L, E, T)`.

    :return: Tuple of water, oil and gas relative permeability, as scalars when every
        saturation and endpoint is a scalar and as arrays otherwise.
    """
    if wettability not in tuple(Wettability):
        raise ValidationError(f"Wettability {wettability!r} not implemented.")

    is_scalar = all(
        np.isscalar(value)
        for value in (
            water_saturation,
            oil_saturation,
            gas_saturation,
            irreducible_water_saturation,
            residual_oil_saturation_water,
            residual_oil_saturation_gas,
            residual_gas_saturation,
        )
    )
    shape, size, inputs = prepare_relperm_inputs(
        water_saturation,
        oil_saturation,
        gas_saturation,
        irreducible_water_saturation,
        residual_oil_saturation_water,
        residual_oil_saturation_gas,
        residual_gas_saturation,
    )
    dtype = get_floating_dtype(water_saturation)
    water_relperm = np.empty(size, dtype=dtype)
    oil_relperm = np.empty(size, dtype=dtype)
    gas_relperm = np.empty(size, dtype=dtype)
    compute_let_values_loop(
        *inputs,
        to_kernel_parameters(water),
        to_kernel_parameters(oil_water),
        to_kernel_parameters(gas_oil),
        to_kernel_parameters(gas),
        maximum_water_relperm,
        maximum_oil_relperm,
        maximum_gas_relperm,
        wettability,
        mixed_wet_water_fraction,
        rule_id,
        saturation_epsilon,
        minimum_mobile_pore_space,
        c.MINIMUM_MOBILE_PORE_SPACE,
        to_kernel_minimum(minimum_water_relperm),
        to_kernel_minimum(minimum_oil_relperm),
        to_kernel_minimum(minimum_gas_relperm),
        size,
        water_relperm,
        oil_relperm,
        gas_relperm,
    )
    if is_scalar:
        return water_relperm.item(), oil_relperm.item(), gas_relperm.item()
    return (
        water_relperm.reshape(shape),
        oil_relperm.reshape(shape),
        gas_relperm.reshape(shape),
    )


def compute_let_relative_permeability_derivatives_compiled(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    water: tuple[Number, Number, Number],
    oil_water: tuple[Number, Number, Number],
    gas_oil: tuple[Number, Number, Number],
    gas: tuple[Number, Number, Number],
    rule_id: Integer,
    saturation_epsilon: Number,
    minimum_mobile_pore_space: Number,
    maximum_water_relperm: Number = 1.0,
    maximum_oil_relperm: Number = 1.0,
    maximum_gas_relperm: Number = 1.0,
    wettability: Wettability = Wettability.WATER_WET,
    mixed_wet_water_fraction: Number = 0.5,
    minimum_water_relperm: Number | None = None,
    minimum_oil_relperm: Number | None = None,
    minimum_gas_relperm: Number | None = None,
) -> RelativePermeabilityDerivatives:
    """
    LET relative permeability derivatives computed by the compiled parallel loop, for
    the built-in mixing rule identified by `rule_id`.

    The parameters match `compute_let_relative_permeability_derivatives`, with the mixing rule
    replaced by its identifier from `get_builtin_rule_id`, each phase's LET parameters given as a
    tuple `(L, E, T)`, and the finite-difference step dropped because the built-in rules have
    analytical derivatives.

    :return: `RelativePermeabilityDerivatives` dictionary, with scalars when every saturation
        and endpoint is a scalar and arrays otherwise.
    """
    if wettability not in tuple(Wettability):
        raise ValidationError(f"Wettability {wettability!r} not implemented.")

    is_scalar = all(
        np.isscalar(value)
        for value in (
            water_saturation,
            oil_saturation,
            gas_saturation,
            irreducible_water_saturation,
            residual_oil_saturation_water,
            residual_oil_saturation_gas,
            residual_gas_saturation,
        )
    )
    shape, size, inputs = prepare_relperm_inputs(
        water_saturation,
        oil_saturation,
        gas_saturation,
        irreducible_water_saturation,
        residual_oil_saturation_water,
        residual_oil_saturation_gas,
        residual_gas_saturation,
    )
    dtype = get_floating_dtype(water_saturation)
    outputs = tuple(np.empty(size, dtype=dtype) for _ in range(9))
    compute_let_derivatives_loop(
        *inputs,
        to_kernel_parameters(water),
        to_kernel_parameters(oil_water),
        to_kernel_parameters(gas_oil),
        to_kernel_parameters(gas),
        maximum_water_relperm,
        maximum_oil_relperm,
        maximum_gas_relperm,
        wettability,
        mixed_wet_water_fraction,
        rule_id,
        saturation_epsilon,
        minimum_mobile_pore_space,
        c.MINIMUM_MOBILE_PORE_SPACE,
        to_kernel_minimum(minimum_water_relperm),
        to_kernel_minimum(minimum_oil_relperm),
        to_kernel_minimum(minimum_gas_relperm),
        size,
        *outputs,
    )
    dkrw_dsw, dkro_dsw, dkrg_dsw, dkrw_dso, dkro_dso, dkrg_dso, dkrw_dsg, dkro_dsg, dkrg_dsg = (
        output.item() if is_scalar else output.reshape(shape) for output in outputs
    )
    return RelativePermeabilityDerivatives(
        dkrw_dsw=dkrw_dsw,
        dkro_dsw=dkro_dsw,
        dkrg_dsw=dkrg_dsw,
        dkrw_dso=dkrw_dso,
        dkro_dso=dkro_dso,
        dkrg_dso=dkrg_dso,
        dkrw_dsg=dkrw_dsg,
        dkro_dsg=dkro_dsg,
        dkrg_dsg=dkrg_dsg,
    )


@numba.njit(cache=True, error_model="numpy")
def compute_let_values_and_derivatives_cell(
    sw: Number,
    so: Number,
    sg: Number,
    swc: Number,
    sorw: Number,
    sorg: Number,
    sgr: Number,
    water: tuple[Number, Number, Number],
    oil_water: tuple[Number, Number, Number],
    gas_oil: tuple[Number, Number, Number],
    gas: tuple[Number, Number, Number],
    maximum_water_relperm: Number,
    maximum_oil_relperm: Number,
    maximum_gas_relperm: Number,
    wettability: Wettability,
    mixed_wet_water_fraction: Number,
    rule_id: Integer,
    saturation_epsilon: Number,
    minimum_mobile_pore_space: Number,
    rule_minimum_mobile_pore_space: Number,
    minimum_water_relperm: Number,
    minimum_oil_relperm: Number,
    minimum_gas_relperm: Number,
) -> tuple[float, float, float, float, float, float, float, float, float, float, float, float]:
    """
    LET relative permeabilities and all nine partial derivatives of one cell, from a
    single evaluation of the underlying two-phase curves and mixing rule.

    The results are identical to `compute_let_values_cell` and
    `compute_let_derivatives_cell`. A minimum of `NO_MINIMUM` means no floor.

    :return: Tuple of water, oil and gas relative permeability followed by `dkrw_dsw`,
        `dkro_dsw`, `dkrg_dsw`, `dkrw_dso`, `dkro_dso`, `dkrg_dso`, `dkrw_dsg`, `dkro_dsg` and
        `dkrg_dsg`.
    """
    sw, so, sg = normalize_cell_saturations(sw, so, sg, saturation_epsilon)
    if wettability == Wettability.WATER_WET:
        cell = compute_let_water_wet_cell(
            sw,
            so,
            sg,
            swc,
            sorw,
            sorg,
            sgr,
            water,
            oil_water,
            gas_oil,
            gas,
            maximum_water_relperm,
            maximum_oil_relperm,
            maximum_gas_relperm,
            rule_id,
            minimum_mobile_pore_space,
            rule_minimum_mobile_pore_space,
        )
        d = finalize_water_wet_cell(
            cell, minimum_water_relperm, minimum_oil_relperm, minimum_gas_relperm
        )
    elif wettability == Wettability.OIL_WET:
        cell = compute_let_oil_wet_cell(
            sw,
            so,
            sg,
            swc,
            sorw,
            sorg,
            sgr,
            water,
            oil_water,
            gas_oil,
            gas,
            maximum_water_relperm,
            maximum_oil_relperm,
            maximum_gas_relperm,
            rule_id,
            minimum_mobile_pore_space,
            rule_minimum_mobile_pore_space,
        )
        d = finalize_oil_wet_cell(
            cell, minimum_water_relperm, minimum_oil_relperm, minimum_gas_relperm
        )
    else:
        cell = blend_cells(
            compute_let_water_wet_cell(
                sw,
                so,
                sg,
                swc,
                sorw,
                sorg,
                sgr,
                water,
                oil_water,
                gas_oil,
                gas,
                maximum_water_relperm,
                maximum_oil_relperm,
                maximum_gas_relperm,
                rule_id,
                minimum_mobile_pore_space,
                rule_minimum_mobile_pore_space,
            ),
            compute_let_oil_wet_cell(
                sw,
                so,
                sg,
                swc,
                sorw,
                sorg,
                sgr,
                water,
                oil_water,
                gas_oil,
                gas,
                maximum_water_relperm,
                maximum_oil_relperm,
                maximum_gas_relperm,
                rule_id,
                minimum_mobile_pore_space,
                rule_minimum_mobile_pore_space,
            ),
            mixed_wet_water_fraction,
        )
        d = finalize_mixed_wet_cell(
            cell, minimum_water_relperm, minimum_oil_relperm, minimum_gas_relperm
        )
    return (
        floor_relperm(clip_unit(cell.krw), minimum_water_relperm),
        floor_relperm(clip_unit(cell.kro), minimum_oil_relperm),
        floor_relperm(clip_unit(cell.krg), minimum_gas_relperm),
        d[0],
        d[1],
        d[2],
        d[3],
        d[4],
        d[5],
        d[6],
        d[7],
        d[8],
    )


@numba.njit(cache=True, parallel=True, error_model="numpy")
def compute_let_values_and_derivatives_loop(
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    irreducible_water_saturation: NumberArray[OneDimension],
    residual_oil_saturation_water: NumberArray[OneDimension],
    residual_oil_saturation_gas: NumberArray[OneDimension],
    residual_gas_saturation: NumberArray[OneDimension],
    water: tuple[Number, Number, Number],
    oil_water: tuple[Number, Number, Number],
    gas_oil: tuple[Number, Number, Number],
    gas: tuple[Number, Number, Number],
    maximum_water_relperm: Number,
    maximum_oil_relperm: Number,
    maximum_gas_relperm: Number,
    wettability: Wettability,
    mixed_wet_water_fraction: Number,
    rule_id: Integer,
    saturation_epsilon: Number,
    minimum_mobile_pore_space: Number,
    rule_minimum_mobile_pore_space: Number,
    minimum_water_relperm: Number,
    minimum_oil_relperm: Number,
    minimum_gas_relperm: Number,
    size: Integer,
    water_relperm: NumberArray[OneDimension],
    oil_relperm: NumberArray[OneDimension],
    gas_relperm: NumberArray[OneDimension],
    dkrw_dsw: NumberArray[OneDimension],
    dkro_dsw: NumberArray[OneDimension],
    dkrg_dsw: NumberArray[OneDimension],
    dkrw_dso: NumberArray[OneDimension],
    dkro_dso: NumberArray[OneDimension],
    dkrg_dso: NumberArray[OneDimension],
    dkrw_dsg: NumberArray[OneDimension],
    dkro_dsg: NumberArray[OneDimension],
    dkrg_dsg: NumberArray[OneDimension],
) -> None:
    """
    Compute the LET relative permeabilities and all nine derivatives of `size` cells in
    parallel and write them to the output arrays.

    Per-cell inputs are flat arrays, or a single value shared by every cell.
    """
    for i in numba.prange(size):
        r = compute_let_values_and_derivatives_cell(
            np.float64(element_at(water_saturation, i)),
            np.float64(element_at(oil_saturation, i)),
            np.float64(element_at(gas_saturation, i)),
            np.float64(element_at(irreducible_water_saturation, i)),
            np.float64(element_at(residual_oil_saturation_water, i)),
            np.float64(element_at(residual_oil_saturation_gas, i)),
            np.float64(element_at(residual_gas_saturation, i)),
            water,
            oil_water,
            gas_oil,
            gas,
            maximum_water_relperm,
            maximum_oil_relperm,
            maximum_gas_relperm,
            wettability,
            mixed_wet_water_fraction,
            rule_id,
            saturation_epsilon,
            minimum_mobile_pore_space,
            rule_minimum_mobile_pore_space,
            minimum_water_relperm,
            minimum_oil_relperm,
            minimum_gas_relperm,
        )
        water_relperm[i] = r[0]
        oil_relperm[i] = r[1]
        gas_relperm[i] = r[2]
        dkrw_dsw[i] = r[3]
        dkro_dsw[i] = r[4]
        dkrg_dsw[i] = r[5]
        dkrw_dso[i] = r[6]
        dkro_dso[i] = r[7]
        dkrg_dso[i] = r[8]
        dkrw_dsg[i] = r[9]
        dkro_dsg[i] = r[10]
        dkrg_dsg[i] = r[11]


def compute_let_relative_permeabilities_and_derivatives_compiled(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    water: tuple[Number, Number, Number],
    oil_water: tuple[Number, Number, Number],
    gas_oil: tuple[Number, Number, Number],
    gas: tuple[Number, Number, Number],
    rule_id: Integer,
    saturation_epsilon: Number,
    minimum_mobile_pore_space: Number,
    maximum_water_relperm: Number = 1.0,
    maximum_oil_relperm: Number = 1.0,
    maximum_gas_relperm: Number = 1.0,
    wettability: Wettability = Wettability.WATER_WET,
    mixed_wet_water_fraction: Number = 0.5,
    minimum_water_relperm: Number | None = None,
    minimum_oil_relperm: Number | None = None,
    minimum_gas_relperm: Number | None = None,
) -> tuple[RelativePermeabilities, RelativePermeabilityDerivatives]:
    """
    LET relative permeabilities and derivatives computed together by the compiled
    parallel loop, for the built-in mixing rule identified by `rule_id`.

    The parameters match `compute_let_relative_permeability_derivatives_compiled`. The
    results are the same as calling the separate relative permeability and derivative functions,
    at roughly the cost of the derivative call alone.

    :return: Tuple of the `RelativePermeabilities` and `RelativePermeabilityDerivatives`
        dictionaries, with scalars when every saturation and endpoint is a scalar.
    """
    if wettability not in tuple(Wettability):
        raise ValidationError(f"Wettability {wettability!r} not implemented.")

    is_scalar = all(
        np.isscalar(value)
        for value in (
            water_saturation,
            oil_saturation,
            gas_saturation,
            irreducible_water_saturation,
            residual_oil_saturation_water,
            residual_oil_saturation_gas,
            residual_gas_saturation,
        )
    )
    shape, size, inputs = prepare_relperm_inputs(
        water_saturation,
        oil_saturation,
        gas_saturation,
        irreducible_water_saturation,
        residual_oil_saturation_water,
        residual_oil_saturation_gas,
        residual_gas_saturation,
    )
    dtype = get_floating_dtype(water_saturation)
    outputs = tuple(np.empty(size, dtype=dtype) for _ in range(12))
    compute_let_values_and_derivatives_loop(
        *inputs,
        to_kernel_parameters(water),
        to_kernel_parameters(oil_water),
        to_kernel_parameters(gas_oil),
        to_kernel_parameters(gas),
        maximum_water_relperm,
        maximum_oil_relperm,
        maximum_gas_relperm,
        wettability,
        mixed_wet_water_fraction,
        rule_id,
        saturation_epsilon,
        minimum_mobile_pore_space,
        c.MINIMUM_MOBILE_PORE_SPACE,
        to_kernel_minimum(minimum_water_relperm),
        to_kernel_minimum(minimum_oil_relperm),
        to_kernel_minimum(minimum_gas_relperm),
        size,
        *outputs,
    )
    (
        water_relperm,
        oil_relperm,
        gas_relperm,
        dkrw_dsw,
        dkro_dsw,
        dkrg_dsw,
        dkrw_dso,
        dkro_dso,
        dkrg_dso,
        dkrw_dsg,
        dkro_dsg,
        dkrg_dsg,
    ) = (output.item() if is_scalar else output.reshape(shape) for output in outputs)
    return (
        RelativePermeabilities(water=water_relperm, oil=oil_relperm, gas=gas_relperm),
        RelativePermeabilityDerivatives(
            dkrw_dsw=dkrw_dsw,
            dkro_dsw=dkro_dsw,
            dkrg_dsw=dkrg_dsw,
            dkrw_dso=dkrw_dso,
            dkro_dso=dkro_dso,
            dkrg_dso=dkrg_dso,
            dkrw_dsg=dkrw_dsg,
            dkro_dsg=dkro_dsg,
            dkrg_dsg=dkrg_dsg,
        ),
    )


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
    rule_id = get_builtin_rule_id(mixing_rule)
    if rule_id is not None:
        return compute_let_relative_permeabilities_compiled(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
            residual_gas_saturation=residual_gas_saturation,
            water=(water_L, water_E, water_T),
            oil_water=(oil_water_L, oil_water_E, oil_water_T),
            gas_oil=(gas_oil_L, gas_oil_E, gas_oil_T),
            gas=(gas_L, gas_E, gas_T),
            rule_id=rule_id,
            saturation_epsilon=saturation_epsilon,
            minimum_mobile_pore_space=minimum_mobile_pore_space,
            maximum_water_relperm=maximum_water_relperm,
            maximum_oil_relperm=maximum_oil_relperm,
            maximum_gas_relperm=maximum_gas_relperm,
            wettability=wettability,
            mixed_wet_water_fraction=mixed_wet_water_fraction,
            minimum_water_relperm=minimum_water_relperm,
            minimum_oil_relperm=minimum_oil_relperm,
            minimum_gas_relperm=minimum_gas_relperm,
        )

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
    effective_saturation = np.where(
        valid,
        np.clip((saturation - lower_saturation) / np.where(valid, mobile_range, 1.0), 0.0, 1.0),
        0.0,
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
    effective_saturation = np.where(valid, np.clip(unclipped_effective_saturation, 0.0, 1.0), 0.0)
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
    saturation_epsilon: Number,
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
    rule_id = get_builtin_rule_id(mixing_rule)
    if rule_id is not None:
        return compute_let_relative_permeability_derivatives_compiled(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
            residual_gas_saturation=residual_gas_saturation,
            water=(water.L, water.E, water.T),
            oil_water=(oil_water.L, oil_water.E, oil_water.T),
            gas_oil=(gas_oil.L, gas_oil.E, gas_oil.T),
            gas=(gas.L, gas.E, gas.T),
            rule_id=rule_id,
            saturation_epsilon=saturation_epsilon,
            minimum_mobile_pore_space=minimum_mobile_pore_space,
            maximum_water_relperm=maximum_water_relperm,
            maximum_oil_relperm=maximum_oil_relperm,
            maximum_gas_relperm=maximum_gas_relperm,
            wettability=wettability,
            mixed_wet_water_fraction=mixed_wet_water_fraction,
            minimum_water_relperm=minimum_water_relperm,
            minimum_oil_relperm=minimum_oil_relperm,
            minimum_gas_relperm=minimum_gas_relperm,
        )
    if wettability not in tuple(Wettability):
        raise ValidationError(f"Wettability {wettability!r} not implemented.")

    sw, so, sg, zeros, is_scalar = broadcast_saturations(
        water_saturation,
        oil_saturation,
        gas_saturation,
        irreducible_water_saturation,
        residual_oil_saturation_water,
        residual_oil_saturation_gas,
        residual_gas_saturation,
    )
    sw, so, sg = validate_and_normalize_saturations(sw, so, sg, saturation_epsilon)

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

        validate_analytical_relperm_parameters(
            irreducible_water_saturation=self.irreducible_water_saturation,
            residual_oil_saturation_water=self.residual_oil_saturation_water,
            residual_oil_saturation_gas=self.residual_oil_saturation_gas,
            residual_gas_saturation=self.residual_gas_saturation,
            maximum_water_relperm=self.maximum_water_relperm,
            maximum_oil_relperm=self.maximum_oil_relperm,
            maximum_gas_relperm=self.maximum_gas_relperm,
            mixed_wet_water_fraction=self.mixed_wet_water_fraction,
        )
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

    def resolve_endpoint_saturations(
        self,
        irreducible_water_saturation: NumberOrArray[NDimension] | None = None,
        residual_oil_saturation_water: NumberOrArray[NDimension] | None = None,
        residual_oil_saturation_gas: NumberOrArray[NDimension] | None = None,
        residual_gas_saturation: NumberOrArray[NDimension] | None = None,
    ) -> tuple[
        NumberOrArray[NDimension],
        NumberOrArray[NDimension],
        NumberOrArray[NDimension],
        NumberOrArray[NDimension],
    ]:
        """
        Endpoint saturations for an evaluation: each argument if given, otherwise the value set
        on the model.

        :param irreducible_water_saturation: Irreducible water saturation, or `None` to use the
            model's value.
        :param residual_oil_saturation_water: Residual oil saturation to water flooding, or
            `None` to use the model's value.
        :param residual_oil_saturation_gas: Residual oil saturation to gas flooding, or `None`
            to use the model's value.
        :param residual_gas_saturation: Residual gas saturation, or `None` to use the model's
            value.
        :raises ValidationError: If an endpoint is neither given nor set on the model.
        :return: Tuple of irreducible water, residual oil to water, residual oil to gas and
            residual gas saturation.
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
        return swc, sorw, sorg, sgr  # type: ignore[return-value]

    def evaluate_with_derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        irreducible_water_saturation: NumberOrArray[NDimension] | None = None,
        residual_oil_saturation_water: NumberOrArray[NDimension] | None = None,
        residual_oil_saturation_gas: NumberOrArray[NDimension] | None = None,
        residual_gas_saturation: NumberOrArray[NDimension] | None = None,
        **kwargs: typing.Any,
    ) -> tuple[RelativePermeabilities, RelativePermeabilityDerivatives]:
        """
        Compute the relative permeabilities and all nine partial derivatives together.

        The results are the same as calling `evaluate` and `derivatives` separately. For the
        built-in mixing rules the two share one pass over the cells, which is about as fast as
        the derivative call alone. A custom mixing rule is evaluated by calling `evaluate` and
        `derivatives` in turn.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param irreducible_water_saturation: Irreducible water saturation, or `None` to use the
            model's value.
        :param residual_oil_saturation_water: Residual oil saturation to water flooding, or
            `None` to use the model's value.
        :param residual_oil_saturation_gas: Residual oil saturation to gas flooding, or `None`
            to use the model's value.
        :param residual_gas_saturation: Residual gas saturation, or `None` to use the model's
            value.
        :param kwargs: Other key word arguments, which this model does not use.
        :return: Tuple of the `RelativePermeabilities` and `RelativePermeabilityDerivatives`
            dictionaries.
        """
        swc, sorw, sorg, sgr = self.resolve_endpoint_saturations(
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
            residual_gas_saturation=residual_gas_saturation,
        )
        rule_id = get_builtin_rule_id(self.mixing_rule)
        if rule_id is None:
            return (
                self.evaluate(
                    water_saturation,
                    oil_saturation,
                    gas_saturation,
                    irreducible_water_saturation=swc,
                    residual_oil_saturation_water=sorw,
                    residual_oil_saturation_gas=sorg,
                    residual_gas_saturation=sgr,
                ),
                self.derivatives(
                    water_saturation,
                    oil_saturation,
                    gas_saturation,
                    irreducible_water_saturation=swc,
                    residual_oil_saturation_water=sorw,
                    residual_oil_saturation_gas=sorg,
                    residual_gas_saturation=sgr,
                ),
            )
        return compute_let_relative_permeabilities_and_derivatives_compiled(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=swc,
            residual_oil_saturation_water=sorw,
            residual_oil_saturation_gas=sorg,
            residual_gas_saturation=sgr,
            water=(self.water.L, self.water.E, self.water.T),
            oil_water=(self.oil_water.L, self.oil_water.E, self.oil_water.T),
            gas_oil=(self.gas_oil.L, self.gas_oil.E, self.gas_oil.T),
            gas=(self.gas.L, self.gas.E, self.gas.T),
            rule_id=rule_id,
            saturation_epsilon=c.SATURATION_EPSILON,
            minimum_mobile_pore_space=c.MINIMUM_MOBILE_PORE_SPACE,
            maximum_water_relperm=self.get_water_relperm_endpoint(),
            maximum_oil_relperm=self.get_oil_relperm_endpoint(),
            maximum_gas_relperm=self.get_gas_relperm_endpoint(),
            wettability=self.wettability,
            mixed_wet_water_fraction=self.mixed_wet_water_fraction,
            minimum_water_relperm=resolve_min_relperm(self.minimum_water_relperm),
            minimum_oil_relperm=resolve_min_relperm(self.minimum_oil_relperm),
            minimum_gas_relperm=resolve_min_relperm(self.minimum_gas_relperm),
        )

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
        swc, sorw, sorg, sgr = self.resolve_endpoint_saturations(
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
            residual_gas_saturation=residual_gas_saturation,
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
        swc, sorw, sorg, sgr = self.resolve_endpoint_saturations(
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
            residual_gas_saturation=residual_gas_saturation,
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
            saturation_epsilon=c.SATURATION_EPSILON,
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
