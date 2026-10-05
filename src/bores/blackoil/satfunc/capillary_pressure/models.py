"""Capillary pressure analytical models/tables for multi-phase flow simulations."""

import typing

import attrs
import numba
import numpy as np
from typing_extensions import Self

from bores.blackoil.satfunc.capillary_pressure.tables import (
    CapillaryPressureTable,
)
from bores.constants import c, get_conversion_factors
from bores.errors import ValidationError
from bores.types import (
    CapillaryPressureDerivatives,
    CapillaryPressures,
    FluidPhase,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    UnitConversionTable,
    UnitSystem,
    Wettability,
)
from bores.utils import atleast_1d

__all__ = [
    "BrooksCoreyCapillaryPressureTable",
    "LeverettJCapillaryPressureTable",
    "VanGenuchtenCapillaryPressureTable",
]


@numba.njit(cache=True)
def compute_brooks_corey_capillary_pressures_scalar(
    water_saturation: Number,
    oil_saturation: Number,
    gas_saturation: Number,
    irreducible_water_saturation: Number,
    residual_oil_saturation_water: Number,
    residual_oil_saturation_gas: Number,
    residual_gas_saturation: Number,
    wettability: Wettability,
    oil_water_entry_pressure_water_wet: Number,
    oil_water_entry_pressure_oil_wet: Number,
    oil_water_pore_size_distribution_index_water_wet: Number,
    oil_water_pore_size_distribution_index_oil_wet: Number,
    gas_oil_entry_pressure: Number,
    gas_oil_pore_size_distribution_index: Number,
    mixed_wet_water_fraction: Number = 0.5,
    saturation_epsilon: Number = 1e-12,
    minimum_mobile_pore_space: Number = 1e-9,
) -> tuple[Number, Number]:
    """
    Scalar variant of Brooks-Corey capillary pressure computation.

    Computes capillary pressures (Pcow, Pcgo) using Brooks-Corey model.

    Pcow is defined as Po - Pw.
    Pcgo is defined as Pg - Po.

    Brooks-Corey model: Pc = Pd * (Se)^(-1/λ)
    where:
    - Pd is the displacement/entry pressure
    - Se is the effective saturation
    - λ is the pore size distribution index

    Wettability behavior:
    - WATER_WET: Pcow > 0, Pcgo > 0 (water preferentially wets rock)
    - OIL_WET:   Pcow < 0, Pcgo > 0 (oil preferentially wets rock)
    - MIXED_WET: Pcow varies with saturation (weighted combination)

    :param water_saturation: Current water saturation (fraction, 0-1).
    :param oil_saturation: Current oil saturation (fraction, 0-1).
    :param gas_saturation: Current gas saturation (fraction, 0-1).
    :param irreducible_water_saturation: Irreducible water saturation (swc).
    :param residual_oil_saturation_water: Residual oil saturation during water flooding (sorw).
    :param residual_oil_saturation_gas: Residual oil saturation during gas flooding (sorg).
    :param residual_gas_saturation: Residual gas saturation (sgr).
    :param wettability: Wettability type (WATER_WET, OIL_WET, or MIXED_WET).
    :param oil_water_entry_pressure_water_wet: Entry pressure for oil-water in water-wet system (psi).
    :param oil_water_entry_pressure_oil_wet: Entry pressure for oil-water in oil-wet system (psi).
    :param oil_water_pore_size_distribution_index_water_wet: Pore size distribution index (λ) for oil-water in water-wet.
    :param oil_water_pore_size_distribution_index_oil_wet: Pore size distribution index (λ) for oil-water in oil-wet.
    :param gas_oil_entry_pressure: Entry pressure for gas-oil (psi).
    :param gas_oil_pore_size_distribution_index: Pore size distribution index (λ) for gas-oil.
    :param mixed_wet_water_fraction: Fraction of pore space that is water-wet (0-1, default 0.5).
    :param saturation_epsilon: Small value to avoid division by zero in saturation calculations.
    :param minimum_mobile_pore_space: Minimum mobile pore space threshold below which Pc is set to zero.
    :return: Tuple of (oil_water_capillary_pressure, gas_oil_capillary_pressure) in psi.
    """
    sw: Number = water_saturation
    so: Number = oil_saturation
    sg: Number = gas_saturation
    swc: Number = irreducible_water_saturation
    sorw: Number = residual_oil_saturation_water
    sorg: Number = residual_oil_saturation_gas
    sgr: Number = residual_gas_saturation

    # Validate saturations
    if sw < 0 or sw > 1 or so < 0 or so > 1 or sg < 0 or sg > 1:
        raise ValidationError("Saturations must be between 0 and 1.")

    # Normalize saturations if they do not sum to 1
    total_saturation = sw + so + sg
    if abs(total_saturation - 1.0) > saturation_epsilon and total_saturation > 0.0:
        sw = sw / total_saturation
        sg = sg / total_saturation

    # Effective pore spaces
    total_mobile_pore_space_water = 1.0 - swc - sorw
    total_mobile_pore_space_gas = 1.0 - swc - sorg - sgr

    # Pcow (Po - Pw)
    oil_water_capillary_pressure: Number = 0.0
    if total_mobile_pore_space_water > minimum_mobile_pore_space:
        effective_water_saturation = (sw - swc) / total_mobile_pore_space_water
        effective_water_saturation = min(max(effective_water_saturation, saturation_epsilon), 1.0)

        if effective_water_saturation < 1.0 - saturation_epsilon:
            if wettability == Wettability.WATER_WET:
                oil_water_capillary_pressure = oil_water_entry_pressure_water_wet * (
                    effective_water_saturation
                    ** (-1.0 / oil_water_pore_size_distribution_index_water_wet)
                )
            elif wettability == Wettability.OIL_WET:
                oil_water_capillary_pressure = -(
                    oil_water_entry_pressure_oil_wet
                    * effective_water_saturation
                    ** (-1.0 / oil_water_pore_size_distribution_index_oil_wet)
                )
            elif wettability == Wettability.MIXED_WET:
                pcow_ww = oil_water_entry_pressure_water_wet * (
                    effective_water_saturation
                    ** (-1.0 / oil_water_pore_size_distribution_index_water_wet)
                )
                pcow_ow = -(
                    oil_water_entry_pressure_oil_wet
                    * effective_water_saturation
                    ** (-1.0 / oil_water_pore_size_distribution_index_oil_wet)
                )
                oil_water_capillary_pressure = (
                    mixed_wet_water_fraction * pcow_ww + (1.0 - mixed_wet_water_fraction) * pcow_ow
                )

    # Pcgo (Pg - Po)
    gas_oil_capillary_pressure: Number = 0.0
    if total_mobile_pore_space_gas > minimum_mobile_pore_space:
        effective_gas_saturation = (sg - sgr) / total_mobile_pore_space_gas
        effective_gas_saturation = min(max(effective_gas_saturation, saturation_epsilon), 1.0)

        if effective_gas_saturation < 1.0 - saturation_epsilon:
            gas_oil_capillary_pressure = gas_oil_entry_pressure * (
                effective_gas_saturation ** (-1.0 / gas_oil_pore_size_distribution_index)
            )

    return oil_water_capillary_pressure, gas_oil_capillary_pressure


@numba.njit(cache=True)
def compute_brooks_corey_capillary_pressures_array(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    wettability: Wettability,
    oil_water_entry_pressure_water_wet: Number,
    oil_water_entry_pressure_oil_wet: Number,
    oil_water_pore_size_distribution_index_water_wet: Number,
    oil_water_pore_size_distribution_index_oil_wet: Number,
    gas_oil_entry_pressure: Number,
    gas_oil_pore_size_distribution_index: Number,
    mixed_wet_water_fraction: Number = 0.5,
    saturation_epsilon: Number = 1e-12,
    minimum_mobile_pore_space: Number = 1e-9,
) -> tuple[NumberArray[NDimension], NumberArray[NDimension]]:
    """
    Array variant of Brooks-Corey capillary pressure computation.

    Computes capillary pressures (Pcow, Pcgo) using Brooks-Corey model.
    Supports array inputs (up to 3D). Always returns NDArrays.

    Pcow is defined as Po - Pw.
    Pcgo is defined as Pg - Po.

    Brooks-Corey model: Pc = Pd * (Se)^(-1/λ)
    where:
    - Pd is the displacement/entry pressure
    - Se is the effective saturation
    - λ is the pore size distribution index

    Wettability behavior:
    - `WATER_WET`: Pcow > 0, Pcgo > 0 (water preferentially wets rock)
    - `OIL_WET`:   Pcow < 0, Pcgo > 0 (oil preferentially wets rock)
    - `MIXED_WET`: Pcow varies with saturation (weighted combination)

    :param water_saturation: Current water saturation (fraction, 0-1) - scalar or array.
    :param oil_saturation: Current oil saturation (fraction, 0-1) - scalar or array.
    :param gas_saturation: Current gas saturation (fraction, 0-1) - scalar or array.
    :param irreducible_water_saturation: Irreducible water saturation (swc) - scalar or array.
    :param residual_oil_saturation_water: Residual oil saturation during water flooding (sorw) - scalar or array.
    :param residual_oil_saturation_gas: Residual oil saturation during gas flooding (sorg) - scalar or array.
    :param residual_gas_saturation: Residual gas saturation (sgr) - scalar or array.
    :param wettability: Wettability type (WATER_WET, OIL_WET, or MIXED_WET).
    :param oil_water_entry_pressure_water_wet: Entry pressure for oil-water in water-wet system (psi).
    :param oil_water_entry_pressure_oil_wet: Entry pressure for oil-water in oil-wet system (psi).
    :param oil_water_pore_size_distribution_index_water_wet: Pore size distribution index (λ) for oil-water in water-wet.
    :param oil_water_pore_size_distribution_index_oil_wet: Pore size distribution index (λ) for oil-water in oil-wet.
    :param gas_oil_entry_pressure: Entry pressure for gas-oil (psi).
    :param gas_oil_pore_size_distribution_index: Pore size distribution index (λ) for gas-oil.
    :param mixed_wet_water_fraction: Fraction of pore space that is water-wet (0-1, default 0.5).
    :param saturation_epsilon: Small value to avoid division by zero in saturation calculations.
    :param minimum_mobile_pore_space: Minimum mobile pore space threshold below which Pc is set to zero.
    :return: Tuple of (oil_water_capillary_pressure, gas_oil_capillary_pressure) NDArrays in psi.
    """
    sw = atleast_1d(water_saturation)
    so = atleast_1d(oil_saturation)
    sg = atleast_1d(gas_saturation)
    swc = atleast_1d(irreducible_water_saturation)
    sorw = atleast_1d(residual_oil_saturation_water)
    sorg = atleast_1d(residual_oil_saturation_gas)
    sgr = atleast_1d(residual_gas_saturation)
    dtype = sw.dtype.type
    oil_water_entry_pressure_water_wet = dtype(oil_water_entry_pressure_water_wet)
    oil_water_entry_pressure_oil_wet = dtype(oil_water_entry_pressure_oil_wet)
    oil_water_pore_size_distribution_index_water_wet = dtype(
        oil_water_pore_size_distribution_index_water_wet
    )
    oil_water_pore_size_distribution_index_oil_wet = dtype(
        oil_water_pore_size_distribution_index_oil_wet
    )
    gas_oil_entry_pressure = dtype(gas_oil_entry_pressure)
    gas_oil_pore_size_distribution_index = dtype(gas_oil_pore_size_distribution_index)
    one = dtype(1.0)
    zero = dtype(0.0)

    # Broadcast all arrays to same shape
    sw, so, sg, swc, sorw, sorg, sgr = np.broadcast_arrays(sw, so, sg, swc, sorw, sorg, sgr)
    # Validate saturations
    if np.any((sw < 0) | (sw > 1) | (so < 0) | (so > 1) | (sg < 0) | (sg > 1)):
        raise ValidationError("Saturations must be between 0 and 1.")

    # Normalize saturations if they do not sum to 1
    total_saturation = sw + so + sg
    needs_norm = (np.abs(total_saturation - 1.0) > saturation_epsilon) & (total_saturation > 0.0)
    if np.any(needs_norm):
        sw = np.where(needs_norm, sw / total_saturation, sw)
        so = np.where(needs_norm, so / total_saturation, so)
        sg = np.where(needs_norm, sg / total_saturation, sg)

    # Effective pore spaces
    total_mobile_pore_space_water = one - swc - sorw
    total_mobile_pore_space_gas = one - swc - sorg - sgr

    # Pcow (Po - Pw)
    oil_water_capillary_pressure = np.zeros_like(sw)
    valid_water = total_mobile_pore_space_water > minimum_mobile_pore_space

    if np.any(valid_water):
        effective_water_saturation = np.where(
            valid_water, (sw - swc) / total_mobile_pore_space_water, zero
        )
        np.clip(
            effective_water_saturation,
            saturation_epsilon,
            one,
            out=effective_water_saturation,
        )
        undersaturated = valid_water & (effective_water_saturation < one - saturation_epsilon)

        if np.any(undersaturated):
            if wettability == Wettability.WATER_WET:
                pcow = oil_water_entry_pressure_water_wet * (
                    effective_water_saturation
                    ** (-one / oil_water_pore_size_distribution_index_water_wet)
                )
                oil_water_capillary_pressure = np.where(undersaturated, pcow, zero)

            elif wettability == Wettability.OIL_WET:
                pcow = -(
                    oil_water_entry_pressure_oil_wet
                    * effective_water_saturation
                    ** (-one / oil_water_pore_size_distribution_index_oil_wet)
                )
                oil_water_capillary_pressure = np.where(undersaturated, pcow, zero)

            elif wettability == Wettability.MIXED_WET:
                pcow_ww = oil_water_entry_pressure_water_wet * (
                    effective_water_saturation
                    ** (-one / oil_water_pore_size_distribution_index_water_wet)
                )
                pcow_ow = -(
                    oil_water_entry_pressure_oil_wet
                    * effective_water_saturation
                    ** (-one / oil_water_pore_size_distribution_index_oil_wet)
                )
                pcow = (
                    mixed_wet_water_fraction * pcow_ww + (one - mixed_wet_water_fraction) * pcow_ow
                )
                oil_water_capillary_pressure = np.where(undersaturated, pcow, zero)

    # Pcgo (Pg - Po)
    gas_oil_capillary_pressure = np.zeros_like(sg)
    valid_gas = total_mobile_pore_space_gas > minimum_mobile_pore_space

    if np.any(valid_gas):
        effective_gas_saturation = np.where(
            valid_gas, (sg - sgr) / total_mobile_pore_space_gas, zero
        )
        np.clip(
            effective_gas_saturation,
            saturation_epsilon,
            one,
            out=effective_gas_saturation,
        )
        is_undersaturated_gas = valid_gas & (effective_gas_saturation < one - saturation_epsilon)
        if np.any(is_undersaturated_gas):
            pcgo = gas_oil_entry_pressure * (
                effective_gas_saturation ** (-one / gas_oil_pore_size_distribution_index)
            )
            gas_oil_capillary_pressure = np.where(is_undersaturated_gas, pcgo, zero)

    return typing.cast(NumberArray[NDimension], oil_water_capillary_pressure), typing.cast(
        NumberArray[NDimension], gas_oil_capillary_pressure
    )


def compute_brooks_corey_capillary_pressures(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    wettability: Wettability,
    oil_water_entry_pressure_water_wet: Number,
    oil_water_entry_pressure_oil_wet: Number,
    oil_water_pore_size_distribution_index_water_wet: Number,
    oil_water_pore_size_distribution_index_oil_wet: Number,
    gas_oil_entry_pressure: Number,
    gas_oil_pore_size_distribution_index: Number,
    mixed_wet_water_fraction: Number = 0.5,
    saturation_epsilon: Number = 1e-12,
    minimum_mobile_pore_space: Number = 1e-9,
) -> tuple[NumberOrArray[NDimension], NumberOrArray[NDimension]]:
    """
    Dispatch function for Brooks-Corey capillary pressure computation.

    Routes to the scalar variant when all inputs are Python scalars, otherwise
    routes to the array variant. This avoids Numba union return-type limitations.

    :return: (Pcow, Pcgo) as (Number, Number) for scalar inputs,
             or (NDArray, NDArray) for array inputs, both in psi.
    """
    if (
        np.isscalar(water_saturation)
        and np.isscalar(oil_saturation)
        and np.isscalar(gas_saturation)
        and np.isscalar(irreducible_water_saturation)
        and np.isscalar(residual_oil_saturation_water)
        and np.isscalar(residual_oil_saturation_gas)
        and np.isscalar(residual_gas_saturation)
    ):
        return compute_brooks_corey_capillary_pressures_scalar(
            water_saturation=water_saturation,  # type: ignore[arg-type]
            oil_saturation=oil_saturation,  # type: ignore[arg-type]
            gas_saturation=gas_saturation,  # type: ignore[arg-type]
            irreducible_water_saturation=irreducible_water_saturation,  # type: ignore[arg-type]
            residual_oil_saturation_water=residual_oil_saturation_water,  # type: ignore[arg-type]
            residual_oil_saturation_gas=residual_oil_saturation_gas,  # type: ignore[arg-type]
            residual_gas_saturation=residual_gas_saturation,  # type: ignore[arg-type]
            wettability=wettability,
            oil_water_entry_pressure_water_wet=oil_water_entry_pressure_water_wet,
            oil_water_entry_pressure_oil_wet=oil_water_entry_pressure_oil_wet,
            oil_water_pore_size_distribution_index_water_wet=oil_water_pore_size_distribution_index_water_wet,
            oil_water_pore_size_distribution_index_oil_wet=oil_water_pore_size_distribution_index_oil_wet,
            gas_oil_entry_pressure=gas_oil_entry_pressure,
            gas_oil_pore_size_distribution_index=gas_oil_pore_size_distribution_index,
            mixed_wet_water_fraction=mixed_wet_water_fraction,
            saturation_epsilon=saturation_epsilon,
            minimum_mobile_pore_space=minimum_mobile_pore_space,
        )
    return compute_brooks_corey_capillary_pressures_array(
        water_saturation=water_saturation,  # type: ignore[arg-type]
        oil_saturation=oil_saturation,  # type: ignore[arg-type]
        gas_saturation=gas_saturation,  # type: ignore[arg-type]
        irreducible_water_saturation=irreducible_water_saturation,  # type: ignore[arg-type]
        residual_oil_saturation_water=residual_oil_saturation_water,  # type: ignore[arg-type]
        residual_oil_saturation_gas=residual_oil_saturation_gas,  # type: ignore[arg-type]
        residual_gas_saturation=residual_gas_saturation,  # type: ignore[arg-type]
        wettability=wettability,
        oil_water_entry_pressure_water_wet=oil_water_entry_pressure_water_wet,
        oil_water_entry_pressure_oil_wet=oil_water_entry_pressure_oil_wet,
        oil_water_pore_size_distribution_index_water_wet=oil_water_pore_size_distribution_index_water_wet,
        oil_water_pore_size_distribution_index_oil_wet=oil_water_pore_size_distribution_index_oil_wet,
        gas_oil_entry_pressure=gas_oil_entry_pressure,
        gas_oil_pore_size_distribution_index=gas_oil_pore_size_distribution_index,
        mixed_wet_water_fraction=mixed_wet_water_fraction,
        saturation_epsilon=saturation_epsilon,
        minimum_mobile_pore_space=minimum_mobile_pore_space,
    )


@numba.njit(cache=True)
def compute_brooks_corey_derivatives_scalar(
    water_saturation: Number,
    gas_saturation: Number,
    irreducible_water_saturation: Number,
    residual_oil_saturation_water: Number,
    residual_oil_saturation_gas: Number,
    residual_gas_saturation: Number,
    wettability: Wettability,
    oil_water_pore_size_distribution_index_water_wet: Number,
    oil_water_pore_size_distribution_index_oil_wet: Number,
    oil_water_entry_pressure_water_wet: Number,
    oil_water_entry_pressure_oil_wet: Number,
    gas_oil_pore_size_distribution_index: Number,
    gas_oil_entry_pressure: Number,
    mixed_wet_water_fraction: Number = 0.5,
    saturation_epsilon: Number = 1e-12,
    minimum_mobile_pore_space: Number = 1e-9,
) -> tuple[Number, Number, Number, Number]:
    """
    Scalar variant of Brooks-Corey capillary pressure derivatives.

    Returns (dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso).

    :param water_saturation: Water saturation (fraction, 0-1).
    :param gas_saturation: Gas saturation (fraction, 0-1).
    :param irreducible_water_saturation: Irreducible water saturation (swc).
    :param residual_oil_saturation_water: Residual oil saturation during water flooding (sorw).
    :param residual_oil_saturation_gas: Residual oil saturation during gas flooding (sorg).
    :param residual_gas_saturation: Residual gas saturation (sgr).
    :param wettability: Wettability type.
    :param oil_water_pore_size_distribution_index_water_wet: λ for oil-water (water-wet).
    :param oil_water_pore_size_distribution_index_oil_wet: λ for oil-water (oil-wet).
    :param oil_water_entry_pressure_water_wet: Entry pressure for oil-water (water-wet) in psi.
    :param oil_water_entry_pressure_oil_wet: Entry pressure for oil-water (oil-wet) in psi.
    :param gas_oil_pore_size_distribution_index: λ for gas-oil.
    :param gas_oil_entry_pressure: Entry pressure for gas-oil in psi.
    :param mixed_wet_water_fraction: Fraction of pore space that is water-wet (0-1).
    :param saturation_epsilon: Small value to avoid division by zero.
    :param minimum_mobile_pore_space: Minimum mobile pore space threshold.
    :return: Tuple of (dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso).
    """
    sw = water_saturation
    sg = gas_saturation
    swc = irreducible_water_saturation
    sorw = residual_oil_saturation_water
    sorg = residual_oil_saturation_gas
    sgr = residual_gas_saturation

    zero = 0.0
    one = 1.0

    # Oil-water derivatives
    mobile_water_range = one - swc - sorw
    se_w = min(max((sw - swc) / mobile_water_range, saturation_epsilon), one)
    valid_water = mobile_water_range > minimum_mobile_pore_space

    dpcow_dsw = zero
    if valid_water and se_w < one - saturation_epsilon:
        if wettability == Wettability.MIXED_WET:
            exp_ww = -one / oil_water_pore_size_distribution_index_water_wet
            exp_ow = -one / oil_water_pore_size_distribution_index_oil_wet
            dse_w_ww = oil_water_entry_pressure_water_wet * exp_ww * (se_w ** (exp_ww - one))
            dse_w_ow = -(oil_water_entry_pressure_oil_wet * exp_ow * (se_w ** (exp_ow - one)))
            dpcow_dse_w = (
                mixed_wet_water_fraction * dse_w_ww + (one - mixed_wet_water_fraction) * dse_w_ow
            )
        else:
            if wettability == Wettability.WATER_WET:
                pore_distribution_index = oil_water_pore_size_distribution_index_water_wet
                entry_pressure = oil_water_entry_pressure_water_wet
                sign = one
            else:  # OIL_WET
                pore_distribution_index = oil_water_pore_size_distribution_index_oil_wet
                entry_pressure = oil_water_entry_pressure_oil_wet
                sign = -one
            exp = -one / pore_distribution_index
            dpcow_dse_w = sign * entry_pressure * exp * (se_w ** (exp - one))

        dpcow_dsw = dpcow_dse_w / mobile_water_range

    dpcow_dso = zero

    # Gas-oil derivatives
    mobile_gas_range = one - swc - sorg - sgr
    se_g = min(max((sg - sgr) / mobile_gas_range, saturation_epsilon), one)
    valid_gas = mobile_gas_range > minimum_mobile_pore_space

    dpcgo_dsg = zero
    if valid_gas:
        exp_gas_oil = -one / gas_oil_pore_size_distribution_index
        dpcgo_dse_g = gas_oil_entry_pressure * exp_gas_oil * (se_g ** (exp_gas_oil - one))
        dpcgo_dsg = dpcgo_dse_g / mobile_gas_range

    dpcgo_dso = zero

    return dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso


@numba.njit(cache=True)
def compute_brooks_corey_derivatives_array(
    water_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    wettability: Wettability,
    oil_water_pore_size_distribution_index_water_wet: Number,
    oil_water_pore_size_distribution_index_oil_wet: Number,
    oil_water_entry_pressure_water_wet: Number,
    oil_water_entry_pressure_oil_wet: Number,
    gas_oil_pore_size_distribution_index: Number,
    gas_oil_entry_pressure: Number,
    mixed_wet_water_fraction: Number = 0.5,
    saturation_epsilon: Number = 1e-12,
    minimum_mobile_pore_space: Number = 1e-9,
) -> tuple[
    NumberArray[NDimension],
    NumberArray[NDimension],
    NumberArray[NDimension],
    NumberArray[NDimension],
]:
    """
    Array variant of Brooks-Corey capillary pressure derivatives.

    Returns (dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso) as NDArrays.

    :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
    :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
    :param irreducible_water_saturation: Irreducible water saturation (swc) - scalar or array.
    :param residual_oil_saturation_water: Residual oil saturation during water flooding (sorw) - scalar or array.
    :param residual_oil_saturation_gas: Residual oil saturation during gas flooding (sorg) - scalar or array.
    :param residual_gas_saturation: Residual gas saturation (sgr) - scalar or array.
    :param wettability: Wettability type.
    :param oil_water_pore_size_distribution_index_water_wet: λ for oil-water (water-wet).
    :param oil_water_pore_size_distribution_index_oil_wet: λ for oil-water (oil-wet).
    :param oil_water_entry_pressure_water_wet: Entry pressure for oil-water (water-wet) in psi.
    :param oil_water_entry_pressure_oil_wet: Entry pressure for oil-water (oil-wet) in psi.
    :param gas_oil_pore_size_distribution_index: λ for gas-oil.
    :param gas_oil_entry_pressure: Entry pressure for gas-oil in psi.
    :param mixed_wet_water_fraction: Fraction of pore space that is water-wet (0-1).
    :param saturation_epsilon: Small value to avoid division by zero.
    :param minimum_mobile_pore_space: Minimum mobile pore space threshold.
    :return: Tuple of (dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso) as NDArrays.
    """
    sw = atleast_1d(water_saturation)
    sg = atleast_1d(gas_saturation)
    swc = atleast_1d(irreducible_water_saturation)
    sorw = atleast_1d(residual_oil_saturation_water)
    sorg = atleast_1d(residual_oil_saturation_gas)
    sgr = atleast_1d(residual_gas_saturation)

    dtype = sw.dtype.type
    one = dtype(1.0)

    # Broadcast arrays
    sw, sg, swc, sorw, sorg, sgr = np.broadcast_arrays(sw, sg, swc, sorw, sorg, sgr)

    # Oil-water derivatives
    mobile_water_range = one - swc - sorw
    valid_water = mobile_water_range > minimum_mobile_pore_space
    se_w = np.clip(
        (sw - swc) / np.where(valid_water, mobile_water_range, one),
        saturation_epsilon,
        one,
    )

    if wettability == Wettability.MIXED_WET:
        water_wet_fraction = dtype(mixed_wet_water_fraction)
        exp_ww = -one / dtype(oil_water_pore_size_distribution_index_water_wet)
        exp_ow = -one / dtype(oil_water_pore_size_distribution_index_oil_wet)
        dse_w_ww = dtype(oil_water_entry_pressure_water_wet) * exp_ww * (se_w ** (exp_ww - one))
        dse_w_ow = -(dtype(oil_water_entry_pressure_oil_wet) * exp_ow * (se_w ** (exp_ow - one)))
        dpcow_dse_w = water_wet_fraction * dse_w_ww + (one - water_wet_fraction) * dse_w_ow
    else:
        if wettability == Wettability.WATER_WET:
            pore_distribution_index = dtype(oil_water_pore_size_distribution_index_water_wet)
            entry_pressure = dtype(oil_water_entry_pressure_water_wet)
            sign = one
        else:  # OIL_WET
            pore_distribution_index = dtype(oil_water_pore_size_distribution_index_oil_wet)
            entry_pressure = dtype(oil_water_entry_pressure_oil_wet)
            sign = -one
        exp = -one / pore_distribution_index
        dpcow_dse_w = sign * entry_pressure * exp * (se_w ** (exp - one))

    dpcow_dsw = np.where(
        valid_water & (se_w < one - saturation_epsilon),
        dpcow_dse_w / mobile_water_range,
        np.zeros_like(sw),
    )
    dpcow_dso = np.zeros_like(sw)

    # Gas-oil derivatives
    mobile_gas_range = one - swc - sorg - sgr
    valid_gas = mobile_gas_range > minimum_mobile_pore_space
    se_g = np.clip(
        (sg - sgr) / np.where(valid_gas, mobile_gas_range, one),
        saturation_epsilon,
        one,
    )

    exp_gas_oil = -one / dtype(gas_oil_pore_size_distribution_index)
    dpcgo_dse_g = dtype(gas_oil_entry_pressure) * exp_gas_oil * (se_g ** (exp_gas_oil - one))
    dpcgo_dsg = np.where(
        valid_gas,
        dpcgo_dse_g / mobile_gas_range,
        np.zeros_like(sg),
    )
    dpcgo_dso = np.zeros_like(sg)
    return (
        typing.cast(NumberArray[NDimension], dpcow_dsw),
        typing.cast(NumberArray[NDimension], dpcow_dso),
        typing.cast(NumberArray[NDimension], dpcgo_dsg),
        typing.cast(NumberArray[NDimension], dpcgo_dso),
    )


def compute_brooks_corey_capillary_pressure_derivatives(
    water_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    wettability: Wettability,
    oil_water_pore_size_distribution_index_water_wet: Number,
    oil_water_pore_size_distribution_index_oil_wet: Number,
    oil_water_entry_pressure_water_wet: Number,
    oil_water_entry_pressure_oil_wet: Number,
    gas_oil_pore_size_distribution_index: Number,
    gas_oil_entry_pressure: Number,
    mixed_wet_water_fraction: Number = 0.5,
    saturation_epsilon: Number = 1e-12,
    minimum_mobile_pore_space: Number = 1e-9,
) -> tuple[
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
]:
    """
    Dispatch function for Brooks-Corey capillary pressure derivatives.

    Routes to the scalar variant when all inputs are Python scalars, otherwise
    routes to the array variant.

    :return: (dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso) as scalars or arrays.
    """
    if (
        np.isscalar(water_saturation)
        and np.isscalar(gas_saturation)
        and np.isscalar(irreducible_water_saturation)
        and np.isscalar(residual_oil_saturation_water)
        and np.isscalar(residual_oil_saturation_gas)
        and np.isscalar(residual_gas_saturation)
    ):
        return compute_brooks_corey_derivatives_scalar(
            water_saturation=water_saturation,  # type: ignore[arg-type]
            gas_saturation=gas_saturation,  # type: ignore[arg-type]
            irreducible_water_saturation=irreducible_water_saturation,  # type: ignore[arg-type]
            residual_oil_saturation_water=residual_oil_saturation_water,  # type: ignore[arg-type]
            residual_oil_saturation_gas=residual_oil_saturation_gas,  # type: ignore[arg-type]
            residual_gas_saturation=residual_gas_saturation,  # type: ignore[arg-type]
            wettability=wettability,
            oil_water_pore_size_distribution_index_water_wet=oil_water_pore_size_distribution_index_water_wet,
            oil_water_pore_size_distribution_index_oil_wet=oil_water_pore_size_distribution_index_oil_wet,
            oil_water_entry_pressure_water_wet=oil_water_entry_pressure_water_wet,
            oil_water_entry_pressure_oil_wet=oil_water_entry_pressure_oil_wet,
            gas_oil_pore_size_distribution_index=gas_oil_pore_size_distribution_index,
            gas_oil_entry_pressure=gas_oil_entry_pressure,
            mixed_wet_water_fraction=mixed_wet_water_fraction,
            saturation_epsilon=saturation_epsilon,
            minimum_mobile_pore_space=minimum_mobile_pore_space,
        )
    return compute_brooks_corey_derivatives_array(
        water_saturation=water_saturation,  # type: ignore[arg-type]
        gas_saturation=gas_saturation,  # type: ignore[arg-type]
        irreducible_water_saturation=irreducible_water_saturation,  # type: ignore[arg-type]
        residual_oil_saturation_water=residual_oil_saturation_water,  # type: ignore[arg-type]
        residual_oil_saturation_gas=residual_oil_saturation_gas,  # type: ignore[arg-type]
        residual_gas_saturation=residual_gas_saturation,  # type: ignore[arg-type]
        wettability=wettability,
        oil_water_pore_size_distribution_index_water_wet=oil_water_pore_size_distribution_index_water_wet,
        oil_water_pore_size_distribution_index_oil_wet=oil_water_pore_size_distribution_index_oil_wet,
        oil_water_entry_pressure_water_wet=oil_water_entry_pressure_water_wet,
        oil_water_entry_pressure_oil_wet=oil_water_entry_pressure_oil_wet,
        gas_oil_pore_size_distribution_index=gas_oil_pore_size_distribution_index,
        gas_oil_entry_pressure=gas_oil_entry_pressure,
        mixed_wet_water_fraction=mixed_wet_water_fraction,
        saturation_epsilon=saturation_epsilon,
        minimum_mobile_pore_space=minimum_mobile_pore_space,
    )


@attrs.frozen(slots=True)
class BrooksCoreyCapillaryPressureTable(CapillaryPressureTable):
    """
    Implements the Brooks-Corey capillary pressure model for three-phase systems.

    Implements the Brooks-Corey model: Pc = Pd * (Se)^(-1/λ)

    Supports water-wet, oil-wet, and mixed-wet systems.
    """

    irreducible_water_saturation: Number | None = None
    """Default irreducible water saturation (swc). Can be overridden per call."""

    residual_oil_saturation_water: Number | None = None
    """Default residual oil saturation after water flood (sorw). Can be overridden per call."""

    residual_oil_saturation_gas: Number | None = None
    """Default residual oil saturation after gas flood (sorg). Can be overridden per call."""

    residual_gas_saturation: Number | None = None
    """Default residual gas saturation (sgr). Can be overridden per call."""

    oil_water_entry_pressure_water_wet: Number = 5.0
    """Entry pressure for oil-water in water-wet system. Units: [pressure] in `unit_system`."""

    oil_water_entry_pressure_oil_wet: Number = 5.0
    """Entry pressure for oil-water in oil-wet system. Units: [pressure] in `unit_system`."""

    oil_water_pore_size_distribution_index_water_wet: Number = 2.0
    """Pore size distribution index (λ) for oil-water in water-wet system."""

    oil_water_pore_size_distribution_index_oil_wet: Number = 2.0
    """Pore size distribution index (λ) for oil-water in oil-wet system."""

    gas_oil_entry_pressure: Number = 1.0
    """Entry pressure for gas-oil. Units: [pressure] in `unit_system`."""

    gas_oil_pore_size_distribution_index: Number = 2.0
    """Pore size distribution index (λ) for gas-oil."""

    wettability: Wettability = Wettability.WATER_WET
    """Wettability type (`WATER_WET`, `OIL_WET`, or `MIXED_WET`)."""

    mixed_wet_water_fraction: Number = 0.5
    """Fraction of pore space that is water-wet in mixed-wet systems (0-1)."""

    unit_system: UnitSystem = attrs.field(default=UnitSystem.FIELD)
    """
    Unit system in which the pressure parameters of this model are expressed.

    All entry pressures (`oil_water_entry_pressure_water_wet`,
    `oil_water_entry_pressure_oil_wet`, `gas_oil_entry_pressure`) and the
    resulting capillary pressure outputs are in this unit system.
    Saturation parameters, pore-size distribution indices, and
    dimensionless ratios are unaffected by unit conversion.

    Use `convert(target)` to produce a copy of this model rescaled to
    another `UnitSystem`.
    """

    supports_vector: bool = attrs.field(init=False, repr=False, default=True)
    """Flag indicating support for array inputs."""

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
    ) -> CapillaryPressures:
        """
        Compute capillary pressures using Brooks-Corey model.

        Supports both scalar and array inputs.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param irreducible_water_saturation: Optional override for swc - scalar or array.
        :param residual_oil_saturation_water: Optional override for sorw - scalar or array.
        :param residual_oil_saturation_gas: Optional override for sorg - scalar or array.
        :param residual_gas_saturation: Optional override for sgr - scalar or array.
        :return: `CapillaryPressures` dictionary.
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
                f"Residual saturations must be provided either as model defaults or in the call. "
                f"Missing: {', '.join(params_missing)}"
            )

        pcow, pcgo = compute_brooks_corey_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=sorg,  # type: ignore[arg-type]
            residual_gas_saturation=sgr,  # type: ignore[arg-type]
            wettability=self.wettability,
            oil_water_entry_pressure_water_wet=self.oil_water_entry_pressure_water_wet,
            oil_water_entry_pressure_oil_wet=self.oil_water_entry_pressure_oil_wet,
            oil_water_pore_size_distribution_index_water_wet=self.oil_water_pore_size_distribution_index_water_wet,
            oil_water_pore_size_distribution_index_oil_wet=self.oil_water_pore_size_distribution_index_oil_wet,
            gas_oil_entry_pressure=self.gas_oil_entry_pressure,
            gas_oil_pore_size_distribution_index=self.gas_oil_pore_size_distribution_index,
            mixed_wet_water_fraction=self.mixed_wet_water_fraction,
            saturation_epsilon=c.SATURATION_EPSILON,
            minimum_mobile_pore_space=c.MINIMUM_MOBILE_PORE_SPACE,
        )
        return CapillaryPressures(oil_water=pcow, gas_oil=pcgo)  # type: ignore[typeddict-item]

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
    ) -> CapillaryPressureDerivatives:
        """
        Compute the partial derivatives of the Brooks-Corey oil-water and gas-oil
        capillary pressures with respect to saturation.

        Returns a dictionary contianing:
        ```
        (dpcow / dsw, dpcow / dso, dpcgo / dsg)
        ```

        - `dpcow/dsw`: analytically derived from the Brooks-Corey power law via
        the chain rule through effective water saturation.
        - `dpcow/dso`: zero for this model - the oil-water capillary pressure is
        a function of water saturation only, regardless of wettability.
        - `dpcgo/dsg`: analytically derived via the chain rule through effective
        gas saturation.

        The Brooks-Corey capillary pressure formulae are:

        ```
        Pcow = Pd_ow * Se_w ^ (-1 / lambda_ow)(water - wet)
        Pcow = -Pd_ow * Se_w ^ (-1 / lambda_ow)(oil - wet)
        Pcgo = Pd_go * Se_g ^ (-1 / lambda_go)
        ```

        where effective water saturation:
        ```
        Se_w = (Sw - irreducible water saturation)
            / (1 - irreducible water saturation
                    - residual oil saturation to water flooding)
        ```

        and effective gas saturation:

        ```
        Se_g = (Sg - residual gas saturation)
            / (1 - irreducible water saturation
                    - residual gas saturation
                    - residual oil saturation to gas flooding)
        ```

        `Pd` is the displacement entry pressure and `lambda` is the pore
        size distribution index.

        :param water_saturation: Water saturation (scalar or array).
        :param oil_saturation: Oil saturation (scalar or array, not used by
            this model but included for API consistency).
        :param gas_saturation: Gas saturation (scalar or array).
        :param irreducible_water_saturation: Optional override for the
            irreducible (connate) water saturation.
        :param residual_oil_saturation_water: Optional override for the residual
            oil saturation to water flooding.
        :param residual_oil_saturation_gas: Optional override for the residual
            oil saturation to gas flooding.
        :param residual_gas_saturation: Optional override for the residual gas
            saturation.
        :return: `CapillaryPressureDerivatives` dictionary containing the partial derivatives as described above.
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
                f"Residual saturations must be provided either as model defaults or in the call. "
                f"Missing: {', '.join(params_missing)}"
            )

        dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso = (
            compute_brooks_corey_capillary_pressure_derivatives(
                water_saturation=water_saturation,
                gas_saturation=gas_saturation,
                irreducible_water_saturation=swc,  # type: ignore[arg-type]
                residual_oil_saturation_water=sorw,  # type: ignore[arg-type]
                residual_oil_saturation_gas=sorg,  # type: ignore[arg-type]
                residual_gas_saturation=sgr,  # type: ignore[arg-type]
                wettability=self.wettability,
                oil_water_pore_size_distribution_index_water_wet=self.oil_water_pore_size_distribution_index_water_wet,
                oil_water_pore_size_distribution_index_oil_wet=self.oil_water_pore_size_distribution_index_oil_wet,
                oil_water_entry_pressure_water_wet=self.oil_water_entry_pressure_water_wet,
                oil_water_entry_pressure_oil_wet=self.oil_water_entry_pressure_oil_wet,
                gas_oil_pore_size_distribution_index=self.gas_oil_pore_size_distribution_index,
                gas_oil_entry_pressure=self.gas_oil_entry_pressure,
                mixed_wet_water_fraction=self.mixed_wet_water_fraction,
                saturation_epsilon=c.SATURATION_EPSILON,
                minimum_mobile_pore_space=c.MINIMUM_MOBILE_PORE_SPACE,
            )
        )
        return CapillaryPressureDerivatives(
            dpcow_dsw=dpcow_dsw,
            dpcow_dso=dpcow_dso,
            dpcgo_dsg=dpcgo_dsg,
            dpcgo_dso=dpcgo_dso,
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `BrooksCoreyCapillaryPressureTable` with all pressure
        parameters rescaled to *target*.

        Entry pressures (`oil_water_entry_pressure_water_wet`,
        `oil_water_entry_pressure_oil_wet`, `gas_oil_entry_pressure`) are
        multiplied by the pressure conversion factor from `self.unit_system`
        to *target*.  Saturation parameters, pore-size distribution indices,
        wettability, and `mixed_wet_water_fraction` are dimensionless and
        are copied unchanged.

        :param target: Target `UnitSystem`.
        :returns: New `BrooksCoreyCapillaryPressureTable` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        pressure_factor = factors["pressure"]
        return attrs.evolve(
            self,
            oil_water_entry_pressure_water_wet=self.oil_water_entry_pressure_water_wet
            * pressure_factor,
            oil_water_entry_pressure_oil_wet=self.oil_water_entry_pressure_oil_wet
            * pressure_factor,
            gas_oil_entry_pressure=self.gas_oil_entry_pressure * pressure_factor,
            unit_system=target,
        )


@numba.njit(cache=True)
def compute_van_genuchten_capillary_pressures_scalar(
    water_saturation: Number,
    oil_saturation: Number,
    gas_saturation: Number,
    irreducible_water_saturation: Number,
    residual_oil_saturation_water: Number,
    residual_oil_saturation_gas: Number,
    residual_gas_saturation: Number,
    wettability: Wettability,
    oil_water_alpha_water_wet: Number,
    oil_water_alpha_oil_wet: Number,
    oil_water_n_water_wet: Number,
    oil_water_n_oil_wet: Number,
    gas_oil_alpha: Number,
    gas_oil_n: Number,
    mixed_wet_water_fraction: Number = 0.5,
    saturation_epsilon: Number = 1e-6,
    minimum_mobile_pore_space: Number = 1e-9,
) -> tuple[Number, Number]:
    """
    Scalar variant of van Genuchten capillary pressure computation.

    Computes capillary pressures using van Genuchten model.

    van Genuchten model: Pc = (1/α) * [(Se^(-1/m) - 1)^(1/n)]
    where m = 1 - 1/n

    :param water_saturation: Current water saturation (fraction, 0-1).
    :param oil_saturation: Current oil saturation (fraction, 0-1).
    :param gas_saturation: Current gas saturation (fraction, 0-1).
    :param irreducible_water_saturation: Irreducible water saturation (swc).
    :param residual_oil_saturation_water: Residual oil saturation during water flooding (sorw).
    :param residual_oil_saturation_gas: Residual oil saturation during gas flooding (sorg).
    :param residual_gas_saturation: Residual gas saturation (sgr).
    :param wettability: Wettability type (WATER_WET, OIL_WET, or MIXED_WET).
    :param oil_water_alpha_water_wet: van Genuchten α parameter for oil-water (water-wet) [1/psi].
    :param oil_water_alpha_oil_wet: van Genuchten α parameter for oil-water (oil-wet) [1/psi].
    :param oil_water_n_water_wet: van Genuchten n parameter for oil-water (water-wet).
    :param oil_water_n_oil_wet: van Genuchten n parameter for oil-water (oil-wet).
    :param gas_oil_alpha: van Genuchten α parameter for gas-oil [1/psi].
    :param gas_oil_n: van Genuchten n parameter for gas-oil.
    :param mixed_wet_water_fraction: Fraction of pore space that is water-wet (0-1, default 0.5).
    :param saturation_epsilon: Small value to avoid division by zero in saturation calculations.
    :param minimum_mobile_pore_space: Minimum mobile pore space threshold below which Pc is set to zero.
    :return: Tuple of (oil_water_capillary_pressure, gas_oil_capillary_pressure) in psi.
    """
    if oil_water_alpha_water_wet <= 0.0 or oil_water_alpha_oil_wet <= 0.0:
        raise ValidationError("Oil-water alpha parameters must be positive.")
    if gas_oil_alpha <= 0.0:
        raise ValidationError("Gas-oil alpha parameter must be positive.")
    if oil_water_n_water_wet <= 1.0 or oil_water_n_oil_wet <= 1.0:
        raise ValidationError("Oil-water n parameters must be greater than 1.")
    if gas_oil_n <= 1.0:
        raise ValidationError("Gas-oil n parameter must be greater than 1.")

    sw: Number = water_saturation
    so: Number = oil_saturation
    sg: Number = gas_saturation
    swc: Number = irreducible_water_saturation
    sorw: Number = residual_oil_saturation_water
    sorg: Number = residual_oil_saturation_gas
    sgr: Number = residual_gas_saturation

    if sw < 0 or sw > 1 or so < 0 or so > 1 or sg < 0 or sg > 1:
        raise ValidationError("Saturations must be between 0 and 1.")

    # Normalize saturations if they do not sum to 1
    total_saturation = sw + so + sg
    if abs(total_saturation - 1.0) > saturation_epsilon and total_saturation > 0.0:
        sw = sw / total_saturation
        sg = sg / total_saturation

    # Effective pore spaces
    total_mobile_pore_space_water = 1.0 - swc - sorw
    total_mobile_pore_space_gas = 1.0 - swc - sorg - sgr

    # Pcow (Po - Pw)
    oil_water_capillary_pressure: Number = 0.0
    if total_mobile_pore_space_water > minimum_mobile_pore_space:
        effective_water_saturation = (sw - swc) / total_mobile_pore_space_water
        effective_water_saturation = min(
            max(effective_water_saturation, saturation_epsilon),
            1.0 - saturation_epsilon,
        )

        if wettability == Wettability.WATER_WET:
            m_ww = 1.0 - 1.0 / oil_water_n_water_wet
            term = (effective_water_saturation ** (-1.0 / m_ww) - 1.0) ** (
                1.0 / oil_water_n_water_wet
            )
            oil_water_capillary_pressure = (1.0 / oil_water_alpha_water_wet) * term

        elif wettability == Wettability.OIL_WET:
            m_ow = 1.0 - 1.0 / oil_water_n_oil_wet
            term = (effective_water_saturation ** (-1.0 / m_ow) - 1.0) ** (
                1.0 / oil_water_n_oil_wet
            )
            oil_water_capillary_pressure = -(1.0 / oil_water_alpha_oil_wet) * term

        elif wettability == Wettability.MIXED_WET:
            m_ww = 1.0 - 1.0 / oil_water_n_water_wet
            term_ww = (effective_water_saturation ** (-1.0 / m_ww) - 1.0) ** (
                1.0 / oil_water_n_water_wet
            )
            pcow_ww = (1.0 / oil_water_alpha_water_wet) * term_ww

            m_ow = 1.0 - 1.0 / oil_water_n_oil_wet
            term_ow = (effective_water_saturation ** (-1.0 / m_ow) - 1.0) ** (
                1.0 / oil_water_n_oil_wet
            )
            pcow_ow = -(1.0 / oil_water_alpha_oil_wet) * term_ow

            oil_water_capillary_pressure = (
                mixed_wet_water_fraction * pcow_ww + (1.0 - mixed_wet_water_fraction) * pcow_ow
            )

    # Pcgo (Pg - Po)
    gas_oil_capillary_pressure: Number = 0.0
    if total_mobile_pore_space_gas > minimum_mobile_pore_space:
        effective_gas_saturation = (sg - sgr) / total_mobile_pore_space_gas
        effective_gas_saturation = min(
            max(effective_gas_saturation, saturation_epsilon),
            1.0 - saturation_epsilon,
        )

        m_gas_oil = 1.0 - 1.0 / gas_oil_n
        term = (effective_gas_saturation ** (-1.0 / m_gas_oil) - 1.0) ** (1.0 / gas_oil_n)
        gas_oil_capillary_pressure = (1.0 / gas_oil_alpha) * term

    return oil_water_capillary_pressure, gas_oil_capillary_pressure


@numba.njit(cache=True)
def compute_van_genuchten_capillary_pressures_array(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    wettability: Wettability,
    oil_water_alpha_water_wet: Number,
    oil_water_alpha_oil_wet: Number,
    oil_water_n_water_wet: Number,
    oil_water_n_oil_wet: Number,
    gas_oil_alpha: Number,
    gas_oil_n: Number,
    mixed_wet_water_fraction: Number = 0.5,
    saturation_epsilon: Number = 1e-6,
    minimum_mobile_pore_space: Number = 1e-9,
) -> tuple[NumberArray[NDimension], NumberArray[NDimension]]:
    """
    Array variant of van Genuchten capillary pressure computation.

    Computes capillary pressures using van Genuchten model.
    Supports array inputs (up to 3D). Always returns NDArrays.

    van Genuchten model: Pc = (1/α) * [(Se^(-1/m) - 1)^(1/n)]
    where m = 1 - 1/n

    :param water_saturation: Current water saturation (fraction, 0-1) - scalar or array.
    :param oil_saturation: Current oil saturation (fraction, 0-1) - scalar or array.
    :param gas_saturation: Current gas saturation (fraction, 0-1) - scalar or array.
    :param irreducible_water_saturation: Irreducible water saturation (swc) - scalar or array.
    :param residual_oil_saturation_water: Residual oil saturation during water flooding (sorw) - scalar or array.
    :param residual_oil_saturation_gas: Residual oil saturation during gas flooding (sorg) - scalar or array.
    :param residual_gas_saturation: Residual gas saturation (sgr) - scalar or array.
    :param wettability: Wettability type (WATER_WET, OIL_WET, or MIXED_WET).
    :param oil_water_alpha_water_wet: van Genuchten α parameter for oil-water (water-wet) [1/psi].
    :param oil_water_alpha_oil_wet: van Genuchten α parameter for oil-water (oil-wet) [1/psi].
    :param oil_water_n_water_wet: van Genuchten n parameter for oil-water (water-wet).
    :param oil_water_n_oil_wet: van Genuchten n parameter for oil-water (oil-wet).
    :param gas_oil_alpha: van Genuchten α parameter for gas-oil [1/psi].
    :param gas_oil_n: van Genuchten n parameter for gas-oil.
    :param mixed_wet_water_fraction: Fraction of pore space that is water-wet (0-1, default 0.5).
    :param saturation_epsilon: Small value to avoid division by zero in saturation calculations.
    :param minimum_mobile_pore_space: Minimum mobile pore space threshold below which Pc is set to zero.
    :return: Tuple of (oil_water_capillary_pressure, gas_oil_capillary_pressure) NDArrays in psi.
    """
    if oil_water_alpha_water_wet <= 0.0 or oil_water_alpha_oil_wet <= 0.0:
        raise ValidationError("Oil-water alpha parameters must be positive.")
    if gas_oil_alpha <= 0.0:
        raise ValidationError("Gas-oil alpha parameter must be positive.")
    if oil_water_n_water_wet <= 1.0 or oil_water_n_oil_wet <= 1.0:
        raise ValidationError("Oil-water n parameters must be greater than 1.")
    if gas_oil_n <= 1.0:
        raise ValidationError("Gas-oil n parameter must be greater than 1.")

    sw = atleast_1d(water_saturation)
    so = atleast_1d(oil_saturation)
    sg = atleast_1d(gas_saturation)
    swc = atleast_1d(irreducible_water_saturation)
    sorw = atleast_1d(residual_oil_saturation_water)
    sorg = atleast_1d(residual_oil_saturation_gas)
    sgr = atleast_1d(residual_gas_saturation)
    dtype = sw.dtype.type
    oil_water_alpha_water_wet = dtype(oil_water_alpha_water_wet)
    oil_water_alpha_oil_wet = dtype(oil_water_alpha_oil_wet)
    oil_water_n_water_wet = dtype(oil_water_n_water_wet)
    oil_water_n_oil_wet = dtype(oil_water_n_oil_wet)
    gas_oil_alpha = dtype(gas_oil_alpha)
    gas_oil_n = dtype(gas_oil_n)
    mixed_wet_water_fraction = dtype(mixed_wet_water_fraction)
    one = dtype(1.0)
    zero = dtype(0.0)

    # Broadcast all arrays to same shape
    sw, so, sg, swc, sorw, sorg, sgr = np.broadcast_arrays(sw, so, sg, swc, sorw, sorg, sgr)

    # Validate saturations
    if np.any((sw < 0) | (sw > 1) | (so < 0) | (so > 1) | (sg < 0) | (sg > 1)):
        raise ValidationError("Saturations must be between 0 and 1.")

    # Normalize saturations if they do not sum to 1
    total_saturation = sw + so + sg
    needs_norm = (np.abs(total_saturation - 1.0) > saturation_epsilon) & (total_saturation > 0.0)
    if np.any(needs_norm):
        sw = np.where(needs_norm, sw / total_saturation, sw)
        so = np.where(needs_norm, so / total_saturation, so)
        sg = np.where(needs_norm, sg / total_saturation, sg)

    # Effective pore spaces
    total_mobile_pore_space_water = one - swc - sorw
    total_mobile_pore_space_gas = one - swc - sorg - sgr

    # Pcow (Po - Pw)
    oil_water_capillary_pressure = np.zeros_like(sw)
    valid_water = total_mobile_pore_space_water > minimum_mobile_pore_space

    if np.any(valid_water):
        effective_water_saturation = np.where(
            valid_water, (sw - swc) / total_mobile_pore_space_water, zero
        )
        effective_water_saturation = np.clip(
            effective_water_saturation, saturation_epsilon, one - saturation_epsilon
        )

        if wettability == Wettability.WATER_WET:
            m_ww = one - one / oil_water_n_water_wet
            term = (effective_water_saturation ** (-one / m_ww) - one) ** (
                one / oil_water_n_water_wet
            )
            pcow = (one / oil_water_alpha_water_wet) * term
            oil_water_capillary_pressure = np.where(valid_water, pcow, zero)

        elif wettability == Wettability.OIL_WET:
            m_ow = one - one / oil_water_n_oil_wet
            term = (effective_water_saturation ** (-one / m_ow) - one) ** (
                one / oil_water_n_oil_wet
            )
            pcow = -(one / oil_water_alpha_oil_wet) * term
            oil_water_capillary_pressure = np.where(valid_water, pcow, zero)

        elif wettability == Wettability.MIXED_WET:
            m_ww = one - one / oil_water_n_water_wet
            term_ww = (effective_water_saturation ** (-one / m_ww) - one) ** (
                one / oil_water_n_water_wet
            )
            pcow_ww = (one / oil_water_alpha_water_wet) * term_ww

            m_ow = one - one / oil_water_n_oil_wet
            term_ow = (effective_water_saturation ** (-one / m_ow) - one) ** (
                one / oil_water_n_oil_wet
            )
            pcow_ow = -(one / oil_water_alpha_oil_wet) * term_ow

            pcow = mixed_wet_water_fraction * pcow_ww + (one - mixed_wet_water_fraction) * pcow_ow
            oil_water_capillary_pressure = np.where(valid_water, pcow, zero)

    # Pcgo (Pg - Po)
    gas_oil_capillary_pressure = np.zeros_like(sg)
    valid_gas = total_mobile_pore_space_gas > minimum_mobile_pore_space

    if np.any(valid_gas):
        effective_gas_saturation = np.where(
            valid_gas, (sg - sgr) / total_mobile_pore_space_gas, zero
        )
        effective_gas_saturation = np.clip(
            effective_gas_saturation, saturation_epsilon, one - saturation_epsilon
        )

        m_gas_oil = one - one / gas_oil_n
        term = (effective_gas_saturation ** (-one / m_gas_oil) - one) ** (one / gas_oil_n)
        pcgo = (one / gas_oil_alpha) * term
        gas_oil_capillary_pressure = np.where(valid_gas, pcgo, zero)

    return typing.cast(NumberArray[NDimension], oil_water_capillary_pressure), typing.cast(
        NumberArray[NDimension], gas_oil_capillary_pressure
    )


def compute_van_genuchten_capillary_pressures(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    wettability: Wettability,
    oil_water_alpha_water_wet: Number,
    oil_water_alpha_oil_wet: Number,
    oil_water_n_water_wet: Number,
    oil_water_n_oil_wet: Number,
    gas_oil_alpha: Number,
    gas_oil_n: Number,
    mixed_wet_water_fraction: Number = 0.5,
    saturation_epsilon: Number = 1e-6,
    minimum_mobile_pore_space: Number = 1e-9,
) -> tuple[NumberOrArray[NDimension], NumberOrArray[NDimension]]:
    """
    Dispatch function for van Genuchten capillary pressure computation.

    :return: (Pcow, Pcgo) as (Number, Number) for scalar inputs,
             or (NDArray, NDArray) for array inputs, both in psi.
    """
    if (
        np.isscalar(water_saturation)
        and np.isscalar(oil_saturation)
        and np.isscalar(gas_saturation)
        and np.isscalar(irreducible_water_saturation)
        and np.isscalar(residual_oil_saturation_water)
        and np.isscalar(residual_oil_saturation_gas)
        and np.isscalar(residual_gas_saturation)
    ):
        return compute_van_genuchten_capillary_pressures_scalar(
            water_saturation=water_saturation,  # type: ignore[arg-type]
            oil_saturation=oil_saturation,  # type: ignore[arg-type]
            gas_saturation=gas_saturation,  # type: ignore[arg-type]
            irreducible_water_saturation=irreducible_water_saturation,  # type: ignore[arg-type]
            residual_oil_saturation_water=residual_oil_saturation_water,  # type: ignore[arg-type]
            residual_oil_saturation_gas=residual_oil_saturation_gas,  # type: ignore[arg-type]
            residual_gas_saturation=residual_gas_saturation,  # type: ignore[arg-type]
            wettability=wettability,
            oil_water_alpha_water_wet=oil_water_alpha_water_wet,
            oil_water_alpha_oil_wet=oil_water_alpha_oil_wet,
            oil_water_n_water_wet=oil_water_n_water_wet,
            oil_water_n_oil_wet=oil_water_n_oil_wet,
            gas_oil_alpha=gas_oil_alpha,
            gas_oil_n=gas_oil_n,
            mixed_wet_water_fraction=mixed_wet_water_fraction,
            saturation_epsilon=saturation_epsilon,
            minimum_mobile_pore_space=minimum_mobile_pore_space,
        )
    return compute_van_genuchten_capillary_pressures_array(
        water_saturation=water_saturation,  # type: ignore[arg-type]
        oil_saturation=oil_saturation,  # type: ignore[arg-type]
        gas_saturation=gas_saturation,  # type: ignore[arg-type]
        irreducible_water_saturation=irreducible_water_saturation,  # type: ignore[arg-type]
        residual_oil_saturation_water=residual_oil_saturation_water,  # type: ignore[arg-type]
        residual_oil_saturation_gas=residual_oil_saturation_gas,  # type: ignore[arg-type]
        residual_gas_saturation=residual_gas_saturation,  # type: ignore[arg-type]
        wettability=wettability,
        oil_water_alpha_water_wet=oil_water_alpha_water_wet,
        oil_water_alpha_oil_wet=oil_water_alpha_oil_wet,
        oil_water_n_water_wet=oil_water_n_water_wet,
        oil_water_n_oil_wet=oil_water_n_oil_wet,
        gas_oil_alpha=gas_oil_alpha,
        gas_oil_n=gas_oil_n,
        mixed_wet_water_fraction=mixed_wet_water_fraction,
        saturation_epsilon=saturation_epsilon,
        minimum_mobile_pore_space=minimum_mobile_pore_space,
    )


@numba.njit(cache=True)
def compute_van_genuchten_pc_slope_wrt_effective_saturation(
    effective_saturation: NumberArray[NDimension],
    alpha: Number,
    n: Number,
    sign: Number,
    saturation_epsilon: Number = 1e-6,
) -> NumberArray[NDimension]:
    """
    Analytical derivative of the van Genuchten capillary pressure with
    respect to effective (normalised) saturation.

    The van Genuchten model is:

    ```
    Pc = sign * (1/alpha) * (Se^(-1/m) - 1)^(1/n)

    where m = 1 - 1/n
    ```

    Let u = Se^(-1/m) - 1.  By the chain rule:

    ```
    dPc/dSe = (dPc/du) * (du/dSe)
            = sign*(1/alpha)*(1/n)*u^(1/n - 1) * (-1/m)*Se^(-1/m - 1)
    ```

    :param effective_saturation: Normalised saturation, clamped to
        (c.SATURATION_EPSILON, 1 - c.SATURATION_EPSILON) internally.
    :param alpha: van Genuchten alpha parameter (1/pressure, positive).
    :param n: van Genuchten n parameter (greater than 1).
    :param sign: +1 for water-wet, -1 for oil-wet oil-water capillary pressure.
    :return: Derivative array with the same shape as `effective_saturation`.
    """
    m = 1.0 - 1.0 / n
    se = np.clip(effective_saturation, saturation_epsilon, 1.0 - saturation_epsilon)
    u = se ** (-1.0 / m) - 1.0
    u_safe = np.where(u > 1e-30, u, 1e-30)
    dpc_du = sign * (1.0 / alpha) * (1.0 / n) * (u_safe ** (1.0 / n - 1.0))
    du_dse = (-1.0 / m) * (se ** (-1.0 / m - 1.0))
    return dpc_du * du_dse  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_van_genuchten_derivatives_scalar(
    water_saturation: Number,
    gas_saturation: Number,
    irreducible_water_saturation: Number,
    residual_oil_saturation_water: Number,
    residual_oil_saturation_gas: Number,
    residual_gas_saturation: Number,
    wettability: Wettability,
    oil_water_alpha_water_wet: Number,
    oil_water_alpha_oil_wet: Number,
    oil_water_n_water_wet: Number,
    oil_water_n_oil_wet: Number,
    gas_oil_alpha: Number,
    gas_oil_n: Number,
    mixed_wet_water_fraction: Number = 0.5,
    saturation_epsilon: Number = 1e-6,
    minimum_mobile_pore_space: Number = 1e-9,
) -> tuple[Number, Number, Number, Number]:
    """
    Scalar variant of van Genuchten capillary pressure derivatives.

    Returns (dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso).

    :param water_saturation: Water saturation (fraction, 0-1).
    :param gas_saturation: Gas saturation (fraction, 0-1).
    :param irreducible_water_saturation: Irreducible water saturation (swc).
    :param residual_oil_saturation_water: Residual oil saturation during water flooding (sorw).
    :param residual_oil_saturation_gas: Residual oil saturation during gas flooding (sorg).
    :param residual_gas_saturation: Residual gas saturation (sgr).
    :param wettability: Wettability type.
    :param oil_water_alpha_water_wet: van Genuchten α for oil-water (water-wet) [1/psi].
    :param oil_water_alpha_oil_wet: van Genuchten α for oil-water (oil-wet) [1/psi].
    :param oil_water_n_water_wet: van Genuchten n for oil-water (water-wet).
    :param oil_water_n_oil_wet: van Genuchten n for oil-water (oil-wet).
    :param gas_oil_alpha: van Genuchten α for gas-oil [1/psi].
    :param gas_oil_n: van Genuchten n for gas-oil.
    :param mixed_wet_water_fraction: Fraction of pore space that is water-wet (0-1).
    :param saturation_epsilon: Small value to avoid division by zero.
    :param minimum_mobile_pore_space: Minimum mobile pore space threshold.
    :return: Tuple of (dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso).
    """
    sw = water_saturation
    sg = gas_saturation
    swc = irreducible_water_saturation
    sorw = residual_oil_saturation_water
    sorg = residual_oil_saturation_gas
    sgr = residual_gas_saturation

    zero = 0.0
    one = 1.0

    # Oil-water derivatives
    mobile_water_range = one - swc - sorw
    valid_water = mobile_water_range > minimum_mobile_pore_space
    se_w = min(
        max((sw - swc) / mobile_water_range, saturation_epsilon),
        one - saturation_epsilon,
    )

    dpcow_dsw = zero
    if valid_water:
        if wettability == Wettability.WATER_WET:
            m_ww = one - one / oil_water_n_water_wet
            u = se_w ** (-one / m_ww) - one
            u_safe = max(u, 1e-30)
            dpc_du = (
                (one / oil_water_alpha_water_wet)
                * (one / oil_water_n_water_wet)
                * (u_safe ** (one / oil_water_n_water_wet - one))
            )
            du_dse = (-one / m_ww) * (se_w ** (-one / m_ww - one))
            dpcow_dse_w = dpc_du * du_dse
        elif wettability == Wettability.OIL_WET:
            m_ow = one - one / oil_water_n_oil_wet
            u = se_w ** (-one / m_ow) - one
            u_safe = max(u, 1e-30)
            dpc_du = (
                -(one / oil_water_alpha_oil_wet)
                * (one / oil_water_n_oil_wet)
                * (u_safe ** (one / oil_water_n_oil_wet - one))
            )
            du_dse = (-one / m_ow) * (se_w ** (-one / m_ow - one))
            dpcow_dse_w = dpc_du * du_dse
        else:  # MIXED_WET
            # Water-wet component
            m_ww = one - one / oil_water_n_water_wet
            u_ww = se_w ** (-one / m_ww) - one
            u_ww_safe = max(u_ww, 1e-30)
            dpc_du_ww = (
                (one / oil_water_alpha_water_wet)
                * (one / oil_water_n_water_wet)
                * (u_ww_safe ** (one / oil_water_n_water_wet - one))
            )
            du_dse_ww = (-one / m_ww) * (se_w ** (-one / m_ww - one))
            d_ww = dpc_du_ww * du_dse_ww

            # Oil-wet component
            m_ow = one - one / oil_water_n_oil_wet
            u_ow = se_w ** (-one / m_ow) - one
            u_ow_safe = max(u_ow, 1e-30)
            dpc_du_ow = (
                -(one / oil_water_alpha_oil_wet)
                * (one / oil_water_n_oil_wet)
                * (u_ow_safe ** (one / oil_water_n_oil_wet - one))
            )
            du_dse_ow = (-one / m_ow) * (se_w ** (-one / m_ow - one))
            d_ow = dpc_du_ow * du_dse_ow

            dpcow_dse_w = mixed_wet_water_fraction * d_ww + (one - mixed_wet_water_fraction) * d_ow

        dpcow_dsw = dpcow_dse_w / mobile_water_range

    dpcow_dso = zero

    # Gas-oil derivatives
    mobile_gas_range = one - swc - sorg - sgr
    valid_gas = mobile_gas_range > minimum_mobile_pore_space
    se_g = min(
        max((sg - sgr) / mobile_gas_range, saturation_epsilon),
        one - saturation_epsilon,
    )

    dpcgo_dsg = zero
    if valid_gas:
        m_gas_oil = one - one / gas_oil_n
        u = se_g ** (-one / m_gas_oil) - one
        u_safe = max(u, 1e-30)
        dpc_du = (one / gas_oil_alpha) * (one / gas_oil_n) * (u_safe ** (one / gas_oil_n - one))
        du_dse = (-one / m_gas_oil) * (se_g ** (-one / m_gas_oil - one))
        dpcgo_dse_g = dpc_du * du_dse
        dpcgo_dsg = dpcgo_dse_g / mobile_gas_range

    dpcgo_dso = zero
    return dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso


@numba.njit(cache=True)
def compute_van_genuchten_derivatives_array(
    water_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    wettability: Wettability,
    oil_water_alpha_water_wet: Number,
    oil_water_alpha_oil_wet: Number,
    oil_water_n_water_wet: Number,
    oil_water_n_oil_wet: Number,
    gas_oil_alpha: Number,
    gas_oil_n: Number,
    mixed_wet_water_fraction: Number = 0.5,
    saturation_epsilon: Number = 1e-6,
    minimum_mobile_pore_space: Number = 1e-9,
) -> tuple[
    NumberArray[NDimension],
    NumberArray[NDimension],
    NumberArray[NDimension],
    NumberArray[NDimension],
]:
    """
    Array variant of van Genuchten capillary pressure derivatives.

    Returns (dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso) as NDArrays.

    :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
    :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
    :param irreducible_water_saturation: Irreducible water saturation (swc) - scalar or array.
    :param residual_oil_saturation_water: Residual oil saturation during water flooding (sorw) - scalar or array.
    :param residual_oil_saturation_gas: Residual oil saturation during gas flooding (sorg) - scalar or array.
    :param residual_gas_saturation: Residual gas saturation (sgr) - scalar or array.
    :param wettability: Wettability type.
    :param oil_water_alpha_water_wet: van Genuchten α for oil-water (water-wet) [1/psi].
    :param oil_water_alpha_oil_wet: van Genuchten α for oil-water (oil-wet) [1/psi].
    :param oil_water_n_water_wet: van Genuchten n for oil-water (water-wet).
    :param oil_water_n_oil_wet: van Genuchten n for oil-water (oil-wet).
    :param gas_oil_alpha: van Genuchten α for gas-oil [1/psi].
    :param gas_oil_n: van Genuchten n for gas-oil.
    :param mixed_wet_water_fraction: Fraction of pore space that is water-wet (0-1).
    :param saturation_epsilon: Small value to avoid division by zero.
    :param minimum_mobile_pore_space: Minimum mobile pore space threshold.
    :return: Tuple of (dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso) as NDArrays.
    """
    sw = atleast_1d(water_saturation)
    sg = atleast_1d(gas_saturation)
    swc = atleast_1d(irreducible_water_saturation)
    sorw = atleast_1d(residual_oil_saturation_water)
    sorg = atleast_1d(residual_oil_saturation_gas)
    sgr = atleast_1d(residual_gas_saturation)

    dtype = sw.dtype.type
    one = dtype(1.0)

    # Broadcast arrays
    sw, sg, swc, sorw, sorg, sgr = np.broadcast_arrays(sw, sg, swc, sorw, sorg, sgr)

    # Oil-water derivatives
    mobile_water_range = one - swc - sorw
    valid_water = mobile_water_range > minimum_mobile_pore_space
    se_w = np.clip(
        (sw - swc) / np.where(valid_water, mobile_water_range, one),
        saturation_epsilon,
        one - saturation_epsilon,
    )

    if wettability == Wettability.WATER_WET:
        m_ww = one - one / dtype(oil_water_n_water_wet)
        u = se_w ** (-one / m_ww) - one
        u_safe = np.where(u > 1e-30, u, 1e-30)
        dpc_du = (
            (one / dtype(oil_water_alpha_water_wet))
            * (one / dtype(oil_water_n_water_wet))
            * (u_safe ** (one / dtype(oil_water_n_water_wet) - one))
        )
        du_dse = (-one / m_ww) * (se_w ** (-one / m_ww - one))
        dpcow_dse_w = dpc_du * du_dse
    elif wettability == Wettability.OIL_WET:
        m_ow = one - one / dtype(oil_water_n_oil_wet)
        u = se_w ** (-one / m_ow) - one
        u_safe = np.where(u > 1e-30, u, 1e-30)
        dpc_du = (
            -(one / dtype(oil_water_alpha_oil_wet))
            * (one / dtype(oil_water_n_oil_wet))
            * (u_safe ** (one / dtype(oil_water_n_oil_wet) - one))
        )
        du_dse = (-one / m_ow) * (se_w ** (-one / m_ow - one))
        dpcow_dse_w = dpc_du * du_dse
    else:  # MIXED_WET
        water_wet_fraction = dtype(mixed_wet_water_fraction)

        # Water-wet component
        m_ww = one - one / dtype(oil_water_n_water_wet)
        u_ww = se_w ** (-one / m_ww) - one
        u_ww_safe = np.where(u_ww > 1e-30, u_ww, 1e-30)
        dpc_du_ww = (
            (one / dtype(oil_water_alpha_water_wet))
            * (one / dtype(oil_water_n_water_wet))
            * (u_ww_safe ** (one / dtype(oil_water_n_water_wet) - one))
        )
        du_dse_ww = (-one / m_ww) * (se_w ** (-one / m_ww - one))
        d_ww = dpc_du_ww * du_dse_ww

        # Oil-wet component
        m_ow = one - one / dtype(oil_water_n_oil_wet)
        u_ow = se_w ** (-one / m_ow) - one
        u_ow_safe = np.where(u_ow > 1e-30, u_ow, 1e-30)
        dpc_du_ow = (
            -(one / dtype(oil_water_alpha_oil_wet))
            * (one / dtype(oil_water_n_oil_wet))
            * (u_ow_safe ** (one / dtype(oil_water_n_oil_wet) - one))
        )
        du_dse_ow = (-one / m_ow) * (se_w ** (-one / m_ow - one))
        d_ow = dpc_du_ow * du_dse_ow

        dpcow_dse_w = water_wet_fraction * d_ww + (one - water_wet_fraction) * d_ow

    dpcow_dsw = np.where(
        valid_water,
        dpcow_dse_w / mobile_water_range,
        np.zeros_like(sw),
    )
    dpcow_dso = np.zeros_like(sw)

    # Gas-oil derivatives
    mobile_gas_range = one - swc - sorg - sgr
    valid_gas = mobile_gas_range > minimum_mobile_pore_space
    se_g = np.clip(
        (sg - sgr) / np.where(valid_gas, mobile_gas_range, one),
        saturation_epsilon,
        one - saturation_epsilon,
    )

    m_gas_oil = one - one / dtype(gas_oil_n)
    u = se_g ** (-one / m_gas_oil) - one
    u_safe = np.where(u > 1e-30, u, 1e-30)
    dpc_du = (
        (one / dtype(gas_oil_alpha))
        * (one / dtype(gas_oil_n))
        * (u_safe ** (one / dtype(gas_oil_n) - one))
    )
    du_dse = (-one / m_gas_oil) * (se_g ** (-one / m_gas_oil - one))
    dpcgo_dse_g = dpc_du * du_dse
    dpcgo_dsg = np.where(
        valid_gas,
        dpcgo_dse_g / mobile_gas_range,
        np.zeros_like(sg),
    )
    dpcgo_dso = np.zeros_like(sg)

    return (
        typing.cast(NumberArray[NDimension], dpcow_dsw),
        typing.cast(NumberArray[NDimension], dpcow_dso),
        typing.cast(NumberArray[NDimension], dpcgo_dsg),
        typing.cast(NumberArray[NDimension], dpcgo_dso),
    )


def compute_van_genuchten_derivatives(
    water_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    wettability: Wettability,
    oil_water_alpha_water_wet: Number,
    oil_water_alpha_oil_wet: Number,
    oil_water_n_water_wet: Number,
    oil_water_n_oil_wet: Number,
    gas_oil_alpha: Number,
    gas_oil_n: Number,
    mixed_wet_water_fraction: Number = 0.5,
    saturation_epsilon: Number = 1e-6,
    minimum_mobile_pore_space: Number = 1e-9,
) -> tuple[
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
]:
    """
    Dispatch function for van Genuchten capillary pressure derivatives.

    Routes to the scalar variant when all inputs are Python scalars, otherwise
    routes to the array variant.

    :return: (dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso) as scalars or arrays.
    """
    if (
        np.isscalar(water_saturation)
        and np.isscalar(gas_saturation)
        and np.isscalar(irreducible_water_saturation)
        and np.isscalar(residual_oil_saturation_water)
        and np.isscalar(residual_oil_saturation_gas)
        and np.isscalar(residual_gas_saturation)
    ):
        return compute_van_genuchten_derivatives_scalar(
            water_saturation=water_saturation,  # type: ignore[arg-type]
            gas_saturation=gas_saturation,  # type: ignore[arg-type]
            irreducible_water_saturation=irreducible_water_saturation,  # type: ignore[arg-type]
            residual_oil_saturation_water=residual_oil_saturation_water,  # type: ignore[arg-type]
            residual_oil_saturation_gas=residual_oil_saturation_gas,  # type: ignore[arg-type]
            residual_gas_saturation=residual_gas_saturation,  # type: ignore[arg-type]
            wettability=wettability,
            oil_water_alpha_water_wet=oil_water_alpha_water_wet,
            oil_water_alpha_oil_wet=oil_water_alpha_oil_wet,
            oil_water_n_water_wet=oil_water_n_water_wet,
            oil_water_n_oil_wet=oil_water_n_oil_wet,
            gas_oil_alpha=gas_oil_alpha,
            gas_oil_n=gas_oil_n,
            mixed_wet_water_fraction=mixed_wet_water_fraction,
            saturation_epsilon=saturation_epsilon,
            minimum_mobile_pore_space=minimum_mobile_pore_space,
        )
    return compute_van_genuchten_derivatives_array(
        water_saturation=water_saturation,  # type: ignore[arg-type]
        gas_saturation=gas_saturation,  # type: ignore[arg-type]
        irreducible_water_saturation=irreducible_water_saturation,  # type: ignore[arg-type]
        residual_oil_saturation_water=residual_oil_saturation_water,  # type: ignore[arg-type]
        residual_oil_saturation_gas=residual_oil_saturation_gas,  # type: ignore[arg-type]
        residual_gas_saturation=residual_gas_saturation,  # type: ignore[arg-type]
        wettability=wettability,
        oil_water_alpha_water_wet=oil_water_alpha_water_wet,
        oil_water_alpha_oil_wet=oil_water_alpha_oil_wet,
        oil_water_n_water_wet=oil_water_n_water_wet,
        oil_water_n_oil_wet=oil_water_n_oil_wet,
        gas_oil_alpha=gas_oil_alpha,
        gas_oil_n=gas_oil_n,
        mixed_wet_water_fraction=mixed_wet_water_fraction,
        saturation_epsilon=saturation_epsilon,
        minimum_mobile_pore_space=minimum_mobile_pore_space,
    )


@attrs.frozen(slots=True)
class VanGenuchtenCapillaryPressureTable(CapillaryPressureTable):
    """
    Implements the van Genuchten capillary pressure model for three-phase systems.

    Implements: Pc = (1/α) * [(Se^(-1/m) - 1)^(1/n)] where m = 1 - 1/n

    Provides smoother transitions than Brooks-Corey model.
    """

    __type__ = "van_genuchten_capillary_pressure_model"

    irreducible_water_saturation: Number | None = None
    """Default irreducible water saturation (swc). Can be overridden per call."""

    residual_oil_saturation_water: Number | None = None
    """Default residual oil saturation after water flood (sorw). Can be overridden per call."""

    residual_oil_saturation_gas: Number | None = None
    """Default residual oil saturation after gas flood (sorg). Can be overridden per call."""

    residual_gas_saturation: Number | None = None
    """Default residual gas saturation (sgr). Can be overridden per call."""

    oil_water_alpha_water_wet: Number = 0.01
    """van Genuchten α parameter for oil-water (water-wet). Units: 1/[pressure] in `unit_system`."""

    oil_water_alpha_oil_wet: Number = 0.01
    """van Genuchten α parameter for oil-water (oil-wet). Units: 1/[pressure] in `unit_system`."""

    oil_water_n_water_wet: Number = 2.0
    """van Genuchten n parameter for oil-water (water-wet)."""

    oil_water_n_oil_wet: Number = 2.0
    """van Genuchten n parameter for oil-water (oil-wet)."""

    gas_oil_alpha: Number = 0.01
    """van Genuchten α parameter for gas-oil. Units: 1/[pressure] in `unit_system`."""

    gas_oil_n: Number = 2.0
    """van Genuchten n parameter for gas-oil."""

    wettability: Wettability = Wettability.WATER_WET
    """Wettability type (`WATER_WET`, `OIL_WET`, or `MIXED_WET`)."""

    mixed_wet_water_fraction: Number = 0.5
    """Fraction of pore space that is water-wet in mixed-wet systems (0-1)."""

    unit_system: UnitSystem = attrs.field(default=UnitSystem.FIELD)
    """
    Unit system in which the pressure parameters of this model are expressed.

    All entry pressures (`oil_water_alpha_water_wet`,
    `oil_water_alpha_oil_wet`, `gas_oil_alpha`) and the
    resulting capillary pressure outputs are in this unit system.
    Saturation parameters and dimensionless ratios are unaffected by unit conversion.

    Use `convert(target)` to produce a copy of this model rescaled to
    another `UnitSystem`.
    """

    supports_vector: bool = attrs.field(init=False, repr=False, default=True)
    """Flag indicating support for array inputs."""

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
    ) -> CapillaryPressures:
        """
        Compute capillary pressures using van Genuchten model.

        Supports both scalar and array inputs.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param irreducible_water_saturation: Optional override for swc - scalar or array.
        :param residual_oil_saturation_water: Optional override for sorw - scalar or array.
        :param residual_oil_saturation_gas: Optional override for sorg - scalar or array.
        :param residual_gas_saturation: Optional override for sgr - scalar or array.
        :return: `CapillaryPressures` dictionary.
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
                f"Residual saturations must be provided either as model defaults or in the call. "
                f"Missing: {', '.join(params_missing)}"
            )

        pcow, pcgo = compute_van_genuchten_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=sorg,  # type: ignore[arg-type]
            residual_gas_saturation=sgr,  # type: ignore[arg-type]
            wettability=self.wettability,
            oil_water_alpha_water_wet=self.oil_water_alpha_water_wet,
            oil_water_alpha_oil_wet=self.oil_water_alpha_oil_wet,
            oil_water_n_water_wet=self.oil_water_n_water_wet,
            oil_water_n_oil_wet=self.oil_water_n_oil_wet,
            gas_oil_alpha=self.gas_oil_alpha,
            gas_oil_n=self.gas_oil_n,
            mixed_wet_water_fraction=self.mixed_wet_water_fraction,
            saturation_epsilon=c.SATURATION_EPSILON,
            minimum_mobile_pore_space=c.MINIMUM_MOBILE_PORE_SPACE,
        )
        return CapillaryPressures(oil_water=pcow, gas_oil=pcgo)  # type: ignore[typeddict-item]

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
    ) -> CapillaryPressureDerivatives:
        """
        Compute the partial derivatives of the van Genuchten oil-water and
        gas-oil capillary pressures with respect to saturation.

        Returns a dictionary containing:

        ```
        (dpcow / dsw, dpcow / dso, dpcgo / dsg)
        ```

        - `dpcow/dsw`: analytically derived via the chain rule through effective water saturation.
        - `dpcow/dso`: zero - the oil-water capillary pressure depends only
        on water saturation in this model.
        - `dpcgo/dsg`: analytically derived via the chain rule through effective gas saturation.

        The van Genuchten model is:
        ```
        Pc = (1/alpha) * [(Se^(-1/m) - 1)^(1/n)]    where m = 1 - 1/n
        ```

        :param water_saturation: Water saturation (scalar or array).
        :param oil_saturation: Oil saturation (scalar or array).
        :param gas_saturation: Gas saturation (scalar or array).
        :param irreducible_water_saturation: Optional override for the
            irreducible (connate) water saturation.
        :param residual_oil_saturation_water: Optional override for the residual
            oil saturation to water flooding.
        :param residual_oil_saturation_gas: Optional override for the residual
            oil saturation to gas flooding.
        :param residual_gas_saturation: Optional override for the residual gas
            saturation.
        :return: `CapillaryPressureDerivatives` dictionary containing the partial derivatives as described above.
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
                f"Residual saturations must be provided either as model defaults or in the call. "
                f"Missing: {', '.join(params_missing)}"
            )

        dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso = compute_van_genuchten_derivatives(
            water_saturation=water_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=sorg,  # type: ignore[arg-type]
            residual_gas_saturation=sgr,  # type: ignore[arg-type]
            wettability=self.wettability,
            oil_water_alpha_water_wet=self.oil_water_alpha_water_wet,
            oil_water_alpha_oil_wet=self.oil_water_alpha_oil_wet,
            oil_water_n_water_wet=self.oil_water_n_water_wet,
            oil_water_n_oil_wet=self.oil_water_n_oil_wet,
            gas_oil_alpha=self.gas_oil_alpha,
            gas_oil_n=self.gas_oil_n,
            mixed_wet_water_fraction=self.mixed_wet_water_fraction,
            saturation_epsilon=c.SATURATION_EPSILON,
            minimum_mobile_pore_space=c.MINIMUM_MOBILE_PORE_SPACE,
        )
        return CapillaryPressureDerivatives(
            dpcow_dsw=dpcow_dsw,
            dpcow_dso=dpcow_dso,
            dpcgo_dsg=dpcgo_dsg,
            dpcgo_dso=dpcgo_dso,
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `VanGenuchtenCapillaryPressureTable` with all pressure
        parameters rescaled to *target*.

        The alpha parameters have units of `1/[pressure]`, so they are
        divided by the pressure conversion factor (equivalently, multiplied
        by `1 / pressure_factor`).  The `n` parameters and saturation
        parameters are dimensionless and are copied unchanged.

        :param target: Target `UnitSystem`.
        :returns: New `VanGenuchtenCapillaryPressureTable` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        pressure_factor = factors["pressure"]
        # alpha has units 1/pressure -> divide by pressure_factor
        alpha_factor = 1.0 / pressure_factor
        return attrs.evolve(
            self,
            oil_water_alpha_water_wet=self.oil_water_alpha_water_wet * alpha_factor,
            oil_water_alpha_oil_wet=self.oil_water_alpha_oil_wet * alpha_factor,
            gas_oil_alpha=self.gas_oil_alpha * alpha_factor,
            unit_system=target,
        )


@numba.njit(cache=True)
def compute_leverett_j_capillary_pressures_scalar(
    water_saturation: Number,
    oil_saturation: Number,
    gas_saturation: Number,
    irreducible_water_saturation: Number,
    residual_oil_saturation_water: Number,
    residual_oil_saturation_gas: Number,
    residual_gas_saturation: Number,
    permeability: Number,
    porosity: Number,
    oil_water_interfacial_tension: Number,
    gas_oil_interfacial_tension: Number,
    oil_water_contact_angle: Number = 0.0,
    gas_oil_contact_angle: Number = 0.0,
    j_function_coefficient: Number = 0.5,
    j_function_exponent: Number = 0.5,
    mixed_wet_water_fraction: Number = 0.5,
    wettability: Wettability = Wettability.WATER_WET,
    saturation_epsilon: Number = 1e-6,
    minimum_mobile_pore_space: Number = 1e-9,
    dyne_per_cm_to_pressure_unit: Number = 4.725e-4,
) -> tuple[Number, Number]:
    """
    Scalar variant of Leverett J-function capillary pressure computation.

    Computes capillary pressures using Leverett J-function approach.

    Pc = σ * cos(θ) * sqrt(φ/k) * J(Se)

    where J(Se) = a * Se^(-b) (power-law form).

    :param water_saturation: Current water saturation (fraction, 0-1).
    :param oil_saturation: Current oil saturation (fraction, 0-1).
    :param gas_saturation: Current gas saturation (fraction, 0-1).
    :param irreducible_water_saturation: Irreducible water saturation (swc).
    :param residual_oil_saturation_water: Residual oil saturation during water flooding (sorw).
    :param residual_oil_saturation_gas: Residual oil saturation during gas flooding (sorg).
    :param residual_gas_saturation: Residual gas saturation (sgr).
    :param permeability: Absolute permeability (mD).
    :param porosity: Porosity (fraction, 0-1).
    :param oil_water_interfacial_tension: Oil-water interfacial tension (dyne/cm).
    :param gas_oil_interfacial_tension: Gas-oil interfacial tension (dyne/cm).
    :param oil_water_contact_angle: Oil-water contact angle in degrees (0° = water-wet).
    :param gas_oil_contact_angle: Gas-oil contact angle in degrees (0° = oil-wet).
    :param j_function_coefficient: Empirical coefficient 'a' in J(Se) = a * Se^(-b).
    :param j_function_exponent: Empirical exponent 'b' in J(Se) = a * Se^(-b).
    :param mixed_wet_water_fraction: Fraction of pore space that is water-wet in mixed-wet systems (0-1).
    :param wettability: Wettability type (affects sign of capillary pressure).
    :param saturation_epsilon: Small value to avoid division by zero in saturation calculations.
    :param minimum_mobile_pore_space: Minimum mobile pore space threshold below which Pc is set to zero.
    :return: Tuple of (oil_water_capillary_pressure, gas_oil_capillary_pressure) in psi.
    """
    sw: Number = water_saturation
    so: Number = oil_saturation
    sg: Number = gas_saturation
    swc: Number = irreducible_water_saturation
    sorw: Number = residual_oil_saturation_water
    sorg: Number = residual_oil_saturation_gas
    sgr: Number = residual_gas_saturation

    # Validate saturations
    if sw < 0 or sw > 1 or so < 0 or so > 1 or sg < 0 or sg > 1:
        raise ValidationError("Saturations must be between 0 and 1.")
    if permeability < 0.0:
        raise ValidationError("Permeability must be positive.")
    if porosity < 0.0 or porosity > 1.0:
        raise ValidationError("Porosity must be between 0 and 1.")

    # Normalize saturations if they do not sum to 1
    total_saturation = sw + so + sg
    if abs(total_saturation - 1.0) > saturation_epsilon and total_saturation > 0.0:
        sw = sw / total_saturation
        sg = sg / total_saturation

    total_mobile_pore_space_water = 1.0 - swc - sorw
    total_mobile_pore_space_gas = 1.0 - swc - sorg - sgr

    oil_water_angle_radians = np.deg2rad(oil_water_contact_angle)
    gas_oil_angle_radians = np.deg2rad(gas_oil_contact_angle)

    valid_rock = permeability > 0 and porosity > 0
    leverett_factor = np.sqrt(porosity / permeability) if valid_rock else 0.0

    # Pcow (Po - Pw)
    oil_water_capillary_pressure: Number = 0.0
    valid_water = total_mobile_pore_space_water > minimum_mobile_pore_space and valid_rock

    if valid_water:
        effective_water_saturation = (sw - swc) / total_mobile_pore_space_water
        effective_water_saturation = min(
            max(effective_water_saturation, saturation_epsilon),
            1.0 - saturation_epsilon,
        )

        j_value_oil_water = j_function_coefficient * (
            effective_water_saturation ** (-j_function_exponent)
        )
        pcow = (
            oil_water_interfacial_tension
            * dyne_per_cm_to_pressure_unit
            * np.cos(oil_water_angle_radians)
            * leverett_factor
            * j_value_oil_water
        )

        if wettability == Wettability.WATER_WET:
            oil_water_capillary_pressure = pcow
        elif wettability == Wettability.OIL_WET:
            oil_water_capillary_pressure = -pcow
        else:  # MIXED_WET
            oil_water_capillary_pressure = mixed_wet_water_fraction * pcow + (
                1.0 - mixed_wet_water_fraction
            ) * (-pcow)

    # Pcgo (Pg - Po)
    gas_oil_capillary_pressure: Number = 0.0
    valid_gas = total_mobile_pore_space_gas > minimum_mobile_pore_space and valid_rock

    if valid_gas:
        effective_gas_saturation = (sg - sgr) / total_mobile_pore_space_gas
        effective_gas_saturation = min(
            max(effective_gas_saturation, saturation_epsilon),
            1.0 - saturation_epsilon,
        )

        j_value_gas_oil = j_function_coefficient * (
            effective_gas_saturation ** (-j_function_exponent)
        )
        gas_oil_capillary_pressure = (
            gas_oil_interfacial_tension
            * dyne_per_cm_to_pressure_unit
            * np.cos(gas_oil_angle_radians)
            * leverett_factor
            * j_value_gas_oil
        )

    return oil_water_capillary_pressure, gas_oil_capillary_pressure


@numba.njit(cache=True)
def compute_leverett_j_capillary_pressures_array(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    permeability: NumberOrArray[NDimension],
    porosity: NumberOrArray[NDimension],
    oil_water_interfacial_tension: Number,
    gas_oil_interfacial_tension: Number,
    oil_water_contact_angle: Number = 0.0,
    gas_oil_contact_angle: Number = 0.0,
    j_function_coefficient: Number = 0.5,
    j_function_exponent: Number = 0.5,
    mixed_wet_water_fraction: Number = 0.5,
    wettability: Wettability = Wettability.WATER_WET,
    saturation_epsilon: Number = 1e-6,
    minimum_mobile_pore_space: Number = 1e-9,
    dyne_per_cm_to_pressure_unit: Number = 4.725e-4,
) -> tuple[NumberArray[NDimension], NumberArray[NDimension]]:
    """
    Array variant of Leverett J-function capillary pressure computation.

    Computes capillary pressures using Leverett J-function approach.
    Supports array inputs (up to 3D). Always returns NDArrays.

    Pc = σ * cos(θ) * sqrt(φ/k) * J(Se)

    where J(Se) = a * Se^(-b) (power-law form).

    :param water_saturation: Current water saturation (fraction, 0-1) - scalar or array.
    :param oil_saturation: Current oil saturation (fraction, 0-1) - scalar or array.
    :param gas_saturation: Current gas saturation (fraction, 0-1) - scalar or array.
    :param irreducible_water_saturation: Irreducible water saturation (swc) - scalar or array.
    :param residual_oil_saturation_water: Residual oil saturation during water flooding (sorw) - scalar or array.
    :param residual_oil_saturation_gas: Residual oil saturation during gas flooding (sorg) - scalar or array.
    :param residual_gas_saturation: Residual gas saturation (sgr) - scalar or array.
    :param permeability: Absolute permeability (mD) - scalar or array.
    :param porosity: Porosity (fraction, 0-1) - scalar or array.
    :param oil_water_interfacial_tension: Oil-water interfacial tension (dyne/cm).
    :param gas_oil_interfacial_tension: Gas-oil interfacial tension (dyne/cm).
    :param oil_water_contact_angle: Oil-water contact angle in degrees (0° = water-wet).
    :param gas_oil_contact_angle: Gas-oil contact angle in degrees (0° = oil-wet).
    :param j_function_coefficient: Empirical coefficient 'a' in J(Se) = a * Se^(-b).
    :param j_function_exponent: Empirical exponent 'b' in J(Se) = a * Se^(-b).
    :param mixed_wet_water_fraction: Fraction of pore space that is water-wet in mixed-wet systems (0-1).
    :param wettability: Wettability type (affects sign of capillary pressure).
    :param saturation_epsilon: Small value to avoid division by zero in saturation calculations.
    :param minimum_mobile_pore_space: Minimum mobile pore space threshold below which Pc is set to zero.
    :return: Tuple of (oil_water_capillary_pressure, gas_oil_capillary_pressure) NDArrays in psi.
    """
    sw = atleast_1d(water_saturation)
    so = atleast_1d(oil_saturation)
    sg = atleast_1d(gas_saturation)
    swc = atleast_1d(irreducible_water_saturation)
    sorw = atleast_1d(residual_oil_saturation_water)
    sorg = atleast_1d(residual_oil_saturation_gas)
    sgr = atleast_1d(residual_gas_saturation)
    perm = atleast_1d(permeability, sw.dtype)
    phi = atleast_1d(porosity, sw.dtype)

    # Broadcast all arrays to same shape
    sw, so, sg, swc, sorw, sorg, sgr, perm, phi = np.broadcast_arrays(
        sw, so, sg, swc, sorw, sorg, sgr, perm, phi
    )
    dtype = sw.dtype.type
    oil_water_interfacial_tension = dtype(oil_water_interfacial_tension)
    gas_oil_interfacial_tension = dtype(gas_oil_interfacial_tension)
    oil_water_contact_angle = dtype(oil_water_contact_angle)
    gas_oil_contact_angle = dtype(gas_oil_contact_angle)
    j_function_coefficient = dtype(j_function_coefficient)
    j_function_exponent = dtype(j_function_exponent)
    mixed_wet_water_fraction = dtype(mixed_wet_water_fraction)
    one = dtype(1.0)
    zero = dtype(0.0)

    # Validate saturations
    if np.any((sw < 0) | (sw > 1) | (so < 0) | (so > 1) | (sg < 0) | (sg > 1)):
        raise ValidationError("Saturations must be between 0 and 1.")
    if np.any(perm < 0.0):
        raise ValidationError("Permeability must be positive.")
    if np.any((phi < 0.0) | (phi > 1.0)):
        raise ValidationError("Porosity must be between 0 and 1.")

    # Normalize saturations if they do not sum to 1
    total_saturation = sw + so + sg
    needs_norm = (np.abs(total_saturation - 1.0) > saturation_epsilon) & (total_saturation > 0.0)
    if np.any(needs_norm):
        sw = np.where(needs_norm, sw / total_saturation, sw)
        so = np.where(needs_norm, so / total_saturation, so)
        sg = np.where(needs_norm, sg / total_saturation, sg)

    total_mobile_pore_space_water = one - swc - sorw
    total_mobile_pore_space_gas = one - swc - sorg - sgr

    oil_water_angle_radians = np.deg2rad(oil_water_contact_angle)
    gas_oil_angle_radians = np.deg2rad(gas_oil_contact_angle)

    valid_rock = (perm > zero) & (phi > zero)
    leverett_factor = np.where(valid_rock, np.sqrt(phi / perm), zero)

    # Pcow (Po - Pw)
    oil_water_capillary_pressure = np.zeros_like(sw)
    valid_water = (total_mobile_pore_space_water > minimum_mobile_pore_space) & valid_rock

    if np.any(valid_water):
        effective_water_saturation = np.where(
            valid_water, (sw - swc) / total_mobile_pore_space_water, zero
        )
        effective_water_saturation = np.clip(
            effective_water_saturation, saturation_epsilon, one - saturation_epsilon
        )
        j_value_oil_water = j_function_coefficient * (
            effective_water_saturation ** (-j_function_exponent)
        )
        pcow = (
            oil_water_interfacial_tension
            * np.cos(oil_water_angle_radians)
            * leverett_factor
            * j_value_oil_water
            * dyne_per_cm_to_pressure_unit
        )
        pcow = pcow.astype(dtype)

        if wettability == Wettability.WATER_WET:
            oil_water_capillary_pressure = np.where(valid_water, pcow, zero)
        elif wettability == Wettability.OIL_WET:
            oil_water_capillary_pressure = np.where(valid_water, -pcow, zero)
        else:  # MIXED_WET
            mixedpcow = pcow * mixed_wet_water_fraction + (one - mixed_wet_water_fraction) * (
                -pcow
            )
            oil_water_capillary_pressure = np.where(valid_water, mixedpcow, zero)

    # Pcgo (Pg - Po)
    gas_oil_capillary_pressure = np.zeros_like(sg)
    valid_gas = (total_mobile_pore_space_gas > minimum_mobile_pore_space) & valid_rock

    if np.any(valid_gas):
        effective_gas_saturation = np.where(
            valid_gas, (sg - sgr) / total_mobile_pore_space_gas, zero
        )
        effective_gas_saturation = np.clip(
            effective_gas_saturation, saturation_epsilon, one - saturation_epsilon
        )
        j_value_gas_oil = j_function_coefficient * (
            effective_gas_saturation ** (-j_function_exponent)
        )
        pcgo = (
            gas_oil_interfacial_tension
            * np.cos(gas_oil_angle_radians)
            * leverett_factor
            * j_value_gas_oil
            * dyne_per_cm_to_pressure_unit
        )
        pcgo = pcgo.astype(dtype)
        gas_oil_capillary_pressure = np.where(valid_gas, pcgo, zero)

    return oil_water_capillary_pressure, gas_oil_capillary_pressure  # type: ignore[return-value]


def compute_leverett_j_capillary_pressures(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    permeability: NumberOrArray[NDimension],
    porosity: NumberOrArray[NDimension],
    oil_water_interfacial_tension: Number,
    gas_oil_interfacial_tension: Number,
    oil_water_contact_angle: Number = 0.0,
    gas_oil_contact_angle: Number = 0.0,
    j_function_coefficient: Number = 0.5,
    j_function_exponent: Number = 0.5,
    mixed_wet_water_fraction: Number = 0.5,
    wettability: Wettability = Wettability.WATER_WET,
    saturation_epsilon: Number = 1e-6,
    minimum_mobile_pore_space: Number = 1e-9,
    dyne_per_cm_to_pressure_unit: Number = 4.725e-4,
) -> tuple[NumberOrArray[NDimension], NumberOrArray[NDimension]]:
    """
    Dispatch function for Leverett J-function capillary pressure computation.

    Routes to the scalar variant when all inputs are Python scalars, otherwise
    routes to the array variant. This avoids Numba union return-type limitations.

    :return: (Pcow, Pcgo) as (Number, Number) for scalar inputs,
        or (NDArray, NDArray) for array inputs, both in psi.
    """
    if (
        np.isscalar(water_saturation)
        and np.isscalar(oil_saturation)
        and np.isscalar(gas_saturation)
        and np.isscalar(irreducible_water_saturation)
        and np.isscalar(residual_oil_saturation_water)
        and np.isscalar(residual_oil_saturation_gas)
        and np.isscalar(residual_gas_saturation)
        and np.isscalar(permeability)
        and np.isscalar(porosity)
    ):
        return compute_leverett_j_capillary_pressures_scalar(
            water_saturation=water_saturation,  # type: ignore[arg-type]
            oil_saturation=oil_saturation,  # type: ignore[arg-type]
            gas_saturation=gas_saturation,  # type: ignore[arg-type]
            irreducible_water_saturation=irreducible_water_saturation,  # type: ignore[arg-type]
            residual_oil_saturation_water=residual_oil_saturation_water,  # type: ignore[arg-type]
            residual_oil_saturation_gas=residual_oil_saturation_gas,  # type: ignore[arg-type]
            residual_gas_saturation=residual_gas_saturation,  # type: ignore[arg-type]
            permeability=permeability,  # type: ignore[arg-type]
            porosity=porosity,  # type: ignore[arg-type]
            oil_water_interfacial_tension=oil_water_interfacial_tension,
            gas_oil_interfacial_tension=gas_oil_interfacial_tension,
            oil_water_contact_angle=oil_water_contact_angle,
            gas_oil_contact_angle=gas_oil_contact_angle,
            j_function_coefficient=j_function_coefficient,
            j_function_exponent=j_function_exponent,
            mixed_wet_water_fraction=mixed_wet_water_fraction,
            wettability=wettability,
            saturation_epsilon=saturation_epsilon,
            minimum_mobile_pore_space=minimum_mobile_pore_space,
            dyne_per_cm_to_pressure_unit=dyne_per_cm_to_pressure_unit,
        )
    return compute_leverett_j_capillary_pressures_array(
        water_saturation=water_saturation,  # type: ignore[arg-type]
        oil_saturation=oil_saturation,  # type: ignore[arg-type]
        gas_saturation=gas_saturation,  # type: ignore[arg-type]
        irreducible_water_saturation=irreducible_water_saturation,  # type: ignore[arg-type]
        residual_oil_saturation_water=residual_oil_saturation_water,  # type: ignore[arg-type]
        residual_oil_saturation_gas=residual_oil_saturation_gas,  # type: ignore[arg-type]
        residual_gas_saturation=residual_gas_saturation,  # type: ignore[arg-type]
        permeability=permeability,  # type: ignore[arg-type]
        porosity=porosity,  # type: ignore[arg-type]
        oil_water_interfacial_tension=oil_water_interfacial_tension,
        gas_oil_interfacial_tension=gas_oil_interfacial_tension,
        oil_water_contact_angle=oil_water_contact_angle,
        gas_oil_contact_angle=gas_oil_contact_angle,
        j_function_coefficient=j_function_coefficient,
        j_function_exponent=j_function_exponent,
        mixed_wet_water_fraction=mixed_wet_water_fraction,
        wettability=wettability,
        saturation_epsilon=saturation_epsilon,
        minimum_mobile_pore_space=minimum_mobile_pore_space,
        dyne_per_cm_to_pressure_unit=dyne_per_cm_to_pressure_unit,
    )


@numba.njit(cache=True)
def compute_leverett_j_derivatives_scalar(
    water_saturation: Number,
    gas_saturation: Number,
    irreducible_water_saturation: Number,
    residual_oil_saturation_water: Number,
    residual_oil_saturation_gas: Number,
    residual_gas_saturation: Number,
    permeability: Number,
    porosity: Number,
    wettability: Wettability,
    oil_water_interfacial_tension: Number,
    gas_oil_interfacial_tension: Number,
    oil_water_contact_angle: Number,
    gas_oil_contact_angle: Number,
    j_function_coefficient: Number,
    j_function_exponent: Number,
    mixed_wet_water_fraction: Number = 0.5,
    saturation_epsilon: Number = 1e-6,
    minimum_mobile_pore_space: Number = 1e-9,
    dyne_per_cm_to_pressure_unit: Number = 4.725e-4,
) -> tuple[Number, Number, Number, Number]:
    """
    Scalar variant of Leverett J-function capillary pressure derivatives.

    Returns (dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso).

    :param water_saturation: Water saturation (fraction, 0-1).
    :param gas_saturation: Gas saturation (fraction, 0-1).
    :param irreducible_water_saturation: Irreducible water saturation (swc).
    :param residual_oil_saturation_water: Residual oil saturation during water flooding (sorw).
    :param residual_oil_saturation_gas: Residual oil saturation during gas flooding (sorg).
    :param residual_gas_saturation: Residual gas saturation (sgr).
    :param permeability: Absolute permeability (mD).
    :param porosity: Porosity (fraction, 0-1).
    :param wettability: Wettability type.
    :param oil_water_interfacial_tension: Oil-water interfacial tension (dyne/cm).
    :param gas_oil_interfacial_tension: Gas-oil interfacial tension (dyne/cm).
    :param oil_water_contact_angle: Oil-water contact angle in degrees.
    :param gas_oil_contact_angle: Gas-oil contact angle in degrees.
    :param j_function_coefficient: Empirical coefficient 'a' in J(Se) = a * Se^(-b).
    :param j_function_exponent: Empirical exponent 'b' in J(Se) = a * Se^(-b).
    :param mixed_wet_water_fraction: Fraction of pore space that is water-wet (0-1).
    :param saturation_epsilon: Small value to avoid division by zero.
    :param minimum_mobile_pore_space: Minimum mobile pore space threshold.
    :param dyne_per_cm_to_pressure_unit: Conversion factor from dyne/cm to psi.
    :return: Tuple of (dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso).
    """
    sw = water_saturation
    sg = gas_saturation
    swc = irreducible_water_saturation
    sorw = residual_oil_saturation_water
    sorg = residual_oil_saturation_gas
    sgr = residual_gas_saturation

    zero = 0.0
    one = 1.0

    leverett_rock_factor = (
        np.sqrt(porosity / permeability) if permeability > 0.0 and porosity > 0.0 else 0.0
    )

    # Oil-water derivatives
    mobile_water_range = one - swc - sorw
    valid_water = (mobile_water_range > minimum_mobile_pore_space) and (leverett_rock_factor > 0.0)
    se_w = min(
        max((sw - swc) / mobile_water_range, saturation_epsilon),
        one - saturation_epsilon,
    )

    dpcow_dsw = zero
    if valid_water:
        dj_dse_w = (
            -j_function_coefficient * j_function_exponent * (se_w ** (-j_function_exponent - one))
        )
        cos_oil_water_angle = np.cos(np.deg2rad(oil_water_contact_angle))
        oil_water_scale = (
            oil_water_interfacial_tension
            * dyne_per_cm_to_pressure_unit
            * cos_oil_water_angle
            * leverett_rock_factor
        )

        if wettability == Wettability.WATER_WET:
            wettability_sign = one
        elif wettability == Wettability.OIL_WET:
            wettability_sign = -one
        else:  # MIXED_WET
            wettability_sign = 2.0 * mixed_wet_water_fraction - one

        dpcow_dsw = wettability_sign * oil_water_scale * dj_dse_w / mobile_water_range

    dpcow_dso = zero

    # Gas-oil derivatives
    mobile_gas_range = one - swc - sorg - sgr
    valid_gas = (mobile_gas_range > minimum_mobile_pore_space) and (leverett_rock_factor > 0.0)
    se_g = min(
        max((sg - sgr) / mobile_gas_range, saturation_epsilon),
        one - saturation_epsilon,
    )

    dpcgo_dsg = zero
    if valid_gas:
        dj_dse_g = (
            -j_function_coefficient * j_function_exponent * (se_g ** (-j_function_exponent - one))
        )
        cos_gas_oil_angle = np.cos(np.deg2rad(gas_oil_contact_angle))
        gas_oil_scale = (
            gas_oil_interfacial_tension
            * dyne_per_cm_to_pressure_unit
            * cos_gas_oil_angle
            * leverett_rock_factor
        )
        dpcgo_dsg = gas_oil_scale * dj_dse_g / mobile_gas_range

    dpcgo_dso = zero
    return dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso


@numba.njit(cache=True)
def compute_leverett_j_derivatives_array(
    water_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    permeability: NumberOrArray[NDimension],
    porosity: NumberOrArray[NDimension],
    wettability: Wettability,
    oil_water_interfacial_tension: Number,
    gas_oil_interfacial_tension: Number,
    oil_water_contact_angle: Number,
    gas_oil_contact_angle: Number,
    j_function_coefficient: Number,
    j_function_exponent: Number,
    mixed_wet_water_fraction: Number = 0.5,
    saturation_epsilon: Number = 1e-6,
    minimum_mobile_pore_space: Number = 1e-9,
    dyne_per_cm_to_pressure_unit: Number = 4.725e-4,
) -> tuple[
    NumberArray[NDimension],
    NumberArray[NDimension],
    NumberArray[NDimension],
    NumberArray[NDimension],
]:
    """
    Array variant of Leverett J-function capillary pressure derivatives.

    Returns (dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso) as NDArrays.

    :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
    :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
    :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
    :param irreducible_water_saturation: Irreducible water saturation (swc) - scalar or array.
    :param residual_oil_saturation_water: Residual oil saturation during water flooding (sorw) - scalar or array.
    :param residual_oil_saturation_gas: Residual oil saturation during gas flooding (sorg) - scalar or array.
    :param residual_gas_saturation: Residual gas saturation (sgr) - scalar or array.
    :param permeability: Absolute permeability (mD) - scalar or array.
    :param porosity: Porosity (fraction, 0-1) - scalar or array.
    :param wettability: Wettability type.
    :param oil_water_interfacial_tension: Oil-water interfacial tension (dyne/cm).
    :param gas_oil_interfacial_tension: Gas-oil interfacial tension (dyne/cm).
    :param oil_water_contact_angle: Oil-water contact angle in degrees.
    :param gas_oil_contact_angle: Gas-oil contact angle in degrees.
    :param j_function_coefficient: Empirical coefficient 'a' in J(Se) = a * Se^(-b).
    :param j_function_exponent: Empirical exponent 'b' in J(Se) = a * Se^(-b).
    :param mixed_wet_water_fraction: Fraction of pore space that is water-wet (0-1).
    :param saturation_epsilon: Small value to avoid division by zero.
    :param minimum_mobile_pore_space: Minimum mobile pore space threshold.
    :param dyne_per_cm_to_pressure_unit: Conversion factor from dyne/cm to psi.
    :return: Tuple of (dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso) as NDArrays.
    """
    sw = atleast_1d(water_saturation)
    sg = atleast_1d(gas_saturation)
    swc = atleast_1d(irreducible_water_saturation)
    sorw = atleast_1d(residual_oil_saturation_water)
    sorg = atleast_1d(residual_oil_saturation_gas)
    sgr = atleast_1d(residual_gas_saturation)
    perm = atleast_1d(permeability, sw.dtype)
    phi = atleast_1d(porosity, sw.dtype)

    dtype = sw.dtype.type
    one = dtype(1.0)
    zero = dtype(0.0)

    # Broadcast arrays
    sw, sg, swc, sorw, sorg, sgr, perm, phi = np.broadcast_arrays(
        sw, sg, swc, sorw, sorg, sgr, perm, phi
    )

    valid_rock = (perm > zero) & (phi > zero)
    leverett_rock_factor = np.where(valid_rock, np.sqrt(phi / perm), zero)

    # Oil-water derivatives
    mobile_water_range = one - swc - sorw
    valid_water = (mobile_water_range > minimum_mobile_pore_space) & valid_rock
    se_w = np.clip(
        (sw - swc) / np.where(valid_water, mobile_water_range, one),
        saturation_epsilon,
        one - saturation_epsilon,
    )

    dj_dse_w = (
        -dtype(j_function_coefficient)
        * dtype(j_function_exponent)
        * (se_w ** (-dtype(j_function_exponent) - one))
    )
    cos_oil_water_angle = np.cos(np.deg2rad(dtype(oil_water_contact_angle)))
    oil_water_scale = (
        dtype(oil_water_interfacial_tension)
        * dtype(dyne_per_cm_to_pressure_unit)
        * cos_oil_water_angle
        * leverett_rock_factor
    )

    if wettability == Wettability.WATER_WET:
        wettability_sign = one
    elif wettability == Wettability.OIL_WET:
        wettability_sign = -one
    else:  # MIXED_WET
        wettability_sign = dtype(2.0) * dtype(mixed_wet_water_fraction) - one

    dpcow_dsw = np.where(
        valid_water,
        wettability_sign * oil_water_scale * dj_dse_w / mobile_water_range,
        np.zeros_like(sw),
    )
    dpcow_dso = np.zeros_like(sw)

    # Gas-oil derivatives
    mobile_gas_range = one - swc - sorg - sgr
    valid_gas = (mobile_gas_range > minimum_mobile_pore_space) & valid_rock
    se_g = np.clip(
        (sg - sgr) / np.where(valid_gas, mobile_gas_range, one),
        saturation_epsilon,
        one - saturation_epsilon,
    )

    dj_dse_g = (
        -dtype(j_function_coefficient)
        * dtype(j_function_exponent)
        * (se_g ** (-dtype(j_function_exponent) - one))
    )
    cos_gas_oil_angle = np.cos(np.deg2rad(dtype(gas_oil_contact_angle)))
    gas_oil_scale = (
        dtype(gas_oil_interfacial_tension)
        * dtype(dyne_per_cm_to_pressure_unit)
        * cos_gas_oil_angle
        * leverett_rock_factor
    )
    dpcgo_dsg = np.where(
        valid_gas,
        gas_oil_scale * dj_dse_g / mobile_gas_range,
        np.zeros_like(sg),
    )
    dpcgo_dso = np.zeros_like(sg)
    return (
        typing.cast(NumberArray[NDimension], dpcow_dsw),
        typing.cast(NumberArray[NDimension], dpcow_dso),
        typing.cast(NumberArray[NDimension], dpcgo_dsg),
        typing.cast(NumberArray[NDimension], dpcgo_dso),
    )


def compute_leverett_j_derivatives(
    water_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    permeability: NumberOrArray[NDimension],
    porosity: NumberOrArray[NDimension],
    wettability: Wettability,
    oil_water_interfacial_tension: Number,
    gas_oil_interfacial_tension: Number,
    oil_water_contact_angle: Number,
    gas_oil_contact_angle: Number,
    j_function_coefficient: Number,
    j_function_exponent: Number,
    mixed_wet_water_fraction: Number = 0.5,
    saturation_epsilon: Number = 1e-6,
    minimum_mobile_pore_space: Number = 1e-9,
    dyne_per_cm_to_pressure_unit: Number = 4.725e-4,
) -> tuple[
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
]:
    """
    Dispatch function for Leverett J-function capillary pressure derivatives.

    Routes to the scalar variant when all inputs are Python scalars, otherwise
    routes to the array variant.

    :return: (dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso) as scalars or arrays.
    """
    if (
        np.isscalar(water_saturation)
        and np.isscalar(gas_saturation)
        and np.isscalar(irreducible_water_saturation)
        and np.isscalar(residual_oil_saturation_water)
        and np.isscalar(residual_oil_saturation_gas)
        and np.isscalar(residual_gas_saturation)
        and np.isscalar(permeability)
        and np.isscalar(porosity)
    ):
        return compute_leverett_j_derivatives_scalar(
            water_saturation=water_saturation,  # type: ignore[arg-type]
            gas_saturation=gas_saturation,  # type: ignore[arg-type]
            irreducible_water_saturation=irreducible_water_saturation,  # type: ignore[arg-type]
            residual_oil_saturation_water=residual_oil_saturation_water,  # type: ignore[arg-type]
            residual_oil_saturation_gas=residual_oil_saturation_gas,  # type: ignore[arg-type]
            residual_gas_saturation=residual_gas_saturation,  # type: ignore[arg-type]
            permeability=permeability,  # type: ignore[arg-type]
            porosity=porosity,  # type: ignore[arg-type]
            wettability=wettability,
            oil_water_interfacial_tension=oil_water_interfacial_tension,
            gas_oil_interfacial_tension=gas_oil_interfacial_tension,
            oil_water_contact_angle=oil_water_contact_angle,
            gas_oil_contact_angle=gas_oil_contact_angle,
            j_function_coefficient=j_function_coefficient,
            j_function_exponent=j_function_exponent,
            mixed_wet_water_fraction=mixed_wet_water_fraction,
            saturation_epsilon=saturation_epsilon,
            minimum_mobile_pore_space=minimum_mobile_pore_space,
            dyne_per_cm_to_pressure_unit=dyne_per_cm_to_pressure_unit,
        )
    return compute_leverett_j_derivatives_array(
        water_saturation=water_saturation,  # type: ignore[arg-type]
        gas_saturation=gas_saturation,  # type: ignore[arg-type]
        irreducible_water_saturation=irreducible_water_saturation,  # type: ignore[arg-type]
        residual_oil_saturation_water=residual_oil_saturation_water,  # type: ignore[arg-type]
        residual_oil_saturation_gas=residual_oil_saturation_gas,  # type: ignore[arg-type]
        residual_gas_saturation=residual_gas_saturation,  # type: ignore[arg-type]
        permeability=permeability,  # type: ignore[arg-type]
        porosity=porosity,  # type: ignore[arg-type]
        wettability=wettability,
        oil_water_interfacial_tension=oil_water_interfacial_tension,
        gas_oil_interfacial_tension=gas_oil_interfacial_tension,
        oil_water_contact_angle=oil_water_contact_angle,
        gas_oil_contact_angle=gas_oil_contact_angle,
        j_function_coefficient=j_function_coefficient,
        j_function_exponent=j_function_exponent,
        mixed_wet_water_fraction=mixed_wet_water_fraction,
        saturation_epsilon=saturation_epsilon,
        minimum_mobile_pore_space=minimum_mobile_pore_space,
        dyne_per_cm_to_pressure_unit=dyne_per_cm_to_pressure_unit,
    )


def convert_dyne_per_cm_to_pressure_unit(unit_system: UnitSystem) -> Number:
    dyne_per_cm_to_psi = c.DYNE_PER_CENTIMETER_TO_PSI
    if unit_system == UnitSystem.FIELD:
        return dyne_per_cm_to_psi
    factors = get_conversion_factors(UnitSystem.FIELD, unit_system)
    return dyne_per_cm_to_psi * factors["pressure"]


@attrs.frozen(slots=True)
class LeverettJCapillaryPressureTable(CapillaryPressureTable):
    """
    Implements the Leverett J-function capillary pressure model for three-phase systems.

    Uses dimensionless J-function correlation to relate capillary pressure
    to rock and fluid properties: Pc = σ * cos(θ) * sqrt(φ/k) * J(Se)

    Useful when capillary pressure data needs to be scaled across different
    rock types or fluid systems.
    """

    irreducible_water_saturation: Number | None = None
    """Default irreducible water saturation (swc). Can be overridden per call."""

    residual_oil_saturation_water: Number | None = None
    """Default residual oil saturation after water flood (sorw). Can be overridden per call."""

    residual_oil_saturation_gas: Number | None = None
    """Default residual oil saturation after gas flood (sorg). Can be overridden per call."""

    residual_gas_saturation: Number | None = None
    """Default residual gas saturation (sgr). Can be overridden per call."""

    permeability: Number = 100.0
    """Absolute permeability (mD)."""

    porosity: Number = 0.2
    """Porosity (fraction, 0-1)."""

    oil_water_interfacial_tension: Number = 30.0
    """Oil-water interfacial tension (dyne/cm)."""

    gas_oil_interfacial_tension: Number = 20.0
    """Gas-oil interfacial tension (dyne/cm)."""

    oil_water_contact_angle: Number = 0.0
    """Oil-water contact angle in degrees (0° = water-wet, 180° = oil-wet)."""

    gas_oil_contact_angle: Number = 0.0
    """Gas-oil contact angle in degrees (0° = oil-wet to gas)."""

    mixed_wet_water_fraction: Number = 0.5
    """Fraction of pore space that is water-wet in mixed-wet systems (0-1)."""

    wettability: Wettability = Wettability.WATER_WET
    """Wettability type (affects sign convention)."""

    j_function_coefficient: Number = 0.5
    """Empirical coefficient 'a' in J(Se) = a * Se^(-b). Fit to core data (can be tuned to match experimental data)."""

    j_function_exponent: Number = 0.5
    """Empirical exponent 'b' in J(Se) = a * Se^(-b). Fit to core data (can be tuned to match experimental data)."""

    unit_system: UnitSystem = attrs.field(default=UnitSystem.FIELD)
    """
    Unit system in which the dimensioned parameters of this model are expressed.

    Saturation parameters and dimensionless ratios are unaffected by unit conversion.

    Use `convert(target)` to produce a copy of this model rescaled to
    another `UnitSystem`.
    """

    supports_vector: bool = attrs.field(init=False, repr=False, default=True)
    """Flag indicating support for array inputs."""

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

    def evaluate(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        irreducible_water_saturation: NumberOrArray[NDimension] | None = None,
        residual_oil_saturation_water: NumberOrArray[NDimension] | None = None,
        residual_oil_saturation_gas: NumberOrArray[NDimension] | None = None,
        residual_gas_saturation: NumberOrArray[NDimension] | None = None,
        porosity: NumberOrArray[NDimension] | None = None,
        permeability: NumberOrArray[NDimension] | None = None,
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures using Leverett J-function.

        Supports both scalar and array inputs.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param irreducible_water_saturation: Optional override for swc - scalar or array.
        :param residual_oil_saturation_water: Optional override for sorw - scalar or array.
        :param residual_oil_saturation_gas: Optional override for sorg - scalar or array.
        :param residual_gas_saturation: Optional override for sgr - scalar or array.
        :param porosity: Optional override for porosity - scalar or array.
        :param permeability: Optional override for permeability - scalar or array.
        :return: `CapillaryPressures` dictionary.
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
        porosity = porosity if porosity is not None else self.porosity
        permeability = permeability if permeability is not None else self.permeability

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
                f"Residual saturations must be provided either as model defaults or in the call. "
                f"Missing: {', '.join(params_missing)}"
            )

        pcow, pcgo = compute_leverett_j_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=sorg,  # type: ignore[arg-type]
            residual_gas_saturation=sgr,  # type: ignore[arg-type]
            permeability=permeability,
            porosity=porosity,
            oil_water_interfacial_tension=self.oil_water_interfacial_tension,
            gas_oil_interfacial_tension=self.gas_oil_interfacial_tension,
            oil_water_contact_angle=self.oil_water_contact_angle,
            gas_oil_contact_angle=self.gas_oil_contact_angle,
            j_function_coefficient=self.j_function_coefficient,
            j_function_exponent=self.j_function_exponent,
            mixed_wet_water_fraction=self.mixed_wet_water_fraction,
            wettability=self.wettability,
            saturation_epsilon=c.SATURATION_EPSILON,
            minimum_mobile_pore_space=c.MINIMUM_MOBILE_PORE_SPACE,
            dyne_per_cm_to_pressure_unit=convert_dyne_per_cm_to_pressure_unit(self.unit_system),
        )
        return CapillaryPressures(oil_water=pcow, gas_oil=pcgo)  # type: ignore[typeddict-item]

    def derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        irreducible_water_saturation: NumberOrArray[NDimension] | None = None,
        residual_oil_saturation_water: NumberOrArray[NDimension] | None = None,
        residual_oil_saturation_gas: NumberOrArray[NDimension] | None = None,
        residual_gas_saturation: NumberOrArray[NDimension] | None = None,
        porosity: NumberOrArray[NDimension] | None = None,
        permeability: NumberOrArray[NDimension] | None = None,
        **kwargs: typing.Any,
    ) -> CapillaryPressureDerivatives:
        """
        Compute the partial derivatives of the Leverett J-function oil-water and
        gas-oil capillary pressures with respect to saturation.

        Returns a dictionary containing:

        ```
        (dpcow / dsw, dpcow / dso, dpcgo / dsg)
        ```

        - `dpcow/dsw`: analytically derived via the chain rule through
        effective water saturation.
        - `dpcow/dso`: zero - the oil-water capillary pressure depends only on
        water saturation in this model.
        - `dpcgo/dsg`: analytically derived via the chain rule through
        effective gas saturation.

        The Leverett J-function model is:

        ```
        Pc = sigma * cos(theta) * sqrt(porosity / permeability) * J(Se)
        J(Se) = j_function_coefficient * Se^(-j_function_exponent)
        ```

        Analytical derivative:

        ```
        dPc/dSe = sigma * cos(theta) * sqrt(phi/k)
                * j_function_coefficient * (-j_function_exponent)
                * Se^(-j_function_exponent - 1)
        dPc/dsw = dPc/dSe / mobile_water_saturation_range
        ```

        Unit conversion from dyne/cm to psi uses the factor 4.725.

        :param water_saturation: Water saturation (scalar or array).
        :param oil_saturation: Oil saturation (scalar or array).
        :param gas_saturation: Gas saturation (scalar or array).
        :param irreducible_water_saturation: Optional override for the
            irreducible (connate) water saturation.
        :param residual_oil_saturation_water: Optional override for the residual
            oil saturation to water flooding.
        :param residual_oil_saturation_gas: Optional override for the residual
            oil saturation to gas flooding.
        :param residual_gas_saturation: Optional override for the residual gas
            saturation.
        :param porosity: Optional override for porosity.
        :param permeability: Optional override for permeability.
        :return: `CapillaryPressureDerivatives` dictionary containing the partial derivatives as described above.
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
        porosity = porosity if porosity is not None else self.porosity
        permeability = permeability if permeability is not None else self.permeability

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
                f"Residual saturations must be provided either as model defaults or in the call. "
                f"Missing: {', '.join(params_missing)}"
            )

        dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso = compute_leverett_j_derivatives(
            water_saturation=water_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=sorg,  # type: ignore[arg-type]
            residual_gas_saturation=sgr,  # type: ignore[arg-type]
            permeability=permeability,
            porosity=porosity,
            wettability=self.wettability,
            oil_water_interfacial_tension=self.oil_water_interfacial_tension,
            gas_oil_interfacial_tension=self.gas_oil_interfacial_tension,
            oil_water_contact_angle=self.oil_water_contact_angle,
            gas_oil_contact_angle=self.gas_oil_contact_angle,
            j_function_coefficient=self.j_function_coefficient,
            j_function_exponent=self.j_function_exponent,
            mixed_wet_water_fraction=self.mixed_wet_water_fraction,
            saturation_epsilon=c.SATURATION_EPSILON,
            minimum_mobile_pore_space=c.MINIMUM_MOBILE_PORE_SPACE,
            dyne_per_cm_to_pressure_unit=convert_dyne_per_cm_to_pressure_unit(self.unit_system),
        )
        return CapillaryPressureDerivatives(
            dpcow_dsw=dpcow_dsw,
            dpcow_dso=dpcow_dso,
            dpcgo_dsg=dpcgo_dsg,
            dpcgo_dso=dpcgo_dso,
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `LeverettJCapillaryPressureTable` rescaled to *target*.

        `permeability` is rescaled from `self.unit_system` to *target* using
        the `permeability` conversion factor so that the rock factor
        `sqrt(phi/k)` remains physically correct.

        :param target: Target `UnitSystem`.
        :returns: New `LeverettJCapillaryPressureTable` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        return attrs.evolve(
            self,
            permeability=self.permeability * factors["permeability"],
            unit_system=target,
        )
