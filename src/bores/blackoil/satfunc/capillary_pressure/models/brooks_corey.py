"""Brooks-Corey capillary pressure model and table."""

import typing

import attrs
import numba
import numpy as np
from typing_extensions import Self

from bores.blackoil.satfunc.capillary_pressure.tables import CapillaryPressureTable
from bores.blackoil.satfunc.utils import (
    element_at,
    get_floating_dtype,
    prepare_kernel_inputs,
    saturations_within_bounds,
    zeros_like_value,
)
from bores.constants import c
from bores.errors import ValidationError
from bores.types import (
    CapillaryPressureDerivatives,
    CapillaryPressures,
    FluidPhase,
    Integer,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
    UnitConversionTable,
    UnitSystem,
    Wettability,
)
from bores.units import get_conversion_factors
from bores.utils import scale

__all__ = [
    "BrooksCoreyCapillaryPressureTable",
]


@numba.njit(cache=True, inline="always", error_model="numpy")
def compute_brooks_corey_capillary_pressures_cell(
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
    Capillary pressures of a single cell, without any validation of the inputs.

    This is the per-cell kernel shared by the scalar function and the array loop. It has the
    same parameters as the scalar function.

    :return: Tuple of (oil_water_capillary_pressure, gas_oil_capillary_pressure).
    """
    sw: Number = water_saturation
    so: Number = oil_saturation
    sg: Number = gas_saturation
    swc: Number = irreducible_water_saturation
    sorw: Number = residual_oil_saturation_water
    sorg: Number = residual_oil_saturation_gas
    sgr: Number = residual_gas_saturation

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
        effective_liquid_saturation = (1.0 - swc - sg - sorg) / total_mobile_pore_space_gas
        effective_liquid_saturation = min(
            max(effective_liquid_saturation, saturation_epsilon), 1.0
        )

        if effective_liquid_saturation < 1.0 - saturation_epsilon:
            gas_oil_capillary_pressure = gas_oil_entry_pressure * (
                effective_liquid_saturation ** (-1.0 / gas_oil_pore_size_distribution_index)
            )

    return oil_water_capillary_pressure, gas_oil_capillary_pressure


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
    if sw < 0 or sw > 1 or so < 0 or so > 1 or sg < 0 or sg > 1:
        raise ValidationError("Saturations must be between 0 and 1.")
    return compute_brooks_corey_capillary_pressures_cell(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        irreducible_water_saturation=irreducible_water_saturation,
        residual_oil_saturation_water=residual_oil_saturation_water,
        residual_oil_saturation_gas=residual_oil_saturation_gas,
        residual_gas_saturation=residual_gas_saturation,
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


@numba.njit(cache=True, parallel=True, error_model="numpy")
def compute_brooks_corey_capillary_pressures_loop(
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    irreducible_water_saturation: NumberArray[OneDimension],
    residual_oil_saturation_water: NumberArray[OneDimension],
    residual_oil_saturation_gas: NumberArray[OneDimension],
    residual_gas_saturation: NumberArray[OneDimension],
    wettability: Wettability,
    oil_water_entry_pressure_water_wet: Number,
    oil_water_entry_pressure_oil_wet: Number,
    oil_water_pore_size_distribution_index_water_wet: Number,
    oil_water_pore_size_distribution_index_oil_wet: Number,
    gas_oil_entry_pressure: Number,
    gas_oil_pore_size_distribution_index: Number,
    mixed_wet_water_fraction: Number,
    saturation_epsilon: Number,
    minimum_mobile_pore_space: Number,
    size: Integer,
    oil_water_capillary_pressure: NumberArray[OneDimension],
    gas_oil_capillary_pressure: NumberArray[OneDimension],
) -> None:
    """
    Compute the capillary pressures of `size` cells in parallel and write them to the output arrays.

    Per-cell inputs are flat arrays, or a single value shared by every cell.
    """
    for i in numba.prange(size):
        oil_water, gas_oil = compute_brooks_corey_capillary_pressures_cell(
            water_saturation=np.float64(element_at(array=water_saturation, index=i)),
            oil_saturation=np.float64(element_at(array=oil_saturation, index=i)),
            gas_saturation=np.float64(element_at(array=gas_saturation, index=i)),
            irreducible_water_saturation=np.float64(
                element_at(array=irreducible_water_saturation, index=i)
            ),
            residual_oil_saturation_water=np.float64(
                element_at(array=residual_oil_saturation_water, index=i)
            ),
            residual_oil_saturation_gas=np.float64(
                element_at(array=residual_oil_saturation_gas, index=i)
            ),
            residual_gas_saturation=np.float64(element_at(array=residual_gas_saturation, index=i)),
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
        oil_water_capillary_pressure[i] = oil_water
        gas_oil_capillary_pressure[i] = gas_oil


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
    (
        shape,
        size,
        (
            water_saturation_flat,
            oil_saturation_flat,
            gas_saturation_flat,
            irreducible_water_saturation_flat,
            residual_oil_saturation_water_flat,
            residual_oil_saturation_gas_flat,
            residual_gas_saturation_flat,
        ),
    ) = prepare_kernel_inputs(
        water_saturation,
        oil_saturation,
        gas_saturation,
        irreducible_water_saturation,
        residual_oil_saturation_water,
        residual_oil_saturation_gas,
        residual_gas_saturation,
    )
    if not saturations_within_bounds(
        water_saturation=water_saturation_flat,
        oil_saturation=oil_saturation_flat,
        gas_saturation=gas_saturation_flat,
        size=size,
    ):
        raise ValidationError("Saturations must be between 0 and 1.")

    dtype = get_floating_dtype(water_saturation)
    oil_water_capillary_pressure = np.empty(size, dtype=dtype)
    gas_oil_capillary_pressure = np.empty(size, dtype=dtype)
    compute_brooks_corey_capillary_pressures_loop(
        water_saturation=water_saturation_flat,
        oil_saturation=oil_saturation_flat,
        gas_saturation=gas_saturation_flat,
        irreducible_water_saturation=irreducible_water_saturation_flat,
        residual_oil_saturation_water=residual_oil_saturation_water_flat,
        residual_oil_saturation_gas=residual_oil_saturation_gas_flat,
        residual_gas_saturation=residual_gas_saturation_flat,
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
        size=size,
        oil_water_capillary_pressure=oil_water_capillary_pressure,
        gas_oil_capillary_pressure=gas_oil_capillary_pressure,
    )
    return typing.cast(
        "NumberArray[NDimension]", oil_water_capillary_pressure.reshape(shape)
    ), typing.cast("NumberArray[NDimension]", gas_oil_capillary_pressure.reshape(shape))


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


@numba.njit(cache=True, inline="always", error_model="numpy")
def compute_brooks_corey_derivatives_scalar(
    water_saturation: Number,
    oil_saturation: Number,
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
    :param oil_saturation: Oil saturation (fraction, 0 to 1).
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
    total_saturation = water_saturation + oil_saturation + gas_saturation
    if abs(total_saturation - 1.0) > saturation_epsilon and total_saturation > 0.0:
        water_saturation = water_saturation / total_saturation
        gas_saturation = gas_saturation / total_saturation

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

    # No slope where the value is clipped (flat) or set to zero
    raw_se_w = (sw - swc) / mobile_water_range
    dpcow_dsw = zero
    if valid_water and raw_se_w > saturation_epsilon and raw_se_w < one - saturation_epsilon:
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
    valid_gas = mobile_gas_range > minimum_mobile_pore_space

    # Pcgo depends on the effective liquid saturation of the gas-oil system, which falls as
    # Sg rises. It has no slope where the value is clipped or set to zero.
    dpcgo_dsg = zero
    if valid_gas:
        se_l = (one - swc - sg - sorg) / mobile_gas_range
        if se_l > saturation_epsilon and se_l < one - saturation_epsilon:
            exp_gas_oil = -one / gas_oil_pore_size_distribution_index
            dpcgo_dse_l = gas_oil_entry_pressure * exp_gas_oil * (se_l ** (exp_gas_oil - one))
            dpcgo_dsg = -dpcgo_dse_l / mobile_gas_range

    dpcgo_dso = zero

    return dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso


@numba.njit(cache=True, parallel=True, error_model="numpy")
def compute_brooks_corey_derivatives_loop(
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    irreducible_water_saturation: NumberArray[OneDimension],
    residual_oil_saturation_water: NumberArray[OneDimension],
    residual_oil_saturation_gas: NumberArray[OneDimension],
    residual_gas_saturation: NumberArray[OneDimension],
    wettability: Wettability,
    oil_water_pore_size_distribution_index_water_wet: Number,
    oil_water_pore_size_distribution_index_oil_wet: Number,
    oil_water_entry_pressure_water_wet: Number,
    oil_water_entry_pressure_oil_wet: Number,
    gas_oil_pore_size_distribution_index: Number,
    gas_oil_entry_pressure: Number,
    mixed_wet_water_fraction: Number,
    saturation_epsilon: Number,
    minimum_mobile_pore_space: Number,
    size: Integer,
    dpcow_dsw: NumberArray[OneDimension],
    dpcow_dso: NumberArray[OneDimension],
    dpcgo_dsg: NumberArray[OneDimension],
    dpcgo_dso: NumberArray[OneDimension],
) -> None:
    """
    Compute the capillary pressure derivatives of `size` cells in parallel and write them to the
    output arrays.

    Per-cell inputs are flat arrays, or a single value shared by every cell.
    """
    for i in numba.prange(size):
        d_ow_sw, d_ow_so, d_go_sg, d_go_so = compute_brooks_corey_derivatives_scalar(
            water_saturation=np.float64(element_at(array=water_saturation, index=i)),
            oil_saturation=np.float64(element_at(array=oil_saturation, index=i)),
            gas_saturation=np.float64(element_at(array=gas_saturation, index=i)),
            irreducible_water_saturation=np.float64(
                element_at(array=irreducible_water_saturation, index=i)
            ),
            residual_oil_saturation_water=np.float64(
                element_at(array=residual_oil_saturation_water, index=i)
            ),
            residual_oil_saturation_gas=np.float64(
                element_at(array=residual_oil_saturation_gas, index=i)
            ),
            residual_gas_saturation=np.float64(element_at(array=residual_gas_saturation, index=i)),
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
        dpcow_dsw[i] = d_ow_sw
        dpcow_dso[i] = d_ow_so
        dpcgo_dsg[i] = d_go_sg
        dpcgo_dso[i] = d_go_so


def compute_brooks_corey_derivatives_array(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
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
    :param oil_saturation: Oil saturation (fraction, 0 to 1).
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
    (
        shape,
        size,
        (
            water_saturation_flat,
            oil_saturation_flat,
            gas_saturation_flat,
            irreducible_water_saturation_flat,
            residual_oil_saturation_water_flat,
            residual_oil_saturation_gas_flat,
            residual_gas_saturation_flat,
        ),
    ) = prepare_kernel_inputs(
        water_saturation,
        oil_saturation,
        gas_saturation,
        irreducible_water_saturation,
        residual_oil_saturation_water,
        residual_oil_saturation_gas,
        residual_gas_saturation,
    )
    dtype = get_floating_dtype(water_saturation)
    dpcow_dsw = np.empty(size, dtype=dtype)
    dpcow_dso = np.empty(size, dtype=dtype)
    dpcgo_dsg = np.empty(size, dtype=dtype)
    dpcgo_dso = np.empty(size, dtype=dtype)
    compute_brooks_corey_derivatives_loop(
        water_saturation=water_saturation_flat,
        oil_saturation=oil_saturation_flat,
        gas_saturation=gas_saturation_flat,
        irreducible_water_saturation=irreducible_water_saturation_flat,
        residual_oil_saturation_water=residual_oil_saturation_water_flat,
        residual_oil_saturation_gas=residual_oil_saturation_gas_flat,
        residual_gas_saturation=residual_gas_saturation_flat,
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
        size=size,
        dpcow_dsw=dpcow_dsw,
        dpcow_dso=dpcow_dso,
        dpcgo_dsg=dpcgo_dsg,
        dpcgo_dso=dpcgo_dso,
    )
    return (
        typing.cast("NumberArray[NDimension]", dpcow_dsw.reshape(shape)),
        typing.cast("NumberArray[NDimension]", dpcow_dso.reshape(shape)),
        typing.cast("NumberArray[NDimension]", dpcgo_dsg.reshape(shape)),
        typing.cast("NumberArray[NDimension]", dpcgo_dso.reshape(shape)),
    )


def compute_brooks_corey_capillary_pressure_derivatives(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
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
        and np.isscalar(oil_saturation)
        and np.isscalar(gas_saturation)
        and np.isscalar(irreducible_water_saturation)
        and np.isscalar(residual_oil_saturation_water)
        and np.isscalar(residual_oil_saturation_gas)
        and np.isscalar(residual_gas_saturation)
    ):
        return compute_brooks_corey_derivatives_scalar(
            water_saturation=water_saturation,  # type: ignore[arg-type]
            oil_saturation=oil_saturation,  # type: ignore[arg-type]
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
        oil_saturation=oil_saturation,  # type: ignore[arg-type]
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


@numba.njit(cache=True, error_model="numpy")
def compute_brooks_corey_capillary_pressures_and_derivatives_cell(
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
) -> tuple[Number, Number, Number, Number, Number, Number]:
    """
    Capillary pressures and their derivatives with respect to saturation for one cell, from a
    single evaluation of the power functions that both need.

    The pressures are identical to those of the value kernel. The derivatives equal those of the
    derivative kernel up to rounding, and share the same saturations and the same activation
    conditions. There is no validation of the inputs.

    :return: Tuple of oil-water and gas-oil capillary pressure, `dpcow_dsw`, `dpcow_dso`,
        `dpcgo_dsg` and `dpcgo_dso`.
    """
    sw: Number = water_saturation
    so: Number = oil_saturation
    sg: Number = gas_saturation
    swc: Number = irreducible_water_saturation
    sorw: Number = residual_oil_saturation_water
    sorg: Number = residual_oil_saturation_gas
    sgr: Number = residual_gas_saturation

    # Normalize saturations if they do not sum to 1
    total_saturation = sw + so + sg
    if abs(total_saturation - 1.0) > saturation_epsilon and total_saturation > 0.0:
        sw = sw / total_saturation
        sg = sg / total_saturation

    total_mobile_pore_space_water = 1.0 - swc - sorw
    total_mobile_pore_space_gas = 1.0 - swc - sorg - sgr

    oil_water_capillary_pressure: Number = 0.0
    dpcow_dsw: Number = 0.0
    if total_mobile_pore_space_water > minimum_mobile_pore_space:
        raw_effective_water_saturation = (sw - swc) / total_mobile_pore_space_water
        effective_water_saturation = min(
            max(raw_effective_water_saturation, saturation_epsilon), 1.0
        )

        if effective_water_saturation < 1.0 - saturation_epsilon:
            # The slope of `entry_pressure * Se ** exponent` is `pc * exponent / Se`, so the power
            # is evaluated once and shared by the value and the slope.
            dpcow_dse_w: Number = 0.0
            if wettability == Wettability.WATER_WET:
                exponent_ww = -1.0 / oil_water_pore_size_distribution_index_water_wet
                pcow_ww = oil_water_entry_pressure_water_wet * (
                    effective_water_saturation**exponent_ww
                )
                oil_water_capillary_pressure = pcow_ww
                dpcow_dse_w = pcow_ww * exponent_ww / effective_water_saturation
            elif wettability == Wettability.OIL_WET:
                exponent_ow = -1.0 / oil_water_pore_size_distribution_index_oil_wet
                pcow_ow = -(
                    oil_water_entry_pressure_oil_wet * effective_water_saturation**exponent_ow
                )
                oil_water_capillary_pressure = pcow_ow
                dpcow_dse_w = pcow_ow * exponent_ow / effective_water_saturation
            elif wettability == Wettability.MIXED_WET:
                exponent_ww = -1.0 / oil_water_pore_size_distribution_index_water_wet
                exponent_ow = -1.0 / oil_water_pore_size_distribution_index_oil_wet
                pcow_ww = oil_water_entry_pressure_water_wet * (
                    effective_water_saturation**exponent_ww
                )
                pcow_ow = -(
                    oil_water_entry_pressure_oil_wet * effective_water_saturation**exponent_ow
                )
                oil_water_capillary_pressure = (
                    mixed_wet_water_fraction * pcow_ww + (1.0 - mixed_wet_water_fraction) * pcow_ow
                )
                dpcow_dse_w = mixed_wet_water_fraction * (
                    pcow_ww * exponent_ww / effective_water_saturation
                ) + (1.0 - mixed_wet_water_fraction) * (
                    pcow_ow * exponent_ow / effective_water_saturation
                )
            if raw_effective_water_saturation > saturation_epsilon:
                dpcow_dsw = dpcow_dse_w / total_mobile_pore_space_water

    gas_oil_capillary_pressure: Number = 0.0
    dpcgo_dsg: Number = 0.0
    if total_mobile_pore_space_gas > minimum_mobile_pore_space:
        raw_effective_liquid_saturation = (1.0 - swc - sg - sorg) / total_mobile_pore_space_gas
        effective_liquid_saturation = min(
            max(raw_effective_liquid_saturation, saturation_epsilon), 1.0
        )

        if effective_liquid_saturation < 1.0 - saturation_epsilon:
            exponent_gas_oil = -1.0 / gas_oil_pore_size_distribution_index
            gas_oil_capillary_pressure = gas_oil_entry_pressure * (
                effective_liquid_saturation**exponent_gas_oil
            )
            if raw_effective_liquid_saturation > saturation_epsilon:
                dpcgo_dse_l = (
                    gas_oil_capillary_pressure * exponent_gas_oil / effective_liquid_saturation
                )
                dpcgo_dsg = -dpcgo_dse_l / total_mobile_pore_space_gas

    return oil_water_capillary_pressure, gas_oil_capillary_pressure, dpcow_dsw, 0.0, dpcgo_dsg, 0.0


@numba.njit(cache=True, parallel=True, error_model="numpy")
def compute_brooks_corey_capillary_pressures_and_derivatives_loop(
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    irreducible_water_saturation: NumberArray[OneDimension],
    residual_oil_saturation_water: NumberArray[OneDimension],
    residual_oil_saturation_gas: NumberArray[OneDimension],
    residual_gas_saturation: NumberArray[OneDimension],
    wettability: Wettability,
    oil_water_entry_pressure_water_wet: Number,
    oil_water_entry_pressure_oil_wet: Number,
    oil_water_pore_size_distribution_index_water_wet: Number,
    oil_water_pore_size_distribution_index_oil_wet: Number,
    gas_oil_entry_pressure: Number,
    gas_oil_pore_size_distribution_index: Number,
    mixed_wet_water_fraction: Number,
    saturation_epsilon: Number,
    minimum_mobile_pore_space: Number,
    size: Integer,
    oil_water_capillary_pressure: NumberArray[OneDimension],
    gas_oil_capillary_pressure: NumberArray[OneDimension],
    dpcow_dsw: NumberArray[OneDimension],
    dpcow_dso: NumberArray[OneDimension],
    dpcgo_dsg: NumberArray[OneDimension],
    dpcgo_dso: NumberArray[OneDimension],
) -> None:
    """
    Compute the capillary pressures and their derivatives of `size` cells in parallel and write
    them to the output arrays.

    Per-cell inputs are flat arrays, or a single value shared by every cell.
    """
    for i in numba.prange(size):
        oil_water, gas_oil, d_ow_sw, d_ow_so, d_go_sg, d_go_so = (
            compute_brooks_corey_capillary_pressures_and_derivatives_cell(
                water_saturation=np.float64(element_at(array=water_saturation, index=i)),
                oil_saturation=np.float64(element_at(array=oil_saturation, index=i)),
                gas_saturation=np.float64(element_at(array=gas_saturation, index=i)),
                irreducible_water_saturation=np.float64(
                    element_at(array=irreducible_water_saturation, index=i)
                ),
                residual_oil_saturation_water=np.float64(
                    element_at(array=residual_oil_saturation_water, index=i)
                ),
                residual_oil_saturation_gas=np.float64(
                    element_at(array=residual_oil_saturation_gas, index=i)
                ),
                residual_gas_saturation=np.float64(
                    element_at(array=residual_gas_saturation, index=i)
                ),
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
        )
        oil_water_capillary_pressure[i] = oil_water
        gas_oil_capillary_pressure[i] = gas_oil
        dpcow_dsw[i] = d_ow_sw
        dpcow_dso[i] = d_ow_so
        dpcgo_dsg[i] = d_go_sg
        dpcgo_dso[i] = d_go_so


def compute_brooks_corey_capillary_pressures_and_derivatives_array(
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
) -> tuple[
    tuple[NumberArray[NDimension], NumberArray[NDimension]],
    tuple[
        NumberArray[NDimension],
        NumberArray[NDimension],
        NumberArray[NDimension],
        NumberArray[NDimension],
    ],
]:
    """
    Compute the capillary pressures and their derivatives for array inputs in one pass.

    The saturations are validated as in the value function. Unlike the separate functions, the
    power functions shared by the pressure and its slope are evaluated once per cell.

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
    :return: Tuple of the (oil-water, gas-oil) capillary pressures and the (`dpcow_dsw`,
        `dpcow_dso`, `dpcgo_dsg`, `dpcgo_dso`) derivatives.
    """
    (
        shape,
        size,
        (
            water_saturation_flat,
            oil_saturation_flat,
            gas_saturation_flat,
            irreducible_water_saturation_flat,
            residual_oil_saturation_water_flat,
            residual_oil_saturation_gas_flat,
            residual_gas_saturation_flat,
        ),
    ) = prepare_kernel_inputs(
        water_saturation,
        oil_saturation,
        gas_saturation,
        irreducible_water_saturation,
        residual_oil_saturation_water,
        residual_oil_saturation_gas,
        residual_gas_saturation,
    )
    if not saturations_within_bounds(
        water_saturation=water_saturation_flat,
        oil_saturation=oil_saturation_flat,
        gas_saturation=gas_saturation_flat,
        size=size,
    ):
        raise ValidationError("Saturations must be between 0 and 1.")

    dtype = get_floating_dtype(water_saturation)
    oil_water_capillary_pressure = np.empty(size, dtype=dtype)
    gas_oil_capillary_pressure = np.empty(size, dtype=dtype)
    dpcow_dsw = np.empty(size, dtype=dtype)
    dpcow_dso = np.empty(size, dtype=dtype)
    dpcgo_dsg = np.empty(size, dtype=dtype)
    dpcgo_dso = np.empty(size, dtype=dtype)
    compute_brooks_corey_capillary_pressures_and_derivatives_loop(
        water_saturation=water_saturation_flat,
        oil_saturation=oil_saturation_flat,
        gas_saturation=gas_saturation_flat,
        irreducible_water_saturation=irreducible_water_saturation_flat,
        residual_oil_saturation_water=residual_oil_saturation_water_flat,
        residual_oil_saturation_gas=residual_oil_saturation_gas_flat,
        residual_gas_saturation=residual_gas_saturation_flat,
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
        size=size,
        oil_water_capillary_pressure=oil_water_capillary_pressure,
        gas_oil_capillary_pressure=gas_oil_capillary_pressure,
        dpcow_dsw=dpcow_dsw,
        dpcow_dso=dpcow_dso,
        dpcgo_dsg=dpcgo_dsg,
        dpcgo_dso=dpcgo_dso,
    )
    return (
        typing.cast("NumberArray[NDimension]", oil_water_capillary_pressure.reshape(shape)),
        typing.cast("NumberArray[NDimension]", gas_oil_capillary_pressure.reshape(shape)),
    ), (
        typing.cast("NumberArray[NDimension]", dpcow_dsw.reshape(shape)),
        typing.cast("NumberArray[NDimension]", dpcow_dso.reshape(shape)),
        typing.cast("NumberArray[NDimension]", dpcgo_dsg.reshape(shape)),
        typing.cast("NumberArray[NDimension]", dpcgo_dso.reshape(shape)),
    )


def compute_brooks_corey_capillary_pressures_and_derivatives(
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
) -> tuple[
    tuple[NumberOrArray[NDimension], NumberOrArray[NDimension]],
    tuple[
        NumberOrArray[NDimension],
        NumberOrArray[NDimension],
        NumberOrArray[NDimension],
        NumberOrArray[NDimension],
    ],
]:
    """
    Dispatch function for the Brooks-Corey fused capillary pressure and derivative computation.

    Computes the capillary pressures and their derivatives together, evaluating the power
    functions that both need once per cell. The results match `compute_brooks_corey_capillary_pressures` and the derivative function of this
    model. Scalar inputs give scalar results.

    :return: Tuple of the (oil-water, gas-oil) capillary pressures and the (`dpcow_dsw`,
        `dpcow_dso`, `dpcgo_dsg`, `dpcgo_dso`) derivatives.
    """
    is_scalar = (
        np.isscalar(water_saturation)
        and np.isscalar(oil_saturation)
        and np.isscalar(gas_saturation)
        and np.isscalar(irreducible_water_saturation)
        and np.isscalar(residual_oil_saturation_water)
        and np.isscalar(residual_oil_saturation_gas)
        and np.isscalar(residual_gas_saturation)
    )
    pressures, derivatives = compute_brooks_corey_capillary_pressures_and_derivatives_array(
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
    if is_scalar:
        return (
            (pressures[0].item(), pressures[1].item()),
            (
                derivatives[0].item(),
                derivatives[1].item(),
                derivatives[2].item(),
                derivatives[3].item(),
            ),
        )
    return pressures, derivatives


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
                f"Residual saturations must be provided either as model defaults or in the call. "
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
    ) -> tuple[CapillaryPressures, CapillaryPressureDerivatives]:
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
        swc, sorw, sorg, sgr = self.resolve_endpoint_saturations(
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
            residual_gas_saturation=residual_gas_saturation,
        )

        (pcow, pcgo), (dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso) = (
            compute_brooks_corey_capillary_pressures_and_derivatives(
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
        )
        return (
            CapillaryPressures(oil_water=pcow, gas_oil=pcgo, gas_water=zeros_like_value(pcow)),  # type: ignore[typeddict-item]
            CapillaryPressureDerivatives(
                dpcow_dsw=dpcow_dsw,
                dpcow_dso=dpcow_dso,
                dpcgo_dsg=dpcgo_dsg,
                dpcgo_dso=dpcgo_dso,
                dpcgw_dsg=zeros_like_value(dpcgo_dsg),
            ),
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
        swc, sorw, sorg, sgr = self.resolve_endpoint_saturations(
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
            residual_gas_saturation=residual_gas_saturation,
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
        return CapillaryPressures(oil_water=pcow, gas_oil=pcgo, gas_water=zeros_like_value(pcow))  # type: ignore[typeddict-item]

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
        Pcgo = Pd_go * Se_l ^ (-1 / lambda_go)
        ```

        where effective water saturation:
        ```
        Se_w = (Sw - irreducible water saturation)
            / (1 - irreducible water saturation
                    - residual oil saturation to water flooding)
        ```

        and effective liquid saturation of the gas-oil system (the wetting liquid, oil at
        connate water, against the gas):

        ```
        Se_l = (1 - irreducible water saturation - Sg
                    - residual oil saturation to gas flooding)
            / (1 - irreducible water saturation
                    - residual gas saturation
                    - residual oil saturation to gas flooding)
        ```

        `Se_l` falls as gas displaces oil, so `Pcgo` rises with `Sg`. It is a function of `Sg`
        only, and has no slope where it is clipped (no mobile gas, or the oil exhausted).

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
        swc, sorw, sorg, sgr = self.resolve_endpoint_saturations(
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
            residual_gas_saturation=residual_gas_saturation,
        )

        dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso = (
            compute_brooks_corey_capillary_pressure_derivatives(
                water_saturation=water_saturation,
                oil_saturation=oil_saturation,
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
            dpcgw_dsg=zeros_like_value(dpcgo_dsg),
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
        scaled by the pressure conversion factor from `self.unit_system`
        to *target*. Saturation parameters, pore-size distribution indices,
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
            oil_water_entry_pressure_water_wet=scale(
                self.oil_water_entry_pressure_water_wet, pressure_factor
            ),
            oil_water_entry_pressure_oil_wet=scale(
                self.oil_water_entry_pressure_oil_wet, pressure_factor
            ),
            gas_oil_entry_pressure=scale(self.gas_oil_entry_pressure, pressure_factor),
            unit_system=target,
        )
