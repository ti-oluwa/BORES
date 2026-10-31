"""Leverett J-function capillary pressure model and table."""

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
    "LeverettJCapillaryPressureTable",
]


@numba.njit(cache=True, inline="always", error_model="numpy")
def compute_leverett_j_capillary_pressures_cell(
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
    dyne_per_cm_to_pressure_unit: Number,
    oil_water_contact_angle: Number = 0.0,
    gas_oil_contact_angle: Number = 0.0,
    j_function_coefficient: Number = 0.5,
    j_function_exponent: Number = 0.5,
    mixed_wet_water_fraction: Number = 0.5,
    wettability: Wettability = Wettability.WATER_WET,
    saturation_epsilon: Number = 1e-6,
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
        effective_liquid_saturation = (1.0 - swc - sg - sorg) / total_mobile_pore_space_gas
        effective_liquid_saturation = min(
            max(effective_liquid_saturation, saturation_epsilon),
            1.0 - saturation_epsilon,
        )

        j_value_gas_oil = j_function_coefficient * (
            effective_liquid_saturation ** (-j_function_exponent)
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
    dyne_per_cm_to_pressure_unit: Number,
    oil_water_contact_angle: Number = 0.0,
    gas_oil_contact_angle: Number = 0.0,
    j_function_coefficient: Number = 0.5,
    j_function_exponent: Number = 0.5,
    mixed_wet_water_fraction: Number = 0.5,
    wettability: Wettability = Wettability.WATER_WET,
    saturation_epsilon: Number = 1e-6,
    minimum_mobile_pore_space: Number = 1e-9,
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
    if sw < 0 or sw > 1 or so < 0 or so > 1 or sg < 0 or sg > 1:
        raise ValidationError("Saturations must be between 0 and 1.")
    if permeability < 0.0:
        raise ValidationError("Permeability must be positive.")
    if porosity < 0.0 or porosity > 1.0:
        raise ValidationError("Porosity must be between 0 and 1.")
    return compute_leverett_j_capillary_pressures_cell(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        irreducible_water_saturation=irreducible_water_saturation,
        residual_oil_saturation_water=residual_oil_saturation_water,
        residual_oil_saturation_gas=residual_oil_saturation_gas,
        residual_gas_saturation=residual_gas_saturation,
        permeability=permeability,
        porosity=porosity,
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


@numba.njit(cache=True, parallel=True, error_model="numpy")
def compute_leverett_j_capillary_pressures_loop(
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    irreducible_water_saturation: NumberArray[OneDimension],
    residual_oil_saturation_water: NumberArray[OneDimension],
    residual_oil_saturation_gas: NumberArray[OneDimension],
    residual_gas_saturation: NumberArray[OneDimension],
    permeability: NumberArray[OneDimension],
    porosity: NumberArray[OneDimension],
    oil_water_interfacial_tension: Number,
    gas_oil_interfacial_tension: Number,
    oil_water_contact_angle: Number,
    gas_oil_contact_angle: Number,
    j_function_coefficient: Number,
    j_function_exponent: Number,
    mixed_wet_water_fraction: Number,
    wettability: Wettability,
    saturation_epsilon: Number,
    minimum_mobile_pore_space: Number,
    dyne_per_cm_to_pressure_unit: Number,
    size: Integer,
    oil_water_capillary_pressure: NumberArray[OneDimension],
    gas_oil_capillary_pressure: NumberArray[OneDimension],
) -> None:
    """
    Compute the capillary pressures of `size` cells in parallel and write them to the output arrays.

    Per-cell inputs are flat arrays, or a single value shared by every cell.
    """
    for i in numba.prange(size):
        oil_water, gas_oil = compute_leverett_j_capillary_pressures_cell(
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
            permeability=np.float64(element_at(array=permeability, index=i)),
            porosity=np.float64(element_at(array=porosity, index=i)),
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
        oil_water_capillary_pressure[i] = oil_water
        gas_oil_capillary_pressure[i] = gas_oil


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
    dyne_per_cm_to_pressure_unit: Number,
    oil_water_contact_angle: Number = 0.0,
    gas_oil_contact_angle: Number = 0.0,
    j_function_coefficient: Number = 0.5,
    j_function_exponent: Number = 0.5,
    mixed_wet_water_fraction: Number = 0.5,
    wettability: Wettability = Wettability.WATER_WET,
    saturation_epsilon: Number = 1e-6,
    minimum_mobile_pore_space: Number = 1e-9,
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
            permeability_flat,
            porosity_flat,
        ),
    ) = prepare_kernel_inputs(
        water_saturation,
        oil_saturation,
        gas_saturation,
        irreducible_water_saturation,
        residual_oil_saturation_water,
        residual_oil_saturation_gas,
        residual_gas_saturation,
        permeability,
        porosity,
    )
    if not saturations_within_bounds(
        water_saturation=water_saturation_flat,
        oil_saturation=oil_saturation_flat,
        gas_saturation=gas_saturation_flat,
        size=size,
    ):
        raise ValidationError("Saturations must be between 0 and 1.")
    if np.any(permeability_flat < 0.0):
        raise ValidationError("Permeability must be positive.")
    if np.any((porosity_flat < 0.0) | (porosity_flat > 1.0)):
        raise ValidationError("Porosity must be between 0 and 1.")

    dtype = get_floating_dtype(water_saturation)
    oil_water_capillary_pressure = np.empty(size, dtype=dtype)
    gas_oil_capillary_pressure = np.empty(size, dtype=dtype)
    compute_leverett_j_capillary_pressures_loop(
        water_saturation=water_saturation_flat,
        oil_saturation=oil_saturation_flat,
        gas_saturation=gas_saturation_flat,
        irreducible_water_saturation=irreducible_water_saturation_flat,
        residual_oil_saturation_water=residual_oil_saturation_water_flat,
        residual_oil_saturation_gas=residual_oil_saturation_gas_flat,
        residual_gas_saturation=residual_gas_saturation_flat,
        permeability=permeability_flat,
        porosity=porosity_flat,
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
        size=size,
        oil_water_capillary_pressure=oil_water_capillary_pressure,
        gas_oil_capillary_pressure=gas_oil_capillary_pressure,
    )
    return typing.cast(
        "NumberArray[NDimension]", oil_water_capillary_pressure.reshape(shape)
    ), typing.cast("NumberArray[NDimension]", gas_oil_capillary_pressure.reshape(shape))


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
    dyne_per_cm_to_pressure_unit: Number,
    oil_water_contact_angle: Number = 0.0,
    gas_oil_contact_angle: Number = 0.0,
    j_function_coefficient: Number = 0.5,
    j_function_exponent: Number = 0.5,
    mixed_wet_water_fraction: Number = 0.5,
    wettability: Wettability = Wettability.WATER_WET,
    saturation_epsilon: Number = 1e-6,
    minimum_mobile_pore_space: Number = 1e-9,
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


@numba.njit(cache=True, inline="always", error_model="numpy")
def compute_leverett_j_derivatives_scalar(
    water_saturation: Number,
    oil_saturation: Number,
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
    dyne_per_cm_to_pressure_unit: Number,
    mixed_wet_water_fraction: Number = 0.5,
    saturation_epsilon: Number = 1e-6,
    minimum_mobile_pore_space: Number = 1e-9,
) -> tuple[Number, Number, Number, Number]:
    """
    Scalar variant of Leverett J-function capillary pressure derivatives.

    Returns (dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso).

    :param water_saturation: Water saturation (fraction, 0-1).
    :param oil_saturation: Oil saturation (fraction, 0 to 1).
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

    # No slope where the value is clipped (flat)
    raw_se_w = (sw - swc) / mobile_water_range
    dpcow_dsw = zero
    if valid_water and raw_se_w > saturation_epsilon and raw_se_w < one - saturation_epsilon:
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

    # Pcgo depends on the effective liquid saturation of the gas-oil system, which falls as
    # Sg rises. It has no slope where the value is clipped.
    dpcgo_dsg = zero
    if valid_gas:
        se_l = (one - swc - sg - sorg) / mobile_gas_range
        if se_l > saturation_epsilon and se_l < one - saturation_epsilon:
            dj_dse_l = (
                -j_function_coefficient
                * j_function_exponent
                * (se_l ** (-j_function_exponent - one))
            )
            cos_gas_oil_angle = np.cos(np.deg2rad(gas_oil_contact_angle))
            gas_oil_scale = (
                gas_oil_interfacial_tension
                * dyne_per_cm_to_pressure_unit
                * cos_gas_oil_angle
                * leverett_rock_factor
            )
            dpcgo_dsg = -gas_oil_scale * dj_dse_l / mobile_gas_range

    dpcgo_dso = zero
    return dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso


@numba.njit(cache=True, parallel=True, error_model="numpy")
def compute_leverett_j_derivatives_loop(
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    irreducible_water_saturation: NumberArray[OneDimension],
    residual_oil_saturation_water: NumberArray[OneDimension],
    residual_oil_saturation_gas: NumberArray[OneDimension],
    residual_gas_saturation: NumberArray[OneDimension],
    permeability: NumberArray[OneDimension],
    porosity: NumberArray[OneDimension],
    wettability: Wettability,
    oil_water_interfacial_tension: Number,
    gas_oil_interfacial_tension: Number,
    oil_water_contact_angle: Number,
    gas_oil_contact_angle: Number,
    j_function_coefficient: Number,
    j_function_exponent: Number,
    mixed_wet_water_fraction: Number,
    saturation_epsilon: Number,
    minimum_mobile_pore_space: Number,
    dyne_per_cm_to_pressure_unit: Number,
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
        d_ow_sw, d_ow_so, d_go_sg, d_go_so = compute_leverett_j_derivatives_scalar(
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
            permeability=np.float64(element_at(array=permeability, index=i)),
            porosity=np.float64(element_at(array=porosity, index=i)),
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
        dpcow_dsw[i] = d_ow_sw
        dpcow_dso[i] = d_ow_so
        dpcgo_dsg[i] = d_go_sg
        dpcgo_dso[i] = d_go_so


def compute_leverett_j_derivatives_array(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
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
    dyne_per_cm_to_pressure_unit: Number,
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
    Array variant of Leverett J-function capillary pressure derivatives.

    Returns (dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso) as NDArrays.

    :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
    :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
    :param oil_saturation: Oil saturation (fraction, 0 to 1).
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
            permeability_flat,
            porosity_flat,
        ),
    ) = prepare_kernel_inputs(
        water_saturation,
        oil_saturation,
        gas_saturation,
        irreducible_water_saturation,
        residual_oil_saturation_water,
        residual_oil_saturation_gas,
        residual_gas_saturation,
        permeability,
        porosity,
    )
    dtype = get_floating_dtype(water_saturation)
    dpcow_dsw = np.empty(size, dtype=dtype)
    dpcow_dso = np.empty(size, dtype=dtype)
    dpcgo_dsg = np.empty(size, dtype=dtype)
    dpcgo_dso = np.empty(size, dtype=dtype)
    compute_leverett_j_derivatives_loop(
        water_saturation=water_saturation_flat,
        oil_saturation=oil_saturation_flat,
        gas_saturation=gas_saturation_flat,
        irreducible_water_saturation=irreducible_water_saturation_flat,
        residual_oil_saturation_water=residual_oil_saturation_water_flat,
        residual_oil_saturation_gas=residual_oil_saturation_gas_flat,
        residual_gas_saturation=residual_gas_saturation_flat,
        permeability=permeability_flat,
        porosity=porosity_flat,
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


def compute_leverett_j_derivatives(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
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
    dyne_per_cm_to_pressure_unit: Number,
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
    Dispatch function for Leverett J-function capillary pressure derivatives.

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
        and np.isscalar(permeability)
        and np.isscalar(porosity)
    ):
        return compute_leverett_j_derivatives_scalar(
            water_saturation=water_saturation,  # type: ignore[arg-type]
            oil_saturation=oil_saturation,  # type: ignore[arg-type]
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
        oil_saturation=oil_saturation,  # type: ignore[arg-type]
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


@numba.njit(cache=True, error_model="numpy")
def compute_leverett_j_capillary_pressures_and_derivatives_cell(
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
    dyne_per_cm_to_pressure_unit: Number,
    oil_water_contact_angle: Number = 0.0,
    gas_oil_contact_angle: Number = 0.0,
    j_function_coefficient: Number = 0.5,
    j_function_exponent: Number = 0.5,
    mixed_wet_water_fraction: Number = 0.5,
    wettability: Wettability = Wettability.WATER_WET,
    saturation_epsilon: Number = 1e-6,
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

    oil_water_angle_radians = np.deg2rad(oil_water_contact_angle)
    gas_oil_angle_radians = np.deg2rad(gas_oil_contact_angle)

    valid_rock = permeability > 0 and porosity > 0
    leverett_factor = np.sqrt(porosity / permeability) if valid_rock else 0.0

    # The J function is `a * Se ** (-b)`, so its slope is `-b * J / Se` and the power is evaluated
    # once and shared by the value and the slope. The rock and angle factors are shared too.
    oil_water_capillary_pressure: Number = 0.0
    dpcow_dsw: Number = 0.0
    valid_water = total_mobile_pore_space_water > minimum_mobile_pore_space and valid_rock

    if valid_water:
        raw_effective_water_saturation = (sw - swc) / total_mobile_pore_space_water
        effective_water_saturation = min(
            max(raw_effective_water_saturation, saturation_epsilon),
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
            wettability_sign = 1.0
        elif wettability == Wettability.OIL_WET:
            oil_water_capillary_pressure = -pcow
            wettability_sign = -1.0
        else:  # MIXED_WET
            oil_water_capillary_pressure = mixed_wet_water_fraction * pcow + (
                1.0 - mixed_wet_water_fraction
            ) * (-pcow)
            wettability_sign = 2.0 * mixed_wet_water_fraction - 1.0

        if (
            raw_effective_water_saturation > saturation_epsilon
            and raw_effective_water_saturation < 1.0 - saturation_epsilon
        ):
            dpcow_dsw = (
                wettability_sign
                * (pcow * (-j_function_exponent) / effective_water_saturation)
                / total_mobile_pore_space_water
            )

    gas_oil_capillary_pressure: Number = 0.0
    dpcgo_dsg: Number = 0.0
    valid_gas = total_mobile_pore_space_gas > minimum_mobile_pore_space and valid_rock

    if valid_gas:
        raw_effective_liquid_saturation = (1.0 - swc - sg - sorg) / total_mobile_pore_space_gas
        effective_liquid_saturation = min(
            max(raw_effective_liquid_saturation, saturation_epsilon),
            1.0 - saturation_epsilon,
        )

        j_value_gas_oil = j_function_coefficient * (
            effective_liquid_saturation ** (-j_function_exponent)
        )
        gas_oil_capillary_pressure = (
            gas_oil_interfacial_tension
            * dyne_per_cm_to_pressure_unit
            * np.cos(gas_oil_angle_radians)
            * leverett_factor
            * j_value_gas_oil
        )
        if (
            raw_effective_liquid_saturation > saturation_epsilon
            and raw_effective_liquid_saturation < 1.0 - saturation_epsilon
        ):
            dpcgo_dsg = (
                -(
                    gas_oil_capillary_pressure
                    * (-j_function_exponent)
                    / effective_liquid_saturation
                )
                / total_mobile_pore_space_gas
            )

    return oil_water_capillary_pressure, gas_oil_capillary_pressure, dpcow_dsw, 0.0, dpcgo_dsg, 0.0


@numba.njit(cache=True, parallel=True, error_model="numpy")
def compute_leverett_j_capillary_pressures_and_derivatives_loop(
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    irreducible_water_saturation: NumberArray[OneDimension],
    residual_oil_saturation_water: NumberArray[OneDimension],
    residual_oil_saturation_gas: NumberArray[OneDimension],
    residual_gas_saturation: NumberArray[OneDimension],
    permeability: NumberArray[OneDimension],
    porosity: NumberArray[OneDimension],
    oil_water_interfacial_tension: Number,
    gas_oil_interfacial_tension: Number,
    oil_water_contact_angle: Number,
    gas_oil_contact_angle: Number,
    j_function_coefficient: Number,
    j_function_exponent: Number,
    mixed_wet_water_fraction: Number,
    wettability: Wettability,
    saturation_epsilon: Number,
    minimum_mobile_pore_space: Number,
    dyne_per_cm_to_pressure_unit: Number,
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
            compute_leverett_j_capillary_pressures_and_derivatives_cell(
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
                permeability=np.float64(element_at(array=permeability, index=i)),
                porosity=np.float64(element_at(array=porosity, index=i)),
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
        )
        oil_water_capillary_pressure[i] = oil_water
        gas_oil_capillary_pressure[i] = gas_oil
        dpcow_dsw[i] = d_ow_sw
        dpcow_dso[i] = d_ow_so
        dpcgo_dsg[i] = d_go_sg
        dpcgo_dso[i] = d_go_so


def compute_leverett_j_capillary_pressures_and_derivatives_array(
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
    dyne_per_cm_to_pressure_unit: Number,
    oil_water_contact_angle: Number = 0.0,
    gas_oil_contact_angle: Number = 0.0,
    j_function_coefficient: Number = 0.5,
    j_function_exponent: Number = 0.5,
    mixed_wet_water_fraction: Number = 0.5,
    wettability: Wettability = Wettability.WATER_WET,
    saturation_epsilon: Number = 1e-6,
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
            permeability_flat,
            porosity_flat,
        ),
    ) = prepare_kernel_inputs(
        water_saturation,
        oil_saturation,
        gas_saturation,
        irreducible_water_saturation,
        residual_oil_saturation_water,
        residual_oil_saturation_gas,
        residual_gas_saturation,
        permeability,
        porosity,
    )
    if not saturations_within_bounds(
        water_saturation=water_saturation_flat,
        oil_saturation=oil_saturation_flat,
        gas_saturation=gas_saturation_flat,
        size=size,
    ):
        raise ValidationError("Saturations must be between 0 and 1.")
    if np.any(permeability_flat < 0.0):
        raise ValidationError("Permeability must be positive.")
    if np.any((porosity_flat < 0.0) | (porosity_flat > 1.0)):
        raise ValidationError("Porosity must be between 0 and 1.")

    dtype = get_floating_dtype(water_saturation)
    oil_water_capillary_pressure = np.empty(size, dtype=dtype)
    gas_oil_capillary_pressure = np.empty(size, dtype=dtype)
    dpcow_dsw = np.empty(size, dtype=dtype)
    dpcow_dso = np.empty(size, dtype=dtype)
    dpcgo_dsg = np.empty(size, dtype=dtype)
    dpcgo_dso = np.empty(size, dtype=dtype)
    compute_leverett_j_capillary_pressures_and_derivatives_loop(
        water_saturation=water_saturation_flat,
        oil_saturation=oil_saturation_flat,
        gas_saturation=gas_saturation_flat,
        irreducible_water_saturation=irreducible_water_saturation_flat,
        residual_oil_saturation_water=residual_oil_saturation_water_flat,
        residual_oil_saturation_gas=residual_oil_saturation_gas_flat,
        residual_gas_saturation=residual_gas_saturation_flat,
        permeability=permeability_flat,
        porosity=porosity_flat,
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


def compute_leverett_j_capillary_pressures_and_derivatives(
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
    dyne_per_cm_to_pressure_unit: Number,
    oil_water_contact_angle: Number = 0.0,
    gas_oil_contact_angle: Number = 0.0,
    j_function_coefficient: Number = 0.5,
    j_function_exponent: Number = 0.5,
    mixed_wet_water_fraction: Number = 0.5,
    wettability: Wettability = Wettability.WATER_WET,
    saturation_epsilon: Number = 1e-6,
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
    Dispatch function for the Leverett J-function fused capillary pressure and derivative computation.

    Computes the capillary pressures and their derivatives together, evaluating the power
    functions that both need once per cell. The results match `compute_leverett_j_capillary_pressures` and the derivative function of this
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
        and np.isscalar(permeability)
        and np.isscalar(porosity)
    )
    pressures, derivatives = compute_leverett_j_capillary_pressures_and_derivatives_array(
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

    dyne_per_cm_to_pressure_unit: Number = attrs.field(init=False, repr=False, default=0.0)
    """Factor from dyne/cm to the pressure unit of `unit_system`. Set from `unit_system`."""

    def __attrs_post_init__(self) -> None:
        object.__setattr__(
            self,
            "dyne_per_cm_to_pressure_unit",
            convert_dyne_per_cm_to_pressure_unit(self.unit_system),
        )

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
        porosity: NumberOrArray[NDimension] | None = None,
        permeability: NumberOrArray[NDimension] | None = None,
        **kwargs: typing.Any,
    ) -> tuple[CapillaryPressures, CapillaryPressureDerivatives]:
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
        swc, sorw, sorg, sgr = self.resolve_endpoint_saturations(
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
            residual_gas_saturation=residual_gas_saturation,
        )
        porosity = porosity if porosity is not None else self.porosity
        permeability = permeability if permeability is not None else self.permeability

        (pcow, pcgo), (dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso) = (
            compute_leverett_j_capillary_pressures_and_derivatives(
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
                dyne_per_cm_to_pressure_unit=self.dyne_per_cm_to_pressure_unit,
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
        swc, sorw, sorg, sgr = self.resolve_endpoint_saturations(
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
            residual_gas_saturation=residual_gas_saturation,
        )
        porosity = porosity if porosity is not None else self.porosity
        permeability = permeability if permeability is not None else self.permeability

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
            dyne_per_cm_to_pressure_unit=self.dyne_per_cm_to_pressure_unit,
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
        - `dpcgo/dsg`: analytically derived via the chain rule through the effective
        liquid saturation, and zero where that saturation is clipped.

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
        swc, sorw, sorg, sgr = self.resolve_endpoint_saturations(
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
            residual_gas_saturation=residual_gas_saturation,
        )
        porosity = porosity if porosity is not None else self.porosity
        permeability = permeability if permeability is not None else self.permeability

        dpcow_dsw, dpcow_dso, dpcgo_dsg, dpcgo_dso = compute_leverett_j_derivatives(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
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
            dyne_per_cm_to_pressure_unit=self.dyne_per_cm_to_pressure_unit,
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
            permeability=scale(self.permeability, factors["permeability"]),
            unit_system=target,
        )
