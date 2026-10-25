"""Shared compiled per-cell building blocks for the analytical three-phase relative permeability models."""

import math
import typing

import numba
import numpy as np

from bores.blackoil.satfunc.relperm.mixing_rules import (
    RULE_BAKER_LINEAR,
    RULE_STONE_I,
    RULE_STONE_II,
    _baker_derivatives,
    _baker_value,
    _min_derivatives,
    _min_value,
    _stone_I_derivatives,
    _stone_I_value,
    _stone_II_derivatives,
    _stone_II_value,
)
from bores.blackoil.satfunc.relperm.tables import show_invalid_saturation
from bores.blackoil.satfunc.utils import (
    prepare_kernel_inputs,
    saturations_within_bounds,
)
from bores.errors import ValidationError
from bores.types import NDimension, Number, NumberArray, NumberOrArray, OneDimension

__all__ = [
    "NO_MINIMUM",
    "RelPermCell",
    "blend_cells",
    "clip_unit",
    "finalize_mixed_wet_cell",
    "finalize_oil_wet_cell",
    "finalize_water_wet_cell",
    "floor_relperm",
    "floor_relperm_derivative",
    "normalize_cell_saturations",
    "prepare_relperm_inputs",
    "rule_derivatives",
    "rule_value",
    "to_kernel_minimum",
]

NO_MINIMUM = -1.0
"""Value that stands for "no minimum relative permeability" inside a compiled kernel."""


def to_kernel_minimum(minimum: Number | None) -> float:
    """
    Convert a resolved minimum relative permeability to the form the compiled kernels take.

    :param minimum: Resolved minimum relative permeability, or `None` for no minimum.
    :return: The minimum as a float, or `NO_MINIMUM` when there is none.
    """
    return NO_MINIMUM if minimum is None else float(minimum)


class RelPermCell(typing.NamedTuple):
    """
    Relative permeabilities and their partial derivatives for one cell of one wettability
    system, before the minimum relative permeability is applied.

    Derivatives that are always zero for the system are stored as zero. The derivatives of
    gas relative permeability with respect to water and oil saturation are not stored
    because gas relative permeability depends on gas saturation only.
    """

    krw: float
    """Water relative permeability."""

    kro: float
    """Oil relative permeability."""

    krg: float
    """Gas relative permeability."""

    dkrw_dsw: float
    """Derivative of water relative permeability with respect to water saturation."""

    dkrw_dso: float
    """Derivative of water relative permeability with respect to oil saturation."""

    dkrw_dsg: float
    """Derivative of water relative permeability with respect to gas saturation."""

    dkro_dsw: float
    """Derivative of oil relative permeability with respect to water saturation."""

    dkro_dso: float
    """Derivative of oil relative permeability with respect to oil saturation."""

    dkro_dsg: float
    """Derivative of oil relative permeability with respect to gas saturation."""

    dkrg_dsg: float
    """Derivative of gas relative permeability with respect to gas saturation."""


@numba.njit(cache=True, inline="always", error_model="numpy")
def clip_unit(value: float) -> float:
    """Limit `value` to the range 0 to 1, leaving NaN unchanged."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@numba.njit(cache=True, inline="always", error_model="numpy")
def floor_relperm(relperm: float, minimum: float) -> float:
    """Smooth floor `sqrt(kr**2 + minimum**2)` of a relative permeability, or `relperm` itself when there is no minimum."""
    if minimum < 0.0:
        return relperm
    return math.sqrt(relperm**2 + minimum**2)


@numba.njit(cache=True, inline="always", error_model="numpy")
def floor_relperm_derivative(derivative: float, relperm: float, minimum: float) -> float:
    """Derivative of the smooth floor, or `derivative` itself when there is no minimum."""
    if minimum < 0.0:
        return derivative
    return derivative * relperm / math.sqrt(relperm**2 + minimum**2)


@numba.njit(cache=True, inline="always", error_model="numpy")
def normalize_cell_saturations(
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
    saturation_epsilon: float,
) -> tuple[float, float, float]:
    """Scale the saturations of a cell to sum to 1 when they differ from 1 by more than `saturation_epsilon`."""
    total = water_saturation + oil_saturation + gas_saturation
    if abs(total - 1.0) > saturation_epsilon and total > 0.0:
        return (
            water_saturation / total,
            oil_saturation / total,
            gas_saturation / total,
        )
    return water_saturation, oil_saturation, gas_saturation


@numba.njit(cache=True, error_model="numpy")
def rule_value(
    rule_id: int,
    kro_w: float,
    kro_g: float,
    krw: float,
    krg: float,
    kr_max: float,
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
    irreducible_water_saturation: float,
    residual_oil_saturation_water: float,
    residual_oil_saturation_gas: float,
    minimum_mobile_pore_space: float,
) -> float:
    """Value of the built-in mixing rule identified by `rule_id`, limited to the range 0 to 1."""
    if rule_id == RULE_BAKER_LINEAR:
        value = _baker_value(
            kro_w,
            kro_g,
            water_saturation,
            gas_saturation,
            irreducible_water_saturation,
            minimum_mobile_pore_space,
        )
    elif rule_id == RULE_STONE_II:
        value = _stone_II_value(kro_w, kro_g, krw, krg, kr_max, minimum_mobile_pore_space)
    elif rule_id == RULE_STONE_I:
        value = _stone_I_value(
            kro_w,
            kro_g,
            kr_max,
            water_saturation,
            oil_saturation,
            gas_saturation,
            irreducible_water_saturation,
            residual_oil_saturation_water,
            residual_oil_saturation_gas,
            minimum_mobile_pore_space,
        )
    else:
        value = _min_value(kro_w, kro_g)
    return clip_unit(value)


@numba.njit(cache=True, error_model="numpy")
def rule_derivatives(
    rule_id: int,
    kro_w: float,
    kro_g: float,
    krw: float,
    krg: float,
    kr_max: float,
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
    irreducible_water_saturation: float,
    residual_oil_saturation_water: float,
    residual_oil_saturation_gas: float,
    minimum_mobile_pore_space: float,
) -> tuple[float, float, float, float, float, float, float]:
    """
    Partial derivatives of the built-in mixing rule identified by `rule_id`.

    :return: Tuple of the derivatives with respect to `kro_w`, `kro_g`, `krw`, `krg`, water
        saturation, oil saturation and gas saturation, in that order.
    """
    if rule_id == RULE_BAKER_LINEAR:
        return _baker_derivatives(
            kro_w,
            kro_g,
            water_saturation,
            gas_saturation,
            irreducible_water_saturation,
            minimum_mobile_pore_space,
        )
    if rule_id == RULE_STONE_II:
        return _stone_II_derivatives(kro_w, kro_g, krw, krg, kr_max, minimum_mobile_pore_space)
    if rule_id == RULE_STONE_I:
        return _stone_I_derivatives(
            kro_w,
            kro_g,
            kr_max,
            water_saturation,
            oil_saturation,
            gas_saturation,
            irreducible_water_saturation,
            residual_oil_saturation_water,
            residual_oil_saturation_gas,
            minimum_mobile_pore_space,
        )
    return _min_derivatives(kro_w, kro_g)


@numba.njit(cache=True, error_model="numpy")
def blend_cells(
    water_wet: RelPermCell, oil_wet: RelPermCell, water_wet_fraction: float
) -> RelPermCell:
    """Blend the water-wet and oil-wet systems of a mixed-wet rock, weighting the water-wet system by `water_wet_fraction`."""
    f = water_wet_fraction
    g = 1.0 - water_wet_fraction
    return RelPermCell(
        f * water_wet.krw + g * oil_wet.krw,
        f * water_wet.kro + g * oil_wet.kro,
        f * water_wet.krg + g * oil_wet.krg,
        f * water_wet.dkrw_dsw + g * oil_wet.dkrw_dsw,
        f * water_wet.dkrw_dso + g * oil_wet.dkrw_dso,
        f * water_wet.dkrw_dsg + g * oil_wet.dkrw_dsg,
        f * water_wet.dkro_dsw + g * oil_wet.dkro_dsw,
        f * water_wet.dkro_dso + g * oil_wet.dkro_dso,
        f * water_wet.dkro_dsg + g * oil_wet.dkro_dsg,
        f * water_wet.dkrg_dsg + g * oil_wet.dkrg_dsg,
    )


@numba.njit(cache=True, error_model="numpy")
def finalize_water_wet_cell(
    cell: RelPermCell, minimum_water: float, minimum_oil: float, minimum_gas: float
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """
    Apply the minimum relative permeability to a purely water-wet cell.

    Water relative permeability depends on water saturation only and gas relative
    permeability on gas saturation only, so their other derivatives are zero.

    :return: Tuple of `dkrw_dsw`, `dkro_dsw`, `dkrg_dsw`, `dkrw_dso`, `dkro_dso`, `dkrg_dso`,
        `dkrw_dsg`, `dkro_dsg` and `dkrg_dsg`.
    """
    return (
        floor_relperm_derivative(cell.dkrw_dsw, cell.krw, minimum_water),
        floor_relperm_derivative(cell.dkro_dsw, cell.kro, minimum_oil),
        0.0,
        0.0,
        floor_relperm_derivative(cell.dkro_dso, cell.kro, minimum_oil),
        0.0,
        0.0,
        floor_relperm_derivative(cell.dkro_dsg, cell.kro, minimum_oil),
        floor_relperm_derivative(cell.dkrg_dsg, cell.krg, minimum_gas),
    )


@numba.njit(cache=True, error_model="numpy")
def finalize_oil_wet_cell(
    cell: RelPermCell, minimum_water: float, minimum_oil: float, minimum_gas: float
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """
    Apply the minimum relative permeability to a purely oil-wet cell.

    Oil relative permeability depends on oil saturation only and gas relative permeability
    on gas saturation only, so their other derivatives are zero.

    :return: Tuple of the nine derivatives in the order of `finalize_water_wet_cell`.
    """
    return (
        floor_relperm_derivative(cell.dkrw_dsw, cell.krw, minimum_water),
        0.0,
        0.0,
        floor_relperm_derivative(cell.dkrw_dso, cell.krw, minimum_water),
        floor_relperm_derivative(cell.dkro_dso, cell.kro, minimum_oil),
        0.0,
        floor_relperm_derivative(cell.dkrw_dsg, cell.krw, minimum_water),
        0.0,
        floor_relperm_derivative(cell.dkrg_dsg, cell.krg, minimum_gas),
    )


@numba.njit(cache=True, error_model="numpy")
def finalize_mixed_wet_cell(
    cell: RelPermCell, minimum_water: float, minimum_oil: float, minimum_gas: float
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """
    Apply the minimum relative permeability to an already blended mixed-wet cell.

    :return: Tuple of the nine derivatives in the order of `finalize_water_wet_cell`.
    """
    return (
        floor_relperm_derivative(cell.dkrw_dsw, cell.krw, minimum_water),
        floor_relperm_derivative(cell.dkro_dsw, cell.kro, minimum_oil),
        0.0,
        floor_relperm_derivative(cell.dkrw_dso, cell.krw, minimum_water),
        floor_relperm_derivative(cell.dkro_dso, cell.kro, minimum_oil),
        0.0,
        floor_relperm_derivative(cell.dkrw_dsg, cell.krw, minimum_water),
        floor_relperm_derivative(cell.dkro_dsg, cell.kro, minimum_oil),
        floor_relperm_derivative(cell.dkrg_dsg, cell.krg, minimum_gas),
    )


def prepare_relperm_inputs(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
) -> tuple[tuple[int, ...], int, tuple[NumberArray[OneDimension], ...]]:
    """
    Validate the saturations and bring all inputs of a relative permeability kernel to flat arrays.

    :raises ValidationError: If any saturation is outside the range 0 to 1.
    :return: Tuple of the broadcast shape, the number of cells and the flattened inputs in
        the order water, oil, gas, irreducible water, residual oil to water, residual oil to
        gas and residual gas saturation.
    """
    shape, size, arrays = prepare_kernel_inputs(
        water_saturation,
        oil_saturation,
        gas_saturation,
        irreducible_water_saturation,
        residual_oil_saturation_water,
        residual_oil_saturation_gas,
        residual_gas_saturation,
    )
    if not saturations_within_bounds(arrays[0], arrays[1], arrays[2], size):
        sw, so, sg = np.broadcast_arrays(
            np.atleast_1d(water_saturation),
            np.atleast_1d(oil_saturation),
            np.atleast_1d(gas_saturation),
        )
        raise ValidationError(
            f"Saturations must be between 0 and 1. Sw: {show_invalid_saturation(sw)}, "
            f"So: {show_invalid_saturation(so)}, Sg: {show_invalid_saturation(sg)}"
        )
    return shape, size, arrays
