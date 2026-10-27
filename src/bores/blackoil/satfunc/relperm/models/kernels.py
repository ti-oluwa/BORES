"""Shared compiled per-cell building blocks for the analytical three-phase relative permeability models."""

import math
import typing

import numba
import numpy as np

from bores.blackoil.satfunc.relperm.mixing_rules import (
    BAKER_LINEAR_RULE,
    STONE_I_RULE,
    STONE_II_RULE,
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
from bores.types import Integer, NDimension, Number, NumberArray, NumberOrArray, OneDimension

__all__ = [
    "NO_MINIMUM",
    "FlatRelPermInputs",
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


def to_kernel_minimum(minimum: Number | None) -> Number:
    """
    Convert a resolved minimum relative permeability to the form the compiled kernels take.

    :param minimum: Resolved minimum relative permeability, or `None` for no minimum.
    :return: The minimum as a Number, or `NO_MINIMUM` when there is none.
    """
    return NO_MINIMUM if minimum is None else minimum


class RelPermCell(typing.NamedTuple):
    """
    Relative permeabilities and their partial derivatives for one cell of one wettability
    system, before the minimum relative permeability is applied.

    Derivatives that are always zero for the system are stored as zero. The derivatives of
    gas relative permeability with respect to water and oil saturation are not stored
    because gas relative permeability depends on gas saturation only.
    """

    krw: Number
    """Water relative permeability."""

    kro: Number
    """Oil relative permeability."""

    krg: Number
    """Gas relative permeability."""

    dkrw_dsw: Number
    """Derivative of water relative permeability with respect to water saturation."""

    dkrw_dso: Number
    """Derivative of water relative permeability with respect to oil saturation."""

    dkrw_dsg: Number
    """Derivative of water relative permeability with respect to gas saturation."""

    dkro_dsw: Number
    """Derivative of oil relative permeability with respect to water saturation."""

    dkro_dso: Number
    """Derivative of oil relative permeability with respect to oil saturation."""

    dkro_dsg: Number
    """Derivative of oil relative permeability with respect to gas saturation."""

    dkrg_dsg: Number
    """Derivative of gas relative permeability with respect to gas saturation."""


@numba.njit(cache=True, inline="always", error_model="numpy")
def clip_unit(value: Number) -> Number:
    """Limit `value` to the range 0 to 1, leaving NaN unchanged."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@numba.njit(cache=True, inline="always", error_model="numpy")
def floor_relperm(relperm: Number, minimum: Number) -> Number:
    """Smooth floor `sqrt(kr**2 + minimum**2)` of a relative permeability, or `relperm` itself when there is no minimum."""
    if minimum < 0.0:
        return relperm
    return math.sqrt(relperm**2 + minimum**2)


@numba.njit(cache=True, inline="always", error_model="numpy")
def floor_relperm_derivative(derivative: Number, relperm: Number, minimum: Number) -> Number:
    """Derivative of the smooth floor, or `derivative` itself when there is no minimum."""
    if minimum < 0.0:
        return derivative
    return derivative * relperm / math.sqrt(relperm**2 + minimum**2)


@numba.njit(cache=True, inline="always", error_model="numpy")
def normalize_cell_saturations(
    water_saturation: Number,
    oil_saturation: Number,
    gas_saturation: Number,
    saturation_epsilon: Number,
) -> tuple[Number, Number, Number]:
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
    rule_id: Integer,
    kro_w: Number,
    kro_g: Number,
    krw: Number,
    krg: Number,
    kr_max: Number,
    water_saturation: Number,
    oil_saturation: Number,
    gas_saturation: Number,
    irreducible_water_saturation: Number,
    residual_oil_saturation_water: Number,
    residual_oil_saturation_gas: Number,
    minimum_mobile_pore_space: Number,
) -> Number:
    """Value of the built-in mixing rule identified by `rule_id`, limited to the range 0 to 1."""
    if rule_id == BAKER_LINEAR_RULE:
        value = _baker_value(
            kro_w=kro_w,
            kro_g=kro_g,
            water_saturation=water_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=irreducible_water_saturation,
            minimum_mobile_pore_space=minimum_mobile_pore_space,
        )
    elif rule_id == STONE_II_RULE:
        value = _stone_II_value(
            kro_w=kro_w,
            kro_g=kro_g,
            krw=krw,
            krg=krg,
            kr_max=kr_max,
            minimum_mobile_pore_space=minimum_mobile_pore_space,
        )
    elif rule_id == STONE_I_RULE:
        value = _stone_I_value(
            kro_w=kro_w,
            kro_g=kro_g,
            kr_max=kr_max,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
            minimum_mobile_pore_space=minimum_mobile_pore_space,
        )
    else:
        value = _min_value(kro_w=kro_w, kro_g=kro_g)
    return clip_unit(value)


@numba.njit(cache=True, error_model="numpy")
def rule_derivatives(
    rule_id: Integer,
    kro_w: Number,
    kro_g: Number,
    krw: Number,
    krg: Number,
    kr_max: Number,
    water_saturation: Number,
    oil_saturation: Number,
    gas_saturation: Number,
    irreducible_water_saturation: Number,
    residual_oil_saturation_water: Number,
    residual_oil_saturation_gas: Number,
    minimum_mobile_pore_space: Number,
) -> tuple[Number, Number, Number, Number, Number, Number, Number]:
    """
    Partial derivatives of the built-in mixing rule identified by `rule_id`.

    :return: Tuple of the derivatives with respect to `kro_w`, `kro_g`, `krw`, `krg`, water
        saturation, oil saturation and gas saturation, in that order.
    """
    if rule_id == BAKER_LINEAR_RULE:
        return _baker_derivatives(
            kro_w=kro_w,
            kro_g=kro_g,
            water_saturation=water_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=irreducible_water_saturation,
            minimum_mobile_pore_space=minimum_mobile_pore_space,
        )
    if rule_id == STONE_II_RULE:
        return _stone_II_derivatives(
            kro_w=kro_w,
            kro_g=kro_g,
            krw=krw,
            krg=krg,
            kr_max=kr_max,
            minimum_mobile_pore_space=minimum_mobile_pore_space,
        )
    if rule_id == STONE_I_RULE:
        return _stone_I_derivatives(
            kro_w=kro_w,
            kro_g=kro_g,
            kr_max=kr_max,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
            minimum_mobile_pore_space=minimum_mobile_pore_space,
        )
    return _min_derivatives(kro_w=kro_w, kro_g=kro_g)


@numba.njit(cache=True, error_model="numpy")
def blend_cells(
    water_wet: RelPermCell, oil_wet: RelPermCell, water_wet_fraction: Number
) -> RelPermCell:
    """Blend the water-wet and oil-wet systems of a mixed-wet rock, weighting the water-wet system by `water_wet_fraction`."""
    f = water_wet_fraction
    g = 1.0 - water_wet_fraction
    return RelPermCell(
        krw=f * water_wet.krw + g * oil_wet.krw,
        kro=f * water_wet.kro + g * oil_wet.kro,
        krg=f * water_wet.krg + g * oil_wet.krg,
        dkrw_dsw=f * water_wet.dkrw_dsw + g * oil_wet.dkrw_dsw,
        dkrw_dso=f * water_wet.dkrw_dso + g * oil_wet.dkrw_dso,
        dkrw_dsg=f * water_wet.dkrw_dsg + g * oil_wet.dkrw_dsg,
        dkro_dsw=f * water_wet.dkro_dsw + g * oil_wet.dkro_dsw,
        dkro_dso=f * water_wet.dkro_dso + g * oil_wet.dkro_dso,
        dkro_dsg=f * water_wet.dkro_dsg + g * oil_wet.dkro_dsg,
        dkrg_dsg=f * water_wet.dkrg_dsg + g * oil_wet.dkrg_dsg,
    )


@numba.njit(cache=True, error_model="numpy")
def finalize_water_wet_cell(
    cell: RelPermCell, minimum_water: Number, minimum_oil: Number, minimum_gas: Number
) -> tuple[Number, Number, Number, Number, Number, Number, Number, Number, Number]:
    """
    Apply the minimum relative permeability to a purely water-wet cell.

    Water relative permeability depends on water saturation only and gas relative
    permeability on gas saturation only, so their other derivatives are zero.

    :return: Tuple of `dkrw_dsw`, `dkro_dsw`, `dkrg_dsw`, `dkrw_dso`, `dkro_dso`, `dkrg_dso`,
        `dkrw_dsg`, `dkro_dsg` and `dkrg_dsg`.
    """
    return (
        floor_relperm_derivative(
            derivative=cell.dkrw_dsw, relperm=cell.krw, minimum=minimum_water
        ),
        floor_relperm_derivative(derivative=cell.dkro_dsw, relperm=cell.kro, minimum=minimum_oil),
        0.0,
        0.0,
        floor_relperm_derivative(derivative=cell.dkro_dso, relperm=cell.kro, minimum=minimum_oil),
        0.0,
        0.0,
        floor_relperm_derivative(derivative=cell.dkro_dsg, relperm=cell.kro, minimum=minimum_oil),
        floor_relperm_derivative(derivative=cell.dkrg_dsg, relperm=cell.krg, minimum=minimum_gas),
    )


@numba.njit(cache=True, error_model="numpy")
def finalize_oil_wet_cell(
    cell: RelPermCell, minimum_water: Number, minimum_oil: Number, minimum_gas: Number
) -> tuple[Number, Number, Number, Number, Number, Number, Number, Number, Number]:
    """
    Apply the minimum relative permeability to a purely oil-wet cell.

    Oil relative permeability depends on oil saturation only and gas relative permeability
    on gas saturation only, so their other derivatives are zero.

    :return: Tuple of the nine derivatives in the order of `finalize_water_wet_cell`.
    """
    return (
        floor_relperm_derivative(
            derivative=cell.dkrw_dsw, relperm=cell.krw, minimum=minimum_water
        ),
        0.0,
        0.0,
        floor_relperm_derivative(
            derivative=cell.dkrw_dso, relperm=cell.krw, minimum=minimum_water
        ),
        floor_relperm_derivative(derivative=cell.dkro_dso, relperm=cell.kro, minimum=minimum_oil),
        0.0,
        floor_relperm_derivative(
            derivative=cell.dkrw_dsg, relperm=cell.krw, minimum=minimum_water
        ),
        0.0,
        floor_relperm_derivative(derivative=cell.dkrg_dsg, relperm=cell.krg, minimum=minimum_gas),
    )


@numba.njit(cache=True, error_model="numpy")
def finalize_mixed_wet_cell(
    cell: RelPermCell, minimum_water: Number, minimum_oil: Number, minimum_gas: Number
) -> tuple[Number, Number, Number, Number, Number, Number, Number, Number, Number]:
    """
    Apply the minimum relative permeability to an already blended mixed-wet cell.

    :return: Tuple of the nine derivatives in the order of `finalize_water_wet_cell`.
    """
    return (
        floor_relperm_derivative(
            derivative=cell.dkrw_dsw, relperm=cell.krw, minimum=minimum_water
        ),
        floor_relperm_derivative(derivative=cell.dkro_dsw, relperm=cell.kro, minimum=minimum_oil),
        0.0,
        floor_relperm_derivative(
            derivative=cell.dkrw_dso, relperm=cell.krw, minimum=minimum_water
        ),
        floor_relperm_derivative(derivative=cell.dkro_dso, relperm=cell.kro, minimum=minimum_oil),
        0.0,
        floor_relperm_derivative(
            derivative=cell.dkrw_dsg, relperm=cell.krw, minimum=minimum_water
        ),
        floor_relperm_derivative(derivative=cell.dkro_dsg, relperm=cell.kro, minimum=minimum_oil),
        floor_relperm_derivative(derivative=cell.dkrg_dsg, relperm=cell.krg, minimum=minimum_gas),
    )


class FlatRelPermInputs(typing.NamedTuple):
    """The saturation and endpoint inputs of a relative permeability kernel as flat arrays."""

    water_saturation: NumberArray[OneDimension]
    """Water saturation, or a single value shared by every cell."""

    oil_saturation: NumberArray[OneDimension]
    """Oil saturation, or a single value shared by every cell."""

    gas_saturation: NumberArray[OneDimension]
    """Gas saturation, or a single value shared by every cell."""

    irreducible_water_saturation: NumberArray[OneDimension]
    """Irreducible water saturation, or a single value shared by every cell."""

    residual_oil_saturation_water: NumberArray[OneDimension]
    """Residual oil saturation to water flooding, or a single value shared by every cell."""

    residual_oil_saturation_gas: NumberArray[OneDimension]
    """Residual oil saturation to gas flooding, or a single value shared by every cell."""

    residual_gas_saturation: NumberArray[OneDimension]
    """Residual gas saturation, or a single value shared by every cell."""


def prepare_relperm_inputs(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
) -> tuple[tuple[Integer, ...], Integer, FlatRelPermInputs]:
    """
    Validate the saturations and bring all inputs of a relative permeability kernel to flat arrays.

    :raises ValidationError: If any saturation is outside the range 0 to 1.
    :return: Tuple of the broadcast shape, the number of cells and the flattened inputs.
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
    if not saturations_within_bounds(
        water_saturation=arrays[0], oil_saturation=arrays[1], gas_saturation=arrays[2], size=size
    ):
        sw, so, sg = np.broadcast_arrays(
            np.atleast_1d(water_saturation),
            np.atleast_1d(oil_saturation),
            np.atleast_1d(gas_saturation),
        )
        raise ValidationError(
            f"Saturations must be between 0 and 1. Sw: {show_invalid_saturation(sw)}, "
            f"So: {show_invalid_saturation(so)}, Sg: {show_invalid_saturation(sg)}"
        )
    return shape, size, FlatRelPermInputs(*arrays)
