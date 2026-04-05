import typing

import attrs
import numba
import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator

from bores.grids.utils import Spacing, make_saturation_grid
from bores.rock_fluid.capillary_pressure import (
    CapillaryPressureTable,
    ThreePhaseCapillaryPressureTable,
    TwoPhaseCapillaryPressureTable,
)
from bores.rock_fluid.relperm import (
    MixingRule,
    RelativePermeabilityTable,
    ThreePhaseRelPermTable,
    TwoPhaseRelPermTable,
)
from bores.serialization import Serializable
from bores.types import (
    CapillaryPressures,
    FluidPhase,
    RelativePermeabilities,
)

__all__ = [
    "RockFluidTables",
    "as_three_phase_capillary_pressure_table",
    "as_three_phase_relperm_table",
]


@typing.final
@attrs.frozen
class RockFluidTables(Serializable):
    """
    Tables defining rock-fluid interactions in the reservoir.

    Made up of a relative permeability table and an optional capillary pressure
    table.
    """

    relative_permeability_table: RelativePermeabilityTable
    capillary_pressure_table: typing.Optional[CapillaryPressureTable] = None

    def get_relative_permeabilities(
        self,
        water_saturation: float,
        oil_saturation: float,
        gas_saturation: float,
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        return self.relative_permeability_table.get_relative_permeabilities(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )

    def get_capillary_pressures(
        self,
        water_saturation: float,
        oil_saturation: float,
        gas_saturation: float,
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        if self.capillary_pressure_table is None:
            raise ValueError("Capillary pressure table is not defined.")
        return self.capillary_pressure_table.get_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )


def _resolve_saturation_endpoint(
    argument_value: typing.Optional[float],
    model: object,
    model_attribute_name: str,
    calling_function_name: str,
) -> float:
    """
    Return `argument_value` if provided, otherwise fall back to the named
    attribute on `model`. Raises `ValueError` when neither source supplies
    a value.

    :param argument_value: Explicit value passed by the caller, or `None` to
        trigger the model attribute fallback.
    :param model: Model object that may carry the endpoint as an attribute.
    :param model_attribute_name: Name of the attribute to read from `model`.
    :param calling_function_name: Name of the public function to include in the
        error message so the user knows where to fix the missing value.
    :return: Resolved saturation endpoint as a Python float.
    """
    if argument_value is not None:
        return float(argument_value)
    attribute_value = getattr(model, model_attribute_name, None)
    if attribute_value is None:
        raise ValueError(
            f"'{model_attribute_name}' must be supplied either as an argument to "
            f"{calling_function_name}() or stored in the model."
        )
    return float(attribute_value)


@numba.njit(cache=True, inline="always")
def _make_saturation_grid_with_minimum_span(
    number_of_points: int,
    saturation_minimum: float,
    saturation_maximum: float,
    spacing_mode: Spacing,
    minimum_span: float,
) -> npt.NDArray:
    """
    Build a saturation grid, enforcing a floor of `minimum_span` on the
    total range so the grid never collapses to a single point.

    `minimum_span` is a plain positional argument rather than keyword-only
    so that Numba can inline this function without a typing error.

    :param number_of_points: Number of points in the output grid.
    :param saturation_minimum: Lower bound of the saturation range.
    :param saturation_maximum: Upper bound of the saturation range.
    :param spacing_mode: Grid spacing strategy (e.g. `'cosine'`, `'linspace'`).
    :param minimum_span: Minimum permitted distance between the lower and upper
        bounds. When the natural span is smaller, the upper bound is extended to
        `saturation_minimum + minimum_span`.
    :return: 1-D array of saturation values.
    """
    if saturation_maximum - saturation_minimum < minimum_span:
        return np.array(
            [
                saturation_minimum,
                max(saturation_minimum + minimum_span, saturation_maximum),
            ],
            dtype=np.float64,
        )
    return make_saturation_grid(
        number_of_points, saturation_minimum, saturation_maximum, spacing_mode
    )


@numba.njit(cache=True, inline="always")
def _clamp_to_closed_interval(
    value: float, lower_bound: float, upper_bound: float
) -> float:
    """
    Clamp `value` to the closed interval [`lower_bound`, `upper_bound`].

    :param value: Value to clamp.
    :param lower_bound: Minimum permitted value.
    :param upper_bound: Maximum permitted value.
    :return: Clamped value.
    """
    return max(lower_bound, min(upper_bound, value))


def _pchip_resample_with_derivative(
    source_saturations: npt.NDArray,
    source_values: npt.NDArray,
    number_of_output_points: int,
    spacing_mode: Spacing,
) -> typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Fit a PCHIP interpolant through (`source_saturations`, `source_values`),
    resample at `number_of_output_points` and return both the resampled values
    and their first derivatives.

    Returning the derivative alongside the values lets callers populate both
    the value and derivative arrays of a two-phase table in one pass, giving
    the Newton solver a C¹-consistent Jacobian without extra model evaluations.

    :param source_saturations: Original saturation knots, strictly increasing.
    :param source_values: Function values at each knot (kr or Pc).
    :param number_of_output_points: Number of points in the resampled grid.
    :param spacing_mode: Grid spacing strategy for the output grid.
    :return: Three-tuple of `(resampled_saturations, resampled_values, resampled_derivatives)`.
    """
    interpolant = PchipInterpolator(source_saturations, source_values)
    resampled_saturations = make_saturation_grid(
        number_of_output_points,
        float(source_saturations[0]),
        float(source_saturations[-1]),
        spacing_mode,
    )
    return (
        resampled_saturations,
        interpolant(resampled_saturations),
        interpolant(resampled_saturations, 1),
    )


@numba.njit(cache=True)
def _build_saturation_reference_grid(
    number_of_base_points: int,
    saturation_lower_bound: float,
    saturation_upper_bound: float,
    spacing_mode: Spacing,
    number_of_endpoint_extra_points: int,
) -> npt.NDArray:
    """
    Build a saturation reference grid with optional endpoint refinement.

    The base grid spans [`saturation_lower_bound`, `saturation_upper_bound`]
    at `number_of_base_points` using `spacing_mode`. When
    `number_of_endpoint_extra_points` > 0, extra knots are injected into the
    first and last 10 % of the range to capture the rapid variation of kr and
    Pc curves near residual saturations.

    `number_of_endpoint_extra_points` is a plain positional argument rather
    than keyword-only so that Numba can compile this function without a typing
    error when inlining `_make_saturation_grid_with_minimum_span`.

    :param number_of_base_points: Number of evenly-spaced base knots.
    :param saturation_lower_bound: Left end of the saturation range.
    :param saturation_upper_bound: Right end of the saturation range.
    :param spacing_mode: Grid spacing strategy (e.g. `'cosine'`, `'linspace'`).
    :param number_of_endpoint_extra_points: Number of extra knots added inside
        each boundary decade. Pass `0` to disable endpoint refinement.
    :return: Sorted 1-D array of unique saturation values.
    """
    minimum_grid_span = 1e-6
    base_grid = _make_saturation_grid_with_minimum_span(
        number_of_base_points,
        saturation_lower_bound,
        saturation_upper_bound,
        spacing_mode,
        minimum_grid_span,
    )
    saturation_span = saturation_upper_bound - saturation_lower_bound
    if saturation_span < minimum_grid_span or number_of_endpoint_extra_points <= 0:
        return base_grid

    endpoint_decade_width = 0.10 * saturation_span
    lower_endpoint_refinement = np.linspace(
        saturation_lower_bound,
        saturation_lower_bound + endpoint_decade_width,
        number_of_endpoint_extra_points + 2,
    )
    upper_endpoint_refinement = np.linspace(
        saturation_upper_bound - endpoint_decade_width,
        saturation_upper_bound,
        number_of_endpoint_extra_points + 2,
    )
    return np.unique(
        np.concatenate(
            (base_grid, lower_endpoint_refinement, upper_endpoint_refinement)
        )
    )


@numba.njit(cache=True)
def _oil_water_point_sweep_along_water_saturation(
    water_saturation: float,
    irreducible_water_saturation: float,
    residual_oil_saturation_water: float,
) -> typing.Tuple[float, float, float]:
    """
    Clamp a water saturation value to the oil-water mobile window and return
    the corresponding (Sw, So, Sg) triple with Sg fixed at zero.

    :param water_saturation: Candidate water saturation to clamp.
    :param irreducible_water_saturation: Connate water saturation (Swc).
    :param residual_oil_saturation_water: Residual oil saturation in a waterflood (Sorw).
    :return: Three-tuple `(Sw, So, Sg)` with Sg = 0.
    """
    sw = _clamp_to_closed_interval(
        water_saturation,
        irreducible_water_saturation,
        1.0 - residual_oil_saturation_water,
    )
    so = _clamp_to_closed_interval(1.0 - sw, 0.0, 1.0 - irreducible_water_saturation)
    return sw, so, 0.0


@numba.njit(cache=True)
def _oil_water_point_sweep_along_oil_saturation(
    oil_saturation: float,
    irreducible_water_saturation: float,
    residual_oil_saturation_water: float,
) -> typing.Tuple[float, float, float]:
    """
    Clamp an oil saturation value to the oil-water mobile window and return
    the corresponding (Sw, So, Sg) triple with Sg fixed at zero.

    :param oil_saturation: Candidate oil saturation to clamp.
    :param irreducible_water_saturation: Connate water saturation (Swc).
    :param residual_oil_saturation_water: Residual oil saturation in a waterflood (Sorw).
    :return: Three-tuple `(Sw, So, Sg)` with Sg = 0.
    """
    so = _clamp_to_closed_interval(
        oil_saturation,
        residual_oil_saturation_water,
        1.0 - irreducible_water_saturation,
    )
    sw = _clamp_to_closed_interval(irreducible_water_saturation, 0.0, 1.0 - so)
    return sw, so, 0.0


@numba.njit(cache=True)
def _gas_oil_point_sweep_along_gas_saturation(
    gas_saturation: float,
    irreducible_water_saturation: float,
    residual_oil_saturation_gas: float,
    residual_gas_saturation: float,
) -> typing.Tuple[float, float, float]:
    """
    Clamp a gas saturation value to the gas-oil mobile window and return the
    corresponding (Sw, So, Sg) triple with Sw fixed at Swc.

    :param gas_saturation: Candidate gas saturation to clamp.
    :param irreducible_water_saturation: Connate water saturation (Swc).
    :param residual_oil_saturation_gas: Residual oil saturation in a gas flood
        (Sorg).
    :param residual_gas_saturation: Residual gas saturation (Sgr).
    :return: Three-tuple `(Sw, So, Sg)` with Sw = Swc.
    """
    sg = _clamp_to_closed_interval(
        gas_saturation,
        residual_gas_saturation,
        1.0 - irreducible_water_saturation - residual_oil_saturation_gas,
    )
    so = _clamp_to_closed_interval(
        1.0 - irreducible_water_saturation - sg,
        0.0,
        1.0 - irreducible_water_saturation,
    )
    return irreducible_water_saturation, so, sg


@numba.njit(cache=True)
def _gas_oil_point_sweep_along_oil_saturation(
    oil_saturation: float,
    irreducible_water_saturation: float,
    residual_oil_saturation_gas: float,
    residual_gas_saturation: float,
) -> typing.Tuple[float, float, float]:
    """
    Clamp an oil saturation value to the gas-oil mobile window and return the
    corresponding (Sw, So, Sg) triple with Sw fixed at Swc.

    :param oil_saturation: Candidate oil saturation to clamp.
    :param irreducible_water_saturation: Connate water saturation (Swc).
    :param residual_oil_saturation_gas: Residual oil saturation in a gas flood (Sorg).
    :param residual_gas_saturation: Residual gas saturation (Sgr).
    :return: Three-tuple `(Sw, So, Sg)` with Sw = Swc.
    """
    so = _clamp_to_closed_interval(
        oil_saturation,
        residual_oil_saturation_gas,
        1.0 - irreducible_water_saturation - residual_gas_saturation,
    )
    sg = _clamp_to_closed_interval(
        1.0 - irreducible_water_saturation - so,
        0.0,
        1.0 - irreducible_water_saturation,
    )
    return irreducible_water_saturation, so, sg


def _oil_water_sweep_axis_is_water_saturation(
    wetting_phase: FluidPhase,
    reference_phase: typing.Literal["wetting", "non_wetting"],
) -> bool:
    """
    Return `True` when the oil-water table should be built by sweeping along
    increasing water saturation.

    :param wetting_phase: Wetting phase declared for the oil-water system.
    :param reference_phase: Whether the reference axis tracks the wetting or
        non-wetting phase saturation.
    :return: `True` if the sweep axis is water saturation.
    """
    return (wetting_phase == FluidPhase.WATER) == (reference_phase == "wetting")


def _gas_oil_sweep_axis_is_gas_saturation(
    wetting_phase: FluidPhase,
    reference_phase: typing.Literal["wetting", "non_wetting"],
) -> bool:
    """
    Return `True` when the gas-oil table should be built by sweeping along
    increasing gas saturation.

    :param wetting_phase: Wetting phase declared for the gas-oil system.
    :param reference_phase: Whether the reference axis tracks the wetting or
        non-wetting phase saturation.
    :return: `True` if the sweep axis is gas saturation.
    """
    return (wetting_phase == FluidPhase.GAS) == (reference_phase == "wetting")


def _sample_oil_water_relative_permeabilities(
    *,
    relperm_model: RelativePermeabilityTable,
    oil_water_reference_saturations: npt.NDArray,
    sweep_axis_is_water_saturation: bool,
    irreducible_water_saturation: float,
    residual_oil_saturation_water: float,
    model_call_kwargs: typing.Dict[str, typing.Any],
    oil_water_wetting_phase: FluidPhase,
    number_of_output_points: int,
    spacing_mode: Spacing,
) -> typing.Tuple[
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    typing.Optional[npt.NDArray],
    typing.Optional[npt.NDArray],
]:
    """
    Sample oil-water relative permeability values and their derivatives across
    the oil-water saturation range.

    For tabular source models (`ThreePhaseRelPermTable`) the existing knots
    are PCHIP-resampled to the denser output grid, recovering C¹-continuous
    values and derivatives in a single pass without extra model evaluations.

    :param relperm_model: Relative permeability model to sample from.
    :param oil_water_reference_saturations: Saturation axis to sample along.
    :param sweep_axis_is_water_saturation: When `True` the reference axis is
        water saturation; when `False` it is oil saturation.
    :param irreducible_water_saturation: Connate water saturation (Swc).
    :param residual_oil_saturation_water: Residual oil saturation in a waterflood (Sorw).
    :param model_call_kwargs: Extra keyword arguments forwarded to every model evaluation call.
    :param oil_water_wetting_phase: Wetting phase for the oil-water system.
    :param number_of_output_points: Number of points used when PCHIP-resampling a tabular source model.
    :param spacing_mode: Grid spacing strategy used during PCHIP resampling.
    :return: Five-tuple of `(reference_saturations, wetting_phase_kr,
        non_wetting_phase_kr, wetting_phase_kr_derivative,
        non_wetting_phase_kr_derivative)`.
    """
    if isinstance(relperm_model, ThreePhaseRelPermTable):
        oil_water_table = relperm_model.oil_water_table
        resampled_saturations, wetting_phase_kr, wetting_phase_kr_derivative = (
            _pchip_resample_with_derivative(
                source_saturations=oil_water_table.reference_saturation,
                source_values=oil_water_table.wetting_phase_relative_permeability,
                number_of_output_points=number_of_output_points,
                spacing_mode=spacing_mode,
            )
        )
        _, non_wetting_phase_kr, non_wetting_phase_kr_derivative = (
            _pchip_resample_with_derivative(
                source_saturations=oil_water_table.reference_saturation,
                source_values=oil_water_table.non_wetting_phase_relative_permeability,
                number_of_output_points=number_of_output_points,
                spacing_mode=spacing_mode,
            )
        )
        return (
            resampled_saturations,
            wetting_phase_kr,
            non_wetting_phase_kr,
            wetting_phase_kr_derivative,
            non_wetting_phase_kr_derivative,
        )

    water_relative_permeability = np.empty(len(oil_water_reference_saturations))
    oil_relative_permeability = np.empty(len(oil_water_reference_saturations))
    water_relative_permeability_derivative = np.empty(
        len(oil_water_reference_saturations)
    )
    oil_relative_permeability_derivative = np.empty(
        len(oil_water_reference_saturations)
    )

    for index, reference_saturation_value in enumerate(oil_water_reference_saturations):
        if sweep_axis_is_water_saturation:
            sw, so, sg = _oil_water_point_sweep_along_water_saturation(
                water_saturation=float(reference_saturation_value),
                irreducible_water_saturation=irreducible_water_saturation,
                residual_oil_saturation_water=residual_oil_saturation_water,
            )
        else:
            sw, so, sg = _oil_water_point_sweep_along_oil_saturation(
                oil_saturation=float(reference_saturation_value),
                irreducible_water_saturation=irreducible_water_saturation,
                residual_oil_saturation_water=residual_oil_saturation_water,
            )

        relative_permeabilities = relperm_model.get_relative_permeabilities(
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            **model_call_kwargs,
        )
        water_relative_permeability[index] = float(relative_permeabilities["water"])
        oil_relative_permeability[index] = float(relative_permeabilities["oil"])

        relative_permeability_derivatives = (
            relperm_model.get_relative_permeability_derivatives(  # type: ignore[union-attr]
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                **model_call_kwargs,
            )
        )
        if sweep_axis_is_water_saturation:
            water_relative_permeability_derivative[index] = float(
                relative_permeability_derivatives["dKrw_dSw"]
            )
            oil_relative_permeability_derivative[index] = float(
                relative_permeability_derivatives["dKro_dSw"]
            )
        else:
            water_relative_permeability_derivative[index] = float(
                relative_permeability_derivatives["dKrw_dSo"]
            )
            oil_relative_permeability_derivative[index] = float(
                relative_permeability_derivatives["dKro_dSo"]  # type: ignore[index]
            )

    if oil_water_wetting_phase == FluidPhase.WATER:
        return (
            oil_water_reference_saturations,
            water_relative_permeability,
            oil_relative_permeability,
            water_relative_permeability_derivative,
            oil_relative_permeability_derivative,
        )
    return (  # type: ignore[return-value]
        oil_water_reference_saturations,
        oil_relative_permeability,
        water_relative_permeability,
        oil_relative_permeability_derivative,
        water_relative_permeability_derivative,
    )


def _sample_gas_oil_relative_permeabilities(
    *,
    relperm_model: RelativePermeabilityTable,
    gas_oil_reference_saturations: npt.NDArray,
    sweep_axis_is_gas_saturation: bool,
    irreducible_water_saturation: float,
    residual_oil_saturation_gas: float,
    residual_gas_saturation: float,
    model_call_kwargs: typing.Dict[str, typing.Any],
    gas_oil_wetting_phase: FluidPhase,
    number_of_output_points: int,
    spacing_mode: Spacing,
) -> typing.Tuple[
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    typing.Optional[npt.NDArray],
    typing.Optional[npt.NDArray],
]:
    """
    Sample gas-oil relative permeability values and their derivatives across
    the gas-oil saturation range.

    For tabular source models (`ThreePhaseRelPermTable`) the existing knots
    are PCHIP-resampled to the denser output grid (fast path).

    :param relperm_model: Relative permeability model to sample from.
    :param gas_oil_reference_saturations: Saturation axis to sample along.
    :param sweep_axis_is_gas_saturation: When `True` the reference axis is
        gas saturation; when `False` it is oil saturation.
    :param irreducible_water_saturation: Connate water saturation (Swc).
    :param residual_oil_saturation_gas: Residual oil saturation in a gas flood (Sorg).
    :param residual_gas_saturation: Residual gas saturation (Sgr).
    :param model_call_kwargs: Extra keyword arguments forwarded to every model evaluation call.
    :param gas_oil_wetting_phase: Wetting phase for the gas-oil system.
    :param number_of_output_points: Number of points used when PCHIP-resampling a tabular source model.
    :param spacing_mode: Grid spacing strategy used during PCHIP resampling.
    :return: Five-tuple of `(reference_saturations, wetting_phase_kr,
        non_wetting_phase_kr, wetting_phase_kr_derivative,
        non_wetting_phase_kr_derivative)`.
    """
    if isinstance(relperm_model, ThreePhaseRelPermTable):
        gas_oil_table = relperm_model.gas_oil_table
        resampled_saturations, wetting_phase_kr, wetting_phase_kr_derivative = (
            _pchip_resample_with_derivative(
                source_saturations=gas_oil_table.reference_saturation,
                source_values=gas_oil_table.wetting_phase_relative_permeability,
                number_of_output_points=number_of_output_points,
                spacing_mode=spacing_mode,
            )
        )
        _, non_wetting_phase_kr, non_wetting_phase_kr_derivative = (
            _pchip_resample_with_derivative(
                source_saturations=gas_oil_table.reference_saturation,
                source_values=gas_oil_table.non_wetting_phase_relative_permeability,
                number_of_output_points=number_of_output_points,
                spacing_mode=spacing_mode,
            )
        )
        return (
            resampled_saturations,
            wetting_phase_kr,
            non_wetting_phase_kr,
            wetting_phase_kr_derivative,
            non_wetting_phase_kr_derivative,
        )

    gas_relative_permeability = np.empty(len(gas_oil_reference_saturations))
    oil_relative_permeability = np.empty(len(gas_oil_reference_saturations))
    gas_relative_permeability_derivative = np.empty(len(gas_oil_reference_saturations))
    oil_relative_permeability_derivative = np.empty(len(gas_oil_reference_saturations))

    for index, reference_saturation_value in enumerate(gas_oil_reference_saturations):
        if sweep_axis_is_gas_saturation:
            sw, so, sg = _gas_oil_point_sweep_along_gas_saturation(
                gas_saturation=float(reference_saturation_value),
                irreducible_water_saturation=irreducible_water_saturation,
                residual_oil_saturation_gas=residual_oil_saturation_gas,
                residual_gas_saturation=residual_gas_saturation,
            )
        else:
            sw, so, sg = _gas_oil_point_sweep_along_oil_saturation(
                oil_saturation=float(reference_saturation_value),
                irreducible_water_saturation=irreducible_water_saturation,
                residual_oil_saturation_gas=residual_oil_saturation_gas,
                residual_gas_saturation=residual_gas_saturation,
            )

        relative_permeabilities = relperm_model.get_relative_permeabilities(
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            **model_call_kwargs,
        )
        gas_relative_permeability[index] = float(relative_permeabilities["gas"])
        oil_relative_permeability[index] = float(relative_permeabilities["oil"])

        relative_permeability_derivatives = (
            relperm_model.get_relative_permeability_derivatives(  # type: ignore[union-attr]
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                **model_call_kwargs,
            )
        )
        if sweep_axis_is_gas_saturation:
            gas_relative_permeability_derivative[index] = float(
                relative_permeability_derivatives["dKrg_dSg"]
            )
            oil_relative_permeability_derivative[index] = float(
                relative_permeability_derivatives["dKro_dSg"]
            )
        else:
            gas_relative_permeability_derivative[index] = float(
                relative_permeability_derivatives["dKrg_dSo"]
            )
            oil_relative_permeability_derivative[index] = float(
                relative_permeability_derivatives["dKro_dSo"]  # type: ignore[index]
            )

    if gas_oil_wetting_phase == FluidPhase.OIL:
        return (
            gas_oil_reference_saturations,
            oil_relative_permeability,
            gas_relative_permeability,
            oil_relative_permeability_derivative,
            gas_relative_permeability_derivative,
        )
    return (  # type: ignore[return-value]
        gas_oil_reference_saturations,
        gas_relative_permeability,
        oil_relative_permeability,
        gas_relative_permeability_derivative,
        oil_relative_permeability_derivative,
    )


def _sample_oil_water_capillary_pressure(
    *,
    capillary_pressure_model: CapillaryPressureTable,
    oil_water_reference_saturations: npt.NDArray,
    sweep_axis_is_water_saturation: bool,
    irreducible_water_saturation: float,
    residual_oil_saturation_water: float,
    model_call_kwargs: typing.Dict[str, typing.Any],
    number_of_output_points: int,
    spacing_mode: Spacing,
) -> typing.Tuple[npt.NDArray, npt.NDArray, typing.Optional[npt.NDArray]]:
    """
    Sample oil-water capillary pressure values and their derivatives across
    the oil-water saturation range.

    For tabular source models (`ThreePhaseCapillaryPressureTable`) the
    existing knots are PCHIP-resampled to the denser output grid (fast path).

    :param capillary_pressure_model: Capillary pressure model to sample from.
    :param oil_water_reference_saturations: Saturation axis to sample along.
    :param sweep_axis_is_water_saturation: When `True` the reference axis is
        water saturation; when `False` it is oil saturation.
    :param irreducible_water_saturation: Connate water saturation (Swc).
    :param residual_oil_saturation_water: Residual oil saturation in a waterflood (Sorw).
    :param model_call_kwargs: Extra keyword arguments forwarded to every model
        evaluation call.
    :param number_of_output_points: Number of points used when PCHIP-resampling
        a tabular source model.
    :param spacing_mode: Grid spacing strategy used during PCHIP resampling.
    :return: Three-tuple of `(reference_saturations, capillary_pressure_values, capillary_pressure_derivatives)`.
    """
    if isinstance(capillary_pressure_model, ThreePhaseCapillaryPressureTable):
        oil_water_table = capillary_pressure_model.oil_water_table
        (
            resampled_saturations,
            capillary_pressure_values,
            capillary_pressure_derivatives,
        ) = _pchip_resample_with_derivative(
            source_saturations=oil_water_table.reference_saturation,
            source_values=oil_water_table.capillary_pressure,
            number_of_output_points=number_of_output_points,
            spacing_mode=spacing_mode,
        )
        return (
            resampled_saturations,
            capillary_pressure_values,
            capillary_pressure_derivatives,
        )

    capillary_pressure_values = np.empty(len(oil_water_reference_saturations))
    capillary_pressure_derivatives = np.empty(len(oil_water_reference_saturations))

    for index, reference_saturation_value in enumerate(oil_water_reference_saturations):
        if sweep_axis_is_water_saturation:
            sw, so, sg = _oil_water_point_sweep_along_water_saturation(
                water_saturation=float(reference_saturation_value),
                irreducible_water_saturation=irreducible_water_saturation,
                residual_oil_saturation_water=residual_oil_saturation_water,
            )
        else:
            sw, so, sg = _oil_water_point_sweep_along_oil_saturation(
                oil_saturation=float(reference_saturation_value),
                irreducible_water_saturation=irreducible_water_saturation,
                residual_oil_saturation_water=residual_oil_saturation_water,
            )

        capillary_pressures = capillary_pressure_model.get_capillary_pressures(
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            **model_call_kwargs,
        )
        capillary_pressure_values[index] = float(capillary_pressures["oil_water"])

        capillary_pressure_derivative_values = (
            capillary_pressure_model.get_capillary_pressure_derivatives(  # type: ignore[union-attr]
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                **model_call_kwargs,
            )
        )
        if sweep_axis_is_water_saturation:
            capillary_pressure_derivatives[index] = float(
                capillary_pressure_derivative_values["dPcow_dSw"]
            )
        else:
            capillary_pressure_derivatives[index] = float(
                capillary_pressure_derivative_values["dPcow_dSo"]
            )

    return (
        oil_water_reference_saturations,
        capillary_pressure_values,
        capillary_pressure_derivatives,
    )


def _sample_gas_oil_capillary_pressure(
    *,
    capillary_pressure_model: CapillaryPressureTable,
    gas_oil_reference_saturations: npt.NDArray,
    sweep_axis_is_gas_saturation: bool,
    irreducible_water_saturation: float,
    residual_oil_saturation_gas: float,
    residual_gas_saturation: float,
    model_call_kwargs: typing.Dict[str, typing.Any],
    number_of_output_points: int,
    spacing_mode: Spacing,
) -> typing.Tuple[npt.NDArray, npt.NDArray, typing.Optional[npt.NDArray]]:
    """
    Sample gas-oil capillary pressure values and their derivatives across the
    gas-oil saturation range.

    For tabular source models (`ThreePhaseCapillaryPressureTable`) the
    existing knots are PCHIP-resampled to the denser output grid (fast path).

    :param capillary_pressure_model: Capillary pressure model to sample from.
    :param gas_oil_reference_saturations: Saturation axis to sample along.
    :param sweep_axis_is_gas_saturation: When `True` the reference axis is
        gas saturation; when `False` it is oil saturation.
    :param irreducible_water_saturation: Connate water saturation (Swc).
    :param residual_oil_saturation_gas: Residual oil saturation in a gas flood (Sorg).
    :param residual_gas_saturation: Residual gas saturation (Sgr).
    :param model_call_kwargs: Extra keyword arguments forwarded to every model evaluation call.
    :param number_of_output_points: Number of points used when PCHIP-resampling a tabular source model.
    :param spacing_mode: Grid spacing strategy used during PCHIP resampling.
    :return: Three-tuple of `(reference_saturations, capillary_pressure_values, capillary_pressure_derivatives)`.
    """
    if isinstance(capillary_pressure_model, ThreePhaseCapillaryPressureTable):
        gas_oil_table = capillary_pressure_model.gas_oil_table
        (
            resampled_saturations,
            capillary_pressure_values,
            capillary_pressure_derivatives,
        ) = _pchip_resample_with_derivative(
            source_saturations=gas_oil_table.reference_saturation,
            source_values=gas_oil_table.capillary_pressure,
            number_of_output_points=number_of_output_points,
            spacing_mode=spacing_mode,
        )
        return (
            resampled_saturations,
            capillary_pressure_values,
            capillary_pressure_derivatives,
        )

    capillary_pressure_values = np.empty(len(gas_oil_reference_saturations))
    capillary_pressure_derivatives = np.empty(len(gas_oil_reference_saturations))

    for index, reference_saturation_value in enumerate(gas_oil_reference_saturations):
        if sweep_axis_is_gas_saturation:
            sw, so, sg = _gas_oil_point_sweep_along_gas_saturation(
                gas_saturation=float(reference_saturation_value),
                irreducible_water_saturation=irreducible_water_saturation,
                residual_oil_saturation_gas=residual_oil_saturation_gas,
                residual_gas_saturation=residual_gas_saturation,
            )
        else:
            sw, so, sg = _gas_oil_point_sweep_along_oil_saturation(
                oil_saturation=float(reference_saturation_value),
                irreducible_water_saturation=irreducible_water_saturation,
                residual_oil_saturation_gas=residual_oil_saturation_gas,
                residual_gas_saturation=residual_gas_saturation,
            )

        capillary_pressures = capillary_pressure_model.get_capillary_pressures(
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            **model_call_kwargs,
        )
        capillary_pressure_values[index] = float(capillary_pressures["gas_oil"])

        capillary_pressure_derivative_values = (
            capillary_pressure_model.get_capillary_pressure_derivatives(  # type: ignore[union-attr]
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                **model_call_kwargs,
            )
        )
        if sweep_axis_is_gas_saturation:
            capillary_pressure_derivatives[index] = float(
                capillary_pressure_derivative_values["dPcgo_dSg"]
            )
        else:
            capillary_pressure_derivatives[index] = float(
                capillary_pressure_derivative_values["dPcgo_dSo"]
            )

    return (
        gas_oil_reference_saturations,
        capillary_pressure_values,
        capillary_pressure_derivatives,
    )


def as_three_phase_relperm_table(
    model: RelativePermeabilityTable,
    *,
    irreducible_water_saturation: typing.Optional[float] = None,
    residual_oil_saturation_water: typing.Optional[float] = None,
    residual_oil_saturation_gas: typing.Optional[float] = None,
    residual_gas_saturation: typing.Optional[float] = None,
    model_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    oil_water_wetting_phase: typing.Optional[typing.Union[FluidPhase, str]] = None,
    gas_oil_wetting_phase: typing.Optional[typing.Union[FluidPhase, str]] = None,
    oil_water_reference_phase: typing.Literal["wetting", "non_wetting"] = "wetting",
    gas_oil_reference_phase: typing.Literal["wetting", "non_wetting"] = "non_wetting",
    n_points: int = 200,
    n_endpoint_extra: int = 20,
    spacing: Spacing = "cosine",
    oil_water_reference_saturation: typing.Optional[npt.ArrayLike] = None,
    gas_oil_reference_saturation: typing.Optional[npt.ArrayLike] = None,
    mixing_rule: typing.Optional[typing.Union[MixingRule, str]] = None,
) -> ThreePhaseRelPermTable:
    """
    Convert any `RelativePermeabilityTable` to a `ThreePhaseRelPermTable`
    backed by piecewise-linear `TwoPhaseRelPermTable` instances.

    Returns `model` unchanged if it is already a `ThreePhaseRelPermTable`.

    Analytical derivatives are sampled at every knot and stored in the
    two-phase sub-tables so that `get_*_derivative` returns smooth,
    consistent values instead of piecewise-linear slopes.

    For tabular source models (`ThreePhaseRelPermTable`) the existing knots
    are PCHIP-resampled to the denser output grid, recovering C¹-continuous
    values and derivatives without any additional model calls.

    :param model: Source analytical or tabular relative permeability model.
    :param irreducible_water_saturation: Irreducible water saturation (Swc).
    :param residual_oil_saturation_water: Residual oil saturation in a waterflood (Sorw).
    :param residual_oil_saturation_gas: Residual oil saturation in a gas flood (Sorg).
    :param residual_gas_saturation: Residual gas saturation (Sgr).
    :param model_kwargs: Extra kwargs forwarded to every model evaluation call.
    :param oil_water_wetting_phase: Wetting phase for the oil-water sub-table.
    :param gas_oil_wetting_phase: Wetting phase for the gas-oil sub-table.
    :param oil_water_reference_phase: Reference saturation axis for the oil-water table.
    :param gas_oil_reference_phase: Reference saturation axis for the gas-oil table.
    :param n_points: Base number of sample points per sub-table.
    :param n_endpoint_extra: Extra knots in each boundary decade. Pass `0` to disable endpoint refinement.
    :param spacing: Grid spacing mode.
    :param oil_water_reference_saturation: Custom saturation axis for the
        oil-water sub-table. Overrides the auto-generated grid when supplied.
    :param gas_oil_reference_saturation: Custom saturation axis for the gas-oil
        sub-table. Overrides the auto-generated grid when supplied.
    :param mixing_rule: Three-phase oil relative permeability mixing rule.
    :return: `ThreePhaseRelPermTable` with piecewise-linear sub-tables.
    """
    if isinstance(model, ThreePhaseRelPermTable):
        return model

    resolved_irreducible_water_saturation = _resolve_saturation_endpoint(
        argument_value=irreducible_water_saturation,
        model=model,
        model_attribute_name="irreducible_water_saturation",
        calling_function_name="as_three_phase_relperm_table",
    )
    resolved_residual_oil_saturation_water = _resolve_saturation_endpoint(
        argument_value=residual_oil_saturation_water,
        model=model,
        model_attribute_name="residual_oil_saturation_water",
        calling_function_name="as_three_phase_relperm_table",
    )
    resolved_residual_oil_saturation_gas = _resolve_saturation_endpoint(
        argument_value=residual_oil_saturation_gas,
        model=model,
        model_attribute_name="residual_oil_saturation_gas",
        calling_function_name="as_three_phase_relperm_table",
    )
    resolved_residual_gas_saturation = _resolve_saturation_endpoint(
        argument_value=residual_gas_saturation,
        model=model,
        model_attribute_name="residual_gas_saturation",
        calling_function_name="as_three_phase_relperm_table",
    )

    model_call_kwargs: typing.Dict[str, typing.Any] = {
        "irreducible_water_saturation": resolved_irreducible_water_saturation,
        "residual_oil_saturation_water": resolved_residual_oil_saturation_water,
        "residual_oil_saturation_gas": resolved_residual_oil_saturation_gas,
        "residual_gas_saturation": resolved_residual_gas_saturation,
        **(model_kwargs or {}),
    }

    oil_water_wetting_phase_resolved: FluidPhase = (
        FluidPhase(oil_water_wetting_phase)
        if oil_water_wetting_phase is not None
        else model.get_oil_water_wetting_phase()
    )
    gas_oil_wetting_phase_resolved: FluidPhase = (
        FluidPhase(gas_oil_wetting_phase)
        if gas_oil_wetting_phase is not None
        else model.get_gas_oil_wetting_phase()
    )
    oil_water_non_wetting_phase = (
        FluidPhase.OIL
        if oil_water_wetting_phase_resolved == FluidPhase.WATER
        else FluidPhase.WATER
    )
    gas_oil_non_wetting_phase = (
        FluidPhase.GAS
        if gas_oil_wetting_phase_resolved == FluidPhase.OIL
        else FluidPhase.OIL
    )

    sweep_oil_water_axis_is_water_saturation = (
        _oil_water_sweep_axis_is_water_saturation(
            wetting_phase=oil_water_wetting_phase_resolved,
            reference_phase=oil_water_reference_phase,
        )
    )
    sweep_gas_oil_axis_is_gas_saturation = _gas_oil_sweep_axis_is_gas_saturation(
        wetting_phase=gas_oil_wetting_phase_resolved,
        reference_phase=gas_oil_reference_phase,
    )

    if oil_water_reference_saturation is not None:
        oil_water_reference_saturations = np.asarray(
            oil_water_reference_saturation, dtype=np.float64
        )
    else:
        if sweep_oil_water_axis_is_water_saturation:
            oil_water_lower_bound = resolved_irreducible_water_saturation
            oil_water_upper_bound = 1.0 - resolved_residual_oil_saturation_water
        else:
            oil_water_lower_bound = resolved_residual_oil_saturation_water
            oil_water_upper_bound = 1.0 - resolved_irreducible_water_saturation
        oil_water_reference_saturations = _build_saturation_reference_grid(
            number_of_base_points=n_points,
            saturation_lower_bound=oil_water_lower_bound,
            saturation_upper_bound=oil_water_upper_bound,
            spacing_mode=spacing,
            number_of_endpoint_extra_points=n_endpoint_extra,
        )

    if gas_oil_reference_saturation is not None:
        gas_oil_reference_saturations = np.asarray(
            gas_oil_reference_saturation, dtype=np.float64
        )
    else:
        if sweep_gas_oil_axis_is_gas_saturation:
            gas_oil_lower_bound = resolved_residual_gas_saturation
            gas_oil_upper_bound = (
                1.0
                - resolved_irreducible_water_saturation
                - resolved_residual_oil_saturation_gas
            )
        else:
            gas_oil_lower_bound = resolved_residual_oil_saturation_gas
            gas_oil_upper_bound = (
                1.0
                - resolved_irreducible_water_saturation
                - resolved_residual_gas_saturation
            )
        gas_oil_reference_saturations = _build_saturation_reference_grid(
            number_of_base_points=n_points,
            saturation_lower_bound=gas_oil_lower_bound,
            saturation_upper_bound=gas_oil_upper_bound,
            spacing_mode=spacing,
            number_of_endpoint_extra_points=n_endpoint_extra,
        )

    (
        oil_water_reference_saturations,
        oil_water_wetting_phase_kr,
        oil_water_non_wetting_phase_kr,
        oil_water_wetting_phase_kr_derivative,
        oil_water_non_wetting_phase_kr_derivative,
    ) = _sample_oil_water_relative_permeabilities(
        relperm_model=model,
        oil_water_reference_saturations=oil_water_reference_saturations,
        sweep_axis_is_water_saturation=sweep_oil_water_axis_is_water_saturation,
        irreducible_water_saturation=resolved_irreducible_water_saturation,
        residual_oil_saturation_water=resolved_residual_oil_saturation_water,
        model_call_kwargs=model_call_kwargs,
        oil_water_wetting_phase=oil_water_wetting_phase_resolved,
        number_of_output_points=n_points,
        spacing_mode=spacing,
    )

    (
        gas_oil_reference_saturations,
        gas_oil_wetting_phase_kr,
        gas_oil_non_wetting_phase_kr,
        gas_oil_wetting_phase_kr_derivative,
        gas_oil_non_wetting_phase_kr_derivative,
    ) = _sample_gas_oil_relative_permeabilities(
        relperm_model=model,
        gas_oil_reference_saturations=gas_oil_reference_saturations,
        sweep_axis_is_gas_saturation=sweep_gas_oil_axis_is_gas_saturation,
        irreducible_water_saturation=resolved_irreducible_water_saturation,
        residual_oil_saturation_gas=resolved_residual_oil_saturation_gas,
        residual_gas_saturation=resolved_residual_gas_saturation,
        model_call_kwargs=model_call_kwargs,
        gas_oil_wetting_phase=gas_oil_wetting_phase_resolved,
        number_of_output_points=n_points,
        spacing_mode=spacing,
    )

    oil_water_table = TwoPhaseRelPermTable(
        wetting_phase=oil_water_wetting_phase_resolved,
        non_wetting_phase=oil_water_non_wetting_phase,
        reference_saturation=oil_water_reference_saturations,
        wetting_phase_relative_permeability=oil_water_wetting_phase_kr,
        non_wetting_phase_relative_permeability=oil_water_non_wetting_phase_kr,
        reference_phase=oil_water_reference_phase,
        wetting_phase_relative_permeability_derivative=oil_water_wetting_phase_kr_derivative,
        non_wetting_phase_relative_permeability_derivative=oil_water_non_wetting_phase_kr_derivative,
    )
    gas_oil_table = TwoPhaseRelPermTable(
        wetting_phase=gas_oil_wetting_phase_resolved,
        non_wetting_phase=gas_oil_non_wetting_phase,
        reference_saturation=gas_oil_reference_saturations,
        wetting_phase_relative_permeability=gas_oil_wetting_phase_kr,
        non_wetting_phase_relative_permeability=gas_oil_non_wetting_phase_kr,
        reference_phase=gas_oil_reference_phase,
        wetting_phase_relative_permeability_derivative=gas_oil_wetting_phase_kr_derivative,
        non_wetting_phase_relative_permeability_derivative=gas_oil_non_wetting_phase_kr_derivative,
    )

    if mixing_rule is None:
        mixing_rule = getattr(model, "mixing_rule", "eclipse_rule")

    return ThreePhaseRelPermTable(
        oil_water_table=oil_water_table,
        gas_oil_table=gas_oil_table,
        mixing_rule=mixing_rule,
    )


def as_three_phase_capillary_pressure_table(
    model: CapillaryPressureTable,
    *,
    irreducible_water_saturation: typing.Optional[float] = None,
    residual_oil_saturation_water: typing.Optional[float] = None,
    residual_oil_saturation_gas: typing.Optional[float] = None,
    residual_gas_saturation: typing.Optional[float] = None,
    model_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    oil_water_wetting_phase: typing.Optional[typing.Union[FluidPhase, str]] = None,
    gas_oil_wetting_phase: typing.Optional[typing.Union[FluidPhase, str]] = None,
    oil_water_reference_phase: typing.Literal["wetting", "non_wetting"] = "wetting",
    gas_oil_reference_phase: typing.Literal["wetting", "non_wetting"] = "non_wetting",
    n_points: int = 200,
    n_endpoint_extra: int = 30,
    spacing: Spacing = "cosine",
    oil_water_reference_saturation: typing.Optional[npt.ArrayLike] = None,
    gas_oil_reference_saturation: typing.Optional[npt.ArrayLike] = None,
) -> ThreePhaseCapillaryPressureTable:
    """
    Convert any `CapillaryPressureTable` to a `ThreePhaseCapillaryPressureTable` backed by piecewise-linear
    `TwoPhaseCapillaryPressureTable` instances.

    Returns `model` unchanged if it is already a `ThreePhaseCapillaryPressureTable`.

    Analytical derivatives are sampled at every knot and stored in the
    two-phase sub-tables. The default `n_endpoint_extra=30` (vs 20 for
    relperm) reflects that Pc curves are unbounded near residual saturation,
    making endpoint fidelity especially important for SI convergence.

    :param model: Source analytical or tabular capillary pressure model.
    :param irreducible_water_saturation: Irreducible water saturation (Swc).
    :param residual_oil_saturation_water: Residual oil saturation in a waterflood (Sorw).
    :param residual_oil_saturation_gas: Residual oil saturation in a gas flood (Sorg).
    :param residual_gas_saturation: Residual gas saturation (Sgr).
    :param model_kwargs: Extra kwargs forwarded to every model evaluation call.
    :param oil_water_wetting_phase: Wetting phase for the oil-water sub-table.
    :param gas_oil_wetting_phase: Wetting phase for the gas-oil sub-table.
    :param oil_water_reference_phase: Reference saturation axis for the oil-water table.
    :param gas_oil_reference_phase: Reference saturation axis for the gas-oil table.
    :param n_points: Base number of sample points per sub-table.
    :param n_endpoint_extra: Extra knots in each boundary decade. Pass `0` to disable endpoint refinement.
    :param spacing: Grid spacing mode.
    :param oil_water_reference_saturation: Custom saturation axis for the
        oil-water sub-table. Overrides the auto-generated grid when supplied.
    :param gas_oil_reference_saturation: Custom saturation axis for the gas-oil
        sub-table. Overrides the auto-generated grid when supplied.
    :return: `ThreePhaseCapillaryPressureTable` backed by piecewise-linear sub-tables.
    """
    if isinstance(model, ThreePhaseCapillaryPressureTable):
        return model

    resolved_irreducible_water_saturation = _resolve_saturation_endpoint(
        argument_value=irreducible_water_saturation,
        model=model,
        model_attribute_name="irreducible_water_saturation",
        calling_function_name="as_three_phase_capillary_pressure_table",
    )
    resolved_residual_oil_saturation_water = _resolve_saturation_endpoint(
        argument_value=residual_oil_saturation_water,
        model=model,
        model_attribute_name="residual_oil_saturation_water",
        calling_function_name="as_three_phase_capillary_pressure_table",
    )
    resolved_residual_oil_saturation_gas = _resolve_saturation_endpoint(
        argument_value=residual_oil_saturation_gas,
        model=model,
        model_attribute_name="residual_oil_saturation_gas",
        calling_function_name="as_three_phase_capillary_pressure_table",
    )
    resolved_residual_gas_saturation = _resolve_saturation_endpoint(
        argument_value=residual_gas_saturation,
        model=model,
        model_attribute_name="residual_gas_saturation",
        calling_function_name="as_three_phase_capillary_pressure_table",
    )

    model_call_kwargs: typing.Dict[str, typing.Any] = {
        "irreducible_water_saturation": resolved_irreducible_water_saturation,
        "residual_oil_saturation_water": resolved_residual_oil_saturation_water,
        "residual_oil_saturation_gas": resolved_residual_oil_saturation_gas,
        "residual_gas_saturation": resolved_residual_gas_saturation,
        **(model_kwargs or {}),
    }

    oil_water_wetting_phase_resolved: FluidPhase = (
        FluidPhase(oil_water_wetting_phase)
        if oil_water_wetting_phase is not None
        else model.get_oil_water_wetting_phase()
    )
    gas_oil_wetting_phase_resolved: FluidPhase = (
        FluidPhase(gas_oil_wetting_phase)
        if gas_oil_wetting_phase is not None
        else model.get_gas_oil_wetting_phase()
    )
    oil_water_non_wetting_phase = (
        FluidPhase.OIL
        if oil_water_wetting_phase_resolved == FluidPhase.WATER
        else FluidPhase.WATER
    )
    gas_oil_non_wetting_phase = (
        FluidPhase.GAS
        if gas_oil_wetting_phase_resolved == FluidPhase.OIL
        else FluidPhase.OIL
    )

    sweep_oil_water_axis_is_water_saturation = (
        _oil_water_sweep_axis_is_water_saturation(
            wetting_phase=oil_water_wetting_phase_resolved,
            reference_phase=oil_water_reference_phase,
        )
    )
    sweep_gas_oil_axis_is_gas_saturation = _gas_oil_sweep_axis_is_gas_saturation(
        wetting_phase=gas_oil_wetting_phase_resolved,
        reference_phase=gas_oil_reference_phase,
    )

    if oil_water_reference_saturation is not None:
        oil_water_reference_saturations = np.asarray(
            oil_water_reference_saturation, dtype=np.float64
        )
    else:
        if sweep_oil_water_axis_is_water_saturation:
            oil_water_lower_bound = resolved_irreducible_water_saturation
            oil_water_upper_bound = 1.0 - resolved_residual_oil_saturation_water
        else:
            oil_water_lower_bound = resolved_residual_oil_saturation_water
            oil_water_upper_bound = 1.0 - resolved_irreducible_water_saturation
        oil_water_reference_saturations = _build_saturation_reference_grid(
            number_of_base_points=n_points,
            saturation_lower_bound=oil_water_lower_bound,
            saturation_upper_bound=oil_water_upper_bound,
            spacing_mode=spacing,
            number_of_endpoint_extra_points=n_endpoint_extra,
        )

    if gas_oil_reference_saturation is not None:
        gas_oil_reference_saturations = np.asarray(
            gas_oil_reference_saturation, dtype=np.float64
        )
    else:
        if sweep_gas_oil_axis_is_gas_saturation:
            gas_oil_lower_bound = resolved_residual_gas_saturation
            gas_oil_upper_bound = (
                1.0
                - resolved_irreducible_water_saturation
                - resolved_residual_oil_saturation_gas
            )
        else:
            gas_oil_lower_bound = resolved_residual_oil_saturation_gas
            gas_oil_upper_bound = (
                1.0
                - resolved_irreducible_water_saturation
                - resolved_residual_gas_saturation
            )
        gas_oil_reference_saturations = _build_saturation_reference_grid(
            number_of_base_points=n_points,
            saturation_lower_bound=gas_oil_lower_bound,
            saturation_upper_bound=gas_oil_upper_bound,
            spacing_mode=spacing,
            number_of_endpoint_extra_points=n_endpoint_extra,
        )

    (
        oil_water_reference_saturations,
        oil_water_capillary_pressure_values,
        oil_water_capillary_pressure_derivatives,
    ) = _sample_oil_water_capillary_pressure(
        capillary_pressure_model=model,
        oil_water_reference_saturations=oil_water_reference_saturations,
        sweep_axis_is_water_saturation=sweep_oil_water_axis_is_water_saturation,
        irreducible_water_saturation=resolved_irreducible_water_saturation,
        residual_oil_saturation_water=resolved_residual_oil_saturation_water,
        model_call_kwargs=model_call_kwargs,
        number_of_output_points=n_points,
        spacing_mode=spacing,
    )

    (
        gas_oil_reference_saturations,
        gas_oil_capillary_pressure_values,
        gas_oil_capillary_pressure_derivatives,
    ) = _sample_gas_oil_capillary_pressure(
        capillary_pressure_model=model,
        gas_oil_reference_saturations=gas_oil_reference_saturations,
        sweep_axis_is_gas_saturation=sweep_gas_oil_axis_is_gas_saturation,
        irreducible_water_saturation=resolved_irreducible_water_saturation,
        residual_oil_saturation_gas=resolved_residual_oil_saturation_gas,
        residual_gas_saturation=resolved_residual_gas_saturation,
        model_call_kwargs=model_call_kwargs,
        number_of_output_points=n_points,
        spacing_mode=spacing,
    )

    oil_water_table = TwoPhaseCapillaryPressureTable(
        wetting_phase=oil_water_wetting_phase_resolved,
        non_wetting_phase=oil_water_non_wetting_phase,
        reference_saturation=oil_water_reference_saturations,
        capillary_pressure=oil_water_capillary_pressure_values,
        reference_phase=oil_water_reference_phase,
        capillary_pressure_derivative=oil_water_capillary_pressure_derivatives,
    )
    gas_oil_table = TwoPhaseCapillaryPressureTable(
        wetting_phase=gas_oil_wetting_phase_resolved,
        non_wetting_phase=gas_oil_non_wetting_phase,
        reference_saturation=gas_oil_reference_saturations,
        capillary_pressure=gas_oil_capillary_pressure_values,
        reference_phase=gas_oil_reference_phase,
        capillary_pressure_derivative=gas_oil_capillary_pressure_derivatives,
    )
    return ThreePhaseCapillaryPressureTable(
        oil_water_table=oil_water_table, gas_oil_table=gas_oil_table
    )
