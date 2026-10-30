import typing

import attrs
import numba
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.blackoil.satfunc.capillary_pressure.tables import (
    CapillaryPressureTable,
    ThreePhaseCapillaryPressureTable,
    TwoPhaseCapillaryPressureTable,
)
from bores.blackoil.satfunc.relperm.mixing_rules import MixingRule
from bores.blackoil.satfunc.relperm.tables import (
    RelativePermeabilityTable,
    ThreePhaseRelPermTable,
    TwoPhaseRelPermTable,
)
from bores.blackoil.satfunc.utils import (
    build_saturation_reference_field,
    pchip_resample,
)
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.types import (
    FluidPhase,
    Number,
    NumberArray,
    OneDimension,
    Spacing,
    UnitSystem,
)
from bores.units import UnitConversionTable, get_unit_system

__all__ = [
    "SatFuncTables",
    "as_three_phase_capillary_pressure_table",
    "as_three_phase_relperm_table",
]


@attrs.frozen(slots=True)
class SatFuncTables:
    """
    Saturation function table(s) defining rock-fluid interactions in the reservoir.

    Made up of a relative permeability table and an optional capillary pressure table.
    """

    relative_permeability: RelativePermeabilityTable
    """Relative permeability table for the rock-fluid system."""

    capillary_pressure: CapillaryPressureTable | None = None
    """Capillary pressure table for the rock-fluid system, or `None` if not present."""

    unit_system: UnitSystem = attrs.field(init=False)
    """
    Unit system of the tables. Taken from the capillary pressure table when there is one.
    Otherwise (relative permeability is dimensionless) it is the unit system active in
    the current `UnitSystemContext`.
    """

    def __attrs_post_init__(self) -> None:
        unit_system = None
        if self.capillary_pressure is not None:
            unit_system = self.capillary_pressure.unit_system
        if unit_system is None:
            unit_system = get_unit_system()
        object.__setattr__(self, "unit_system", unit_system)

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `SatFuncTables` with capillary pressure in every
        region rescaled to *target*.

        Relative permeability is dimensionless and is unaffected.

        :param target: Target `UnitSystem`.
        :returns: New `SatFuncTables` in *target* units.
        """
        if self.capillary_pressure is None:
            # Nothing dimensional to rescale, but the declared unit system still changes
            converted = attrs.evolve(self)
            object.__setattr__(converted, "unit_system", target)
            return converted
        return attrs.evolve(
            self,
            capillary_pressure=self.capillary_pressure.convert(target, table=table),
        )


def _resolve_saturation_endpoint(
    arg_value: Number | None, table: typing.Any, attr_name: str, caller: str
) -> Number:
    """
    Return `arg_value` if provided, otherwise fall back to the named
    attribute on `table`. Raises `ValueError` when neither source supplies
    a value.

    :param arg_value: Explicit value passed by the caller, or `None` to
        trigger the table attribute fallback.
    :param table: Model object that may carry the endpoint as an attribute.
    :param attr_name: Name of the attribute to read from `table`.
    :param caller: Name of the public function to include in the
        error message so the user knows where to fix the missing value.
    :return: Resolved saturation endpoint as a Python Number.
    """
    if arg_value is not None:
        return arg_value

    attribute_value = getattr(table, attr_name, None)
    if attribute_value is None:
        raise ValueError(
            f"'{attr_name}' must be supplied either as an argument to "
            f"{caller}() or stored in the table."
        )
    return attribute_value


@numba.njit(cache=True, inline="always")
def _clamp(value: Number, lower_bound: Number, upper_bound: Number) -> Number:
    """
    Clamp `value` to the closed interval [`lower_bound`, `upper_bound`].

    :param value: Value to clamp.
    :param lower_bound: Minimum permitted value.
    :param upper_bound: Maximum permitted value.
    :return: Clamped value.
    """
    return max(lower_bound, min(upper_bound, value))


@numba.njit(cache=True)
def get_oil_water_point_sweep_along_water_saturation(
    water_saturation: Number,
    irreducible_water_saturation: Number,
    residual_oil_saturation_water: Number,
) -> tuple[Number, Number, Number]:
    """
    Clamp a water saturation value to the oil-water mobile window and return
    the corresponding (Sw, So, Sg) triple with Sg fixed at zero.

    :param water_saturation: Candidate water saturation to clamp.
    :param irreducible_water_saturation: Connate water saturation (swc).
    :param residual_oil_saturation_water: Residual oil saturation in a waterflood (sorw).
    :return: Three-tuple `(Sw, So, Sg)` with Sg = 0.
    """
    water_saturation = _clamp(
        water_saturation,
        irreducible_water_saturation,
        1.0 - residual_oil_saturation_water,
    )
    oil_saturation = _clamp(1.0 - water_saturation, 0.0, 1.0 - irreducible_water_saturation)
    return water_saturation, oil_saturation, 0.0


@numba.njit(cache=True)
def get_oil_water_point_sweep_along_oil_saturation(
    oil_saturation: Number,
    irreducible_water_saturation: Number,
    residual_oil_saturation_water: Number,
) -> tuple[Number, Number, Number]:
    """
    Clamp an oil saturation value to the oil-water mobile window and return
    the corresponding (Sw, So, Sg) triple with Sg fixed at zero.

    :param oil_saturation: Candidate oil saturation to clamp.
    :param irreducible_water_saturation: Connate water saturation (swc).
    :param residual_oil_saturation_water: Residual oil saturation in a waterflood (sorw).
    :return: Three-tuple `(Sw, So, Sg)` with Sg = 0.
    """
    oil_saturation = _clamp(
        oil_saturation,
        residual_oil_saturation_water,
        1.0 - irreducible_water_saturation,
    )
    # Two-phase oil-water point (Sg = 0), so water fills whatever oil does not
    water_saturation = _clamp(
        1.0 - oil_saturation,
        irreducible_water_saturation,
        1.0 - residual_oil_saturation_water,
    )
    return water_saturation, oil_saturation, 0.0


@numba.njit(cache=True)
def get_gas_oil_point_sweep_along_gas_saturation(
    gas_saturation: Number,
    irreducible_water_saturation: Number,
    residual_oil_saturation_gas: Number,
    residual_gas_saturation: Number,
) -> tuple[Number, Number, Number]:
    """
    Clamp a gas saturation value to the gas-oil mobile window and return the
    corresponding (Sw, So, Sg) triple with Sw fixed at swc.

    :param gas_saturation: Candidate gas saturation to clamp.
    :param irreducible_water_saturation: Connate water saturation (swc).
    :param residual_oil_saturation_gas: Residual oil saturation in a gas flood
        (sorg).
    :param residual_gas_saturation: Residual gas saturation (sgr).
    :return: Three-tuple `(Sw, So, Sg)` with Sw = swc.
    """
    gas_saturation = _clamp(
        gas_saturation,
        residual_gas_saturation,
        1.0 - irreducible_water_saturation - residual_oil_saturation_gas,
    )
    oil_saturation = _clamp(
        1.0 - irreducible_water_saturation - gas_saturation,
        0.0,
        1.0 - irreducible_water_saturation,
    )
    return irreducible_water_saturation, oil_saturation, gas_saturation


@numba.njit(cache=True)
def get_gas_oil_point_sweep_along_oil_saturation(
    oil_saturation: Number,
    irreducible_water_saturation: Number,
    residual_oil_saturation_gas: Number,
    residual_gas_saturation: Number,
) -> tuple[Number, Number, Number]:
    """
    Clamp an oil saturation value to the gas-oil mobile window and return the
    corresponding (Sw, So, Sg) triple with Sw fixed at swc.

    :param oil_saturation: Candidate oil saturation to clamp.
    :param irreducible_water_saturation: Connate water saturation (swc).
    :param residual_oil_saturation_gas: Residual oil saturation in a gas flood (sorg).
    :param residual_gas_saturation: Residual gas saturation (sgr).
    :return: Three-tuple `(Sw, So, Sg)` with Sw = swc.
    """
    oil_saturation = _clamp(
        oil_saturation,
        residual_oil_saturation_gas,
        1.0 - irreducible_water_saturation - residual_gas_saturation,
    )
    gas_saturation = _clamp(
        1.0 - irreducible_water_saturation - oil_saturation,
        0.0,
        1.0 - irreducible_water_saturation,
    )
    return irreducible_water_saturation, oil_saturation, gas_saturation


def check_oil_water_sweep_axis_is_water_saturation(
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


def check_gas_oil_sweep_axis_is_gas_saturation(
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


def _resolve_system_orientation(
    *,
    system: str,
    source_table: TwoPhaseRelPermTable | TwoPhaseCapillaryPressureTable | None,
    default_wetting_phase: FluidPhase,
    default_reference_phase: typing.Literal["wetting", "non_wetting"],
    wetting_phase: FluidPhase | str | None,
    reference_phase: typing.Literal["wetting", "non_wetting"] | None,
) -> tuple[FluidPhase, typing.Literal["wetting", "non_wetting"]]:
    """
    Resolve the wetting phase and reference phase of one two-phase system.

    PCHIP resampling reuses the source saturation axis and value arrays, so it
    cannot re-orient a tabular source. A mismatch would not fail on its own, it
    would silently relabel the data (e.g. an So-indexed table treated as Sg-indexed).
    So for a tabular source an omitted argument takes the source's value, and an
    explicit argument that disagrees with the source is rejected.

    For an analytical source (`source_table` is `None`) an omitted wetting phase
    takes `default_wetting_phase` and an omitted reference phase takes
    `default_reference_phase`.

    :param system: `"oil-water"` or `"gas-oil"`, for the error message.
    :param source_table: Two-phase sub-table of a tabular source, or `None` for an
        analytical source.
    :param default_wetting_phase: Wetting phase the source reports for this system.
    :param default_reference_phase: Reference phase used for an analytical source
        when `reference_phase` is omitted.
    :param wetting_phase: Requested wetting phase, or `None` to use the default.
    :param reference_phase: Requested reference phase, or `None` to use the default.
    :return: Two-tuple of `(wetting_phase, reference_phase)`.
    :raises ValidationError: When a tabular source is given an explicit orientation
        that differs from its own.
    """
    requested_wetting_phase = FluidPhase(wetting_phase) if wetting_phase is not None else None

    if source_table is None:
        return (
            requested_wetting_phase or default_wetting_phase,
            reference_phase or default_reference_phase,
        )

    source_wetting_phase = FluidPhase(source_table.wetting_phase)
    source_reference_phase = source_table.reference_phase
    if (
        requested_wetting_phase is not None and requested_wetting_phase != source_wetting_phase
    ) or (reference_phase is not None and reference_phase != source_reference_phase):
        raise ValidationError(
            f"Cannot re-orient a tabular {system} table by resampling. The source has "
            f"`wetting_phase={source_wetting_phase!r}` and "
            f"`reference_phase={source_reference_phase!r}`, but "
            f"`wetting_phase={requested_wetting_phase!r}` and "
            f"`reference_phase={reference_phase!r}` were requested. Omit these arguments "
            f"to keep the source orientation."
        )
    return source_wetting_phase, source_reference_phase


def _build_tabular_reference_saturations(
    *,
    source_saturations: npt.NDArray,
    custom_saturations: npt.ArrayLike | None,
    n_points: int,
    n_endpoint_extra: int,
    spacing: Spacing,
    dtype: npt.DTypeLike,
) -> npt.NDArray:
    """
    Build the saturation axis a tabular source is resampled onto.

    The axis spans the source table's own saturation range, since that is the only
    range its data covers. Endpoint enrichment is applied inside that range.

    :param source_saturations: Saturation axis of the source two-phase table.
    :param custom_saturations: User-supplied axis, used as given when not `None`.
    :param n_points: Number of base points.
    :param n_endpoint_extra: Extra knots in each end region of the range.
    :param spacing: Grid spacing strategy.
    :param dtype: Floating-point dtype of the axis.
    :return: Strictly increasing 1-D saturation axis.
    """
    if custom_saturations is not None:
        return np.asarray(custom_saturations, dtype=dtype)
    return build_saturation_reference_field(
        number_of_base_points=n_points,
        saturation_lower_bound=source_saturations[0],
        saturation_upper_bound=source_saturations[-1],
        spacing=spacing,
        number_of_endpoint_extra_points=n_endpoint_extra,
        dtype=dtype,
    )


def _resample_tabular_relperm(
    *,
    source_table: TwoPhaseRelPermTable,
    reference_saturations: npt.NDArray,
    dtype: npt.DTypeLike,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    PCHIP-evaluate a tabular two-phase relative permeability table on a new axis.

    :param source_table: Two-phase sub-table to resample.
    :param reference_saturations: Saturation axis to evaluate on. Values outside the
        source range take the nearest end value.
    :param dtype: Floating-point dtype of the returned saturations.
    :return: Three-tuple of `(reference_saturations, wetting_phase_kr, non_wetting_phase_kr)`.
    """
    resampled_saturations, wetting_phase_kr = pchip_resample(
        source_saturations=source_table.reference_saturation,
        source_values=source_table.wetting_phase_relative_permeability,
        target_saturations=reference_saturations,
        dtype=dtype,
    )
    _, non_wetting_phase_kr = pchip_resample(
        source_saturations=source_table.reference_saturation,
        source_values=source_table.non_wetting_phase_relative_permeability,
        target_saturations=reference_saturations,
        dtype=dtype,
    )
    return resampled_saturations, wetting_phase_kr, non_wetting_phase_kr


def _resample_tabular_capillary_pressure(
    *,
    source_table: TwoPhaseCapillaryPressureTable,
    reference_saturations: npt.NDArray,
    dtype: npt.DTypeLike,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    PCHIP-evaluate a tabular two-phase capillary pressure table on a new axis.

    :param source_table: Two-phase sub-table to resample.
    :param reference_saturations: Saturation axis to evaluate on. Values outside the
        source range take the nearest end value.
    :param dtype: Floating-point dtype of the returned saturations.
    :return: Two-tuple of `(reference_saturations, capillary_pressure_values)`.
    """
    return pchip_resample(
        source_saturations=source_table.reference_saturation,
        source_values=source_table.capillary_pressure,
        target_saturations=reference_saturations,
        dtype=dtype,
    )


def sample_oil_water_relative_permeabilities(
    *,
    relperm_table: RelativePermeabilityTable,
    oil_water_reference_saturations: npt.NDArray,
    sweep_axis_is_water_saturation: bool,
    irreducible_water_saturation: Number,
    residual_oil_saturation_water: Number,
    model_kwargs: dict[str, typing.Any],
    oil_water_wetting_phase: FluidPhase,
    dtype: npt.DTypeLike,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Sample oil-water relative permeability values across the oil-water saturation range.

    The table is evaluated at every point of the reference axis.

    :param relperm_table: Relative permeability table to sample from.
    :param oil_water_reference_saturations: Saturation axis to sample along.
    :param sweep_axis_is_water_saturation: When `True` the reference axis is
        water saturation; when `False` it is oil saturation.
    :param irreducible_water_saturation: Connate water saturation (swc).
    :param residual_oil_saturation_water: Residual oil saturation in a waterflood (sorw).
    :param model_kwargs: Extra keyword arguments forwarded to every table evaluation call.
    :param oil_water_wetting_phase: Wetting phase for the oil-water system.
    :param dtype: Floating-point dtype of the returned arrays.
    :return: Three-tuple of `(reference_saturations, wetting_phase_kr, non_wetting_phase_kr)`.
    """
    water_relative_permeability = np.empty(len(oil_water_reference_saturations), dtype=dtype)
    oil_relative_permeability = np.empty(len(oil_water_reference_saturations), dtype=dtype)

    for index, reference_saturation_value in enumerate(oil_water_reference_saturations):
        if sweep_axis_is_water_saturation:
            water_saturation, oil_saturation, gas_saturation = (
                get_oil_water_point_sweep_along_water_saturation(
                    water_saturation=reference_saturation_value,
                    irreducible_water_saturation=irreducible_water_saturation,
                    residual_oil_saturation_water=residual_oil_saturation_water,
                )
            )
        else:
            water_saturation, oil_saturation, gas_saturation = (
                get_oil_water_point_sweep_along_oil_saturation(
                    oil_saturation=reference_saturation_value,
                    irreducible_water_saturation=irreducible_water_saturation,
                    residual_oil_saturation_water=residual_oil_saturation_water,
                )
            )

        relative_permeabilities = relperm_table.evaluate(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **model_kwargs,
        )
        water_relative_permeability[index] = relative_permeabilities["water"]
        oil_relative_permeability[index] = relative_permeabilities["oil"]

    if oil_water_wetting_phase == FluidPhase.WATER:
        return (
            oil_water_reference_saturations,
            water_relative_permeability,
            oil_relative_permeability,
        )
    return (
        oil_water_reference_saturations,
        oil_relative_permeability,
        water_relative_permeability,
    )


def sample_gas_oil_relative_permeabilities(
    *,
    relperm_table: RelativePermeabilityTable,
    gas_oil_reference_saturations: npt.NDArray,
    sweep_axis_is_gas_saturation: bool,
    irreducible_water_saturation: Number,
    residual_oil_saturation_gas: Number,
    residual_gas_saturation: Number,
    model_kwargs: dict[str, typing.Any],
    gas_oil_wetting_phase: FluidPhase,
    dtype: npt.DTypeLike,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Sample gas-oil relative permeability values across the gas-oil saturation range.

    The table is evaluated at every point of the reference axis.

    :param relperm_table: Relative permeability table to sample from.
    :param gas_oil_reference_saturations: Saturation axis to sample along.
    :param sweep_axis_is_gas_saturation: When `True` the reference axis is
        gas saturation; when `False` it is oil saturation.
    :param irreducible_water_saturation: Connate water saturation (swc).
    :param residual_oil_saturation_gas: Residual oil saturation in a gas flood (sorg).
    :param residual_gas_saturation: Residual gas saturation (sgr).
    :param model_kwargs: Extra keyword arguments forwarded to every table evaluation call.
    :param gas_oil_wetting_phase: Wetting phase for the gas-oil system.
    :param dtype: Floating-point dtype of the returned arrays.
    :return: Three-tuple of `(reference_saturations, wetting_phase_kr, non_wetting_phase_kr)`.
    """
    gas_relative_permeability = np.empty(len(gas_oil_reference_saturations), dtype=dtype)
    oil_relative_permeability = np.empty(len(gas_oil_reference_saturations), dtype=dtype)

    for index, reference_saturation_value in enumerate(gas_oil_reference_saturations):
        if sweep_axis_is_gas_saturation:
            water_saturation, oil_saturation, gas_saturation = (
                get_gas_oil_point_sweep_along_gas_saturation(
                    gas_saturation=reference_saturation_value,
                    irreducible_water_saturation=irreducible_water_saturation,
                    residual_oil_saturation_gas=residual_oil_saturation_gas,
                    residual_gas_saturation=residual_gas_saturation,
                )
            )
        else:
            water_saturation, oil_saturation, gas_saturation = (
                get_gas_oil_point_sweep_along_oil_saturation(
                    oil_saturation=reference_saturation_value,
                    irreducible_water_saturation=irreducible_water_saturation,
                    residual_oil_saturation_gas=residual_oil_saturation_gas,
                    residual_gas_saturation=residual_gas_saturation,
                )
            )

        relative_permeabilities = relperm_table.evaluate(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **model_kwargs,
        )
        gas_relative_permeability[index] = relative_permeabilities["gas"]
        oil_relative_permeability[index] = relative_permeabilities["oil"]

    if gas_oil_wetting_phase == FluidPhase.OIL:
        return (
            gas_oil_reference_saturations,
            oil_relative_permeability,
            gas_relative_permeability,
        )
    return (
        gas_oil_reference_saturations,
        gas_relative_permeability,
        oil_relative_permeability,
    )


def sample_oil_water_capillary_pressure(
    *,
    capillary_pressure_table: CapillaryPressureTable,
    oil_water_reference_saturations: npt.NDArray,
    sweep_axis_is_water_saturation: bool,
    irreducible_water_saturation: Number,
    residual_oil_saturation_water: Number,
    model_kwargs: dict[str, typing.Any],
    dtype: npt.DTypeLike,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Sample oil-water capillary pressure values across the oil-water saturation range.

    The table is evaluated at every point of the reference axis.

    :param capillary_pressure_table: Capillary pressure table to sample from.
    :param oil_water_reference_saturations: Saturation axis to sample along.
    :param sweep_axis_is_water_saturation: When `True` the reference axis is
        water saturation; when `False` it is oil saturation.
    :param irreducible_water_saturation: Connate water saturation (swc).
    :param residual_oil_saturation_water: Residual oil saturation in a waterflood (sorw).
    :param model_kwargs: Extra keyword arguments forwarded to every table
        evaluation call.
    :param dtype: Floating-point dtype of the returned arrays.
    :return: Two-tuple of `(reference_saturations, capillary_pressure_values)`.
    """
    capillary_pressure_values = np.empty(len(oil_water_reference_saturations), dtype=dtype)
    for index, reference_saturation_value in enumerate(oil_water_reference_saturations):
        if sweep_axis_is_water_saturation:
            water_saturation, oil_saturation, gas_saturation = (
                get_oil_water_point_sweep_along_water_saturation(
                    water_saturation=reference_saturation_value,
                    irreducible_water_saturation=irreducible_water_saturation,
                    residual_oil_saturation_water=residual_oil_saturation_water,
                )
            )
        else:
            water_saturation, oil_saturation, gas_saturation = (
                get_oil_water_point_sweep_along_oil_saturation(
                    oil_saturation=reference_saturation_value,
                    irreducible_water_saturation=irreducible_water_saturation,
                    residual_oil_saturation_water=residual_oil_saturation_water,
                )
            )

        capillary_pressures = capillary_pressure_table.evaluate(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **model_kwargs,
        )
        capillary_pressure_values[index] = capillary_pressures["oil_water"]

    return oil_water_reference_saturations, capillary_pressure_values


def sample_gas_oil_capillary_pressure(
    *,
    capillary_pressure_table: CapillaryPressureTable,
    gas_oil_reference_saturations: npt.NDArray,
    sweep_axis_is_gas_saturation: bool,
    irreducible_water_saturation: Number,
    residual_oil_saturation_gas: Number,
    residual_gas_saturation: Number,
    model_kwargs: dict[str, typing.Any],
    dtype: npt.DTypeLike,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Sample gas-oil capillary pressure values across the gas-oil saturation range.

    The table is evaluated at every point of the reference axis.

    :param capillary_pressure_table: Capillary pressure table to sample from.
    :param gas_oil_reference_saturations: Saturation axis to sample along.
    :param sweep_axis_is_gas_saturation: When `True` the reference axis is
        gas saturation; when `False` it is oil saturation.
    :param irreducible_water_saturation: Connate water saturation (swc).
    :param residual_oil_saturation_gas: Residual oil saturation in a gas flood (sorg).
    :param residual_gas_saturation: Residual gas saturation (sgr).
    :param model_kwargs: Extra keyword arguments forwarded to every table evaluation call.
    :param dtype: Floating-point dtype of the returned arrays.
    :return: Two-tuple of `(reference_saturations, capillary_pressure_values)`.
    """
    capillary_pressure_values = np.empty(len(gas_oil_reference_saturations), dtype=dtype)
    for index, reference_saturation_value in enumerate(gas_oil_reference_saturations):
        if sweep_axis_is_gas_saturation:
            water_saturation, oil_saturation, gas_saturation = (
                get_gas_oil_point_sweep_along_gas_saturation(
                    gas_saturation=reference_saturation_value,
                    irreducible_water_saturation=irreducible_water_saturation,
                    residual_oil_saturation_gas=residual_oil_saturation_gas,
                    residual_gas_saturation=residual_gas_saturation,
                )
            )
        else:
            water_saturation, oil_saturation, gas_saturation = (
                get_gas_oil_point_sweep_along_oil_saturation(
                    oil_saturation=reference_saturation_value,
                    irreducible_water_saturation=irreducible_water_saturation,
                    residual_oil_saturation_gas=residual_oil_saturation_gas,
                    residual_gas_saturation=residual_gas_saturation,
                )
            )

        capillary_pressures = capillary_pressure_table.evaluate(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **model_kwargs,
        )
        capillary_pressure_values[index] = capillary_pressures["gas_oil"]
    return gas_oil_reference_saturations, capillary_pressure_values


def _resample_three_phase_relperm_table(
    table: ThreePhaseRelPermTable,
    *,
    oil_water_wetting_phase: FluidPhase | str | None,
    gas_oil_wetting_phase: FluidPhase | str | None,
    oil_water_reference_phase: typing.Literal["wetting", "non_wetting"] | None,
    gas_oil_reference_phase: typing.Literal["wetting", "non_wetting"] | None,
    n_points: int,
    n_endpoint_extra: int,
    spacing: Spacing,
    oil_water_reference_saturation: npt.ArrayLike | None,
    gas_oil_reference_saturation: npt.ArrayLike | None,
    mixing_rule: MixingRule | str | None,
    dtype: np.dtype,
) -> ThreePhaseRelPermTable:
    """
    Resample a tabular `ThreePhaseRelPermTable` onto a new saturation field.

    Needs none of the saturation endpoints. The new field spans each sub-table's own
    saturation range, with `n_endpoint_extra` extra knots in each end region, unless
    a custom axis is given. Orientation and the minimum kr floors are taken from the source.

    :raises ValidationError: When an explicit orientation differs from the source's.
    """
    oil_water_source = table.oil_water_table
    gas_oil_source = table.gas_oil_table

    oil_water_wetting, oil_water_reference = _resolve_system_orientation(
        system="oil-water",
        source_table=oil_water_source,
        default_wetting_phase=table.get_oil_water_wetting_phase(),
        default_reference_phase="wetting",
        wetting_phase=oil_water_wetting_phase,
        reference_phase=oil_water_reference_phase,
    )
    gas_oil_wetting, gas_oil_reference = _resolve_system_orientation(
        system="gas-oil",
        source_table=gas_oil_source,
        default_wetting_phase=table.get_gas_oil_wetting_phase(),
        default_reference_phase="non_wetting",
        wetting_phase=gas_oil_wetting_phase,
        reference_phase=gas_oil_reference_phase,
    )

    oil_water_axis, oil_water_wetting_kr, oil_water_non_wetting_kr = _resample_tabular_relperm(
        source_table=oil_water_source,
        reference_saturations=_build_tabular_reference_saturations(
            source_saturations=oil_water_source.reference_saturation,
            custom_saturations=oil_water_reference_saturation,
            n_points=n_points,
            n_endpoint_extra=n_endpoint_extra,
            spacing=spacing,
            dtype=dtype,
        ),
        dtype=dtype,
    )
    gas_oil_axis, gas_oil_wetting_kr, gas_oil_non_wetting_kr = _resample_tabular_relperm(
        source_table=gas_oil_source,
        reference_saturations=_build_tabular_reference_saturations(
            source_saturations=gas_oil_source.reference_saturation,
            custom_saturations=gas_oil_reference_saturation,
            n_points=n_points,
            n_endpoint_extra=n_endpoint_extra,
            spacing=spacing,
            dtype=dtype,
        ),
        dtype=dtype,
    )

    oil_water_table = TwoPhaseRelPermTable(
        wetting_phase=oil_water_wetting,
        non_wetting_phase=oil_water_source.non_wetting_phase,
        reference_saturation=typing.cast("NumberArray[OneDimension]", oil_water_axis),
        wetting_phase_relative_permeability=typing.cast(
            "NumberArray[OneDimension]", oil_water_wetting_kr
        ),
        non_wetting_phase_relative_permeability=typing.cast(
            "NumberArray[OneDimension]", oil_water_non_wetting_kr
        ),
        reference_phase=oil_water_reference,
        min_wetting_relperm=oil_water_source.min_wetting_relperm,
        min_non_wetting_relperm=oil_water_source.min_non_wetting_relperm,
        dtype=dtype,
    )
    gas_oil_table = TwoPhaseRelPermTable(
        wetting_phase=gas_oil_wetting,
        non_wetting_phase=gas_oil_source.non_wetting_phase,
        reference_saturation=typing.cast("NumberArray[OneDimension]", gas_oil_axis),
        wetting_phase_relative_permeability=typing.cast(
            "NumberArray[OneDimension]", gas_oil_wetting_kr
        ),
        non_wetting_phase_relative_permeability=typing.cast(
            "NumberArray[OneDimension]", gas_oil_non_wetting_kr
        ),
        reference_phase=gas_oil_reference,
        min_wetting_relperm=gas_oil_source.min_wetting_relperm,
        min_non_wetting_relperm=gas_oil_source.min_non_wetting_relperm,
        dtype=dtype,
    )
    return ThreePhaseRelPermTable(
        oil_water_table=oil_water_table,
        gas_oil_table=gas_oil_table,
        mixing_rule=mixing_rule if mixing_rule is not None else table.mixing_rule,
    )


def as_three_phase_relperm_table(
    table: RelativePermeabilityTable,
    *,
    irreducible_water_saturation: Number | None = None,
    residual_oil_saturation_water: Number | None = None,
    residual_oil_saturation_gas: Number | None = None,
    residual_gas_saturation: Number | None = None,
    model_kwargs: dict[str, typing.Any] | None = None,
    oil_water_wetting_phase: FluidPhase | str | None = None,
    gas_oil_wetting_phase: FluidPhase | str | None = None,
    oil_water_reference_phase: typing.Literal["wetting", "non_wetting"] | None = None,
    gas_oil_reference_phase: typing.Literal["wetting", "non_wetting"] | None = None,
    n_points: int = 200,
    n_endpoint_extra: int = 20,
    spacing: Spacing = "cosine",
    oil_water_reference_saturation: npt.ArrayLike | None = None,
    gas_oil_reference_saturation: npt.ArrayLike | None = None,
    mixing_rule: MixingRule | str | None = None,
    dtype: npt.DTypeLike = None,
) -> ThreePhaseRelPermTable:
    """
    Convert any `RelativePermeabilityTable` to a `ThreePhaseRelPermTable`
    backed by piecewise-linear `TwoPhaseRelPermTable` instances.

    Analytical derivatives are sampled at every knot and stored in the
    two-phase sub-tables so that `get_*_derivative` returns smooth,
    consistent values instead of piecewise-linear slopes.

    For tabular source models (`ThreePhaseRelPermTable`) the existing knots are
    PCHIP-resampled onto a new field spanning each sub-table's own saturation range,
    built from `n_points`, `n_endpoint_extra` and `spacing` (or the custom
    `*_reference_saturation` axes), recovering C¹-continuous values and derivatives
    without any additional table calls. The saturation endpoints are not needed for a
    tabular source and are ignored. Orientation and minimum kr floors are taken from the
    source; an explicit wetting or reference phase that differs from the source's raises
    a `ValidationError`, since resampling cannot re-orient a table.

    :param table: Source analytical or tabular relative permeability table.
    :param irreducible_water_saturation: Irreducible water saturation (swc).
    :param residual_oil_saturation_water: Residual oil saturation in a waterflood (sorw).
    :param residual_oil_saturation_gas: Residual oil saturation in a gas flood (sorg).
    :param residual_gas_saturation: Residual gas saturation (sgr).
    :param model_kwargs: Extra kwargs forwarded to every table evaluation call.
    :param oil_water_wetting_phase: Wetting phase for the oil-water sub-table.
    :param gas_oil_wetting_phase: Wetting phase for the gas-oil sub-table.
    :param oil_water_reference_phase: Reference saturation axis for the oil-water table.
        Defaults to `"wetting"`, or to the source's for a tabular source.
    :param gas_oil_reference_phase: Reference saturation axis for the gas-oil table.
        Defaults to `"non_wetting"`, or to the source's for a tabular source.
    :param n_points: Base number of sample points per sub-table.
    :param n_endpoint_extra: Extra knots in each end region (first and last 10 % of the
        range). Pass `0` to disable endpoint refinement.
    :param spacing: Grid spacing mode.
    :param oil_water_reference_saturation: Custom saturation axis for the
        oil-water sub-table. Overrides the auto-generated field when supplied.
    :param gas_oil_reference_saturation: Custom saturation axis for the gas-oil
        sub-table. Overrides the auto-generated field when supplied.
    :param mixing_rule: Three-phase oil relative permeability mixing rule.
    :param dtype: Floating-point dtype of the returned table. Defaults to the source
        table's dtype, else the global dtype.
    :return: `ThreePhaseRelPermTable` with piecewise-linear sub-tables.
    """
    dtype = np.dtype(dtype if dtype is not None else getattr(table, "dtype", get_dtype()))
    if isinstance(table, ThreePhaseRelPermTable):
        return _resample_three_phase_relperm_table(
            table,
            oil_water_wetting_phase=oil_water_wetting_phase,
            gas_oil_wetting_phase=gas_oil_wetting_phase,
            oil_water_reference_phase=oil_water_reference_phase,
            gas_oil_reference_phase=gas_oil_reference_phase,
            n_points=n_points,
            n_endpoint_extra=n_endpoint_extra,
            spacing=spacing,
            oil_water_reference_saturation=oil_water_reference_saturation,
            gas_oil_reference_saturation=gas_oil_reference_saturation,
            mixing_rule=mixing_rule,
            dtype=dtype,
        )

    resolved_irreducible_water_saturation = _resolve_saturation_endpoint(
        arg_value=irreducible_water_saturation,
        table=table,
        attr_name="irreducible_water_saturation",
        caller="as_three_phase_relperm_table",
    )
    resolved_residual_oil_saturation_water = _resolve_saturation_endpoint(
        arg_value=residual_oil_saturation_water,
        table=table,
        attr_name="residual_oil_saturation_water",
        caller="as_three_phase_relperm_table",
    )
    resolved_residual_oil_saturation_gas = _resolve_saturation_endpoint(
        arg_value=residual_oil_saturation_gas,
        table=table,
        attr_name="residual_oil_saturation_gas",
        caller="as_three_phase_relperm_table",
    )
    resolved_residual_gas_saturation = _resolve_saturation_endpoint(
        arg_value=residual_gas_saturation,
        table=table,
        attr_name="residual_gas_saturation",
        caller="as_three_phase_relperm_table",
    )

    model_kwargs = {
        "irreducible_water_saturation": resolved_irreducible_water_saturation,
        "residual_oil_saturation_water": resolved_residual_oil_saturation_water,
        "residual_oil_saturation_gas": resolved_residual_oil_saturation_gas,
        "residual_gas_saturation": resolved_residual_gas_saturation,
        **(model_kwargs or {}),
    }

    oil_water_wetting_phase_resolved, oil_water_reference_phase = _resolve_system_orientation(
        system="oil-water",
        source_table=None,
        default_wetting_phase=table.get_oil_water_wetting_phase(),
        default_reference_phase="wetting",
        wetting_phase=oil_water_wetting_phase,
        reference_phase=oil_water_reference_phase,
    )
    gas_oil_wetting_phase_resolved, gas_oil_reference_phase = _resolve_system_orientation(
        system="gas-oil",
        source_table=None,
        default_wetting_phase=table.get_gas_oil_wetting_phase(),
        default_reference_phase="non_wetting",
        wetting_phase=gas_oil_wetting_phase,
        reference_phase=gas_oil_reference_phase,
    )
    oil_water_non_wetting_phase = (
        FluidPhase.OIL
        if oil_water_wetting_phase_resolved == FluidPhase.WATER
        else FluidPhase.WATER
    )
    gas_oil_non_wetting_phase = (
        FluidPhase.GAS if gas_oil_wetting_phase_resolved == FluidPhase.OIL else FluidPhase.OIL
    )

    sweep_oil_water_axis_is_water_saturation = check_oil_water_sweep_axis_is_water_saturation(
        wetting_phase=oil_water_wetting_phase_resolved,
        reference_phase=oil_water_reference_phase,
    )
    sweep_gas_oil_axis_is_gas_saturation = check_gas_oil_sweep_axis_is_gas_saturation(
        wetting_phase=gas_oil_wetting_phase_resolved,
        reference_phase=gas_oil_reference_phase,
    )

    if oil_water_reference_saturation is not None:
        oil_water_reference_saturations = np.asarray(oil_water_reference_saturation, dtype=dtype)
    else:
        if sweep_oil_water_axis_is_water_saturation:
            oil_water_lower_bound = resolved_irreducible_water_saturation
            oil_water_upper_bound = 1.0 - resolved_residual_oil_saturation_water
        else:
            oil_water_lower_bound = resolved_residual_oil_saturation_water
            oil_water_upper_bound = 1.0 - resolved_irreducible_water_saturation
        oil_water_reference_saturations = build_saturation_reference_field(
            number_of_base_points=n_points,
            saturation_lower_bound=oil_water_lower_bound,
            saturation_upper_bound=oil_water_upper_bound,
            spacing=spacing,
            number_of_endpoint_extra_points=n_endpoint_extra,
            dtype=dtype,
        )

    if gas_oil_reference_saturation is not None:
        gas_oil_reference_saturations = np.asarray(gas_oil_reference_saturation, dtype=dtype)
    else:
        if sweep_gas_oil_axis_is_gas_saturation:
            gas_oil_lower_bound = resolved_residual_gas_saturation
            gas_oil_upper_bound = (
                1.0 - resolved_irreducible_water_saturation - resolved_residual_oil_saturation_gas
            )
        else:
            gas_oil_lower_bound = resolved_residual_oil_saturation_gas
            gas_oil_upper_bound = (
                1.0 - resolved_irreducible_water_saturation - resolved_residual_gas_saturation
            )
        gas_oil_reference_saturations = build_saturation_reference_field(
            number_of_base_points=n_points,
            saturation_lower_bound=gas_oil_lower_bound,
            saturation_upper_bound=gas_oil_upper_bound,
            spacing=spacing,
            number_of_endpoint_extra_points=n_endpoint_extra,
            dtype=dtype,
        )

    (
        oil_water_reference_saturations,
        oil_water_wetting_phase_kr,
        oil_water_non_wetting_phase_kr,
    ) = sample_oil_water_relative_permeabilities(
        relperm_table=table,
        oil_water_reference_saturations=oil_water_reference_saturations,
        sweep_axis_is_water_saturation=sweep_oil_water_axis_is_water_saturation,
        irreducible_water_saturation=resolved_irreducible_water_saturation,
        residual_oil_saturation_water=resolved_residual_oil_saturation_water,
        model_kwargs=model_kwargs,
        oil_water_wetting_phase=oil_water_wetting_phase_resolved,
        dtype=dtype,
    )

    (
        gas_oil_reference_saturations,
        gas_oil_wetting_phase_kr,
        gas_oil_non_wetting_phase_kr,
    ) = sample_gas_oil_relative_permeabilities(
        relperm_table=table,
        gas_oil_reference_saturations=gas_oil_reference_saturations,
        sweep_axis_is_gas_saturation=sweep_gas_oil_axis_is_gas_saturation,
        irreducible_water_saturation=resolved_irreducible_water_saturation,
        residual_oil_saturation_gas=resolved_residual_oil_saturation_gas,
        residual_gas_saturation=resolved_residual_gas_saturation,
        model_kwargs=model_kwargs,
        gas_oil_wetting_phase=gas_oil_wetting_phase_resolved,
        dtype=dtype,
    )

    oil_water_table = TwoPhaseRelPermTable(
        wetting_phase=oil_water_wetting_phase_resolved,
        non_wetting_phase=oil_water_non_wetting_phase,
        reference_saturation=typing.cast(
            "NumberArray[OneDimension]", oil_water_reference_saturations
        ),
        wetting_phase_relative_permeability=typing.cast(
            "NumberArray[OneDimension]", oil_water_wetting_phase_kr
        ),
        non_wetting_phase_relative_permeability=typing.cast(
            "NumberArray[OneDimension]", oil_water_non_wetting_phase_kr
        ),
        reference_phase=oil_water_reference_phase,
        dtype=dtype,
    )
    gas_oil_table = TwoPhaseRelPermTable(
        wetting_phase=gas_oil_wetting_phase_resolved,
        non_wetting_phase=gas_oil_non_wetting_phase,
        reference_saturation=typing.cast(
            "NumberArray[OneDimension]", gas_oil_reference_saturations
        ),
        wetting_phase_relative_permeability=typing.cast(
            "NumberArray[OneDimension]", gas_oil_wetting_phase_kr
        ),
        non_wetting_phase_relative_permeability=typing.cast(
            "NumberArray[OneDimension]", gas_oil_non_wetting_phase_kr
        ),
        reference_phase=gas_oil_reference_phase,
        dtype=dtype,
    )

    if mixing_rule is None:
        mixing_rule = getattr(table, "mixing_rule", "baker_linear_rule")
    return ThreePhaseRelPermTable(
        oil_water_table=oil_water_table,
        gas_oil_table=gas_oil_table,
        mixing_rule=mixing_rule,
    )


def _resample_three_phase_capillary_pressure_table(
    table: ThreePhaseCapillaryPressureTable,
    *,
    current_unit_system: UnitSystem,
    oil_water_wetting_phase: FluidPhase | str | None,
    gas_oil_wetting_phase: FluidPhase | str | None,
    oil_water_reference_phase: typing.Literal["wetting", "non_wetting"] | None,
    gas_oil_reference_phase: typing.Literal["wetting", "non_wetting"] | None,
    n_points: int,
    n_endpoint_extra: int,
    spacing: Spacing,
    oil_water_reference_saturation: npt.ArrayLike | None,
    gas_oil_reference_saturation: npt.ArrayLike | None,
    dtype: np.dtype,
) -> ThreePhaseCapillaryPressureTable:
    """
    Resample a tabular `ThreePhaseCapillaryPressureTable` onto a new saturation field.

    Needs none of the saturation endpoints. The new field spans each sub-table's own
    saturation range, with `n_endpoint_extra` extra knots in each end region, unless
    a custom axis is given. Orientation is taken from the source.

    :param current_unit_system: Unit system of the source table, kept on the result.
    :raises ValidationError: When an explicit orientation differs from the source's.
    """
    oil_water_source = table.oil_water_table
    gas_oil_source = table.gas_oil_table

    oil_water_wetting, oil_water_reference = _resolve_system_orientation(
        system="oil-water",
        source_table=oil_water_source,
        default_wetting_phase=table.get_oil_water_wetting_phase(),
        default_reference_phase="wetting",
        wetting_phase=oil_water_wetting_phase,
        reference_phase=oil_water_reference_phase,
    )
    gas_oil_wetting, gas_oil_reference = _resolve_system_orientation(
        system="gas-oil",
        source_table=gas_oil_source,
        default_wetting_phase=table.get_gas_oil_wetting_phase(),
        default_reference_phase="non_wetting",
        wetting_phase=gas_oil_wetting_phase,
        reference_phase=gas_oil_reference_phase,
    )

    oil_water_axis, oil_water_values = _resample_tabular_capillary_pressure(
        source_table=oil_water_source,
        reference_saturations=_build_tabular_reference_saturations(
            source_saturations=oil_water_source.reference_saturation,
            custom_saturations=oil_water_reference_saturation,
            n_points=n_points,
            n_endpoint_extra=n_endpoint_extra,
            spacing=spacing,
            dtype=dtype,
        ),
        dtype=dtype,
    )
    gas_oil_axis, gas_oil_values = _resample_tabular_capillary_pressure(
        source_table=gas_oil_source,
        reference_saturations=_build_tabular_reference_saturations(
            source_saturations=gas_oil_source.reference_saturation,
            custom_saturations=gas_oil_reference_saturation,
            n_points=n_points,
            n_endpoint_extra=n_endpoint_extra,
            spacing=spacing,
            dtype=dtype,
        ),
        dtype=dtype,
    )

    return ThreePhaseCapillaryPressureTable(
        oil_water_table=TwoPhaseCapillaryPressureTable(
            wetting_phase=oil_water_wetting,
            non_wetting_phase=oil_water_source.non_wetting_phase,
            reference_saturation=typing.cast("NumberArray[OneDimension]", oil_water_axis),
            capillary_pressure=typing.cast("NumberArray[OneDimension]", oil_water_values),
            reference_phase=oil_water_reference,
            dtype=dtype,
            unit_system=current_unit_system,
        ),
        gas_oil_table=TwoPhaseCapillaryPressureTable(
            wetting_phase=gas_oil_wetting,
            non_wetting_phase=gas_oil_source.non_wetting_phase,
            reference_saturation=typing.cast("NumberArray[OneDimension]", gas_oil_axis),
            capillary_pressure=typing.cast("NumberArray[OneDimension]", gas_oil_values),
            reference_phase=gas_oil_reference,
            dtype=dtype,
            unit_system=current_unit_system,
        ),
    )


def as_three_phase_capillary_pressure_table(
    table: CapillaryPressureTable,
    *,
    irreducible_water_saturation: Number | None = None,
    residual_oil_saturation_water: Number | None = None,
    residual_oil_saturation_gas: Number | None = None,
    residual_gas_saturation: Number | None = None,
    model_kwargs: dict[str, typing.Any] | None = None,
    oil_water_wetting_phase: FluidPhase | str | None = None,
    gas_oil_wetting_phase: FluidPhase | str | None = None,
    oil_water_reference_phase: typing.Literal["wetting", "non_wetting"] | None = None,
    gas_oil_reference_phase: typing.Literal["wetting", "non_wetting"] | None = None,
    n_points: int = 200,
    n_endpoint_extra: int = 30,
    spacing: Spacing = "cosine",
    oil_water_reference_saturation: npt.ArrayLike | None = None,
    gas_oil_reference_saturation: npt.ArrayLike | None = None,
    unit_system: UnitSystem | None = None,
    dtype: npt.DTypeLike = None,
) -> ThreePhaseCapillaryPressureTable:
    """
    Convert any `CapillaryPressureTable` to a `ThreePhaseCapillaryPressureTable`
    backed by piecewise-linear `TwoPhaseCapillaryPressureTable` instances.

    Analytical derivatives are sampled at every knot and stored in the
    two-phase sub-tables. The default `n_endpoint_extra=30` (vs 20 for
    relperm) reflects that Pc curves are unbounded near residual saturation,
    making endpoint fidelity especially important for implicit convergence.

    For tabular source models (`ThreePhaseCapillaryPressureTable`) the existing knots are
    PCHIP-resampled onto a new field spanning each sub-table's own saturation range,
    built from `n_points`, `n_endpoint_extra` and `spacing` (or the custom
    `*_reference_saturation` axes), recovering C¹-continuous values and derivatives
    without any additional table calls. The saturation endpoints are not needed for a
    tabular source and are ignored. Orientation is taken from the source; an explicit
    wetting or reference phase that differs from the source's raises a `ValidationError`,
    since resampling cannot re-orient a table.

    :param table: Source analytical or tabular capillary pressure table.
    :param irreducible_water_saturation: Irreducible water saturation (swc).
    :param residual_oil_saturation_water: Residual oil saturation in a waterflood (sorw).
    :param residual_oil_saturation_gas: Residual oil saturation in a gas flood (sorg).
    :param residual_gas_saturation: Residual gas saturation (sgr).
    :param model_kwargs: Extra kwargs forwarded to every table evaluation call.
    :param oil_water_wetting_phase: Wetting phase for the oil-water sub-table.
    :param gas_oil_wetting_phase: Wetting phase for the gas-oil sub-table.
    :param oil_water_reference_phase: Reference saturation axis for the oil-water table.
        Defaults to `"wetting"`, or to the source's for a tabular source.
    :param gas_oil_reference_phase: Reference saturation axis for the gas-oil table.
        Defaults to `"non_wetting"`, or to the source's for a tabular source.
    :param n_points: Base number of sample points per sub-table.
    :param n_endpoint_extra: Extra knots in each end region (first and last 10 % of the
        range). Pass `0` to disable endpoint refinement.
    :param spacing: Grid spacing mode.
    :param oil_water_reference_saturation: Custom saturation axis for the
        oil-water sub-table. Overrides the auto-generated field when supplied.
    :param gas_oil_reference_saturation: Custom saturation axis for the gas-oil
        sub-table. Overrides the auto-generated field when supplied.
    :param unit_system: Unit system to convert the result to. When omitted the result
        stays in the source table's unit system.
    :param dtype: Floating-point dtype of the returned table. Defaults to the source
        table's dtype, else the global dtype.
    :return: `ThreePhaseCapillaryPressureTable` backed by piecewise-linear sub-tables.
    """
    dtype = np.dtype(dtype if dtype is not None else getattr(table, "dtype", get_dtype()))
    current_unit_system = table.unit_system
    if current_unit_system is None:
        raise ValidationError(
            "Cannot determine the unit system of the provided table. Pass `unit_system`."
        )
    should_convert = unit_system is not None and current_unit_system != unit_system

    if isinstance(table, ThreePhaseCapillaryPressureTable):
        resampled_table = _resample_three_phase_capillary_pressure_table(
            table,
            current_unit_system=current_unit_system,
            oil_water_wetting_phase=oil_water_wetting_phase,
            gas_oil_wetting_phase=gas_oil_wetting_phase,
            oil_water_reference_phase=oil_water_reference_phase,
            gas_oil_reference_phase=gas_oil_reference_phase,
            n_points=n_points,
            n_endpoint_extra=n_endpoint_extra,
            spacing=spacing,
            oil_water_reference_saturation=oil_water_reference_saturation,
            gas_oil_reference_saturation=gas_oil_reference_saturation,
            dtype=dtype,
        )
        if should_convert:
            return resampled_table.convert(unit_system)  # type: ignore[arg-type]
        return resampled_table

    resolved_irreducible_water_saturation = _resolve_saturation_endpoint(
        arg_value=irreducible_water_saturation,
        table=table,
        attr_name="irreducible_water_saturation",
        caller="as_three_phase_capillary_pressure_table",
    )
    resolved_residual_oil_saturation_water = _resolve_saturation_endpoint(
        arg_value=residual_oil_saturation_water,
        table=table,
        attr_name="residual_oil_saturation_water",
        caller="as_three_phase_capillary_pressure_table",
    )
    resolved_residual_oil_saturation_gas = _resolve_saturation_endpoint(
        arg_value=residual_oil_saturation_gas,
        table=table,
        attr_name="residual_oil_saturation_gas",
        caller="as_three_phase_capillary_pressure_table",
    )
    resolved_residual_gas_saturation = _resolve_saturation_endpoint(
        arg_value=residual_gas_saturation,
        table=table,
        attr_name="residual_gas_saturation",
        caller="as_three_phase_capillary_pressure_table",
    )

    model_kwargs = {
        "irreducible_water_saturation": resolved_irreducible_water_saturation,
        "residual_oil_saturation_water": resolved_residual_oil_saturation_water,
        "residual_oil_saturation_gas": resolved_residual_oil_saturation_gas,
        "residual_gas_saturation": resolved_residual_gas_saturation,
        **(model_kwargs or {}),
    }

    oil_water_wetting_phase_resolved, oil_water_reference_phase = _resolve_system_orientation(
        system="oil-water",
        source_table=None,
        default_wetting_phase=table.get_oil_water_wetting_phase(),
        default_reference_phase="wetting",
        wetting_phase=oil_water_wetting_phase,
        reference_phase=oil_water_reference_phase,
    )
    gas_oil_wetting_phase_resolved, gas_oil_reference_phase = _resolve_system_orientation(
        system="gas-oil",
        source_table=None,
        default_wetting_phase=table.get_gas_oil_wetting_phase(),
        default_reference_phase="non_wetting",
        wetting_phase=gas_oil_wetting_phase,
        reference_phase=gas_oil_reference_phase,
    )
    oil_water_non_wetting_phase = (
        FluidPhase.OIL
        if oil_water_wetting_phase_resolved == FluidPhase.WATER
        else FluidPhase.WATER
    )
    gas_oil_non_wetting_phase = (
        FluidPhase.GAS if gas_oil_wetting_phase_resolved == FluidPhase.OIL else FluidPhase.OIL
    )

    sweep_oil_water_axis_is_water_saturation = check_oil_water_sweep_axis_is_water_saturation(
        wetting_phase=oil_water_wetting_phase_resolved,
        reference_phase=oil_water_reference_phase,
    )
    sweep_gas_oil_axis_is_gas_saturation = check_gas_oil_sweep_axis_is_gas_saturation(
        wetting_phase=gas_oil_wetting_phase_resolved,
        reference_phase=gas_oil_reference_phase,
    )

    if oil_water_reference_saturation is not None:
        oil_water_reference_saturations = np.asarray(oil_water_reference_saturation, dtype=dtype)
    else:
        if sweep_oil_water_axis_is_water_saturation:
            oil_water_lower_bound = resolved_irreducible_water_saturation
            oil_water_upper_bound = 1.0 - resolved_residual_oil_saturation_water
        else:
            oil_water_lower_bound = resolved_residual_oil_saturation_water
            oil_water_upper_bound = 1.0 - resolved_irreducible_water_saturation
        oil_water_reference_saturations = build_saturation_reference_field(
            number_of_base_points=n_points,
            saturation_lower_bound=oil_water_lower_bound,
            saturation_upper_bound=oil_water_upper_bound,
            spacing=spacing,
            number_of_endpoint_extra_points=n_endpoint_extra,
            dtype=dtype,
        )

    if gas_oil_reference_saturation is not None:
        gas_oil_reference_saturations = np.asarray(gas_oil_reference_saturation, dtype=dtype)
    else:
        if sweep_gas_oil_axis_is_gas_saturation:
            gas_oil_lower_bound = resolved_residual_gas_saturation
            gas_oil_upper_bound = (
                1.0 - resolved_irreducible_water_saturation - resolved_residual_oil_saturation_gas
            )
        else:
            gas_oil_lower_bound = resolved_residual_oil_saturation_gas
            gas_oil_upper_bound = (
                1.0 - resolved_irreducible_water_saturation - resolved_residual_gas_saturation
            )
        gas_oil_reference_saturations = build_saturation_reference_field(
            number_of_base_points=n_points,
            saturation_lower_bound=gas_oil_lower_bound,
            saturation_upper_bound=gas_oil_upper_bound,
            spacing=spacing,
            number_of_endpoint_extra_points=n_endpoint_extra,
            dtype=dtype,
        )

    (
        oil_water_reference_saturations,
        oil_water_capillary_pressure_values,
    ) = sample_oil_water_capillary_pressure(
        capillary_pressure_table=table,
        oil_water_reference_saturations=oil_water_reference_saturations,
        sweep_axis_is_water_saturation=sweep_oil_water_axis_is_water_saturation,
        irreducible_water_saturation=resolved_irreducible_water_saturation,
        residual_oil_saturation_water=resolved_residual_oil_saturation_water,
        model_kwargs=model_kwargs,
        dtype=dtype,
    )

    (
        gas_oil_reference_saturations,
        gas_oil_capillary_pressure_values,
    ) = sample_gas_oil_capillary_pressure(
        capillary_pressure_table=table,
        gas_oil_reference_saturations=gas_oil_reference_saturations,
        sweep_axis_is_gas_saturation=sweep_gas_oil_axis_is_gas_saturation,
        irreducible_water_saturation=resolved_irreducible_water_saturation,
        residual_oil_saturation_gas=resolved_residual_oil_saturation_gas,
        residual_gas_saturation=resolved_residual_gas_saturation,
        model_kwargs=model_kwargs,
        dtype=dtype,
    )

    oil_water_table = TwoPhaseCapillaryPressureTable(
        wetting_phase=oil_water_wetting_phase_resolved,
        non_wetting_phase=oil_water_non_wetting_phase,
        reference_saturation=typing.cast(
            "NumberArray[OneDimension]", oil_water_reference_saturations
        ),
        capillary_pressure=typing.cast(
            "NumberArray[OneDimension]", oil_water_capillary_pressure_values
        ),
        reference_phase=oil_water_reference_phase,
        dtype=dtype,
        unit_system=current_unit_system,
    )
    gas_oil_table = TwoPhaseCapillaryPressureTable(
        wetting_phase=gas_oil_wetting_phase_resolved,
        non_wetting_phase=gas_oil_non_wetting_phase,
        reference_saturation=typing.cast(
            "NumberArray[OneDimension]", gas_oil_reference_saturations
        ),
        capillary_pressure=typing.cast(
            "NumberArray[OneDimension]", gas_oil_capillary_pressure_values
        ),
        reference_phase=gas_oil_reference_phase,
        dtype=dtype,
        unit_system=current_unit_system,
    )
    three_phase_table = ThreePhaseCapillaryPressureTable(
        oil_water_table=oil_water_table, gas_oil_table=gas_oil_table
    )
    if should_convert:
        return three_phase_table.convert(unit_system)  # type: ignore[arg-type]
    return three_phase_table
