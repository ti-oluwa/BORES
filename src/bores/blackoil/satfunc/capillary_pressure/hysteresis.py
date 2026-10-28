"""
Hysteresis models for capillary pressure.

Implements a Killough-type scanning-curve model for capillary pressure hysteresis.

**References**:

- Killough, J.E. (1976). "Reservoir Simulation With History-Dependent
  Saturation Functions". SPE 5106.
- Land, C.S. (1968). "Calculation of Imbibition Relative Permeability for
  Two- and Three-Phase Flow from Rock Properties". SPE 1942.
- Carlson, F.M. (1981). "Simulation of Relative Permeability Hysteresis to
  the Non-Wetting Phase". SPE 10157.
"""

import math
import typing

import attrs
import numpy as np

from bores.blackoil.satfunc.capillary_pressure.tables import (
    CapillaryPressureTable,
    ThreePhaseCapillaryPressureTable,
    TwoPhaseCapillaryPressureTable,
    query_gas_oil_capillary_pressure,
    query_oil_water_capillary_pressure,
)
from bores.blackoil.satfunc.utils import flatten_kernel_input, scan_curve_kernel
from bores.errors import ValidationError
from bores.types import (
    BooleanArray,
    CapillaryPressureDerivatives,
    CapillaryPressures,
    FluidPhase,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
)

__all__ = ["KilloughCapillaryPressureTable"]

AnyCapillaryTable = TwoPhaseCapillaryPressureTable | CapillaryPressureTable


def query_oil_water_curve(
    table: AnyCapillaryTable,
    *,
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    with_derivatives: bool,
    memo: dict[int, typing.Any],
    **kwargs: typing.Any,
) -> tuple[typing.Any, typing.Any]:
    """
    Oil-water capillary pressure of a table and its slope with respect to water saturation.

    A two-phase table is queried at the oil-water system's own saturations (`So = 1 - Sw`),
    whichever phase its reference axis uses. A three-phase table is evaluated at the three-phase
    point, and `memo` lets a table that serves both systems be evaluated once.

    :param table: Two-phase or three-phase capillary pressure table.
    :param water_saturation: Flat water saturation array.
    :param oil_saturation: Flat oil saturation array.
    :param gas_saturation: Flat gas saturation array.
    :param with_derivatives: Whether to compute the slope. It is `None` otherwise.
    :param memo: Evaluations of three-phase tables already made in this call, by table id.
    :param kwargs: Extra arguments forwarded to a three-phase table.
    :return: Tuple of `Pcow` and `dPcow/dSw`.
    """
    if isinstance(table, TwoPhaseCapillaryPressureTable):
        value = query_oil_water_capillary_pressure(table, water_saturation)
        if not with_derivatives:
            return value, None
        return value, query_oil_water_capillary_pressure(table, water_saturation, derivative=True)
    values, slopes = evaluate_three_phase_table(
        table,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        with_derivatives=with_derivatives,
        memo=memo,
        **kwargs,
    )
    return values["oil_water"], None if slopes is None else slopes["dpcow_dsw"]


def query_gas_oil_curve(
    table: AnyCapillaryTable,
    *,
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    connate_water_saturation: Number,
    with_derivatives: bool,
    memo: dict[int, typing.Any],
    **kwargs: typing.Any,
) -> tuple[typing.Any, typing.Any]:
    """
    Gas-oil capillary pressure of a table and its slope with respect to gas saturation.

    A two-phase table is queried at the gas-oil system's own saturations
    (`So = 1 - swc - Sg`), whichever phase its reference axis uses. A three-phase table is
    evaluated at the three-phase point, and `memo` lets a table that serves both systems be
    evaluated once.

    :param table: Two-phase or three-phase capillary pressure table.
    :param water_saturation: Flat water saturation array.
    :param oil_saturation: Flat oil saturation array.
    :param gas_saturation: Flat gas saturation array.
    :param connate_water_saturation: Connate water saturation, which places the gas-oil system.
    :param with_derivatives: Whether to compute the slope. It is `None` otherwise.
    :param memo: Evaluations of three-phase tables already made in this call, by table id.
    :param kwargs: Extra arguments forwarded to a three-phase table.
    :return: Tuple of `Pcgo` and `dPcgo/dSg`.
    """
    if isinstance(table, TwoPhaseCapillaryPressureTable):
        value = query_gas_oil_capillary_pressure(table, gas_saturation, connate_water_saturation)
        if not with_derivatives:
            return value, None
        return value, query_gas_oil_capillary_pressure(
            table, gas_saturation, connate_water_saturation, derivative=True
        )
    values, slopes = evaluate_three_phase_table(
        table,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        with_derivatives=with_derivatives,
        memo=memo,
        **kwargs,
    )
    return values["gas_oil"], None if slopes is None else slopes["dpcgo_dsg"]


def evaluate_three_phase_table(
    table: CapillaryPressureTable,
    *,
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    with_derivatives: bool,
    memo: dict[int, typing.Any],
    **kwargs: typing.Any,
) -> tuple[CapillaryPressures, CapillaryPressureDerivatives | None]:
    """
    Evaluate a three-phase capillary pressure table, once per call for a table used by both systems.

    The values and derivatives come from a single `evaluate_with_derivatives` call when the
    derivatives are wanted, which shares work inside models that can.
    """
    cached = memo.get(id(table))
    if cached is not None and (cached[1] is not None or not with_derivatives):
        return cached
    arguments = dict(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    if with_derivatives:
        result = table.evaluate_with_derivatives(**arguments)
    else:
        result = (table.evaluate(**arguments), None)
    memo[id(table)] = result
    return result


def scan_curve(
    *,
    size: int,
    flat_shape: tuple[int, ...],
    saturation: NumberArray[OneDimension],
    maximum_saturation: NumberArray[OneDimension],
    imbibition_flag: NumberArray[OneDimension],
    reversal_saturation: NumberArray[OneDimension],
    scanning_exponent: Number,
    drainage: tuple[typing.Any, typing.Any],
    imbibition: tuple[typing.Any, typing.Any],
    with_derivatives: bool,
) -> tuple[typing.Any, typing.Any]:
    """
    Scan one capillary pressure curve between its drainage and imbibition bounds.

    :param size: Number of cells.
    :param flat_shape: Shape the per-cell inputs are broadcast to.
    :param saturation: Saturation the curve is scanned over (flat).
    :param maximum_saturation: Historical maximum of that saturation (flat).
    :param imbibition_flag: Imbibition flag of that phase (flat).
    :param reversal_saturation: Saturation at the last reversal (flat).
    :param scanning_exponent: Killough exponent *n*.
    :param drainage: Drainage value and slope.
    :param imbibition: Imbibition value and slope.
    :param with_derivatives: Whether the slopes are present and wanted.
    :return: Scanned value and slope (`None` when the slope is not wanted).
    """
    drainage_value = np.asarray(drainage[0]).reshape(-1)
    imbibition_value = np.asarray(imbibition[0]).reshape(-1)
    dtype = drainage_value.dtype if drainage_value.dtype.kind == "f" else np.dtype(np.float64)
    placeholder = np.zeros(1, dtype=dtype)
    drainage_slope = np.asarray(drainage[1]).reshape(-1) if with_derivatives else placeholder
    imbibition_slope = np.asarray(imbibition[1]).reshape(-1) if with_derivatives else placeholder

    value = np.empty(size, dtype=dtype)
    slope = np.empty(size if with_derivatives else 1, dtype=dtype)
    scan_curve_kernel(
        size=size,
        saturation=saturation,
        reversal_saturation=reversal_saturation,
        maximum_historical_saturation=maximum_saturation,
        is_imbibition=imbibition_flag,
        scanning_exponent=scanning_exponent,
        numerical_epsilon=1e-12,
        drainage_value=drainage_value,
        imbibition_value=imbibition_value,
        drainage_derivative=drainage_slope,
        imbibition_derivative=imbibition_slope,
        want_derivative=with_derivatives,
        out_value=value,
        out_derivative=slope,
    )
    return value, slope if with_derivatives else None


@attrs.frozen(slots=True)
class KilloughCapillaryPressureTable(CapillaryPressureTable):
    """
    Implements the Killough capillary pressure hysteresis model/table.

    Capillary pressure hysteresis involves no trapping. When the displacement
    direction reverses, capillary pressure traces a *scanning curve* that
    interpolates between the primary drainage and imbibition bounds.

    Both two-phase (`TwoPhaseCapillaryPressureTable`) and full three-phase
    (`CapillaryPressureTable`) backing tables are supported. Wetting and
    non-wetting phase roles are resolved through the canonical API
    (`get_oil_water_wetting_phase` / `get_gas_oil_wetting_phase`) so the
    model is wettability-agnostic.

    The oil-water scanning curve scans over *water* saturation; the gas-oil
    scanning curve scans over *gas* saturation.

    Saturation history is passed as keyword arguments to `evaluate`, `derivatives` and
    `evaluate_with_derivatives`. A sub-system whose history is not supplied stays on its primary
    drainage curve. The imbibition curve of a sub-system is only evaluated when it has history
    and an imbibition table that differs from its drainage table, since a scan between identical
    curves is the curve itself.
    """

    __type__ = "killough_capillary_pressure_model"

    oil_water_drainage_table: TwoPhaseCapillaryPressureTable | CapillaryPressureTable
    """Primary drainage capillary pressure table for the oil-water system."""

    gas_oil_drainage_table: TwoPhaseCapillaryPressureTable | CapillaryPressureTable
    """Primary drainage capillary pressure table for the gas-oil system."""

    oil_water_imbibition_table: TwoPhaseCapillaryPressureTable | CapillaryPressureTable | None = (
        None
    )
    """Primary imbibition Pc table for the oil-water system. Defaults to the drainage table."""

    gas_oil_imbibition_table: TwoPhaseCapillaryPressureTable | CapillaryPressureTable | None = None
    """Primary imbibition Pc table for the gas-oil system. Defaults to the drainage table."""

    scanning_interpolation_exponent: Number = 1.0
    """Killough interpolation exponent *n* (1 = linear)."""

    supports_vector: bool = attrs.field(init=False, repr=False, default=True)

    def __attrs_post_init__(self) -> None:
        for name, table, phases in (
            (
                "oil_water_drainage_table",
                self.oil_water_drainage_table,
                {FluidPhase.WATER, FluidPhase.OIL},
            ),
            (
                "gas_oil_drainage_table",
                self.gas_oil_drainage_table,
                {FluidPhase.OIL, FluidPhase.GAS},
            ),
            (
                "oil_water_imbibition_table",
                self.oil_water_imbibition_table,
                {FluidPhase.WATER, FluidPhase.OIL},
            ),
            (
                "gas_oil_imbibition_table",
                self.gas_oil_imbibition_table,
                {FluidPhase.OIL, FluidPhase.GAS},
            ),
        ):
            if (
                isinstance(table, TwoPhaseCapillaryPressureTable)
                and {
                    table.wetting_phase,
                    table.non_wetting_phase,
                }
                != phases
            ):
                involved = " and ".join(sorted(phase.value for phase in phases))
                raise ValidationError(f"`{name}` must involve {involved} phases.")

    def get_oil_water_wetting_phase(self) -> FluidPhase:
        """
        Return the wetting phase for the oil-water sub-system.

        :return: `FluidPhase.WATER` for water-wet or `FluidPhase.OIL` for
            oil-wet systems, as reported by the drainage table.
        """
        return self.oil_water_drainage_table.get_oil_water_wetting_phase()

    def get_connate_water_saturation(self) -> Number:
        """
        Connate water saturation of the system, read off the oil-water drainage table. It
        places the gas-oil system (`So = 1 - swc - Sg`) when the gas-oil table is a
        two-phase table.

        :return: Connate water saturation.
        """
        table = self.oil_water_drainage_table
        if isinstance(table, TwoPhaseCapillaryPressureTable | ThreePhaseCapillaryPressureTable):
            return table.get_connate_water_saturation()
        return table.irreducible_water_saturation  # type: ignore[attr-defined]

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        """
        Return the wetting phase for the gas-oil sub-system.

        :return: `FluidPhase.OIL` for oil-wet or `FluidPhase.GAS` for
            gas-wet systems, as reported by the drainage table.
        """
        return self.gas_oil_drainage_table.get_gas_oil_wetting_phase()

    def compute(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        *,
        with_derivatives: bool,
        max_water_saturation: NumberOrArray[NDimension] | None = None,
        max_gas_saturation: NumberOrArray[NDimension] | None = None,
        water_imbibition_flag: bool | BooleanArray[NDimension] | None = None,
        gas_imbibition_flag: bool | BooleanArray[NDimension] | None = None,
        water_reversal_saturation: NumberOrArray[NDimension] | None = None,
        gas_reversal_saturation: NumberOrArray[NDimension] | None = None,
        **kwargs: typing.Any,
    ) -> tuple[CapillaryPressures, CapillaryPressureDerivatives | None]:
        """
        Capillary pressures, and optionally their derivatives, in one pass.

        This is the one implementation behind `evaluate`, `derivatives` and
        `evaluate_with_derivatives`.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param with_derivatives: Whether to also compute the derivatives.
        :param max_water_saturation: Historical maximum water saturation per cell.
        :param max_gas_saturation: Historical maximum gas saturation per cell.
        :param water_imbibition_flag: Per-cell flag, true while water saturation increases.
        :param gas_imbibition_flag: Per-cell flag, true while gas saturation increases.
        :param water_reversal_saturation: Water saturation at the last reversal. Defaults to
            `max_water_saturation`.
        :param gas_reversal_saturation: Gas saturation at the last reversal. Defaults to
            `max_gas_saturation`.
        :param kwargs: Extra arguments forwarded to the backing tables.
        :return: The capillary pressures and, when requested, their derivatives (otherwise
            `None`).
        """
        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
        )
        use_water_hysteresis = (
            max_water_saturation is not None
            and water_imbibition_flag is not None
            and self.is_distinct_imbibition(
                self.oil_water_imbibition_table, self.oil_water_drainage_table
            )
        )
        use_gas_hysteresis = (
            max_gas_saturation is not None
            and gas_imbibition_flag is not None
            and self.is_distinct_imbibition(
                self.gas_oil_imbibition_table, self.gas_oil_drainage_table
            )
        )

        inputs = [water_saturation, oil_saturation, gas_saturation]
        if use_water_hysteresis:
            inputs += [max_water_saturation, water_imbibition_flag]
            if water_reversal_saturation is not None:
                inputs.append(water_reversal_saturation)
        if use_gas_hysteresis:
            inputs += [max_gas_saturation, gas_imbibition_flag]
            if gas_reversal_saturation is not None:
                inputs.append(gas_reversal_saturation)
        shape = np.broadcast_shapes(*(np.shape(value) for value in inputs))
        flat_shape = shape or (1,)
        size = math.prod(flat_shape)

        sw = flatten_kernel_input(water_saturation, shape=flat_shape)
        so = flatten_kernel_input(oil_saturation, shape=flat_shape)
        sg = flatten_kernel_input(gas_saturation, shape=flat_shape)

        connate_water_saturation = self.get_connate_water_saturation()
        memo: dict[int, typing.Any] = {}
        arguments = dict(
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            with_derivatives=with_derivatives,
            memo=memo,
        )
        oil_water_drainage = query_oil_water_curve(
            self.oil_water_drainage_table, **arguments, **kwargs
        )
        gas_oil_drainage = query_gas_oil_curve(
            self.gas_oil_drainage_table,
            connate_water_saturation=connate_water_saturation,
            **arguments,
            **kwargs,
        )

        oil_water, oil_water_slope = oil_water_drainage
        if use_water_hysteresis:
            oil_water_imbibition = query_oil_water_curve(
                typing.cast(AnyCapillaryTable, self.oil_water_imbibition_table),
                **arguments,
                **kwargs,
            )
            maximum = flatten_kernel_input(max_water_saturation, shape=flat_shape)
            oil_water, oil_water_slope = scan_curve(
                size=size,
                flat_shape=flat_shape,
                saturation=sw,
                maximum_saturation=maximum,
                imbibition_flag=flatten_kernel_input(
                    water_imbibition_flag, shape=flat_shape, as_float=False
                ),
                reversal_saturation=maximum
                if water_reversal_saturation is None
                else flatten_kernel_input(water_reversal_saturation, shape=flat_shape),
                scanning_exponent=self.scanning_interpolation_exponent,
                drainage=oil_water_drainage,
                imbibition=oil_water_imbibition,
                with_derivatives=with_derivatives,
            )

        gas_oil, gas_oil_slope = gas_oil_drainage
        if use_gas_hysteresis:
            gas_oil_imbibition = query_gas_oil_curve(
                typing.cast(AnyCapillaryTable, self.gas_oil_imbibition_table),
                connate_water_saturation=connate_water_saturation,
                **arguments,
                **kwargs,
            )
            maximum = flatten_kernel_input(max_gas_saturation, shape=flat_shape)
            gas_oil, gas_oil_slope = scan_curve(
                size=size,
                flat_shape=flat_shape,
                saturation=sg,
                maximum_saturation=maximum,
                imbibition_flag=flatten_kernel_input(
                    gas_imbibition_flag, shape=flat_shape, as_float=False
                ),
                reversal_saturation=maximum
                if gas_reversal_saturation is None
                else flatten_kernel_input(gas_reversal_saturation, shape=flat_shape),
                scanning_exponent=self.scanning_interpolation_exponent,
                drainage=gas_oil_drainage,
                imbibition=gas_oil_imbibition,
                with_derivatives=with_derivatives,
            )

        def finish(array: typing.Any) -> typing.Any:
            result = np.asarray(array)
            if result.size == 1 and size > 1:
                result = np.broadcast_to(result, (size,))
            result = result.reshape(shape)
            return result.item() if is_scalar else result

        pressures = CapillaryPressures(oil_water=finish(oil_water), gas_oil=finish(gas_oil))
        if not with_derivatives:
            return pressures, None
        zero = finish(np.zeros(size, dtype=np.asarray(oil_water).dtype))
        return pressures, CapillaryPressureDerivatives(
            dpcow_dsw=finish(oil_water_slope),
            dpcow_dso=zero,
            dpcgo_dsg=finish(gas_oil_slope),
            dpcgo_dso=zero,
        )

    @staticmethod
    def is_distinct_imbibition(
        imbibition_table: AnyCapillaryTable | None, drainage_table: AnyCapillaryTable
    ) -> bool:
        """Whether the imbibition curve can differ from the drainage curve."""
        return imbibition_table is not None and imbibition_table is not drainage_table

    def evaluate(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute three-phase capillary pressures with Killough hysteresis.

        Without saturation history the result is the primary drainage capillary pressure.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param kwargs: History arguments (`max_water_saturation`, `max_gas_saturation`,
            `water_imbibition_flag`, `gas_imbibition_flag`, `water_reversal_saturation`,
            `gas_reversal_saturation`) and any other arguments the backing tables accept.
        :return: `CapillaryPressures` dictionary with keys `"oil_water"` and `"gas_oil"`.
        """
        pressures, _ = self.compute(
            water_saturation, oil_saturation, gas_saturation, with_derivatives=False, **kwargs
        )
        return pressures

    def derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> CapillaryPressureDerivatives:
        """
        Compute partial derivatives of capillary pressures with Killough hysteresis.

        Returns the following non-zero entries.

        - `dpcow_dsw`: dpcow/dsw - oil-water Pc scanned over Sw.
        - `dpcow_dso`: zero - Pcow does not depend directly on So in this model.
        - `dpcgo_dsg`: dpcgo/dsg - gas-oil Pc scanned over Sg.
        - `dpcgo_dso`: zero - Pcgo does not depend directly on So.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param kwargs: Same history and backing table arguments as `evaluate`.
        :return: `CapillaryPressureDerivatives` dictionary containing
            `dpcow_dsw`, `dpcow_dso`, `dpcgo_dsg`, and `dpcgo_dso`.
        """
        _, derivatives = self.compute(
            water_saturation, oil_saturation, gas_saturation, with_derivatives=True, **kwargs
        )
        assert derivatives is not None
        return derivatives

    def evaluate_with_derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> tuple[CapillaryPressures, CapillaryPressureDerivatives]:
        """
        Compute the capillary pressures and their derivatives in one pass.

        Every bounding curve is evaluated once, where calling `evaluate` and `derivatives`
        separately would evaluate each twice.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param kwargs: Same history and backing table arguments as `evaluate`.
        :return: Tuple of the `CapillaryPressures` and `CapillaryPressureDerivatives`.
        """
        pressures, derivatives = self.compute(
            water_saturation, oil_saturation, gas_saturation, with_derivatives=True, **kwargs
        )
        assert derivatives is not None
        return pressures, derivatives
