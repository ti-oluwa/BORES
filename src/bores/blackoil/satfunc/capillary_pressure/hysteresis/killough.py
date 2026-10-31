"""
Killough (1976) scanning curves for capillary pressure.

- Killough, J.E. (1976). "Reservoir Simulation With History-Dependent Saturation Functions".
  SPE 5106.
"""

import math
import typing

import attrs
import numpy as np

from bores.blackoil.satfunc.capillary_pressure.hysteresis.base import (
    AnyCapillaryTable,
    HysteresisCapillaryPressureTable,
    query_gas_oil_curve,
    query_oil_water_curve,
)
from bores.blackoil.satfunc.utils import flatten_kernel_input, scan_curve_kernel
from bores.types import (
    Boolean,
    BooleanArray,
    CapillaryPressureDerivatives,
    CapillaryPressures,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
)

__all__ = ["KilloughCapillaryPressureTable", "scan_curve"]


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
    with_derivatives: Boolean,
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
        with_derivatives=with_derivatives,
        out_value=value,
        out_derivative=slope,
    )
    return value, slope if with_derivatives else None


@attrs.frozen(slots=True)
class KilloughCapillaryPressureTable(HysteresisCapillaryPressureTable):
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

    scanning_interpolation_exponent: Number = 1.0
    """Killough interpolation exponent *n* (1 = linear)."""

    def compute(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        *,
        with_derivatives: Boolean,
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

        sw = flatten_kernel_input(value=water_saturation, shape=flat_shape)
        so = flatten_kernel_input(value=oil_saturation, shape=flat_shape)
        sg = flatten_kernel_input(value=gas_saturation, shape=flat_shape)

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
            maximum = flatten_kernel_input(value=max_water_saturation, shape=flat_shape)
            oil_water, oil_water_slope = scan_curve(
                size=size,
                flat_shape=flat_shape,
                saturation=sw,
                maximum_saturation=maximum,
                imbibition_flag=flatten_kernel_input(
                    value=water_imbibition_flag, shape=flat_shape, as_float=False
                ),
                reversal_saturation=maximum
                if water_reversal_saturation is None
                else flatten_kernel_input(value=water_reversal_saturation, shape=flat_shape),
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
            maximum = flatten_kernel_input(value=max_gas_saturation, shape=flat_shape)
            gas_oil, gas_oil_slope = scan_curve(
                size=size,
                flat_shape=flat_shape,
                saturation=sg,
                maximum_saturation=maximum,
                imbibition_flag=flatten_kernel_input(
                    value=gas_imbibition_flag, shape=flat_shape, as_float=False
                ),
                reversal_saturation=maximum
                if gas_reversal_saturation is None
                else flatten_kernel_input(value=gas_reversal_saturation, shape=flat_shape),
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

        pressures = CapillaryPressures(
            oil_water=finish(array=oil_water), gas_oil=finish(array=gas_oil)
        )
        if not with_derivatives:
            return pressures, None
        zero = finish(array=np.zeros(size, dtype=np.asarray(oil_water).dtype))
        return pressures, CapillaryPressureDerivatives(
            dpcow_dsw=finish(array=oil_water_slope),
            dpcow_dso=zero,
            dpcgo_dsg=finish(array=gas_oil_slope),
            dpcgo_dso=zero,
        )
