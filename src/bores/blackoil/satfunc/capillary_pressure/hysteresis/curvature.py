"""
Killough (1976) capillary pressure scanning curves with a curvature parameter.

The capillary pressure on a scanning curve is a weighted average of the drainage and imbibition
curves, `pc = pc_drn + F * (pc_imb - pc_drn)`, with `F` rising from 0 on the drainage curve to 1 on
the imbibition curve. The weight is not a power of the saturation's progress through the scanning
range, as the Killough model of this library has it, but depends on a small curvature `ε`:

    imbibition scan:  F = [1 / (s - s_low + ε) - 1 / ε] / [1 / (s_high - s_low + ε) - 1 / ε]
    drainage scan:    F = 1 - [1 / (s_high - s + ε) - 1 / ε] / [1 / (s_high - s_low + ε) - 1 / ε]

where `s_low` is the lowest wetting phase saturation reached, `s_high` the highest, and the curve
is on the drainage curve at `s_low` and on the imbibition curve at `s_high`. The closer `ε` is to
zero, the sooner the scanning curve moves over to the other bound. This is `EHYSTR` item 1.

- Killough, J.E. (1976). "Reservoir Simulation With History-Dependent Saturation Functions".
  SPE 5106.
"""

import math
import typing

import attrs
import numba
import numpy as np

from bores.blackoil.satfunc.capillary_pressure.hysteresis.base import (
    AnyCapillaryTable,
    HysteresisCapillaryPressureTable,
    query_gas_oil_curve,
    query_oil_water_curve,
)
from bores.blackoil.satfunc.utils import element_at, flatten_kernel_input
from bores.constants import c
from bores.errors import ValidationError
from bores.types import (
    Boolean,
    BooleanArray,
    CapillaryPressureDerivatives,
    CapillaryPressures,
    FluidPhase,
    Integer,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
)

__all__ = [
    "CurvatureCapillaryPressureTable",
    "compute_curvature_scanning_weight",
    "scan_curve_curvature",
]


@numba.njit(cache=True, error_model="numpy")
def compute_curvature_scanning_weight(
    saturation: Number,
    lowest_saturation: Number,
    highest_saturation: Number,
    is_imbibition: Number,
    curvature: Number,
    numerical_epsilon: Number,
) -> tuple[Number, Number]:
    """
    Weight `F` of the imbibition curve on a scanning curve, and its derivative with respect to the
    saturation.

    With `x` the distance travelled from the end the curve left, and `X` the length of the
    scanning range, both forms reduce to `x * (X + ε) / (X * (x + ε))`, which is 0 at the start,
    1 at the far end, and bends faster the smaller `ε` is. An imbibition scan leaves the drainage
    curve at the lowest saturation and travels up, so `F` is that value. A drainage scan leaves the
    imbibition curve at the highest saturation and travels down, so the weight of the imbibition
    curve is one minus it. A saturation outside the range is on a bounding curve, where the
    derivative is zero. A collapsed range has no scanning curve, so the drainage curve is used.

    :param saturation: The wetting phase saturation.
    :param lowest_saturation: Lowest saturation reached, where an imbibition scan starts and a
        drainage scan ends.
    :param highest_saturation: Highest saturation reached, where a drainage scan starts and an
        imbibition scan ends.
    :param is_imbibition: Non-zero while the wetting phase is imbibing, so rising.
    :param curvature: The curvature `ε`, not negative.
    :param numerical_epsilon: Smallest range that is scanned.
    :returns: The weight of the imbibition curve, and its derivative.
    """
    span = highest_saturation - lowest_saturation
    if span <= numerical_epsilon:
        return 0.0, 0.0
    if is_imbibition != 0:
        travelled = saturation - lowest_saturation
    else:
        travelled = highest_saturation - saturation
    if travelled <= 0.0:
        fraction = 0.0
        fraction_slope = 0.0
    elif travelled >= span:
        fraction = 1.0
        fraction_slope = 0.0
    else:
        fraction = travelled * (span + curvature) / (span * (travelled + curvature))
        fraction_slope = (span + curvature) * curvature / (span * (travelled + curvature) ** 2)
    if is_imbibition != 0:
        return fraction, fraction_slope
    return 1.0 - fraction, fraction_slope


@numba.njit(cache=True, error_model="numpy")
def scan_curve_curvature_kernel(
    size: Integer,
    saturation: NumberArray[OneDimension],
    reversal_saturation: NumberArray[OneDimension],
    maximum_saturation: NumberArray[OneDimension],
    minimum_saturation: NumberArray[OneDimension],
    is_imbibition: NumberArray[OneDimension],
    curvature: Number,
    numerical_epsilon: Number,
    drainage_value: NumberArray[OneDimension],
    imbibition_value: NumberArray[OneDimension],
    drainage_derivative: NumberArray[OneDimension],
    imbibition_derivative: NumberArray[OneDimension],
    with_derivatives: Boolean,
    out_value: NumberArray[OneDimension],
    out_derivative: NumberArray[OneDimension],
) -> None:
    """
    Scan one capillary pressure curve between its bounds for every cell, writing the value and
    optionally the derivative in a single pass.

    An imbibing cell scans up from `reversal_saturation`, where it left the drainage curve, to
    `maximum_saturation`. A draining cell scans down from `reversal_saturation`, where it left the
    imbibition curve, to `minimum_saturation`. Every array argument is flat and holds either one
    value per cell or one value shared by all cells.
    """
    for i in range(size):
        imbibing = element_at(array=is_imbibition, index=i)
        reversal = element_at(array=reversal_saturation, index=i)
        if imbibing != 0:
            lowest = reversal
            highest = element_at(array=maximum_saturation, index=i)
        else:
            lowest = element_at(array=minimum_saturation, index=i)
            highest = reversal
        weight, weight_derivative = compute_curvature_scanning_weight(
            saturation=element_at(array=saturation, index=i),
            lowest_saturation=lowest,
            highest_saturation=highest,
            is_imbibition=imbibing,
            curvature=curvature,
            numerical_epsilon=numerical_epsilon,
        )
        drainage = element_at(array=drainage_value, index=i)
        imbibition = element_at(array=imbibition_value, index=i)
        out_value[i] = drainage + (imbibition - drainage) * weight
        if with_derivatives:
            drainage_slope = element_at(array=drainage_derivative, index=i)
            imbibition_slope = element_at(array=imbibition_derivative, index=i)
            out_derivative[i] = (
                drainage_slope
                + (imbibition_slope - drainage_slope) * weight
                + (imbibition - drainage) * weight_derivative
            )


def scan_curve_curvature(
    *,
    size: int,
    saturation: NumberArray[OneDimension],
    maximum_saturation: NumberArray[OneDimension],
    minimum_saturation: NumberArray[OneDimension],
    imbibition_flag: NumberArray[OneDimension],
    reversal_saturation: NumberArray[OneDimension],
    curvature: Number,
    drainage: tuple[typing.Any, typing.Any],
    imbibition: tuple[typing.Any, typing.Any],
    with_derivatives: Boolean,
) -> tuple[typing.Any, typing.Any]:
    """
    Scan one capillary pressure curve between its drainage and imbibition bounds, with the curvature
    weight.

    :param size: Number of cells.
    :param saturation: Wetting phase saturation the curve is scanned over (flat).
    :param maximum_saturation: Historical maximum of that saturation (flat).
    :param minimum_saturation: Historical minimum of that saturation (flat).
    :param imbibition_flag: Imbibition flag of that phase (flat).
    :param reversal_saturation: Saturation at the last reversal (flat).
    :param curvature: The curvature `ε`.
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
    scan_curve_curvature_kernel(
        size=size,
        saturation=saturation,
        reversal_saturation=reversal_saturation,
        maximum_saturation=maximum_saturation,
        minimum_saturation=minimum_saturation,
        is_imbibition=imbibition_flag,
        curvature=curvature,
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
class CurvatureCapillaryPressureTable(HysteresisCapillaryPressureTable):
    """
    Capillary pressure hysteresis with Killough's scanning curves and a curvature parameter (see the
    module notes), as Eclipse's `EHYSTR` item 1 sets it.

    The oil-water capillary pressure scans over water saturation, between the lowest water
    saturation reached, the highest, and the saturation of the last reversal, which are all in the
    saturation history. The gas-oil capillary pressure follows its drainage curve: its scanning
    needs the history of the lowest gas saturation, which is not tracked, so a gas-oil imbibition
    table is rejected rather than ignored. Water has to be the wetting phase of the oil-water
    system for an oil-water imbibition table to scan, since the scanning range is then in the
    wetting phase's saturation.

    Saturation history is passed as keyword arguments to `evaluate`, `derivatives` and
    `evaluate_with_derivatives`. A system whose history is not supplied stays on its drainage curve.
    """

    __type__ = "curvature_capillary_pressure_model"

    scanning_curvature: Number = attrs.field(
        factory=lambda: typing.cast(float, c.HYSTERESIS_CAPILLARY_PRESSURE_CURVATURE)
    )
    """
    Curvature `ε` of the scanning curves. The closer it is to zero, the sooner a scanning curve
    moves over to the other bound. Not negative.
    """

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.scanning_curvature < 0.0:
            raise ValidationError("`scanning_curvature` must not be negative.")
        if self.gas_oil_imbibition_table is not None:
            raise ValidationError(
                "The curvature strategy does not scan gas-oil capillary pressure yet, because the "
                "lowest gas saturation is not tracked. Give no `gas_oil_imbibition_table`."
            )
        if (
            self.oil_water_imbibition_table is not None
            and self.get_oil_water_wetting_phase() != FluidPhase.WATER
        ):
            raise ValidationError(
                "The curvature strategy scans oil-water capillary pressure over water saturation, "
                "so water has to be the wetting phase."
            )

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
        min_water_saturation: NumberOrArray[NDimension] | None = None,
        **kwargs: typing.Any,
    ) -> tuple[CapillaryPressures, CapillaryPressureDerivatives | None]:
        """
        Capillary pressures, and optionally their derivatives, in one pass.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param with_derivatives: Whether to also compute the derivatives.
        :param max_water_saturation: Historical maximum water saturation per cell.
        :param max_gas_saturation: Historical maximum gas saturation per cell. Not used.
        :param water_imbibition_flag: Per-cell flag, true while water is imbibing, so rising.
        :param gas_imbibition_flag: Per-cell gas imbibition flag. Not used.
        :param water_reversal_saturation: Water saturation at the last reversal. Defaults to
            `max_water_saturation`.
        :param gas_reversal_saturation: Gas saturation at the last reversal. Not used.
        :param min_water_saturation: Historical minimum water saturation per cell, where a
            drainage scan ends.
        :param kwargs: Extra arguments forwarded to the backing tables.
        :return: The capillary pressures and, when requested, their derivatives (otherwise `None`).
        """
        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
        )
        use_water_hysteresis = (
            max_water_saturation is not None
            and water_imbibition_flag is not None
            and min_water_saturation is not None
            and self.is_distinct_imbibition(
                self.oil_water_imbibition_table, self.oil_water_drainage_table
            )
        )

        inputs = [water_saturation, oil_saturation, gas_saturation]
        if use_water_hysteresis:
            inputs += [max_water_saturation, water_imbibition_flag, min_water_saturation]
            if water_reversal_saturation is not None:
                inputs.append(water_reversal_saturation)

        shape = np.broadcast_shapes(*(np.shape(value) for value in inputs))
        flat_shape = shape or (1,)
        size = math.prod(flat_shape)

        sw = flatten_kernel_input(value=water_saturation, shape=flat_shape)
        so = flatten_kernel_input(value=oil_saturation, shape=flat_shape)
        sg = flatten_kernel_input(value=gas_saturation, shape=flat_shape)

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
        gas_oil, gas_oil_slope = query_gas_oil_curve(
            self.gas_oil_drainage_table,
            connate_water_saturation=self.get_connate_water_saturation(),
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
            oil_water, oil_water_slope = scan_curve_curvature(
                size=size,
                saturation=sw,
                maximum_saturation=maximum,
                minimum_saturation=flatten_kernel_input(
                    value=min_water_saturation, shape=flat_shape
                ),
                imbibition_flag=flatten_kernel_input(
                    value=water_imbibition_flag, shape=flat_shape, as_float=False
                ),
                reversal_saturation=maximum
                if water_reversal_saturation is None
                else flatten_kernel_input(value=water_reversal_saturation, shape=flat_shape),
                curvature=self.scanning_curvature,
                drainage=oil_water_drainage,
                imbibition=oil_water_imbibition,
                with_derivatives=with_derivatives,
            )

        def finish(array: typing.Any) -> typing.Any:
            result = np.asarray(array)
            if result.size == 1 and size > 1:
                result = np.broadcast_to(result, (size,))
            result = result.reshape(shape)
            return result.item() if is_scalar else result

        zero_pressure = finish(array=np.zeros(size, dtype=np.asarray(oil_water).dtype))
        pressures = CapillaryPressures(
            oil_water=finish(array=oil_water),
            gas_oil=finish(array=gas_oil),
            gas_water=zero_pressure,
        )
        if not with_derivatives:
            return pressures, None
        zero = zero_pressure
        return pressures, CapillaryPressureDerivatives(
            dpcow_dsw=finish(array=oil_water_slope),
            dpcow_dso=zero,
            dpcgo_dsg=finish(array=gas_oil_slope),
            dpcgo_dso=zero,
            dpcgw_dsg=zero,
        )
