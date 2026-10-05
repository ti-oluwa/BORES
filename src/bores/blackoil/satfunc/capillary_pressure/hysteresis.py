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

import typing

import attrs
import numpy as np
import numpy.typing as npt

from bores.blackoil.satfunc.capillary_pressure.tables import (
    CapillaryPressureTable,
    TwoPhaseCapillaryPressureTable,
)
from bores.blackoil.satfunc.utils import (
    compute_killough_scanning_curve,
    compute_killough_scanning_curve_derivative,
)
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
)

__all__ = ["KilloughCapillaryPressureTable"]


def get_oil_water_capillary_pressure(
    oil_water_capillary_pressure_table: TwoPhaseCapillaryPressureTable | CapillaryPressureTable,
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    **kwargs: typing.Any,
) -> NumberOrArray[NDimension]:
    """
    Extract Pcow from an oil-water capillary pressure table, dispatching
    saturations correctly via the canonical wetting-phase API.

    :param oil_water_capillary_pressure_table: Oil-water capillary pressure table (two-phase or three-phase).
    :param water_saturation: Water saturation (scalar or array).
    :param oil_saturation: Oil saturation (scalar or array).
    :param gas_saturation: Gas saturation (scalar or array).
    :param kwargs: Additional keyword arguments forwarded to parametric tables.
    :return: Oil-water capillary pressure Pcow = Po - Pw (scalar or array).
    """
    oil_water_wetting_phase = oil_water_capillary_pressure_table.get_oil_water_wetting_phase()
    if isinstance(oil_water_capillary_pressure_table, TwoPhaseCapillaryPressureTable):
        if oil_water_wetting_phase == FluidPhase.WATER:
            wetting_saturation = water_saturation
            non_wetting_saturation = oil_saturation
        else:
            wetting_saturation = oil_saturation
            non_wetting_saturation = water_saturation
        return oil_water_capillary_pressure_table.get_capillary_pressure(
            wetting_saturation=wetting_saturation,
            non_wetting_saturation=non_wetting_saturation,
        )

    result = oil_water_capillary_pressure_table.evaluate(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    return result["oil_water"]


def get_gas_oil_capillary_pressure(
    gas_oil_capillary_pressure_table: TwoPhaseCapillaryPressureTable | CapillaryPressureTable,
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    **kwargs: typing.Any,
) -> NumberOrArray[NDimension]:
    """
    Extract Pcgo from a gas-oil capillary pressure table, dispatching
    saturations correctly via the canonical wetting-phase API.

    :param gas_oil_capillary_pressure_table: Gas-oil capillary pressure table (two-phase or three-phase).
    :param water_saturation: Water saturation (scalar or array).
    :param oil_saturation: Oil saturation (scalar or array).
    :param gas_saturation: Gas saturation (scalar or array).
    :param kwargs: Additional keyword arguments forwarded to parametric tables.
    :return: Gas-oil capillary pressure Pcgo = Pg - Po (scalar or array).
    """
    gas_oil_wetting_phase = gas_oil_capillary_pressure_table.get_gas_oil_wetting_phase()
    if isinstance(gas_oil_capillary_pressure_table, TwoPhaseCapillaryPressureTable):
        if gas_oil_wetting_phase == FluidPhase.OIL:
            wetting_saturation = oil_saturation
            non_wetting_saturation = gas_saturation
        else:
            wetting_saturation = gas_saturation
            non_wetting_saturation = oil_saturation
        return gas_oil_capillary_pressure_table.get_capillary_pressure(
            wetting_saturation=wetting_saturation,
            non_wetting_saturation=non_wetting_saturation,
        )

    result = gas_oil_capillary_pressure_table.evaluate(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    return result["gas_oil"]


def get_oil_water_capillary_pressure_derivative(
    oil_water_capillary_pressure_table: TwoPhaseCapillaryPressureTable | CapillaryPressureTable,
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    **kwargs: typing.Any,
) -> NumberOrArray[NDimension]:
    """
    Return dpcow/d(reference_sat) for the oil-water capillary pressure table.

    For `TwoPhaseCapillaryPressureTable` this is the derivative w.r.t. the
    table's own reference saturation (Sw in water-wet, So in oil-wet). For
    three-phase tables `dpcow/dsw` is returned to match the scanning variable
    used by the hysteresis layer.

    :param oil_water_capillary_pressure_table: Oil-water capillary pressure table (two-phase or three-phase).
    :param water_saturation: Water saturation (scalar or array).
    :param oil_saturation: Oil saturation (scalar or array).
    :param gas_saturation: Gas saturation (scalar or array).
    :param kwargs: Additional keyword arguments forwarded to parametric tables.
    :return: Derivative of Pcow w.r.t. the reference saturation axis (scalar or array).
    """
    oil_water_wetting_phase = oil_water_capillary_pressure_table.get_oil_water_wetting_phase()

    if isinstance(oil_water_capillary_pressure_table, TwoPhaseCapillaryPressureTable):
        if oil_water_wetting_phase == FluidPhase.WATER:
            wetting_saturation = water_saturation
            non_wetting_saturation = oil_saturation
        else:
            wetting_saturation = oil_saturation
            non_wetting_saturation = water_saturation
        return oil_water_capillary_pressure_table.get_capillary_pressure_derivative(
            wetting_saturation=wetting_saturation,
            non_wetting_saturation=non_wetting_saturation,
        )

    derivatives = oil_water_capillary_pressure_table.derivatives(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    return derivatives["dpcow_dsw"]


def get_gas_oil_capillary_pressure_derivative(
    gas_oil_capillary_pressure_table: TwoPhaseCapillaryPressureTable | CapillaryPressureTable,
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    **kwargs: typing.Any,
) -> NumberOrArray[NDimension]:
    """
    Return dpcgo/d(reference_sat) for the gas-oil capillary pressure table.

    For `TwoPhaseCapillaryPressureTable` this is the derivative w.r.t. the
    table's own reference saturation. For three-phase tables
    `dpcgo/dsg` is returned to match the scanning variable used by the
    hysteresis layer.

    :param gas_oil_capillary_pressure_table: Gas-oil capillary pressure table (two-phase or three-phase).
    :param water_saturation: Water saturation (scalar or array).
    :param oil_saturation: Oil saturation (scalar or array).
    :param gas_saturation: Gas saturation (scalar or array).
    :param kwargs: Additional keyword arguments forwarded to parametric tables.
    :return: Derivative of Pcgo w.r.t. the reference saturation axis
        (scalar or array).
    """
    gas_oil_wetting_phase = gas_oil_capillary_pressure_table.get_gas_oil_wetting_phase()

    if isinstance(gas_oil_capillary_pressure_table, TwoPhaseCapillaryPressureTable):
        if gas_oil_wetting_phase == FluidPhase.OIL:
            wetting_saturation = oil_saturation
            non_wetting_saturation = gas_saturation
        else:
            wetting_saturation = gas_saturation
            non_wetting_saturation = oil_saturation
        return gas_oil_capillary_pressure_table.get_capillary_pressure_derivative(
            wetting_saturation=wetting_saturation,
            non_wetting_saturation=non_wetting_saturation,
        )

    derivatives = gas_oil_capillary_pressure_table.derivatives(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    return derivatives["dpcgo_dsg"]


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

    Saturation history is passed as additional keyword arguments to
    `evaluate` and `derivatives`.
    When these arguments are absent the model returns primary drainage Pc
    values.
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
        if isinstance(self.oil_water_drainage_table, TwoPhaseCapillaryPressureTable) and {
            self.oil_water_drainage_table.wetting_phase,
            self.oil_water_drainage_table.non_wetting_phase,
        } != {FluidPhase.WATER, FluidPhase.OIL}:
            raise ValidationError("`oil_water_drainage_table` must involve water and oil phases.")

        if isinstance(self.gas_oil_drainage_table, TwoPhaseCapillaryPressureTable) and {
            self.gas_oil_drainage_table.wetting_phase,
            self.gas_oil_drainage_table.non_wetting_phase,
        } != {FluidPhase.OIL, FluidPhase.GAS}:
            raise ValidationError("`gas_oil_drainage_table` must involve oil and gas phases.")

        if (
            self.oil_water_imbibition_table is not None
            and isinstance(self.oil_water_imbibition_table, TwoPhaseCapillaryPressureTable)
            and {
                self.oil_water_imbibition_table.wetting_phase,
                self.oil_water_imbibition_table.non_wetting_phase,
            }
            != {FluidPhase.WATER, FluidPhase.OIL}
        ):
            raise ValidationError(
                "`oil_water_imbibition_table` must involve water and oil phases."
            )

        if (
            self.gas_oil_imbibition_table is not None
            and isinstance(self.gas_oil_imbibition_table, TwoPhaseCapillaryPressureTable)
            and {
                self.gas_oil_imbibition_table.wetting_phase,
                self.gas_oil_imbibition_table.non_wetting_phase,
            }
            != {FluidPhase.OIL, FluidPhase.GAS}
        ):
            raise ValidationError("`gas_oil_imbibition_table` must involve oil and gas phases.")

    def get_oil_water_wetting_phase(self) -> FluidPhase:
        """
        Return the wetting phase for the oil-water sub-system.

        :return: `FluidPhase.WATER` for water-wet or `FluidPhase.OIL` for
            oil-wet systems, as reported by the drainage table.
        """
        return self.oil_water_drainage_table.get_oil_water_wetting_phase()

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        """
        Return the wetting phase for the gas-oil sub-system.

        :return: `FluidPhase.OIL` for oil-wet or `FluidPhase.GAS` for
            gas-wet systems, as reported by the drainage table.
        """
        return self.gas_oil_drainage_table.get_gas_oil_wetting_phase()

    def _parse_hysteresis_kwargs(
        self,
        water_saturation: npt.NDArray,
        gas_saturation: npt.NDArray,
        max_water_saturation: NumberOrArray[NDimension] | None,
        max_gas_saturation: NumberOrArray[NDimension] | None,
        water_imbibition_flag: bool | BooleanArray[NDimension] | None,
        gas_imbibition_flag: bool | BooleanArray[NDimension] | None,
        water_reversal_saturation: NumberOrArray[NDimension] | None,
        gas_reversal_saturation: NumberOrArray[NDimension] | None,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Parse and broadcast saturation-history arrays.

        The oil-water and gas-oil sub-systems are resolved **independently**:
        each falls back to its own primary-drainage state (no scanning curve)
        if its own history is incomplete, rather than requiring both
        sub-systems' history to be present before either can engage
        hysteresis. Supplying `max_water_saturation`/`water_imbibition_flag`
        alone is enough to activate oil-water hysteresis even if the gas-oil
        history is entirely absent, and vice versa.

        :param water_saturation: Broadcast-ready water saturation array.
        :param gas_saturation: Broadcast-ready gas saturation array.
        :param max_water_saturation: Historical maximum water saturation or `None`.
        :param max_gas_saturation: Historical maximum gas saturation or `None`.
        :param water_imbibition_flag: Per-cell oil-water imbibition flag or `None`.
        :param gas_imbibition_flag: Per-cell gas-oil imbibition flag or `None`.
        :param water_reversal_saturation: Water saturation at last oil-water reversal,
            or `None` (defaults to `max_water_saturation`).
        :param gas_reversal_saturation: Gas saturation at last gas-oil reversal, or
            `None` (defaults to `max_gas_saturation`).
        :return: Six broadcast-compatible arrays: `(sw_max, sg_max, sw_imb, sg_imb,
            sw_rev, sg_rev)`.
        """
        use_water_hysteresis = (
            max_water_saturation is not None and water_imbibition_flag is not None
        )
        use_gas_hysteresis = max_gas_saturation is not None and gas_imbibition_flag is not None

        if use_water_hysteresis:
            maximum_water_saturation = np.atleast_1d(max_water_saturation)  # type: ignore
            water_imbibition_flag = np.atleast_1d(water_imbibition_flag)  # type: ignore
            water_reversal_saturation = typing.cast(
                NumberArray[NDimension],
                np.atleast_1d(water_reversal_saturation)
                if water_reversal_saturation is not None
                else maximum_water_saturation.copy(),
            )
        else:
            maximum_water_saturation = water_saturation.copy()
            water_imbibition_flag = typing.cast(
                BooleanArray[NDimension],
                np.zeros_like(water_saturation, dtype=np.bool_),
            )
            water_reversal_saturation = typing.cast(
                NumberArray[NDimension], water_saturation.copy()
            )

        if use_gas_hysteresis:
            maximum_gas_saturation = np.atleast_1d(max_gas_saturation)  # type: ignore
            gas_imbibition_flag = np.atleast_1d(gas_imbibition_flag)  # type: ignore
            gas_reversal_saturation = typing.cast(
                NumberArray[NDimension],
                np.atleast_1d(gas_reversal_saturation)
                if gas_reversal_saturation is not None
                else maximum_gas_saturation.copy(),
            )
        else:
            maximum_gas_saturation = gas_saturation.copy()
            gas_imbibition_flag = typing.cast(
                BooleanArray[NDimension], np.zeros_like(gas_saturation, dtype=np.bool_)
            )
            gas_reversal_saturation = typing.cast(NumberArray[NDimension], gas_saturation.copy())

        (
            maximum_water_saturation,
            maximum_gas_saturation,
            water_imbibition_flag,
            gas_imbibition_flag,
            water_reversal_saturation,
            gas_reversal_saturation,
        ) = np.broadcast_arrays(
            maximum_water_saturation,
            maximum_gas_saturation,
            water_imbibition_flag,  # type: ignore
            gas_imbibition_flag,  # type: ignore
            water_reversal_saturation,
            gas_reversal_saturation,
        )
        return (  # type: ignore
            maximum_water_saturation,
            maximum_gas_saturation,
            water_imbibition_flag,
            gas_imbibition_flag,
            water_reversal_saturation,
            gas_reversal_saturation,
        )

    def evaluate(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        max_water_saturation: NumberOrArray[NDimension] | None = None,
        max_gas_saturation: NumberOrArray[NDimension] | None = None,
        water_imbibition_flag: bool | BooleanArray[NDimension] | None = None,
        gas_imbibition_flag: bool | BooleanArray[NDimension] | None = None,
        water_reversal_saturation: NumberOrArray[NDimension] | None = None,
        gas_reversal_saturation: NumberOrArray[NDimension] | None = None,
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute three-phase capillary pressures with Killough hysteresis.

        When the saturation-history keyword arguments are absent the method
        returns primary drainage Pc values.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param max_water_saturation: Historical maximum water saturation per cell.
        :param max_gas_saturation: Historical maximum gas saturation per cell.
        :param water_imbibition_flag: Per-cell oil-water imbibition flag (1 / 0),
            or `None`.
        :param gas_imbibition_flag: Per-cell gas-oil imbibition flag (1 / 0),
            or `None`.
        :param water_reversal_saturation: Water saturation at the last oil-water
            reversal, or `None` (defaults to `max_water_saturation`).
        :param gas_reversal_saturation: Gas saturation at the last gas-oil reversal,
            or `None` (defaults to `max_gas_saturation`).
        :param kwargs: Additional keyword arguments forwarded to the backing tables.
        :return: `CapillaryPressures` dictionary with keys `"oil_water"` and
            `"gas_oil"`.
        """
        is_scalar = np.isscalar(water_saturation)
        water_saturation = np.atleast_1d(water_saturation)  # type: ignore
        oil_saturation = np.atleast_1d(oil_saturation)  # type: ignore
        gas_saturation = np.atleast_1d(gas_saturation)  # type: ignore
        water_saturation, oil_saturation, gas_saturation = np.broadcast_arrays(  # type: ignore
            water_saturation, oil_saturation, gas_saturation
        )

        (
            maximum_water_saturation,
            maximum_gas_saturation,
            water_imbibition_flag,
            gas_imbibition_flag,
            water_reversal_saturation,
            gas_reversal_saturation,
        ) = self._parse_hysteresis_kwargs(  # type: ignore
            water_saturation=water_saturation,  # type: ignore
            gas_saturation=gas_saturation,  # type: ignore
            max_water_saturation=max_water_saturation,
            max_gas_saturation=max_gas_saturation,
            water_imbibition_flag=water_imbibition_flag,
            gas_imbibition_flag=gas_imbibition_flag,
            water_reversal_saturation=water_reversal_saturation,
            gas_reversal_saturation=gas_reversal_saturation,
        )

        oil_water_drainage_table = self.oil_water_drainage_table
        oil_water_imbibition_table = self.oil_water_imbibition_table or oil_water_drainage_table
        gas_oil_drainage_table = self.gas_oil_drainage_table
        gas_oil_imbibition_table = self.gas_oil_imbibition_table or gas_oil_drainage_table

        # Oil-water Pc: scan over water saturation
        oil_water_capillary_pressure_drainage = get_oil_water_capillary_pressure(
            oil_water_capillary_pressure_table=oil_water_drainage_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        oil_water_capillary_pressure_imbibition = get_oil_water_capillary_pressure(
            oil_water_capillary_pressure_table=oil_water_imbibition_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        oil_water_capillary_pressure = compute_killough_scanning_curve(
            saturation=water_saturation,
            drainage_curve_value=oil_water_capillary_pressure_drainage,
            imbibition_curve_value=oil_water_capillary_pressure_imbibition,
            reversal_saturation=water_reversal_saturation,  # type: ignore
            maximum_historical_saturation=maximum_water_saturation,
            is_imbibition=water_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )

        # Gas-oil Pc: scan over gas saturation
        gas_oil_capillary_pressure_drainage = get_gas_oil_capillary_pressure(
            gas_oil_capillary_pressure_table=gas_oil_drainage_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        gas_oil_capillary_pressure_imbibition = get_gas_oil_capillary_pressure(
            gas_oil_capillary_pressure_table=gas_oil_imbibition_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        gas_oil_capillary_pressure = compute_killough_scanning_curve(
            saturation=gas_saturation,
            drainage_curve_value=gas_oil_capillary_pressure_drainage,
            imbibition_curve_value=gas_oil_capillary_pressure_imbibition,
            reversal_saturation=gas_reversal_saturation,  # type: ignore
            maximum_historical_saturation=maximum_gas_saturation,
            is_imbibition=gas_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )

        if is_scalar:
            return CapillaryPressures(
                oil_water=oil_water_capillary_pressure.item(),  # type: ignore
                gas_oil=gas_oil_capillary_pressure.item(),  # type: ignore
            )
        return CapillaryPressures(
            oil_water=oil_water_capillary_pressure,  # type: ignore[typeddict-item]
            gas_oil=gas_oil_capillary_pressure,  # type: ignore[typeddict-item]
        )

    def derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        max_water_saturation: NumberOrArray[NDimension] | None = None,
        max_gas_saturation: NumberOrArray[NDimension] | None = None,
        water_imbibition_flag: bool | BooleanArray[NDimension] | None = None,
        gas_imbibition_flag: bool | BooleanArray[NDimension] | None = None,
        water_reversal_saturation: NumberOrArray[NDimension] | None = None,
        gas_reversal_saturation: NumberOrArray[NDimension] | None = None,
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
        :param max_water_saturation: Historical maximum water saturation per cell,
            or `None` to disable hysteresis.
        :param max_gas_saturation: Historical maximum gas saturation per cell,
            or `None` to disable hysteresis.
        :param water_imbibition_flag: Per-cell oil-water imbibition flag or `None`.
        :param gas_imbibition_flag: Per-cell gas-oil imbibition flag or `None`.
        :param water_reversal_saturation: Water saturation at the last oil-water
            reversal, or `None` (defaults to `max_water_saturation`).
        :param gas_reversal_saturation: Gas saturation at the last gas-oil reversal,
            or `None` (defaults to `max_gas_saturation`).
        :param kwargs: Additional keyword arguments forwarded to the backing tables.
        :return: `CapillaryPressureDerivatives` dictionary containing
            `dpcow_dsw`, `dpcow_dso`, `dpcgo_dsg`, and `dpcgo_dso`.
        """
        is_scalar = np.isscalar(water_saturation)
        water_saturation = np.atleast_1d(water_saturation)  # type: ignore
        oil_saturation = np.atleast_1d(oil_saturation)  # type: ignore
        gas_saturation = np.atleast_1d(gas_saturation)  # type: ignore
        water_saturation, oil_saturation, gas_saturation = np.broadcast_arrays(  # type: ignore
            water_saturation, oil_saturation, gas_saturation
        )
        zeros = np.zeros_like(water_saturation)

        (
            maximum_water_saturation,
            maximum_gas_saturation,
            water_imbibition_flag,
            gas_imbibition_flag,
            water_reversal_saturation,
            gas_reversal_saturation,
        ) = self._parse_hysteresis_kwargs(  # type: ignore
            water_saturation=water_saturation,  # type: ignore
            gas_saturation=gas_saturation,  # type: ignore
            max_water_saturation=max_water_saturation,
            max_gas_saturation=max_gas_saturation,
            water_imbibition_flag=water_imbibition_flag,
            gas_imbibition_flag=gas_imbibition_flag,
            water_reversal_saturation=water_reversal_saturation,
            gas_reversal_saturation=gas_reversal_saturation,
        )

        oil_water_drainage_table = self.oil_water_drainage_table
        oil_water_imbibition_table = self.oil_water_imbibition_table or oil_water_drainage_table
        gas_oil_drainage_table = self.gas_oil_drainage_table
        gas_oil_imbibition_table = self.gas_oil_imbibition_table or gas_oil_drainage_table

        # Oil-water
        oil_water_capillary_pressure_drainage = get_oil_water_capillary_pressure(
            oil_water_capillary_pressure_table=oil_water_drainage_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        oil_water_capillary_pressure_imbibition = get_oil_water_capillary_pressure(
            oil_water_capillary_pressure_table=oil_water_imbibition_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        oil_water_capillary_pressure_drainage_derivative = (
            get_oil_water_capillary_pressure_derivative(
                oil_water_capillary_pressure_table=oil_water_drainage_table,
                water_saturation=water_saturation,
                oil_saturation=oil_saturation,
                gas_saturation=gas_saturation,
                **kwargs,
            )
        )
        oil_water_capillary_pressure_imbibition_derivative = (
            get_oil_water_capillary_pressure_derivative(
                oil_water_capillary_pressure_table=oil_water_imbibition_table,
                water_saturation=water_saturation,
                oil_saturation=oil_saturation,
                gas_saturation=gas_saturation,
                **kwargs,
            )
        )

        oil_water_capillary_pressure_derivative = compute_killough_scanning_curve_derivative(
            saturation=water_saturation,
            drainage_curve_value=oil_water_capillary_pressure_drainage,
            imbibition_curve_value=oil_water_capillary_pressure_imbibition,
            drainage_curve_derivative=oil_water_capillary_pressure_drainage_derivative,
            imbibition_curve_derivative=oil_water_capillary_pressure_imbibition_derivative,
            reversal_saturation=water_reversal_saturation,  # type: ignore
            maximum_historical_saturation=maximum_water_saturation,
            is_imbibition=water_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )

        # Gas-oil
        gas_oil_capillary_pressure_drainage = get_gas_oil_capillary_pressure(
            gas_oil_capillary_pressure_table=gas_oil_drainage_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        gas_oil_capillary_pressure_imbibition = get_gas_oil_capillary_pressure(
            gas_oil_capillary_pressure_table=gas_oil_imbibition_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        gas_oil_capillary_pressure_drainage_derivative = get_gas_oil_capillary_pressure_derivative(
            gas_oil_capillary_pressure_table=gas_oil_drainage_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        gas_oil_capillary_pressure_imbibition_derivative = (
            get_gas_oil_capillary_pressure_derivative(
                gas_oil_capillary_pressure_table=gas_oil_imbibition_table,
                water_saturation=water_saturation,
                oil_saturation=oil_saturation,
                gas_saturation=gas_saturation,
                **kwargs,
            )
        )

        gas_oil_capillary_pressure_derivative = compute_killough_scanning_curve_derivative(
            saturation=gas_saturation,
            drainage_curve_value=gas_oil_capillary_pressure_drainage,
            imbibition_curve_value=gas_oil_capillary_pressure_imbibition,
            drainage_curve_derivative=gas_oil_capillary_pressure_drainage_derivative,
            imbibition_curve_derivative=gas_oil_capillary_pressure_imbibition_derivative,
            reversal_saturation=gas_reversal_saturation,  # type: ignore
            maximum_historical_saturation=maximum_gas_saturation,
            is_imbibition=gas_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )

        if is_scalar:
            return CapillaryPressureDerivatives(
                dpcow_dsw=oil_water_capillary_pressure_derivative.item(),  # type: ignore
                dpcow_dso=0.0,
                dpcgo_dsg=gas_oil_capillary_pressure_derivative.item(),  # type: ignore
                dpcgo_dso=0.0,
            )

        return CapillaryPressureDerivatives(
            dpcow_dsw=oil_water_capillary_pressure_derivative,
            dpcow_dso=zeros.copy(),
            dpcgo_dsg=gas_oil_capillary_pressure_derivative,
            dpcgo_dso=zeros.copy(),
        )
