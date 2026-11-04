"""
Well and field summary vectors, built on `bores.schedule.summary`.

A leading `F` is a field-wide total and a leading `W` is one well. Every class has a short alias
named after its deck mnemonic (`FOPR = FieldOilProductionRate`). All rates are surface-condition.

Production vectors only count producers and injection vectors only count injectors, so a
well's injection vectors read zero for a producer and its production vectors read
zero for an injector. Every volume and rate is in the model's own unit system, so FIELD
reports cubic feet and cubic feet per day, not barrels. Cumulative totals
(`FOPT`, `WWIT`, ...) read the workspace's running volumes, which the run advances every
accepted step with `bores.wells.workspace.accumulate_well_volumes`.

A well that has not come into effect yet, or has no control, reads zero for every rate and
total. A well with no open connection also reads zero. Reading a rate from a well that is in
effect and has a control, but has not been resolved, is an error rather than a zero, so a well
the run failed to resolve can not pass for one that is not flowing. Pressure vectors (`WBHP`,
`WTHP`) read `NaN` for a well with no flowing pressure.

Vectors that report on the reservoir itself (fluids in place, average pressure, per-cell values)
live in `bores.reservoir.summary`.
"""

import typing

import attrs
import numpy as np

from bores.errors import SummaryError
from bores.schedule.base import ScheduleContext
from bores.schedule.summary import Summary, get_workspace
from bores.types import Integer, Number
from bores.wells.compile import (
    CompiledWellSystem,
    InjectorControlModeTag,
    ProducerControlModeTag,
    WellKind,
)
from bores.wells.schedule import resolve_well

if typing.TYPE_CHECKING:
    from bores.blackoil.compile import CompiledBlackOilModel
    from bores.simulation.workspace import SimulationWorkspace

__all__ = [
    "FGIR",
    "FGIT",
    "FGOR",
    "FGPR",
    "FGPT",
    "FLPR",
    "FOIR",
    "FOIT",
    "FOPR",
    "FOPT",
    "FWCT",
    "FWIR",
    "FWIT",
    "FWPR",
    "FWPT",
    "WBHP",
    "WGIR",
    "WGIT",
    "WGOR",
    "WGPR",
    "WGPT",
    "WLPR",
    "WOIR",
    "WOIT",
    "WOPR",
    "WOPT",
    "WTHP",
    "WWCT",
    "WWIR",
    "WWIT",
    "WWPR",
    "WWPT",
    "FieldGasInjectionRate",
    "FieldGasInjectionTotal",
    "FieldGasOilRatio",
    "FieldGasProductionRate",
    "FieldGasProductionTotal",
    "FieldLiquidProductionRate",
    "FieldOilInjectionRate",
    "FieldOilInjectionTotal",
    "FieldOilProductionRate",
    "FieldOilProductionTotal",
    "FieldWaterCut",
    "FieldWaterInjectionRate",
    "FieldWaterInjectionTotal",
    "FieldWaterProductionRate",
    "FieldWaterProductionTotal",
    "WellBottomHolePressure",
    "WellGasInjectionRate",
    "WellGasInjectionTotal",
    "WellGasOilRatio",
    "WellGasProductionRate",
    "WellGasProductionTotal",
    "WellLiquidProductionRate",
    "WellOilInjectionRate",
    "WellOilInjectionTotal",
    "WellOilProductionRate",
    "WellOilProductionTotal",
    "WellTubingHeadPressure",
    "WellWaterCut",
    "WellWaterInjectionRate",
    "WellWaterInjectionTotal",
    "WellWaterProductionRate",
    "WellWaterProductionTotal",
    "get_wells",
    "sum_rates",
]


def get_wells(*, model: "CompiledBlackOilModel") -> CompiledWellSystem:
    """
    Reads the model's compiled wells.

    :param model: The model to read from.
    :returns: The model's `CompiledWellSystem`.
    :raises SummaryError: If the model has no wells.
    """
    if model.wells is None:
        raise SummaryError("Well summary vectors need a model with wells, but it has none.")
    return model.wells


def is_expected_to_flow(*, wells: CompiledWellSystem, well_row: Integer) -> bool:
    """
    :param wells: The compiled wells.
    :param well_row: The well's row in the compiled well system.
    :returns: Whether the well is in effect and has a control, so its rates should have been
        resolved. A well that is not yet in effect, or has no control, has none to resolve.
    """
    unset_mode = (
        InjectorControlModeTag.UNSET
        if wells.well_kinds[well_row] == WellKind.INJECTOR
        else ProducerControlModeTag.UNSET
    )
    return bool(
        wells.schedule_statuses[well_row] == 1
        and wells.controls.control_modes[well_row] != unset_mode
    )


def find_unresolved_wells(
    *, wells: CompiledWellSystem, values: typing.Any, kind: WellKind | None = None
) -> list[str]:
    """
    Finds the wells that should have a resolved value but still have `NaN`.

    A well should have a value when it is in effect and has a control. A well that is not yet
    in effect, or has no control, legitimately has none.

    :param wells: The compiled wells.
    :param values: Shape `(n_wells,)` array of one resolved quantity per well.
    :param kind: Only wells of this kind are checked. Every well if not given.
    :returns: The names of the wells that should have a value and do not.
    """
    well_kinds = wells.well_kinds
    unset_modes = np.where(
        well_kinds == WellKind.INJECTOR,
        InjectorControlModeTag.UNSET,
        ProducerControlModeTag.UNSET,
    )
    expected = (wells.schedule_statuses == 1) & (wells.controls.control_modes != unset_modes)
    if kind is not None:
        expected &= well_kinds == kind
    return [str(wells.names[row]) for row in np.flatnonzero(expected & np.isnan(values))]


def raise_for_unresolved_wells(*, names: typing.Sequence[str], array_name: str) -> None:
    """
    :param names: The wells that should have a value and do not.
    :param array_name: The `WellsWorkspace` field that was read.
    :raises SummaryError: If `names` is not empty.
    """
    if names:
        raise SummaryError(
            f"Wells {list(names)} are in effect and have a control, but `{array_name}` has no "
            "value for them. The run has not resolved them yet, so no summary can be read."
        )


def sum_rates(
    *, wells: CompiledWellSystem, workspace: "SimulationWorkspace", array_name: str, kind: WellKind
) -> Number:
    """
    Sums one per-well rate array over every well of one kind.

    :param wells: The compiled wells, for each well's kind.
    :param workspace: The run's workspace, holding the rate arrays.
    :param array_name: The `WellsWorkspace` field to sum, e.g. `"surface_oil_rates"`.
    :param kind: Only wells of this kind are counted.
    :returns: The total. A well that is not yet in effect, or has no control, counts as zero.
    :raises SummaryError: If a well that is in effect and has a control has not been resolved.
    """
    rates = getattr(workspace.wells, array_name)
    raise_for_unresolved_wells(
        names=find_unresolved_wells(wells=wells, values=rates, kind=kind), array_name=array_name
    )
    return np.nansum(rates[wells.well_kinds == kind])


def divide(*, numerator: Number, denominator: Number) -> Number:
    """
    Divides, returning zero instead of raising when there is nothing to divide by.

    :param numerator: The dividend.
    :param denominator: The divisor.
    :returns: `numerator / denominator`, or `0.0` if `denominator` is zero.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


@attrs.frozen(kw_only=True, slots=True)
class FieldRate(Summary["CompiledBlackOilModel"]):
    """Base for a field-wide surface rate. Subclasses set `mnemonic`, `array_name`, and `kind`."""

    mnemonic = ""
    array_name = ""
    kind = WellKind.PRODUCER

    @property
    def key(self) -> str:
        """
        This vector's report key.

        :returns: The deck mnemonic, e.g. `"FOPR"`.
        """
        return self.mnemonic

    def __call__(self, model: "CompiledBlackOilModel", context: ScheduleContext) -> Number:
        """
        Sums the rate over every well of this vector's kind.

        :param model: The model to read wells from.
        :param context: The current moment's context, carrying the workspace.
        :returns: The field's total rate.
        """
        return sum_rates(
            wells=get_wells(model=model),
            workspace=get_workspace(context=context),
            array_name=self.array_name,
            kind=self.kind,
        )


@attrs.frozen(kw_only=True, slots=True)
class FieldOilProductionRate(FieldRate):
    """Field oil production rate (`FOPR`)."""

    mnemonic = "FOPR"
    array_name = "surface_oil_rates"


@attrs.frozen(kw_only=True, slots=True)
class FieldWaterProductionRate(FieldRate):
    """Field water production rate (`FWPR`)."""

    mnemonic = "FWPR"
    array_name = "surface_water_rates"


@attrs.frozen(kw_only=True, slots=True)
class FieldGasProductionRate(FieldRate):
    """Field gas production rate (`FGPR`)."""

    mnemonic = "FGPR"
    array_name = "surface_gas_rates"


@attrs.frozen(kw_only=True, slots=True)
class FieldWaterInjectionRate(FieldRate):
    """Field water injection rate (`FWIR`)."""

    mnemonic = "FWIR"
    array_name = "surface_water_rates"
    kind = WellKind.INJECTOR


@attrs.frozen(kw_only=True, slots=True)
class FieldOilInjectionRate(FieldRate):
    """Field oil injection rate (`FOIR`). Rarely used."""

    mnemonic = "FOIR"
    array_name = "surface_oil_rates"
    kind = WellKind.INJECTOR


@attrs.frozen(kw_only=True, slots=True)
class FieldGasInjectionRate(FieldRate):
    """Field gas injection rate (`FGIR`)."""

    mnemonic = "FGIR"
    array_name = "surface_gas_rates"
    kind = WellKind.INJECTOR


@attrs.frozen(kw_only=True, slots=True)
class FieldLiquidProductionRate(FieldRate):
    """Field liquid (oil plus water) production rate (`FLPR`)."""

    mnemonic = "FLPR"

    def __call__(self, model: "CompiledBlackOilModel", context: ScheduleContext) -> Number:
        """
        Sums oil and water production rates over every producer.

        :param model: The model to read wells from.
        :param context: The current moment's context, carrying the workspace.
        :returns: The field's total liquid production rate.
        """
        wells = get_wells(model=model)
        workspace = get_workspace(context=context)
        oil_rate = sum_rates(
            wells=wells,
            workspace=workspace,
            array_name="surface_oil_rates",
            kind=WellKind.PRODUCER,
        )
        water_rate = sum_rates(
            wells=wells,
            workspace=workspace,
            array_name="surface_water_rates",
            kind=WellKind.PRODUCER,
        )
        return oil_rate + water_rate


@attrs.frozen(kw_only=True, slots=True)
class FieldWaterCut(FieldRate):
    """Field water cut (`FWCT`): water over total liquid production, `0` with no liquid."""

    mnemonic = "FWCT"

    def __call__(self, model: "CompiledBlackOilModel", context: ScheduleContext) -> Number:
        """
        Divides water production by oil plus water production over every producer.

        :param model: The model to read wells from.
        :param context: The current moment's context, carrying the workspace.
        :returns: The field's water cut, between `0` and `1`.
        """
        wells = get_wells(model=model)
        workspace = get_workspace(context=context)
        oil_rate = sum_rates(
            wells=wells,
            workspace=workspace,
            array_name="surface_oil_rates",
            kind=WellKind.PRODUCER,
        )
        water_rate = sum_rates(
            wells=wells,
            workspace=workspace,
            array_name="surface_water_rates",
            kind=WellKind.PRODUCER,
        )
        return divide(numerator=water_rate, denominator=oil_rate + water_rate)


@attrs.frozen(kw_only=True, slots=True)
class FieldGasOilRatio(FieldRate):
    """Field gas-oil ratio (`FGOR`): gas over oil production, `0` with no oil."""

    mnemonic = "FGOR"

    def __call__(self, model: "CompiledBlackOilModel", context: ScheduleContext) -> Number:
        """
        Divides gas production by oil production over every producer.

        :param model: The model to read wells from.
        :param context: The current moment's context, carrying the workspace.
        :returns: The field's gas-oil ratio, in the model's unit system.
        """
        wells = get_wells(model=model)
        workspace = get_workspace(context=context)
        oil_rate = sum_rates(
            wells=wells,
            workspace=workspace,
            array_name="surface_oil_rates",
            kind=WellKind.PRODUCER,
        )
        gas_rate = sum_rates(
            wells=wells,
            workspace=workspace,
            array_name="surface_gas_rates",
            kind=WellKind.PRODUCER,
        )
        return divide(numerator=gas_rate, denominator=oil_rate)


@attrs.frozen(kw_only=True, slots=True)
class WellVector(Summary["CompiledBlackOilModel"]):
    """Base for a per-well vector. Subclasses set `mnemonic` and implement `get_value`."""

    mnemonic = ""

    kind: typing.ClassVar[WellKind | None] = None
    """
    The kind of well this vector reports on. A well of any other kind reads `0`.
    `None` reports every well, whatever its kind.
    """

    is_rate: typing.ClassVar[bool] = False
    """
    Whether this vector is built from the wells' flow rates. A well that is not yet in effect,
    or has no control, reads `0` for such a vector, and one that is in effect and has a control
    but has not been resolved is an error.
    """

    well_name: str
    """The well to report on."""

    @property
    def key(self) -> str:
        """
        This vector's report key.

        :returns: The deck mnemonic and well name, e.g. `"WOPR:PROD1"`.
        """
        return f"{self.mnemonic}:{self.well_name}"

    def get_value(
        self,
        *,
        well_row: Integer,
        workspace: "SimulationWorkspace",
    ) -> Number:
        """
        Reads this vector's value for one well. Must be overridden.

        :param well_row: The well's row in the compiled well system.
        :param workspace: The run's workspace.
        :returns: The well's value.
        """
        raise NotImplementedError

    def __call__(self, model: "CompiledBlackOilModel", context: ScheduleContext) -> Number:
        """
        Reads this vector's value for `well_name`.

        :param model: The model to look the well up in.
        :param context: The current moment's context, carrying the workspace.
        :returns: The well's value, or `0` if the well is not of this vector's kind, or if this
            is a rate vector and the well is not yet in effect or has no control.
        :raises ValidationError: If the model has no well named `well_name`.
        :raises SummaryError: If this is a rate vector and the well is in effect and has a
            control but has not been resolved.
        """
        well_row, wells = resolve_well(model=model, well_name=self.well_name)
        if self.kind is not None and wells.well_kinds[well_row] != self.kind:
            return 0.0
        value = self.get_value(well_row=well_row, workspace=get_workspace(context=context))
        if self.is_rate and np.isnan(value):
            if is_expected_to_flow(wells=wells, well_row=well_row):
                raise SummaryError(
                    f"Well {self.well_name!r} is in effect and has a control, but the run has "
                    f"not resolved it, so {self.mnemonic} can not be read."
                )
            return 0.0
        return value


@attrs.frozen(kw_only=True, slots=True)
class WellRate(WellVector):
    """Base for a per-well surface rate. Subclasses set `mnemonic` and `array_name`."""

    is_rate = True
    array_name = ""

    def get_value(self, *, well_row: Integer, workspace: "SimulationWorkspace") -> Number:
        """
        Reads the well's rate.

        :param well_row: The well's row in the compiled well system.
        :param workspace: The run's workspace.
        :returns: The well's rate for `array_name`.
        """
        return getattr(workspace.wells, self.array_name)[well_row]


@attrs.frozen(kw_only=True, slots=True)
class WellOilProductionRate(WellRate):
    """Well oil production rate (`WOPR`)."""

    mnemonic = "WOPR"
    kind = WellKind.PRODUCER
    array_name = "surface_oil_rates"


@attrs.frozen(kw_only=True, slots=True)
class WellWaterProductionRate(WellRate):
    """Well water production rate (`WWPR`)."""

    mnemonic = "WWPR"
    kind = WellKind.PRODUCER
    array_name = "surface_water_rates"


@attrs.frozen(kw_only=True, slots=True)
class WellGasProductionRate(WellRate):
    """Well gas production rate (`WGPR`)."""

    mnemonic = "WGPR"
    kind = WellKind.PRODUCER
    array_name = "surface_gas_rates"


@attrs.frozen(kw_only=True, slots=True)
class WellWaterInjectionRate(WellRate):
    """Well water injection rate (`WWIR`)."""

    mnemonic = "WWIR"
    kind = WellKind.INJECTOR
    array_name = "surface_water_rates"


@attrs.frozen(kw_only=True, slots=True)
class WellOilInjectionRate(WellRate):
    """Well oil injection rate (`WOIR`). Rarely used."""

    mnemonic = "WOIR"
    kind = WellKind.INJECTOR
    array_name = "surface_oil_rates"


@attrs.frozen(kw_only=True, slots=True)
class WellGasInjectionRate(WellRate):
    """Well gas injection rate (`WGIR`)."""

    mnemonic = "WGIR"
    kind = WellKind.INJECTOR
    array_name = "surface_gas_rates"


@attrs.frozen(kw_only=True, slots=True)
class WellLiquidProductionRate(WellVector):
    """Well liquid (oil plus water) production rate (`WLPR`)."""

    mnemonic = "WLPR"
    is_rate = True
    kind = WellKind.PRODUCER

    def get_value(self, *, well_row: Integer, workspace: "SimulationWorkspace") -> Number:
        """
        Sums the well's oil and water rates.

        :param well_row: The well's row in the compiled well system.
        :param workspace: The run's workspace.
        :returns: The well's liquid production rate.
        """
        return (
            workspace.wells.surface_oil_rates[well_row]
            + workspace.wells.surface_water_rates[well_row]
        )


@attrs.frozen(kw_only=True, slots=True)
class WellWaterCut(WellVector):
    """Well water cut (`WWCT`): water over total liquid, `0` with no liquid."""

    mnemonic = "WWCT"
    is_rate = True
    kind = WellKind.PRODUCER

    def get_value(self, *, well_row: Integer, workspace: "SimulationWorkspace") -> Number:
        """
        Divides the well's water rate by its oil plus water rate.

        :param well_row: The well's row in the compiled well system.
        :param workspace: The run's workspace.
        :returns: The well's water cut, between `0` and `1`.
        """
        oil_rate = workspace.wells.surface_oil_rates[well_row]
        water_rate = workspace.wells.surface_water_rates[well_row]
        return divide(numerator=water_rate, denominator=oil_rate + water_rate)


@attrs.frozen(kw_only=True, slots=True)
class WellGasOilRatio(WellVector):
    """Well gas-oil ratio (`WGOR`): gas over oil, `0` with no oil."""

    mnemonic = "WGOR"
    is_rate = True
    kind = WellKind.PRODUCER

    def get_value(self, *, well_row: Integer, workspace: "SimulationWorkspace") -> Number:
        """
        Divides the well's gas rate by its oil rate.

        :param well_row: The well's row in the compiled well system.
        :param workspace: The run's workspace.
        :returns: The well's gas-oil ratio, in the model's unit system.
        """
        return divide(
            numerator=workspace.wells.surface_gas_rates[well_row],
            denominator=workspace.wells.surface_oil_rates[well_row],
        )


@attrs.frozen(kw_only=True, slots=True)
class WellBottomHolePressure(WellVector):
    """Well bottom-hole pressure (`WBHP`). `NaN` for a well that is not yet in effect, not yet resolved, or has no open connection."""

    mnemonic = "WBHP"

    def get_value(self, *, well_row: Integer, workspace: "SimulationWorkspace") -> Number:
        """
        Reads the well's flowing bottom-hole pressure.

        :param well_row: The well's row in the compiled well system.
        :param workspace: The run's workspace.
        :returns: The well's bottom-hole pressure.
        """
        return workspace.wells.bhps[well_row]


@attrs.frozen(kw_only=True, slots=True)
class WellTubingHeadPressure(WellVector):
    """Well tubing-head pressure (`WTHP`). `NaN` where it wasn't computed."""

    mnemonic = "WTHP"

    def get_value(self, *, well_row: Integer, workspace: "SimulationWorkspace") -> Number:
        """
        Reads the well's tubing-head pressure.

        :param well_row: The well's row in the compiled well system.
        :param workspace: The run's workspace.
        :returns: The well's tubing-head pressure.
        """
        return workspace.wells.thps[well_row]


@attrs.frozen(kw_only=True, slots=True)
class FieldOilProductionTotal(FieldRate):
    """Field cumulative oil production (`FOPT`)."""

    mnemonic = "FOPT"
    array_name = "cumulative_oil_volumes"


@attrs.frozen(kw_only=True, slots=True)
class FieldWaterProductionTotal(FieldRate):
    """Field cumulative water production (`FWPT`)."""

    mnemonic = "FWPT"
    array_name = "cumulative_water_volumes"


@attrs.frozen(kw_only=True, slots=True)
class FieldGasProductionTotal(FieldRate):
    """Field cumulative gas production (`FGPT`)."""

    mnemonic = "FGPT"
    array_name = "cumulative_gas_volumes"


@attrs.frozen(kw_only=True, slots=True)
class FieldWaterInjectionTotal(FieldRate):
    """Field cumulative water injection (`FWIT`)."""

    mnemonic = "FWIT"
    array_name = "cumulative_water_volumes"
    kind = WellKind.INJECTOR


@attrs.frozen(kw_only=True, slots=True)
class FieldOilInjectionTotal(FieldRate):
    """Field cumulative oil injection (`FOIT`). Rarely used."""

    mnemonic = "FOIT"
    array_name = "cumulative_oil_volumes"
    kind = WellKind.INJECTOR


@attrs.frozen(kw_only=True, slots=True)
class FieldGasInjectionTotal(FieldRate):
    """Field cumulative gas injection (`FGIT`)."""

    mnemonic = "FGIT"
    array_name = "cumulative_gas_volumes"
    kind = WellKind.INJECTOR


@attrs.frozen(kw_only=True, slots=True)
class WellOilProductionTotal(WellRate):
    """Well cumulative oil production (`WOPT`)."""

    mnemonic = "WOPT"
    kind = WellKind.PRODUCER
    array_name = "cumulative_oil_volumes"


@attrs.frozen(kw_only=True, slots=True)
class WellWaterProductionTotal(WellRate):
    """Well cumulative water production (`WWPT`)."""

    mnemonic = "WWPT"
    kind = WellKind.PRODUCER
    array_name = "cumulative_water_volumes"


@attrs.frozen(kw_only=True, slots=True)
class WellGasProductionTotal(WellRate):
    """Well cumulative gas production (`WGPT`)."""

    mnemonic = "WGPT"
    kind = WellKind.PRODUCER
    array_name = "cumulative_gas_volumes"


@attrs.frozen(kw_only=True, slots=True)
class WellWaterInjectionTotal(WellRate):
    """Well cumulative water injection (`WWIT`)."""

    mnemonic = "WWIT"
    kind = WellKind.INJECTOR
    array_name = "cumulative_water_volumes"


@attrs.frozen(kw_only=True, slots=True)
class WellOilInjectionTotal(WellRate):
    """Well cumulative oil injection (`WOIT`). Rarely used."""

    mnemonic = "WOIT"
    kind = WellKind.INJECTOR
    array_name = "cumulative_oil_volumes"


@attrs.frozen(kw_only=True, slots=True)
class WellGasInjectionTotal(WellRate):
    """Well cumulative gas injection (`WGIT`)."""

    mnemonic = "WGIT"
    kind = WellKind.INJECTOR
    array_name = "cumulative_gas_volumes"


FOPR = FieldOilProductionRate
FWPR = FieldWaterProductionRate
FGPR = FieldGasProductionRate
FLPR = FieldLiquidProductionRate
FWIR = FieldWaterInjectionRate
FOIR = FieldOilInjectionRate
FGIR = FieldGasInjectionRate
FWCT = FieldWaterCut
FGOR = FieldGasOilRatio
WOPR = WellOilProductionRate
WWPR = WellWaterProductionRate
WGPR = WellGasProductionRate
WLPR = WellLiquidProductionRate
WWIR = WellWaterInjectionRate
WOIR = WellOilInjectionRate
WGIR = WellGasInjectionRate
WWCT = WellWaterCut
WGOR = WellGasOilRatio
WBHP = WellBottomHolePressure
WTHP = WellTubingHeadPressure
FOPT = FieldOilProductionTotal
FWPT = FieldWaterProductionTotal
FGPT = FieldGasProductionTotal
FWIT = FieldWaterInjectionTotal
FOIT = FieldOilInjectionTotal
FGIT = FieldGasInjectionTotal
WOPT = WellOilProductionTotal
WWPT = WellWaterProductionTotal
WGPT = WellGasProductionTotal
WWIT = WellWaterInjectionTotal
WOIT = WellOilInjectionTotal
WGIT = WellGasInjectionTotal
