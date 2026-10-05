"""
Well and field summary vectors, built on `bores.schedule.summary`.

Names follow Eclipse's own summary mnemonics. a leading `F` is a field-wide
total, a leading `W` is one well. Every class has a short alias
(`FOPR = FieldOilProductionRate`). All rates are surface-condition, and
production/injection totals only count wells of the matching kind. Cumulative totals
(`FOPT`, `WWIT`, ...) read the workspace's running volumes, which the run advances every
accepted step with `bores.wells.workspace.accumulate_well_volumes`.
"""

import typing

import attrs
import numpy as np
from typing_extensions import Self

from bores.constants import c
from bores.errors import SummaryError
from bores.schedule.base import ScheduleContext
from bores.schedule.summary import Summary
from bores.types import CellArray, IntCellArray, Integer, Number, UnitSystem
from bores.wells.compile import CompiledWellSystem, WellKind
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
    "RGIP",
    "ROIP",
    "RWIP",
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
    "RegionGasInPlace",
    "RegionInPlace",
    "RegionOilInPlace",
    "RegionWaterInPlace",
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
    "get_workspace",
    "sum_rates",
]


def get_workspace(*, context: ScheduleContext) -> "SimulationWorkspace":
    """
    Reads the run's `SimulationWorkspace` from `context.extra["workspace"]`.

    :param context: The current moment's context.
    :returns: The run's `SimulationWorkspace`.
    :raises SummaryError: If the key is missing or isn't a `SimulationWorkspace`.
    """
    from bores.simulation.workspace import SimulationWorkspace

    workspace = context.extra.get("workspace")
    if not isinstance(workspace, SimulationWorkspace):
        raise SummaryError(
            "Well summary vectors need key 'workspace' in `context.extra`, "
            f"with a `SimulationWorkspace` value, but got {workspace!r}."
        )
    return workspace


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


def sum_rates(
    *, wells: CompiledWellSystem, workspace: "SimulationWorkspace", array_name: str, kind: WellKind
) -> Number:
    """
    Sums one per-well rate array over every well of one kind.

    :param wells: The compiled wells, for each well's kind.
    :param workspace: The run's workspace, holding the rate arrays.
    :param array_name: The `WellsWorkspace` field to sum, e.g. `"surface_oil_rates"`.
    :param kind: Only wells of this kind are counted.
    :returns: The total. A well not yet resolved (`NaN`) counts as zero.
    """
    rates = getattr(workspace.wells, array_name)
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

        :returns: The Eclipse mnemonic, e.g. `"FOPR"`.
        """
        return self.mnemonic

    def __dump__(self) -> dict[str, typing.Any]:
        """
        Dumps this vector. A field-wide vector has nothing to configure.

        :returns: An empty mapping.
        """
        return {}

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        """
        Loads this vector. A field-wide vector has nothing to configure.

        :param data: Ignored.
        :returns: A new instance.
        """
        return cls()

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
    """Field oil injection rate (`FOIR`). Rare in practice; Eclipse still defines it."""

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

    well_name: str
    """The well to report on."""

    @property
    def key(self) -> str:
        """
        This vector's report key.

        :returns: The Eclipse mnemonic and well name, e.g. `"WOPR:PROD1"`.
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
        :returns: The well's value.
        :raises ValidationError: If the model has no well named `well_name`.
        """
        well_row, _ = resolve_well(model=model, well_name=self.well_name)
        return self.get_value(well_row=well_row, workspace=get_workspace(context=context))


@attrs.frozen(kw_only=True, slots=True)
class WellRate(WellVector):
    """Base for a per-well surface rate. Subclasses set `mnemonic` and `array_name`."""

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
    array_name = "surface_oil_rates"


@attrs.frozen(kw_only=True, slots=True)
class WellWaterProductionRate(WellRate):
    """Well water production rate (`WWPR`)."""

    mnemonic = "WWPR"
    array_name = "surface_water_rates"


@attrs.frozen(kw_only=True, slots=True)
class WellGasProductionRate(WellRate):
    """Well gas production rate (`WGPR`)."""

    mnemonic = "WGPR"
    array_name = "surface_gas_rates"


@attrs.frozen(kw_only=True, slots=True)
class WellWaterInjectionRate(WellRate):
    """Well water injection rate (`WWIR`)."""

    mnemonic = "WWIR"
    array_name = "surface_water_rates"


@attrs.frozen(kw_only=True, slots=True)
class WellOilInjectionRate(WellRate):
    """Well oil injection rate (`WOIR`). Rare in practice; Eclipse still defines it."""

    mnemonic = "WOIR"
    array_name = "surface_oil_rates"


@attrs.frozen(kw_only=True, slots=True)
class WellGasInjectionRate(WellRate):
    """Well gas injection rate (`WGIR`)."""

    mnemonic = "WGIR"
    array_name = "surface_gas_rates"


@attrs.frozen(kw_only=True, slots=True)
class WellLiquidProductionRate(WellVector):
    """Well liquid (oil plus water) production rate (`WLPR`)."""

    mnemonic = "WLPR"

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
    """Well bottom-hole pressure (`WBHP`). `NaN` for a well not yet resolved."""

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
    """Field cumulative oil injection (`FOIT`). Rare in practice; Eclipse still defines it."""

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
    array_name = "cumulative_oil_volumes"


@attrs.frozen(kw_only=True, slots=True)
class WellWaterProductionTotal(WellRate):
    """Well cumulative water production (`WWPT`)."""

    mnemonic = "WWPT"
    array_name = "cumulative_water_volumes"


@attrs.frozen(kw_only=True, slots=True)
class WellGasProductionTotal(WellRate):
    """Well cumulative gas production (`WGPT`)."""

    mnemonic = "WGPT"
    array_name = "cumulative_gas_volumes"


@attrs.frozen(kw_only=True, slots=True)
class WellWaterInjectionTotal(WellRate):
    """Well cumulative water injection (`WWIT`)."""

    mnemonic = "WWIT"
    array_name = "cumulative_water_volumes"


@attrs.frozen(kw_only=True, slots=True)
class WellOilInjectionTotal(WellRate):
    """Well cumulative oil injection (`WOIT`). Rare in practice; Eclipse still defines it."""

    __type__: typing.ClassVar[str] = "well_oil_injection_total"

    mnemonic = "WOIT"
    array_name = "cumulative_oil_volumes"


@attrs.frozen(kw_only=True, slots=True)
class WellGasInjectionTotal(WellRate):
    """Well cumulative gas injection (`WGIT`)."""

    mnemonic = "WGIT"
    array_name = "cumulative_gas_volumes"


@attrs.frozen(kw_only=True, slots=True)
class RegionInPlace(Summary["CompiledBlackOilModel"]):
    """
    Base for a fluid-in-place vector over one fluid-in-place region (`FIPNUM`).

    Subclasses set `mnemonic` and implement `get_value`. A model with no fluid-in-place regions
    has a single region, number 1, covering every cell.
    """

    mnemonic = ""

    region: Integer = 1
    """The fluid-in-place region number to report on."""

    @property
    def key(self) -> str:
        """
        This vector's report key.

        :returns: The Eclipse mnemonic and region number, e.g. `"ROIP:2"`.
        """
        return f"{self.mnemonic}:{self.region}"

    def get_value(
        self,
        *,
        region_mask: IntCellArray,
        pore_volumes: CellArray,
        workspace: "SimulationWorkspace",
        unit_system: UnitSystem,
    ) -> Number:
        """
        Reads this vector's value over the cells of the region. Must be overridden.

        :param region_mask: Shape `(n_cells,)` boolean mask of the region's cells.
        :param pore_volumes: Shape `(n_cells,)` pore volume of every cell.
        :param workspace: The run's workspace.
        :param unit_system: The model's unit system.
        :returns: The in-place volume.
        """
        raise NotImplementedError

    def __call__(self, model: "CompiledBlackOilModel", context: ScheduleContext) -> Number:
        """
        Reads this vector's value over the cells of `region`.

        :param model: The model to read the regions and pore volumes from.
        :param context: The current moment's context, carrying the workspace.
        :returns: The in-place volume of the region.
        :raises SummaryError: If the region has no cells.
        """
        reservoir = model.reservoir
        pore_volumes = reservoir.pore_volumes
        fluid_in_place_regions = (
            reservoir.regions.fluid_in_place_region if reservoir.regions is not None else None
        )
        if fluid_in_place_regions is None:
            region_mask = np.full(pore_volumes.shape, self.region == 1)
        else:
            region_mask = np.asarray(fluid_in_place_regions) == self.region

        if not region_mask.any():
            raise SummaryError(
                f"{self.mnemonic}: fluid-in-place region {self.region} has no cells."
            )
        return self.get_value(
            region_mask=region_mask,
            pore_volumes=pore_volumes,
            workspace=get_workspace(context=context),
            unit_system=model.unit_system,
        )


def get_volume_factor(unit_system: UnitSystem) -> Number:
    """
    Factor from a stock-tank volume in the internal volume unit
    to the reported oil/water unit.

    :param unit_system: The model's unit system.
    :returns: Cubic feet to barrels for FIELD, `1` for every other unit system.
    """
    return c.CUBIC_FEET_TO_BARRELS if unit_system == UnitSystem.FIELD else 1.0


@attrs.frozen(kw_only=True, slots=True)
class RegionOilInPlace(RegionInPlace):
    """
    Oil in place in a fluid-in-place region (`ROIP`), as stock-tank oil.

    Counts oil in the oil phase only (`So * PV / Bo`); oil vaporized in free gas is not included.
    """

    mnemonic = "ROIP"

    def get_value(
        self,
        *,
        region_mask: IntCellArray,
        pore_volumes: CellArray,
        workspace: "SimulationWorkspace",
        unit_system: UnitSystem,
    ) -> Number:
        oil_volume = np.empty_like(pore_volumes)
        np.multiply(
            workspace.reservoir.oil_saturation,
            pore_volumes,
            out=oil_volume,
        )
        np.divide(
            oil_volume,
            workspace.physics.pvt.oil_formation_volume_factor,
            out=oil_volume,
        )
        return oil_volume[region_mask].sum() * get_volume_factor(unit_system)


@attrs.frozen(kw_only=True, slots=True)
class RegionWaterInPlace(RegionInPlace):
    """Water in place in a fluid-in-place region (`RWIP`), as stock-tank water."""

    mnemonic = "RWIP"

    def get_value(
        self,
        *,
        region_mask: IntCellArray,
        pore_volumes: CellArray,
        workspace: "SimulationWorkspace",
        unit_system: UnitSystem,
    ) -> Number:
        water_volume = np.empty_like(pore_volumes)
        np.multiply(
            workspace.reservoir.water_saturation,
            pore_volumes,
            out=water_volume,
        )
        np.divide(
            water_volume,
            workspace.physics.pvt.water_formation_volume_factor,
            out=water_volume,
        )
        return water_volume[region_mask].sum() * get_volume_factor(unit_system)


@attrs.frozen(kw_only=True, slots=True)
class RegionGasInPlace(RegionInPlace):
    """
    Gas in place in a fluid-in-place region (`RGIP`): free gas plus gas dissolved in the oil.

    Reported in standard cubic feet for FIELD units (cubic metres for METRIC).
    """

    mnemonic = "RGIP"

    def get_value(
        self,
        *,
        region_mask: IntCellArray,
        pore_volumes: CellArray,
        workspace: "SimulationWorkspace",
        unit_system: UnitSystem,
    ) -> Number:
        state = workspace.reservoir
        pvt = workspace.physics.pvt

        free_gas = np.empty_like(pore_volumes)
        np.multiply(state.gas_saturation, pore_volumes, out=free_gas)
        np.divide(free_gas, pvt.gas_formation_volume_factor, out=free_gas)

        stock_tank_oil = np.empty_like(pore_volumes)
        np.multiply(state.oil_saturation, pore_volumes, out=stock_tank_oil)
        np.divide(stock_tank_oil, pvt.oil_formation_volume_factor, out=stock_tank_oil)

        dissolved_gas = np.empty_like(pore_volumes)
        np.multiply(state.solution_gor, stock_tank_oil, out=dissolved_gas)
        np.multiply(dissolved_gas, get_volume_factor(unit_system), out=dissolved_gas)

        return (free_gas + dissolved_gas)[region_mask].sum()


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
ROIP = RegionOilInPlace
RWIP = RegionWaterInPlace
RGIP = RegionGasInPlace
