"""
Well and field summary vectors, built on `bores.schedule.summary`.

Names follow Eclipse's summary mnemonics. A leading `F` is a field-wide
total, a leading `W` is one well. Every class has a short alias
(`FOPR = FieldOilProductionRate`). All rates are surface-condition.

Production vectors only count producers and injection vectors only count injectors, so a
well's injection vectors read zero for a producer and its production vectors read
zero for an injector. Every volume and rate is in the model's own unit system, so FIELD
reports cubic feet and cubic feet per day, not barrels. Cumulative totals
(`FOPT`, `WWIT`, ...) read the workspace's running volumes, which the run advances every
accepted step with `bores.wells.workspace.accumulate_well_volumes`.

Block vectors (`BPR`, `BGSAT`, ...) report one grid cell. They take the cell as 0-based
`i`, `j`, `k`, like `Grid.flat_index`, and name it with Eclipse's 1-based indices in their
report key, e.g. `"BPR:1,1,1"` for the first cell.
"""

import typing

import attrs
import numpy as np

from bores.constants import c
from bores.errors import SummaryError
from bores.schedule.base import ScheduleContext
from bores.schedule.summary import Summary
from bores.types import BooleanCellArray, CellArray, Integer, Number, UnitSystem
from bores.wells.compile import CompiledWellSystem, WellKind
from bores.wells.schedule import resolve_well

if typing.TYPE_CHECKING:
    from bores.blackoil.compile import CompiledBlackOilModel
    from bores.simulation.workspace import SimulationWorkspace

__all__ = [
    "BGKR",
    "BGSAT",
    "BOKR",
    "BOSAT",
    "BPR",
    "BPRESSUR",
    "BRS",
    "BRV",
    "BSGAS",
    "BSOIL",
    "BSWAT",
    "BWKR",
    "BWSAT",
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
    "BlockGasOilRatio",
    "BlockGasRelativePermeability",
    "BlockGasSaturation",
    "BlockOilRelativePermeability",
    "BlockOilSaturation",
    "BlockPressure",
    "BlockVaporizedOilGasRatio",
    "BlockVector",
    "BlockWaterRelativePermeability",
    "BlockWaterSaturation",
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

    kind: typing.ClassVar[WellKind | None] = None
    """
    The kind of well this vector reports on. A well of any other kind reads `0`.
    `None` reports every well, whatever its kind.
    """

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
        :returns: The well's value, or `0` if the well is not of this vector's kind.
        :raises ValidationError: If the model has no well named `well_name`.
        """
        well_row, wells = resolve_well(model=model, well_name=self.well_name)
        if self.kind is not None and wells.well_kinds[well_row] != self.kind:
            return 0.0
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
    """Well oil injection rate (`WOIR`). Rare in practice; Eclipse still defines it."""

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
    """Well cumulative oil injection (`WOIT`). Rare in practice; Eclipse still defines it."""

    mnemonic = "WOIT"
    kind = WellKind.INJECTOR
    array_name = "cumulative_oil_volumes"


@attrs.frozen(kw_only=True, slots=True)
class WellGasInjectionTotal(WellRate):
    """Well cumulative gas injection (`WGIT`)."""

    mnemonic = "WGIT"
    kind = WellKind.INJECTOR
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
        region_mask: BooleanCellArray,
        pore_volumes: CellArray,
        workspace: "SimulationWorkspace",
        unit_system: UnitSystem,
    ) -> Number:
        """
        Reads this vector's value over the cells of the region. Must be overridden.

        :param region_mask: Shape `(n_cells,)` boolean mask of the region's cells.
        :param pore_volumes: Shape `(n_cells,)` pore volume of every cell at the current pressure.
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
        :returns: The in-place volume of the region, at the pore volume the run holds now.
        :raises SummaryError: If the region has no cells.
        """
        workspace = get_workspace(context=context)
        reservoir = model.reservoir
        pore_volumes = reservoir.compute_pore_volumes(workspace.reservoir.pressure)
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
            workspace=workspace,
            unit_system=model.unit_system,
        )


def get_stock_tank_barrels_per_volume(unit_system: UnitSystem) -> Number:
    """
    The factor between a volume in the model's own volume unit and stock-tank barrels.

    The gas-oil, oil-gas and gas-water ratios are per stock-tank barrel in FIELD units, while
    the volumes the vectors report are in cubic feet. Every other unit system states these
    ratios per unit of volume, so nothing needs converting.

    :param unit_system: The model's unit system.
    :returns: Stock-tank barrels per cubic foot for FIELD, `1` for every other unit system.
    """
    return c.CUBIC_FEET_TO_STB if unit_system == UnitSystem.FIELD else 1.0


def finite_or_zero(values: CellArray) -> CellArray:
    """
    :param values: A per-cell ratio that is `NaN` where the model doesn't track it.
    :returns: `values`, with `NaN` read as `0`.
    """
    return typing.cast(CellArray, np.nan_to_num(values, nan=0.0))


@attrs.frozen(kw_only=True, slots=True)
class RegionOilInPlace(RegionInPlace):
    """
    Oil in place in a fluid-in-place region (`ROIP`), as stock-tank oil.

    Counts the oil in the oil phase plus the oil vaporized in free gas, when the model has a
    vaporized oil-gas ratio. Reported in the model's own volume unit.
    """

    mnemonic = "ROIP"

    def get_value(
        self,
        *,
        region_mask: BooleanCellArray,
        pore_volumes: CellArray,
        workspace: "SimulationWorkspace",
        unit_system: UnitSystem,
    ) -> Number:
        state = workspace.reservoir
        pvt = workspace.physics.pvt

        oil_in_oil_phase = state.oil_saturation * pore_volumes / pvt.oil_formation_volume_factor
        free_gas = state.gas_saturation * pore_volumes / pvt.gas_formation_volume_factor
        vaporized_oil = (
            finite_or_zero(pvt.vaporized_oil_gas_ratio)
            * free_gas
            / get_stock_tank_barrels_per_volume(unit_system)
        )
        return (oil_in_oil_phase + vaporized_oil)[region_mask].sum()


@attrs.frozen(kw_only=True, slots=True)
class RegionWaterInPlace(RegionInPlace):
    """
    Water in place in a fluid-in-place region (`RWIP`), as stock-tank water.

    Reported in the model's own volume unit.
    """

    mnemonic = "RWIP"

    def get_value(
        self,
        *,
        region_mask: BooleanCellArray,
        pore_volumes: CellArray,
        workspace: "SimulationWorkspace",
        unit_system: UnitSystem,
    ) -> Number:
        water_volume = (
            workspace.reservoir.water_saturation
            * pore_volumes
            / workspace.physics.pvt.water_formation_volume_factor
        )
        return water_volume[region_mask].sum()


@attrs.frozen(kw_only=True, slots=True)
class RegionGasInPlace(RegionInPlace):
    """
    Gas in place in a fluid-in-place region (`RGIP`): free gas, plus gas dissolved in the oil,
    plus gas dissolved in the water when the model has a gas solubility in water.

    Reported in the model's own volume unit, standard cubic feet for FIELD.
    """

    mnemonic = "RGIP"

    def get_value(
        self,
        *,
        region_mask: BooleanCellArray,
        pore_volumes: CellArray,
        workspace: "SimulationWorkspace",
        unit_system: UnitSystem,
    ) -> Number:
        state = workspace.reservoir
        pvt = workspace.physics.pvt
        barrels_per_volume = get_stock_tank_barrels_per_volume(unit_system)

        free_gas = state.gas_saturation * pore_volumes / pvt.gas_formation_volume_factor
        oil_in_oil_phase = state.oil_saturation * pore_volumes / pvt.oil_formation_volume_factor
        water_volume = state.water_saturation * pore_volumes / pvt.water_formation_volume_factor
        gas_in_oil = state.solution_gor * oil_in_oil_phase * barrels_per_volume
        gas_in_water = (
            finite_or_zero(state.gas_solubility_in_water) * water_volume * barrels_per_volume
        )
        return (free_gas + gas_in_oil + gas_in_water)[region_mask].sum()


@attrs.frozen(kw_only=True, slots=True)
class BlockVector(Summary["CompiledBlackOilModel"]):
    """Base for a per-cell vector. Subclasses set `mnemonic` and implement `get_value`."""

    mnemonic = ""

    i: Integer
    """The cell's 0-based x index."""

    j: Integer
    """The cell's 0-based y index."""

    k: Integer
    """The cell's 0-based z index."""

    @property
    def key(self) -> str:
        """
        This vector's report key.

        :returns: The Eclipse mnemonic and the cell's 1-based indices, e.g. `"BPR:1,1,1"`.
        """
        return f"{self.mnemonic}:{self.i + 1},{self.j + 1},{self.k + 1}"

    def get_value(self, *, cell_index: Integer, workspace: "SimulationWorkspace") -> Number:
        """
        Reads this vector's value for one cell. Must be overridden.

        :param cell_index: The cell's flat index.
        :param workspace: The run's workspace.
        :returns: The cell's value.
        """
        raise NotImplementedError

    def __call__(self, model: "CompiledBlackOilModel", context: ScheduleContext) -> Number:
        """
        Reads this vector's value for the cell at `i`, `j`, `k`.

        :param model: The model to look the cell up in.
        :param context: The current moment's context, carrying the workspace.
        :returns: The cell's value.
        :raises SummaryError: If the cell is outside the model's grid, or the grid has no
            `i`, `j`, `k` layout.
        """
        dimensions = model.reservoir.grid.dimensions
        if dimensions is None:
            raise SummaryError(f"{self.mnemonic}: the model's grid has no `i`, `j`, `k` layout.")
        
        if not (
            0 <= self.i < dimensions.nx
            and 0 <= self.j < dimensions.ny
            and 0 <= self.k < dimensions.nz
        ):
            raise SummaryError(
                f"{self.key}: cell is outside the {dimensions.nx} x {dimensions.ny} x "
                f"{dimensions.nz} grid."
            )
        return self.get_value(
            cell_index=dimensions.flat_index(self.i, self.j, self.k),
            workspace=get_workspace(context=context),
        )


@attrs.frozen(kw_only=True, slots=True)
class BlockPressure(BlockVector):
    """Oil phase pressure in a cell (`BPR`, also written `BPRESSUR`)."""

    mnemonic = "BPR"

    def get_value(self, *, cell_index: Integer, workspace: "SimulationWorkspace") -> Number:
        return workspace.reservoir.pressure[cell_index]


@attrs.frozen(kw_only=True, slots=True)
class BlockGasSaturation(BlockVector):
    """Gas saturation in a cell (`BGSAT`, also written `BSGAS`)."""

    mnemonic = "BGSAT"

    def get_value(self, *, cell_index: Integer, workspace: "SimulationWorkspace") -> Number:
        return workspace.reservoir.gas_saturation[cell_index]


@attrs.frozen(kw_only=True, slots=True)
class BlockOilSaturation(BlockVector):
    """Oil saturation in a cell (`BOSAT`, also written `BSOIL`)."""

    mnemonic = "BOSAT"

    def get_value(self, *, cell_index: Integer, workspace: "SimulationWorkspace") -> Number:
        return workspace.reservoir.oil_saturation[cell_index]


@attrs.frozen(kw_only=True, slots=True)
class BlockWaterSaturation(BlockVector):
    """Water saturation in a cell (`BWSAT`, also written `BSWAT`)."""

    mnemonic = "BWSAT"

    def get_value(self, *, cell_index: Integer, workspace: "SimulationWorkspace") -> Number:
        return workspace.reservoir.water_saturation[cell_index]


@attrs.frozen(kw_only=True, slots=True)
class BlockGasOilRatio(BlockVector):
    """Gas dissolved in the oil of a cell (`BRS`), per unit of stock-tank oil."""

    mnemonic = "BRS"

    def get_value(self, *, cell_index: Integer, workspace: "SimulationWorkspace") -> Number:
        return workspace.reservoir.solution_gor[cell_index]


@attrs.frozen(kw_only=True, slots=True)
class BlockVaporizedOilGasRatio(BlockVector):
    """Oil vaporized in the gas of a cell (`BRV`), per unit of gas. `0` where the model doesn't track it."""

    mnemonic = "BRV"

    def get_value(self, *, cell_index: Integer, workspace: "SimulationWorkspace") -> Number:
        value = workspace.reservoir.vaporized_oil_gas_ratio[cell_index]
        return 0.0 if np.isnan(value) else value


@attrs.frozen(kw_only=True, slots=True)
class BlockOilRelativePermeability(BlockVector):
    """Oil relative permeability in a cell (`BOKR`)."""

    mnemonic = "BOKR"

    def get_value(self, *, cell_index: Integer, workspace: "SimulationWorkspace") -> Number:
        return workspace.physics.satfunc.oil_relative_permeability[cell_index]


@attrs.frozen(kw_only=True, slots=True)
class BlockWaterRelativePermeability(BlockVector):
    """Water relative permeability in a cell (`BWKR`)."""

    mnemonic = "BWKR"

    def get_value(self, *, cell_index: Integer, workspace: "SimulationWorkspace") -> Number:
        return workspace.physics.satfunc.water_relative_permeability[cell_index]


@attrs.frozen(kw_only=True, slots=True)
class BlockGasRelativePermeability(BlockVector):
    """Gas relative permeability in a cell (`BGKR`)."""

    mnemonic = "BGKR"

    def get_value(self, *, cell_index: Integer, workspace: "SimulationWorkspace") -> Number:
        return workspace.physics.satfunc.gas_relative_permeability[cell_index]


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
BPR = BlockPressure
BPRESSUR = BlockPressure
BGSAT = BlockGasSaturation
BSGAS = BlockGasSaturation
BOSAT = BlockOilSaturation
BSOIL = BlockOilSaturation
BWSAT = BlockWaterSaturation
BSWAT = BlockWaterSaturation
BRS = BlockGasOilRatio
BRV = BlockVaporizedOilGasRatio
BOKR = BlockOilRelativePermeability
BWKR = BlockWaterRelativePermeability
BGKR = BlockGasRelativePermeability
