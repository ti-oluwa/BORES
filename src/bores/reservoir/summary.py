"""
Summary vectors that report on the reservoir itself: the fluids in place, the average pressure
and pore volume, and the values in individual grid cells.

Vectors come in three scopes, named by their leading letter:

- Field vectors (`F...`) cover every cell, e.g. `FOIP` for the oil in place in the whole field.
- Region vectors (`R...`) cover one fluid-in-place region, e.g. `ROIP:2`. A model with no
  fluid-in-place regions has a single region, number 1, that covers every cell.
- Block vectors (`B...`) report one grid cell. They take the cell as 0-based `i`, `j`, `k`, like
  `Grid.flat_index`, and name it with 1-based indices in their report key, e.g. `"BPR:1,1,1"` for
  the first cell.

Every volume is in the model's own unit system and is stock-tank volume (standard volume for
gas), so FIELD reports cubic feet, not barrels. Pressures are oil phase pressures.
Every class has a short alias named after its deck mnemonic (`FOIP = FieldOilInPlace`).
"""

import enum
import typing

import attrs
import numpy as np

from bores.constants import c
from bores.errors import SummaryError
from bores.schedule.base import ScheduleContext
from bores.schedule.summary import Summary, get_workspace
from bores.types import BooleanCellArray, CellArray, Integer, Number, UnitSystem

if typing.TYPE_CHECKING:
    from bores.blackoil.compile import CompiledBlackOilModel
    from bores.simulation.workspace import SimulationWorkspace

__all__ = [
    "BGKR",
    "BGSAT",
    "BOKR",
    "BOSAT",
    "BPORV",
    "BPR",
    "BPRESSUR",
    "BRS",
    "BRV",
    "BSGAS",
    "BSOIL",
    "BSWAT",
    "BWKR",
    "BWSAT",
    "FGIP",
    "FGIPG",
    "FGIPL",
    "FOIP",
    "FOIPG",
    "FOIPL",
    "FPORV",
    "FPR",
    "FPRH",
    "FWIP",
    "RGIP",
    "RGIPG",
    "RGIPL",
    "ROIP",
    "ROIPG",
    "ROIPL",
    "RPORV",
    "RPR",
    "RPRH",
    "RWIP",
    "BlockGasOilRatio",
    "BlockGasRelativePermeability",
    "BlockGasSaturation",
    "BlockOilRelativePermeability",
    "BlockOilSaturation",
    "BlockPoreVolume",
    "BlockPressure",
    "BlockVaporizedOilGasRatio",
    "BlockVector",
    "BlockWaterRelativePermeability",
    "BlockWaterSaturation",
    "FieldAveragePressure",
    "FieldGasInPlace",
    "FieldGasInPlaceDissolved",
    "FieldGasInPlaceFree",
    "FieldHydrocarbonAveragePressure",
    "FieldOilInPlace",
    "FieldOilInPlaceInGas",
    "FieldOilInPlaceInOil",
    "FieldPoreVolume",
    "FieldWaterInPlace",
    "InPlaceQuantity",
    "PressureWeight",
    "RegionAveragePressure",
    "RegionGasInPlace",
    "RegionGasInPlaceDissolved",
    "RegionGasInPlaceFree",
    "RegionHydrocarbonAveragePressure",
    "RegionOilInPlace",
    "RegionOilInPlaceInGas",
    "RegionOilInPlaceInOil",
    "RegionPoreVolume",
    "RegionWaterInPlace",
    "compute_cell_in_place",
    "get_stock_tank_barrels_per_volume",
]


class InPlaceQuantity(enum.Enum):
    """A fluid volume that a cell holds, as stock-tank (or standard) volume."""

    OIL = "oil"
    """All stock-tank oil: the oil in the oil phase plus the oil vaporized in free gas."""

    OIL_IN_OIL_PHASE = "oil_in_oil_phase"
    """Stock-tank oil that sits in the oil phase."""

    OIL_IN_GAS_PHASE = "oil_in_gas_phase"
    """Stock-tank oil vaporized in free gas. Zero where the model has no vaporized oil-gas ratio."""

    GAS = "gas"
    """All gas: free gas, plus gas dissolved in the oil, plus gas dissolved in the water."""

    GAS_IN_LIQUID = "gas_in_liquid"
    """Gas dissolved in the oil and in the water."""

    GAS_FREE = "gas_free"
    """Free gas, the gas in the gas phase."""

    WATER = "water"
    """Stock-tank water."""


class PressureWeight(enum.Enum):
    """What an average pressure is weighted by."""

    PORE_VOLUME = "pore_volume"
    """Weight each cell's pressure by its pore volume."""

    HYDROCARBON_PORE_VOLUME = "hydrocarbon_pore_volume"
    """Weight each cell's pressure by the pore volume its oil and gas fill."""


def get_stock_tank_barrels_per_volume(unit_system: UnitSystem) -> Number:
    """
    Returns the factor between a volume in the model's own volume unit and stock-tank barrels.

    The gas-oil, oil-gas and gas-water ratios are per stock-tank barrel in FIELD units, while
    the volumes the vectors report are in cubic feet. Every other unit system states these
    ratios per unit of volume, so nothing needs converting.

    :param unit_system: The model's unit system.
    :returns: Stock-tank barrels per cubic foot for FIELD, `1` for every other unit system.
    """
    return c.CUBIC_FEET_TO_STB if unit_system == UnitSystem.FIELD else 1.0


def finite_or_zero(values: CellArray) -> CellArray:
    """
    Replaces the `NaN` entries of a per-cell ratio with zero.

    :param values: A per-cell ratio that is `NaN` where the model doesn't track it.
    :returns: `values`, with `NaN` read as `0`.
    """
    return typing.cast("CellArray", np.nan_to_num(values, nan=0.0))


def compute_cell_in_place(
    *,
    quantity: InPlaceQuantity,
    workspace: "SimulationWorkspace",
    pore_volumes: CellArray,
    unit_system: UnitSystem,
) -> CellArray:
    """
    Computes the volume of one fluid in every cell, at the pressures the run holds now.

    :param quantity: Which fluid volume to compute.
    :param workspace: The run's workspace.
    :param pore_volumes: Shape `(n_cells,)` pore volume of every cell at the current pressure.
    :param unit_system: The model's unit system.
    :returns: Shape `(n_cells,)` volume of `quantity` in every cell, in the model's volume unit.
    """
    state = workspace.reservoir
    pvt = workspace.physics.pvt
    barrels_per_volume = get_stock_tank_barrels_per_volume(unit_system)

    if quantity is InPlaceQuantity.WATER:
        return typing.cast(
            "CellArray",
            state.water_saturation * pore_volumes / pvt.water_formation_volume_factor,
        )

    oil_in_oil_phase = state.oil_saturation * pore_volumes / pvt.oil_formation_volume_factor
    if quantity is InPlaceQuantity.OIL_IN_OIL_PHASE:
        return typing.cast("CellArray", oil_in_oil_phase)

    free_gas = state.gas_saturation * pore_volumes / pvt.gas_formation_volume_factor
    if quantity is InPlaceQuantity.GAS_FREE:
        return typing.cast("CellArray", free_gas)

    if quantity is InPlaceQuantity.OIL_IN_GAS_PHASE:
        vaporized_oil = finite_or_zero(pvt.vaporized_oil_gas_ratio) * free_gas / barrels_per_volume
        return typing.cast("CellArray", vaporized_oil)

    if quantity is InPlaceQuantity.OIL:
        vaporized_oil = finite_or_zero(pvt.vaporized_oil_gas_ratio) * free_gas / barrels_per_volume
        return typing.cast("CellArray", oil_in_oil_phase + vaporized_oil)

    water_volume = state.water_saturation * pore_volumes / pvt.water_formation_volume_factor
    gas_in_oil = state.solution_gor * oil_in_oil_phase * barrels_per_volume
    gas_in_water = (
        finite_or_zero(state.gas_solubility_in_water) * water_volume * barrels_per_volume
    )
    if quantity is InPlaceQuantity.GAS_IN_LIQUID:
        return typing.cast("CellArray", gas_in_oil + gas_in_water)
    return typing.cast("CellArray", free_gas + gas_in_oil + gas_in_water)


def compute_average_pressure(
    *,
    weight: PressureWeight,
    cell_mask: BooleanCellArray,
    pore_volumes: CellArray,
    workspace: "SimulationWorkspace",
) -> Number:
    """
    Computes the weighted average oil phase pressure over a set of cells.

    :param weight: What each cell's pressure is weighted by.
    :param cell_mask: Shape `(n_cells,)` boolean mask of the cells to average over.
    :param pore_volumes: Shape `(n_cells,)` pore volume of every cell at the current pressure.
    :param workspace: The run's workspace.
    :returns: The weighted average pressure, or `NaN` when the cells carry no weight (for example
        a hydrocarbon-weighted average over cells that hold no oil or gas).
    """
    state = workspace.reservoir
    if weight is PressureWeight.HYDROCARBON_PORE_VOLUME:
        weights = (state.oil_saturation + state.gas_saturation) * pore_volumes
    else:
        weights = pore_volumes
    total_weight = weights[cell_mask].sum()
    if total_weight <= 0.0:
        return float("nan")
    return (state.pressure[cell_mask] * weights[cell_mask]).sum() / total_weight


@attrs.frozen(kw_only=True, slots=True)
class FieldVector(Summary["CompiledBlackOilModel"]):
    """
    Base for a vector over every cell of the field.

    Subclasses set `mnemonic` and implement `get_value`.
    """

    mnemonic: typing.ClassVar[str] = ""

    @property
    def key(self) -> str:
        """
        Returns this vector's report key.

        :returns: The mnemonic, e.g. `"FOIP"`.
        """
        return self.mnemonic

    def get_value(
        self,
        *,
        cell_mask: BooleanCellArray,
        pore_volumes: CellArray,
        workspace: "SimulationWorkspace",
        unit_system: UnitSystem,
    ) -> Number:
        """
        Computes this vector's value over a set of cells. Must be overridden.

        :param cell_mask: Shape `(n_cells,)` boolean mask of the cells to report on.
        :param pore_volumes: Shape `(n_cells,)` pore volume of every cell at the current pressure.
        :param workspace: The run's workspace.
        :param unit_system: The model's unit system.
        :returns: The vector's value.
        """
        raise NotImplementedError

    def __call__(self, model: "CompiledBlackOilModel", context: ScheduleContext) -> Number:
        """
        Computes this vector's value over every cell.

        :param model: The model to read the pore volumes from.
        :param context: The current moment's context, carrying the workspace.
        :returns: The vector's value, at the pore volume the run holds now.
        """
        workspace = get_workspace(context=context)
        pore_volumes = model.reservoir.compute_pore_volumes(workspace.reservoir.pressure)
        return self.get_value(
            cell_mask=np.ones(pore_volumes.shape, dtype=bool),
            pore_volumes=pore_volumes,
            workspace=workspace,
            unit_system=model.unit_system,
        )


@attrs.frozen(kw_only=True, slots=True)
class RegionVector(Summary["CompiledBlackOilModel"]):
    """
    Base for a vector over the cells of one fluid-in-place region.

    Subclasses set `mnemonic` and implement `get_value`. A model with no fluid-in-place regions
    has a single region, number 1, covering every cell.
    """

    mnemonic: typing.ClassVar[str] = ""

    region: Integer = 1
    """The fluid-in-place region number to report on."""

    @property
    def key(self) -> str:
        """
        Returns this vector's report key.

        :returns: The mnemonic and region number, e.g. `"ROIP:2"`.
        """
        return f"{self.mnemonic}:{self.region}"

    def get_value(
        self,
        *,
        cell_mask: BooleanCellArray,
        pore_volumes: CellArray,
        workspace: "SimulationWorkspace",
        unit_system: UnitSystem,
    ) -> Number:
        """
        Computes this vector's value over the cells of the region. Must be overridden.

        :param cell_mask: Shape `(n_cells,)` boolean mask of the region's cells.
        :param pore_volumes: Shape `(n_cells,)` pore volume of every cell at the current pressure.
        :param workspace: The run's workspace.
        :param unit_system: The model's unit system.
        :returns: The vector's value.
        """
        raise NotImplementedError

    def __call__(self, model: "CompiledBlackOilModel", context: ScheduleContext) -> Number:
        """
        Computes this vector's value over the cells of `region`.

        :param model: The model to read the regions and pore volumes from.
        :param context: The current moment's context, carrying the workspace.
        :returns: The vector's value, at the pore volume the run holds now.
        :raises SummaryError: If the region has no cells.
        """
        workspace = get_workspace(context=context)
        reservoir = model.reservoir
        pore_volumes = reservoir.compute_pore_volumes(workspace.reservoir.pressure)
        fluid_in_place_regions = (
            reservoir.regions.fluid_in_place_region if reservoir.regions is not None else None
        )
        if fluid_in_place_regions is None:
            cell_mask = np.full(pore_volumes.shape, self.region == 1)
        else:
            cell_mask = np.asarray(fluid_in_place_regions) == self.region

        if not cell_mask.any():
            raise SummaryError(
                f"{self.mnemonic}: fluid-in-place region {self.region} has no cells."
            )
        return self.get_value(
            cell_mask=cell_mask,
            pore_volumes=pore_volumes,
            workspace=workspace,
            unit_system=model.unit_system,
        )


@attrs.frozen(kw_only=True, slots=True)
class FieldInPlace(FieldVector):
    """Base for the volume of one fluid in the whole field. Subclasses set `quantity`."""

    quantity: typing.ClassVar[InPlaceQuantity]

    def get_value(
        self,
        *,
        cell_mask: BooleanCellArray,
        pore_volumes: CellArray,
        workspace: "SimulationWorkspace",
        unit_system: UnitSystem,
    ) -> Number:
        cell_volumes = compute_cell_in_place(
            quantity=self.quantity,
            workspace=workspace,
            pore_volumes=pore_volumes,
            unit_system=unit_system,
        )
        return cell_volumes[cell_mask].sum()


@attrs.frozen(kw_only=True, slots=True)
class RegionInPlace(RegionVector):
    """Base for the volume of one fluid in a fluid-in-place region. Subclasses set `quantity`."""

    quantity: typing.ClassVar[InPlaceQuantity]

    def get_value(
        self,
        *,
        cell_mask: BooleanCellArray,
        pore_volumes: CellArray,
        workspace: "SimulationWorkspace",
        unit_system: UnitSystem,
    ) -> Number:
        cell_volumes = compute_cell_in_place(
            quantity=self.quantity,
            workspace=workspace,
            pore_volumes=pore_volumes,
            unit_system=unit_system,
        )
        return cell_volumes[cell_mask].sum()


@attrs.frozen(kw_only=True, slots=True)
class FieldAveragePressure(FieldVector):
    """Average oil phase pressure of the whole field, weighted by pore volume (`FPR`)."""

    mnemonic = "FPR"
    weight: typing.ClassVar[PressureWeight] = PressureWeight.PORE_VOLUME

    def get_value(
        self,
        *,
        cell_mask: BooleanCellArray,
        pore_volumes: CellArray,
        workspace: "SimulationWorkspace",
        unit_system: UnitSystem,
    ) -> Number:
        return compute_average_pressure(
            weight=self.weight,
            cell_mask=cell_mask,
            pore_volumes=pore_volumes,
            workspace=workspace,
        )


@attrs.frozen(kw_only=True, slots=True)
class FieldHydrocarbonAveragePressure(FieldAveragePressure):
    """Average oil phase pressure of the whole field, weighted by hydrocarbon pore volume (`FPRH`)."""

    mnemonic = "FPRH"
    weight: typing.ClassVar[PressureWeight] = PressureWeight.HYDROCARBON_PORE_VOLUME


@attrs.frozen(kw_only=True, slots=True)
class RegionAveragePressure(RegionVector):
    """Average oil phase pressure of a fluid-in-place region, weighted by pore volume (`RPR`)."""

    mnemonic = "RPR"
    weight: typing.ClassVar[PressureWeight] = PressureWeight.PORE_VOLUME

    def get_value(
        self,
        *,
        cell_mask: BooleanCellArray,
        pore_volumes: CellArray,
        workspace: "SimulationWorkspace",
        unit_system: UnitSystem,
    ) -> Number:
        return compute_average_pressure(
            weight=self.weight,
            cell_mask=cell_mask,
            pore_volumes=pore_volumes,
            workspace=workspace,
        )


@attrs.frozen(kw_only=True, slots=True)
class RegionHydrocarbonAveragePressure(RegionAveragePressure):
    """
    Average oil phase pressure of a fluid-in-place region, weighted by hydrocarbon pore volume
    (`RPRH`).
    """

    mnemonic = "RPRH"
    weight: typing.ClassVar[PressureWeight] = PressureWeight.HYDROCARBON_PORE_VOLUME


@attrs.frozen(kw_only=True, slots=True)
class FieldPoreVolume(FieldVector):
    """Pore volume of the whole field at the pressures the run holds now (`FPORV`)."""

    mnemonic = "FPORV"

    def get_value(
        self,
        *,
        cell_mask: BooleanCellArray,
        pore_volumes: CellArray,
        workspace: "SimulationWorkspace",
        unit_system: UnitSystem,
    ) -> Number:
        return pore_volumes[cell_mask].sum()


@attrs.frozen(kw_only=True, slots=True)
class RegionPoreVolume(RegionVector):
    """Pore volume of a fluid-in-place region at the pressures the run holds now (`RPORV`)."""

    mnemonic = "RPORV"

    def get_value(
        self,
        *,
        cell_mask: BooleanCellArray,
        pore_volumes: CellArray,
        workspace: "SimulationWorkspace",
        unit_system: UnitSystem,
    ) -> Number:
        return pore_volumes[cell_mask].sum()


@attrs.frozen(kw_only=True, slots=True)
class FieldOilInPlace(FieldInPlace):
    """
    Stock-tank oil in the whole field (`FOIP`).

    Counts the oil in the oil phase plus the oil vaporized in free gas, when the model has a
    vaporized oil-gas ratio.
    """

    mnemonic = "FOIP"
    quantity = InPlaceQuantity.OIL


@attrs.frozen(kw_only=True, slots=True)
class FieldOilInPlaceInOil(FieldInPlace):
    """Stock-tank oil in the whole field that sits in the oil phase (`FOIPL`)."""

    mnemonic = "FOIPL"
    quantity = InPlaceQuantity.OIL_IN_OIL_PHASE


@attrs.frozen(kw_only=True, slots=True)
class FieldOilInPlaceInGas(FieldInPlace):
    """Stock-tank oil in the whole field that is vaporized in free gas (`FOIPG`)."""

    mnemonic = "FOIPG"
    quantity = InPlaceQuantity.OIL_IN_GAS_PHASE


@attrs.frozen(kw_only=True, slots=True)
class FieldGasInPlace(FieldInPlace):
    """
    Gas in the whole field (`FGIP`): free gas, plus gas dissolved in the oil, plus gas dissolved
    in the water when the model has a gas solubility in water.
    """

    mnemonic = "FGIP"
    quantity = InPlaceQuantity.GAS


@attrs.frozen(kw_only=True, slots=True)
class FieldGasInPlaceDissolved(FieldInPlace):
    """Gas in the whole field that is dissolved in the oil and the water (`FGIPL`)."""

    mnemonic = "FGIPL"
    quantity = InPlaceQuantity.GAS_IN_LIQUID


@attrs.frozen(kw_only=True, slots=True)
class FieldGasInPlaceFree(FieldInPlace):
    """Free gas in the whole field (`FGIPG`)."""

    mnemonic = "FGIPG"
    quantity = InPlaceQuantity.GAS_FREE


@attrs.frozen(kw_only=True, slots=True)
class FieldWaterInPlace(FieldInPlace):
    """Stock-tank water in the whole field (`FWIP`)."""

    mnemonic = "FWIP"
    quantity = InPlaceQuantity.WATER


@attrs.frozen(kw_only=True, slots=True)
class RegionOilInPlace(RegionInPlace):
    """
    Stock-tank oil in a fluid-in-place region (`ROIP`).

    Counts the oil in the oil phase plus the oil vaporized in free gas, when the model has a
    vaporized oil-gas ratio.
    """

    mnemonic = "ROIP"
    quantity = InPlaceQuantity.OIL


@attrs.frozen(kw_only=True, slots=True)
class RegionOilInPlaceInOil(RegionInPlace):
    """Stock-tank oil in a fluid-in-place region that sits in the oil phase (`ROIPL`)."""

    mnemonic = "ROIPL"
    quantity = InPlaceQuantity.OIL_IN_OIL_PHASE


@attrs.frozen(kw_only=True, slots=True)
class RegionOilInPlaceInGas(RegionInPlace):
    """Stock-tank oil in a fluid-in-place region that is vaporized in free gas (`ROIPG`)."""

    mnemonic = "ROIPG"
    quantity = InPlaceQuantity.OIL_IN_GAS_PHASE


@attrs.frozen(kw_only=True, slots=True)
class RegionGasInPlace(RegionInPlace):
    """
    Gas in a fluid-in-place region (`RGIP`): free gas, plus gas dissolved in the oil, plus gas
    dissolved in the water when the model has a gas solubility in water.
    """

    mnemonic = "RGIP"
    quantity = InPlaceQuantity.GAS


@attrs.frozen(kw_only=True, slots=True)
class RegionGasInPlaceDissolved(RegionInPlace):
    """Gas in a fluid-in-place region that is dissolved in the oil and the water (`RGIPL`)."""

    mnemonic = "RGIPL"
    quantity = InPlaceQuantity.GAS_IN_LIQUID


@attrs.frozen(kw_only=True, slots=True)
class RegionGasInPlaceFree(RegionInPlace):
    """Free gas in a fluid-in-place region (`RGIPG`)."""

    mnemonic = "RGIPG"
    quantity = InPlaceQuantity.GAS_FREE


@attrs.frozen(kw_only=True, slots=True)
class RegionWaterInPlace(RegionInPlace):
    """Stock-tank water in a fluid-in-place region (`RWIP`)."""

    mnemonic = "RWIP"
    quantity = InPlaceQuantity.WATER


@attrs.frozen(kw_only=True, slots=True)
class BlockVector(Summary["CompiledBlackOilModel"]):
    """Base for a per-cell vector. Subclasses set `mnemonic` and implement `get_value`."""

    mnemonic: typing.ClassVar[str] = ""

    i: Integer
    """The cell's 0-based x index."""

    j: Integer
    """The cell's 0-based y index."""

    k: Integer
    """The cell's 0-based z index."""

    @property
    def key(self) -> str:
        """
        Returns this vector's report key.

        :returns: The mnemonic and the cell's 1-based indices, e.g. `"BPR:1,1,1"`.
        """
        return f"{self.mnemonic}:{self.i + 1},{self.j + 1},{self.k + 1}"

    def get_value(
        self,
        *,
        cell_index: Integer,
        workspace: "SimulationWorkspace",
        model: "CompiledBlackOilModel",
    ) -> Number:
        """
        Reads this vector's value for one cell. Must be overridden.

        :param cell_index: The cell's flat index.
        :param workspace: The run's workspace.
        :param model: The model the cell belongs to.
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
            model=model,
        )


@attrs.frozen(kw_only=True, slots=True)
class BlockPressure(BlockVector):
    """Oil phase pressure in a cell (`BPR`, also written `BPRESSUR`)."""

    mnemonic = "BPR"

    def get_value(
        self,
        *,
        cell_index: Integer,
        workspace: "SimulationWorkspace",
        model: "CompiledBlackOilModel",
    ) -> Number:
        return workspace.reservoir.pressure[cell_index]


@attrs.frozen(kw_only=True, slots=True)
class BlockPoreVolume(BlockVector):
    """Pore volume of a cell at the pressure the run holds now (`BPORV`)."""

    mnemonic = "BPORV"

    def get_value(
        self,
        *,
        cell_index: Integer,
        workspace: "SimulationWorkspace",
        model: "CompiledBlackOilModel",
    ) -> Number:
        pore_volumes = model.reservoir.compute_pore_volumes(workspace.reservoir.pressure)
        return pore_volumes[cell_index]


@attrs.frozen(kw_only=True, slots=True)
class BlockGasSaturation(BlockVector):
    """Gas saturation in a cell (`BGSAT`, also written `BSGAS`)."""

    mnemonic = "BGSAT"

    def get_value(
        self,
        *,
        cell_index: Integer,
        workspace: "SimulationWorkspace",
        model: "CompiledBlackOilModel",
    ) -> Number:
        return workspace.reservoir.gas_saturation[cell_index]


@attrs.frozen(kw_only=True, slots=True)
class BlockOilSaturation(BlockVector):
    """Oil saturation in a cell (`BOSAT`, also written `BSOIL`)."""

    mnemonic = "BOSAT"

    def get_value(
        self,
        *,
        cell_index: Integer,
        workspace: "SimulationWorkspace",
        model: "CompiledBlackOilModel",
    ) -> Number:
        return workspace.reservoir.oil_saturation[cell_index]


@attrs.frozen(kw_only=True, slots=True)
class BlockWaterSaturation(BlockVector):
    """Water saturation in a cell (`BWSAT`, also written `BSWAT`)."""

    mnemonic = "BWSAT"

    def get_value(
        self,
        *,
        cell_index: Integer,
        workspace: "SimulationWorkspace",
        model: "CompiledBlackOilModel",
    ) -> Number:
        return workspace.reservoir.water_saturation[cell_index]


@attrs.frozen(kw_only=True, slots=True)
class BlockGasOilRatio(BlockVector):
    """Gas dissolved in the oil of a cell (`BRS`), per unit of stock-tank oil."""

    mnemonic = "BRS"

    def get_value(
        self,
        *,
        cell_index: Integer,
        workspace: "SimulationWorkspace",
        model: "CompiledBlackOilModel",
    ) -> Number:
        return workspace.reservoir.solution_gor[cell_index]


@attrs.frozen(kw_only=True, slots=True)
class BlockVaporizedOilGasRatio(BlockVector):
    """Oil vaporized in the gas of a cell (`BRV`), per unit of gas. `0` where the model doesn't track it."""

    mnemonic = "BRV"

    def get_value(
        self,
        *,
        cell_index: Integer,
        workspace: "SimulationWorkspace",
        model: "CompiledBlackOilModel",
    ) -> Number:
        value = workspace.reservoir.vaporized_oil_gas_ratio[cell_index]
        return 0.0 if np.isnan(value) else value


@attrs.frozen(kw_only=True, slots=True)
class BlockOilRelativePermeability(BlockVector):
    """Oil relative permeability in a cell (`BOKR`)."""

    mnemonic = "BOKR"

    def get_value(
        self,
        *,
        cell_index: Integer,
        workspace: "SimulationWorkspace",
        model: "CompiledBlackOilModel",
    ) -> Number:
        return workspace.physics.satfunc.oil_relative_permeability[cell_index]


@attrs.frozen(kw_only=True, slots=True)
class BlockWaterRelativePermeability(BlockVector):
    """Water relative permeability in a cell (`BWKR`)."""

    mnemonic = "BWKR"

    def get_value(
        self,
        *,
        cell_index: Integer,
        workspace: "SimulationWorkspace",
        model: "CompiledBlackOilModel",
    ) -> Number:
        return workspace.physics.satfunc.water_relative_permeability[cell_index]


@attrs.frozen(kw_only=True, slots=True)
class BlockGasRelativePermeability(BlockVector):
    """Gas relative permeability in a cell (`BGKR`)."""

    mnemonic = "BGKR"

    def get_value(
        self,
        *,
        cell_index: Integer,
        workspace: "SimulationWorkspace",
        model: "CompiledBlackOilModel",
    ) -> Number:
        return workspace.physics.satfunc.gas_relative_permeability[cell_index]


FPR = FieldAveragePressure
FPRH = FieldHydrocarbonAveragePressure
FPORV = FieldPoreVolume
FOIP = FieldOilInPlace
FOIPL = FieldOilInPlaceInOil
FOIPG = FieldOilInPlaceInGas
FGIP = FieldGasInPlace
FGIPL = FieldGasInPlaceDissolved
FGIPG = FieldGasInPlaceFree
FWIP = FieldWaterInPlace
RPR = RegionAveragePressure
RPRH = RegionHydrocarbonAveragePressure
RPORV = RegionPoreVolume
ROIP = RegionOilInPlace
ROIPL = RegionOilInPlaceInOil
ROIPG = RegionOilInPlaceInGas
RGIP = RegionGasInPlace
RGIPL = RegionGasInPlaceDissolved
RGIPG = RegionGasInPlaceFree
RWIP = RegionWaterInPlace
BPORV = BlockPoreVolume
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
