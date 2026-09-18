"""A black-oil simulation case: a model, its initial state, and everything needed to run it."""

import typing

import attrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.blackoil.compile import CompiledBlackOilModel, compile_model
from bores.blackoil.fluids.model import BlackOil
from bores.blackoil.model import BlackOilModel
from bores.blackoil.pvt.regions import PVT
from bores.blackoil.satfunc.regions import SatFunc
from bores.blackoil.satfunc.relperm.tables import MinimumRelPerm
from bores.deck.file import DeckFile
from bores.errors import CaseLoadError, CaseValidationError
from bores.grids.base import Grid
from bores.initialization import N_SATURATION_SAMPLES, initialize_reservoir_state
from bores.precision import get_dtype
from bores.reservoir.model import Reservoir
from bores.reservoir.regions import Regions
from bores.reservoir.rock.base import Rock
from bores.reservoir.state import Equilibrium, ReservoirState
from bores.reservoir.state.base import Hysteresis
from bores.reservoir.temperature import Temperature
from bores.schedule.base import Schedule
from bores.serde.base import Serializable
from bores.simulation.spec import RunSpec
from bores.simulation.workspace import SimulationWorkspace, build_simulation_workspace
from bores.types import CellArray, InterpolationMethod, Number, UnitSystem
from bores.wells.deck import load_schedule
from bores.wells.hydraulics.base import WellBoreModel
from bores.wells.model import WellSystem

__all__ = ["SimulationCase", "load_case"]


@attrs.define(kw_only=True, slots=True, frozen=True)
class SimulationCase(Serializable):
    """A black-oil simulation case."""

    model: CompiledBlackOilModel
    """The black-oil model this case runs against."""

    initial_state: ReservoirState
    """The reservoir's state at the start of the run."""

    runspec: RunSpec = attrs.field(factory=RunSpec)
    """Run configuration - unit system, constants, timer, start date, resolver tuning."""

    schedule: Schedule[CompiledBlackOilModel] = attrs.field(factory=Schedule)
    """Every scheduled well/model edit for the run."""

    summary: typing.Any = None
    """Summary-vector request/output configuration. Not built out yet."""

    salinity: CellArray | None = None
    """
    Optional salinity values for the reservoir cells.
    
    Leaves `None` if salinity is not used in the simulation. If provided, 
    it must be a 1D array of length equal to the number of cells in the reservoir grid, 
    representing the salinity in each cell.
    """

    hysteresis: Hysteresis | None = None
    """
    Optional hysteresis configuration for the simulation. 

    Only used if `RunSpec.enable_hysteresis` is `True`. If so, the case's initial state
    must have been initialized with `hysteresis_enabled=True` and `saturation_samples>0`.
    If `hysteresis` is `None`, the case's initial state's hysteresis configuration is used.
    """

    dtype: npt.DTypeLike = attrs.field(factory=get_dtype)
    """Array dtype for every buffer. `bores.precision.get_dtype()` if not provided"""

    _workspace: SimulationWorkspace | None = attrs.field(default=None, init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        if self.model.unit_system != self.runspec.unit_system:
            raise CaseValidationError(
                "Simulation case's model and runspec must share the same unit system."
            )
        if self.model.unit_system != self.initial_state.unit_system:
            raise CaseValidationError(
                "Simulation case's model and initial state must share the same unit system."
            )

    @property
    def unit_system(self) -> UnitSystem:
        """
        The simulation case's unit system.

        :returns: The case's unit system.
        """
        return self.model.unit_system

    @property
    def workspace(self) -> SimulationWorkspace:
        """
        The simulation case's workspace, building it on first access.

        :returns: The case's `SimulationWorkspace`.
        """
        if self._workspace is None:
            object.__setattr__(self, "_workspace", self.build_workspace(dtype=self.dtype))
        return typing.cast(SimulationWorkspace, self._workspace)

    def build_workspace(self, *, dtype: npt.DTypeLike = None) -> SimulationWorkspace:
        dtype = np.dtype(dtype) if dtype is not None else self.dtype
        model = self.model
        wells = model.wells
        n_wells = len(wells.names) if wells is not None else 0
        n_connections = wells.perforations.well_offsets[-1] if wells is not None else 0
        regions = model.reservoir.regions
        initial_state = self.initial_state
        if regions is None:
            regions = Regions()

        hysteresis = self.hysteresis
        hysteresis_enabled = self.runspec.hysteresis_enabled
        if not hysteresis_enabled:
            hysteresis = None
        elif hysteresis_enabled and hysteresis is None:
            if initial_state.hysteresis is not None:
                hysteresis = initial_state.hysteresis
            else:
                hysteresis = Hysteresis.from_saturation(
                    water_saturation=initial_state.water_saturation,
                    gas_saturation=initial_state.gas_saturation,
                    dtype=dtype,
                )

        return build_simulation_workspace(
            reservoir=model.reservoir,
            regions=regions,
            fluid=model.fluid,
            initial_state=initial_state,
            n_wells=n_wells,
            n_connections=n_connections,
            runspec=self.runspec,
            hysteresis=hysteresis,
            salinity=self.salinity,
            dtype=dtype,
        )

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        default_wellbore: WellBoreModel,
        temperature: Temperature | Number | None = None,
        mixing_rule: str = "eclipse_rule",
        compiled_at: float = 0.0,
        runspec: RunSpec | None = None,
        summary: typing.Any = None,
        min_wetting_relperm: MinimumRelPerm = None,
        min_non_wetting_relperm: MinimumRelPerm = None,
        include_capillary_pressure: bool = True,
        hysteresis_enabled: bool | None = None,
        saturation_samples: int = N_SATURATION_SAMPLES,
        interpolation_method: InterpolationMethod = "linear",
        unit_system: UnitSystem | None = None,
        dtype: npt.DTypeLike = None,
    ) -> Self:
        """
        Loads a `SimulationCase` from a deck. See `load_case`.

        :param deck_file: The deck to load from.
        :param default_wellbore: Wellbore hydraulics model for every well
            that doesn't override it. Deck loading has no sensible universal
            default (tubing diameter is deck-specific), so this is required.
        :param temperature: Reservoir temperature, either a `Temperature` or
            a constant value in the deck's own unit system.
        :param mixing_rule: Three-phase relative permeability mixing rule for `SatFunc.from_deck`.
        :param compiled_at: The point on the schedule clock this case's
            model reflects. Forwarded to `bores.wells.deck.load_schedule`.
            Only schedule events strictly after this remain as actions.
        :param runspec: Run configuration. `RunSpec`'s own defaults if not given.
        :param summary: Summary-vector request/output configuration. Not built out yet.
        :param min_wetting_relperm: Minimum wetting-phase relative permeability for `SatFunc.from_deck`.
        :param min_non_wetting_relperm: Minimum non-wetting-phase relative permeability for `SatFunc.from_deck`.
        :param include_capillary_pressure: Whether to include capillary pressure in the saturation functions.
            Forwarded to `SatFunc.from_deck`.
        :param hysteresis_enabled: Whether to include hysteresis in the initial state.
            Forwarded to `initialize_reservoir_state`.
        :param saturation_samples: Number of saturation samples for the initial state.
            Forwarded to `initialize_reservoir_state`.
        :param interpolation_method: Interpolation method for `Rock.from_deck` and `PVT.from_deck`.
        :param unit_system: Unit system for the case. Defaults to the deck's unit system.
        :param dtype: Array dtype for every buffer. Defaults to `bores.precision.get_dtype()`.
        :returns: The loaded `SimulationCase`.
        """
        case = load_case(
            deck_file,
            default_wellbore=default_wellbore,
            temperature=temperature,
            mixing_rule=mixing_rule,
            compiled_at=compiled_at,
            runspec=runspec,
            summary=summary,
            min_wetting_relperm=min_wetting_relperm,
            min_non_wetting_relperm=min_non_wetting_relperm,
            include_capillary_pressure=include_capillary_pressure,
            hysteresis_enabled=hysteresis_enabled,
            saturation_samples=saturation_samples,
            interpolation_method=interpolation_method,
            unit_system=unit_system,
            dtype=dtype,
        )
        return typing.cast(Self, case)


def load_case(
    deck_file: DeckFile,
    *,
    default_wellbore: WellBoreModel,
    temperature: Temperature | Number | None = None,
    mixing_rule: str = "eclipse_rule",
    compiled_at: float = 0.0,
    runspec: RunSpec | None = None,
    summary: typing.Any = None,
    min_wetting_relperm: MinimumRelPerm = None,
    min_non_wetting_relperm: MinimumRelPerm = None,
    include_capillary_pressure: bool = True,
    hysteresis_enabled: bool | None = None,
    saturation_samples: int = N_SATURATION_SAMPLES,
    interpolation_method: InterpolationMethod = "linear",
    unit_system: UnitSystem | None = None,
    dtype: npt.DTypeLike = None,
) -> SimulationCase:
    """
    Loads a full `SimulationCase` from a deck: grid, rock, fluid, initial
    state, wells, and schedule.

    :param deck_file: The deck to load from.
    :param default_wellbore: Wellbore hydraulics model for every well
        that doesn't override it. Deck loading has no sensible universal
        default (tubing diameter is deck-specific), so this is required.
    :param temperature: Reservoir temperature, either a `Temperature` or
        a constant value in the deck's own unit system.
    :param mixing_rule: Three-phase relative permeability mixing rule for `SatFunc.from_deck`.
    :param compiled_at: The point on the schedule clock this case's
        model reflects. Forwarded to `bores.wells.deck.load_schedule`.
        Only schedule events strictly after this remain as actions.
    :param runspec: Run configuration. `RunSpec`'s own defaults if not given.
    :param summary: Summary-vector request/output configuration. Not built out yet.
    :param min_wetting_relperm: Minimum wetting-phase relative permeability for `SatFunc.from_deck`.
    :param min_non_wetting_relperm: Minimum non-wetting-phase relative permeability for `SatFunc.from_deck`.
    :param include_capillary_pressure: Whether to include capillary pressure in the saturation functions.
        Forwarded to `SatFunc.from_deck`.
    :param hysteresis_enabled: Whether to include hysteresis in the initial state.
        Forwarded to `initialize_reservoir_state`.
    :param saturation_samples: Number of saturation samples for the initial state.
        Forwarded to `initialize_reservoir_state`.
    :param interpolation_method: Interpolation method for `Rock.from_deck` and `PVT.from_deck`.
    :param unit_system: Unit system for the case. Defaults to the deck's unit system.
    :param dtype: Array dtype for every buffer. Defaults to `bores.precision.get_dtype()`.
    :returns: The loaded `SimulationCase`.
    :raises CaseLoadError: If any loading stage fails. The
        original exception is chained as the cause.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    unit_system = unit_system if unit_system is not None else deck_file.unit_system
    runspec = (
        runspec.convert(unit_system) if runspec is not None else RunSpec(unit_system=unit_system)
    )
    if temperature is None:
        temperature = Temperature.from_deck(deck_file, dtype=dtype)
    else:
        temperature = (
            temperature.convert(unit_system)
            if isinstance(temperature, Temperature)
            else Temperature(temperature, unit_system=unit_system)
        )

    try:
        grid = Grid.from_deck(deck_file)
    except Exception as exc:
        raise CaseLoadError(f"Failed to load grid from {deck_file!r}.") from exc

    try:
        regions = Regions.from_deck(deck_file, n_cells=grid.n_cells, use_default=True)
    except Exception as exc:
        raise CaseLoadError(f"Failed to load regions from {deck_file!r}.") from exc

    try:
        satfunc = SatFunc.from_deck(
            deck_file,
            mixing_rule=mixing_rule,
            min_wetting_relperm=min_wetting_relperm,
            min_non_wetting_relperm=min_non_wetting_relperm,
            include_capillary_pressure=include_capillary_pressure,
            dtype=dtype,
        )
    except Exception as exc:
        raise CaseLoadError(f"Failed to load saturation functions from {deck_file!r}.") from exc

    try:
        rock = Rock.from_deck(
            deck_file,
            grid=grid,
            rock_region=regions.rock_region,
            satfunc=satfunc,
            saturation_region=regions.saturation_region,
            interpolation_method=interpolation_method,
            dtype=dtype,
        )
    except Exception as exc:
        raise CaseLoadError(f"Failed to load rock properties from {deck_file!r}.") from exc

    reservoir = Reservoir(grid=grid, rock=rock, regions=regions)

    try:
        pvt = PVT.from_deck(
            deck_file,
            temperature=temperature,
            interpolation_method=interpolation_method,
        )
    except Exception as exc:
        raise CaseLoadError(f"Failed to load PVT tables from {deck_file!r}.") from exc

    fluid = BlackOil(pvt=pvt, satfunc=satfunc)

    try:
        equilibrium = Equilibrium.from_deck(deck_file)
        initial_state = initialize_reservoir_state(
            reservoir=reservoir,
            pvt=pvt,
            deck_file=deck_file,
            equilibrium=equilibrium,
            satfunc=satfunc,
            temperature=temperature,
            hysteresis_enabled=hysteresis_enabled
            if hysteresis_enabled is not None
            else runspec.hysteresis_enabled,
            saturation_samples=saturation_samples,
            dtype=dtype,
        )
    except Exception as exc:
        raise CaseLoadError(f"Failed to initialize reservoir state from {deck_file!r}.") from exc

    try:
        wells = WellSystem.from_deck(deck_file, grid=grid, default_wellbore=default_wellbore)
    except Exception as exc:
        raise CaseLoadError(f"Failed to load wells from {deck_file!r}.") from exc

    model = BlackOilModel(reservoir=reservoir, fluid=fluid, wells=wells, unit_system=unit_system)

    try:
        schedule = load_schedule(deck_file, compiled_at=compiled_at)
    except Exception as exc:
        raise CaseLoadError(f"Failed to load schedule from {deck_file!r}.") from exc

    return SimulationCase(
        model=compile_model(model, dtype=dtype),
        initial_state=initial_state,
        runspec=runspec,
        schedule=schedule,
        summary=summary,
        dtype=dtype,
    )
