"""A black-oil simulation case: a model, its initial state, and everything needed to run it."""

import typing

import attrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.blackoil.compile import CompiledBlackOilModel, compile_model
from bores.blackoil.fluids.model import BlackOil
from bores.blackoil.model import BlackOilModel
from bores.blackoil.pvt.regions import OIL_PVT_KEYWORDS, PVT
from bores.blackoil.satfunc.regions import SATFUNC_KEYWORDS, SatFunc
from bores.blackoil.satfunc.relperm.tables import MinimumRelPerm
from bores.deck.file import DeckFile
from bores.deck.keywords.runspec import START
from bores.deck.precheck import find_missing
from bores.errors import CaseLoadError, CaseValidationError, ValidationError
from bores.grids.base import CellStatus, Grid
from bores.grids.factories.corner_point import InvertedCellPolicy
from bores.grids.io.grdecl import grid_geometry_requirements
from bores.initialization import (
    N_SATURATION_SAMPLES,
    compute_water_phase_pressure,
    initialize_reservoir_state,
)
from bores.precision import get_dtype
from bores.reservoir.boundary.conditions import BoundaryConditions
from bores.reservoir.boundary.deck import load_boundary_conditions
from bores.reservoir.equilibrium import Equilibrium
from bores.reservoir.model import Reservoir
from bores.reservoir.regions import Regions
from bores.reservoir.rock.model import ROCK_REQUIRED_KEYWORDS, Rock
from bores.reservoir.state import Hysteresis, ReservoirState
from bores.reservoir.temperature import TEMPERATURE_KEYWORDS, Temperature
from bores.schedule.base import Schedule
from bores.schedule.summary import SummaryReport
from bores.simulation.spec import RunSpec
from bores.simulation.workspace import SimulationWorkspace, build_simulation_workspace
from bores.timestep import TimeTargets, TimeTargetSource, build_time_targets
from bores.types import (
    Boolean,
    CellArray,
    GridIntersectionMethod,
    InterpolationMethod,
    Number,
    Spacing,
    UnitSystem,
)
from bores.wells.deck import load_schedule
from bores.wells.model import TubingPerformance, WellSystem
from bores.wells.schedule import check_vfp_references

__all__ = ["SimulationCase", "load_case"]


@attrs.define(kw_only=True, slots=True, frozen=True)
class SimulationCase:
    """A black-oil simulation case."""

    title: str
    """A label for the case, such as the deck's `TITLE` or a name given when loading."""

    model: CompiledBlackOilModel
    """The black-oil model this case runs against."""

    initial_state: ReservoirState
    """The reservoir's state at the start of the run."""

    runspec: RunSpec = attrs.field(factory=RunSpec)
    """Run configuration - unit system, constants, timer, start date, resolver tuning."""

    schedule: Schedule[CompiledBlackOilModel] = attrs.field(
        factory=Schedule[CompiledBlackOilModel]
    )
    """Every scheduled well/model edit for the run."""

    time_targets: TimeTargets = attrs.field(factory=TimeTargets)
    """
    Times the run must land on that do not come from the schedule: the deck's report times and
    any custom ones, in the case's own time unit. The schedule's own event times are added by
    `sync_timer`, since they change whenever the schedule does.
    """

    summary: SummaryReport = attrs.field(factory=SummaryReport)
    """
    Summary-vector report for the run, from the schedule of `RecordSummary` actions. 
    """

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
        timer = self.runspec.timer
        if timer is not None and timer.unit_system != self.runspec.unit_system:
            raise CaseValidationError(
                "Simulation case's timer and runspec must share the same unit system."
            )

    def sync_timer(self) -> Boolean:
        """
        Makes the run's timer land on this case's schedule events, report times and custom
        times. It does nothing if the run has no timer, or if neither the schedule nor the
        times have changed since the last call, so it is cheap to call before every step. A run
        that swaps in an edited schedule, for example with `attrs.evolve(case, schedule=...)`,
        only needs to call it again on the new case.

        :returns: `True` if the timer's targets were rebuilt, `False` if nothing needed doing.
        """
        timer = self.runspec.timer
        if timer is None:
            return False
        return timer.update_time_targets(
            self.schedule, self.time_targets, start_date=self.runspec.start_date
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
            boundary_conditions=model.boundary_conditions,
            hysteresis=hysteresis,
            salinity=self.salinity,
            dtype=dtype,
        )

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        title: str | None = None,
        tubing_performance: TubingPerformance
        | typing.Mapping[str, TubingPerformance | None]
        | None = None,
        temperature: Temperature | Number | None = None,
        mixing_rule: str = "baker_linear_rule",
        compiled_at: float = 0.0,
        load_summary: bool = True,
        spacing: Spacing = "cosine",
        warn_on_extrapolation: bool = False,
        horizontal_tolerance: Number | None = None,
        intersection_method: GridIntersectionMethod = "aabb",
        search_radius: Number | None = None,
        grid: Grid | None = None,
        regions: Regions | None = None,
        satfunc: SatFunc | None = None,
        rock: Rock | None = None,
        pvt: PVT | None = None,
        equilibrium: Equilibrium | None = None,
        initial_state: ReservoirState | None = None,
        wells: WellSystem | None = None,
        schedule: Schedule[CompiledBlackOilModel] | None = None,
        runspec: RunSpec | None = None,
        summary: SummaryReport | None = None,
        min_wetting_relperm: MinimumRelPerm = None,
        min_non_wetting_relperm: MinimumRelPerm = None,
        include_capillary_pressure: bool = True,
        allow_two_phase_satfunc: bool = False,
        hysteresis_enabled: bool | None = None,
        saturation_samples: int = N_SATURATION_SAMPLES,
        interpolation_method: InterpolationMethod = "linear",
        boundary_conditions: BoundaryConditions | None = None,
        salinity: CellArray | None = None,
        hysteresis: Hysteresis | None = None,
        unit_system: UnitSystem | None = None,
        on_inverted_cells: InvertedCellPolicy | str = InvertedCellPolicy.RAISE,
        extra_time_targets: typing.Iterable[TimeTargetSource] = (),
        dtype: npt.DTypeLike = None,
    ) -> Self:
        """
        Loads a `SimulationCase` from a deck. See `load_case`.

        :param deck_file: The deck to load from.
        :param title: A label for the case. The deck's `TITLE` is used if not given. Required
            when the deck has no `TITLE`.
        :param tubing_performance: What relates each well's BHP to its THP: a `WellBoreModel` or
            `VFPTables` for every well, or a per-well mapping of them. Omit if nothing in the
            deck needs one; the deck's own `VFPPROD`/`VFPINJ` tables are then used if it has
            any. See `WellSystem.from_deck`.
        :param temperature: Reservoir temperature, either a `Temperature` or
            a constant value in the deck's own unit system.
        :param mixing_rule: Three-phase relative permeability mixing rule for `SatFunc.from_deck`.
        :param compiled_at: The point on the schedule clock this case's
            model reflects. Forwarded to `bores.wells.deck.load_schedule`.
            Only schedule events strictly after this remain as actions.
        :param load_summary: Whether to add summary-vector recording items
            from the deck's `SUMMARY` section to the schedule.
        :param spacing: Saturation-function table grid spacing passed to `SatFunc.from_deck`.
        :param warn_on_extrapolation: Whether PVT table extrapolation logs warnings.
            Forwarded to `PVT.from_deck`.
        :param horizontal_tolerance: Optional horizontal tolerance for well-grid intersection.
            Forwarded to `compile_model`.
        :param intersection_method: Well-grid intersection strategy passed to `compile_model`.
        :param search_radius: Optional search radius for well-grid intersection.
            Forwarded to `compile_model`.
        :param grid: The grid to use in place of the deck's. Everything read from the deck that has one
            value per cell (rock properties, regions, the initial state) must have the same number of
            cells and the same cell ordering as this grid. Converted to the deck's unit system.
        :param regions: The region assignments to use in place of the deck's. Each array must have one
            entry per grid cell.
        :param satfunc: The saturation functions to use in place of the deck's. `mixing_rule`, `spacing`,
            `include_capillary_pressure`, and `allow_two_phase_satfunc` are then not used, and `min_wetting_relperm` and
            `min_non_wetting_relperm` must not be given. Converted to the deck's unit system.
        :param rock: The rock properties to use in place of the deck's. They must have one value per
            grid cell, and must have been built against the same saturation functions and saturation
            regions that the case uses. `interpolation_method` is then not used for the rock. Converted
            to the deck's unit system.
        :param pvt: The PVT tables to use in place of the deck's. Converted to the deck's unit system.
        :param equilibrium: The equilibration data to use in place of the deck's when computing the
            initial state. Not allowed together with `initial_state`. Converted to the deck's unit system.
        :param initial_state: The reservoir state to start the run from, in place of one computed from
            the deck's equilibration data. It must have one value per grid cell. Its pressures and
            saturations should be consistent with the PVT tables and saturation functions the case uses.
            `equilibrium`, `hysteresis_enabled`, and `saturation_samples` are then not used. Converted
            to the deck's unit system.
        :param wells: The wells, their controls, and their groups to use in place of those in the deck.
            Its own tubing performance is used, so `tubing_performance` must not be given. The schedule
            is still checked for references to VFP tables this system does not have.
        :param schedule: The schedule of well and model edits to use in place of the deck's. `compiled_at`
            and `load_summary` are then not used. It must already be expressed in the case's unit system.
        :param runspec: Run configuration. `RunSpec`'s own defaults if not given. If it has no
        `start_date`, the deck's `START` is used. If it has a timer, the timer is set up to land
        on the deck's report times, the schedule's events and `extra_time_targets`.
        :param summary: Summary-vector reports for the run. Empty if not given.
        :param min_wetting_relperm: Minimum wetting-phase relative permeability for `SatFunc.from_deck`.
        :param min_non_wetting_relperm: Minimum non-wetting-phase relative permeability for `SatFunc.from_deck`.
        :param include_capillary_pressure: Whether to include capillary pressure in the saturation functions.
            Forwarded to `SatFunc.from_deck`.
        :param allow_two_phase_satfunc: Whether a saturation region whose deck defines only one of the
            oil-water and gas-oil systems may be given a two-phase saturation function, instead of the
            load failing. It covers relative permeability and capillary pressure together. The phase
            that the region has no system for gets zero relative permeability and capillary pressure.
            Regions that define both systems are unaffected. Forwarded to `SatFunc.from_deck` as
            `allow_two_phase`.
        :param hysteresis_enabled: Whether to include hysteresis in the initial state.
            Forwarded to `initialize_reservoir_state`.
        :param saturation_samples: Number of saturation samples for the initial state.
            Forwarded to `initialize_reservoir_state`.
        :param interpolation_method: Interpolation method for `Rock.from_deck` and `PVT.from_deck`.
        :param salinity: Optional per-cell water salinity (ppm NaCl) for the simulation.
        :param hysteresis: Optional hysteresis history/configuration for the simulation workspace.
        :param unit_system: Unit system the case is expressed in: the model, the initial state, the
            run spec, the schedule's event times and values, and the time targets. Defaults to the
            deck's unit system. A different one converts everything read from the deck. Values you
            pass in that are plain numbers, namely `temperature` and `compiled_at`, stay in the
            deck's unit system. Objects that carry their own unit system are converted as needed.
        :param on_inverted_cells: Policy to apply when the deck grid contains inverted cells.
            Forwarded to `Grid.from_deck`.
        :param extra_time_targets: Custom times the run must land on, besides the deck's report
            times and the schedule's event times. Elapsed times in the case's time unit, dates
            (which need a start date), `PeriodicTarget`, or lists of those. See `build_time_targets`.
        :param dtype: Array dtype for every buffer. Defaults to `bores.precision.get_dtype()`.
        :returns: The loaded `SimulationCase`.
        """
        case = load_case(
            deck_file,
            title=title,
            tubing_performance=tubing_performance,
            temperature=temperature,
            mixing_rule=mixing_rule,
            compiled_at=compiled_at,
            load_summary=load_summary,
            spacing=spacing,
            warn_on_extrapolation=warn_on_extrapolation,
            horizontal_tolerance=horizontal_tolerance,
            intersection_method=intersection_method,
            search_radius=search_radius,
            grid=grid,
            regions=regions,
            satfunc=satfunc,
            rock=rock,
            pvt=pvt,
            equilibrium=equilibrium,
            initial_state=initial_state,
            wells=wells,
            schedule=schedule,
            runspec=runspec,
            summary=summary,
            min_wetting_relperm=min_wetting_relperm,
            min_non_wetting_relperm=min_non_wetting_relperm,
            include_capillary_pressure=include_capillary_pressure,
            allow_two_phase_satfunc=allow_two_phase_satfunc,
            hysteresis_enabled=hysteresis_enabled,
            saturation_samples=saturation_samples,
            interpolation_method=interpolation_method,
            boundary_conditions=boundary_conditions,
            salinity=salinity,
            hysteresis=hysteresis,
            unit_system=unit_system,
            on_inverted_cells=on_inverted_cells,
            extra_time_targets=extra_time_targets,
            dtype=dtype,
        )
        return typing.cast(Self, case)


def check_cell_counts(
    *,
    n_cells: int,
    regions: Regions | None,
    rock: Rock | None,
    initial_state: ReservoirState | None,
) -> None:
    """
    Checks that every supplied per-cell component has one entry per grid cell.

    :param n_cells: Number of cells in the grid.
    :param regions: Supplied region assignments, if any.
    :param rock: Supplied rock properties, if any.
    :param initial_state: Supplied initial state, if any.
    :raises CaseLoadError: If any supplied component has a different number of cells.
    """
    mismatches: list[str] = []
    if regions is not None:
        for field in attrs.fields(Regions):
            values = getattr(regions, field.name)
            if values is not None and np.size(values) != n_cells:
                mismatches.append(f"regions.{field.name} has {np.size(values)}")
    if rock is not None and np.size(rock.porosity) != n_cells:
        mismatches.append(f"rock has {np.size(rock.porosity)}")
    if initial_state is not None and np.size(initial_state.pressure) != n_cells:
        mismatches.append(f"initial_state has {np.size(initial_state.pressure)}")
    if mismatches:
        raise CaseLoadError(
            f"The grid has {n_cells} cells but a supplied component does not match: "
            + ", ".join(mismatches)
            + "."
        )


def check_deck_requirements(
    deck_file: DeckFile,
    *,
    title: str,
    needs_temperature: bool,
    needs_grid: bool,
    needs_satfunc: bool,
    needs_rock: bool,
    needs_pvt: bool,
    needs_equilibrium: bool,
) -> None:
    """
    Check that the deck has what every stage still to be loaded from it cannot do without.

    Loading the grid, saturation functions and rock parses large arrays and tables, so a deck
    that lacks (say) its `EQUIL` would otherwise only be reported after all of that work. This
    looks at the deck's keyword index only, reports every missing requirement at once, and leaves
    out the stages whose component the caller supplied. Optional keywords (wells, schedule,
    boundary conditions, regions, ...) are never required here.

    :param deck_file: The deck the case is being loaded from.
    :param title: The case title, for the error message.
    :param needs_temperature: Whether a `Temperature` has to be read from the deck.
    :param needs_grid: Whether the grid has to be read from the deck.
    :param needs_satfunc: Whether the saturation functions have to be read from the deck.
    :param needs_rock: Whether the rock properties have to be read from the deck.
    :param needs_pvt: Whether the PVT tables have to be read from the deck.
    :param needs_equilibrium: Whether the equilibration data has to be read from the deck.
    :raises CaseLoadError: If any needed keyword is missing, naming each one by the stage that
        needs it.
    """
    all_of: list[str] = []
    any_of: list[tuple[str, ...]] = []
    stage_of: dict[str, str] = {}

    def require(
        stage: str,
        *,
        all_of_keywords: tuple[str, ...] = (),
        any_of_keywords: typing.Sequence[tuple[str, ...]] = (),
    ) -> None:
        for keyword in all_of_keywords:
            all_of.append(keyword)
            stage_of.setdefault(f"`{keyword}`", stage)
        for group in any_of_keywords:
            any_of.append(tuple(group))
            stage_of.setdefault("one of " + ", ".join(f"`{k}`" for k in group), stage)

    problems: list[str] = []
    if needs_grid:
        if deck_file.dimensions is None:
            problems.append("grid: `SPECGRID` or `DIMENS`")
        geometry = grid_geometry_requirements(deck_file.has("COORD"))
        require(
            "grid",
            all_of_keywords=geometry.get("all_of", ()),
            any_of_keywords=geometry.get("any_of", ()),
        )
    if needs_temperature:
        require("temperature", any_of_keywords=(TEMPERATURE_KEYWORDS,))
    if needs_satfunc:
        require("saturation functions", any_of_keywords=(SATFUNC_KEYWORDS,))
    if needs_rock:
        require("rock", all_of_keywords=ROCK_REQUIRED_KEYWORDS)
    if needs_pvt:
        require("PVT", any_of_keywords=(OIL_PVT_KEYWORDS,))
    if needs_equilibrium:
        require("initial state", all_of_keywords=("EQUIL",))

    for entry in find_missing(deck_file, all_of=all_of, any_of=any_of):
        problems.append(f"{stage_of[entry]}: {entry}")
    if problems:
        raise CaseLoadError(
            f"The deck for case {title!r} is missing what is needed to load it: "
            + "; ".join(problems)
            + ". Supply the component to `load_case` directly, or add the keyword to the deck."
        )


def load_case(
    deck_file: DeckFile,
    *,
    title: str | None = None,
    tubing_performance: TubingPerformance
    | typing.Mapping[str, TubingPerformance | None]
    | None = None,
    temperature: Temperature | Number | None = None,
    mixing_rule: str = "baker_linear_rule",
    compiled_at: float = 0.0,
    load_summary: bool = True,
    spacing: Spacing = "cosine",
    warn_on_extrapolation: bool = False,
    horizontal_tolerance: Number | None = None,
    intersection_method: GridIntersectionMethod = "aabb",
    search_radius: Number | None = None,
    grid: Grid | None = None,
    regions: Regions | None = None,
    satfunc: SatFunc | None = None,
    rock: Rock | None = None,
    pvt: PVT | None = None,
    equilibrium: Equilibrium | None = None,
    initial_state: ReservoirState | None = None,
    wells: WellSystem | None = None,
    schedule: Schedule[CompiledBlackOilModel] | None = None,
    runspec: RunSpec | None = None,
    summary: SummaryReport | None = None,
    min_wetting_relperm: MinimumRelPerm = None,
    min_non_wetting_relperm: MinimumRelPerm = None,
    include_capillary_pressure: bool = True,
    allow_two_phase_satfunc: bool = False,
    hysteresis_enabled: bool | None = None,
    saturation_samples: int = N_SATURATION_SAMPLES,
    interpolation_method: InterpolationMethod = "linear",
    boundary_conditions: BoundaryConditions | None = None,
    salinity: CellArray | None = None,
    hysteresis: Hysteresis | None = None,
    unit_system: UnitSystem | None = None,
    on_inverted_cells: InvertedCellPolicy | str = InvertedCellPolicy.RAISE,
    extra_time_targets: typing.Iterable[TimeTargetSource] = (),
    dtype: npt.DTypeLike = None,
) -> SimulationCase:
    """
    Loads a full `SimulationCase` from a deck: grid, rock, fluid, initial
    state, wells, and schedule.

    :param deck_file: The deck to load from.
    :param title: A label for the case. The deck's `TITLE` is used if not given. Required when
        the deck has no `TITLE`.
    :param tubing_performance: What relates each well's BHP to its THP: a `WellBoreModel` or
        `VFPTables` for every well, or a per-well mapping of them. Omit if nothing in the deck
        needs one; the deck's own `VFPPROD`/`VFPINJ` tables are then used if it has any.
        See `WellSystem.from_deck`.
    :param temperature: Reservoir temperature, either a `Temperature` or
        a constant value in the deck's own unit system.
    :param mixing_rule: Three-phase relative permeability mixing rule for `SatFunc.from_deck`.
    :param compiled_at: The point on the schedule clock this case's
        model reflects. Forwarded to `bores.wells.deck.load_schedule`.
        Only schedule events strictly after this remain as actions.
    :param load_summary: Whether to add summary-vector recording items
        from the deck's `SUMMARY` section to the schedule.
    :param spacing: Saturation-function table grid spacing passed to `SatFunc.from_deck`.
    :param warn_on_extrapolation: Whether PVT table extrapolation logs warnings.
        Forwarded to `PVT.from_deck`.
    :param horizontal_tolerance: Optional horizontal tolerance for well-grid intersection.
        Forwarded to `compile_model`.
    :param intersection_method: Well-grid intersection strategy passed to `compile_model`.
    :param search_radius: Optional search radius for well-grid intersection.
        Forwarded to `compile_model`.
    :param grid: The grid to use in place of the deck's. Everything read from the deck that has one
        value per cell (rock properties, regions, the initial state) must have the same number of
        cells and the same cell ordering as this grid. Converted to the deck's unit system.
    :param regions: The region assignments to use in place of the deck's. Each array must have one
        entry per grid cell.
    :param satfunc: The saturation functions to use in place of the deck's. `mixing_rule`, `spacing`,
        `include_capillary_pressure`, and `allow_two_phase_satfunc` are then not used, and `min_wetting_relperm` and
        `min_non_wetting_relperm` must not be given. Converted to the deck's unit system.
    :param rock: The rock properties to use in place of the deck's. They must have one value per
        grid cell, and must have been built against the same saturation functions and saturation
        regions that the case uses. `interpolation_method` is then not used for the rock. Converted
        to the deck's unit system.
    :param pvt: The PVT tables to use in place of the deck's. Converted to the deck's unit system.
    :param equilibrium: The equilibration data to use in place of the deck's when computing the
        initial state. Not allowed together with `initial_state`. Converted to the deck's unit system.
    :param initial_state: The reservoir state to start the run from, in place of one computed from
        the deck's equilibration data. It must have one value per grid cell. Its pressures and
        saturations should be consistent with the PVT tables and saturation functions the case uses.
        `equilibrium`, `hysteresis_enabled`, and `saturation_samples` are then not used. Converted
        to the deck's unit system.
    :param wells: The wells, their controls, and their groups to use in place of those in the deck.
        Its own tubing performance is used, so `tubing_performance` must not be given. The schedule
        is still checked for references to VFP tables this system does not have.
    :param schedule: The schedule of well and model edits to use in place of the deck's. `compiled_at`
        and `load_summary` are then not used. It must already be expressed in the case's unit system.
    :param runspec: Run configuration. `RunSpec`'s own defaults if not given.
    :param summary: Summary-vector reports for the run. Empty if not given.
    :param min_wetting_relperm: Minimum wetting-phase relative permeability for `SatFunc.from_deck`.
    :param min_non_wetting_relperm: Minimum non-wetting-phase relative permeability for `SatFunc.from_deck`.
    :param include_capillary_pressure: Whether to include capillary pressure in the saturation functions.
        Forwarded to `SatFunc.from_deck`.
    :param allow_two_phase_satfunc: Whether a saturation region whose deck defines only one of the
        oil-water and gas-oil systems may be given a two-phase saturation function, instead of the
        load failing. It covers relative permeability and capillary pressure together. The phase
        that the region has no system for gets zero relative permeability and capillary pressure.
        Regions that define both systems are unaffected. Forwarded to `SatFunc.from_deck` as
        `allow_two_phase`.
    :param hysteresis_enabled: Whether to include hysteresis in the initial state.
        Forwarded to `initialize_reservoir_state`.
    :param saturation_samples: Number of saturation samples for the initial state.
        Forwarded to `initialize_reservoir_state`.
    :param interpolation_method: Interpolation method for `Rock.from_deck` and `PVT.from_deck`.
    :param salinity: Optional per-cell water salinity (ppm NaCl) for the simulation.
    :param hysteresis: Optional hysteresis history/configuration for the simulation workspace.
    :param unit_system: Unit system the case is expressed in: the model, the initial state, the
        run spec, the schedule's event times and values, and the time targets. Defaults to the
        deck's unit system. A different one converts everything read from the deck. Values you
        pass in that are plain numbers, namely `temperature` and `compiled_at`, stay in the
        deck's unit system. Objects that carry their own unit system are converted as needed.
    :param on_inverted_cells: Policy to apply when the deck grid contains inverted cells.
        Forwarded to `Grid.from_deck`.
    :param dtype: Array dtype for every buffer. Defaults to `bores.precision.get_dtype()`.
    :param boundary_conditions: Updates or overrides the boundary conditions
        loaded from the deck's `AQUCT`/`AQUFETP`/`AQUFLUX`/`AQUANCON` keywords.
        A region here replaces a same-named deck-loaded region, and a
        region with a new name is added; see `BoundaryConditions.override`.
        Converted to `unit_system` if given in a different one. If the
        deck defines no boundary conditions, this is used as-is.
    :returns: The loaded `SimulationCase`.
    :raises CaseLoadError: If the deck has no `TITLE` and no `title` is given, if an option is
        given that has no effect because the component it configures was supplied instead (see
        `satfunc`, `equilibrium`, `initial_state`, and `wells`), if a supplied component does not
        match the grid's cell count, or if any loading stage fails. For a failed stage, the
        original exception is chained as the cause.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    raw_deck_title = deck_file.get("TITLE")
    raw_deck_title = raw_deck_title.strip() if isinstance(raw_deck_title, str) else None

    if title is not None and not title.strip():
        raise CaseLoadError("`title` must not be empty. Omit it to use the deck's `TITLE`.")
    if title is not None:
        title = title.strip()
    elif raw_deck_title:
        title = raw_deck_title
    else:
        raise CaseLoadError(
            "The deck has no `TITLE`, so a case title is required. Pass `title` to name the case."
        )

    if satfunc is not None and (
        min_wetting_relperm is not None or min_non_wetting_relperm is not None
    ):
        raise CaseLoadError(
            "`min_wetting_relperm` and `min_non_wetting_relperm` only apply to saturation "
            "functions loaded from the deck, so they cannot be given together with `satfunc`."
        )
    if wells is not None and tubing_performance is not None:
        raise CaseLoadError(
            "`tubing_performance` only applies to wells loaded from the deck. Set it on the "
            "`WellSystem` given as `wells` instead."
        )
    if initial_state is not None and equilibrium is not None:
        raise CaseLoadError(
            "`equilibrium` is only used to compute the initial state, so it cannot be given "
            "together with `initial_state`."
        )
    if initial_state is not None and hysteresis_enabled is not None:
        raise CaseLoadError(
            "`hysteresis_enabled` only applies to an initial state computed from the deck. "
            "Build the `initial_state` given with the hysteresis tracking it needs instead."
        )

    check_deck_requirements(
        deck_file,
        title=title,
        needs_temperature=temperature is None and (pvt is None or initial_state is None),
        needs_grid=grid is None,
        needs_satfunc=satfunc is None,
        needs_rock=rock is None,
        needs_pvt=pvt is None,
        needs_equilibrium=initial_state is None and equilibrium is None,
    )

    deck_unit_system = deck_file.unit_system
    unit_system = unit_system if unit_system is not None else deck_unit_system
    runspec = (
        runspec.convert(unit_system) if runspec is not None else RunSpec(unit_system=unit_system)
    )
    if temperature is None:
        # Only the PVT tables and the initial state need a temperature.
        if pvt is None or initial_state is None:
            temperature = Temperature.from_deck(deck_file, dtype=dtype)
    else:
        temperature = (
            temperature.convert(deck_unit_system)
            if isinstance(temperature, Temperature)
            else Temperature(temperature, unit_system=deck_unit_system)
        )

    # Components that are supplied are expressed in the deck's unit system from here on, as the
    # ones loaded from the deck are, since the stages below combine them.
    if grid is None:
        try:
            print("Loading grid")
            grid = Grid.from_deck(deck_file, on_inverted_cells=on_inverted_cells)
            print("Loaded grid")
        except Exception as exc:
            raise CaseLoadError(f"Failed to load grid from deck file: {title!r}.") from exc
    else:
        grid = grid.convert(deck_unit_system)

    if regions is None:
        try:
            print("Loading regions")
            # Decks write 0 as the region of inactive cells, which Eclipse ignores.
            active = (
                grid.cell_statuses != int(CellStatus.INACTIVE)
                if grid.cell_statuses is not None
                else None
            )
            regions = Regions.from_deck(
                deck_file,
                n_cells=grid.n_cells,
                use_default=True,
                active=active,
            )
            print("Loaded regions")
        except Exception as exc:
            raise CaseLoadError(f"Failed to load regions from deck file: {title!r}.") from exc

    if satfunc is None:
        try:
            print("Loading satfunc")
            satfunc = SatFunc.from_deck(
                deck_file,
                mixing_rule=mixing_rule,
                spacing=spacing,
                min_wetting_relperm=min_wetting_relperm,
                min_non_wetting_relperm=min_non_wetting_relperm,
                include_capillary_pressure=include_capillary_pressure,
                allow_two_phase=allow_two_phase_satfunc,
                dtype=dtype,
            )
            print("Loaded satfunc")
        except Exception as exc:
            raise CaseLoadError(
                f"Failed to load saturation functions from deck file: {title!r}."
            ) from exc
    else:
        satfunc = satfunc.convert(deck_unit_system)

    if rock is None:
        try:
            print("Loading rock")
            rock = Rock.from_deck(
                deck_file,
                grid=grid,
                satfunc=satfunc,
                saturation_region=regions.saturation_region,
                interpolation_method=interpolation_method,
                dtype=dtype,
            )
            print("Loaded rock")
        except Exception as exc:
            raise CaseLoadError(
                f"Failed to load rock properties from deck file: {title!r}."
            ) from exc
    else:
        rock = rock.convert(deck_unit_system)

    if initial_state is not None:
        initial_state = initial_state.convert(deck_unit_system)

    check_cell_counts(
        n_cells=grid.n_cells,
        regions=regions,
        rock=rock,
        initial_state=initial_state,
    )
    reservoir = Reservoir(grid=grid, rock=rock, regions=regions)

    if pvt is None:
        try:
            print("Loading PVT")
            pvt = PVT.from_deck(
                deck_file,
                temperature=typing.cast(Temperature, temperature),
                interpolation_method=interpolation_method,
                warn_on_extrapolation=warn_on_extrapolation,
            )
            print("Loaded PVT")
        except Exception as exc:
            raise CaseLoadError(f"Failed to load PVT tables from deck file: {title!r}.") from exc
    else:
        pvt = pvt.convert(deck_unit_system)

    fluid = BlackOil(pvt=pvt, satfunc=satfunc)

    if initial_state is None:
        try:
            print("Loading state")
            equilibrium = (
                Equilibrium.from_deck(deck_file)
                if equilibrium is None
                else equilibrium.convert(deck_unit_system)
            )
            initial_state = initialize_reservoir_state(
                reservoir=reservoir,
                pvt=pvt,
                deck_file=deck_file,
                equilibrium=equilibrium,
                satfunc=satfunc,
                temperature=typing.cast(Temperature, temperature),
                hysteresis_enabled=hysteresis_enabled
                if hysteresis_enabled is not None
                else runspec.hysteresis_enabled,
                saturation_samples=saturation_samples,
                dtype=dtype,
            )
            print("Loaded state")
        except Exception as exc:
            raise CaseLoadError(
                f"Failed to initialize reservoir state from deck file: {title!r}."
            ) from exc

    if wells is None:
        try:
            print("Loading wells")
            wells = WellSystem.from_deck(
                deck_file,
                grid=grid,
                tubing_performance=tubing_performance,
            )
            print("Loaded wells")
        except Exception as exc:
            raise CaseLoadError(f"Failed to load wells from deck file: {title!r}.") from exc

    try:
        print("Loading BC")
        resolved_boundary_conditions = load_boundary_conditions(
            deck_file,
            grid,
            pvt=pvt,
            reservoir_pressure=compute_water_phase_pressure(reservoir, initial_state, satfunc),
        )
        print("Loaded BC")
    except Exception as exc:
        raise CaseLoadError(
            f"Failed to load boundary conditions from deck file: {title!r}."
        ) from exc

    if boundary_conditions is not None:
        boundary_conditions = boundary_conditions.convert(deck_unit_system)
        resolved_boundary_conditions = (
            resolved_boundary_conditions.override(boundary_conditions)
            if resolved_boundary_conditions is not None
            else boundary_conditions
        )

    model = BlackOilModel(
        reservoir=reservoir,
        fluid=fluid,
        wells=wells,
        boundary_conditions=resolved_boundary_conditions,
        unit_system=unit_system,
    )

    if schedule is None:
        try:
            print("Loading schedule")
            schedule = load_schedule(
                deck_file,
                compiled_at=compiled_at,
                load_summary=load_summary,
                unit_system=unit_system,
            )
            print("Loaded schedule")
        except Exception as exc:
            raise CaseLoadError(f"Failed to load schedule from deck file: {title!r}.") from exc

    try:
        check_vfp_references(well_system=wells, schedule=schedule)
    except ValidationError as exc:
        raise CaseLoadError(f"Invalid VFP table reference in deck file: {title!r}: {exc}") from exc

    start_date = runspec.start_date
    if start_date is None:
        start_date = deck_file.get(START)
        if start_date is not None:
            runspec = attrs.evolve(runspec, start_date=start_date)

    try:
        time_targets = build_time_targets(
            deck_file,
            list(extra_time_targets),
            start_date=start_date,
            unit_system=unit_system,
        )
    except Exception as exc:
        raise CaseLoadError(f"Failed to load time targets from deck file: {title!r}.") from exc

    case = SimulationCase(
        title=title,
        model=compile_model(
            model,
            horizontal_tolerance=horizontal_tolerance,
            intersection_method=intersection_method,
            search_radius=search_radius,
            dtype=dtype,
        ),
        initial_state=initial_state.convert(unit_system),
        runspec=runspec,
        schedule=schedule,
        time_targets=time_targets,
        summary=summary if summary is not None else SummaryReport(),
        salinity=salinity,
        hysteresis=hysteresis,
        dtype=dtype,
    )
    try:
        case.sync_timer()
    except Exception as exc:
        raise CaseLoadError(
            f"Failed to set up the timer's time targets for deck file: {title!r}."
        ) from exc
    return case
