"""Model analysis tools for reservoir performance evaluation over simulation states."""

import logging
import typing

import attrs
import numpy as np
from scipy.optimize import curve_fit

from bores.cells import CellFilter, Cells
from bores.constants import c
from bores.correlations.arrays import compute_hydrocarbon_in_place
from bores.errors import ValidationError
from bores.grids.base import uniform_grid
from bores.states import ModelState
from bores.types import FluidPhase, NDimension, NDimensionalGrid
from bores.utils import clip
from bores.wells.base import Wells, _expand_intervals

logger = logging.getLogger(__name__)

__all__ = ["ModelAnalyst"]


def _ensure_cells(cells: typing.Union[Cells, CellFilter]) -> typing.Optional[Cells]:
    """Convert CellFilter to Cells if needed."""
    if cells is None:
        return None
    if isinstance(cells, Cells):
        return cells
    return Cells.from_filter(cells)


@attrs.frozen(slots=True)
class ReservoirVolumetrics:
    """Reservoir volumetrics analysis results."""

    oil_in_place: float
    """Total oil in place in stock tank barrels (STB)."""
    gas_in_place: float
    """Total gas in place (including solution gas) in standard cubic feet (SCF)."""
    water_in_place: float
    """Total water in place in stock tank barrels (STB)."""
    pore_volume: float
    """Total pore volume in cubic feet (ft³)."""
    hydrocarbon_pore_volume: float
    """Hydrocarbon pore volume in cubic feet (ft³)."""


@attrs.frozen(slots=True)
class InstantaneousRates:
    """Instantaneous production/injection rates."""

    oil_rate: float
    """Oil production/injection rate in stock tank barrels per day (STB/day)."""
    gas_rate: float
    """Total gas rate (free gas + solution gas from oil) in standard cubic feet per day (SCF/day)."""
    water_rate: float
    """Water production/injection rate in stock tank barrels per day (STB/day)."""
    total_liquid_rate: float
    """Total liquid (oil + water) rate in stock tank barrels per day (STB/day)."""
    gas_oil_ratio: float
    """Produced gas-oil ratio (free gas + solution gas) / oil in SCF/STB."""
    water_cut: float
    """Water cut as a fraction (0 to 1) of total liquid production."""
    free_gas_rate: float = 0.0
    """Free gas phase rate in standard cubic feet per day (SCF/day)."""
    solution_gas_rate: float = 0.0
    """Solution gas (dissolved in produced oil, released at surface) rate in SCF/day."""


@attrs.frozen(slots=True)
class CumulativeProduction:
    """Cumulative production analysis results."""

    cumulative_oil: float
    """Cumulative oil produced in stock tank barrels (STB)."""
    cumulative_free_gas: float
    """Cumulative free gas produced in standard cubic feet (SCF)."""
    cumulative_water: float
    """Cumulative water produced in stock tank barrels (STB)."""
    oil_recovery_factor: float
    """Oil recovery factor as a fraction (0 to 1) of initial oil in place."""
    gas_recovery_factor: float
    """Gas recovery factor as a fraction (0 to 1) of initial gas in place."""


@attrs.frozen(slots=True)
class MaterialBalance:
    """
    Results of a Havlena-Odeh material balance analysis at a specific time step.

    The generalized material balance equation expresses that total underground
    withdrawal must equal the sum of all expansion and influx terms:

        F = N*(Eo + m*Eg + Efw) + We

    Where:
        F    = Underground withdrawal (reservoir bbl)
        N    = Stock tank oil initially in place (STB)
        Eo   = Oil + dissolved gas expansion (bbl/STB)
        m    = Gas cap ratio (dimensionless)
        Eg   = Gas cap expansion (bbl/STB)
        Efw  = Rock + connate water expansion (bbl/STB)
        We   = Water influx (bbl)

    Drive indices partition F into the fraction supplied by each mechanism
    and always sum to 1.0.
    """

    pressure: float
    """Current reservoir pressure (psia), oil-saturation weighted mean."""

    pressure_decline: float
    """Pressure decline from initial conditions (psi). Positive = depleting."""

    oil_expansion_factor: float
    """
    Ratio of current to initial oil FVF (dimensionless).

        oil_expansion_factor = Bo / Boi

    > 1.0 while reservoir is above bubble point (undersaturated expansion).
    Peaks at bubble point, then declines as gas comes out of solution.
    """

    producing_gor: float
    """
    Cumulative producing gas-oil ratio (SCF/STB).

        Rp = (cumulative free gas + cumulative solution gas) / cumulative oil

    Equal to Rsi early in life when only solution gas is produced. Rises
    sharply once free gas starts being produced below bubble point.
    """

    gas_cap_ratio: float
    """
    Ratio of initial gas cap reservoir volume to initial oil reservoir volume
    (dimensionless).

        m = (GIIP * Bgi) / (N * Boi)

    Zero for reservoirs with no initial free gas cap.
    """

    underground_withdrawal: float
    """
    Total underground withdrawal F (reservoir bbl).

        F = Np*(Bo + (Rp - Rs)*Bg) + Wp*Bw - Wi*Bwi - Gi*Bginj

    The left-hand side of the MBE. Primary normaliser for drive indices.
    """

    solution_gas_drive: float
    """
    Energy supplied by oil and dissolved gas expansion (reservoir bbl).

        solution_gas_drive = N * Eo
                           = N * [(Bo - Boi) + (Rsi - Rs)*Bg]

    Primary mechanism in solution-gas-drive reservoirs. Typical recovery: 5-30% STOIIP.
    """

    gas_cap_drive: float
    """
    Energy supplied by free gas cap expansion (reservoir bbl).

        gas_cap_drive = N * m * Eg
                      = N * m * Boi * [(Bg/Bgi) - 1]

    Zero if no initial free gas cap exists. Typical recovery with gas cap: 20-40% STOIIP.
    """

    water_drive: float
    """
    Energy supplied by water influx (reservoir bbl).

        water_drive = We = max(0, F - N*(Eo + m*Eg + Efw))

    Captures natural aquifer influx. Water injection is already subtracted
    from F so this represents aquifer support only.
    Typical recovery with strong water drive: 35-75% STOIIP.
    """

    compaction_drive: float
    """
    Energy supplied by rock and connate water compressibility (reservoir bbl).

        compaction_drive = N * Efw
                         = N * (1+m) * Boi * [(Swi*cw + cf) / (1-Swi)] * ΔP

    Small in most conventional reservoirs (1-5% of total). Dominant above
    bubble point where no gas expansion is available.
    """

    solution_gas_drive_index: float
    """
    Fraction of total withdrawal supplied by oil and dissolved gas expansion
    (dimensionless, 0.0 to 1.0).

    All four drive indices sum to 1.0.
    """

    gas_cap_drive_index: float
    """
    Fraction of total withdrawal supplied by gas cap expansion
    (dimensionless, 0.0 to 1.0).

    All four drive indices sum to 1.0. Zero if no initial free gas cap.
    """

    water_drive_index: float
    """
    Fraction of total withdrawal supplied by water influx
    (dimensionless, 0.0 to 1.0).

    All four drive indices sum to 1.0. High values (> 0.5) indicate
    strong aquifer support.
    """

    compaction_drive_index: float
    """
    Fraction of total withdrawal supplied by rock and connate water
    compressibility (dimensionless, 0.0 to 1.0).

    All four drive indices sum to 1.0. Typically < 0.05 in conventional
    reservoirs.
    """

    aquifer_influx: float
    """
    Estimated natural aquifer influx We (reservoir bbl).

    Identical to `water_drive`. Exposed separately as a named field for
    convenience since aquifer influx is commonly reported standalone in
    reservoir engineering workflows.
    """


@attrs.frozen(slots=True)
class MaterialBalanceError:
    """
    Results of a material balance error analysis over a simulation interval.

    Combines two complementary checks:

    1. Phase-level MBE - simulator volume conservation check:

        MBE_phase = (ΔPV_phase - net_flux_phase) / PV_phase_initial

       All volumes at reservoir conditions (ft³). Directly tests whether the
       saturation solver conserved pore-volume occupancy for each phase.
       Zero means perfect volume conservation for that phase.

    2. Total MBE - Havlena-Odeh physical balance check:

        total_mbe = (F - N*(Eo + m*Eg + Efw) - We) / F

       Where F is total underground withdrawal (reservoir bbl). Checks whether
       observed production is physically explained by fluid and rock expansion.
       Zero means expansion exactly accounts for withdrawal.

    The two checks diagnose different problems:
        - Large phase MBE  → simulator numerical conservation error
        - Large total MBE  → PVT / STOIIP / drive mechanism mismatch

    Drive indices are sourced from the Havlena-Odeh terms and always sum to 1.0:

        solution_gas_drive_index + gas_cap_drive_index +
        water_drive_index + compaction_drive_index = 1.0

    MBE Quality Thresholds (based on worst of all four MBE values):
        < 0.1%      Excellent    - results are reliable
        0.1% - 1%   Acceptable   - monitor for drift
        1% - 5%     Marginal     - refine grid or reduce timestep
        > 5%        Unacceptable - investigate before using results

    References:
        Havlena, D. and Odeh, A.S.: "The Material Balance as an Equation of a
        Straight Line," JPT (August 1963).
    """

    total_mbe: float
    """
    Fractional Havlena-Odeh material balance error (dimensionless).

        total_mbe = (F - N*(Eo + m*Eg + Efw) - We) / F

    Where We = max(0, F - N*(Eo + m*Eg + Efw)).  Because water influx is
    computed as the residual between withdrawal and expansion (clamped to
    zero), this value is always <= 0.0 by construction.

    Interpretation:
        0.0    Perfect balance (expansion + aquifer influx accounts for withdrawal)
        < 0.0  Expansion exceeds withdrawal (overestimated STOIIP or PVT errors)
    """

    oil_mbe: float
    """
    Fractional volume conservation error for the oil phase (dimensionless).

    Computed as:

        oil_mbe = (ΔPV_oil - net_oil_flux) / PV_oil_initial

    Where:
        ΔPV_oil        = current_oil_pv - initial_oil_pv                (ft³)
        net_oil_flux   = oil_injected - oil_produced                    (ft³)
        PV_oil_initial = initial pore volume occupied by oil            (ft³)

    All volumes at reservoir conditions (ft³). Directly tests whether the
    saturation solver conserved oil pore-volume occupancy (PV * So).

    Zero means the simulator perfectly conserved oil volume over the interval.

    Common causes of large `oil_mbe`:
        - Time step too large (explicit solver instability)
        - Grid refinement needed near wells
        - Convergence tolerance too loose
        - Phase transfer (gas liberation/dissolution) volume imbalance
    """

    water_mbe: float
    """
    Fractional volume conservation error for the water phase (dimensionless).

    Computed as:

        water_mbe = (ΔPV_water - net_water_flux) / PV_water_initial

    Where:
        ΔPV_water        = current_water_pv - initial_water_pv          (ft³)
        net_water_flux   = water_injected - water_produced              (ft³)
        PV_water_initial = initial pore volume occupied by water        (ft³)

    All volumes at reservoir conditions (ft³).

    Zero means the simulator perfectly conserved water volume over the interval.

    Common causes of large `water_mbe`:
        - Water injection rates not correctly accounted for
        - Capillary pressure discontinuities causing numerical water movement
        - Time step too large during water breakthrough
    """

    gas_mbe: float
    """
    Fractional volume conservation error for the gas phase (dimensionless).

    Computed as:

        gas_mbe = (ΔPV_gas - net_gas_flux) / PV_gas_initial

    Where:
        ΔPV_gas        = current_gas_pv - initial_gas_pv                (ft³)
        net_gas_flux   = gas_injected - gas_produced                    (ft³)
        PV_gas_initial = initial pore volume occupied by free gas       (ft³)

    All volumes at reservoir conditions (ft³). Directly tests whether the
    saturation solver conserved gas pore-volume occupancy (PV * Sg).

    Zero means the simulator perfectly conserved gas volume over the interval.

    Note: `gas_mbe` will be 0.0 if there is no initial free gas
    (PV_gas_initial = 0), since a fractional error is undefined
    without an initial free gas volume to normalise against.
    """

    solution_gas_drive: float
    """
    Energy supplied by oil and dissolved gas expansion over the interval
    (reservoir bbl).

        solution_gas_drive = N * Eo
                           = N * [(Bo - Boi) + (Rsi - Rs) * Bg]
    """

    gas_cap_drive: float
    """
    Energy supplied by free gas cap expansion over the interval (reservoir bbl).

        gas_cap_drive = N * m * Eg
                      = N * m * Boi * [(Bg/Bgi) - 1]
    """

    water_drive: float
    """
    Energy supplied by water influx over the interval (reservoir bbl).

        water_drive = We = max(0, F - N*(Eo + m*Eg + Efw))

    See `MaterialBalance.water_drive` for full documentation.
    """

    compaction_drive: float
    """
    Energy supplied by rock and connate water compressibility over the interval
    (reservoir bbl).

        compaction_drive = N * Efw
                         = N * (1+m) * Boi * [(Swi*cw + cf) / (1-Swi)] * ΔP

    See `MaterialBalance.compaction_drive` for full documentation.
    """

    solution_gas_drive_index: float
    """
    Fraction of total withdrawal supplied by oil and dissolved gas expansion
    (dimensionless, 0.0 to 1.0).

    All four drive indices sum to 1.0.
    """

    gas_cap_drive_index: float
    """
    Fraction of total withdrawal supplied by gas cap expansion
    (dimensionless, 0.0 to 1.0).

    All four drive indices sum to 1.0. Zero if no initial free gas cap.
    """

    water_drive_index: float
    """
    Fraction of total withdrawal supplied by water influx
    (dimensionless, 0.0 to 1.0). 

    All four drive indices sum to 1.0.
    """

    compaction_drive_index: float
    """
    Fraction of total withdrawal supplied by rock and connate water
    compressibility (dimensionless, 0.0 to 1.0). 

    All four drive indices sum to 1.0.
    """

    underground_withdrawal: float
    """
    Total underground withdrawal F over the interval (reservoir bbl).

        F = Np*(Bo + (Rp - Rs)*Bg) + Wp*Bw - Wi*Bwi - Gi*Bginj

    Primary normaliser for `total_mbe`. See `MaterialBalance.underground_withdrawal`
    for full documentation.
    """

    quality: typing.Literal["excellent", "acceptable", "marginal", "unacceptable"]
    """
    Qualitative rating based on the worst of all four MBE values
    (`total_mbe`, `oil_mbe`, `water_mbe`, `gas_mbe`).

    ==================  ===========================  ================================
    Rating              max(|MBE|) threshold         Recommended action
    ==================  ===========================  ================================
    `"excellent"`     < 0.1%                       Results are reliable
    `"acceptable"`    0.1% - 1%                    Monitor for systematic drift
    `"marginal"`      1% - 5%                      Refine grid or reduce timestep
    `"unacceptable"`  > 5%                         Investigate before using results
    ==================  ===========================  ================================

    Using the worst of all four MBEs means a perfect Havlena-Odeh balance
    cannot mask a simulator phase conservation error, and vice versa.
    """

    from_step: int
    """The starting time step of the analysis interval (inclusive)."""

    to_step: int
    """The ending time step of the analysis interval (inclusive)."""


@attrs.frozen(slots=True)
class ProductivityAnalysis:
    """Well productivity analysis results based on actual flow rates and reservoir properties."""

    total_flow_rate: float
    """Total flow rate in stock tank barrels per day (STB/day) or standard cubic feet per day (SCF/day)."""
    average_reservoir_pressure: float
    """Average reservoir pressure at perforation intervals in psia."""
    skin_factor: float
    """Dimensionless skin factor indicating wellbore damage or stimulation."""
    flow_efficiency: float
    """Flow efficiency as a fraction (0 to 1) accounting for skin effects."""
    well_index: float
    """Geometric well index in reservoir barrels per day per psi (rb/day/psi)."""
    average_mobility: float
    """Average phase mobility at perforation intervals in 1/cp."""


@attrs.frozen(slots=True)
class SweepEfficiencyAnalysis:
    """Sweep efficiency analysis results."""

    volumetric_sweep_efficiency: float
    """Volumetric sweep efficiency as a fraction (0 to 1) of initial oil contacted."""

    displacement_efficiency: float
    """Displacement efficiency as a fraction (0 to 1) in contacted zones."""

    recovery_efficiency: float
    """Overall recovery efficiency as a fraction (0 to 1) combining sweep and displacement."""

    contacted_oil: float
    """Oil in contacted reservoir zones in stock tank barrels (STB)."""

    uncontacted_oil: float
    """Oil in uncontacted reservoir zones in stock tank barrels (STB)."""

    areal_sweep_efficiency: float = 0.0
    """Areal sweep efficiency (0 to 1) computed from contacted planform area."""

    vertical_sweep_efficiency: float = 0.0
    """Vertical sweep efficiency (0 to 1) computed using saturation-weighted column fractions."""


@attrs.frozen(slots=True)
class DeclineCurveResult:
    """Decline curve analysis results."""

    decline_type: typing.Literal["exponential", "hyperbolic", "harmonic"]
    """Type of decline curve analysis performed."""
    initial_rate: float
    """Initial production rate in stock tank barrels per day (STB/day) for oil/water or standard cubic feet per day (SCF/day) for gas."""
    decline_rate_per_timestep: float
    """Decline rate per time step as a fraction."""
    b_factor: float
    """Hyperbolic decline exponent (0 for exponential, 1 for harmonic)."""
    r_squared: float
    """Coefficient of determination (R²) indicating goodness of fit."""
    phase: typing.Literal["oil", "gas", "water"] = "oil"
    """Phase analyzed ('oil', 'gas', 'water')."""
    error: typing.Optional[str] = None
    """Error message if analysis could not be completed."""
    steps: typing.Optional[typing.List[int]] = None
    """Time steps used in the analysis."""
    actual_rates: typing.Optional[typing.List[float]] = None
    """Actual production rates in STB/day or SCF/day depending on phase."""
    predicted_rates: typing.Optional[typing.List[float]] = None
    """Predicted production rates from decline curve in STB/day or SCF/day depending on phase."""


@attrs.frozen(slots=True)
class InjectionFrontAnalysis:
    """Spatial injection-front analysis results for a single time step."""

    phase: typing.Literal["water", "gas"]
    """Phase whose front is being tracked."""

    front_cells: np.ndarray
    """
    Boolean mask (same shape as the reservoir grid) that is True at every cell
    whose saturation has increased by at least `threshold` relative to the
    initial state.  This is the *contacted* region (equivalent to the swept
    zone boundary).
    """

    front_cell_count: int
    """Number of cells on/inside the front (i.e. contacted cells)."""

    front_volume_fraction: float
    """Fraction of total pore volume that is contacted (0-1), weighted by
    cell pore volume (thickness * porosity * NTG * cell area)."""

    average_front_saturation: float
    """
    Mean saturation of the *displacing phase* in contacted cells only (fraction).
    Indicates how thoroughly the swept zone is saturated.
    """

    max_front_saturation: float
    """Peak displacing-phase saturation across all contacted cells (fraction)."""

    saturation_delta_grid: np.ndarray
    """
    Full-field grid of (current_saturation - initial_saturation) for the
    displacing phase.  Positive values indicate invaded cells; negative values
    indicate cells where the phase has receded (e.g. gas dissolution behind a
    waterflood front).

    Units: fraction (dimensionless).
    """

    front_centroid: typing.Tuple[float, float, float]
    """
    Saturation-delta weighted centre-of-mass of the contacted region
    expressed as (i_centroid, j_centroid, k_centroid) in cell-index units.
    Cells with larger saturation changes pull the centroid toward them,
    giving a more physically representative plume centre than an unweighted
    geometric mean.  Useful for tracking plume migration direction over time.
    """


def _build_injected_fvf_grids(
    wells: Wells,
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    grid_shape: NDimension,
) -> typing.Tuple[NDimensionalGrid[NDimension], NDimensionalGrid[NDimension]]:
    """
    Build per-cell grids of injected gas and water FVF using each
    injector's actual InjectedFluid properties and vectorized PVT correlations.

    Cells with no injection activity are NaN. Handles WAG and multi-fluid
    scenarios correctly - each injector's fluid is used only for the cells
    it perforates, using that fluid's specific gravity / salinity for FVF.

    :param wells: Wells object containing injection wells.
    :param pressure_grid: 3D pressure grid (psi).
    :param temperature_grid: 3D temperature grid (°F).
    :param grid_shape: Shape of the reservoir grid (nx, ny, nz).
    :return: (injected_gas_fvf_grid, injected_water_fvf_grid)
        all shaped `grid_shape`, NaN where no injection occurs.
    """
    injected_gas_fvf_grid = np.full(grid_shape, np.nan)
    injected_water_fvf_grid = np.full(grid_shape, np.nan)

    for well in wells.injection_wells:
        if not well.is_open or well.injected_fluid is None:
            continue

        fluid = well.injected_fluid

        # Collect all perforated cell indices for this well in one pass
        i_coords, j_coords, k_coords = [], [], []
        for start, end in well.perforating_intervals:
            for i in range(start[0], end[0] + 1):
                for j in range(start[1], end[1] + 1):
                    for k in range(start[2], end[2] + 1):
                        i_coords.append(i)
                        j_coords.append(j)
                        k_coords.append(k)

        if not i_coords:
            continue

        idx = (
            np.array(i_coords, dtype=np.intp),
            np.array(j_coords, dtype=np.intp),
            np.array(k_coords, dtype=np.intp),
        )

        # Extract pressure and temperature arrays for all perforated cells at once
        cell_pressures = pressure_grid[idx].astype(np.float64)
        cell_temperatures = temperature_grid[idx].astype(np.float64)

        # Single vectorized FVF call for all perforated cells of this injector
        fvf_values = fluid.get_formation_volume_factor(
            pressure=cell_pressures,
            temperature=cell_temperatures,
        )

        if fluid.phase == FluidPhase.GAS:
            injected_gas_fvf_grid[idx] = fvf_values
        elif fluid.phase == FluidPhase.WATER:
            injected_water_fvf_grid[idx] = fvf_values

    return injected_gas_fvf_grid, injected_water_fvf_grid  # type: ignore[return-value]


class ModelAnalyst(typing.Generic[NDimension]):
    """
    Analysis tools for evaluating reservoir model performance over a series of states
    captured during a simulation run.
    """

    def __init__(
        self,
        states: typing.Iterable[ModelState[NDimension]],
        stoiip: typing.Optional[float] = None,
        stgiip: typing.Optional[float] = None,
        free_giip: typing.Optional[float] = None,
        stwiip: typing.Optional[float] = None,
    ) -> None:
        """
        Initializes the model analyst with a series of model states.

        :param states: An iterable of `ModelState` objects representing the reservoir model states
            captured at different time steps during a simulation run.
        :param stoiip: Optional pre-calculated stock tank oil initially in place (STB).
            Use this when the initial state (step 0) is not available, e.g., in EOR simulations
            that start from a depleted state.
        :param stgiip: Optional pre-calculated stock tank gas initially in place (including solution gas) (SCF).
            Use this when the initial state (step 0) is not available.
        :param free_giip: Optional pre-calculated free gas initially in place (no solution gas included) (SCF).
            Use this when the initial state (step 0) is not available.
        :param stwiip: Optional pre-calculated stock tank water initially in place (STB).
            Use this when the initial state (step 0) is not available.
        """
        self._states = {int(state.step): state for state in states}

        if not self._states:
            raise ValidationError(
                "No states provided. ModelAnalyst requires at least one state."
            )

        self._min_step = min(self._states.keys())
        self._max_step = max(self._states.keys())
        self._state_count = len(self._states)
        self._sorted_steps = sorted(self._states.keys())

        # Store user-provided initial values for EOR/continuation scenarios
        self._stoiip = stoiip
        self._stgiip = stgiip
        self._stwiip = stwiip
        self._free_giip = free_giip

        if self._max_step != (self._state_count - 1 + self._min_step):
            logger.debug(
                f"Model states have non-sequential time steps. Min step: {self._min_step}, Max step: {self._max_step}, "
                "State count: {self._state_count}. Some production metrics may be inaccurate."
            )

        # We use per-instance memoization caches prevent memory leaks.
        # Using `@functools.cache` on instance methods inserts self into a class-level LRU cache,
        # preventing garbage collection for the lifetime of the class.
        self._cell_area_cache: typing.Dict[typing.Tuple, float] = {}
        self._oil_in_place_cache: typing.Dict[int, float] = {}
        self._free_gas_in_place_cache: typing.Dict[int, float] = {}
        self._total_gas_in_place_cache: typing.Dict[int, float] = {}
        self._water_in_place_cache: typing.Dict[int, float] = {}
        self._oil_produced_cache: typing.Dict[typing.Tuple, float] = {}
        self._gas_produced_cache: typing.Dict[typing.Tuple, float] = {}
        self._water_produced_cache: typing.Dict[typing.Tuple, float] = {}
        self._oil_injected_cache: typing.Dict[typing.Tuple, float] = {}
        self._gas_injected_cache: typing.Dict[typing.Tuple, float] = {}
        self._water_injected_cache: typing.Dict[typing.Tuple, float] = {}
        self._instantaneous_production_rates_cache: typing.Dict[
            typing.Tuple, InstantaneousRates
        ] = {}
        self._instantaneous_injection_rates_cache: typing.Dict[
            typing.Tuple, InstantaneousRates
        ] = {}
        self._productivity_analysis_cache: typing.Dict[
            typing.Tuple, ProductivityAnalysis
        ] = {}

    @property
    def min_step(self) -> int:
        """The minimum (first) step number in the available states."""
        return self._min_step

    @property
    def max_step(self) -> int:
        """The maximum (last) step number in the available states."""
        return self._max_step

    @property
    def available_steps(self) -> typing.List[int]:
        """List of all available step numbers in sorted order."""
        return self._sorted_steps.copy()

    def _resolve_step(self, step: int) -> int:
        """
        Resolve negative step indices to actual step numbers.

        :param step: Step number (can be negative for indexing from end)
        :return: Resolved step number
        """
        if step < 0:
            # -1 should give `max_step`, -2 should give second-to-last, etc.
            return int(self._max_step + step + 1)
        return int(step)

    def get_state(self, step: int) -> typing.Optional[ModelState[NDimension]]:
        """
        Retrieves the model state for a specific time step.

        :param step: The time step index to retrieve the state for.
            Negative values index from the end: -1 is the last step, -2 is second-to-last, etc.
        :return: The `ModelState` corresponding to the specified time step, or None if not found.
        """
        step = self._resolve_step(step)

        state = self._states.get(step, None)
        if state is None:
            logger.debug(
                f"Time step {step} not found. Available time steps: "
                f"{self._sorted_steps}"
            )
        else:
            logger.debug(f"Retrieved state at time step {step}")
        return state

    @property
    def stock_tank_oil_initially_in_place(self) -> float:
        """
        The stock tank oil initially in place (STOIIP) in stock tank barrels (STB).

        If `stoiip` was provided at initialization, returns that value.
        Otherwise, computes from the earliest available state (which may not be step 0
        in EOR/continuation scenarios).
        """
        if self._stoiip is not None:
            return self._stoiip
        return self.oil_in_place(self._min_step)

    stoiip = stock_tank_oil_initially_in_place
    """The stock tank oil initially in place (STOIIP) in stock tank barrels (STB)."""

    @property
    def stock_tank_gas_initially_in_place(self) -> float:
        """
        The stock tank gas initially in place (STGIIP) (including solution gas) in standard cubic feet (SCF).

        If `stgiip` was provided at initialization, returns that value.
        Otherwise, computes from the earliest available state.
        """
        if self._stgiip is not None:
            return self._stgiip
        return self.gas_in_place(self._min_step)

    stgiip = stock_tank_gas_initially_in_place
    """The stock tank gas initially in place (STGIIP) in standard cubic feet (SCF)."""

    @property
    def stock_tank_water_initially_in_place(self) -> float:
        """
        The stock tank water initially in place in stock tank barrels (STB).

        If `stwiip` was provided at initialization, returns that value.
        Otherwise, computes from the earliest available state.
        """
        if self._stwiip is not None:
            return self._stwiip
        return self.water_in_place(self._min_step)

    @property
    def cumulative_oil_produced(self) -> float:
        """The cumulative oil produced in stock tank barrels (STB) from the earliest available state to the latest."""
        return self.oil_produced(self._min_step, -1)

    No = cumulative_oil_produced
    """Cumulative oil produced in stock tank barrels (STB)."""

    @property
    def cumulative_gas_produced(self) -> float:
        """Return the cumulative free gas produced in standard cubic feet (SCF) from the earliest available state to the latest."""
        return self.gas_produced(self._min_step, -1)

    Ng = cumulative_gas_produced
    """Cumulative free gas produced in standard cubic feet (SCF)."""

    @property
    def cumulative_water_produced(self) -> float:
        """Return the cumulative water produced in stock tank barrels (STB) from the earliest available state to the latest."""
        return self.water_produced(self._min_step, -1)

    Nw = cumulative_water_produced
    """Cumulative water produced in stock tank barrels (STB)."""

    @property
    def oil_recovery_factor(self) -> float:
        """
        The oil recovery factor based on initial oil in place and cumulative oil produced
        over the entire simulation period.

        Oil Recovery Factor (RF) = Cumulative Oil Produced / Stock Tank Oil Initially in Place

        This is the fundamental measure of reservoir performance, representing the fraction
        of original oil in place that has been recovered. The recovery factor depends on:

        Primary Recovery Mechanisms:
            - Solution Gas Drive: 5-30% RF (gas expansion as pressure drops)
            - Gas Cap Drive: 20-40% RF (gas cap expansion displaces oil)
            - Natural Water Drive: 35-75% RF (aquifer influx maintains pressure)
            - Gravity Drainage: 40-80% RF (oil drains to bottom under gravity)
            - Combination Drive: 25-50% RF (multiple mechanisms)

        Secondary Recovery (Pressure Maintenance):
            - Water Flooding: 30-60% RF (injected water displaces oil)
            - Gas Injection: 40-70% RF (gas maintains pressure, displaces oil)

        Enhanced Oil Recovery (EOR):
            - Chemical Flooding: 50-70% RF (surfactants reduce interfacial tension)
            - Thermal Recovery: 50-80% RF (steam/combustion reduces viscosity)
            - CO2 Flooding: 40-60% RF (miscible displacement)

        Factors Affecting Recovery:
            - Reservoir properties: porosity, permeability, heterogeneity
            - Fluid properties: viscosity, density, API gravity
            - Drive mechanism: water drive > gas cap > solution gas
            - Well spacing and placement
            - Operating conditions: production rates, pressures
            - Sweep efficiency: areal, vertical, and displacement
            - Residual oil saturation

        Units:
            Numerator: Cumulative oil produced in STB (stock tank barrels)
            Denominator: Initial oil in place in STB (stock tank barrels)
            Result: Dimensionless fraction (0.0 to 1.0, or 0% to 100%)

        Calculation Details:
            STOIIP = Σ [Area x h x φ x (1-Swi) x net_to_gross / Boi]
            Np = Σ [q_oil x Δt / Bo]
            RF = Np / STOIIP

        Where:
            Area = Grid cell area (acres)
            h = Net pay thickness (ft)
            φ = Porosity (fraction)
            Swi = Initial water saturation (fraction)
            net_to_gross = Net-to-gross ratio (fraction)
            Boi = Initial oil formation volume factor (bbl/STB)
            q_oil = Oil production rate (STB/day)
            Δt = Time step (days)
            Bo = Oil formation volume factor (bbl/STB)

        :return: The oil recovery factor as a fraction (0 to 1)

        Example:
        ```python
        analyst = ModelAnalyst(states)
        rf = analyst.oil_recovery_factor
        print(f"Oil Recovery Factor: {rf:.2%}")
        # Oil Recovery Factor: 32.50%

        stoiip = analyst.stoiip
        cumulative = analyst.cumulative_oil_produced
        remaining = stoiip - cumulative
        print(f"Recovered: {cumulative:,.0f} STB ({rf:.1%})")
        print(f"Remaining: {remaining:,.0f} STB ({1-rf:.1%})")
        # Recovered: 1,250,000 STB (32.5%)
        # Remaining: 2,596,154 STB (67.5%)
        ```

        Notes:
            - Recovery factor increases monotonically over production life
            - Ultimate recovery factor (URF) is RF at economic abandonment
            - Low RF (<20%) suggests poor drive mechanism or early production stage
            - High RF (>50%) indicates good drive mechanism or enhanced recovery
            - For mature fields, RF typically plateaus unless EOR is applied
            - Compare RF to material balance drive indices to understand mechanism
            - Use sweep efficiency analysis to identify uncontacted oil
        """
        if self.stock_tank_oil_initially_in_place == 0:
            return 0.0
        return self.cumulative_oil_produced / self.stock_tank_oil_initially_in_place

    @property
    def gas_recovery_factor(self) -> float:
        """
        The total gas recovery factor based on all initial gas (free + solution)
        and all cumulative gas produced over the entire simulation period.

        This comprehensive recovery factor accounts for:
            1. Free gas produced from gas cap or gas phase
            2. Solution gas produced (dissolved gas liberated from oil production)

        Total Gas Recovery Factor = (Free Gas + Solution Gas Produced) /
                                    (Initial Free Gas + Initial Solution Gas)

        Calculation:
            Initial Total Gas = GIIP_free + (STOIIP x Rs_initial)
            Produced Total Gas = G_produced_free + (N_produced x Rs_avg)

        Where:
            GIIP_free = Initial free gas in place (scf)
            STOIIP = Stock tank oil initially in place (STB)
            Rs_initial = Initial solution gas-oil ratio (scf/STB)
            G_produced_free = Cumulative free gas produced (scf)
            N_produced = Cumulative oil produced (STB)
            Rs_avg = Average solution GOR during production (scf/STB)

        Typical Values:
            - Solution gas drive oil reservoirs: 15-30% total gas recovery
            - Gas cap drive oil reservoirs: 30-50% total gas recovery
            - Pure gas reservoirs: 60-90% total gas recovery
            - Combination drive: 25-45% total gas recovery

        Use this metric for:
            - Oil reservoirs with significant solution gas
            - Combined gas cap and solution gas drive
            - Total gas resource recovery assessment
            - Gas sales and revenue calculations
            - Material balance validation

        :return: The total gas recovery factor as a fraction (0 to 1)
        """
        # Get initial gas in place
        stgiip = self.stock_tank_gas_initially_in_place  # scf
        if stgiip == 0:
            return 0.0

        # Get cumulative free gas produced
        free_gas_produced = self.cumulative_gas_produced  # scf (free gas only)

        solution_gas_produced = 0.0
        dissolved_gas_produced = 0.0
        days_per_second = c.DAYS_PER_SECOND
        ft3_to_bbl = c.CUBIC_FEET_TO_BARRELS
        for s in self._sorted_steps:
            st = self._states[s]
            solution_gor_grid = st.model.fluid_properties.solution_gas_to_oil_ratio_grid
            gas_solubility_in_water_grid = (
                st.model.fluid_properties.gas_solubility_in_water_grid
            )
            oil_production = st.production_rates.oil
            water_production = st.production_rates.water

            step_in_days = st.step_size * days_per_second
            oil_fvf_grid = st.model.fluid_properties.oil_formation_volume_factor_grid
            water_fvf_grid = (
                st.model.fluid_properties.water_formation_volume_factor_grid
            )
            oil_production_stb = oil_production * ft3_to_bbl / oil_fvf_grid
            water_production_stb = water_production * ft3_to_bbl / water_fvf_grid
            solution_gas_produced += float(
                np.nansum(solution_gor_grid * oil_production_stb) * step_in_days
            )
            dissolved_gas_produced += float(
                np.nansum(gas_solubility_in_water_grid * water_production_stb)
                * step_in_days
            )

        # Total gas produced
        total_gas_produced = (
            free_gas_produced + solution_gas_produced + dissolved_gas_produced
        )
        return float(total_gas_produced / stgiip)

    def compute_cell_area(self, x_dim: float, y_dim: float) -> float:
        """
        Computes the area of a grid cell in acres.

        :param x_dim: The dimension of the cell in the x-direction (ft).
        :param y_dim: The dimension of the cell in the y-direction (ft).
        :return: The area of the cell in acres.
        """
        key = (x_dim, y_dim)
        if key not in self._cell_area_cache:
            cell_area_in_ft2 = x_dim * y_dim
            self._cell_area_cache[key] = cell_area_in_ft2 * c.SQUARE_FEET_TO_ACRES
        return self._cell_area_cache[key]

    def oil_in_place(self, step: int = -1) -> float:
        """
        Computes the total oil in place at a specific time step.

        :param step: The time step index to compute oil in place for.
        :return: The total oil in place in STB
        """
        step = self._resolve_step(step)
        if step in self._oil_in_place_cache:
            return self._oil_in_place_cache[step]

        stoiip = self._stoiip
        if stoiip is None:
            if self._min_step in self._oil_in_place_cache:
                stoiip = self._oil_in_place_cache[self._min_step]
            else:
                initial_state = self.get_state(self._min_step)
                if initial_state is None:
                    logger.debug(
                        f"State at initial time step {self._min_step} not available. Returning 0.0 for oil in place."
                    )
                    return 0.0

                model = initial_state.model
                cell_area_in_acres = self.compute_cell_area(*model.cell_dimension[:2])
                logger.debug(
                    f"Computing oil in place at time step {self._min_step}, cell area={cell_area_in_acres:.4f} acres"
                )
                cell_area_grid = uniform_grid(
                    grid_shape=model.grid_shape, value=cell_area_in_acres
                )
                stoiip_grid = compute_hydrocarbon_in_place(
                    area=cell_area_grid,
                    thickness=model.thickness_grid,
                    porosity=model.rock_properties.porosity_grid,
                    phase_saturation=model.fluid_properties.oil_saturation_grid,
                    formation_volume_factor=model.fluid_properties.oil_formation_volume_factor_grid,
                    net_to_gross_ratio=model.rock_properties.net_to_gross_ratio_grid,
                    hydrocarbon_type="oil",
                    acre_ft_to_bbl=c.ACRE_FOOT_TO_BARRELS,
                    acre_ft_to_ft3=c.ACRE_FOOT_TO_CUBIC_FEET,
                )
                stoiip = float(np.nansum(stoiip_grid))  # type: ignore[return-value]
                self._oil_in_place_cache[self._min_step] = stoiip

        self._oil_in_place_cache[step] = stoiip - self.oil_produced(
            from_step=self._min_step, to_step=step
        )
        return self._oil_in_place_cache[step]

    def free_gas_in_place(self, step: int = -1) -> float:
        """
        Computes the free gas in place at a specific time step.

        :param step: The time step index to compute gas in place for.
        :return: The total free gas in place in SCF
        """
        step = self._resolve_step(step)
        if step in self._free_gas_in_place_cache:
            return self._free_gas_in_place_cache[step]

        free_giip = self._free_giip
        if free_giip is None:
            if self._min_step in self._free_gas_in_place_cache:
                free_giip = self._free_gas_in_place_cache[self._min_step]
            else:
                initial_state = self.get_state(self._min_step)
                if initial_state is None:
                    logger.debug(
                        f"State at initial time step {self._min_step} not available. Returning 0.0 for free gas in place."
                    )
                    return 0.0

                model = initial_state.model
                cell_area_in_acres = self.compute_cell_area(*model.cell_dimension[:2])
                logger.debug(
                    f"Computing free gas in place at time step {self._min_step}, cell area={cell_area_in_acres:.4f} acres"
                )
                cell_area_grid = uniform_grid(
                    grid_shape=model.grid_shape, value=cell_area_in_acres
                )
                giip_grid = compute_hydrocarbon_in_place(
                    area=cell_area_grid,
                    thickness=model.thickness_grid,
                    porosity=model.rock_properties.porosity_grid,
                    phase_saturation=model.fluid_properties.gas_saturation_grid,
                    formation_volume_factor=model.fluid_properties.gas_formation_volume_factor_grid,
                    net_to_gross_ratio=model.rock_properties.net_to_gross_ratio_grid,
                    hydrocarbon_type="gas",
                    acre_ft_to_bbl=c.ACRE_FOOT_TO_BARRELS,
                    acre_ft_to_ft3=c.ACRE_FOOT_TO_CUBIC_FEET,
                )
                free_giip = float(np.nansum(giip_grid))
                self._free_gas_in_place_cache[self._min_step] = free_giip

        # Current free gas = GIIP - cumulative free gas produced + cumulative gas injected
        self._free_gas_in_place_cache[step] = (
            free_giip
            + self.gas_injected(from_step=self._min_step, to_step=step)
            - self.gas_produced(from_step=self._min_step, to_step=step)
        )
        return self._free_gas_in_place_cache[step]

    def gas_in_place(self, step: int = -1) -> float:
        """
        Computes the total gas in place (including solution gas) at a specific time step.

        :param step: The time step index to compute gas in place for.
        :return: The total gas in place (free + solution) in SCF
        """
        step = self._resolve_step(step)
        if step in self._total_gas_in_place_cache:
            return self._total_gas_in_place_cache[step]

        # STGIIP = GIIP_free + (STOIIP * Rs_initial)
        stgiip = self._stgiip
        if stgiip is None:
            if self._min_step in self._total_gas_in_place_cache:
                stgiip = self._total_gas_in_place_cache[self._min_step]
            else:
                initial_state = self.get_state(self._min_step)
                if initial_state is None:
                    logger.debug(
                        f"State at initial time step {self._min_step} not available. Returning 0.0 for total gas in place."
                    )
                    return 0.0

                model = initial_state.model
                cell_area_in_acres = self.compute_cell_area(*model.cell_dimension[:2])
                logger.debug(
                    f"Computing total gas in place at time step {self._min_step}, cell area={cell_area_in_acres:.4f} acres"
                )
                cell_area_grid = uniform_grid(
                    grid_shape=model.grid_shape, value=cell_area_in_acres
                )
                acre_ft_to_bbl = c.ACRE_FOOT_TO_BARRELS
                acre_ft_to_ft3 = c.ACRE_FOOT_TO_CUBIC_FEET

                giip_grid = compute_hydrocarbon_in_place(
                    area=cell_area_grid,
                    thickness=model.thickness_grid,
                    porosity=model.rock_properties.porosity_grid,
                    phase_saturation=model.fluid_properties.gas_saturation_grid,
                    formation_volume_factor=model.fluid_properties.gas_formation_volume_factor_grid,
                    net_to_gross_ratio=model.rock_properties.net_to_gross_ratio_grid,
                    hydrocarbon_type="gas",
                    acre_ft_to_bbl=acre_ft_to_bbl,
                    acre_ft_to_ft3=acre_ft_to_ft3,
                )
                stoiip_grid = compute_hydrocarbon_in_place(
                    area=cell_area_grid,
                    thickness=model.thickness_grid,
                    porosity=model.rock_properties.porosity_grid,
                    phase_saturation=model.fluid_properties.oil_saturation_grid,
                    formation_volume_factor=model.fluid_properties.oil_formation_volume_factor_grid,
                    net_to_gross_ratio=model.rock_properties.net_to_gross_ratio_grid,
                    hydrocarbon_type="oil",
                    acre_ft_to_bbl=acre_ft_to_bbl,
                    acre_ft_to_ft3=acre_ft_to_ft3,
                )
                stgiip_grid = giip_grid + (
                    stoiip_grid * model.fluid_properties.solution_gas_to_oil_ratio_grid
                )
                stgiip = float(np.nansum(stgiip_grid))
                self._total_gas_in_place_cache[self._min_step] = stgiip

        # Total gas change = free gas produced (net of injection) + solution gas produced with oil
        # Solution gas produced = sum over steps of Rs * oil_production_stb * dt
        # (Rs varies with pressure so we must integrate step by step, not use a single average)
        free_gas_produced = self.gas_produced(from_step=self._min_step, to_step=step)
        free_gas_injected = self.gas_injected(from_step=self._min_step, to_step=step)

        solution_gas_produced = 0.0
        days_per_second = c.DAYS_PER_SECOND
        ft3_to_bbl = c.CUBIC_FEET_TO_BARRELS
        for s in self._sorted_steps:
            if s > step:
                break

            st = self._states[s]
            oil_production = st.production_rates.oil
            step_in_days = st.step_size * days_per_second
            oil_fvf_grid = st.model.fluid_properties.oil_formation_volume_factor_grid
            solution_gor_grid = st.model.fluid_properties.solution_gas_to_oil_ratio_grid
            oil_production_stb = oil_production * ft3_to_bbl / oil_fvf_grid
            solution_gas_produced += float(
                np.nansum(solution_gor_grid * oil_production_stb) * step_in_days
            )

        self._total_gas_in_place_cache[step] = (
            stgiip + free_gas_injected - free_gas_produced - solution_gas_produced
        )
        return self._total_gas_in_place_cache[step]

    def water_in_place(self, step: int = -1) -> float:
        """
        Computes the total water in place at a specific time step.

        :param step: The time step index to compute water in place for.
        :return: The total water in place in STB
        """
        step = self._resolve_step(step)
        if step in self._water_in_place_cache:
            return self._water_in_place_cache[step]

        stwiip = self._stwiip
        if stwiip is None:
            if self._min_step in self._water_in_place_cache:
                stwiip = self._water_in_place_cache[self._min_step]
            else:
                initial_state = self.get_state(self._min_step)
                if initial_state is None:
                    logger.debug(
                        f"State at initial time step {self._min_step} not available. Returning 0.0 for water in place."
                    )
                    return 0.0

                model = initial_state.model
                logger.debug(f"Computing water in place at time step {self._min_step}")
                cell_area_in_acres = self.compute_cell_area(*model.cell_dimension[:2])
                cell_area_grid = uniform_grid(
                    grid_shape=model.grid_shape, value=cell_area_in_acres
                )
                stwiip_grid = compute_hydrocarbon_in_place(
                    area=cell_area_grid,
                    thickness=model.thickness_grid,
                    porosity=model.rock_properties.porosity_grid,
                    phase_saturation=model.fluid_properties.water_saturation_grid,
                    formation_volume_factor=model.fluid_properties.water_formation_volume_factor_grid,
                    net_to_gross_ratio=model.rock_properties.net_to_gross_ratio_grid,
                    hydrocarbon_type="water",
                    acre_ft_to_bbl=c.ACRE_FOOT_TO_BARRELS,
                    acre_ft_to_ft3=c.ACRE_FOOT_TO_CUBIC_FEET,
                )
                stwiip = float(np.nansum(stwiip_grid))
                self._water_in_place_cache[self._min_step] = stwiip

        # Current water = initial water + injected - produced
        self._water_in_place_cache[step] = (
            stwiip
            + self.water_injected(from_step=self._min_step, to_step=step)
            - self.water_produced(from_step=self._min_step, to_step=step)
        )
        return self._water_in_place_cache[step]

    def oil_in_place_history(
        self, from_step: int = 0, to_step: int = -1, interval: int = 1
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Get the oil in place history between two time steps.

        :param from_step: The starting time step index (inclusive).
        :param to_step: The ending time step index (inclusive). Use -1 for the last step.
        :return: A generator yielding tuples of time step and oil in place in (STB).
        """
        to_step = self._resolve_step(to_step)

        for t in range(from_step, to_step + 1, interval):
            if t in self._states:
                yield (t, self.oil_in_place(t))

    def gas_in_place_history(
        self, from_step: int = 0, to_step: int = -1, interval: int = 1
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Computes the free gas in place history between two time steps.

        :param from_step: The starting time step index (inclusive).
        :param to_step: The ending time step index (inclusive). Use -1 for the last step.
        :return: A generator yielding tuples of time step and gas in place in (SCF).
        """
        to_step = self._resolve_step(to_step)

        for t in range(from_step, to_step + 1, interval):
            if t in self._states:
                yield (t, self.free_gas_in_place(t))

    def water_in_place_history(
        self, from_step: int = 0, to_step: int = -1, interval: int = 1
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Computes the water in place history between two time steps.

        :param from_step: The starting time step index (inclusive).
        :param to_step: The ending time step index (inclusive). Use -1 for the last step.
        :return: A generator yielding tuples of time step and water in place in (STB).
        """
        to_step = self._resolve_step(to_step)

        for t in range(from_step, to_step + 1, interval):
            if t in self._states:
                yield (t, self.water_in_place(t))

    def oil_produced(
        self,
        from_step: int,
        to_step: int,
        cells: typing.Union[Cells, CellFilter] = None,
    ) -> float:
        """
        Computes the cumulative oil produced between two time steps.

        If:
        - production rates are present, they contribute positively to production.
        - `from_step` equals `to_step`, the production at that time step is returned.
        - `from_step` is min_step and `to_step` is -1, the total cumulative production is returned.

        :param from_step: The starting time step index (inclusive).
        :param to_step: The ending time step index (inclusive). Use -1 for the last step.
        :param cells: Optional filter for specific cells, well name, or region.
            - None: Entire reservoir (default)
            - str: Well name (e.g., "PROD-1")
            - (i, j, k): Single cell
            - ((i1,j1,k1), ...): Tuple of cells
            - (slice, slice, slice): Region
            - Cells object: Pre-constructed Cells instance
        :return: The cumulative oil produced in STB
        """
        # Resolve step in the public method so cache key is canonical
        # (oil_produced(0, -1) and oil_produced(0, 9) won't be cached separately when 9 is last)
        to_step = self._resolve_step(to_step)
        # Convert cells to Cells (hashable) before delegating to cached implementation
        cells_obj = _ensure_cells(cells)
        return self._oil_produced(from_step, to_step, cells_obj)

    def _oil_produced(
        self,
        from_step: int,
        to_step: int,
        cells_obj: typing.Optional[Cells],
    ) -> float:
        """Internal cached implementation of `oil_produced`."""
        key = (from_step, to_step, cells_obj)
        if key in self._oil_produced_cache:
            return self._oil_produced_cache[key]

        logger.debug(
            f"Computing oil produced from time step {from_step} to {to_step}, cells filter: {cells_obj}"
        )
        total_production = 0.0

        # Compute mask once before the loop; `grid_shape` is constant across steps
        mask = None
        if cells_obj is not None:
            first_state = next(
                (
                    self._states[t]
                    for t in range(from_step, to_step + 1)
                    if t in self._states
                ),
                None,
            )
            if first_state is not None:
                mask = cells_obj.get_mask(
                    first_state.model.grid_shape, first_state.wells
                )

        days_per_second = c.DAYS_PER_SECOND
        ft3_to_bbl = c.CUBIC_FEET_TO_BARRELS
        for t in range(from_step, to_step + 1):
            if t not in self._states:
                continue
            state = self._states[t]

            # Production is in ft³/day, convert to STB using FVF
            oil_production = state.production_rates.oil
            step_in_days = state.step_size * days_per_second
            oil_fvf_grid = state.model.fluid_properties.oil_formation_volume_factor_grid
            oil_production_stb = oil_production * ft3_to_bbl / oil_fvf_grid

            # Apply mask if filtering
            if mask is not None:
                oil_production_stb = oil_production_stb * mask

            total_production += np.nansum(oil_production_stb * step_in_days)

        self._oil_produced_cache[key] = float(total_production)
        return self._oil_produced_cache[key]

    def gas_produced(
        self,
        from_step: int,
        to_step: int,
        cells: typing.Union[Cells, CellFilter] = None,
    ) -> float:
        """
        Computes the cumulative free gas produced between two time steps.

        If:
        - production rates are present, they contribute positively to production.
        - `from_step` equals `to_step`, the production at that time step is returned.
        - `from_step` is 0 and `to_step` is -1, the total cumulative production is returned.

        :param from_step: The starting time step index (inclusive).
        :param to_step: The ending time step index (inclusive).
        :param cells: Optional filter for specific cells, well name, or region.
            - None: Entire reservoir (default)
            - str: Well name (e.g., "PROD-1")
            - (i, j, k): Single cell
            - [(i1,j1,k1), ...]: List of cells
            - (slice, slice, slice): Region
        :return: The cumulative gas produced in SCF
        """
        to_step = self._resolve_step(to_step)
        cells_obj = _ensure_cells(cells)
        return self._gas_produced(from_step, to_step, cells_obj)

    def _gas_produced(
        self,
        from_step: int,
        to_step: int,
        cells_obj: typing.Optional[Cells],
    ) -> float:
        """Internal cached implementation of `gas_produced`."""
        key = (from_step, to_step, cells_obj)
        if key in self._gas_produced_cache:
            return self._gas_produced_cache[key]

        total_production = 0.0
        mask = None
        if cells_obj is not None:
            first_state = next(
                (
                    self._states[t]
                    for t in range(from_step, to_step + 1)
                    if t in self._states
                ),
                None,
            )
            if first_state is not None:
                mask = cells_obj.get_mask(
                    first_state.model.grid_shape, first_state.wells
                )

        days_per_second = c.DAYS_PER_SECOND
        for t in range(from_step, to_step + 1):
            if t not in self._states:
                continue
            state = self._states[t]

            # Production is in ft³/day, convert to SCF using FVF
            gas_production = state.production_rates.gas
            step_in_days = state.step_size * days_per_second
            gas_fvf_grid = state.model.fluid_properties.gas_formation_volume_factor_grid
            gas_production_scf = gas_production / gas_fvf_grid

            # Apply mask if filtering
            if mask is not None:
                gas_production_scf = gas_production_scf * mask

            total_production += np.nansum(gas_production_scf * step_in_days)

        self._gas_produced_cache[key] = float(total_production)
        return self._gas_produced_cache[key]

    def water_produced(
        self,
        from_step: int,
        to_step: int,
        cells: typing.Union[Cells, CellFilter] = None,
    ) -> float:
        """
        Computes the cumulative water produced between two time steps.

        If:
        - production rates are present, they contribute positively to production.
        - `from_step` equals `to_step`, the production at that time step is returned.
        - `from_step` is 0 and `to_step` is -1, the total cumulative production is returned.

        :param from_step: The starting time step index (inclusive).
        :param to_step: The ending time step index (inclusive).
        :param cells: Optional filter for specific cells, well name, or region.
            - None: Entire reservoir (default)
            - str: Well name (e.g., "PROD-1")
            - (i, j, k): Single cell
            - [(i1,j1,k1), ...]: List of cells
            - (slice, slice, slice): Region
        :return: The cumulative water produced in STB
        """
        to_step = self._resolve_step(to_step)
        cells_obj = _ensure_cells(cells)
        return self._water_produced(from_step, to_step, cells_obj)

    def _water_produced(
        self,
        from_step: int,
        to_step: int,
        cells_obj: typing.Optional[Cells],
    ) -> float:
        """Internal cached implementation of `water_produced`."""
        key = (from_step, to_step, cells_obj)
        if key in self._water_produced_cache:
            return self._water_produced_cache[key]

        total_production = 0.0
        mask = None
        if cells_obj is not None:
            first_state = next(
                (
                    self._states[t]
                    for t in range(from_step, to_step + 1)
                    if t in self._states
                ),
                None,
            )
            if first_state is not None:
                mask = cells_obj.get_mask(
                    first_state.model.grid_shape, first_state.wells
                )

        days_per_second = c.DAYS_PER_SECOND
        ft3_to_bbl = c.CUBIC_FEET_TO_BARRELS
        for t in range(from_step, to_step + 1):
            if t not in self._states:
                continue
            state = self._states[t]

            # Production is in ft³/day, convert to STB using FVF
            water_production = state.production_rates.water
            if water_production is None:
                continue

            step_in_days = state.step_size * days_per_second
            water_fvf_grid = (
                state.model.fluid_properties.water_formation_volume_factor_grid
            )
            water_production_stb = water_production * ft3_to_bbl / water_fvf_grid

            # Apply mask if filtering
            if mask is not None:
                water_production_stb = water_production_stb * mask

            total_production += np.nansum(water_production_stb * step_in_days)

        self._water_produced_cache[key] = float(total_production)
        return self._water_produced_cache[key]

    def oil_injected(
        self,
        from_step: int,
        to_step: int,
        cells: typing.Union[Cells, CellFilter] = None,
    ) -> float:
        """
        Computes the cumulative oil injected between two time steps.

        If:
        - injection rates are present, they contribute positively to injection.
        - `from_step` equals `to_step`, the injection at that time step is returned.
        - `from_step` is 0 and `to_step` is -1, the total cumulative injection is returned.

        :param from_step: The starting time step index (inclusive).
        :param to_step: The ending time step index (inclusive).
        :param cells: Optional filter for specific cells, well name, or region.
            - None: Entire reservoir (default)
            - str: Well name (e.g., "INJ-1")
            - (i, j, k): Single cell
            - [(i1,j1,k1), ...]: List of cells
            - (slice, slice, slice): Region
        :return: The cumulative oil injected in STB
        """
        to_step = self._resolve_step(to_step)
        cells_obj = _ensure_cells(cells)
        return self._oil_injected(from_step, to_step, cells_obj)

    def _oil_injected(
        self,
        from_step: int,
        to_step: int,
        cells_obj: typing.Optional[Cells],
    ) -> float:
        """Internal cached implementation of `oil_injected`."""
        key = (from_step, to_step, cells_obj)
        if key in self._oil_injected_cache:
            return self._oil_injected_cache[key]

        total_injection = 0.0
        # Compute mask once before the loop
        mask = None
        if cells_obj is not None:
            first_state = next(
                (
                    self._states[t]
                    for t in range(from_step, to_step + 1)
                    if t in self._states
                ),
                None,
            )
            if first_state is not None:
                mask = cells_obj.get_mask(
                    first_state.model.grid_shape, first_state.wells
                )

        days_per_second = c.DAYS_PER_SECOND
        ft3_to_bbl = c.CUBIC_FEET_TO_BARRELS
        for t in range(from_step, to_step + 1):
            if t not in self._states:
                continue
            state = self._states[t]

            # Injection is in ft³/day, convert to STB using FVF
            oil_injection = state.injection_rates.oil
            step_in_days = state.step_size * days_per_second
            oil_fvf_grid = state.model.fluid_properties.oil_formation_volume_factor_grid
            oil_injection_stb = oil_injection * ft3_to_bbl / oil_fvf_grid

            # Apply mask if filtering
            if mask is not None:
                oil_injection_stb = oil_injection_stb * mask

            total_injection += np.nansum(oil_injection_stb * step_in_days)

        self._oil_injected_cache[key] = float(total_injection)
        return self._oil_injected_cache[key]

    def gas_injected(
        self,
        from_step: int,
        to_step: int,
        cells: typing.Union[Cells, CellFilter] = None,
    ) -> float:
        """
        Computes the cumulative gas injected between two time steps.

        If:
        - injection rates are present, they contribute positively to injection.
        - `from_step` equals `to_step`, the injection at that time step is returned.
        - `from_step` is 0 and `to_step` is -1, the total cumulative injection is returned.

        :param from_step: The starting time step index (inclusive).
        :param to_step: The ending time step index (inclusive).
        :param cells: Optional filter for specific cells, well name, or region.
            - None: Entire reservoir (default)
            - str: Well name (e.g., "INJ-1")
            - (i, j, k): Single cell
            - [(i1,j1,k1), ...]: List of cells
            - (slice, slice, slice): Region
        :return: The cumulative gas injected in SCF
        """
        to_step = self._resolve_step(to_step)
        cells_obj = _ensure_cells(cells)
        return self._gas_injected(from_step, to_step, cells_obj)

    def _gas_injected(
        self,
        from_step: int,
        to_step: int,
        cells_obj: typing.Optional[Cells],
    ) -> float:
        """Internal cached implementation of `gas_injected`."""
        # Per-instance dict cache prevents memory leaks
        key = (from_step, to_step, cells_obj)
        if key in self._gas_injected_cache:
            return self._gas_injected_cache[key]

        total_injection = 0.0
        mask = None
        if cells_obj is not None:
            first_state = next(
                (
                    self._states[t]
                    for t in range(from_step, to_step + 1)
                    if t in self._states
                ),
                None,
            )
            if first_state is not None:
                mask = cells_obj.get_mask(
                    first_state.model.grid_shape, first_state.wells
                )

        days_per_second = c.DAYS_PER_SECOND
        for t in range(from_step, to_step + 1):
            if t not in self._states:
                continue
            state = self._states[t]

            # Injection is in ft³/day, convert to SCF using FVF
            gas_injection = state.injection_rates.gas
            step_in_days = state.step_size * days_per_second
            injected_gas_fvf_grid, _ = _build_injected_fvf_grids(
                wells=state.wells,
                pressure_grid=state.model.fluid_properties.pressure_grid,
                temperature_grid=state.model.fluid_properties.temperature_grid,
                grid_shape=state.model.grid_shape,
            )
            # NaN where no injector: `gas_injection` is also 0 there, nansum skips correctly
            gas_injection_scf = gas_injection / injected_gas_fvf_grid

            # Apply mask if filtering
            if mask is not None:
                gas_injection_scf = gas_injection_scf * mask

            total_injection += np.nansum(gas_injection_scf * step_in_days)

        self._gas_injected_cache[key] = float(total_injection)
        return self._gas_injected_cache[key]

    def water_injected(
        self,
        from_step: int,
        to_step: int,
        cells: typing.Union[Cells, CellFilter] = None,
    ) -> float:
        """
        Computes the cumulative water injected between two time steps.

        If:
        - injection rates are present, they contribute positively to injection.
        - `from_step` equals `to_step`, the injection at that time step is returned.
        - `from_step` is 0 and `to_step` is -1, the total cumulative injection is returned.

        :param from_step: The starting time step index (inclusive).
        :param to_step: The ending time step index (inclusive).
        :param cells: Optional filter for specific cells, well name, or region.
            - None: Entire reservoir (default)
            - str: Well name (e.g., "INJ-1")
            - (i, j, k): Single cell
            - [(i1,j1,k1), ...]: List of cells
            - (slice, slice, slice): Region
        :return: The cumulative water injected in STB
        """
        to_step = self._resolve_step(to_step)
        cells_obj = _ensure_cells(cells)
        return self._water_injected(from_step, to_step, cells_obj)

    def _water_injected(
        self,
        from_step: int,
        to_step: int,
        cells_obj: typing.Optional[Cells],
    ) -> float:
        """Internal cached implementation of `water_injected`."""
        key = (from_step, to_step, cells_obj)
        if key in self._water_injected_cache:
            return self._water_injected_cache[key]

        total_injection = 0.0
        mask = None
        if cells_obj is not None:
            first_state = next(
                (
                    self._states[t]
                    for t in range(from_step, to_step + 1)
                    if t in self._states
                ),
                None,
            )
            if first_state is not None:
                mask = cells_obj.get_mask(
                    first_state.model.grid_shape, first_state.wells
                )

        days_per_second = c.DAYS_PER_SECOND
        ft3_to_bbl = c.CUBIC_FEET_TO_BARRELS
        for t in range(from_step, to_step + 1):
            if t not in self._states:
                continue
            state = self._states[t]

            # Injection is in ft³/day, convert to STB using FVF
            water_injection = state.injection_rates.water
            step_in_days = state.step_size * days_per_second
            _, injected_water_fvf_grid = _build_injected_fvf_grids(
                wells=state.wells,
                pressure_grid=state.model.fluid_properties.pressure_grid,
                temperature_grid=state.model.fluid_properties.temperature_grid,
                grid_shape=state.model.grid_shape,
            )
            water_injection_stb = water_injection * ft3_to_bbl / injected_water_fvf_grid

            # Apply mask if filtering
            if mask is not None:
                water_injection_stb = water_injection_stb * mask

            total_injection += np.nansum(water_injection_stb * step_in_days)

        self._water_injected_cache[key] = float(total_injection)
        return self._water_injected_cache[key]

    def oil_production_history(
        self,
        from_step: int = 0,
        to_step: int = -1,
        interval: int = 1,
        cumulative: bool = False,
        cells: CellFilter = None,
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Get the oil production history between two time steps.

        :param from_step: The starting time step index (inclusive).
        :param to_step: The ending time step index (inclusive).
        :param interval: Time step interval for sampling.
        :param cumulative: If True (default), returns cumulative production from start. If False, returns production at each time step.
        :param cells: Optional filter for specific cells, well name, or region.
            - None: Entire reservoir (default)
            - str: Well name (e.g., "PROD-1")
            - (i, j, k): Single cell
            - [(i1,j1,k1), ...]: List of cells
            - (slice, slice, slice): Region
        :return: A generator yielding tuples of time step and oil produced (cumulative or exclusive).
        """
        cells_obj = _ensure_cells(cells)
        to_step = self._resolve_step(to_step)

        if cumulative:
            # Use incremental accumulation instead of recalculating from `min_step` each time
            cumulative_total = 0.0
            # First, catch up from `min_step` to `from_step` - 1 (if needed)
            if from_step > self._min_step:
                cumulative_total = self.oil_produced(
                    self._min_step, from_step - 1, cells=cells_obj
                )

            for t in range(from_step, to_step + 1, interval):
                # Add production for steps since last yield
                if t == from_step:
                    cumulative_total += self.oil_produced(
                        from_step, from_step, cells=cells_obj
                    )
                else:
                    # Add production from last yielded step to current step
                    prev_t = t - interval
                    cumulative_total += self.oil_produced(
                        prev_t + 1, t, cells=cells_obj
                    )
                yield (t, cumulative_total)
        else:
            for t in range(from_step, to_step + 1, interval):
                # Calculate production at time step t (exclusive)
                # Use time step t for both from and to to get production at that step
                yield (t, self.oil_produced(t, t, cells=cells_obj))

    def gas_production_history(
        self,
        from_step: int = 0,
        to_step: int = -1,
        interval: int = 1,
        cumulative: bool = False,
        cells: CellFilter = None,
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Get the free gas production history between two time steps.

        :param from_step: The starting time step index (inclusive).
        :param to_step: The ending time step index (inclusive).
        :param interval: Time step interval for sampling.
        :param cumulative: If True (default), returns cumulative production from start. If False, returns production at each time step.
        :param cells: Optional filter for specific cells, well name, or region.
            - None: Entire reservoir (default)
            - str: Well name (e.g., "PROD-1")
            - (i, j, k): Single cell
            - [(i1,j1,k1), ...]: List of cells
            - (slice, slice, slice): Region
        :return: A generator yielding tuples of time step and gas produced (cumulative or exclusive).
        """
        cells_obj = _ensure_cells(cells)
        to_step = self._resolve_step(to_step)

        if cumulative:
            # Use incremental accumulation instead of recalculating from `min_step` each time
            cumulative_total = 0.0
            # First, catch up from `min_step` to `from_step` - 1 (if needed)
            if from_step > self._min_step:
                cumulative_total = self.gas_produced(
                    self._min_step, from_step - 1, cells=cells_obj
                )

            for t in range(from_step, to_step + 1, interval):
                # Add production for steps since last yield
                if t == from_step:
                    cumulative_total += self.gas_produced(
                        from_step, from_step, cells=cells_obj
                    )
                else:
                    # Add production from last yielded step to current step
                    prev_t = t - interval
                    cumulative_total += self.gas_produced(
                        prev_t + 1, t, cells=cells_obj
                    )
                yield (t, cumulative_total)
        else:
            for t in range(from_step, to_step + 1, interval):
                # Calculate production at time step t (exclusive)
                # Use time step t for both from and to to get production at that step
                yield (t, self.gas_produced(t, t, cells=cells_obj))

    def water_production_history(
        self,
        from_step: int = 0,
        to_step: int = -1,
        interval: int = 1,
        cumulative: bool = False,
        cells: CellFilter = None,
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Get the water production history between two time steps.

        :param from_step: The starting time step index (inclusive).
        :param to_step: The ending time step index (inclusive).
        :param interval: Time step interval for sampling.
        :param cumulative: If True (default), returns cumulative production from start. If False, returns production at each time step.
        :param cells: Optional filter for specific cells, well name, or region.
            - None: Entire reservoir (default)
            - str: Well name (e.g., "PROD-1")
            - (i, j, k): Single cell
            - [(i1,j1,k1), ...]: List of cells
            - (slice, slice, slice): Region
        :return: A generator yielding tuples of time step and water produced (cumulative or exclusive).
        """
        cells_obj = _ensure_cells(cells)
        to_step = self._resolve_step(to_step)

        if cumulative:
            # Use incremental accumulation instead of recalculating from `min_step` each time
            cumulative_total = 0.0
            # First, catch up from `min_step` to `from_step` - 1 (if needed)
            if from_step > self._min_step:
                cumulative_total = self.water_produced(
                    self._min_step, from_step - 1, cells=cells_obj
                )

            for t in range(from_step, to_step + 1, interval):
                # Add production for steps since last yield
                if t == from_step:
                    cumulative_total += self.water_produced(
                        from_step, from_step, cells=cells_obj
                    )
                else:
                    # Add production from last yielded step to current step
                    prev_t = t - interval
                    cumulative_total += self.water_produced(
                        prev_t + 1, t, cells=cells_obj
                    )
                yield (t, cumulative_total)
        else:
            for t in range(from_step, to_step + 1, interval):
                # Calculate production at time step t (exclusive)
                # Use time step t for both from and to to get production at that step
                yield (t, self.water_produced(t, t, cells=cells_obj))

    def oil_injection_history(
        self,
        from_step: int = 0,
        to_step: int = -1,
        interval: int = 1,
        cumulative: bool = False,
        cells: CellFilter = None,
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Get the oil injection history between two time steps.

        :param from_step: The starting time step index (inclusive).
        :param to_step: The ending time step index (inclusive).
        :param interval: Time step interval for sampling.
        :param cumulative: If True (default), returns cumulative injection from start. If False, returns injection at each time step.
        :param cells: Optional filter for specific cells, well name, or region.
            - None: Entire reservoir (default)
            - str: Well name (e.g., "INJ-1")
            - (i, j, k): Single cell
            - [(i1,j1,k1), ...]: List of cells
            - (slice, slice, slice): Region
        :return: A generator yielding tuples of time step and oil injected (cumulative or exclusive).
        """
        cells_obj = _ensure_cells(cells)
        to_step = self._resolve_step(to_step)

        if cumulative:
            # Use incremental accumulation instead of recalculating from `min_step` each time
            cumulative_total = 0.0
            # First, catch up from `min_step` to `from_step` - 1 (if needed)
            if from_step > self._min_step:
                cumulative_total = self.oil_injected(
                    self._min_step, from_step - 1, cells=cells_obj
                )

            for t in range(from_step, to_step + 1, interval):
                # Add injection for steps since last yield
                if t == from_step:
                    cumulative_total += self.oil_injected(
                        from_step, from_step, cells=cells_obj
                    )
                else:
                    # Add injection from last yielded step to current step
                    prev_t = t - interval
                    cumulative_total += self.oil_injected(
                        prev_t + 1, t, cells=cells_obj
                    )
                yield (t, cumulative_total)
        else:
            for t in range(from_step, to_step + 1, interval):
                # Calculate injection at time step t (exclusive)
                # Use time step t for both from and to to get injection at that step
                yield (t, self.oil_injected(t, t, cells=cells_obj))

    def gas_injection_history(
        self,
        from_step: int = 0,
        to_step: int = -1,
        interval: int = 1,
        cumulative: bool = False,
        cells: CellFilter = None,
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Get the gas injection history between two time steps.

        :param from_step: The starting time step index (inclusive).
        :param to_step: The ending time step index (inclusive).
        :param interval: Time step interval for sampling.
        :param cumulative: If True (default), returns cumulative injection from start. If False, returns injection at each time step.
        :param cells: Optional filter for specific cells, well name, or region.
            - None: Entire reservoir (default)
            - str: Well name (e.g., "INJ-1")
            - (i, j, k): Single cell
            - [(i1,j1,k1), ...]: List of cells
            - (slice, slice, slice): Region
        :return: A generator yielding tuples of time step and gas injected (cumulative or exclusive).
        """
        cells_obj = _ensure_cells(cells)
        to_step = self._resolve_step(to_step)

        if cumulative:
            # Use incremental accumulation instead of recalculating from `min_step` each time
            cumulative_total = 0.0
            # First, catch up from `min_step` to `from_step` - 1 (if needed)
            if from_step > self._min_step:
                cumulative_total = self.gas_injected(
                    self._min_step, from_step - 1, cells=cells_obj
                )

            for t in range(from_step, to_step + 1, interval):
                # Add injection for steps since last yield
                if t == from_step:
                    cumulative_total += self.gas_injected(
                        from_step, from_step, cells=cells_obj
                    )
                else:
                    # Add injection from last yielded step to current step
                    prev_t = t - interval
                    cumulative_total += self.gas_injected(
                        prev_t + 1, t, cells=cells_obj
                    )
                yield (t, cumulative_total)
        else:
            for t in range(from_step, to_step + 1, interval):
                # Calculate injection at time step t (exclusive)
                # Use time step t for both from and to to get injection at that step
                yield (t, self.gas_injected(t, t, cells=cells_obj))

    def water_injection_history(
        self,
        from_step: int = 0,
        to_step: int = -1,
        interval: int = 1,
        cumulative: bool = False,
        cells: CellFilter = None,
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Get the water injection history between two time steps.

        :param from_step: The starting time step index (inclusive).
        :param to_step: The ending time step index (inclusive).
        :param interval: Time step interval for sampling.
        :param cumulative: If True (default), returns cumulative injection from start. If False, returns injection at each time step.
        :param cells: Optional filter for specific cells, well name, or region.
            - None: Entire reservoir (default)
            - str: Well name (e.g., "INJ-1")
            - (i, j, k): Single cell
            - [(i1,j1,k1), ...]: List of cells
            - (slice, slice, slice): Region
        :return: A generator yielding tuples of time step and water injected (cumulative or exclusive).
        """
        cells_obj = _ensure_cells(cells)
        to_step = self._resolve_step(to_step)

        if cumulative:
            # Use incremental accumulation instead of recalculating from `min_step` each time
            cumulative_total = 0.0
            # First, catch up from `min_step` to `from_step` - 1 (if needed)
            if from_step > self._min_step:
                cumulative_total = self.water_injected(
                    self._min_step, from_step - 1, cells=cells_obj
                )

            for t in range(from_step, to_step + 1, interval):
                # Add injection for steps since last yield
                if t == from_step:
                    cumulative_total += self.water_injected(
                        from_step, from_step, cells=cells_obj
                    )
                else:
                    # Add injection from last yielded step to current step
                    prev_t = t - interval
                    cumulative_total += self.water_injected(
                        prev_t + 1, t, cells=cells_obj
                    )
                yield (t, cumulative_total)
        else:
            for t in range(from_step, to_step + 1, interval):
                # Calculate injection at time step t (exclusive)
                # Use time step t for both from and to to get injection at that step
                yield (t, self.water_injected(t, t, cells=cells_obj))

    def oil_recovery_factor_history(
        self,
        from_step: int = 0,
        to_step: int = -1,
        interval: int = 1,
        cells: CellFilter = None,
        stoiip: typing.Optional[float] = None,
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Get the oil recovery factor history over time.

        This function computes the oil recovery factor at each time step, showing
        how the recovery factor evolves throughout the simulation. The recovery factor
        at each time step is calculated as:

        RF(t) = Cumulative Oil Produced(0, t) / Stock Tank Oil Initially in Place

        This allows tracking of:
        - Recovery factor growth rate over time
        - Identification of plateau periods
        - Evaluation of different recovery stages (primary, secondary, tertiary)
        - Comparison of recovery efficiency between time periods

        :param from_step: The starting time step index (inclusive). Default is 0.
        :param to_step: The ending time step index (inclusive). Default is -1 (last time step).
        :param interval: Time step interval for sampling. Default is 1.
        :param cells: Optional filter for specific cells, well name, or region.
            - None: Entire reservoir (default)
            - str: Well name (e.g., "PROD-1")
            - (i, j, k): Single cell
            - [(i1,j1,k1), ...]: List of cells
            - (slice, slice, slice): Region
            When cells is specified, STOIIP is also calculated for the same filtered region.
        :param stoiip: Optional pre-calculated STOIIP value to use instead of computing from initial state.
            This will be needed if the initial state of the reservoir is not available in the states provided.
            Especially the the case of EOR simulations where the initial state may not be included, and its starting
            point is after some production has already occurred.
        :return: A generator yielding tuples of (step, recovery_factor).

        Example:
        ```python
        analyst = ModelAnalyst(states)

        # Get full recovery factor history
        for t, rf in analyst.oil_recovery_factor_history():
            print(f"Time step {t}: RF = {rf:.2%}")

        # Plot recovery factor evolution
        times, rfs = zip(*analyst.oil_recovery_factor_history())
        plt.plot(times, rfs)
        plt.ylabel("Oil Recovery Factor")
        plt.xlabel("Time Step")
        ```
        """
        cells_obj = _ensure_cells(cells)
        to_step = self._resolve_step(to_step)

        # If cells filter is specified, compute STOIIP for that region
        if stoiip is not None:
            oiip = stoiip
        elif cells_obj is None:
            oiip = self.stock_tank_oil_initially_in_place
        else:
            initial_state = self.get_state(self._min_step)
            if initial_state is None:
                raise ValidationError(
                    f"Initial state (step {self._min_step}) not available"
                )
            mask = cells_obj.get_mask(
                initial_state.model.grid_shape, initial_state.wells
            )
            model = initial_state.model
            oil_saturation = model.fluid_properties.oil_saturation_grid
            oil_fvf = model.fluid_properties.oil_formation_volume_factor_grid
            porosity = model.rock_properties.porosity_grid
            net_to_gross = model.rock_properties.net_to_gross_ratio_grid
            thickness = model.thickness_grid
            cell_area_in_acres = self.compute_cell_area(*model.cell_dimension[:2])
            cell_area_grid = uniform_grid(
                grid_shape=model.grid_shape, value=cell_area_in_acres
            )
            stoiip_grid = compute_hydrocarbon_in_place(
                area=cell_area_grid,
                thickness=thickness,
                porosity=porosity,
                phase_saturation=oil_saturation,
                formation_volume_factor=oil_fvf,
                net_to_gross_ratio=net_to_gross,
                hydrocarbon_type="oil",
                acre_ft_to_bbl=c.ACRE_FOOT_TO_BARRELS,
                acre_ft_to_ft3=c.ACRE_FOOT_TO_CUBIC_FEET,
            )
            if mask is not None:
                stoiip_grid = np.where(mask, stoiip_grid, 0.0)
            oiip = float(np.nansum(stoiip_grid))

        if oiip == 0:
            # If no initial oil, recovery factor is always 0
            for t in range(from_step, to_step + 1, interval):
                yield (t, 0.0)
        else:
            for t in range(from_step, to_step + 1, interval):
                cumulative_oil_produced = self.oil_produced(
                    self._min_step, t, cells=cells_obj
                )
                rf = cumulative_oil_produced / oiip
                yield (t, rf)

    def gas_recovery_factor_history(
        self,
        from_step: int = 0,
        to_step: int = -1,
        interval: int = 1,
        cells: CellFilter = None,
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Get the total gas recovery factor history over time.

        This function computes the total gas recovery factor (free + solution gas) at each
        time step, showing how total gas recovery evolves throughout the simulation.

        Total Gas RF(t) = (Free Gas Produced(0,t) + Solution Gas Produced(0,t)) /
                          (Initial Free Gas + Initial Solution Gas)

        :param from_step: The starting time step index (inclusive). Default is 0.
        :param to_step: The ending time step index (inclusive). Default is -1 (last time step).
        :param interval: Time step interval for sampling. Default is 1.
        :param cells: Optional filter for specific cells, well name, or region.
            - None: Entire reservoir (default)
            - str: Well name (e.g., "PROD-1")
            - (i, j, k): Single cell
            - [(i1,j1,k1), ...]: List of cells
            - (slice, slice, slice): Region
            When cells is specified, initial gas is also calculated for the same filtered region.
        :return: A generator yielding tuples of (step, recovery_factor).

        Example:
        ```python
        analyst = ModelAnalyst(states)

        # Track total gas recovery (free + solution) over time
        for t, rf in analyst.gas_recovery_factor_history():
            print(f"Time step {t}: Total Gas RF = {rf:.2%}")
        ```
        """
        cells_obj = _ensure_cells(cells)
        to_step = self._resolve_step(to_step)

        initial_state = self.get_state(self._min_step)
        if initial_state is None:
            raise ValidationError(
                f"Initial state (time step {self._min_step}) is not available. Cannot compute total gas recovery factor history."
            )

        # Get initial total gas
        if cells_obj is None:
            total_initial_gas = self.stock_tank_gas_initially_in_place
        else:
            mask = cells_obj.get_mask(
                initial_state.model.grid_shape, initial_state.wells
            )
            model = initial_state.model

            # Calculate GIIP for the filtered region
            gas_saturation = model.fluid_properties.gas_saturation_grid
            gas_fvf = model.fluid_properties.gas_formation_volume_factor_grid
            porosity = model.rock_properties.porosity_grid
            net_to_gross = model.rock_properties.net_to_gross_ratio_grid
            thickness = model.thickness_grid
            cell_area_in_acres = self.compute_cell_area(*model.cell_dimension[:2])
            cell_area_grid = uniform_grid(
                grid_shape=model.grid_shape, value=cell_area_in_acres
            )
            giip_grid = compute_hydrocarbon_in_place(
                area=cell_area_grid,
                thickness=thickness,
                porosity=porosity,
                phase_saturation=gas_saturation,
                formation_volume_factor=gas_fvf,
                net_to_gross_ratio=net_to_gross,
                hydrocarbon_type="gas",
                acre_ft_to_bbl=c.ACRE_FOOT_TO_BARRELS,
                acre_ft_to_ft3=c.ACRE_FOOT_TO_CUBIC_FEET,
            )
            if mask is not None:
                giip_grid = np.where(mask, giip_grid, 0.0)

            # Calculate STOIIP for the filtered region
            oil_saturation = model.fluid_properties.oil_saturation_grid
            oil_fvf = model.fluid_properties.oil_formation_volume_factor_grid
            oiip_grid = compute_hydrocarbon_in_place(
                area=cell_area_grid,
                thickness=thickness,
                porosity=porosity,
                phase_saturation=oil_saturation,
                formation_volume_factor=oil_fvf,
                net_to_gross_ratio=net_to_gross,
                hydrocarbon_type="oil",
                acre_ft_to_bbl=c.ACRE_FOOT_TO_BARRELS,
                acre_ft_to_ft3=c.ACRE_FOOT_TO_CUBIC_FEET,
            )
            if mask is not None:
                oiip_grid = np.where(mask, oiip_grid, 0.0)

            initial_solution_gas_grid = (
                oiip_grid
                * initial_state.model.fluid_properties.solution_gas_to_oil_ratio_grid
            )
            total_initial_gas_grid = giip_grid + initial_solution_gas_grid
            total_initial_gas = np.nansum(total_initial_gas_grid)

        if total_initial_gas == 0:
            for t in range(from_step, to_step + 1, interval):
                yield (t, 0.0)
        else:
            # Pre-compute per-step solution gas contributions in O(N),
            # then accumulate incrementally to avoid O(N²) re-summation.
            step_solution_gas: dict[int, float] = {}
            days_per_second = c.DAYS_PER_SECOND
            ft3_to_bbl = c.CUBIC_FEET_TO_BARRELS
            for s in self._sorted_steps:
                if s > to_step:
                    break
                st = self._states[s]
                oil_production = st.production_rates.oil
                step_in_days = st.step_size * days_per_second
                oil_fvf_grid = (
                    st.model.fluid_properties.oil_formation_volume_factor_grid
                )
                solution_gor_grid = (
                    st.model.fluid_properties.solution_gas_to_oil_ratio_grid
                )
                oil_production_stb = oil_production * ft3_to_bbl / oil_fvf_grid
                if cells_obj is not None:
                    mask = cells_obj.get_mask(st.model.grid_shape, st.wells)
                    oil_production_stb = np.where(
                        mask,  # type: ignore[arg-type]
                        oil_production_stb,
                        0.0,
                    )
                    solution_gor_grid = np.where(mask, solution_gor_grid, 0.0)  # type: ignore[arg-type]

                step_solution_gas[s] = float(
                    np.nansum(solution_gor_grid * oil_production_stb) * step_in_days
                )

            cumulative_solution_gas = 0.0
            sorted_idx = 0
            for t in range(from_step, to_step + 1, interval):
                # Accumulate solution gas contributions up to step t
                while (
                    sorted_idx < len(self._sorted_steps)
                    and self._sorted_steps[sorted_idx] <= t
                ):
                    s = self._sorted_steps[sorted_idx]
                    cumulative_solution_gas += step_solution_gas.get(s, 0.0)
                    sorted_idx += 1

                cumulative_free_gas = self.gas_produced(
                    self._min_step, t, cells=cells_obj
                )
                total_gas_produced = cumulative_free_gas + cumulative_solution_gas
                rf = float(total_gas_produced / total_initial_gas)
                yield (t, rf)

    def reservoir_volumetrics_analysis(self, step: int = -1) -> ReservoirVolumetrics:
        """
        Comprehensive reservoir volumetrics analysis at a specific time step.

        :param step: The time step index to analyze volumetrics for.
        :return: `ReservoirVolumetrics` containing detailed volume analysis.
        """
        state = self.get_state(step)
        if state is None:
            return ReservoirVolumetrics(
                oil_in_place=0.0,
                gas_in_place=0.0,
                water_in_place=0.0,
                pore_volume=0.0,
                hydrocarbon_pore_volume=0.0,
            )

        model = state.model
        cell_area_in_acres = self.compute_cell_area(*model.cell_dimension[:2])
        cell_area_grid = uniform_grid(
            grid_shape=model.grid_shape, value=cell_area_in_acres
        )
        pore_volume_grid = (
            cell_area_grid
            * model.thickness_grid
            * model.rock_properties.porosity_grid
            * model.rock_properties.net_to_gross_ratio_grid
            * c.ACRE_FOOT_TO_CUBIC_FEET  # Convert acre-ft to ft³
        )
        total_pore_volume = np.nansum(pore_volume_grid)

        hydrocarbon_saturation_grid = (
            model.fluid_properties.oil_saturation_grid
            + model.fluid_properties.gas_saturation_grid
        )
        hydrocarbon_pore_volume = np.nansum(
            pore_volume_grid * hydrocarbon_saturation_grid
        )
        return ReservoirVolumetrics(
            oil_in_place=self.oil_in_place(step),
            gas_in_place=self.free_gas_in_place(step),
            water_in_place=self.water_in_place(step),
            pore_volume=total_pore_volume,
            hydrocarbon_pore_volume=hydrocarbon_pore_volume,
        )

    def instantaneous_production_rates(
        self, step: int = -1, cells: typing.Union[Cells, CellFilter] = None
    ) -> InstantaneousRates:
        """
        Calculates instantaneous production rates at a specific time step.

        :param step: The time step index to calculate rates for.
        :param cells: Optional filter for specific cells, well name, or region.
            - None: Entire reservoir (default)
            - str: Well name (e.g., "PROD-1")
            - (i, j, k): Single cell
            - [(i1,j1,k1), ...]: List of cells
            - (slice, slice, slice): Region
        :return: `InstantaneousRates` containing detailed rate analysis.
        """
        # Resolve step and convert cells before building cache key
        step = self._resolve_step(step)
        cells_obj = _ensure_cells(cells)
        cache_key = (step, cells_obj)
        if cache_key in self._instantaneous_production_rates_cache:
            return self._instantaneous_production_rates_cache[cache_key]

        state = self.get_state(step)
        if state is None:
            logger.debug(
                f"State at time step {step} is not available. Returning zero production rates."
            )
            return InstantaneousRates(
                oil_rate=0.0,
                gas_rate=0.0,
                water_rate=0.0,
                total_liquid_rate=0.0,
                gas_oil_ratio=0.0,
                water_cut=0.0,
            )

        # Get cell mask for filtering
        mask = (
            cells_obj.get_mask(state.model.grid_shape, state.wells)
            if cells_obj
            else None
        )

        oil_rate = 0.0
        free_gas_rate = 0.0
        solution_gas_rate = 0.0
        water_rate = 0.0
        oil_production_stb = None

        # Sum production rates from all grid cells
        if (oil_production := state.production_rates.oil) is not None:
            # Convert from ft³/day to STB/day using oil FVF
            oil_fvf_grid = state.model.fluid_properties.oil_formation_volume_factor_grid
            oil_production_stb = oil_production * c.CUBIC_FEET_TO_BARRELS / oil_fvf_grid
            if mask is not None:
                oil_production_stb = np.where(mask, oil_production_stb, 0.0)
            oil_rate = np.nansum(oil_production_stb)

        if (gas_production := state.production_rates.gas) is not None:
            # Convert from ft³/day to SCF/day using gas FVF (free gas phase only)
            gas_fvf_grid = state.model.fluid_properties.gas_formation_volume_factor_grid
            gas_production_scf = gas_production / gas_fvf_grid
            if mask is not None:
                gas_production_scf = np.where(mask, gas_production_scf, 0.0)
            free_gas_rate = float(np.nansum(gas_production_scf))

        # Solution gas: gas dissolved in produced oil that flashes out at surface conditions.
        # Amount = Rs (SCF/STB) * oil production (STB/day)
        if oil_production_stb is not None:
            solution_gor_grid = (
                state.model.fluid_properties.solution_gas_to_oil_ratio_grid
            )
            solution_gas_rate = float(np.nansum(solution_gor_grid * oil_production_stb))

        # Total gas = free gas phase + solution gas from oil
        gas_rate = free_gas_rate + solution_gas_rate

        if (water_production := state.production_rates.water) is not None:
            # Convert from ft³/day to STB/day using water FVF
            water_fvf_grid = (
                state.model.fluid_properties.water_formation_volume_factor_grid
            )
            water_production_stb = (
                water_production * c.CUBIC_FEET_TO_BARRELS / water_fvf_grid
            )
            if mask is not None:
                water_production_stb = np.where(mask, water_production_stb, 0.0)
            water_rate = np.nansum(water_production_stb)

        total_liquid_rate = oil_rate + water_rate
        gas_oil_ratio = gas_rate / oil_rate if oil_rate > 0 else 0.0
        water_cut = water_rate / total_liquid_rate if total_liquid_rate > 0 else 0.0

        logger.debug(
            f"Instantaneous production rates at time step {step}: "
            f"oil={oil_rate:.2f} STB/day, gas={gas_rate:.2f} SCF/day "
            f"(free={free_gas_rate:.2f}, solution={solution_gas_rate:.2f}), "
            f"water={water_rate:.2f} STB/day, "
            f"GOR={gas_oil_ratio:.2f}, WaterCut={water_cut:.4f}"
        )
        result = InstantaneousRates(
            oil_rate=oil_rate,
            gas_rate=gas_rate,
            water_rate=water_rate,
            total_liquid_rate=total_liquid_rate,
            gas_oil_ratio=gas_oil_ratio,
            water_cut=water_cut,
            free_gas_rate=free_gas_rate,
            solution_gas_rate=solution_gas_rate,
        )
        self._instantaneous_production_rates_cache[cache_key] = result
        return result

    def instantaneous_injection_rates(
        self, step: int = -1, cells: typing.Union[Cells, CellFilter] = None
    ) -> InstantaneousRates:
        """
        Calculates instantaneous injection rates at a specific time step.

        :param step: The time step index to calculate rates for.
        :param cells: Optional filter for specific cells, well name, or region.
            - None: Entire reservoir (default)
            - str: Well name (e.g., "INJ-1")
            - (i, j, k): Single cell
            - [(i1,j1,k1), ...]: List of cells
            - (slice, slice, slice): Region
        :return: `InstantaneousRates` containing detailed injection rate analysis.
        """
        step = self._resolve_step(step)
        cells_obj = _ensure_cells(cells)
        cache_key = (step, cells_obj)
        if cache_key in self._instantaneous_injection_rates_cache:
            return self._instantaneous_injection_rates_cache[cache_key]

        state = self.get_state(step)
        if state is None:
            logger.debug(
                f"State at time step {step} is not available. Returning zero injection rates."
            )
            return InstantaneousRates(
                oil_rate=0.0,
                gas_rate=0.0,
                water_rate=0.0,
                total_liquid_rate=0.0,
                gas_oil_ratio=0.0,
                water_cut=0.0,
            )

        # Get cell mask for filtering
        mask = (
            cells_obj.get_mask(state.model.grid_shape, state.wells)
            if cells_obj
            else None
        )

        oil_rate = 0.0
        gas_rate = 0.0
        water_rate = 0.0

        # Sum injection rates from all grid cells
        if (oil_injection := state.injection_rates.oil) is not None:
            # Convert from ft³/day to STB/day using oil FVF
            oil_fvf_grid = state.model.fluid_properties.oil_formation_volume_factor_grid
            oil_injection_stb = oil_injection * c.CUBIC_FEET_TO_BARRELS / oil_fvf_grid
            if mask is not None:
                oil_injection_stb = np.where(mask, oil_injection_stb, 0.0)
            oil_rate = np.nansum(oil_injection_stb)

        injected_gas_fvf_grid, injected_water_fvf_grid = _build_injected_fvf_grids(
            wells=state.wells,
            pressure_grid=state.model.fluid_properties.pressure_grid,
            temperature_grid=state.model.fluid_properties.temperature_grid,
            grid_shape=state.model.grid_shape,
        )
        if (gas_injection := state.injection_rates.gas) is not None:
            gas_injection_scf = gas_injection / injected_gas_fvf_grid
            if mask is not None:
                gas_injection_scf = np.where(mask, gas_injection_scf, 0.0)
            gas_rate = np.nansum(gas_injection_scf)

        if (water_injection := state.injection_rates.water) is not None:
            water_injection_stb = (
                water_injection * c.CUBIC_FEET_TO_BARRELS / injected_water_fvf_grid
            )
            if mask is not None:
                water_injection_stb = np.where(mask, water_injection_stb, 0.0)
            water_rate = np.nansum(water_injection_stb)

        total_liquid_rate = oil_rate + water_rate
        gas_oil_ratio = gas_rate / oil_rate if oil_rate > 0 else 0.0
        water_cut = water_rate / total_liquid_rate if total_liquid_rate > 0 else 0.0
        result = InstantaneousRates(
            oil_rate=oil_rate,
            gas_rate=gas_rate,
            water_rate=water_rate,
            total_liquid_rate=total_liquid_rate,
            gas_oil_ratio=gas_oil_ratio,
            water_cut=water_cut,
            free_gas_rate=gas_rate,
            solution_gas_rate=0.0,
        )
        self._instantaneous_injection_rates_cache[cache_key] = result
        return result

    def cumulative_production_analysis(self, step: int = -1) -> CumulativeProduction:
        """
        Comprehensive cumulative production analysis at a specific time step.

        :param step: The time step index to analyze cumulative production for.
        :return: `CumulativeProduction` containing detailed cumulative analysis.
        """
        cumulative_oil = self.oil_produced(self._min_step, step)
        cumulative_free_gas = self.gas_produced(self._min_step, step)
        cumulative_water = self.water_produced(self._min_step, step)
        return CumulativeProduction(
            cumulative_oil=cumulative_oil,
            cumulative_free_gas=cumulative_free_gas,
            cumulative_water=cumulative_water,
            oil_recovery_factor=self.oil_recovery_factor,
            gas_recovery_factor=self.gas_recovery_factor,
        )

    def material_balance(self, step: int = -1) -> MaterialBalance:
        """
        Material balance analysis for reservoir drive mechanism identification
        using the Havlena-Odeh formulation.

        Computes drive mechanisms, drive indices, and diagnostic pressure/GOR
        metrics at a specific time step using:

            F = N*(Eo + m*Eg + Efw) + We

        Where:
            F    = Underground withdrawal (reservoir bbl)
            Eo   = Oil + dissolved gas expansion term (bbl/STB)
            Eg   = Gas cap expansion term (bbl/STB)
            Efw  = Rock + connate water expansion term (bbl/STB)
            We   = Water influx (bbl)
            m    = Gas cap ratio = (GIIP*Bgi) / (N*Boi)

        All PVT values are weighted by oil saturation so that gas cap cells
        do not skew Boi, Rsi, or Bgi.

        :param step: The time step index to analyze. Use -1 for the latest step.
        :return: `MaterialBalance` containing drive volumes, drive indices,
            pressure, producing GOR, and aquifer influx.
        """
        step = self._resolve_step(step)
        state = self.get_state(step)
        initial_state = self.get_state(self._min_step)

        if state is None or initial_state is None:
            logger.debug(
                f"State at time step {step} or initial state (step {self._min_step}) "
                "is not available. Returning zero material balance."
            )
            return MaterialBalance(
                pressure=0.0,
                pressure_decline=0.0,
                oil_expansion_factor=0.0,
                producing_gor=0.0,
                gas_cap_ratio=0.0,
                underground_withdrawal=0.0,
                solution_gas_drive=0.0,
                gas_cap_drive=0.0,
                water_drive=0.0,
                compaction_drive=0.0,
                solution_gas_drive_index=0.0,
                gas_cap_drive_index=0.0,
                water_drive_index=0.0,
                compaction_drive_index=0.0,
                aquifer_influx=0.0,
            )

        initial_model = initial_state.model
        current_model = state.model

        # Oil-saturation weighted means for Bo, Rs, Bw, pressure
        # (prevents gas cap cells from skewing oil-zone PVT)
        initial_oil_saturation_grid = initial_model.fluid_properties.oil_saturation_grid
        initial_oil_weights = np.where(
            initial_oil_saturation_grid > 0, initial_oil_saturation_grid, 0.0
        )
        initial_oil_weight_sum = float(np.nansum(initial_oil_weights))

        current_oil_saturation_grid = current_model.fluid_properties.oil_saturation_grid
        current_oil_weights = np.where(
            current_oil_saturation_grid > 0, current_oil_saturation_grid, 0.0
        )
        current_oil_weight_sum = float(np.nansum(current_oil_weights))

        # Gas-saturation weighted means for gas cap Bg/Bgi
        # (gas cap expansion Eg needs FVF representative of the gas cap, not the oil zone)
        initial_gas_saturation_grid = initial_model.fluid_properties.gas_saturation_grid
        initial_gas_weights = np.where(
            initial_gas_saturation_grid > 0, initial_gas_saturation_grid, 0.0
        )
        initial_gas_weight_sum = float(np.nansum(initial_gas_weights))

        current_gas_saturation_grid = current_model.fluid_properties.gas_saturation_grid
        current_gas_weights = np.where(
            current_gas_saturation_grid > 0, current_gas_saturation_grid, 0.0
        )
        current_gas_weight_sum = float(np.nansum(current_gas_weights))

        def _initial_oil_weighted_mean(grid: np.ndarray) -> float:
            if initial_oil_weight_sum == 0:
                return float(np.nanmean(grid))
            return float(np.nansum(grid * initial_oil_weights) / initial_oil_weight_sum)

        def _current_oil_weighted_mean(grid: np.ndarray) -> float:
            if current_oil_weight_sum == 0:
                return float(np.nanmean(grid))
            return float(np.nansum(grid * current_oil_weights) / current_oil_weight_sum)

        def _initial_gas_weighted_mean(grid: np.ndarray) -> float:
            if initial_gas_weight_sum == 0:
                return float(np.nanmean(grid))
            return float(np.nansum(grid * initial_gas_weights) / initial_gas_weight_sum)

        def _current_gas_weighted_mean(grid: np.ndarray) -> float:
            if current_gas_weight_sum == 0:
                return float(np.nanmean(grid))
            return float(np.nansum(grid * current_gas_weights) / current_gas_weight_sum)

        initial_oil_fvf = _initial_oil_weighted_mean(
            initial_model.fluid_properties.oil_formation_volume_factor_grid
        )
        initial_solution_gor = _initial_oil_weighted_mean(
            initial_model.fluid_properties.solution_gas_to_oil_ratio_grid
        )
        # Gas-weighted Bgi for gas cap expansion Eg (representative of gas cap conditions)
        # Oil-zone Bgi is not needed: Eo and F only use current Bg.
        initial_gas_cap_fvf = _initial_gas_weighted_mean(
            initial_model.fluid_properties.gas_formation_volume_factor_grid
        )

        initial_water_sat = _initial_oil_weighted_mean(
            initial_model.fluid_properties.water_saturation_grid
        )
        initial_pressure = _initial_oil_weighted_mean(
            initial_model.fluid_properties.pressure_grid
        )

        current_oil_fvf = _current_oil_weighted_mean(
            current_model.fluid_properties.oil_formation_volume_factor_grid
        )
        current_solution_gor = _current_oil_weighted_mean(
            current_model.fluid_properties.solution_gas_to_oil_ratio_grid
        )
        # Oil-weighted Bg for Eo and F
        current_gas_fvf = _current_oil_weighted_mean(
            current_model.fluid_properties.gas_formation_volume_factor_grid
        )
        # Gas-weighted Bg for gas cap expansion Eg
        current_gas_cap_fvf = _current_gas_weighted_mean(
            current_model.fluid_properties.gas_formation_volume_factor_grid
        )
        current_water_fvf = _current_oil_weighted_mean(
            current_model.fluid_properties.water_formation_volume_factor_grid
        )
        current_pressure = _current_oil_weighted_mean(
            current_model.fluid_properties.pressure_grid
        )
        pressure_decline = initial_pressure - current_pressure

        rock_compressibility = float(initial_model.rock_properties.compressibility)
        water_compressibility = float(
            np.nanmean(initial_model.fluid_properties.water_compressibility_grid)
        )

        initial_oil_in_place = self.oil_in_place(self._min_step)
        initial_free_gas_in_place = self.free_gas_in_place(self._min_step)

        gas_cap_ratio = (
            (initial_free_gas_in_place * initial_gas_cap_fvf)
            / (initial_oil_in_place * initial_oil_fvf)
            if (initial_oil_in_place * initial_oil_fvf) > 0
            else 0.0
        )

        cumulative_oil_produced = self.oil_produced(self._min_step, step)
        cumulative_water_produced = self.water_produced(self._min_step, step)
        cumulative_free_gas_produced = self.gas_produced(self._min_step, step)
        cumulative_water_injected = self.water_injected(self._min_step, step)
        cumulative_gas_injected = self.gas_injected(self._min_step, step)

        cumulative_solution_gas_produced = 0.0
        days_per_second = c.DAYS_PER_SECOND
        ft3_to_bbl = c.CUBIC_FEET_TO_BARRELS
        for s in self._sorted_steps:
            if s > step:
                break

            st = self._states[s]
            oil_production = st.production_rates.oil
            step_in_days = st.step_size * days_per_second
            oil_fvf_grid = st.model.fluid_properties.oil_formation_volume_factor_grid
            solution_gor_grid = st.model.fluid_properties.solution_gas_to_oil_ratio_grid
            oil_production_stb = oil_production * ft3_to_bbl / oil_fvf_grid
            cumulative_solution_gas_produced += float(
                np.nansum(solution_gor_grid * oil_production_stb) * step_in_days
            )

        cumulative_total_gas_produced = (
            cumulative_free_gas_produced + cumulative_solution_gas_produced
        )
        producing_gor = (
            cumulative_total_gas_produced / cumulative_oil_produced
            if cumulative_oil_produced > 0
            else initial_solution_gor
        )

        injected_gas_fvf_grid, injected_water_fvf_grid = _build_injected_fvf_grids(
            wells=state.wells,
            pressure_grid=current_model.fluid_properties.pressure_grid,
            temperature_grid=current_model.fluid_properties.temperature_grid,
            grid_shape=current_model.grid_shape,
        )
        injected_water_fvf = (
            float(np.nanmean(injected_water_fvf_grid))
            if not np.all(np.isnan(injected_water_fvf_grid))
            else current_water_fvf
        )
        injected_gas_fvf = (
            float(np.nanmean(injected_gas_fvf_grid))
            if not np.all(np.isnan(injected_gas_fvf_grid))
            else current_gas_fvf
        )

        # Havlena-Odeh terms
        # Underground withdrawal, `F` in bbl
        underground_withdrawal = (
            cumulative_oil_produced
            * (
                current_oil_fvf
                + (producing_gor - current_solution_gor) * current_gas_fvf
            )
            + cumulative_water_produced * current_water_fvf
            - cumulative_water_injected * injected_water_fvf
            - cumulative_gas_injected * injected_gas_fvf
        )

        # Expansion terms (bbl/STB)
        oil_expansion = (current_oil_fvf - initial_oil_fvf) + (
            initial_solution_gor - current_solution_gor
        ) * current_gas_fvf
        gas_cap_expansion = (
            initial_oil_fvf * ((current_gas_cap_fvf / initial_gas_cap_fvf) - 1)
            if initial_gas_cap_fvf > 0
            else 0.0
        )
        rock_and_water_expansion = (
            (1 + gas_cap_ratio)
            * initial_oil_fvf
            * (
                (initial_water_sat * water_compressibility + rock_compressibility)
                / (1 - initial_water_sat)
            )
            * pressure_decline
            if (1 - initial_water_sat) > 0
            else 0.0
        )

        # Drive volumes in bbl
        solution_gas_drive = initial_oil_in_place * oil_expansion
        gas_cap_drive = initial_oil_in_place * gas_cap_ratio * gas_cap_expansion
        compaction_drive = initial_oil_in_place * rock_and_water_expansion
        drive_from_expansion = solution_gas_drive + gas_cap_drive + compaction_drive

        # Water influx `We`: residual between withdrawal and expansion in bbl
        water_drive = max(0.0, underground_withdrawal - drive_from_expansion)

        total_drive = drive_from_expansion + water_drive
        if total_drive > 0:
            solution_gas_drive_index = solution_gas_drive / total_drive
            gas_cap_drive_index = gas_cap_drive / total_drive
            compaction_drive_index = compaction_drive / total_drive
            water_drive_index = water_drive / total_drive
        else:
            solution_gas_drive_index = gas_cap_drive_index = compaction_drive_index = (
                water_drive_index
            ) = 0.0

        logger.debug(
            f"Material balance at step {step}: "
            f"P={current_pressure:.2f} psia, ΔP={pressure_decline:.2f} psi, "
            f"Rp={producing_gor:.1f} scf/STB, m={gas_cap_ratio:.4f}, "
            f"F={underground_withdrawal:.1f} bbl, We={water_drive:.1f} bbl | "
            f"SGD={solution_gas_drive_index:.3f}, GCD={gas_cap_drive_index:.3f}, "
            f"WD={water_drive_index:.3f}, CD={compaction_drive_index:.3f}"
        )
        return MaterialBalance(
            pressure=float(current_pressure),
            pressure_decline=float(pressure_decline),
            oil_expansion_factor=float(current_oil_fvf / initial_oil_fvf),
            producing_gor=float(producing_gor),
            gas_cap_ratio=float(gas_cap_ratio),
            underground_withdrawal=float(underground_withdrawal),
            solution_gas_drive=float(solution_gas_drive),
            gas_cap_drive=float(gas_cap_drive),
            water_drive=float(water_drive),
            compaction_drive=float(compaction_drive),
            solution_gas_drive_index=float(solution_gas_drive_index),
            gas_cap_drive_index=float(gas_cap_drive_index),
            water_drive_index=float(water_drive_index),
            compaction_drive_index=float(compaction_drive_index),
            aquifer_influx=float(water_drive),
        )

    mbal = material_balance

    def material_balance_error(
        self,
        from_step: typing.Optional[int] = None,
        to_step: int = -1,
    ) -> MaterialBalanceError:
        """
        Compute material balance error over a simulation interval using two
        complementary approaches.

        Phase-level MBE (oil, water, gas) - simulator volume conservation check:

            MBE_phase = (ΔPV_phase - net_flux_phase) / PV_phase_initial

        Where all volumes are at reservoir conditions (ft³). Directly tests
        whether the saturation solver conserved pore-volume occupancy for each
        phase. A value of 0.0 means the simulator perfectly conserved that phase.

        Total MBE - Havlena-Odeh physical balance check, sourced directly from
        `material_balance()` to avoid recomputing PVT terms:

            total_mbe = (F - N*(Eo + m*Eg + Efw) - We) / F

        Where F is the total underground withdrawal (reservoir bbl). A value of
        0.0 means expansion exactly accounts for withdrawal.

        The two checks diagnose different problems:
            - Large phase MBE → simulator numerical conservation error
            - Large total MBE → PVT / STOIIP / drive mechanism mismatch

        MBE Quality Thresholds:
            |MBE| < 0.1%      Excellent   - results are reliable
            |MBE| < 1%        Acceptable  - monitor for drift
            |MBE| < 5%        Marginal    - refine grid or reduce timestep
            |MBE| >= 5%       Unacceptable - investigate before using results

        :param from_step: Starting time step of the interval (inclusive).
            Defaults to `self._min_step` when None.
        :param to_step: Ending time step of the interval (inclusive).
            Use -1 for the latest available state.
        :return: `MaterialBalanceError` with per-phase and total MBE, drive
            terms, drive indices, and quality rating.
        """
        if from_step is None:
            from_step = self._min_step

        to_step = self._resolve_step(to_step)
        initial_state = self.get_state(from_step)
        current_state = self.get_state(to_step)

        if initial_state is None or current_state is None:
            return MaterialBalanceError(
                oil_mbe=0.0,
                water_mbe=0.0,
                gas_mbe=0.0,
                total_mbe=0.0,
                solution_gas_drive=0.0,
                gas_cap_drive=0.0,
                water_drive=0.0,
                compaction_drive=0.0,
                solution_gas_drive_index=0.0,
                gas_cap_drive_index=0.0,
                water_drive_index=0.0,
                compaction_drive_index=0.0,
                underground_withdrawal=0.0,
                quality="unacceptable",
                from_step=from_step,
                to_step=to_step,
            )

        initial_model = initial_state.model
        current_model = current_state.model

        mbal = self.material_balance(to_step)

        # total_mbe = (F - expansion - We) / F
        # By construction We = max(0, F - expansion), so if `We` was clamped to 0
        # (expansion > F, which shouldn't happen physically but can numerically)
        # `total_mbe` will be non-zero. We = water_drive.
        drive_from_expansion = (
            mbal.solution_gas_drive + mbal.gas_cap_drive + mbal.compaction_drive
        )
        total_mbe = (
            (mbal.underground_withdrawal - drive_from_expansion - mbal.water_drive)
            / mbal.underground_withdrawal
            if abs(mbal.underground_withdrawal) > 0
            else 0.0
        )

        # Phase-level MBE in reservoir ft³
        # The saturation solver operates in reservoir space so this is the
        # correct basis for checking simulator volume conservation.
        # Pore volume changes with pressure via rock compressibility:
        #   PV(P) = PV_ref * (1 + cf * (P - P_ref))
        cell_area_ft2 = (
            initial_model.cell_dimension[0] * initial_model.cell_dimension[1]
        )
        reference_pore_volume_grid = (
            cell_area_ft2
            * initial_model.thickness_grid
            * initial_model.rock_properties.porosity_grid
            * initial_model.rock_properties.net_to_gross_ratio_grid
        )
        rock_compressibility = float(initial_model.rock_properties.compressibility)
        initial_pressure_grid = initial_model.fluid_properties.pressure_grid
        current_pressure_grid = current_model.fluid_properties.pressure_grid

        initial_pore_volume_grid = reference_pore_volume_grid
        current_pore_volume_grid = reference_pore_volume_grid * (
            1.0 + rock_compressibility * (current_pressure_grid - initial_pressure_grid)
        )

        initial_oil_pv_ft3 = float(
            np.nansum(
                initial_pore_volume_grid
                * initial_model.fluid_properties.oil_saturation_grid
            )
        )
        initial_water_pv_ft3 = float(
            np.nansum(
                initial_pore_volume_grid
                * initial_model.fluid_properties.water_saturation_grid
            )
        )
        initial_gas_pv_ft3 = float(
            np.nansum(
                initial_pore_volume_grid
                * initial_model.fluid_properties.gas_saturation_grid
            )
        )

        current_oil_pv_ft3 = float(
            np.nansum(
                current_pore_volume_grid
                * current_model.fluid_properties.oil_saturation_grid
            )
        )
        current_water_pv_ft3 = float(
            np.nansum(
                current_pore_volume_grid
                * current_model.fluid_properties.water_saturation_grid
            )
        )
        current_gas_pv_ft3 = float(
            np.nansum(
                current_pore_volume_grid
                * current_model.fluid_properties.gas_saturation_grid
            )
        )

        oil_pv_change_ft3 = current_oil_pv_ft3 - initial_oil_pv_ft3
        water_pv_change_ft3 = current_water_pv_ft3 - initial_water_pv_ft3
        gas_pv_change_ft3 = current_gas_pv_ft3 - initial_gas_pv_ft3

        # Net flux in reservoir ft³
        # Production/injection grids are all in ft³/day
        oil_produced_ft3 = 0.0
        oil_injected_ft3 = 0.0
        water_produced_ft3 = 0.0
        water_injected_ft3 = 0.0
        gas_produced_ft3 = 0.0
        gas_injected_ft3 = 0.0
        days_per_second = c.DAYS_PER_SECOND
        for s in self._sorted_steps:
            if s < from_step or s > to_step:
                continue

            st = self._states[s]
            step_in_days = st.step_size * days_per_second
            production = st.production_rates
            oil_produced_ft3 += (
                float(np.nansum(production.oil)) * step_in_days
                if production.oil is not None
                else 0.0
            )
            water_produced_ft3 += (
                float(np.nansum(production.water)) * step_in_days
                if production.water is not None
                else 0.0
            )
            gas_produced_ft3 += (
                float(np.nansum(production.gas)) * step_in_days
                if production.gas is not None
                else 0.0
            )

            injection = st.injection_rates
            oil_injected_ft3 += (
                float(np.nansum(injection.oil)) * step_in_days
                if injection.oil is not None
                else 0.0
            )
            water_injected_ft3 += (
                float(np.nansum(injection.water)) * step_in_days
                if injection.water is not None
                else 0.0
            )
            gas_injected_ft3 += (
                float(np.nansum(injection.gas)) * step_in_days
                if injection.gas is not None
                else 0.0
            )

        # Positive net flux indicates net flow into reservoir
        oil_net_flux_ft3 = oil_injected_ft3 - oil_produced_ft3
        water_net_flux_ft3 = water_injected_ft3 - water_produced_ft3
        gas_net_flux_ft3 = gas_injected_ft3 - gas_produced_ft3

        oil_mbe = (
            (oil_pv_change_ft3 - oil_net_flux_ft3) / initial_oil_pv_ft3
            if initial_oil_pv_ft3 > 0
            else 0.0
        )
        water_mbe = (
            (water_pv_change_ft3 - water_net_flux_ft3) / initial_water_pv_ft3
            if initial_water_pv_ft3 > 0
            else 0.0
        )
        gas_mbe = (
            (gas_pv_change_ft3 - gas_net_flux_ft3) / initial_gas_pv_ft3
            if initial_gas_pv_ft3 > 0
            else 0.0
        )

        # Quality rate using the worst of all four MBEs
        worst_mbe = max(abs(total_mbe), abs(oil_mbe), abs(water_mbe), abs(gas_mbe))
        if worst_mbe < 0.001:
            quality = "excellent"
        elif worst_mbe < 0.01:
            quality = "acceptable"
        elif worst_mbe < 0.05:
            quality = "marginal"
        else:
            quality = "unacceptable"

        logger.debug(
            f"MBE [{from_step} → {to_step}]: "
            f"oil={oil_mbe:.4%}, water={water_mbe:.4%}, gas={gas_mbe:.4%}, "
            f"total={total_mbe:.4%}, quality={quality} | "
            f"F={mbal.underground_withdrawal:.1f} bbl, "
            f"SGD={mbal.solution_gas_drive_index:.3f}, GCD={mbal.gas_cap_drive_index:.3f}, "
            f"WD={mbal.water_drive_index:.3f}, CD={mbal.compaction_drive_index:.3f}"
        )
        return MaterialBalanceError(
            oil_mbe=float(oil_mbe),
            water_mbe=float(water_mbe),
            gas_mbe=float(gas_mbe),
            total_mbe=float(total_mbe),
            solution_gas_drive=float(mbal.solution_gas_drive),
            gas_cap_drive=float(mbal.gas_cap_drive),
            water_drive=float(mbal.water_drive),
            compaction_drive=float(mbal.compaction_drive),
            solution_gas_drive_index=float(mbal.solution_gas_drive_index),
            gas_cap_drive_index=float(mbal.gas_cap_drive_index),
            water_drive_index=float(mbal.water_drive_index),
            compaction_drive_index=float(mbal.compaction_drive_index),
            underground_withdrawal=float(mbal.underground_withdrawal),
            quality=quality,
            from_step=from_step,
            to_step=to_step,
        )

    mbe = material_balance_error

    def sweep_efficiency_analysis(
        self,
        step: int = -1,
        displacing_phase: typing.Literal["oil", "water", "gas"] = "water",
        delta_water_saturation_threshold: float = 0.02,
        delta_gas_saturation_threshold: float = 0.01,
        solvent_concentration_threshold: float = 0.01,
    ) -> SweepEfficiencyAnalysis:
        """
        Sweep efficiency analysis to evaluate reservoir contact and displacement.

        Contact detection is performed using changes in the displacing phase saturation
        (or solvent concentration for miscible floods). Displacement efficiency is
        computed as the saturation-weighted oil removal in contacted cells using the
        initial oil formation volume factor to convert pore volume to STB.

        Vertical sweep efficiency is computed using the saturation-weighted column fraction:
            For each (i,j): F_ij = sum_z (phi * h * (So_init - So_cur)) / sum_z (phi * h * So_init)
            Then VSE = sum_ij (F_ij * initial_oil_stb_ij) / sum_ij (initial_oil_stb_ij)

        Areal sweep efficiency is computed from planform contacted columns:
            E_A = contacted_planform_area / total_planform_area
            where planform area = dx * dy per column from model.cell_dimension.

        :param step: Time step index to analyze; -1 gives the latest state.
        :param displacing_phase: Phase doing the displacing ('oil', 'water' or 'gas').
        :param delta_water_saturation_threshold: Threshold on ΔSw to declare a cell contacted when displacing phase is water.
        :param delta_gas_saturation_threshold: Threshold on ΔSg to declare a cell contacted when displacing phase is gas.
        :param solvent_concentration_threshold: Threshold on solvent concentration to mark contact in miscible runs.
        :return: ``SweepEfficiencyAnalysis`` with volumetric, displacement, recovery efficiencies,
                contacted/uncontacted oil (STB), areal sweep, and vertical sweep metrics.
        """
        state = self.get_state(step)
        initial_state = self.get_state(self._min_step)
        if state is None or initial_state is None:
            logger.debug(
                f"State at time step {step} or initial state (step {self._min_step}) is not available. Returning zeros.",
            )
            return SweepEfficiencyAnalysis(
                volumetric_sweep_efficiency=0.0,
                displacement_efficiency=0.0,
                recovery_efficiency=0.0,
                contacted_oil=0.0,
                uncontacted_oil=0.0,
                areal_sweep_efficiency=0.0,
                vertical_sweep_efficiency=0.0,
            )

        initial_model = initial_state.model
        current_model = state.model
        grid_shape = current_model.grid_shape

        initial_oil_saturation = initial_model.fluid_properties.oil_saturation_grid
        current_oil_saturation = current_model.fluid_properties.oil_saturation_grid
        initial_water_saturation = initial_model.fluid_properties.water_saturation_grid
        current_water_saturation = current_model.fluid_properties.water_saturation_grid
        initial_gas_saturation = initial_model.fluid_properties.gas_saturation_grid
        current_gas_saturation = current_model.fluid_properties.gas_saturation_grid
        solvent_concentration_grid = (
            current_model.fluid_properties.solvent_concentration_grid
        )

        # Determine contacted mask based on displacing phase and thresholds
        if displacing_phase == "water":
            saturation_delta = current_water_saturation - initial_water_saturation
            contacted_mask = saturation_delta > delta_water_saturation_threshold
        elif displacing_phase == "gas":
            saturation_delta = current_gas_saturation - initial_gas_saturation
            contacted_mask = saturation_delta > delta_gas_saturation_threshold
            if solvent_concentration_grid is not None:
                contacted_mask = contacted_mask | (
                    solvent_concentration_grid > solvent_concentration_threshold
                )
        else:  # displacing_phase == "oil"
            # treat as decrease in oil
            saturation_delta = initial_oil_saturation - current_oil_saturation
            contacted_mask = saturation_delta > 0.0

        cell_dimension_x, cell_dimension_y = (
            current_model.cell_dimension[0],
            current_model.cell_dimension[1],
        )
        cell_area_ft2 = cell_dimension_x * cell_dimension_y
        thickness_grid = current_model.thickness_grid
        porosity_grid = current_model.rock_properties.porosity_grid
        net_to_gross_grid = current_model.rock_properties.net_to_gross_ratio_grid

        oil_formation_volume_factor_initial_grid = (
            initial_model.fluid_properties.oil_formation_volume_factor_grid
        )
        # Convert Bo from bbl/STB to ft³/STB
        initial_oil_formation_volume_factor_grid_ft3_per_stb = (
            oil_formation_volume_factor_initial_grid * c.BARRELS_TO_CUBIC_FEET
        )
        initial_oil_formation_volume_factor_grid_ft3_per_stb = np.where(
            initial_oil_formation_volume_factor_grid_ft3_per_stb <= 0.0,
            np.nan,
            initial_oil_formation_volume_factor_grid_ft3_per_stb,
        )

        # Compute initial oil (pore) volume per cell (ft³)
        initial_oil_volume_ft3 = (
            cell_area_ft2
            * thickness_grid
            * porosity_grid
            * net_to_gross_grid
            * initial_oil_saturation
        )
        initial_oil_volume_stb = np.divide(
            initial_oil_volume_ft3,
            initial_oil_formation_volume_factor_grid_ft3_per_stb,
            out=np.zeros_like(initial_oil_volume_ft3),
            where=~np.isnan(initial_oil_formation_volume_factor_grid_ft3_per_stb),
        )

        total_initial_oil_volume_stb = float(np.nansum(initial_oil_volume_stb))
        # Current oil (convert using initial Bo for STB conversion)
        current_oil_volume_ft3 = (
            cell_area_ft2
            * thickness_grid
            * porosity_grid
            * net_to_gross_grid
            * current_oil_saturation
        )
        current_oil_volume_stb = np.divide(
            current_oil_volume_ft3,
            initial_oil_formation_volume_factor_grid_ft3_per_stb,
            out=np.zeros_like(current_oil_volume_ft3),
            where=~np.isnan(initial_oil_formation_volume_factor_grid_ft3_per_stb),
        )

        # Contacted / uncontacted oil volumes (STB) based on mask
        contacted_initial_oil_volume_stb = float(
            np.nansum(initial_oil_volume_stb[contacted_mask])
        )
        contacted_oil_volume_remaining_stb = float(
            np.nansum(current_oil_volume_stb[contacted_mask])
        )
        uncontacted_oil_volume_stb = float(
            np.nansum(initial_oil_volume_stb[~contacted_mask])
        )

        # Volumetric sweep efficiency: fraction of initial oil contacted
        volumetric_sweep_efficiency = (
            contacted_initial_oil_volume_stb / total_initial_oil_volume_stb
            if total_initial_oil_volume_stb > 0
            else 0.0
        )

        # Displacement efficiency in contacted volume (saturation-weighted/volume-weighted)
        # numerator: oil removed in contacted cells (STB)
        if contacted_initial_oil_volume_stb > 0:
            oil_removed_contacted_stb = (
                contacted_initial_oil_volume_stb - contacted_oil_volume_remaining_stb
            )
            displacement_efficiency = float(
                clip(
                    oil_removed_contacted_stb / contacted_initial_oil_volume_stb,
                    0.0,
                    1.0,
                )
            )
        else:
            displacement_efficiency = 0.0

        recovery_efficiency = volumetric_sweep_efficiency * displacement_efficiency

        # AREAL SWEEP EFFICIENCY: contacted planform area / total planform area
        # A column is contacted if any cell in that (i,j) column is contacted
        # Guard axis=2 operations for 2D grids (where len(grid_shape) < 3)
        if len(grid_shape) >= 3:
            mask_reshaped = contacted_mask.reshape(grid_shape)
            column_contacted = np.any(mask_reshaped, axis=2)
            contacted_planform_cells = int(np.count_nonzero(column_contacted))
            total_planform_cells = grid_shape[0] * grid_shape[1]
        else:
            # 2D grid: every cell is its own planform column
            contacted_planform_cells = int(np.count_nonzero(contacted_mask))
            total_planform_cells = int(np.prod(grid_shape))

        areal_sweep_efficiency = (
            (contacted_planform_cells * cell_area_ft2)
            / (total_planform_cells * cell_area_ft2)
            if total_planform_cells > 0
            else 0.0
        )

        # VERTICAL SWEEP EFFICIENCY (saturation-weighted)
        # For each (i,j) compute:
        #   denom_ij = sum_z (phi*h*So_init)_ij
        #   numer_ij = sum_z (phi*h*(So_init - So_cur))_ij  (clamped >= 0)
        # Then column_fraction_ij = numer_ij / denom_ij  (if denom_ij > 0)
        # Global VSE = sum_ij (column_fraction_ij * denom_ij) / sum_ij denom_ij
        porosity_thickness_initial_oil_saturation = (
            porosity_grid * thickness_grid * initial_oil_saturation
        )
        porosity_thickness_oil_saturation_delta = (
            porosity_grid
            * thickness_grid
            * np.maximum(initial_oil_saturation - current_oil_saturation, 0.0)
        )

        # Guard axis=2 for 2D grids
        if len(grid_shape) >= 3:
            denominator_per_column = np.sum(
                porosity_thickness_initial_oil_saturation.reshape(grid_shape),
                axis=2,
            )
            numerator_per_column = np.sum(
                porosity_thickness_oil_saturation_delta.reshape(grid_shape),
                axis=2,
            )
            initial_oil_volume_stb_per_column = np.nansum(
                initial_oil_volume_stb.reshape(grid_shape), axis=2
            )
        else:
            denominator_per_column = porosity_thickness_initial_oil_saturation
            numerator_per_column = porosity_thickness_oil_saturation_delta
            initial_oil_volume_stb_per_column = initial_oil_volume_stb

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            column_fraction = np.where(
                denominator_per_column > 0.0,
                numerator_per_column / denominator_per_column,
                0.0,
            )

        # Weight by initial oil (denominator_per_column * cell_area_ft2 / Bo)
        total_initial_oil_volume_stb_columns = np.nansum(
            initial_oil_volume_stb_per_column
        )
        if total_initial_oil_volume_stb_columns > 0.0:
            vertical_sweep_efficiency = (
                np.nansum(column_fraction * initial_oil_volume_stb_per_column)
                / total_initial_oil_volume_stb_columns
            )
        else:
            vertical_sweep_efficiency = 0.0
        return SweepEfficiencyAnalysis(
            volumetric_sweep_efficiency=float(volumetric_sweep_efficiency),
            displacement_efficiency=float(displacement_efficiency),
            recovery_efficiency=float(recovery_efficiency),
            contacted_oil=contacted_initial_oil_volume_stb,
            uncontacted_oil=uncontacted_oil_volume_stb,
            areal_sweep_efficiency=float(areal_sweep_efficiency),
            vertical_sweep_efficiency=float(vertical_sweep_efficiency),
        )

    def injection_front_analysis(
        self,
        step: int = -1,
        phase: typing.Literal["water", "gas"] = "water",
        threshold: float = 0.02,
    ) -> InjectionFrontAnalysis:
        """
        Track the spatial position and character of an injection-fluid front.

        The front is defined as every cell whose displacing-phase saturation has
        risen by at least *threshold* above its value in the earliest available
        state.  This is consistent with the contact detection used in
        :meth:`sweep_efficiency_analysis` but returns richer spatial information.

        Unlike `sweep_efficiency_analysis`, which reports aggregate
        efficiency scalars, this method returns the full saturation-delta grid and
        a front centroid so you can track plume migration step-by-step.

        :param step: Time step index to analyse; -1 gives the latest state.
        :param phase: Displacing phase to track - `"water"` or `"gas"`.
        :param threshold: Minimum saturation increase (fraction, e.g. 0.02) that
            declares a cell as contacted.  Lower values are more sensitive to
            numerical noise; higher values are more conservative.
        :return: `InjectionFrontAnalysis` with spatial front information.
        """
        state = self.get_state(step)
        initial_state = self.get_state(self._min_step)

        if state is None or initial_state is None:
            empty = np.zeros(1, dtype=bool)
            return InjectionFrontAnalysis(
                phase=phase,
                front_cells=empty,
                front_cell_count=0,
                front_volume_fraction=0.0,
                average_front_saturation=0.0,
                max_front_saturation=0.0,
                saturation_delta_grid=empty.astype(float),
                front_centroid=(0.0, 0.0, 0.0),
            )

        if phase == "water":
            current_sat = state.model.fluid_properties.water_saturation_grid
            initial_sat = initial_state.model.fluid_properties.water_saturation_grid
        else:
            current_sat = state.model.fluid_properties.gas_saturation_grid
            initial_sat = initial_state.model.fluid_properties.gas_saturation_grid

        delta = current_sat - initial_sat
        front_mask = delta > threshold  # contacted cells

        front_cell_count = int(np.count_nonzero(front_mask))

        # Pore-volume weighted volume fraction (accounts for heterogeneous
        # thickness, porosity, and NTG across the grid)
        grid_shape = state.model.grid_shape
        model = state.model
        cell_area_ft2 = model.cell_dimension[0] * model.cell_dimension[1]
        pore_volume_grid = (
            cell_area_ft2
            * model.thickness_grid
            * model.rock_properties.porosity_grid
            * model.rock_properties.net_to_gross_ratio_grid
        )
        total_pore_volume = float(np.nansum(pore_volume_grid))
        front_volume_fraction = (
            float(np.nansum(pore_volume_grid[front_mask])) / total_pore_volume
            if total_pore_volume > 0
            else 0.0
        )

        avg_front_saturation = (
            float(np.nanmean(current_sat[front_mask])) if front_cell_count > 0 else 0.0
        )
        max_front_saturation = (
            float(np.nanmax(current_sat[front_mask])) if front_cell_count > 0 else 0.0
        )

        # Saturation-delta weighted centroid (in cell-index units)
        # Weights the centre-of-mass toward cells with larger saturation
        # changes, giving a more physically representative plume centre.
        reshaped_mask = front_mask.reshape(grid_shape)
        reshaped_delta = delta.reshape(grid_shape)
        if front_cell_count > 0:
            idx = np.argwhere(reshaped_mask)  # shape (N, ndim)
            weights = reshaped_delta[reshaped_mask]  # saturation delta per front cell
            weight_sum = float(np.nansum(weights))
            if weight_sum > 0:
                centroid = tuple(
                    float(np.nansum(idx[:, d] * weights) / weight_sum)
                    for d in range(idx.shape[1])
                )
            else:
                centroid = tuple(float(np.mean(idx[:, d])) for d in range(idx.shape[1]))
            # Pad to 3-tuple if the grid is 2-D
            while len(centroid) < 3:
                centroid = centroid + (0.0,)
        else:
            centroid = (0.0, 0.0, 0.0)

        return InjectionFrontAnalysis(
            phase=phase,
            front_cells=front_mask.reshape(grid_shape),
            front_cell_count=front_cell_count,
            front_volume_fraction=front_volume_fraction,
            average_front_saturation=avg_front_saturation,
            max_front_saturation=max_front_saturation,
            saturation_delta_grid=delta.reshape(grid_shape),
            front_centroid=centroid,  # type: ignore[arg-type]
        )

    def recommend_ipr_method(
        self, step: int = -1
    ) -> typing.Literal["vogel", "linear", "fetkovich", "jones"]:
        """
        Recommend the most appropriate IPR method based on reservoir conditions.

        This method analyzes the current reservoir state and suggests the most
        suitable IPR correlation based on fluid properties and well conditions.

        :param step: The time step to analyze for IPR method recommendation
        :return: Recommended IPR method
        """
        state = self.get_state(step)
        if state is None:
            logger.debug(
                f"State at time step {step} is not available. Defaulting to 'vogel' IPR method."
            )
            return "vogel"

        avg_oil_saturation = np.nanmean(
            state.model.fluid_properties.oil_saturation_grid
        )
        avg_gas_saturation = np.nanmean(
            state.model.fluid_properties.gas_saturation_grid
        )
        reservoir_pressure = np.nanmean(state.model.fluid_properties.pressure_grid)
        estimated_bubble_point = np.nanmean(
            state.model.fluid_properties.oil_bubble_point_pressure_grid
        )

        # Check if this is primarily a gas reservoir
        if avg_gas_saturation >= 0.6:
            return "fetkovich"  # Best for gas wells

        # Check if we're above bubble point (single-phase oil)
        elif reservoir_pressure > estimated_bubble_point and avg_oil_saturation > 0.7:
            return "linear"  # Best for undersaturated oil

        # Check if we have significant multi-phase flow
        elif avg_oil_saturation > 0.3 and avg_gas_saturation > 0.2:
            return "jones"  # Best for complex multi-phase systems

        # Default to Vogel for solution gas drive reservoirs
        return "vogel"  # Best for two-phase oil/gas systems

    def productivity_analysis(
        self,
        step: int = -1,
        phase: typing.Literal["oil", "gas", "water"] = "oil",
        cells: typing.Union[Cells, CellFilter] = None,
    ) -> ProductivityAnalysis:
        """
        Well productivity analysis based on actual flow rates and reservoir properties.

        Analyzes well performance without requiring bottom hole pressure data. Computes
        flow rates, reservoir conditions, well indices, and flow efficiency metrics using
        only production data and formation properties.

        :param step: The time step index to analyze productivity for.
        :param phase: Phase to analyze ('oil', 'gas', 'water').
        :param cells: Optional filter for specific cells, well name, or region.
            - None: All production wells (default)
            - str: Specific well name (e.g., "PROD-1")
            - (i, j, k): Single cell
            - [(i1,j1,k1), ...]: List of cells
            - (slice, slice, slice): Region
        :return: `ProductivityAnalysis` containing productivity metrics based on actual production data.
        """
        # Resolve step and ensure cells before building cache key
        step = self._resolve_step(step)
        cells_obj = _ensure_cells(cells)
        cache_key = (step, phase, cells_obj)
        if cache_key in self._productivity_analysis_cache:
            return self._productivity_analysis_cache[cache_key]

        state = self.get_state(step)
        if state is None:
            logger.debug(
                f"State at time step {step} is not available. Returning zero productivity analysis."
            )
            return ProductivityAnalysis(
                total_flow_rate=0.0,
                average_reservoir_pressure=0.0,
                skin_factor=0.0,
                flow_efficiency=1.0,
                well_index=0.0,
                average_mobility=0.0,
            )

        if state.production_rates is None:
            logger.warning("No production data available for productivity analysis")
            return ProductivityAnalysis(
                total_flow_rate=0.0,
                average_reservoir_pressure=0.0,
                skin_factor=0.0,
                flow_efficiency=1.0,
                well_index=0.0,
                average_mobility=0.0,
            )

        production_wells = state.wells.production_wells
        if not production_wells:
            logger.warning("No production wells found for productivity analysis")
            return ProductivityAnalysis(
                total_flow_rate=0.0,
                average_reservoir_pressure=0.0,
                skin_factor=0.0,
                flow_efficiency=1.0,
                well_index=0.0,
                average_mobility=0.0,
            )

        # Get cell mask for filtering
        mask = (
            cells_obj.get_mask(state.model.grid_shape, state.wells)
            if cells_obj
            else None
        )

        # Accumulate metrics across all wells
        total_flow_rate = 0.0
        total_reservoir_pressure = 0.0
        total_skin_factor = 0.0
        total_flow_efficiency = 0.0
        total_well_index = 0.0
        total_mobility = 0.0
        active_wells = 0
        active_cells = 0

        ft3_to_bbl = c.CUBIC_FEET_TO_BARRELS
        for well in production_wells:
            if not well.is_open:
                continue

            # If filtering by well name, skip if not matching
            if isinstance(cells, str) and well.name != cells:
                continue

            active_wells += 1
            actual_skin_factor = well.skin_factor

            cell_locations = _expand_intervals(
                well.perforating_intervals, orientation=well.orientation
            )
            for cell_location in cell_locations:
                i, j, k = cell_location

                # Check if cell matches mask (if filtering)
                if mask is not None and not mask[i, j, k]:
                    continue

                cell_pressure = float(
                    state.model.fluid_properties.pressure_grid[i, j, k]
                )

                # Get actual cell flow rate and fluid properties
                if phase == "oil":
                    if state.production_rates.oil is None:
                        continue
                    oil_fvf = float(
                        state.model.fluid_properties.oil_formation_volume_factor_grid[
                            i, j, k
                        ]
                    )
                    cell_flow_rate_ft3 = state.production_rates.oil[i, j, k]  # type: ignore  # ft³/day
                    cell_flow_rate_stb = (
                        cell_flow_rate_ft3 * ft3_to_bbl / oil_fvf
                    )  # STB/day

                elif phase == "water":
                    if state.production_rates.water is None:
                        continue
                    water_fvf = float(
                        state.model.fluid_properties.water_formation_volume_factor_grid[
                            i, j, k
                        ]
                    )
                    cell_flow_rate_ft3 = state.production_rates.water[  # type: ignore
                        i, j, k
                    ]  # ft³/day
                    cell_flow_rate_stb = (
                        cell_flow_rate_ft3 * ft3_to_bbl / water_fvf
                    )  # STB/day

                else:  # gas
                    if state.production_rates.gas is None:
                        continue
                    gas_fvf = float(
                        state.model.fluid_properties.gas_formation_volume_factor_grid[
                            i, j, k
                        ]
                    )
                    cell_flow_rate_ft3 = state.production_rates.gas[i, j, k]  # type: ignore  # ft³/day
                    cell_flow_rate_stb = cell_flow_rate_ft3 / gas_fvf  # SCF/day

                if cell_flow_rate_stb == 0:
                    continue

                # Get well index and phase mobility for this cell
                interval_thickness = state.model.cell_dimension
                perm_x = float(
                    state.model.rock_properties.absolute_permeability.x[i, j, k]
                )
                perm_y = float(
                    state.model.rock_properties.absolute_permeability.y[i, j, k]
                )
                perm_z = float(
                    state.model.rock_properties.absolute_permeability.z[i, j, k]
                )
                permeability = (perm_x, perm_y, perm_z)

                well_index = well.get_well_index(
                    interval_thickness=interval_thickness,
                    permeability=permeability,
                    skin_factor=actual_skin_factor,
                )

                # Get phase mobility
                if phase == "oil":
                    phase_mobility = float(
                        state.relative_mobilities.oil_relative_mobility[i, j, k]
                    )
                elif phase == "water":
                    phase_mobility = float(
                        state.relative_mobilities.water_relative_mobility[i, j, k]
                    )
                else:  # gas
                    phase_mobility = float(
                        state.relative_mobilities.gas_relative_mobility[i, j, k]
                    )

                if phase_mobility <= 0 or well_index <= 0:
                    continue

                # Accumulate metrics
                total_flow_rate += abs(cell_flow_rate_stb)
                total_reservoir_pressure += cell_pressure
                total_well_index += well_index
                total_mobility += phase_mobility
                active_cells += 1

            # Flow efficiency using actual skin factor
            well_flow_efficiency = (
                1.0 / (1.0 + actual_skin_factor) if actual_skin_factor > -1 else 1.0
            )
            total_skin_factor += actual_skin_factor
            total_flow_efficiency += well_flow_efficiency

        # Calculate averages
        if active_wells > 0 and active_cells > 0:
            avg_flow_rate = total_flow_rate
            avg_reservoir_pressure = total_reservoir_pressure / active_cells
            avg_skin_factor = total_skin_factor / active_wells
            avg_flow_efficiency = total_flow_efficiency / active_wells
            avg_well_index = total_well_index / active_cells
            avg_mobility = total_mobility / active_cells
            logger.debug(
                f"Productivity analysis complete: active_wells={active_wells}, "
                f"flow_rate={avg_flow_rate:.2f}, pressure={avg_reservoir_pressure:.2f} psia, "
                f"skin={avg_skin_factor:.4f}, efficiency={avg_flow_efficiency:.4f}, "
                f"WI={avg_well_index:.4f}, mobility={avg_mobility:.4f}"
            )
        else:
            avg_flow_rate = 0.0
            avg_reservoir_pressure = 0.0
            avg_skin_factor = 0.0
            avg_flow_efficiency = 1.0
            avg_well_index = 0.0
            avg_mobility = 0.0
            logger.warning(
                "No active production wells or cells found for productivity analysis"
            )

        result = ProductivityAnalysis(
            total_flow_rate=float(avg_flow_rate),
            average_reservoir_pressure=float(avg_reservoir_pressure),
            skin_factor=avg_skin_factor,
            flow_efficiency=avg_flow_efficiency,
            well_index=float(avg_well_index),
            average_mobility=float(avg_mobility),
        )
        self._productivity_analysis_cache[cache_key] = result
        return result

    def voidage_replacement_ratio(
        self, step: int = -1, cells: typing.Union[Cells, CellFilter] = None
    ) -> float:
        """
        Calculates the voidage replacement ratio (VRR) at a specific time step.

        The VRR measures the ratio of injected reservoir volumes to produced reservoir volumes:
        VRR = (Injected volumes) / (Produced volumes)

        VRR = (Wi*Bwi + Ggi*Bginj) / (Np*Bo + Wp*Bwp + Bg*(GOR - Rs)*Np)

        Where:
        - Wi = Water injected (m3 or STB)
        - Bwi = Injected water formation volume factor (m3/m3 or bbl/STB)
        - Ggi = Injected gas volume (m3 or SCF)
        - Bginj = Injected gas formation volume factor at reservoir pressure (m3/m3 or bbl/SCF)
        - Np = Oil produced (m3 or STB)
        - Bo = Oil formation volume factor (m3/m3 or bbl/STB)
        - Wp = Water produced (m3 or STB)
        - Bwp = Produced water formation volume factor (m3/m3 or bbl/STB)
        - Bg = Gas formation volume factor (m3/m3 or bbl/SCF)
        - GOR = Produced gas oil ratio (m3/m3 or SCF/STB)
        - Rs = Solution gas oil ratio (m3/m3 or SCF/STB)

        A VRR > 1.0 indicates pressure maintenance through injection.
        A VRR = 1.0 indicates a balanced reservoir.
        A VRR < 1.0 indicates reservoir pressure is declining.

        :param step: The time step index to calculate VRR for.
        :param cells: Optional filter for specific cells, well name, or region.
            - None: Entire reservoir (default)
            - str: Well name (e.g., "PROD-1", "INJ-1")
            - (i, j, k): Single cell
            - [(i1,j1,k1), ...]: List of cells
            - (slice, slice, slice): Region
        :return: The voidage replacement ratio (dimensionless fraction).
        """
        cells_obj = _ensure_cells(cells)

        state = self.get_state(step)
        if state is None:
            logger.debug(
                f"State at time step {step} is not available. Returning zero VRR."
            )
            return 0.0

        # Get cumulative injection/production volumes
        cumulative_water_injected = self.water_injected(
            self._min_step, step, cells=cells_obj
        )
        cumulative_gas_injected = self.gas_injected(
            self._min_step, step, cells=cells_obj
        )
        cumulative_oil_produced = self.oil_produced(
            self._min_step, step, cells=cells_obj
        )
        cumulative_water_produced = self.water_produced(
            self._min_step, step, cells=cells_obj
        )
        free_gas_produced = self.gas_produced(self._min_step, step, cells=cells_obj)

        injected_gas_fvf_grid, injected_water_fvf_grid = _build_injected_fvf_grids(
            wells=state.wells,
            pressure_grid=state.model.fluid_properties.pressure_grid,
            temperature_grid=state.model.fluid_properties.temperature_grid,
            grid_shape=state.model.grid_shape,
        )

        avg_oil_fvf = np.nanmean(
            state.model.fluid_properties.oil_formation_volume_factor_grid
        )
        avg_gas_fvf = np.nanmean(
            state.model.fluid_properties.gas_formation_volume_factor_grid
        )
        avg_water_fvf_produced = np.nanmean(
            state.model.fluid_properties.water_formation_volume_factor_grid
        )

        avg_injected_water_fvf = (
            float(np.nanmean(injected_water_fvf_grid))
            if not np.all(np.isnan(injected_water_fvf_grid))
            else avg_water_fvf_produced
        )
        avg_injected_gas_fvf = (
            float(np.nanmean(injected_gas_fvf_grid))
            if not np.all(np.isnan(injected_gas_fvf_grid))
            else avg_gas_fvf
        )

        # Calculate injected reservoir volumes (numerator)
        ft3_to_bbl = c.CUBIC_FEET_TO_BARRELS
        injected_water_bbl = cumulative_water_injected * avg_injected_water_fvf
        injected_gas_bbl = cumulative_gas_injected * avg_injected_gas_fvf * ft3_to_bbl
        total_injected_volume = injected_water_bbl + injected_gas_bbl

        # Calculate produced reservoir volumes (denominator)
        produced_oil_bbl = cumulative_oil_produced * avg_oil_fvf
        produced_water_bbl = cumulative_water_produced * avg_water_fvf_produced

        # Free gas produced (already free gas only from `gas_produced`)
        # Plus solution gas that came out of the oil
        # Calculate solution gas step-by-step to account for pressure-dependent Rs
        solution_gas_produced = 0.0
        days_per_second = c.DAYS_PER_SECOND
        for s in self._sorted_steps:
            if s > step:
                break
            st = self._states[s]
            solution_gor_grid = st.model.fluid_properties.solution_gas_to_oil_ratio_grid
            oil_production = st.production_rates.oil
            step_in_days = st.step_size * days_per_second
            oil_fvf_grid = st.model.fluid_properties.oil_formation_volume_factor_grid
            oil_production_stb = oil_production * ft3_to_bbl / oil_fvf_grid
            solution_gas_produced += float(
                np.nansum(solution_gor_grid * oil_production_stb) * step_in_days
            )

        total_gas_produced = free_gas_produced + solution_gas_produced  # SCF
        produced_gas_bbl = total_gas_produced * avg_gas_fvf * ft3_to_bbl

        # Calculate VRR
        total_produced_volume = produced_oil_bbl + produced_water_bbl + produced_gas_bbl
        if total_produced_volume <= 0:
            return 0.0

        voidage_replacement_ratio = total_injected_volume / total_produced_volume
        return float(voidage_replacement_ratio)

    vrr = VRR = voidage_replacement_ratio

    def recommend_decline_model(
        self,
        from_step: int = 0,
        to_step: int = -1,
        phase: typing.Literal["oil", "gas", "water"] = "oil",
        max_decline_per_year: float = 2.0,
    ) -> typing.Tuple[str, typing.Dict[str, DeclineCurveResult]]:
        """
        Automatically recommend the best decline curve model based on statistical fit and physical constraints.

        This method fits all three decline curve models (exponential, harmonic, and hyperbolic) to the
        historical production data and recommends the most appropriate model based on multiple criteria:

        1. Statistical Fit Quality (R² coefficient of determination)
        2. Physical Reasonableness of Parameters
        3. Standard Industry Guidelines

        Model Selection Criteria:

        Exponential Decline - Best For:
        - Wells in boundary-dominated flow (late-time production)
        - Wells with relatively constant pressure drawdown
        - Single-phase flow above bubble point
        - Conservative reserves estimates
        - Short production history where flow regime is uncertain

        Characteristics:
        - Constant percentage decline rate
        - Most conservative EUR estimates
        - Simple to extrapolate and understand
        - Commonly required for SEC reserves reporting

        Equation: q(t) = qi x exp(-Di x t)

        Harmonic Decline - Best For:
        - Wells with linear flow geometry (fractured wells, channel reservoirs)
        - Wells experiencing stable water influx
        - Gas wells in steady-state flow
        - Certain infinite-acting reservoirs

        Characteristics:
        - Decline rate decreases inversely with time
        - Can overestimate EUR if misapplied
        - Special case of hyperbolic with b = 1

        Equation: q(t) = qi / (1 + Di x t)

        Hyperbolic Decline - Best For:
        - Wells transitioning between flow regimes
        - Multi-phase flow systems
        - Wells with changing flowing conditions
        - Modern unconventional wells (shale, tight formations)
        - Wells with sufficient production history (>1-2 years)

        Characteristics:
        - Most flexible model with b-factor controlling curvature
        - 0 < b < 1: Most common for conventional oil wells
        - b ≈ 0.5: Typical for many hydraulically fractured wells
        - Requires more data for reliable parameter estimation

        Equation: q(t) = qi / (1 + b x Di x t)^(1/b)

        Parameter Validation Rules:

        Decline Rate (Di):
        - Valid range: 0.0 to 2.0 per year (0% to 200% annual decline)
        - Typical ranges:
          * Conventional oil wells: 5-25% per year (0.05-0.25)
          * Tight oil/shale: 30-80% first year (0.3-0.8)
          * Gas wells: 10-40% per year (0.1-0.4)
          * Mature fields: <10% per year (<0.1)
        - Models with Di outside valid range are rejected

        Hyperbolic Exponent (b):
        - Valid range: 0.0 to 2.0 (dimensionless)
        - Physical interpretation:
          * b = 0: Exponential decline
          * 0 < b < 0.3: Slightly curved, approaching exponential
          * 0.3 < b < 0.7: Moderate curvature, common for fractured wells
          * 0.7 < b < 1.0: Strong curvature, approaching harmonic
          * b = 1.0: Harmonic decline
          * b > 1.0: Very slow decline, often unrealistic for long-term forecast
        - SEC guidelines: b > 1.0 often requires justification
        - Models with b outside 0.0-2.0 range are rejected

        R² (Coefficient of Determination):
        - Measures goodness of fit: R² = 1 - (SS_residual / SS_total)
        - R² ranges from 0.0 to 1.0:
          * R² > 0.95: Excellent fit
          * 0.90 < R² < 0.95: Good fit
          * 0.80 < R² < 0.90: Acceptable fit
          * R² < 0.80: Poor fit, model may not be appropriate
        - Higher R² is preferred, but must also pass physical constraints

        Selection Algorithm:
        1. Fit all three models to historical data
        2. Eliminate models with physically unreasonable parameters
        3. Among remaining models, select the one with highest R²
        4. If all models fail validation, return "exponential" as conservative default

        Minimum Data Requirements:
        - Exponential: Minimum 6-12 months (2-4 data points)
        - Harmonic: Minimum 12-18 months (4-6 data points)
        - Hyperbolic: Minimum 18-24 months (6-8 data points) for reliable b-factor

        :param from_step: Starting time step index for analysis (inclusive).
            Should include stable production period, avoiding
            initial transient flow or well cleanup period.
        :param to_step: Ending time step index for analysis (inclusive).
            Use -1 for most recent time step. More data generally
            improves fit quality, but ensure data represents similar
            operating conditions.
        :param phase: Phase to analyze - "oil", "gas", or "water".
            Different phases may have different optimal decline models.
            Gas typically has steeper decline than oil.
        :param max_decline_per_year: Maximum acceptable rate decline factor per year (e.g 2.0 for 200%)
        :return: Tuple containing:
            1. str: Recommended model name ("exponential", "harmonic", or "hyperbolic")
            2. dict: Dictionary with all three model results for comparison
                Keys: "exponential", "harmonic", "hyperbolic"
                Values: DeclineCurveResult objects with parameters and fit statistics

        Example:
        ```python
        # Get recommendation and all model results
        recommended, all_models = analyst.recommend_decline_model(
            from_step=0,
            to_step=-1,
            phase="oil"
        )
        print(f"Recommended model: {recommended}")
        print(f"R² = {all_models[recommended].r_squared:.4f}")
        print(f"Di = {all_models[recommended].decline_rate_per_timestep:.3f} per year")
        if recommended == "hyperbolic":
            print(f"b = {all_models[recommended].b_factor:.3f}")

        # Compare EUR estimates from all models
        for model_name, result in all_models.items():
            if not result.error:
                eur = analyst.estimate_economic_ultimate_recovery(result, forecast_steps=30)
                print(f"{model_name}: EUR = {eur:,.0f} STB")

        # Use recommended model for forecasting
        best_model = all_models[recommended]
        forecast = analyst.forecast_production(best_model, steps=365*10)
        ```

        Notes:
        - Always review all three model fits visually before accepting recommendation
        - High R² does not guarantee accurate long-term forecasts
        - Consider using multiple models for P10/P50/P90 reserve scenarios:
            * P90 (conservative): Exponential or steepest decline
            * P50 (best estimate): Recommended model
            * P10 (optimistic): Harmonic or shallowest decline
        - For regulatory reporting (SEC), exponential is often required regardless of fit
        - Hyperbolic decline should transition to exponential for long-term forecasts
            (typically after 3-5 years or when bxDixt > 0.5)
        - Be cautious with harmonic decline (b=1.0) as it may overestimate reserves
        - If well has undergone stimulation, workover, or facility changes during
            history period, consider analyzing pre- and post-change periods separately
        - Water production typically follows different decline patterns than oil/gas
        - For unconventional wells, hyperbolic is usually most appropriate for
            first 2-5 years, then transition to exponential

        Industry Best Practices:
        - SPE: Use hyperbolic to exponential transition for proved reserves
        - SEC: Often requires exponential for proved undeveloped reserves
        - Operators: Use model that best matches analog well performance
        - Always perform sensitivity analysis on decline parameters
        - Update decline curves annually as new production data becomes available
        """
        models = ["exponential", "hyperbolic", "harmonic"]
        results: typing.Dict[str, DeclineCurveResult] = {}

        for model in models:
            result = self.decline_curve_analysis(
                from_step=from_step,
                to_step=to_step,
                decline_type=model,  # type: ignore
                phase=phase,
                max_decline_per_year=max_decline_per_year,
            )
            results[model] = result

        # Get time step size to convert decline rate bounds
        state = self.get_state(from_step)
        if state is None:
            raise ValidationError(
                f"State at time step {from_step} is not available for decline model recommendation."
            )
        step_size_seconds = state.step_size
        timesteps_per_year = c.SECONDS_PER_YEAR / step_size_seconds

        # Convert typical decline rate bounds from per-year to per-timestep
        max_decline_per_timestep = max_decline_per_year / timesteps_per_year

        # Recommend based on R² and physical reasonableness
        best_model = "exponential"
        best_r_squared = 0.0

        for model, result in results.items():
            if result.error:
                continue

            # Check for physically reasonable parameters (adjusted for timestep units)
            if (
                result.decline_rate_per_timestep < 0
                or result.decline_rate_per_timestep > max_decline_per_timestep
            ):
                continue

            if model == "hyperbolic" and (result.b_factor < 0 or result.b_factor > 2.0):
                continue

            if result.r_squared > best_r_squared:
                best_r_squared = result.r_squared
                best_model = model

        return best_model, results

    def decline_curve_analysis(
        self,
        from_step: int = 0,
        to_step: int = -1,
        decline_type: typing.Literal[
            "exponential", "hyperbolic", "harmonic"
        ] = "exponential",
        phase: typing.Literal["oil", "gas", "water"] = "oil",
        max_decline_per_year: float = 2.0,
    ) -> DeclineCurveResult:
        """
        Fits a decline curve model to production data over a specified time range.

        Performs decline curve analysis using exponential, hyperbolic, or harmonic models:
        - Exponential: q = qi * exp(-Di * t)
        - Hyperbolic: q = qi / (1 + b * Di * t)^(1/b)
        - Harmonic: q = qi / (1 + Di * t) [special case of hyperbolic with b=1]

        NOTE: All decline rates are stored per time step (not per year).

        :param from_step: Starting time step for analysis.
        :param to_step: Ending time step for analysis.
        :param decline_type: Type of decline curve ('exponential', 'hyperbolic', 'harmonic').
        :param phase: Phase to analyze ('oil', 'gas', 'water').
        :param max_decline_per_year: Maximum acceptable rate decline factor per year (e.g 2.0 for 200%)
        :return: `DeclineCurveResult` containing fitted decline curve parameters and forecasts.
        """
        if decline_type not in {"exponential", "hyperbolic", "harmonic"}:
            raise ValueError("Invalid decline type specified for analysis.")

        if phase not in {"oil", "gas", "water"}:
            raise ValueError("Invalid phase specified for decline curve analysis.")

        to_step = self._resolve_step(to_step)

        # Collect production rate data over specified time range
        steps_list = []
        production_rates_list = []

        for step in range(from_step, to_step + 1):
            instantaneous_rates = self.instantaneous_production_rates(step)
            steps_list.append(step)

            if phase == "oil":
                production_rates_list.append(instantaneous_rates.oil_rate)
            elif phase == "gas":
                production_rates_list.append(instantaneous_rates.gas_rate)
            else:  # phase == "water"
                production_rates_list.append(instantaneous_rates.water_rate)

        steps_array = np.array(steps_list)
        production_rates_array = np.array(production_rates_list)

        # Filter out zero and negative rates for meaningful decline analysis
        positive_production_mask = production_rates_array > 0
        positive_count = np.sum(positive_production_mask)
        logger.debug(
            f"Decline curve analysis: phase={phase}, decline_type={decline_type}, "
            f"steps={from_step} to {to_step}, positive_rates={positive_count}"
        )
        if positive_count < 2:
            logger.warning(
                f"Insufficient positive {phase} rate data for decline curve analysis: "
                f"only {positive_count} positive rate(s)"
            )
            return DeclineCurveResult(
                decline_type=decline_type,
                initial_rate=0.0,
                decline_rate_per_timestep=0.0,  # Will be 0 for error cases
                b_factor=0.0,
                r_squared=0.0,
                phase=phase,
                error=f"Insufficient positive {phase} rate data for analysis",
                steps=None,
                actual_rates=None,
                predicted_rates=None,
            )

        filtered_steps = steps_array[positive_production_mask]
        positive_production_rates = production_rates_array[positive_production_mask]

        if decline_type == "exponential":
            # Exponential decline: q = qi * exp(-Di*t)
            # Use linear regression on ln(q) vs t to find parameters
            log_production_rates = np.log(positive_production_rates)
            linear_regression_coefficients = np.polyfit(
                filtered_steps, log_production_rates, 1
            )

            exponential_decline_rate_per_timestep = -linear_regression_coefficients[0]
            log_initial_rate_intercept = linear_regression_coefficients[1]
            exponential_initial_production_rate = np.exp(log_initial_rate_intercept)

            # Calculate coefficient of determination (R²) for goodness of fit
            predicted_exponential_rates = exponential_initial_production_rate * np.exp(
                -exponential_decline_rate_per_timestep * filtered_steps
            )
            sum_squared_residuals = np.sum(
                (positive_production_rates - predicted_exponential_rates) ** 2
            )
            total_sum_squares = np.sum(
                (positive_production_rates - np.mean(positive_production_rates)) ** 2
            )
            exponential_r_squared = (
                1 - (sum_squared_residuals / total_sum_squares)
                if total_sum_squares > 0
                else 0.0
            )

            # Return fitted qi as `initial_rate` (using last actual rate leads to forecast
            # discontinuities and doesn't reflect the regression result)
            return DeclineCurveResult(
                decline_type="exponential",
                initial_rate=float(exponential_initial_production_rate),
                decline_rate_per_timestep=exponential_decline_rate_per_timestep,  # Now per timestep
                b_factor=0.0,
                r_squared=exponential_r_squared,
                phase=phase,
                error=None,
                steps=filtered_steps.tolist(),
                actual_rates=positive_production_rates.tolist(),
                predicted_rates=predicted_exponential_rates.tolist(),
            )

        if decline_type == "harmonic":
            # Harmonic decline: q = qi / (1 + Di*t) [special case of hyperbolic with b=1]
            # Use linear regression on 1/q vs t to find parameters
            reciprocal_production_rates = 1.0 / positive_production_rates
            harmonic_regression_coefficients = np.polyfit(
                filtered_steps, reciprocal_production_rates, 1
            )

            harmonic_decline_rate_per_timestep = harmonic_regression_coefficients[0]
            reciprocal_initial_rate_intercept = harmonic_regression_coefficients[1]
            harmonic_initial_production_rate = 1.0 / reciprocal_initial_rate_intercept

            # Calculate predicted rates and R²
            predicted_harmonic_rates = harmonic_initial_production_rate / (
                1.0 + harmonic_decline_rate_per_timestep * filtered_steps
            )
            harmonic_sum_squared_residuals = np.sum(
                (positive_production_rates - predicted_harmonic_rates) ** 2
            )
            harmonic_total_sum_squares = np.sum(
                (positive_production_rates - np.mean(positive_production_rates)) ** 2
            )
            harmonic_r_squared = (
                1 - (harmonic_sum_squared_residuals / harmonic_total_sum_squares)
                if harmonic_total_sum_squares > 0
                else 0.0
            )
            return DeclineCurveResult(
                decline_type="harmonic",
                initial_rate=float(harmonic_initial_production_rate),
                decline_rate_per_timestep=harmonic_decline_rate_per_timestep,  # Now per timestep
                b_factor=1.0,  # Harmonic decline has b=1
                r_squared=harmonic_r_squared,
                phase=phase,
                error=None,
                steps=filtered_steps.tolist(),
                actual_rates=positive_production_rates.tolist(),
                predicted_rates=predicted_harmonic_rates.tolist(),
            )

        # decline_type == "hyperbolic":
        # Hyperbolic decline: q = qi / (1 + b*Di*t)^(1/b)
        # This requires non-linear regression using SciPy `curve_fit`
        def hyperbolic_decline_function(
            time_array, initial_rate_param, decline_rate_param, b_factor_param
        ):
            """Hyperbolic decline curve function for curve fitting."""
            return initial_rate_param / (
                1 + b_factor_param * decline_rate_param * time_array
            ) ** (1 / b_factor_param)

        # Initial parameter estimates
        estimated_initial_rate = positive_production_rates[0]
        estimated_decline_rate = 0.001  # Small initial estimate per timestep
        estimated_b_factor = 0.5  # Typical hyperbolic exponent

        # Get time step size to set appropriate bounds for decline rate
        # Typical decline rates: 0.05-0.8 per year for oil wells
        # Convert to per-timestep by dividing by number of timesteps per year
        state = self.get_state(from_step)
        if state is None:
            logger.error(
                f"State at time step {from_step} not available for decline curve analysis."
            )
            raise ValidationError(f"State at time step {from_step} is not available.")

        step_size_seconds = state.step_size
        timesteps_per_year = c.SECONDS_PER_YEAR / step_size_seconds
        logger.debug(
            f"Decline curve analysis: timesteps_per_year={timesteps_per_year:.2f}, "
            f"step_size={step_size_seconds:.2f}s"
        )

        # Upper bound for decline rate: 2.0 per year → 2.0/timesteps_per_year per timestep
        max_decline_per_timestep = max_decline_per_year / timesteps_per_year

        # Perform non-linear curve fitting
        optimized_parameters, _ = curve_fit(
            hyperbolic_decline_function,
            filtered_steps,
            positive_production_rates,
            p0=[estimated_initial_rate, estimated_decline_rate, estimated_b_factor],
            bounds=([0, 0, 0.1], [np.inf, max_decline_per_timestep, 2.0]),
            maxfev=5000,
        )

        (
            hyperbolic_initial_rate,
            hyperbolic_decline_rate_per_timestep,
            hyperbolic_b_factor,
        ) = optimized_parameters

        # Calculate predicted rates and R²
        predicted_hyperbolic_rates = hyperbolic_decline_function(
            filtered_steps,
            hyperbolic_initial_rate,
            hyperbolic_decline_rate_per_timestep,
            hyperbolic_b_factor,
        )
        hyperbolic_sum_squared_residuals = np.sum(
            (positive_production_rates - predicted_hyperbolic_rates) ** 2
        )
        hyperbolic_total_sum_squares = np.sum(
            (positive_production_rates - np.mean(positive_production_rates)) ** 2
        )
        hyperbolic_r_squared = (
            1 - (hyperbolic_sum_squared_residuals / hyperbolic_total_sum_squares)
            if hyperbolic_total_sum_squares > 0
            else 0.0
        )

        logger.info(
            f"Decline curve analysis complete: type=hyperbolic, phase={phase}, "
            f"qi={hyperbolic_initial_rate:.2f}, Di={hyperbolic_decline_rate_per_timestep:.6f}/timestep, "
            f"b={hyperbolic_b_factor:.4f}, R²={hyperbolic_r_squared:.4f}"
        )
        return DeclineCurveResult(
            decline_type="hyperbolic",
            initial_rate=float(hyperbolic_initial_rate),  #  use fitted qi
            decline_rate_per_timestep=float(
                hyperbolic_decline_rate_per_timestep
            ),  # Now per timestep
            b_factor=float(hyperbolic_b_factor),
            r_squared=hyperbolic_r_squared,
            phase=phase,
            error=None,
            steps=filtered_steps.tolist(),
            actual_rates=positive_production_rates.tolist(),
            predicted_rates=predicted_hyperbolic_rates.tolist(),
        )

    dca = decline_curve_analysis

    def forecast_production(
        self,
        decline_result: DeclineCurveResult,
        steps: int,
        economic_limit: typing.Optional[float] = None,
    ) -> typing.List[typing.Tuple[int, float]]:
        """
        Forecast future production rates based on fitted decline curve parameters.

        This method extrapolates production rates into the future using the decline curve
        model and parameters obtained from historical data analysis. The forecast continues
        until either the specified number of time steps is reached or the production rate
        falls below the economic limit.

        Forecast Equations by Decline Type:

        1. Exponential Decline:
        q(t) = qi x exp(-Di x t)

        2. Harmonic Decline:
        q(t) = qi / (1 + Di x t)

        3. Hyperbolic Decline:
        q(t) = qi / (1 + b x Di x t)^(1/b)

        Where:
        - q(t) = production rate at time step t (STB/day for oil/water, scf/day for gas)
        - qi = initial production rate at start of forecast
        - Di = decline rate per time step (NOT per year)
        - b = hyperbolic exponent (0 < b ≤ 2)
        - t = number of time steps since start of forecast

        NOTE: Production rates are in "per day" units (STB/day, scf/day) regardless of
        time step size. The decline rate is per timestep, but rates remain in per-day units.

        :param decline_result: `DeclineCurveResult` object containing fitted decline parameters
            from `decline_curve_analysis()` method. Must have valid parameters
            (no error flag set).
        :param steps: Number of time steps to forecast into the future.
        :param economic_limit: Optional minimum economic production rate in per-day units
            (STB/day for oil/water, SCF/day for gas). Forecasting stops when predicted
            rate falls below this value. Default is None (no economic limit applied).
        :return: List of tuples containing (step, forecasted_rate). Rates are in
            per-day units (STB/day or scf/day). Time steps are absolute values continuing
            from the last historical time step. Returns empty list if `decline_result`
            contains errors.
        """
        if decline_result.error:
            return []

        last_step = decline_result.steps[-1] if decline_result.steps else 0
        forecast = []
        decline_rate_per_timestep = decline_result.decline_rate_per_timestep

        for t in range(1, steps + 1):
            future_time = last_step + t
            time_since_start = t  # Time steps since start of forecast

            if decline_result.decline_type == "exponential":
                rate = decline_result.initial_rate * np.exp(
                    -decline_rate_per_timestep * time_since_start
                )
            elif decline_result.decline_type == "harmonic":
                rate = decline_result.initial_rate / (
                    1 + decline_rate_per_timestep * time_since_start
                )
            elif decline_result.decline_type == "hyperbolic":
                rate = decline_result.initial_rate / (
                    1
                    + decline_result.b_factor
                    * decline_rate_per_timestep
                    * time_since_start
                ) ** (1 / decline_result.b_factor)
            else:
                rate = 0.0

            if economic_limit is not None and rate < economic_limit:
                break

            forecast.append((future_time, float(rate)))
        return forecast

    def estimate_economic_ultimate_recovery(
        self,
        decline_result: DeclineCurveResult,
        forecast_steps: int,
        economic_limit: float = 0.0,
    ) -> float:
        """
        Calculate Estimated Ultimate Recovery (EUR) using decline curve analysis.

        EUR represents the total cumulative production expected from a well or reservoir
        over its economic life. This method uses analytical integration of decline curve
        equations to calculate cumulative production.

        Cumulative Production Equations by Decline Type:

        1. Exponential Decline:
        Np = (qi - qf) / Di

        2. Harmonic Decline:
        Np = (qi / Di) x ln(qi / qf)

        3. Hyperbolic Decline:
        For b ≠ 1:
        Np = (qi^b / (Di x (1-b))) x [qi^(1-b) - qf^(1-b)]

        For b = 1 (harmonic case):
        Np = (qi / Di) x ln(qi / qf)

        Where:
        - Np = cumulative production (STB for oil/water, scf for gas)
        - qi = initial production rate (STB/day or scf/day)
        - qf = final production rate (STB/day or scf/day, at forecast end or economic limit)
        - Di = decline rate per time step
        - b = hyperbolic exponent (0 < b ≤ 2)

        NOTE: The returned EUR is in stock tank barrels (STB) for oil/water or standard
        cubic feet (scf) for gas. The calculation accounts for the time step size by
        using the integrated decline equations which naturally handle the time step units.

        :param decline_result: `DeclineCurveResult` object from `decline_curve_analysis()` containing
            the fitted decline parameters (qi, Di, b) and decline type.
        :param forecast_steps: Number of time steps to forecast for EUR calculation.
        :param economic_limit: Minimum economic production rate in per-day units
            (STB/day for oil/water, scf/day for gas). Production below this
            rate is assumed to be uneconomic. Default is 0.0 (no limit).
        :return: Estimated Ultimate Recovery (EUR) in stock tank barrels (STB) for oil/water
            or standard cubic feet (scf) for gas. Returns 0.0 if decline_result contains
            errors or if parameters are invalid.
        """
        if decline_result.error:
            logger.error(decline_result.error)
            return 0.0

        qi = decline_result.initial_rate  # STB/day or scf/day
        di = decline_result.decline_rate_per_timestep
        b = decline_result.b_factor

        if di <= 0:
            return 0.0

        # Calculate final rate after `forecast_steps` (in STB/day or scf/day)
        if decline_result.decline_type == "exponential":
            q_final = qi * np.exp(-di * forecast_steps)
        elif decline_result.decline_type == "harmonic":
            q_final = qi / (1 + di * forecast_steps)
        elif decline_result.decline_type == "hyperbolic":
            q_final = qi / (1 + b * di * forecast_steps) ** (1 / b)
        else:
            return 0.0

        # Apply economic limit (both in per-day units)
        q_final = max(q_final, economic_limit)

        # Calculate time to reach economic limit if applicable
        if economic_limit > 0 and q_final == economic_limit:
            if decline_result.decline_type == "exponential":
                time_to_limit = (
                    -np.log(economic_limit / qi) / di if qi > economic_limit else 0
                )
            elif decline_result.decline_type == "harmonic":
                time_to_limit = (
                    (qi - economic_limit) / (di * economic_limit)
                    if economic_limit > 0
                    else 0
                )
            elif decline_result.decline_type == "hyperbolic":
                if b > 0:
                    time_to_limit = ((qi / economic_limit) ** b - 1) / (b * di)
                else:
                    time_to_limit = 0
            else:
                time_to_limit = 0

            # Preserve fractional steps; truncating to int before min() discards
            # the fractional portion and can undercount EUR by up to one step's production
            effective_steps = min(float(forecast_steps), time_to_limit)

            # Recalculate q_final at effective time
            if decline_result.decline_type == "exponential":
                q_final = qi * np.exp(-di * effective_steps)
            elif decline_result.decline_type == "harmonic":
                q_final = qi / (1 + di * effective_steps)
            elif decline_result.decline_type == "hyperbolic":
                q_final = qi / (1 + b * di * effective_steps) ** (1 / b)

        # Calculate cumulative production analytically
        # NOTE: These formulas give results in units of [rate x time]
        # Since rate is in per-day units and time is in timesteps, we get [per-day x timesteps]
        # This naturally gives us the correct volume units (STB or scf) as long as
        # the decline rate Di is per timestep and qi, qf are per day

        if decline_result.decline_type == "exponential":
            # Q = (qi - q_final) / Di
            # Units: [(STB/day) / (1/timestep)] = [STB/day x timestep]
            # If timestep = 1 day, this gives STB ✓
            cumulative = (qi - q_final) / di

        elif decline_result.decline_type == "harmonic":
            # Q = (qi / Di) * ln(qi / q_final)
            if q_final > 0:
                cumulative = (qi / di) * np.log(qi / q_final)
            else:
                cumulative = 0.0

        elif decline_result.decline_type == "hyperbolic":
            # Q = qi^b / (Di*(1-b)) * [qi^(1-b) - q_final^(1-b)]
            if abs(b - 1.0) < 0.001:  # Close to harmonic
                if q_final > 0:
                    cumulative = (qi / di) * np.log(qi / q_final)
                else:
                    cumulative = 0.0
            else:
                cumulative = (
                    (qi**b) / (di * (1 - b)) * (qi ** (1 - b) - q_final ** (1 - b))
                )
        else:
            cumulative = 0.0

        return float(cumulative)

    def mobility_ratio(
        self,
        displaced_phase: typing.Literal["oil", "water"] = "oil",
        displacing_phase: typing.Literal["oil", "water", "gas"] = "water",
        step: int = -1,
    ) -> float:
        """
        Calculate the mobility ratio between two fluid phases in the reservoir.

        Mobility ratio (M) is defined as the ratio of the mobility of the displacing
        phase to the mobility of the displaced phase:

            M = λd / λr

        Where:
        - λd = mobility of displacing phase (e.g., water)
        - λr = mobility of displaced phase (e.g., oil)

        Mobility (λ) is calculated as:

            λ = kr_α / μ

        Where:
        - kr_α = Relative permeability of the phase (fraction)
        - μ = viscosity of the phase (cP)

        A mobility ratio less than 1 (M < 1) indicates a stable displacement,
        while a mobility ratio greater than 1 (M > 1) indicates an unstable displacement

        :param displaced_phase: The phase being displaced ('oil' or 'water').
        :param displacing_phase: The phase doing the displacing ('oil' or 'water').
        :param step: Time step index to analyze. Use -1 for the most recent time step.
        :return: Mobility ratio (M). Returns float('inf') if calculation is not possible.
        """
        if displaced_phase not in {"oil", "water"}:
            raise ValidationError(
                "Invalid displaced phase specified for mobility ratio."
            )

        if displacing_phase not in {"oil", "water", "gas"}:
            raise ValidationError(
                "Invalid displacing phase specified for mobility ratio."
            )

        state = self.get_state(step)
        if state is None:
            logger.debug(
                f"State at time step {step} not available for mobility ratio calculation. "
                "Returning inf."
            )
            return float("inf")

        fluid_properties = state.model.fluid_properties
        relative_permeabilities = state.relative_permeabilities
        logger.debug(
            f"Computing mobility ratio: displaced={displaced_phase}, displacing={displacing_phase}, "
            f"step={step}"
        )

        if displacing_phase == "water":
            displacing_viscosity_grid = fluid_properties.water_viscosity_grid
            displacing_rel_perm_grid = relative_permeabilities.krw
        elif displacing_phase == "oil":
            displacing_viscosity_grid = fluid_properties.oil_viscosity_grid
            displacing_rel_perm_grid = relative_permeabilities.kro
        else:  # displacing_phase == "gas"
            displacing_viscosity_grid = fluid_properties.gas_viscosity_grid
            displacing_rel_perm_grid = relative_permeabilities.krg

        if displaced_phase == "water":
            displaced_viscosity_grid = fluid_properties.water_viscosity_grid
            displaced_rel_perm_grid = relative_permeabilities.krw
        else:  # displaced_phase == "oil"
            displaced_viscosity_grid = fluid_properties.oil_viscosity_grid
            displaced_rel_perm_grid = relative_permeabilities.kro

        # Calculate average mobility for displacing phase
        displacing_mobility_grid = np.divide(
            displacing_rel_perm_grid,
            displacing_viscosity_grid,
            out=np.zeros_like(displacing_rel_perm_grid),
            where=displacing_viscosity_grid != 0,
        )
        avg_displacing_mobility = np.mean(displacing_mobility_grid)
        # Calculate average mobility for displaced phase
        displaced_mobility_grid = np.divide(
            displaced_rel_perm_grid,
            displaced_viscosity_grid,
            out=np.zeros_like(displaced_rel_perm_grid),
            where=displaced_viscosity_grid != 0,
        )
        avg_displaced_mobility = np.mean(displaced_mobility_grid)
        if avg_displaced_mobility == 0:
            return float("inf")

        # Calculate mobility ratio
        return float(avg_displacing_mobility / avg_displaced_mobility)

    mr = mobility_ratio

    def reservoir_volumetrics_history(
        self, from_step: int = 0, to_step: int = -1, interval: int = 1
    ) -> typing.Generator[typing.Tuple[int, ReservoirVolumetrics], None, None]:
        """
        Generator for reservoir volumetrics history over time.

        :param from_step: Starting time step index.
        :param to_step: Ending time step index.
        :param interval: Time step interval.
        :return: Generator yielding (step, `ReservoirVolumetrics`) tuples.
        """
        to_step = self._resolve_step(to_step)

        for t in range(from_step, to_step + 1, interval):
            yield (t, self.reservoir_volumetrics_analysis(t))

    def instantaneous_rates_history(
        self,
        from_step: int = 0,
        to_step: int = -1,
        interval: int = 1,
        rate_type: typing.Literal["production", "injection"] = "production",
        cells: typing.Union[Cells, CellFilter] = None,
    ) -> typing.Generator[typing.Tuple[int, InstantaneousRates], None, None]:
        """
        Generator for instantaneous rates history over time.

        :param from_step: Starting time step index.
        :param to_step: Ending time step index.
        :param interval: Time step interval.
        :param rate_type: Type of rates ('production' or 'injection').
        :return: Generator yielding (step, `InstantaneousRates`) tuples.
        """
        to_step = self._resolve_step(to_step)

        instantaneous_rate_method = (
            self.instantaneous_production_rates
            if rate_type == "production"
            else self.instantaneous_injection_rates
        )

        for t in range(from_step, to_step + 1, interval):
            yield (t, instantaneous_rate_method(t, cells=cells))

    def cumulative_production_history(
        self, from_step: int = 0, to_step: int = -1, interval: int = 1
    ) -> typing.Generator[typing.Tuple[int, CumulativeProduction], None, None]:
        """
        Generator for cumulative production history over time.

        :param from_step: Starting time step index.
        :param to_step: Ending time step index.
        :param interval: Time step interval.
        :return: Generator yielding (step, `CumulativeProduction`) tuples.
        """
        to_step = self._resolve_step(to_step)

        for t in range(from_step, to_step + 1, interval):
            yield (t, self.cumulative_production_analysis(t))

    def material_balance_history(
        self, from_step: int = 0, to_step: int = -1, interval: int = 1
    ) -> typing.Generator[typing.Tuple[int, MaterialBalance], None, None]:
        """
        Generator for material balance history over time.

        :param from_step: Starting time step index.
        :param to_step: Ending time step index.
        :param interval: Time step interval.
        :return: Generator yielding (step, `MaterialBalance`) tuples.
        """
        to_step = self._resolve_step(to_step)

        for t in range(from_step, to_step + 1, interval):
            yield (t, self.material_balance(t))

    mbal_history = material_balance_history

    def material_balance_error_history(
        self,
        from_step: typing.Optional[int] = None,
        to_step: int = -1,
        interval: int = 1,
    ) -> typing.Generator[typing.Tuple[int, MaterialBalanceError], None, None]:
        """
        Generator for cumulative material balance error over time.

        Each yielded value is the *cumulative* MBE from `from_step` up to step
        *t*, so you can see whether errors are drifting monotonically (systematic
        problem) or oscillating around zero (acceptable numerical noise).

        :param from_step: Starting time step; defaults to `self._min_step`.
        :param to_step: Ending time step; -1 for the last available.
        :param interval: Sampling interval.
        :return: Generator yielding `(step, MaterialBalanceError)` tuples.
        """
        if from_step is None:
            from_step = self._min_step
        to_step = self._resolve_step(to_step)

        for t in range(from_step, to_step + 1, interval):
            yield (t, self.material_balance_error(from_step=from_step, to_step=t))

    mbe_history = material_balance_error_history

    def sweep_efficiency_history(
        self,
        from_step: int = 0,
        to_step: int = -1,
        interval: int = 1,
        displacing_phase: typing.Literal["oil", "water", "gas"] = "water",
        delta_water_saturation_threshold: float = 0.02,
        delta_gas_saturation_threshold: float = 0.01,
        solvent_concentration_threshold: float = 0.01,
    ) -> typing.Generator[typing.Tuple[int, SweepEfficiencyAnalysis], None, None]:
        """
        Generator for sweep efficiency analysis history over time.

        :param from_step: Starting time step index.
        :param to_step: Ending time step index.
        :param interval: Time step interval.
        :return: Generator yielding (step, `SweepEfficiencyAnalysis`) tuples.
        """
        to_step = self._resolve_step(to_step)

        for t in range(from_step, to_step + 1, interval):
            yield (
                t,
                self.sweep_efficiency_analysis(
                    step=t,
                    displacing_phase=displacing_phase,
                    delta_water_saturation_threshold=delta_water_saturation_threshold,
                    delta_gas_saturation_threshold=delta_gas_saturation_threshold,
                    solvent_concentration_threshold=solvent_concentration_threshold,
                ),
            )

    def injection_front_history(
        self,
        from_step: int = 0,
        to_step: int = -1,
        interval: int = 1,
        phase: typing.Literal["water", "gas"] = "water",
        threshold: float = 0.02,
    ) -> typing.Generator[typing.Tuple[int, InjectionFrontAnalysis], None, None]:
        """
        Generator for injection-front analysis over a range of time steps.

        :param from_step: Starting time step (inclusive).
        :param to_step: Ending time step (inclusive); -1 for the last available.
        :param interval: Sampling interval.
        :param phase: Displacing phase - `"water"` or `"gas"`.
        :param threshold: Saturation-increase threshold for contact detection (fraction).
        :return: Generator yielding `(step, InjectionFrontAnalysis)` tuples.
        """
        to_step = self._resolve_step(to_step)
        for t in range(from_step, to_step + 1, interval):
            yield (
                t,
                self.injection_front_analysis(step=t, phase=phase, threshold=threshold),
            )

    def productivity_history(
        self,
        from_step: int = 0,
        to_step: int = -1,
        interval: int = 1,
        phase: typing.Literal["oil", "gas", "water"] = "oil",
    ) -> typing.Generator[typing.Tuple[int, ProductivityAnalysis], None, None]:
        """
        Generator for productivity analysis history over time.

        :param from_step: Starting time step index.
        :param to_step: Ending time step index.
        :param interval: Time step interval.
        :param phase: Phase to analyze ('oil', 'gas', 'water').
        :return: Generator yielding (step, `ProductivityAnalysis`) tuples.
        """
        to_step = self._resolve_step(to_step)

        for t in range(from_step, to_step + 1, interval):
            yield (t, self.productivity_analysis(t, phase=phase))

    def voidage_replacement_ratio_history(
        self,
        from_step: int = 0,
        to_step: int = -1,
        interval: int = 1,
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Generator for voidage replacement ratio (VRR) history over time.

        The VRR tracks how well injection is maintaining reservoir pressure by
        comparing injected volumes to produced volumes on a reservoir volume basis.

        :param from_step: Starting time step index (inclusive).
        :param to_step: Ending time step index (inclusive).
        :param interval: Time step interval for sampling.
        :return: Generator yielding tuples of (step, voidage_replacement_ratio).
        """
        to_step = self._resolve_step(to_step)

        for t in range(from_step, to_step + 1, interval):
            yield (t, self.voidage_replacement_ratio(t))

    vrr_history = voidage_replacement_ratio_history

    def mobility_ratio_history(
        self,
        displaced_phase: typing.Literal["oil", "water"] = "oil",
        displacing_phase: typing.Literal["oil", "water", "gas"] = "water",
        from_step: int = 0,
        to_step: int = -1,
        interval: int = 1,
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Generator for mobility ratio history over time.

        :param displaced_phase: The phase being displaced ('oil' or 'water').
        :param displacing_phase: The phase doing the displacing ('oil', 'water', or 'gas').
        :param from_step: Starting time step index.
        :param to_step: Ending time step index.
        :param interval: Time step interval.
        :return: Generator yielding (step, mobility_ratio) tuples.
        """
        to_step = self._resolve_step(to_step)

        for t in range(from_step, to_step + 1, interval):
            yield (
                t,
                self.mobility_ratio(
                    displaced_phase=displaced_phase,
                    displacing_phase=displacing_phase,
                    step=t,
                ),
            )

    mr_history = mobility_ratio_history
