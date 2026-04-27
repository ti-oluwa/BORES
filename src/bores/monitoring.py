"""Utility API for simulation monitoring"""

import itertools
import logging
import time
import typing
from dataclasses import dataclass, field

import numpy as np
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from tqdm import tqdm

from bores.config import Config
from bores.constants import c
from bores.datastructures import FormationVolumeFactors, Rates, SparseTensor
from bores.errors import ValidationError
from bores.models import ReservoirModel
from bores.simulate import Run, StepCallback, StepResult, run
from bores.states import ModelState
from bores.types import ThreeDimensions
from bores.utils import _close_iter
from bores.wells.base import InjectionWell, ProductionWell

__all__ = [
    "MonitorConfig",
    "RunStats",
    "StepDiagnostics",
    "monitor",
]

logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    """
    Configuration for the simulation monitor.

    Controls which display backends are active and how often they refresh.
    """

    use_rich: bool = True
    """
    Show a live Rich panel with solver diagnostics and physics summary.

    The panel updates in-place every `refresh_interval` accepted steps
    and is left in terminal history when the run ends.
    """

    use_tqdm: bool = False
    """
    Show a tqdm progress bar tracking simulation-time completion.

    The bar runs from 0 % to 100 % of total simulation time and displays
    a postfix with current step, average pressure, water saturation, and
    per-step wall time.
    """

    refresh_interval: int = 1
    """
    How often (in accepted steps) to refresh the Rich live display.

    Set higher for very fast simulations to avoid terminal flicker.
    Has no effect when `use_rich=False`.
    """

    show_wells: bool = True
    """
    Include a per-well performance section in the Rich panel showing
    injection and production rates by well name.
    """

    show_mbe: bool = True
    """
    Include a material balance error section in the Rich panel showing
    total and phase absolute and relative material balance errors.
    """

    color_theme: str = "dark"
    """
    Color theme for the Rich panel.

    `"dark"` uses a charcoal background with amber accents.
    `"light"` uses off-white with navy accents.
    """

    def __post_init__(self) -> None:
        if self.use_rich and self.use_tqdm:
            raise ValidationError(
                "`use_tqdm=True` and `use_rich=True` is not allowed. Choose one"
            )


@dataclass
class WellRateDiagnostics:
    """Surface-condition rates for a single well."""

    name: str
    """Well name."""
    well_type: str
    """'injection' or 'production'."""
    oil_rate: float
    """Oil rate in STB/day."""
    water_rate: float
    """Water rate in STB/day."""
    gas_rate: float
    """Gas rate in SCF/day."""
    average_oil_bhp: float = 0.0
    """Average oil bottom hole pressure across all perforations, in psi."""
    average_water_bhp: float = 0.0
    """Average water bottom hole pressure across all perforations, in psi."""
    average_gas_bhp: float = 0.0
    """Average gas bottom hole pressure across all perforations, in psi."""


@dataclass
class StepDiagnostics:
    """
    Cheap scalar snapshot captured after each accepted time step.

    Only aggregates (mean, min, max) are stored.
    All pressure values are in psi; saturation values are dimensionless
    fractions in [0, 1]; rates are in surface conditions (STB/day or SCF/day).
    """

    step: int
    """Accepted step index (1-based)."""

    elapsed_time: float
    """Total simulation time elapsed at the end of this step (s)."""

    step_size: float
    """Time step size Δt used for this step (s)."""

    wall_time_ms: float
    """Wall-clock time consumed by this step, measured by the monitor (ms)."""

    average_pressure: float
    """Mean reservoir pressure over all interior cells (psi)."""

    minimum_pressure: float
    """Minimum cell pressure (psi)."""

    maximum_pressure: float
    """Maximum cell pressure (psi)."""

    maximum_pressure_change: float
    """Largest pressure change in any cell during this step (psi)."""

    average_water_saturation: float
    """Mean water saturation over all interior cells."""

    average_oil_saturation: float
    """Mean oil saturation over all interior cells."""

    average_gas_saturation: float
    """Mean gas saturation over all interior cells."""

    minimum_water_saturation: float
    """Minimum water saturation."""

    maximum_water_saturation: float
    """Maximum water saturation."""

    minimum_oil_saturation: float
    """Minimum oil saturation."""

    maximum_oil_saturation: float
    """Maximum oil saturation."""

    minimum_gas_saturation: float
    """Minimum gas saturation."""

    maximum_gas_saturation: float
    """Maximum gas saturation."""

    maximum_water_saturation_change: float
    """Maximum water saturation change during this step."""

    maximum_oil_saturation_change: float
    """Maximum oil saturation change during this step."""

    maximum_gas_saturation_change: float
    """Maximum gas saturation change during this step."""

    newton_iterations: int
    """
    Newton-Raphson iterations used by the saturation solver.

    Set to `-1` for schemes that do not use Newton iteration (e.g. explicit
    saturation in IMPES, or the fully explicit scheme).
    """

    maximum_cfl: float
    """
    Maximum CFL number encountered during the explicit transport step.

    Set to `-1.0` for fully implicit schemes where CFL is not tracked.
    """

    oil_injection_rate: float
    """Total oil injection rate in STB/day."""

    water_injection_rate: float
    """Total water injection rate in STB/day."""

    gas_injection_rate: float
    """Total gas injection rate in SCF/day."""

    oil_production_rate: float
    """Total oil production rate in STB/day."""

    water_production_rate: float
    """Total water production rate in STB/day."""

    gas_production_rate: float
    """Total gas production rate in SCF/day."""

    well_diagnostics: typing.List[WellRateDiagnostics] = field(default_factory=list)
    """Per-well rate diagnostics."""

    absolute_oil_mbe: float = 0.0
    """Absolute oil material balance error (res ft³). See `MaterialBalanceErrors.absolute_oil_mbe`."""

    absolute_water_mbe: float = 0.0
    """Absolute water material balance error (res ft³). See `MaterialBalanceErrors.absolute_water_mbe`."""

    absolute_gas_mbe: float = 0.0
    """Absolute gas material balance error (res ft³), including dissolved gas. See `MaterialBalanceErrors.absolute_gas_mbe`."""

    total_absolute_mbe: float = 0.0
    """Sum of absolute phase MBEs (res ft³). See `MaterialBalanceErrors.total_absolute_mbe`."""

    relative_oil_mbe: float = 0.0
    """Relative oil MBE as a dimensionless fraction. Multiply by 100 for percent."""

    relative_water_mbe: float = 0.0
    """Relative water MBE as a dimensionless fraction. Multiply by 100 for percent."""

    relative_gas_mbe: float = 0.0
    """Relative gas MBE as a dimensionless fraction. Multiply by 100 for percent."""

    total_relative_mbe: float = 0.0
    """Total relative MBE as a dimensionless fraction. Multiply by 100 for percent."""

    converged: bool = True
    """`True` if the step was accepted without any convergence fallback."""


@dataclass
class RunStats:
    """
    Accumulates per-step diagnostics and produces end-of-run summaries.

    Updated in-place by `monitored_run` after every accepted step.
    Callers may read any field or property at any point during iteration;
    the object is safe to inspect after the loop as well.

    Running totals (`_total_wall_time_ms`, `_total_newton_iterations`, `_newton_count`)
    are maintained so that derived properties (`average_step_wall_ms`,
    `average_newton_iterations`) compute in O(1) without iterating `steps`.
    """

    steps: typing.List[StepDiagnostics] = field(default_factory=list)
    """All per-step diagnostic snapshots in chronological order."""

    total_wall_time: float = 0.0
    """Cumulative wall-clock time across all accepted steps (seconds)."""

    rejected_steps: int = 0
    """
    Number of proposed time steps that were rejected by the timer.

    Note: `monitor(...)` cannot directly observe rejections from the
    public `run()` generator; this counter is incremented only when
    the caller explicitly signals a rejection via `record_rejection()`.
    """

    accepted_steps: int = 0
    """Number of time steps accepted and recorded."""

    _total_wall_time_ms: float = field(default=0.0, repr=False)
    _total_newton_iterations: int = field(default=0, repr=False)
    _newton_count: int = field(default=0, repr=False)

    def record(self, diagnostics: StepDiagnostics) -> None:
        """
        Append a step diagnostic and update all running accumulators.

        :param diagnostics: Diagnostic snapshot for the just-completed step.
        """
        self.steps.append(diagnostics)
        self.accepted_steps += 1
        self._total_wall_time_ms += diagnostics.wall_time_ms
        self.total_wall_time += diagnostics.wall_time_ms / 1000.0
        if diagnostics.newton_iterations >= 0:
            self._total_newton_iterations += diagnostics.newton_iterations
            self._newton_count += 1

    def record_rejection(self) -> None:
        """Increment the rejected-step counter by one."""
        self.rejected_steps += 1

    @property
    def average_step_wall_ms(self) -> float:
        """Mean per-step wall time across all accepted steps (ms)."""
        return (
            self._total_wall_time_ms / self.accepted_steps
            if self.accepted_steps
            else 0.0
        )

    @property
    def average_newton_iterations(self) -> float:
        """
        Mean Newton iterations per step, counting only steps where Newton was used.

        Returns `0.0` if the scheme never uses Newton iteration.
        """
        return (
            self._total_newton_iterations / self._newton_count
            if self._newton_count
            else 0.0
        )

    @property
    def step_wall_times_ms(self) -> typing.List[float]:
        """Per-step wall times in chronological order (ms)."""
        return [d.wall_time_ms for d in self.steps]

    def get_percentile_wall_time_ms(self, percentage: float) -> float:
        """
        Return the *percentage*-th percentile of per-step wall times using nearest-rank.

        Returns `0.0` if no steps have been recorded yet.

        :param percentage: Percentile in the range [0, 100].
        :return: Wall time at the requested percentile (ms).
        """
        if not self.steps:
            return 0.0
        arr = sorted(self.step_wall_times_ms)
        idx = min(int(len(arr) * percentage / 100), len(arr) - 1)
        return arr[idx]

    def summary_table(self) -> Table:
        """
        Return a Rich Table summarizing the simulation run.

        Includes step counts, wall-time statistics, final physics state,
        and Newton iteration averages when applicable.

        :return: Rich Table with run summary.
        """
        if not self.steps:
            table = Table(show_header=False, box=box.SIMPLE)
            table.add_row("No steps recorded.")
            return table

        last = self.steps[-1]
        wt = self.step_wall_times_ms

        table = Table(
            title="[bold]SIMULATION RUN SUMMARY[/bold]",
            box=box.DOUBLE_EDGE,
            show_header=True,
            header_style="bold cyan",
            title_style="bold white on blue",
            expand=False,
        )

        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="white", justify="right")

        # Step counts
        table.add_row("Accepted steps", f"{self.accepted_steps:,}")
        table.add_row("Rejected steps", f"{self.rejected_steps:,}")

        # Wall time statistics
        table.add_section()
        table.add_row("Total wall time", f"{self.total_wall_time:.3f} s")
        table.add_row("Avg step time", f"{self.average_step_wall_ms:.3f} ms")
        table.add_row("p50 step time", f"{self.get_percentile_wall_time_ms(50):.3f} ms")
        table.add_row("p95 step time", f"{self.get_percentile_wall_time_ms(95):.3f} ms")
        table.add_row("Max step time", f"{max(wt):.3f} ms")

        # Simulation state
        table.add_section()
        table.add_row("Simulation time", f"{last.elapsed_time:.4f} s")
        table.add_row("Final avg pressure", f"{last.average_pressure:.2f} psi")
        table.add_row(
            "Final Sw / So / Sg",
            f"{last.average_water_saturation:.4f} / "
            f"{last.average_oil_saturation:.4f} / "
            f"{last.average_gas_saturation:.4f}",
        )

        # Newton iterations (if applicable)
        if self._newton_count:
            table.add_section()
            table.add_row(
                "Avg Newton iterations", f"{self.average_newton_iterations:.2f}"
            )

        # Material balance errors
        table.add_section()
        table.add_row(
            "Total absolute MBE",
            f"{last.total_absolute_mbe:.3e} lbm",
        )
        table.add_row(
            "Total relative MBE",
            f"{last.total_relative_mbe * 100:.2e}%",
        )
        table.add_row(
            "Oil MBE (rel)",
            f"{last.relative_oil_mbe * 100:.2e}%",
        )
        table.add_row(
            "Water MBE (rel)",
            f"{last.relative_water_mbe * 100:.2e}%",
        )
        table.add_row(
            "Gas MBE (rel)",
            f"{last.relative_gas_mbe * 100:.2e}%",
        )
        return table

    def summary(self) -> str:
        """
        Return a plain-text end-of-run summary suitable for logging.

        Includes step counts, wall-time statistics, final physics state,
        and Newton iteration averages when applicable.

        :return: Multi-line summary string.
        """
        if not self.steps:
            return "No steps recorded."

        last = self.steps[-1]
        wt = self.step_wall_times_ms
        lines = [
            "",
            "═" * 62,
            "  SIMULATION RUN SUMMARY",
            "═" * 62,
            f"  Accepted steps    : {self.accepted_steps}",
            f"  Rejected steps    : {self.rejected_steps}",
            f"  Total wall time   : {self.total_wall_time:.3f} s",
            f"  Avg step time     : {self.average_step_wall_ms:.3f} ms",
            f"  p50 step time     : {self.get_percentile_wall_time_ms(50):.3f} ms",
            f"  p95 step time     : {self.get_percentile_wall_time_ms(95):.3f} ms",
            f"  Max step time     : {max(wt):.3f} ms",
            "",
            f"  Simulation time   : {last.elapsed_time:.4f} s",
            f"  Final avg pressure: {last.average_pressure:.2f} psi",
            f"  Final Sw/So/Sg    : {last.average_water_saturation:.4f} / {last.average_oil_saturation:.4f} / {last.average_gas_saturation:.4f}",
        ]
        if self._newton_count:
            lines.append(
                f"  Average Newton iterations  : {self.average_newton_iterations:.2f}"
            )
        lines.append("")
        lines.append(f"  Total absolute MBE        : {last.total_absolute_mbe:.3e} lbm")
        lines.append(
            f"  Total relative MBE        : {last.total_relative_mbe * 100:.2e}%"
        )
        lines.append(
            f"  Oil MBE (relative)        : {last.relative_oil_mbe * 100:.2e}%"
        )
        lines.append(
            f"  Water MBE (relative)      : {last.relative_water_mbe * 100:.2e}%"
        )
        lines.append(
            f"  Gas MBE (relative)        : {last.relative_gas_mbe * 100:.2e}%"
        )
        lines.append("═" * 62)
        return "\n".join(lines)


def _surface_rate_for_well_cells(
    cell_keys: typing.List[typing.Tuple[int, ...]],
    rate_tensor: SparseTensor,
    fvf_tensor: SparseTensor,
    phase: str,
) -> float:
    """
    Convert reservoir-condition volumetric rates to surface-condition rates
    for a specific set of cell keys (a single well's perforations).

    :param cell_keys: List of grid index tuples for this well's perforations.
    :param rate_tensor: Sparse tensor of reservoir-condition flow rates.
    :param fvf_tensor: Sparse tensor of formation volume factors.
    :param phase: Phase name ('oil', 'water', or 'gas').
    :return: Surface-condition rate in STB/day (oil, water) or SCF/day (gas).
    """
    ft3_to_bbl = c.CUBIC_FEET_TO_BARRELS
    total = 0.0
    for key in cell_keys:
        # Only sum cells that have a non-zero rate entry
        if key not in rate_tensor:
            continue
        reservoir_rate = abs(float(rate_tensor[key]))
        fvf = float(fvf_tensor[key]) if key in fvf_tensor else 0.0
        if fvf > 0:
            if phase in ("oil", "water"):
                reservoir_rate *= ft3_to_bbl
            total += reservoir_rate / fvf
    return total


def _expand_well_cells(
    well: typing.Union[InjectionWell, ProductionWell],
) -> typing.List[typing.Tuple[int, ...]]:
    """
    Return all perforation cell indices for a well as a flat list of tuples.

    Handles wells of any dimensionality by expanding all perforating intervals.
    """
    cells: typing.List[typing.Tuple[int, ...]] = []
    for start, end in well.perforating_intervals:
        # Normalise and produce all integer steps along the dominant axis
        lo = tuple(min(s, e) for s, e in zip(start, end))
        hi = tuple(max(s, e) for s, e in zip(start, end))
        ranges = [range(l, h + 1) for l, h in zip(lo, hi)]
        cells.extend(tuple(int(i) for i in idx) for idx in itertools.product(*ranges))
    return cells


def _convert_to_total_surface_rate(
    rates: Rates[float, ThreeDimensions],
    fvfs: FormationVolumeFactors[float, ThreeDimensions],
    phase: str,
) -> float:
    """
    Convert reservoir-condition volumetric rate to surface-condition rate.

    :param rates: Rates object containing sparse tensors for each phase.
    :param fvfs: Formation volume factors object containing sparse tensors for each phase.
    :param phase: Phase name ('oil', 'water', or 'gas').
    :return: Surface-condition rate in STB/day (oil, water) or SCF/day (gas).
    """
    rate_tensor: typing.Optional[SparseTensor] = getattr(rates, phase, None)
    fvf_tensor: typing.Optional[SparseTensor] = getattr(fvfs, phase, None)
    if rate_tensor is None or fvf_tensor is None:
        return 0.0

    total_surface_rate = 0.0
    ft3_to_bbl = c.CUBIC_FEET_TO_BARRELS
    for key in rate_tensor:
        reservoir_rate = abs(float(rate_tensor[key]))
        fvf = float(fvf_tensor[key])
        if fvf > 0:
            if phase in ("oil", "water"):
                reservoir_rate *= ft3_to_bbl

            surface_rate = reservoir_rate / fvf
            total_surface_rate += surface_rate

    return total_surface_rate


def _compute_average_bhp_for_cells(
    cell_keys: typing.List[typing.Tuple[int, ...]],
    bhp_tensor: SparseTensor,
) -> float:
    """
    Compute the average BHP across a well's perforation cells.

    Only cells with an explicitly stored BHP entry are included.
    Returns 0.0 if no cells have a stored entry.

    :param cell_keys: List of grid index tuples for this well's perforations.
    :param bhp_tensor: Sparse tensor of bottom hole pressures (psi).
    :return: Mean BHP in psi, or 0.0 if no entries found.
    """
    values = [float(bhp_tensor[key]) for key in cell_keys if key in bhp_tensor]
    return sum(values) / len(values) if values else 0.0


def build_well_diagnostics(
    state: ModelState[ThreeDimensions],
) -> typing.List[WellRateDiagnostics]:
    """
    Build per-well rate diagnostics from a model state snapshot.

    Iterates over all injection and production wells, sums the surface-condition
    rates across all perforation cells for each phase, and returns a list of
    `WellRateDiagnostics` — one entry per well.

    :param state: Model state snapshot.
    :return: List of per-well diagnostics.
    """
    results: typing.List[WellRateDiagnostics] = []
    wells = state.wells

    for well in wells.injection_wells:
        cells = _expand_well_cells(well)
        oil_rate = _surface_rate_for_well_cells(
            cells,
            state.rates.injection_rates.oil,
            state.rates.injection_fvfs.oil,
            "oil",
        )
        water_rate = _surface_rate_for_well_cells(
            cells,
            state.rates.injection_rates.water,
            state.rates.injection_fvfs.water,
            "water",
        )
        gas_rate = _surface_rate_for_well_cells(
            cells,
            state.rates.injection_rates.gas,
            state.rates.injection_fvfs.gas,
            "gas",
        )
        average_oil_bhp = _compute_average_bhp_for_cells(
            cells, state.rates.injection_bhps.oil
        )
        average_water_bhp = _compute_average_bhp_for_cells(
            cells, state.rates.injection_bhps.water
        )
        average_gas_bhp = _compute_average_bhp_for_cells(
            cells, state.rates.injection_bhps.gas
        )
        results.append(
            WellRateDiagnostics(
                name=well.name,
                well_type="injection",
                oil_rate=oil_rate,
                water_rate=water_rate,
                gas_rate=gas_rate,
                average_oil_bhp=average_oil_bhp,
                average_water_bhp=average_water_bhp,
                average_gas_bhp=average_gas_bhp,
            )
        )

    for well in wells.production_wells:
        cells = _expand_well_cells(well)
        oil_rate = _surface_rate_for_well_cells(
            cells,
            state.rates.production_rates.oil,
            state.rates.production_fvfs.oil,
            "oil",
        )
        water_rate = _surface_rate_for_well_cells(
            cells,
            state.rates.production_rates.water,
            state.rates.production_fvfs.water,
            "water",
        )
        gas_rate = _surface_rate_for_well_cells(
            cells,
            state.rates.production_rates.gas,
            state.rates.production_fvfs.gas,
            "gas",
        )
        average_oil_bhp = _compute_average_bhp_for_cells(
            cells, state.rates.production_bhps.oil
        )
        average_water_bhp = _compute_average_bhp_for_cells(
            cells, state.rates.production_bhps.water
        )
        average_gas_bhp = _compute_average_bhp_for_cells(
            cells, state.rates.production_bhps.gas
        )
        results.append(
            WellRateDiagnostics(
                name=well.name,
                well_type="production",
                oil_rate=oil_rate,
                water_rate=water_rate,
                gas_rate=gas_rate,
                average_oil_bhp=average_oil_bhp,
                average_water_bhp=average_water_bhp,
                average_gas_bhp=average_gas_bhp,
            )
        )

    return results


def build_step_diagnostics(
    state: ModelState[ThreeDimensions],
    wall_time_ms: float,
    step_result: typing.Optional[StepResult[ThreeDimensions]],
) -> StepDiagnostics:
    """
    Build a `StepDiagnostics` instance from a `ModelState`.

    No per-cell arrays are retained; only grid-level aggregates are kept.
    Rates are converted from reservoir conditions to surface conditions using FVFs.

    :param state: Model state snapshot yielded by the simulation generator.
    :param wall_time_ms: Wall-clock time consumed by this step (ms).
    :param step_result: Optional StepResult with pressure and saturation changes,
        newton_iterations, and maximum_cfl_encountered from timer_context.
    :return: `StepDiagnostics` instance with all scalar fields populated.
    """
    fluid_properties = state.model.fluid_properties
    pressure = fluid_properties.pressure_grid
    water_saturation = fluid_properties.water_saturation_grid
    oil_saturation = fluid_properties.oil_saturation_grid
    gas_saturation = fluid_properties.gas_saturation_grid

    # Aggregate totals
    oil_injection_rate = _convert_to_total_surface_rate(
        rates=state.rates.injection_rates,
        fvfs=state.rates.injection_fvfs,
        phase="oil",
    )
    water_injection_rate = _convert_to_total_surface_rate(
        rates=state.rates.injection_rates,
        fvfs=state.rates.injection_fvfs,
        phase="water",
    )
    gas_injection_rate = _convert_to_total_surface_rate(
        rates=state.rates.injection_rates,
        fvfs=state.rates.injection_fvfs,
        phase="gas",
    )
    oil_production_rate = _convert_to_total_surface_rate(
        rates=state.rates.production_rates,
        fvfs=state.rates.production_fvfs,
        phase="oil",
    )
    water_production_rate = _convert_to_total_surface_rate(
        rates=state.rates.production_rates,
        fvfs=state.rates.production_fvfs,
        phase="water",
    )
    gas_production_rate = _convert_to_total_surface_rate(
        rates=state.rates.production_rates,
        fvfs=state.rates.production_fvfs,
        phase="gas",
    )

    # Compute MBE
    mbe = None if step_result is None else step_result.material_balance_errors
    if mbe is not None:
        mbe_kwargs = {
            "absolute_oil_mbe": float(mbe.absolute_oil_mbe or 0.0),
            "absolute_water_mbe": float(mbe.absolute_water_mbe or 0.0),
            "absolute_gas_mbe": float(mbe.absolute_gas_mbe or 0.0),
            "total_absolute_mbe": float(mbe.total_absolute_mbe or 0.0),
            "relative_oil_mbe": float(mbe.relative_oil_mbe or 0.0),
            "relative_water_mbe": float(mbe.relative_water_mbe or 0.0),
            "relative_gas_mbe": float(mbe.relative_gas_mbe or 0.0),
            "total_relative_mbe": float(mbe.total_relative_mbe or 0.0),
        }
    else:
        mbe_kwargs = {}

    well_diagnostics = build_well_diagnostics(state)
    timer_context = {} if step_result is None else step_result.timer_context
    return StepDiagnostics(
        step=state.step,
        elapsed_time=state.time,
        step_size=state.step_size,
        wall_time_ms=wall_time_ms,
        average_pressure=float(np.mean(pressure)),
        minimum_pressure=float(np.min(pressure)),
        maximum_pressure=float(np.max(pressure)),
        maximum_pressure_change=float(
            (step_result.maximum_pressure_change if step_result else None) or 0.0
        ),
        average_water_saturation=float(np.mean(water_saturation)),
        average_oil_saturation=float(np.mean(oil_saturation)),
        average_gas_saturation=float(np.mean(gas_saturation)),
        minimum_water_saturation=float(np.min(water_saturation)),
        maximum_water_saturation=float(np.max(water_saturation)),
        minimum_oil_saturation=float(np.min(oil_saturation)),
        maximum_oil_saturation=float(np.max(oil_saturation)),
        minimum_gas_saturation=float(np.min(gas_saturation)),
        maximum_gas_saturation=float(np.max(gas_saturation)),
        maximum_water_saturation_change=float(
            (step_result.maximum_water_saturation_change if step_result else None)
            or 0.0
        ),
        maximum_oil_saturation_change=float(
            (step_result.maximum_oil_saturation_change if step_result else None) or 0.0
        ),
        maximum_gas_saturation_change=float(
            (step_result.maximum_gas_saturation_change if step_result else None) or 0.0
        ),
        newton_iterations=int(timer_context.get("newton_iterations", -1) or -1),
        maximum_cfl=float(timer_context.get("maximum_cfl_encountered", -1.0) or -1.0),
        oil_injection_rate=oil_injection_rate,
        water_injection_rate=water_injection_rate,
        gas_injection_rate=gas_injection_rate,
        oil_production_rate=oil_production_rate,
        water_production_rate=water_production_rate,
        gas_production_rate=gas_production_rate,
        well_diagnostics=well_diagnostics,
        **mbe_kwargs,
        converged=True,
    )


def _format_time(v: float, /) -> str:
    """
    Format a simulation time value into a human-readable string.

    Chooses the most readable unit (days, hours, minutes, or seconds)
    based on magnitude.

    :param v: Time in seconds.
    :return: Formatted string with unit suffix.
    """
    if v >= 86400:
        return f"{v / 86400:.2f} d"
    if v >= 3600:
        return f"{v / 3600:.2f} h"
    if v >= 60:
        return f"{v / 60:.2f} min"
    return f"{v:.2f} s"


def build_rich_panel(
    diagnostics: StepDiagnostics,
    stats: RunStats,
    total_simulation_time: float,
    show_wells: bool,
    show_mbe: bool,
    theme: str,
) -> Panel:
    """
    Build the Rich renderable for the live monitor panel.

    Constructs a compact two-column `Panel` with progress, physics, solver
    diagnostics, performance metrics, and optional per-well rates.

    :param diagnostics: Diagnostic snapshot for the most recently accepted step.
    :param stats: Accumulated run statistics used for performance metrics.
    :param total_simulation_time: Total simulation time (s), used to compute progress %.
    :param show_wells: Whether to append the per-well rate summary table.
    :param show_mbe: Whether to append the material balance error in the solver table.
    :param theme: `"dark"` or `"light"`. Controls Rich color styles.
    :return: A `rich.panel.Panel` ready to pass to `Live.update()`.
    """
    if theme == "light":
        hdr = "bold navy_blue"
        val = "dark_blue"
        good = "green4"
        warn = "dark_orange"
        dim = "grey50"
        title_style = "bold navy_blue on grey93"
        inj_color = "steel_blue1"
        prod_color = "dark_orange"
    else:
        hdr = "bold bright_yellow"
        val = "bright_white"
        good = "bright_green"
        warn = "yellow"
        dim = "grey62"
        title_style = "bold bright_yellow on grey11"
        inj_color = "cyan1"
        prod_color = "orange1"

    percentage = (
        min(diagnostics.elapsed_time / total_simulation_time * 100, 100.0)
        if total_simulation_time
        else 0.0
    )

    # Progress bar
    bar_width = 44
    filled = int(bar_width * percentage / 100)
    bar = f"[{good}]{'█' * filled}[/{good}][{dim}]{'░' * (bar_width - filled)}[/{dim}]"
    progress_line = Text.assemble(
        ("Progress ", dim),
        Text.from_markup(bar),
        (f" {percentage:.1f}%", good if percentage >= 99.9 else val),
        (f"  step {diagnostics.step:,}", dim),
        (f"  {_format_time(diagnostics.elapsed_time)}", val),
        (f" / {_format_time(total_simulation_time)}", dim),
    )

    # Left column: physics
    physics_table = Table(
        box=box.HEAVY_HEAD,
        show_header=True,
        expand=True,
        header_style=hdr,
        border_style=dim,
        padding=(0, 1),
    )
    physics_table.add_column("Quantity", style=dim, no_wrap=True)
    physics_table.add_column("Avg", style=val, no_wrap=True, justify="right")
    physics_table.add_column("Min", style=dim, no_wrap=True, justify="right")
    physics_table.add_column("Max", style=warn, no_wrap=True, justify="right")

    physics_table.add_row(
        "P (psi)",
        f"{diagnostics.average_pressure:,.2f}",
        f"{diagnostics.minimum_pressure:,.2f}",
        f"{diagnostics.maximum_pressure:,.2f}",
    )
    physics_table.add_row(
        "Sw",
        f"{diagnostics.average_water_saturation:.4f}",
        f"{diagnostics.minimum_water_saturation:.4f}",
        f"{diagnostics.maximum_water_saturation:.4f}",
    )
    physics_table.add_row(
        "So",
        f"{diagnostics.average_oil_saturation:.4f}",
        f"{diagnostics.minimum_oil_saturation:.4f}",
        f"{diagnostics.maximum_oil_saturation:.4f}",
    )
    physics_table.add_row(
        "Sg",
        f"{diagnostics.average_gas_saturation:.4f}",
        f"{diagnostics.minimum_gas_saturation:.4f}",
        f"{diagnostics.maximum_gas_saturation:.4f}",
    )

    # Right column: solver + performance (stacked)
    solver_table = Table(
        box=box.HEAVY,
        show_header=False,
        expand=True,
        border_style=dim,
        padding=(0, 1),
    )
    solver_table.add_column("Metric", style=dim, no_wrap=True)
    solver_table.add_column("Value", style=val, no_wrap=True, justify="right")

    # Step timing (compact: Δt and wall time on one row each)
    solver_table.add_row("Δt", _format_time(diagnostics.step_size))
    solver_table.add_row("Wall", f"{diagnostics.wall_time_ms:.2f} ms")
    solver_table.add_row("Avg/step", f"{stats.average_step_wall_ms:.2f} ms")

    if diagnostics.newton_iterations >= 0:
        ni_style = good if diagnostics.newton_iterations <= 5 else warn
        solver_table.add_row(
            "Newton",
            f"[{ni_style}]{diagnostics.newton_iterations}[/{ni_style}]",
        )
    if diagnostics.maximum_cfl >= 0:
        cfl_style = good if diagnostics.maximum_cfl <= 0.7 else warn
        solver_table.add_row(
            "CFL max",
            f"[{cfl_style}]{diagnostics.maximum_cfl:.3f}[/{cfl_style}]",
        )

    solver_table.add_row("ΔP max", f"{diagnostics.maximum_pressure_change:,.2f} psi")
    solver_table.add_row(
        "ΔS (Sw/So/Sg)",
        f"{diagnostics.maximum_water_saturation_change:.2e} / "
        f"{diagnostics.maximum_oil_saturation_change:.2e} / "
        f"{diagnostics.maximum_gas_saturation_change:.2e}",
    )

    # Performance extras
    if stats.accepted_steps >= 2:
        solver_table.add_row(
            "p95 wall", f"{stats.get_percentile_wall_time_ms(95):.2f} ms"
        )
        if stats._newton_count:
            solver_table.add_row("Avg Newton", f"{stats.average_newton_iterations:.2f}")

    reject_style = warn if stats.rejected_steps else dim
    solver_table.add_row(
        "Rejected",
        f"[{reject_style}]{stats.rejected_steps}[/{reject_style}]"
        f"[{dim}] / [{dim}][{good}]{stats.accepted_steps}[/{good}]",
    )
    solver_table.add_row("Run time", f"{stats.total_wall_time:.2f} s")

    # Material Balance Errors
    if show_mbe and diagnostics.step > 1:  # Only show after first step
        solver_table.add_row("", "")  # Separator

        # Color code MBE based on magnitude
        total_mbe = diagnostics.total_relative_mbe
        if abs(total_mbe) < 0.0001:  # < 0.01 %
            mbe_style = good
        elif abs(total_mbe) < 0.001:  # < 0.1 %
            mbe_style = val
        else:
            mbe_style = warn

        solver_table.add_row(
            "MBE (total)", f"[{mbe_style}]{total_mbe * 100:.2e}%[/{mbe_style}]"
        )
        solver_table.add_row("MBE (oil)", f"{diagnostics.relative_oil_mbe * 100:.2e}%")
        solver_table.add_row(
            "MBE (water)", f"{diagnostics.relative_water_mbe * 100:.2e}%"
        )
        solver_table.add_row("MBE (gas)", f"{diagnostics.relative_gas_mbe * 100:.2e}%")

    # Two-column grid: physics left, solver right
    cols_table = Table.grid(expand=True, padding=(0, 2))
    cols_table.add_column(ratio=3)
    cols_table.add_column(ratio=2)
    cols_table.add_row(
        Group(Text("  Reservoir State", style=hdr), physics_table),
        Group(Text("  Step / Solver", style=hdr), solver_table),
    )

    renderables: typing.List[typing.Any] = [progress_line, Text(""), cols_table]

    if show_wells and diagnostics.well_diagnostics:
        well_table = Table(
            box=box.HEAVY_HEAD,
            show_header=True,
            expand=True,
            header_style=hdr,
            border_style=dim,
            padding=(0, 1),
        )
        well_table.add_column("Well", style=val, no_wrap=True)
        well_table.add_column("Type", style=dim, no_wrap=True)
        well_table.add_column("Oil Rate", no_wrap=True, justify="right")
        well_table.add_column("Water Rate", no_wrap=True, justify="right")
        well_table.add_column("Gas Rate", no_wrap=True, justify="right")
        well_table.add_column("Oil BHP", no_wrap=True, justify="right")
        well_table.add_column("Water BHP", no_wrap=True, justify="right")
        well_table.add_column("Gas BHP", no_wrap=True, justify="right")

        for wd in diagnostics.well_diagnostics:
            is_inj = wd.well_type == "injection"
            type_markup = (
                f"[{inj_color}]INJ[/{inj_color}]"
                if is_inj
                else f"[{prod_color}]PROD[/{prod_color}]"
            )
            rate_style = inj_color if is_inj else prod_color

            # Format rate values
            oil_rate_str = (
                f"[{rate_style}]{wd.oil_rate:.3e}[/{rate_style}] STB/d"
                if wd.oil_rate > 0.0
                else f"[{dim}]—[/{dim}]"
            )
            water_rate_str = (
                f"[{rate_style}]{wd.water_rate:.3e}[/{rate_style}] STB/d"
                if wd.water_rate > 0.0
                else f"[{dim}]—[/{dim}]"
            )
            gas_rate_str = (
                f"[{rate_style}]{wd.gas_rate:.3e}[/{rate_style}] SCF/d"
                if wd.gas_rate > 0.0
                else f"[{dim}]—[/{dim}]"
            )

            # Format BHP values
            oil_bhp_str = (
                f"[{rate_style}]{wd.average_oil_bhp:,.2f} psi[/{rate_style}]"
                if wd.average_oil_bhp > 0.0
                else f"[{dim}]—[/{dim}]"
            )
            water_bhp_str = (
                f"[{rate_style}]{wd.average_water_bhp:,.2f} psi[/{rate_style}]"
                if wd.average_water_bhp > 0.0
                else f"[{dim}]—[/{dim}]"
            )
            gas_bhp_str = (
                f"[{rate_style}]{wd.average_gas_bhp:,.2f} psi[/{rate_style}]"
                if wd.average_gas_bhp > 0.0
                else f"[{dim}]—[/{dim}]"
            )

            well_table.add_row(
                wd.name,
                type_markup,
                oil_rate_str,
                water_rate_str,
                gas_rate_str,
                oil_bhp_str,
                water_bhp_str,
                gas_bhp_str,
            )

        renderables += [Text(""), Text("  Well Rates & BHPs", style=hdr), well_table]

    return Panel(
        Group(*renderables),
        title="[bold]⬛ BORES — Simulation Monitor[/bold]",
        title_align="left",
        style=title_style,
        border_style=hdr,
        padding=(1, 2),
        highlight=True,
        expand=False,
    )


@typing.overload
def monitor(
    input: typing.Union[
        ReservoirModel[ThreeDimensions],
        Run,
        typing.Iterable[ModelState[ThreeDimensions]],
    ],
    config: typing.Optional[Config] = ...,
    *,
    monitor: typing.Optional[MonitorConfig] = ...,
    on_step_rejected: typing.Optional[StepCallback] = ...,
    on_step_accepted: typing.Optional[StepCallback] = ...,
    return_stats: typing.Literal[False] = ...,
) -> typing.Generator[
    ModelState[ThreeDimensions],
    None,
    None,
]: ...


@typing.overload
def monitor(
    input: typing.Union[
        ReservoirModel[ThreeDimensions],
        Run,
        typing.Iterable[ModelState[ThreeDimensions]],
    ],
    config: typing.Optional[Config] = ...,
    *,
    monitor: typing.Optional[MonitorConfig] = ...,
    on_step_rejected: typing.Optional[StepCallback] = ...,
    on_step_accepted: typing.Optional[StepCallback] = ...,
    return_stats: typing.Literal[True],
) -> typing.Generator[
    typing.Tuple[ModelState[ThreeDimensions], RunStats],
    None,
    None,
]: ...


def monitor(
    input: typing.Union[
        ReservoirModel[ThreeDimensions],
        Run,
        typing.Iterable[ModelState[ThreeDimensions]],
    ],
    config: typing.Optional[Config] = None,
    *,
    monitor: typing.Optional[MonitorConfig] = None,
    on_step_rejected: typing.Optional[StepCallback] = None,
    on_step_accepted: typing.Optional[StepCallback] = None,
    return_stats: bool = False,
) -> typing.Generator[
    typing.Union[
        ModelState[ThreeDimensions], typing.Tuple[ModelState[ThreeDimensions], RunStats]
    ],
    None,
    None,
]:
    """
    Wraps `bores.run(...)` with live monitoring and statistics collection.

    Yields `(ModelState, RunStats)` pairs at the same cadence as
    `bores.run` (i.e. every `output_frequency` accepted steps).
    `RunStats` is the same object throughout the run and accumulates
    data in-place, so it remains valid for inspection after the loop.

    The Rich live panel and/or tqdm bar run concurrently with the simulation
    loop and are torn down cleanly on both normal completion and exceptions.
    A summary table is always emitted to the logger at INFO level when
    the generator is exhausted or closed.

    :param input: A `ReservoirModel`, `Run`, or an iterable that yields `ModelState`s - identical to
        the first argument of `bores.run`.
    :param config: Simulation configuration. Required when `input` is a
        `ReservoirModel`; optional when `input` is a `Run` (overrides
        the config stored on the `Run` when provided).
    :param monitor: `MonitorConfig` controlling display options. Defaults
        to `MonitorConfig()` (rich live panel enabled by default).
    :param on_step_rejected: Optional callback to be invoked whenever a
        proposed time step is rejected by the timer. The callback receives the
        same arguments as the `on_step_rejected` callback of `bores.run`.
    :param on_step_accepted: Optional callback to be invoked whenever a time
        step is accepted by the timer. The callback receives the same arguments
        as the `on_step_accepted` callback of `bores.run(...)`.
    :param return_stats: If `True`, yield a tuple of `(ModelState, RunStats)` at each output step.
        If `False`, yield only the `ModelState` (default). Note that `RunStats` is updated in-place,
        so the same object is yielded at every step and can be inspected after the loop to access the full history of diagnostics and summaries.
    :yields: Tuple of `(state, stats)` - the model state and the live `RunStats`, if `return_stats` is True.
        Else, the model state only is returned.
        accumulator after each accepted output step.
    :raises ValidationError: If `input` is a `ReservoirModel` and `config` is not provided.
    """
    if monitor is None:
        monitor = MonitorConfig()

    if not monitor.use_rich and not monitor.use_tqdm:
        logger.warning(
            "Monitor config has both `use_rich` and `use_tqdm` set to False; no live progress display will be shown."
        )

    is_generic_input = not isinstance(input, (ReservoirModel, Run))
    if isinstance(input, Run):
        config = config if config is not None else input.config
    else:
        if config is None and not is_generic_input:
            raise ValueError(
                "Must provide `config` when `input` is a `ReservoirModel`."
            )
        config = typing.cast(Config, config)

    # Suppress logging from `run(...)`; monitor handles all output and stats
    config = config.with_updates(log_interval=0)
    total_simulation_time: float = float(config.timer.simulation_time)
    stats = RunStats()
    _step_result: typing.Optional[StepResult] = None

    def _on_step_rejected(step_result: StepResult) -> None:
        nonlocal _step_result, on_step_rejected
        stats.record_rejection()
        _step_result = step_result
        if on_step_rejected is not None:
            on_step_rejected(step_result)

    def _on_step_accepted(step_result: StepResult) -> None:
        nonlocal _step_result, on_step_accepted
        _step_result = step_result
        if on_step_accepted is not None:
            on_step_accepted(step_result)

    tqdm_bar: typing.Optional[tqdm] = None  # type: ignore[type-arg]
    if monitor.use_tqdm:
        tqdm_bar = tqdm(
            total=100,
            desc="Simulation",
            unit="%",
            bar_format=(
                "{desc}: {percentage:3.1f}%|{bar:40}| "
                "{n:.2f}/{total:.2f}% "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            ),
            colour="green",
            dynamic_ncols=True,
        )

    # All logging from 'bores.*' is redirected through the same `Console` that
    # owns the `Live` panel. This prevents the default stderr handler from
    # writing lines that force Rich to re-render and print a new panel frame
    # on every log record emitted inside simulation.
    live: typing.Optional[Live] = None
    rich_console: typing.Optional[Console] = None
    bores_logger = logging.getLogger("bores")
    original_propagate = bores_logger.propagate  # Save original propagate setting
    original_handlers: typing.List[logging.Handler] = []
    rich_log_handler: typing.Optional[RichHandler] = None

    if monitor.use_rich:
        rich_console = Console()
        live = Live(
            console=rich_console,
            refresh_per_second=4,
            transient=False,
        )
        live.__enter__()

        # Swap out all existing handlers on the bores root logger and replace
        # them with a single `RichHandler` that writes through the `Live` console.
        original_handlers = bores_logger.handlers[:]
        rich_log_handler = RichHandler(
            console=rich_console,
            show_time=False,
            show_path=False,
            markup=False,
            rich_tracebacks=False,
        )
        rich_log_handler.setLevel(logging.DEBUG)
        bores_logger.handlers = [rich_log_handler]
        bores_logger.propagate = False  # Prevent bubbling to root

    step_start = time.perf_counter()
    last_percentage = 0.0
    last_diagnostics: typing.Optional[StepDiagnostics] = None

    simulation = None
    error = None
    try:
        if is_generic_input:
            simulation = input
        elif isinstance(input, Run):
            simulation = input(
                config=config,
                on_step_rejected=_on_step_rejected,
                on_step_accepted=_on_step_accepted,
            )
        else:
            simulation = run(
                input,  # type: ignore[arg-type]
                config,
                on_step_rejected=_on_step_rejected,
                on_step_accepted=_on_step_accepted,
            )

        for state in simulation:  # type: ignore[arg-type]
            step_end = time.perf_counter()
            wall_ms = (step_end - step_start) * 1000.0
            step_start = step_end
            diagnostics = build_step_diagnostics(
                state=state,
                wall_time_ms=wall_ms,
                step_result=_step_result,
            )
            stats.record(diagnostics)
            last_diagnostics = diagnostics

            if (
                live is not None
                and stats.accepted_steps % monitor.refresh_interval == 0
            ):
                live.update(
                    build_rich_panel(
                        diagnostics=diagnostics,
                        stats=stats,
                        total_simulation_time=total_simulation_time,
                        show_wells=monitor.show_wells,
                        show_mbe=monitor.show_mbe,
                        theme=monitor.color_theme,
                    )
                )

            if tqdm_bar is not None:
                new_percentage = min(
                    diagnostics.elapsed_time / total_simulation_time * 100, 100.0
                )
                change = float(new_percentage - last_percentage)
                if change > 0:
                    tqdm_bar.update(change)
                last_percentage = new_percentage
                tqdm_bar.set_postfix_str(
                    f"step={diagnostics.step} "
                    f"P={diagnostics.average_pressure:.2f} psi "
                    f"Sw={diagnostics.average_water_saturation:.4f} "
                    f"wall={wall_ms:.2f} ms"
                )

            if return_stats:
                yield state, stats
            else:
                yield state

    except Exception as exc:
        error = exc

    finally:
        if live is not None:
            if error is None and last_diagnostics is not None:
                # Render the final state into the panel before stopping so
                # the completed view stays in terminal scroll-back history.
                live.update(
                    build_rich_panel(
                        diagnostics=last_diagnostics,
                        stats=stats,
                        total_simulation_time=total_simulation_time,
                        show_wells=monitor.show_wells,
                        show_mbe=monitor.show_mbe,
                        theme=monitor.color_theme,
                    )
                )
            live.__exit__(None, None, None)

            # Restore the bores logger to whatever handlers it had before we
            # started, so logging behaves normally after the run completes.
            bores_logger.handlers = original_handlers
            bores_logger.propagate = original_propagate

        if error is None and tqdm_bar is not None:
            tqdm_bar.update(float(100.0 - last_percentage))  # Ensure bar reaches 100 %
            tqdm_bar.close()

        # Print the summary table
        if rich_console is not None:
            rich_console.print()
            rich_console.print(stats.summary_table())
        else:
            logger.info(stats.summary())

        if simulation is not None:
            _close_iter(simulation)

        if error is not None:
            raise error
