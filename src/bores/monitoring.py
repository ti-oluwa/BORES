"""Utility API for simulation monitoring"""

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
from bores.datastructures import Rates, FormationVolumeFactors, SparseTensor
from bores.models import ReservoirModel
from bores.simulate import Run, StepCallback, StepResult, run
from bores.states import ModelState
from bores.types import ThreeDimensions

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

    extended_every: int = 10
    """
    Every this many accepted steps, include extended stats (p95 wall time,
    average Newton iterations) in the Rich performance line.

    Set to `0` to disable extended stats entirely.
    """

    show_wells: bool = True
    """
    Include a compact well performance section in the Rich panel showing
    per-phase injection and production rates.
    """

    color_theme: str = "dark"
    """
    Color theme for the Rich panel.

    `"dark"` uses a charcoal background with amber accents.
    `"light"` uses off-white with navy accents.
    """


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

    maximum_saturation_change: float
    """Largest per-cell saturation change (any phase) during this step."""

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
    """Oil injection rate in STB/day."""

    water_injection_rate: float
    """Water injection rate in STB/day."""

    gas_injection_rate: float
    """Gas injection rate in SCF/day."""

    oil_production_rate: float
    """Oil production rate in STB/day."""

    water_production_rate: float
    """Water production rate in STB/day."""

    gas_production_rate: float
    """Gas production rate in SCF/day."""

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
        lines.append("═" * 62)
        return "\n".join(lines)


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


def _build_step_diagnostics(
    state: ModelState[ThreeDimensions],
    wall_time_ms: float,
    timer_kwargs: typing.Dict[str, typing.Any],
) -> StepDiagnostics:
    """
    Build a `StepDiagnostics` instance from a `ModelState`.

    No per-cell arrays are retained; only grid-level aggregates are kept.
    Rates are converted from reservoir conditions to surface conditions using FVFs.

    :param state: Model state snapshot yielded by the simulation generator.
    :param wall_time_ms: Wall-clock time consumed by this step (ms).
    :param timer_kwargs: Dict from `state.timer_state`; may contain
        `maximum_pressure_change`, `maximum_saturation_change`,
        `newton_iterations`, and `maximum_cfl_encountered`.
    :return: `StepDiagnostics` instance with all scalar fields populated.
    """
    fluid_properties = state.model.fluid_properties
    pressure = fluid_properties.pressure_grid
    water_saturation = fluid_properties.water_saturation_grid
    oil_saturation = fluid_properties.oil_saturation_grid
    gas_saturation = fluid_properties.gas_saturation_grid

    # Convert rates to surface conditions using FVFs
    oil_injection_rate = _convert_to_total_surface_rate(
        rates=state.injection_rates,
        fvfs=state.injection_formation_volume_factors,
        phase="oil",
    )
    water_injection_rate = _convert_to_total_surface_rate(
        rates=state.injection_rates,
        fvfs=state.injection_formation_volume_factors,
        phase="water",
    )
    gas_injection_rate = _convert_to_total_surface_rate(
        rates=state.injection_rates,
        fvfs=state.injection_formation_volume_factors,
        phase="gas",
    )

    oil_production_rate = _convert_to_total_surface_rate(
        rates=state.production_rates,
        fvfs=state.production_formation_volume_factors,
        phase="oil",
    )
    water_production_rate = _convert_to_total_surface_rate(
        rates=state.production_rates,
        fvfs=state.production_formation_volume_factors,
        phase="water",
    )
    gas_production_rate = _convert_to_total_surface_rate(
        rates=state.production_rates,
        fvfs=state.production_formation_volume_factors,
        phase="gas",
    )

    return StepDiagnostics(
        step=state.step,
        elapsed_time=state.time,
        step_size=state.step_size,
        wall_time_ms=wall_time_ms,
        average_pressure=float(np.mean(pressure)),
        minimum_pressure=float(np.min(pressure)),
        maximum_pressure=float(np.max(pressure)),
        maximum_pressure_change=float(
            timer_kwargs.get("maximum_pressure_change", 0.0) or 0.0
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
        maximum_saturation_change=float(
            timer_kwargs.get("maximum_saturation_change", 0.0) or 0.0
        ),
        newton_iterations=int(timer_kwargs.get("newton_iterations", -1) or -1),
        maximum_cfl=float(timer_kwargs.get("maximum_cfl_encountered", -1.0) or -1.0),
        oil_injection_rate=oil_injection_rate,
        water_injection_rate=water_injection_rate,
        gas_injection_rate=gas_injection_rate,
        oil_production_rate=oil_production_rate,
        water_production_rate=water_production_rate,
        gas_production_rate=gas_production_rate,
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


def _build_rich_panel(
    diagnostics: StepDiagnostics,
    stats: RunStats,
    total_simulation_time: float,
    extended: bool,
    show_wells: bool,
    theme: str,
) -> Panel:
    """
    Build the Rich renderable for the live monitor panel.

    Constructs a `Panel` containing a progress bar, time/step info table,
    physics table (pressure and saturation aggregates), solver diagnostics
    table, performance table, and optional well summary.

    :param diagnostics: Diagnostic snapshot for the most recently accepted step.
    :param stats: Accumulated run statistics used for performance metrics.
    :param total_simulation_time: Total simulation time (s), used to compute progress %.
    :param extended: Whether to include p95 wall time and avg Newton count
        in the performance metrics.
    :param show_wells: Whether to append the well rate summary table.
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
    else:
        hdr = "bold bright_yellow"
        val = "bright_white"
        good = "bright_green"
        warn = "yellow"
        dim = "grey62"
        title_style = "bold bright_yellow on grey11"

    percentage = (
        min(diagnostics.elapsed_time / total_simulation_time * 100, 100.0)
        if total_simulation_time
        else 0.0
    )

    # Progress bar
    bar_width = 38
    filled = int(bar_width * percentage / 100)
    bar = f"[{good}]{'█' * filled}[/{good}][{dim}]{'░' * (bar_width - filled)}[/{dim}]"
    progress_line = Text.assemble(
        ("Progress  ", dim),
        Text.from_markup(bar),
        (f"  {percentage:.1f}%", good if percentage >= 99.9 else val),
    )

    # Time & Step Info Table
    time_info_table = Table(
        box=box.SIMPLE,
        show_header=False,
        expand=True,
        border_style=dim,
        padding=(0, 0),
    )
    time_info_table.add_column("Label", style=dim, no_wrap=True)
    time_info_table.add_column("Value", style=val, no_wrap=True, justify="right")

    time_info_table.add_row("Step", f"{diagnostics.step:,}")
    time_info_table.add_row(
        "Elapsed Simulation Time", _format_time(diagnostics.elapsed_time)
    )
    time_info_table.add_row(
        "Current Step Size (Δt)", _format_time(diagnostics.step_size)
    )
    time_info_table.add_row("Step Wall Time", f"{diagnostics.wall_time_ms:.3f} ms")

    # Physics table
    physics_table = Table(
        box=box.SIMPLE,
        show_header=True,
        expand=True,
        header_style=hdr,
        border_style=dim,
    )
    physics_table.add_column("Quantity", style=dim, no_wrap=True, min_width=16)
    physics_table.add_column("Average", style=val, no_wrap=True, min_width=12)
    physics_table.add_column("Min.", style=dim, no_wrap=True, min_width=12)
    physics_table.add_column("Max.", style=warn, no_wrap=True, min_width=12)

    physics_table.add_row(
        "Pressure (psi)",
        f"{diagnostics.average_pressure:,.2f}",
        f"{diagnostics.minimum_pressure:,.2f}",
        f"{diagnostics.maximum_pressure:,.2f}",
    )
    physics_table.add_row(
        "Water Saturation (Sw)",
        f"{diagnostics.average_water_saturation:.5f}",
        f"{diagnostics.minimum_water_saturation:.5f}",
        f"{diagnostics.maximum_water_saturation:.5f}",
    )
    physics_table.add_row(
        "Oil Saturation (So)",
        f"{diagnostics.average_oil_saturation:.5f}",
        f"{diagnostics.minimum_oil_saturation:.5f}",
        f"{diagnostics.maximum_oil_saturation:.5f}",
    )
    physics_table.add_row(
        "Gas Saturation (Sg)",
        f"{diagnostics.average_gas_saturation:.5f}",
        f"{diagnostics.minimum_gas_saturation:.5f}",
        f"{diagnostics.maximum_gas_saturation:.5f}",
    )

    # Solver Diagnostics Table
    solver_table = Table(
        box=box.SIMPLE,
        show_header=False,
        expand=True,
        border_style=dim,
        padding=(0, 0),
    )
    solver_table.add_column("Metric", style=dim, no_wrap=True)
    solver_table.add_column("Value", style=val, no_wrap=True, justify="right")

    if diagnostics.newton_iterations >= 0:
        ni_style = good if diagnostics.newton_iterations <= 5 else warn
        solver_table.add_row(
            "Newton Iterations",
            f"[{ni_style}]{diagnostics.newton_iterations}[/{ni_style}]",
        )
    else:
        solver_table.add_row("Newton Iterations", "[dim]N/A[/dim]")

    if diagnostics.maximum_cfl >= 0:
        cfl_style = good if diagnostics.maximum_cfl <= 0.7 else warn
        solver_table.add_row(
            "Maximum CFL", f"[{cfl_style}]{diagnostics.maximum_cfl:.4f}[/{cfl_style}]"
        )
    else:
        solver_table.add_row("Maximum CFL", "[dim]N/A[/dim]")

    solver_table.add_row(
        "Pressure Change (ΔP)", f"{diagnostics.maximum_pressure_change:,.2f} psi"
    )
    solver_table.add_row(
        "Saturation Change (ΔS)", f"{diagnostics.maximum_saturation_change:.2e}"
    )

    # Performance Metrics Table
    performance_table = Table(
        box=box.SIMPLE,
        show_header=False,
        expand=True,
        border_style=dim,
        padding=(0, 0),
    )
    performance_table.add_column("Metric", style=dim, no_wrap=True)
    performance_table.add_column("Value", style=val, no_wrap=True, justify="right")

    performance_table.add_row(
        "Avg. Time Per Step", f"{stats.average_step_wall_ms:.2f} ms"
    )
    performance_table.add_row("Total Run Time", f"{stats.total_wall_time:.4f} s")
    performance_table.add_row(
        "Accepted Steps", f"[{good}]{stats.accepted_steps}[/{good}]"
    )

    reject_style = warn if stats.rejected_steps else dim
    performance_table.add_row(
        "Rejected Steps", f"[{reject_style}]{stats.rejected_steps}[/{reject_style}]"
    )

    if extended and stats.accepted_steps >= 2:
        performance_table.add_row(
            "p95 Step Time", f"{stats.get_percentile_wall_time_ms(95):.3f} ms"
        )
        if stats._newton_count:
            performance_table.add_row(
                "Avg. Newton Iterations", f"{stats.average_newton_iterations:.2f}"
            )

    # Wells table
    well_section: typing.Optional[Table] = None
    if show_wells:
        well_section = Table(
            box=box.SIMPLE,
            show_header=True,
            expand=True,
            header_style=hdr,
            border_style=dim,
        )
        well_section.add_column("Phase", style=dim, no_wrap=True)
        well_section.add_column(
            "Injection Rate", style=good, no_wrap=True, justify="right"
        )
        well_section.add_column(
            "Production Rate", style=warn, no_wrap=True, justify="right"
        )

        well_section.add_row(
            "Oil",
            f"{diagnostics.oil_injection_rate:.6f} STB/day",
            f"{diagnostics.oil_production_rate:.6f} STB/day",
        )
        well_section.add_row(
            "Water",
            f"{diagnostics.water_injection_rate:.6f} STB/day",
            f"{diagnostics.water_production_rate:.6f} STB/day",
        )
        well_section.add_row(
            "Gas",
            f"{diagnostics.gas_injection_rate:.6f} SCF/day",
            f"{diagnostics.gas_production_rate:.6f} SCF/day",
        )

    # Assemble sections with headers
    renderables: typing.List[typing.Any] = [
        progress_line,
        Text(""),
        Text("Time & Step", style=hdr),
        time_info_table,
        Text(""),
        Text("Reservoir Physics", style=hdr),
        physics_table,
        Text(""),
        Text("Solver Diagnostics", style=hdr),
        solver_table,
        Text(""),
        Text("Performance", style=hdr),
        performance_table,
    ]

    if well_section is not None:
        renderables += [
            Text(""),
            Text("Well Rates", style=hdr),
            well_section,
        ]

    return Panel(
        Group(*renderables),
        title="[bold]⬛ BORES - Simulation Monitor[/bold]",
        title_align="left",
        style=title_style,
        border_style=hdr,
        padding=(1, 2),
        highlight=True,
        expand=False
    )


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
) -> typing.Generator[typing.Tuple[ModelState[ThreeDimensions], RunStats], None, None]:
    """
    Wraps `bores.run(...)` with optional live monitoring and statistics collection.

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
        as the `on_step_accepted` callback of `bores.run`.
    :yields: Tuple of `(state, stats)` - the model state and the live `RunStats`
        accumulator after each accepted output step.
    :raises ValueError: If `input` is a `ReservoirModel` and `config`
        is not provided.
    """
    if monitor is None:
        monitor = MonitorConfig()

    if not monitor.use_rich and not monitor.use_tqdm:
        logger.warning(
            "Monitor config has both `use_rich` and `use_tqdm` set to False; no live progress display will be shown."
        )

    is_generic_input = not isinstance(input, (ReservoirModel, Run))
    if isinstance(input, Run):
        config = config or input.config
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
    _timer_kwargs: typing.Dict[str, typing.Any] = {}

    def _on_step_rejected(
        step_result: StepResult, step_size: float, elapsed_time: float
    ) -> None:
        nonlocal _timer_kwargs, on_step_rejected
        stats.record_rejection()
        _timer_kwargs.clear()
        _timer_kwargs.update(step_result.timer_kwargs)
        if on_step_rejected is not None:
            on_step_rejected(step_result, step_size, elapsed_time)

    def _on_step_accepted(
        step_result: StepResult, step_size: float, elapsed_time: float
    ) -> None:
        nonlocal _timer_kwargs, on_step_accepted
        _timer_kwargs.clear()
        _timer_kwargs.update(step_result.timer_kwargs)
        if on_step_accepted is not None:
            on_step_accepted(step_result, step_size, elapsed_time)

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
    _rich_console: typing.Optional[Console] = None
    _bores_logger = logging.getLogger("bores")
    _original_propagate = _bores_logger.propagate  # Save original propagate setting
    _original_handlers: typing.List[logging.Handler] = []
    _rich_log_handler: typing.Optional[RichHandler] = None

    if monitor.use_rich:
        _rich_console = Console()
        live = Live(
            console=_rich_console,
            refresh_per_second=4,
            transient=False,
        )
        live.__enter__()

        # Swap out all existing handlers on the bores root logger and replace
        # them with a single `RichHandler` that writes through the `Live` console.
        # Rich's `Live` context knows how to interleave log lines above the panel
        # without triggering a full re-render of the live display.
        _original_handlers = _bores_logger.handlers[:]
        _rich_log_handler = RichHandler(
            console=_rich_console,
            show_time=False,
            show_path=False,
            markup=False,
            rich_tracebacks=False,
        )
        _rich_log_handler.setLevel(logging.DEBUG)
        _bores_logger.handlers = [_rich_log_handler]
        _bores_logger.propagate = False  # Prevent bubbling to root

    step_start = time.perf_counter()
    last_percentage = 0.0
    last_diagnostics: typing.Optional[StepDiagnostics] = None

    try:
        if is_generic_input:
            simulation = input
        elif isinstance(input, Run):
            simulation = run(  # type: ignore
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
            diagnostics = _build_step_diagnostics(state, wall_ms, _timer_kwargs)
            stats.record(diagnostics)
            last_diagnostics = diagnostics
            extended = (
                monitor.extended_every > 0
                and stats.accepted_steps % monitor.extended_every == 0
            )

            if (
                live is not None
                and stats.accepted_steps % monitor.refresh_interval == 0
            ):
                live.update(
                    _build_rich_panel(
                        diagnostics=diagnostics,
                        stats=stats,
                        total_simulation_time=total_simulation_time,
                        extended=extended,
                        show_wells=monitor.show_wells,
                        theme=monitor.color_theme,
                    )
                )

            if tqdm_bar is not None:
                new_percentage = min(
                    diagnostics.elapsed_time / total_simulation_time * 100, 100.0
                )
                change = new_percentage - last_percentage
                if change > 0:
                    tqdm_bar.update(change)
                last_percentage = new_percentage
                tqdm_bar.set_postfix_str(
                    f"step={diagnostics.step} "
                    f"P={diagnostics.average_pressure:.2f} psi "
                    f"Sw={diagnostics.average_water_saturation:.4f} "
                    f"wall={wall_ms:.2f} ms"
                )

            yield state, stats

    finally:
        if live is not None:
            if last_diagnostics is not None:
                # Render the final state into the panel before stopping so
                # the completed view stays in terminal scroll-back history.
                live.update(
                    _build_rich_panel(
                        diagnostics=last_diagnostics,
                        stats=stats,
                        total_simulation_time=total_simulation_time,
                        extended=True,
                        show_wells=monitor.show_wells,
                        theme=monitor.color_theme,
                    )
                )
            live.__exit__(None, None, None)

            # Restore the bores logger to whatever handlers it had before we
            # started, so logging behaves normally after the run completes.
            _bores_logger.handlers = _original_handlers
            _bores_logger.propagate = _original_propagate

        if tqdm_bar is not None:
            tqdm_bar.update(100.0 - last_percentage)  # Ensure that bar reaches 100 %
            tqdm_bar.close()

        # Print the summary table
        if _rich_console is not None:
            _rich_console.print()
            _rich_console.print(stats.summary_table())
        else:
            logger.info(stats.summary())
