"""Simulation timestep controller with smart and adaptive time stepping."""

import datetime
import logging
import math
import typing
from collections import deque

import attrs
from typing_extensions import Self, TypedDict

from bores.constants import c
from bores.errors import TimingError, ValidationError
from bores.timestep.targets import (
    PeriodicTarget,
    TimeTargets,
    TimeTargetSource,
    build_time_targets,
)
from bores.types import Boolean, Number, UnitConversionTable, UnitSystem
from bores.units import get_conversion_factors
from bores.utils import scale

__all__ = ["Time", "Timer", "TimerState"]

logger = logging.getLogger(__name__)


def Time(
    *,
    milliseconds: float = 0,
    seconds: float = 0,
    minutes: float = 0,
    hours: float = 0,
    days: float = 0,
    weeks: float = 0,
    months: float = 0,
    years: float = 0,
    unit_system: UnitSystem | None = UnitSystem.FIELD,
    table: UnitConversionTable | None = None,
) -> Number:
    """
    Expresses time components as a total in the requested unit system's
    native time unit.

    :param milliseconds: Number of milliseconds.
    :param seconds: Number of seconds.
    :param minutes: Number of minutes.
    :param hours: Number of hours.
    :param days: Number of days.
    :param weeks: Number of weeks.
    :param months: Number of months.
    :param years: Number of years.
    :param unit_system: Target `UnitSystem` to express the total in (day for
        FIELD and METRIC, hour for LAB, second for SI). Defaults to FIELD, the
        same default as `Timer`, so `Time(days=5)` is 5 for a default `Timer`.
        `None` returns the total in seconds, unconverted.
    :param table: Optional custom conversion table; `None` uses the default.
    :return: Total time, in `unit_system`'s native time unit, or in seconds
        if `unit_system` is `None`.
    """
    days_per_year = c.DAYS_PER_YEAR
    total_days = (
        days
        + weeks * c.DAYS_PER_WEEK
        + years * days_per_year
        + months * (days_per_year / c.MONTHS_PER_YEAR)
    )
    total_seconds = (
        total_days * c.SECONDS_PER_DAY
        + hours * c.SECONDS_PER_HOUR
        + minutes * c.SECONDS_PER_MINUTE
        + seconds
        + milliseconds / 1000.0
    )
    if unit_system is None:
        return total_seconds

    factors = get_conversion_factors(UnitSystem.SI, unit_system, table=table)
    return scale(total_seconds, factors["time"])


class StepMetricsDict(TypedDict):
    """Dictionary representation of step metrics."""

    step: int
    step_size: Number
    cfl: Number | None
    newton_iterations: int | None
    success: bool


class TimerState(TypedDict):
    """Complete state of a timer instance for serialization."""

    initial_step_size: Number
    maximum_step_size: Number
    minimum_step_size: Number
    simulation_time: Number
    unit_system: str
    maximum_cfl: Number
    cfl_safety_margin: Number
    ramp_up_factor: Number | None
    backoff_factor: Number
    aggressive_backoff_factor: Number
    maximum_steps: int | None
    maximum_absolute_oil_mbe: Number | None
    maximum_absolute_water_mbe: Number | None
    maximum_absolute_gas_mbe: Number | None
    maximum_total_absolute_mbe: Number | None
    maximum_relative_oil_mbe: Number | None
    maximum_relative_water_mbe: Number | None
    maximum_relative_gas_mbe: Number | None
    maximum_total_relative_mbe: Number | None
    use_mbe_for_step_size: bool
    maximum_rejections: int
    maximum_growth_per_step: Number
    step_size_smoothing: Number
    growth_cooldown_steps: int
    failure_memory_window: int
    metrics_history_size: int
    target_utilization: Number
    target_stretch: Number
    enforce_constant_step_size: bool
    elapsed_time: Number
    step: int
    step_size: Number
    next_step_size: Number
    last_step_failed: bool
    rejection_count: int
    steps_since_last_failure: int
    recent_metrics: list[StepMetricsDict]
    failed_step_sizes: list[Number]
    last_successful_step_size: Number
    failure_ceiling: Number | None


TIMER_SETTING_NAMES: tuple[str, ...] = (
    "maximum_cfl",
    "cfl_safety_margin",
    "ramp_up_factor",
    "backoff_factor",
    "aggressive_backoff_factor",
    "maximum_steps",
    "maximum_absolute_oil_mbe",
    "maximum_absolute_water_mbe",
    "maximum_absolute_gas_mbe",
    "maximum_total_absolute_mbe",
    "maximum_relative_oil_mbe",
    "maximum_relative_water_mbe",
    "maximum_relative_gas_mbe",
    "maximum_total_relative_mbe",
    "use_mbe_for_step_size",
    "maximum_rejections",
    "maximum_growth_per_step",
    "step_size_smoothing",
    "growth_cooldown_steps",
    "failure_memory_window",
    "metrics_history_size",
    "target_utilization",
    "target_stretch",
    "enforce_constant_step_size",
)
"""Names of the `Timer` settings that `dump_state` and `load_state` save and restore as they are."""


@attrs.frozen(slots=True)
class StepMetrics:
    """Metrics for a single time step."""

    step: int
    """The 1-indexed step count this metric belongs to."""
    step_size: Number
    """The step size attempted, in the owning `Timer`'s `unit_system` native time unit."""
    cfl: Number | None = None
    """Maximum CFL number encountered during the step, if tracked."""
    newton_iterations: int | None = None
    """Number of Newton iterations taken by the transport solver, if tracked."""
    success: bool = True
    """Whether the step was accepted."""


def compute_utilization(actual: Number | None, limit: Number | None) -> Number | None:
    """Return actual/limit, or None if either is None/zero."""
    if actual is None or limit is None or limit <= 0.0:
        return None
    return actual / limit


def check_same_source(old: TimeTargetSource, new: TimeTargetSource) -> bool:
    """Whether two time target sources are the same object, or equal plain values."""
    if old is new:
        return True
    if isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)):
        return len(old) == len(new) and all(
            check_same_source(a, b) for a, b in zip(old, new, strict=True)
        )
    value_types = (int, float, str, datetime.datetime, TimeTargets, PeriodicTarget)
    return isinstance(old, value_types) and type(old) is type(new) and old == new


@attrs.mutable(kw_only=True, slots=True)
class Timer:
    """
    Simulation time manager for smart and adaptive time stepping.

    The timer proposes each step's size, lands exactly on every time in `time_targets` (event
    times, report times and dates), and adapts the size to how hard the solver found the
    previous steps. Every signal the solver reports (CFL, saturation change, pressure change,
    Newton iterations, material balance error) gives the largest step it would tolerate. The
    smallest of those wins, a step that is too large is cut straight away, and growth is
    limited so that the step size recovers gradually.
    """

    initial_step_size: Number
    """Initial time step size, in `unit_system`'s native time unit."""
    maximum_step_size: Number
    """Maximum allowable time step size, in `unit_system`'s native time unit."""
    minimum_step_size: Number
    """Minimum allowable time step size, in `unit_system`'s native time unit."""
    simulation_time: Number
    """Total simulation time, in `unit_system`'s native time unit."""
    unit_system: UnitSystem = UnitSystem.FIELD
    """
    Unit system the timer's time-valued fields are expressed in.

    Determines the native time unit every step-size and elapsed/simulation
    time field is read and written in: day for FIELD and METRIC, hour for
    LAB, second for SI. Use `convert(target)` to produce a rescaled copy.
    """
    time_targets: TimeTargets = attrs.field(factory=TimeTargets)
    """
    Times the timer must land on exactly, in `unit_system`'s native time unit. Fill it from the
    schedule and the deck's report times before the run starts. These are not saved by
    `dump_state`, since they are rebuilt from the schedule when a run is restarted.
    """
    maximum_cfl: Number = 0.9
    """Default CFL limit (max CFL number) for time step adjustments."""
    ramp_up_factor: Number | None = None
    """
    Largest growth factor per step while the solver is comfortable (every utilization below 50%
    and `growth_cooldown_steps` clean steps behind it). `None` uses `maximum_growth_per_step`.
    Set it above `maximum_growth_per_step` to recover faster through easy stretches of a run.
    """
    backoff_factor: Number = 0.5
    """
    Default factor by which to reduce time step size on failed steps.

    Only used when there is not enough information to intelligently determine a
    suitable backoff factor.
    """
    aggressive_backoff_factor: Number = 0.25
    """Factor by which to aggressively reduce time step size on failed steps."""
    maximum_steps: int | None = None
    """Maximum number of time steps to run for."""

    # Adaptive parameters
    growth_cooldown_steps: int = 5
    """Number of clean steps required after a rejection before growth beyond 10% per step is allowed."""
    maximum_growth_per_step: Number = 1.3
    """Maximum multiplicative growth allowed per step (e.g., 1.3 = 30% max growth)."""
    cfl_safety_margin: Number = 0.95
    """Fraction of the CFL limit the step size aims for."""
    target_utilization: Number = 0.8
    """
    Fraction of a saturation, pressure or material balance limit the step size aims for. A step
    that used less than this lets the step size grow, and one that used more than about 90% of
    the limit shrinks it.
    """
    step_size_smoothing: Number = 0.5
    """
    How much of an allowed increase in step size is taken at once, from 0 (exclusive) to 1.

    Only growth is smoothed. A reduction is always applied in full, so that the timer never
    delays a cut the solver needs. Set to 1.0 to take every allowed increase immediately.
    """
    target_stretch: Number = 0.1
    """
    How far above the proposed size a step may stretch to land on the next time target instead
    of splitting the distance into more steps (0.1 allows 10%). Never exceeds `maximum_step_size`.
    """
    enforce_constant_step_size: bool = False
    """
    Only matters when `initial_step_size`, `maximum_step_size` and `minimum_step_size` are equal.
    By default a constant step is shortened where needed to land on a time target, and a warning
    says so once. Set to `True` to keep every step exactly constant and let events fire at the first
    step boundary after their time. The final step still ends exactly at `simulation_time`.
    """
    metrics_history_size: int = 10
    """Number of recent steps to track for performance analysis."""
    failure_memory_window: int = 5
    """Number of recent failures to remember for adaptive behavior."""

    # MBE-based step control
    maximum_absolute_oil_mbe: Number | None = None
    """Absolute oil MBE threshold, in `unit_system`'s native volume unit. Reject step if exceeded."""
    maximum_absolute_water_mbe: Number | None = None
    """Absolute water MBE threshold, in `unit_system`'s native volume unit. Reject step if exceeded."""
    maximum_absolute_gas_mbe: Number | None = None
    """Absolute gas MBE threshold, in `unit_system`'s native volume unit. Reject step if exceeded."""
    maximum_total_absolute_mbe: Number | None = None
    """Total absolute MBE threshold, in `unit_system`'s native volume unit. Reject step if exceeded."""
    maximum_relative_oil_mbe: Number | None = None
    """Relative oil MBE threshold (fraction, e.g. 0.01 = 1%). Reject step if exceeded."""
    maximum_relative_water_mbe: Number | None = None
    """Relative water MBE threshold (fraction). Reject step if exceeded."""
    maximum_relative_gas_mbe: Number | None = None
    """Relative gas MBE threshold (fraction). Reject step if exceeded."""
    maximum_total_relative_mbe: Number | None = None
    """Total relative MBE threshold (fraction). Reject step if exceeded."""
    use_mbe_for_step_size: bool = False
    """
    If True, MBE values also limit and grow the step size on accepted steps, like the CFL,
    saturation and pressure signals. When False (default), MBE only triggers rejection when a
    configured limit is exceeded.
    """

    # Runtime state
    elapsed_time: Number = attrs.field(init=False, default=0.0)
    """Current simulation time, in `unit_system`'s native time unit (sum of all accepted steps)."""
    step_size: Number = attrs.field(init=False, default=0.0)
    """The time step size used for the most recently accepted step, in `unit_system`'s native time unit."""
    next_step_size: Number = attrs.field(init=False, default=0.0)
    """
    The step size the solver can currently sustain, in `unit_system`'s native time unit. The size
    actually proposed for the next step is this, shortened if needed to land on a time target.
    """
    step: int = attrs.field(init=False, default=0)
    """Number of accepted time steps completed so far."""
    last_step_failed: bool = attrs.field(init=False, default=False)
    """Whether the most recent step attempt was rejected."""
    maximum_rejections: int = 10
    """Maximum number of consecutive time step rejections allowed."""
    rejection_count: int = attrs.field(init=False, default=0)
    """Count of consecutive time step rejections."""
    steps_since_last_failure: int = attrs.field(init=False, default=0)
    """Number of successful steps since the last failure."""
    use_constant_step_size: bool = attrs.field(init=False, default=False)
    """Whether to use a constant time step size (init == max == min)."""
    last_successful_step_size: Number = attrs.field(init=False, default=0.0)
    """The step size used in the most recently accepted step, in `unit_system`'s native time unit."""
    failure_ceiling: Number | None = attrs.field(init=False, default=None)
    """
    The most recently rejected step size, in `unit_system`'s native time unit, while it still
    limits growth. It loosens with every accepted step and is dropped once it no longer matters.
    """

    # Performance tracking
    recent_metrics: deque[StepMetrics] = attrs.field(init=False)
    """Recent step performance metrics (bounded deque, newest at right)."""
    failed_step_sizes: deque[Number] = attrs.field(init=False)
    """Recent failed step sizes, in `unit_system`'s native time unit."""

    _cfl_window: deque[Number] = attrs.field(init=False)
    """Sliding window of the last N successful CFL values (N = metrics_history_size)."""
    _newton_window: deque[int] = attrs.field(init=False)
    """Sliding window of the last N successful Newton iteration counts."""
    _constant_step_warned: bool = attrs.field(init=False, default=False)
    _target_sources: tuple[TimeTargetSource, ...] | None = attrs.field(init=False, default=None)
    """What `update_time_targets` last built `time_targets` from, to spot when it has changed."""
    _target_start_date: datetime.datetime | None = attrs.field(init=False, default=None)

    def __attrs_post_init__(self) -> None:
        self.validate()
        self.next_step_size = self.initial_step_size
        self.step_size = self.initial_step_size
        self.last_successful_step_size = self.initial_step_size
        self.steps_since_last_failure = self.growth_cooldown_steps
        self.use_constant_step_size = (
            self.initial_step_size == self.maximum_step_size
            and self.initial_step_size == self.minimum_step_size
        )
        self.recent_metrics = deque(maxlen=self.metrics_history_size)
        self.failed_step_sizes = deque(maxlen=self.failure_memory_window)
        self._cfl_window = deque(maxlen=self.metrics_history_size)
        self._newton_window = deque(maxlen=self.metrics_history_size)

    def validate(self) -> None:
        """
        Checks that the configuration is usable.

        :raises ValidationError: If any setting is out of range or inconsistent with another.
        """
        problems: list[str] = []
        if not self.simulation_time > 0:
            problems.append("`simulation_time` must be positive")
        if not 0 < self.minimum_step_size <= self.initial_step_size <= self.maximum_step_size:
            problems.append(
                "step sizes must satisfy `0 < minimum_step_size <= initial_step_size <= "
                "maximum_step_size`"
            )
        if not 0 < self.backoff_factor < 1:
            problems.append("`backoff_factor` must be between 0 and 1, exclusive")
        if not 0 < self.aggressive_backoff_factor <= self.backoff_factor:
            problems.append(
                "`aggressive_backoff_factor` must be positive and no larger than `backoff_factor`"
            )
        if not 0 < self.step_size_smoothing <= 1:
            problems.append("`step_size_smoothing` must be above 0 and at most 1")
        if not self.maximum_growth_per_step > 1:
            problems.append("`maximum_growth_per_step` must be above 1")
        if self.ramp_up_factor is not None and not self.ramp_up_factor > 1:
            problems.append("`ramp_up_factor` must be above 1 when given")
        if not self.maximum_cfl > 0:
            problems.append("`maximum_cfl` must be positive")
        if not 0 < self.cfl_safety_margin <= 1:
            problems.append("`cfl_safety_margin` must be above 0 and at most 1")
        if not 0 < self.target_utilization < 0.9:
            problems.append("`target_utilization` must be above 0 and below 0.9")
        if self.target_stretch < 0:
            problems.append("`target_stretch` cannot be negative")
        if self.maximum_rejections < 1:
            problems.append("`maximum_rejections` must be at least 1")
        if self.growth_cooldown_steps < 0:
            problems.append("`growth_cooldown_steps` cannot be negative")
        if self.metrics_history_size < 1 or self.failure_memory_window < 1:
            problems.append(
                "`metrics_history_size` and `failure_memory_window` must be at least 1"
            )
        if problems:
            raise ValidationError(
                f"Invalid `{type(self).__name__}` configuration: " + "; ".join(problems) + "."
            )

    @property
    def time_tolerance(self) -> Number:
        """
        How close two times must be to count as the same, in `unit_system`'s native time unit.
        Scaled to the length of the run so it stays above floating point error.
        """
        return 1e-9 * max(self.simulation_time, 1.0)

    @property
    def next_step(self) -> int:
        """Returns the next time step count (1-indexed)."""
        return self.step + 1

    def done(self) -> bool:
        """True once the simulation has reached its end criteria."""
        if self.elapsed_time >= self.simulation_time - self.time_tolerance:
            return True
        return self.maximum_steps is not None and self.step >= self.maximum_steps

    @property
    def time_remaining(self) -> Number:
        """Remaining simulation time, in `unit_system`'s native time unit."""
        remaining = self.simulation_time - self.elapsed_time
        return 0.0 if remaining <= self.time_tolerance else remaining

    @property
    def is_last_step(self) -> bool:
        """True if the latest accepted step completed the simulation."""
        return self.done()

    @property
    def next_target_time(self) -> Number:
        """
        The next time the clock must land on. The earliest time target after the current time,
        or `simulation_time` if that comes first. A constant step size with
        `enforce_constant_step_size` ignores `time_targets`.
        """
        if self.use_constant_step_size and self.enforce_constant_step_size:
            return self.simulation_time

        target = self.time_targets.next_after(self.elapsed_time, tolerance=self.time_tolerance)
        if target is None or target > self.simulation_time:
            return self.simulation_time
        return target

    @property
    def average_cfl(self) -> Number | None:
        """Average CFL over the recent window, or None if no data."""
        if not self._cfl_window:
            return None
        return sum(self._cfl_window) / len(self._cfl_window)

    @property
    def average_newton_iterations(self) -> float | None:
        """Average Newton iterations over the recent window, or None if no data."""
        if not self._newton_window:
            return None
        return sum(self._newton_window) / len(self._newton_window)

    def compute_performance_factor(self) -> float:
        """
        Derive a multiplicative adjustment factor from recent solver trends.

        Returns a value in (0.5, 1.0]. Values below 1.0 indicate the solver
        is under stress and growth should be dampened.
        """
        factor = 1.0

        average_cfl = self.average_cfl
        if average_cfl is not None and len(self._cfl_window) >= 3:
            if average_cfl > 0.75 * self.maximum_cfl:
                factor *= 0.95
            cfl_list = list(self._cfl_window)
            if len(cfl_list) >= 4:
                trend = cfl_list[-1] - cfl_list[0]
                if trend > 0.15:
                    factor *= 0.90

        average_newton_iterations = self.average_newton_iterations
        if average_newton_iterations is not None and len(self._newton_window) >= 3:
            if average_newton_iterations > 8:
                factor *= 0.85
            recent_newton_iterations = list(self._newton_window)[-3:]
            if len(recent_newton_iterations) == 3 and all(
                iteration_count > 10 for iteration_count in recent_newton_iterations
            ):
                factor *= 0.75

        return max(factor, 0.5)

    def check_mbe_violations(
        self,
        absolute_oil_mbe: Number | None = None,
        absolute_water_mbe: Number | None = None,
        absolute_gas_mbe: Number | None = None,
        total_absolute_mbe: Number | None = None,
        relative_oil_mbe: Number | None = None,
        relative_water_mbe: Number | None = None,
        relative_gas_mbe: Number | None = None,
        total_relative_mbe: Number | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Check if any MBE limits are violated. Errors are compared by magnitude, so a
        negative error counts the same as a positive one.

        :return: `(any_violated, message_list)`.
        """
        mbe_checks: list[tuple[Number | None, Number | None, str]] = [
            (absolute_oil_mbe, self.maximum_absolute_oil_mbe, "abs oil MBE"),
            (absolute_water_mbe, self.maximum_absolute_water_mbe, "abs water MBE"),
            (absolute_gas_mbe, self.maximum_absolute_gas_mbe, "abs gas MBE"),
            (total_absolute_mbe, self.maximum_total_absolute_mbe, "total abs MBE"),
            (relative_oil_mbe, self.maximum_relative_oil_mbe, "rel oil MBE"),
            (relative_water_mbe, self.maximum_relative_water_mbe, "rel water MBE"),
            (relative_gas_mbe, self.maximum_relative_gas_mbe, "rel gas MBE"),
            (total_relative_mbe, self.maximum_total_relative_mbe, "total rel MBE"),
        ]
        messages = []
        for actual, limit, label in mbe_checks:
            if limit is None or actual is None:
                continue
            if abs(actual) > limit:
                messages.append(f"{label}: |{abs(actual):.3e}| > {limit:.3e}")
        return bool(messages), messages

    def propose_step_size(self) -> Number:
        """
        Propose the next time step size without updating any internal state.

        The proposal is the size the solver can currently sustain, shortened so the clock lands
        exactly on the next time target (or on `simulation_time`). A step may stretch up to
        `target_stretch` above the sustainable size to reach a target that is just out of reach,
        but never beyond `maximum_step_size`. When the target is between one and two steps
        away the distance is halved, so no tiny leftover step is left behind. A step that
        lands on a target may be smaller than `minimum_step_size`.

        :return: Proposed time step size, in `unit_system`'s native time unit.
        """
        constant = self.use_constant_step_size
        dt = self.initial_step_size if constant else self.next_step_size
        horizon = self.next_target_time
        distance = horizon - self.elapsed_time
        if distance <= 0.0:
            return dt

        stretch = 0.0 if constant else self.target_stretch
        if distance <= dt * (1.0 + stretch):
            proposed = distance
        elif distance < 2.0 * dt:
            proposed = distance / 2.0
        else:
            proposed = dt
        if proposed > self.maximum_step_size * (1.0 + 1e-9):
            proposed = distance / math.ceil(distance / self.maximum_step_size - 1e-9)

        if constant and horizon < self.simulation_time and proposed < dt * (1.0 - 1e-9):
            if not self._constant_step_warned:
                self._constant_step_warned = True
                logger.warning(
                    "The constant step size is being shortened to land on time targets such as "
                    "event times and report times. Set `enforce_constant_step_size=True` to keep "
                    "every step constant instead."
                )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Proposing time step of size %.6e for step %d at elapsed time %.4f "
                "(sustainable %.6e, next target %.6f)",
                proposed,
                self.next_step,
                self.elapsed_time,
                dt,
                horizon,
            )
        return proposed

    def reject_step(
        self,
        step_size: Number,
        *,
        maximum_cfl_encountered: Number | None = None,
        cfl_threshold: Number | None = None,
        newton_iterations: int | None = None,
        maximum_saturation_change: Number | None = None,
        maximum_allowed_saturation_change: Number | None = None,
        maximum_pressure_change: Number | None = None,
        maximum_allowed_pressure_change: Number | None = None,
        absolute_oil_mbe: Number | None = None,
        absolute_water_mbe: Number | None = None,
        absolute_gas_mbe: Number | None = None,
        total_absolute_mbe: Number | None = None,
        relative_oil_mbe: Number | None = None,
        relative_water_mbe: Number | None = None,
        relative_gas_mbe: Number | None = None,
        total_relative_mbe: Number | None = None,
        aggressive: Boolean = False,
    ) -> Number:
        """
        Register a rejected time step and compute a smaller step size from the specific failure
        criteria encountered.

        Each diagnostic argument gives a proportional backoff factor and the most conservative
        one is applied to the step that failed (not to the size proposed before it, which a
        time target may have shortened). When the result would fall below the last known good
        size and the failure is not severe (backoff factor of 0.5 or more), it is held at 90% of
        that size so the timer does not over-cut.

        :param step_size: The step size that was rejected, in `unit_system`'s
            native time unit.
        :param aggressive: When True, the final factor is additionally capped
            by `aggressive_backoff_factor`. Intended for use after repeated
            consecutive rejections.
        :param maximum_cfl_encountered: Maximum CFL number observed during the
            attempted step.
        :param cfl_threshold: The CFL threshold that was active during the step.
            Falls back to `maximum_cfl` when None.
        :param newton_iterations: Number of Newton iterations attempted before
            the transport solve failed or was deemed too expensive.
        :param maximum_saturation_change: Maximum phase saturation change
            encountered during the attempted step (fraction, 0-1).
        :param maximum_allowed_saturation_change: The configured saturation
            change limit that was violated.
        :param maximum_pressure_change: Maximum absolute pressure change
            encountered during the attempted step, in the run's pressure unit.
        :param maximum_allowed_pressure_change: The configured pressure change
            limit that was violated, in the run's pressure unit.
        :param absolute_oil_mbe: Absolute oil material balance error.
        :param absolute_water_mbe: Absolute water material balance error.
        :param absolute_gas_mbe: Absolute gas material balance error,
            including dissolved gas contribution.
        :param total_absolute_mbe: Sum of all absolute phase MBEs.
        :param relative_oil_mbe: Oil MBE as a fraction of the previous-step oil
            pore volume.
        :param relative_water_mbe: Water MBE as a fraction of the previous-step
            water pore volume.
        :param relative_gas_mbe: Gas MBE as a fraction of the previous-step gas
            pore volume equivalent.
        :param total_relative_mbe: Total MBE as a fraction of the previous-step
            total pore volume.
        :return: The new sustainable time step size, in `unit_system`'s native
            time unit.
        :raises TimingError: If the number of consecutive rejections has reached
            `maximum_rejections`, if the step that failed was already at
            `minimum_step_size`, or if the step size is constant and so cannot be reduced.
        """
        if self.use_constant_step_size:
            raise TimingError(
                f"The step of size {step_size} was rejected, but the step size is constant "
                "and cannot be reduced. Allow a range between `minimum_step_size` and "
                "`maximum_step_size` to let the timer recover from rejected steps."
            )

        if self.rejection_count >= self.maximum_rejections:
            raise TimingError("Maximum number of consecutive time step rejections exceeded")

        if step_size <= self.minimum_step_size * (1.0 + 1e-9):
            raise TimingError(
                f"The step of size {step_size} was rejected, but it is already at the minimum "
                f"step size {self.minimum_step_size}. Retrying it would fail the same way."
            )

        rejection_threshold = int(0.7 * self.maximum_rejections)
        if rejection_threshold <= self.rejection_count < self.maximum_rejections:
            logger.warning(
                "Time step rejection count (%d) is approaching maximum allowed (%d).",
                self.rejection_count,
                self.maximum_rejections,
            )

        self.failed_step_sizes.append(step_size)
        self.recent_metrics.append(
            StepMetrics(
                step=self.next_step,
                step_size=step_size,
                cfl=maximum_cfl_encountered,
                newton_iterations=newton_iterations,
                success=False,
            )
        )

        factor = self.compute_backoff_factor(
            maximum_cfl_encountered=maximum_cfl_encountered,
            cfl_threshold=cfl_threshold,
            newton_iterations=newton_iterations,
            maximum_saturation_change=maximum_saturation_change,
            maximum_allowed_saturation_change=maximum_allowed_saturation_change,
            maximum_pressure_change=maximum_pressure_change,
            maximum_allowed_pressure_change=maximum_allowed_pressure_change,
            absolute_oil_mbe=absolute_oil_mbe,
            absolute_water_mbe=absolute_water_mbe,
            absolute_gas_mbe=absolute_gas_mbe,
            total_absolute_mbe=total_absolute_mbe,
            relative_oil_mbe=relative_oil_mbe,
            relative_water_mbe=relative_water_mbe,
            relative_gas_mbe=relative_gas_mbe,
            total_relative_mbe=total_relative_mbe,
            aggressive=aggressive,
        )
        new_step_size = step_size * factor

        if (
            self.last_successful_step_size > 0.0
            and step_size > self.last_successful_step_size
            and factor >= 0.5
            and new_step_size < self.last_successful_step_size
        ):
            new_step_size = max(new_step_size, self.last_successful_step_size * 0.90)

        if new_step_size < self.minimum_step_size:
            logger.warning(
                "Step size %.6e would be below minimum %.6e. Clamping to minimum.",
                new_step_size,
                self.minimum_step_size,
            )
            new_step_size = self.minimum_step_size

        self.next_step_size = min(self.next_step_size, new_step_size)
        self.failure_ceiling = step_size
        self.last_step_failed = True
        self.rejection_count += 1
        self.steps_since_last_failure = 0

        logger.debug(
            "Step size %.6e rejected (step %d, elapsed %.4f). Backoff factor: %.3f -> new size: %.6e",
            step_size,
            self.next_step,
            self.elapsed_time,
            factor,
            self.next_step_size,
        )
        return self.next_step_size

    def compute_backoff_factor(
        self,
        *,
        maximum_cfl_encountered: Number | None = None,
        cfl_threshold: Number | None = None,
        newton_iterations: int | None = None,
        maximum_saturation_change: Number | None = None,
        maximum_allowed_saturation_change: Number | None = None,
        maximum_pressure_change: Number | None = None,
        maximum_allowed_pressure_change: Number | None = None,
        absolute_oil_mbe: Number | None = None,
        absolute_water_mbe: Number | None = None,
        absolute_gas_mbe: Number | None = None,
        total_absolute_mbe: Number | None = None,
        relative_oil_mbe: Number | None = None,
        relative_water_mbe: Number | None = None,
        relative_gas_mbe: Number | None = None,
        total_relative_mbe: Number | None = None,
        aggressive: Boolean = False,
    ) -> Number:
        """
        Compute the most conservative (smallest) backoff factor from all
        failure signals. Returns a value in (0, 1].
        """
        factors: list[Number] = []

        # CFL
        cfl_limit = (
            cfl_threshold
            if (cfl_threshold is not None and cfl_threshold > 0)
            else self.maximum_cfl
        )
        if maximum_cfl_encountered is not None and maximum_cfl_encountered > cfl_limit:
            ratio = maximum_cfl_encountered / cfl_limit
            if ratio > 2.0:
                factors.append(0.30)
            elif ratio > 1.5:
                factors.append(0.50)
            else:
                factors.append(max((cfl_limit * 0.90) / maximum_cfl_encountered, 0.60))

        # Saturation change
        saturation_utilization = compute_utilization(
            actual=maximum_saturation_change, limit=maximum_allowed_saturation_change
        )
        if saturation_utilization is not None and saturation_utilization > 1.0:
            if saturation_utilization > 3.0:
                factors.append(0.25)
            elif saturation_utilization > 2.0:
                factors.append(0.40)
            else:
                factors.append(max(1.0 / saturation_utilization, 0.50))

        # Pressure change
        pressure_utilization = compute_utilization(
            actual=maximum_pressure_change, limit=maximum_allowed_pressure_change
        )
        if pressure_utilization is not None and pressure_utilization > 1.0:
            if pressure_utilization > 3.0:
                factors.append(0.25)
            elif pressure_utilization > 2.0:
                factors.append(0.40)
            else:
                factors.append(max(1.0 / pressure_utilization, 0.50))

        # Newton iterations
        if newton_iterations is not None:
            if newton_iterations > 20:
                factors.append(0.30)
            elif newton_iterations > 15:
                factors.append(0.50)
            elif newton_iterations > 10:
                factors.append(0.70)

        # MBE violations
        mbe_pairs = [
            (absolute_oil_mbe, self.maximum_absolute_oil_mbe),
            (absolute_water_mbe, self.maximum_absolute_water_mbe),
            (absolute_gas_mbe, self.maximum_absolute_gas_mbe),
            (total_absolute_mbe, self.maximum_total_absolute_mbe),
            (relative_oil_mbe, self.maximum_relative_oil_mbe),
            (relative_water_mbe, self.maximum_relative_water_mbe),
            (relative_gas_mbe, self.maximum_relative_gas_mbe),
            (total_relative_mbe, self.maximum_total_relative_mbe),
        ]
        for actual, limit in mbe_pairs:
            utilization = compute_utilization(actual, limit)
            if utilization is not None and utilization > 1.0:
                if utilization > 3.0:
                    factors.append(0.40)
                elif utilization > 1.5:
                    factors.append(0.60)
                else:
                    factors.append(max(0.90 / utilization, 0.70))

        if factors:
            final = min(factors)
            if aggressive:
                final = min(final, self.aggressive_backoff_factor)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Backoff factor: %.3f (from %d criteria)", final, len(factors))
            return final

        # Fallback when no diagnostic information is available
        return self.aggressive_backoff_factor if aggressive else self.backoff_factor

    def is_acceptable(
        self,
        *,
        maximum_cfl_encountered: Number | None = None,
        cfl_threshold: Number | None = None,
        maximum_saturation_change: Number | None = None,
        maximum_allowed_saturation_change: Number | None = None,
        maximum_pressure_change: Number | None = None,
        maximum_allowed_pressure_change: Number | None = None,
        absolute_oil_mbe: Number | None = None,
        absolute_water_mbe: Number | None = None,
        absolute_gas_mbe: Number | None = None,
        total_absolute_mbe: Number | None = None,
        relative_oil_mbe: Number | None = None,
        relative_water_mbe: Number | None = None,
        relative_gas_mbe: Number | None = None,
        total_relative_mbe: Number | None = None,
        **kwargs: typing.Any,
    ) -> tuple[bool, str]:
        """
        Determine whether a completed time step meets all acceptance criteria.

        Each argument is checked against its corresponding configured limit.
        The first violated criterion causes an immediate `(False, message)`
        return; subsequent criteria are not evaluated. When all criteria pass
        `(True, 'Step acceptable')` is returned.

        :param maximum_cfl_encountered: Maximum CFL number observed during the
            step. Checked against `cfl_threshold` (or `maximum_cfl` when
            `cfl_threshold` is `None`).
        :param cfl_threshold: The CFL threshold that was active during the step.
        :param maximum_saturation_change: Maximum phase saturation change
            encountered (fraction, 0-1).
        :param maximum_allowed_saturation_change: Configured saturation change
            limit. Both this and `maximum_saturation_change` must be non-`None`
            for the check to be active.
        :param maximum_pressure_change: Maximum absolute pressure change
            encountered (psi).
        :param maximum_allowed_pressure_change: Configured pressure change limit
            (psi). Both this and `maximum_pressure_change` must be non-`None`
            for the check to be active.
        :param absolute_oil_mbe: Absolute oil MBE (res ft³) to check against
            `maximum_absolute_oil_mbe`.
        :param absolute_water_mbe: Absolute water MBE (res ft³) to check against
            `maximum_absolute_water_mbe`.
        :param absolute_gas_mbe: Absolute gas MBE (res ft³) to check against
            `maximum_absolute_gas_mbe`.
        :param total_absolute_mbe: Total absolute MBE (res ft³) to check against
            `maximum_total_absolute_mbe`.
        :param relative_oil_mbe: Relative oil MBE fraction to check against
            `maximum_relative_oil_mbe`.
        :param relative_water_mbe: Relative water MBE fraction to check against
            `maximum_relative_water_mbe`.
        :param relative_gas_mbe: Relative gas MBE fraction to check against
            `maximum_relative_gas_mbe`.
        :param total_relative_mbe: Total relative MBE fraction to check against
            `maximum_total_relative_mbe`.
        :param kwargs: Additional keyword arguments forwarded from
            `StepResult.timer_context`; silently ignored.
        :return: Tuple of `(acceptable, message)`. `acceptable` is `True`
            when the step passes all checks. `message` describes the first
            violated criterion, or `'Step acceptable'` on success.
        """
        cfl_limit = cfl_threshold if cfl_threshold is not None else self.maximum_cfl
        if maximum_cfl_encountered is not None and maximum_cfl_encountered > cfl_limit:
            return (
                False,
                f"Maximum CFL ({cfl_limit}) violated. Encountered: {maximum_cfl_encountered}",
            )

        saturation_utilization = compute_utilization(
            actual=maximum_saturation_change, limit=maximum_allowed_saturation_change
        )
        if saturation_utilization is not None and saturation_utilization > 1.0:
            return (
                False,
                f"Maximum allowed saturation change ({maximum_allowed_saturation_change}) violated. "
                f"Encountered: {maximum_saturation_change}",
            )

        pressure_utilization = compute_utilization(
            actual=maximum_pressure_change, limit=maximum_allowed_pressure_change
        )
        if pressure_utilization is not None and pressure_utilization > 1.0:
            return (
                False,
                f"Maximum allowed pressure change ({maximum_allowed_pressure_change}) violated. "
                f"Encountered: {maximum_pressure_change}",
            )

        mbe_violated, mbe_messages = self.check_mbe_violations(
            absolute_oil_mbe=absolute_oil_mbe,
            absolute_water_mbe=absolute_water_mbe,
            absolute_gas_mbe=absolute_gas_mbe,
            total_absolute_mbe=total_absolute_mbe,
            relative_oil_mbe=relative_oil_mbe,
            relative_water_mbe=relative_water_mbe,
            relative_gas_mbe=relative_gas_mbe,
            total_relative_mbe=total_relative_mbe,
        )
        if mbe_violated:
            return False, "MBE limits violated: " + "; ".join(mbe_messages)

        return True, "Step acceptable"

    def add_time_targets(
        self, *sources: TimeTargetSource, start_date: datetime.datetime | None = None
    ) -> None:
        """
        Adds more times the timer must land on, to the ones it already has.

        :param sources: Anything `build_time_targets` accepts: a schedule, a deck file, elapsed
            times in `unit_system`'s native time unit, dates, `PeriodicTarget` or `TimeTargets`.
        :param start_date: The simulation's start date, needed for dates and for `DateEvent`.
        """
        self.time_targets = self.time_targets | build_time_targets(
            *sources, start_date=start_date, unit_system=self.unit_system
        )

    def update_time_targets(
        self, *sources: TimeTargetSource, start_date: datetime.datetime | None = None
    ) -> Boolean:
        """
        Makes the timer's time targets match `sources`, rebuilding them only if `sources` is not
        the same as in the previous call. Call it before every step of a run: a schedule is
        immutable, so editing one produces a new object, and passing that object is all it takes
        for added events to be landed on and removed events to stop being landed on.

        This replaces any targets the timer had, whether set directly or from an earlier call.
        A source counts as unchanged if it is the very same object, or an equal number or date.

        :param sources: Anything `build_time_targets` accepts.
        :param start_date: The simulation's start date, needed for dates and for `DateEvent`.
        :returns: `True` if the targets were rebuilt, `False` if nothing had changed.
        """
        previous = self._target_sources
        if (
            previous is not None
            and self._target_start_date == start_date
            and len(previous) == len(sources)
            and all(
                check_same_source(old, new) for old, new in zip(previous, sources, strict=True)
            )
        ):
            return False

        self.time_targets = build_time_targets(
            *sources, start_date=start_date, unit_system=self.unit_system
        )
        self._target_sources = sources
        self._target_start_date = start_date
        return True

    def snap_time(self, time: Number) -> Number:
        """
        Moves a time that landed within `time_tolerance` of a time target, or of
        `simulation_time`, onto it exactly, so an event at that time is never missed by
        floating point error.

        :param time: A clock value that may have landed just beside a target.
        :return: The target it landed on, or `time` itself if it landed on none.
        """
        tolerance = self.time_tolerance
        if abs(time - self.simulation_time) <= tolerance:
            return self.simulation_time
        if self.use_constant_step_size and self.enforce_constant_step_size:
            return time

        snapped = self.time_targets.snap(time, tolerance=tolerance)
        if snapped is None:
            return time
        return min(snapped, self.simulation_time)

    def accept_step(
        self,
        step_size: Number,
        *,
        maximum_cfl_encountered: Number | None = None,
        cfl_threshold: Number | None = None,
        newton_iterations: int | None = None,
        maximum_saturation_change: Number | None = None,
        maximum_allowed_saturation_change: Number | None = None,
        maximum_pressure_change: Number | None = None,
        maximum_allowed_pressure_change: Number | None = None,
        absolute_oil_mbe: Number | None = None,
        absolute_water_mbe: Number | None = None,
        absolute_gas_mbe: Number | None = None,
        total_absolute_mbe: Number | None = None,
        relative_oil_mbe: Number | None = None,
        relative_water_mbe: Number | None = None,
        relative_gas_mbe: Number | None = None,
        total_relative_mbe: Number | None = None,
    ) -> Number:
        """
        Register an accepted time step and work out the step size the solver can sustain next.

        The CFL, saturation, pressure and material balance diagnostics each give the largest
        step the solver would tolerate, found by scaling the step just taken by how far the
        measurement sat from its target. That holds for a step that was shortened to land on a
        time target too. Newton iteration counts and recent trends adjust the sustainable size
        directly. The smallest result is the proposal. A proposal below the current size is
        adopted at once, apart from a small band (90% to 100% of the current size) where the
        size is held to avoid jitter. A proposal above the current size is limited by
        `maximum_growth_per_step` (or `ramp_up_factor` while comfortable), by recent failures,
        and by `step_size_smoothing`.

        :param step_size: The time step size that was just accepted, in
            `unit_system`'s native time unit.
        :param maximum_cfl_encountered: Maximum CFL number observed during the step.
        :param cfl_threshold: The CFL threshold that was active during the step.
            Falls back to `maximum_cfl` when `None`.
        :param newton_iterations: Number of Newton iterations taken by the
            transport solver. Counts above 10 reduce the step size, and counts below 4 allow
            growth when nothing else reports.
        :param maximum_saturation_change: Maximum phase saturation change
            encountered during the step (fraction, 0-1).
        :param maximum_allowed_saturation_change: Configured saturation change
            limit used to compute the utilization ratio.
        :param maximum_pressure_change: Maximum absolute pressure change
            encountered during the step, in the run's pressure unit.
        :param maximum_allowed_pressure_change: Configured pressure change limit
            used to compute the utilization ratio.
        :param absolute_oil_mbe: Absolute oil material balance error.
            Only used when `use_mbe_for_step_size` is `True`.
        :param absolute_water_mbe: Absolute water material balance error.
            Only used when `use_mbe_for_step_size` is `True`.
        :param absolute_gas_mbe: Absolute gas material balance error,
            including dissolved gas. Only used when `use_mbe_for_step_size`
            is `True`.
        :param total_absolute_mbe: Sum of all absolute phase MBEs.
            Only used when `use_mbe_for_step_size` is `True`.
        :param relative_oil_mbe: Oil MBE as a fraction of the previous-step oil
            pore volume. Only used when `use_mbe_for_step_size` is `True`.
        :param relative_water_mbe: Water MBE as a fraction of the previous-step
            water pore volume. Only used when `use_mbe_for_step_size` is `True`.
        :param relative_gas_mbe: Gas MBE as a fraction of the previous-step gas
            pore volume equivalent. Only used when `use_mbe_for_step_size`
            is `True`.
        :param total_relative_mbe: Total MBE as a fraction of the previous-step
            total pore volume. Only used when `use_mbe_for_step_size` is `True`.
        :return: The step size the solver can sustain next, in `unit_system`'s native
            time unit. Use `propose_step_size` for the size to actually take.
        :raises TimingError: If `step_size` exceeds the remaining simulation time,
            which indicates a bug in the calling time-stepping logic.
        """
        if step_size > self.time_remaining + self.time_tolerance:
            raise TimingError(
                f"Step size {step_size} exceeds remaining time {self.time_remaining}."
            )

        sustainable = (
            self.initial_step_size if self.use_constant_step_size else self.next_step_size
        )

        self.elapsed_time = self.snap_time(self.elapsed_time + step_size)
        self.step_size = step_size
        self.last_successful_step_size = step_size
        self.step += 1
        self.steps_since_last_failure += 1

        self.recent_metrics.append(
            StepMetrics(
                step=self.step,
                step_size=step_size,
                cfl=maximum_cfl_encountered,
                newton_iterations=newton_iterations,
                success=True,
            )
        )
        if maximum_cfl_encountered is not None and maximum_cfl_encountered > 0.0:
            self._cfl_window.append(maximum_cfl_encountered)
        if newton_iterations is not None and newton_iterations > 0:
            self._newton_window.append(newton_iterations)

        self.last_step_failed = False
        self.rejection_count = 0

        if self.use_constant_step_size:
            return self.initial_step_size

        # Each stress signal (CFL, saturation, pressure, MBE) grows with the step size, so a
        # step of any length says how large a step the solver can sustain: the step taken scaled
        # by how far the signal sat from its target. That stays valid for a step that was
        # shortened to land on a time target.
        stress_ratios: list[Number] = []
        utilizations: list[Number] = []

        cfl_limit = cfl_threshold if cfl_threshold is not None else self.maximum_cfl
        if maximum_cfl_encountered is not None and maximum_cfl_encountered > 0.0:
            stress_ratios.append(cfl_limit * self.cfl_safety_margin / maximum_cfl_encountered)
            utilizations.append(maximum_cfl_encountered / cfl_limit)

        signals: list[tuple[Number | None, Number | None]] = [
            (maximum_saturation_change, maximum_allowed_saturation_change),
            (maximum_pressure_change, maximum_allowed_pressure_change),
        ]
        if self.use_mbe_for_step_size:
            signals.extend([
                (absolute_oil_mbe, self.maximum_absolute_oil_mbe),
                (absolute_water_mbe, self.maximum_absolute_water_mbe),
                (absolute_gas_mbe, self.maximum_absolute_gas_mbe),
                (total_absolute_mbe, self.maximum_total_absolute_mbe),
                (relative_oil_mbe, self.maximum_relative_oil_mbe),
                (relative_water_mbe, self.maximum_relative_water_mbe),
                (relative_gas_mbe, self.maximum_relative_gas_mbe),
                (total_relative_mbe, self.maximum_total_relative_mbe),
            ])

        for actual, limit in signals:
            utilization = compute_utilization(
                None if actual is None else abs(float(actual)), limit
            )
            if utilization is not None and utilization > 0.0:
                stress_ratios.append(self.target_utilization / utilization)
                utilizations.append(utilization)

        # Newton iteration counts and the recent trend do not scale with the step taken, so
        # they adjust the sustainable size directly.
        sustainable_ratios: list[Number] = []
        clipped = step_size < sustainable * (1.0 - 1e-9)
        newton_is_easy = False
        if newton_iterations is not None:
            if newton_iterations > 10:
                sustainable_ratios.append(max(0.5, 10.0 / newton_iterations))
            elif newton_iterations < 4 and self.steps_since_last_failure >= 3:
                newton_is_easy = True

        if newton_is_easy and not stress_ratios and not clipped:
            sustainable_ratios.append(1.2)

        performance_factor = self.compute_performance_factor()
        if performance_factor < 1.0:
            sustainable_ratios.append(performance_factor)

        candidates: list[Number] = []
        if stress_ratios:
            candidates.append(step_size * min(stress_ratios))
        if sustainable_ratios:
            candidates.append(sustainable * min(sustainable_ratios))

        proposal = min(candidates) if candidates else sustainable
        if 0.9 * sustainable <= proposal < sustainable:
            proposal = sustainable

        if proposal <= sustainable:
            next_size = proposal
        else:
            cleared = self.steps_since_last_failure >= self.growth_cooldown_steps
            comfortable = max(utilizations) < 0.5 if utilizations else newton_is_easy
            if not cleared:
                growth_cap = min(self.maximum_growth_per_step, 1.1)
            elif comfortable and self.ramp_up_factor is not None:
                growth_cap = self.ramp_up_factor
            else:
                growth_cap = self.maximum_growth_per_step

            allowed = min(proposal, sustainable * growth_cap)
            next_size = sustainable + self.step_size_smoothing * (allowed - sustainable)

        if self.failure_ceiling is not None:
            if self.failure_ceiling >= self.maximum_step_size:
                self.failure_ceiling = None
            else:
                next_size = min(next_size, 0.95 * self.failure_ceiling)
                self.failure_ceiling *= 1.15

        self.next_step_size = min(max(next_size, self.minimum_step_size), self.maximum_step_size)

        logger.debug(
            "Step %.6e accepted (step %d, elapsed %.4f). Sustainable next: %.6e (ratio %.3f)",
            step_size,
            self.step,
            self.elapsed_time,
            self.next_step_size,
            proposal / sustainable if sustainable else 1.0,
        )
        return self.next_step_size

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `Timer` with every time-valued field rescaled to `target`.

        Step sizes, simulation/elapsed time, time targets and step-size history are rescaled
        from `unit_system`'s native time unit to `target`'s, and the absolute material balance
        thresholds from its native volume unit to `target`'s. Every other field carries over.

        :param target: Desired `UnitSystem`.
        :param table: Optional custom conversion table; `None` uses the default.
        :returns: New `Timer` in `target`'s units, with accrued runtime state preserved and rescaled.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        time_factor = factors["time"]
        volume_factor = factors["volume"]

        converted = attrs.evolve(
            self,
            initial_step_size=scale(self.initial_step_size, time_factor),
            maximum_step_size=scale(self.maximum_step_size, time_factor),
            minimum_step_size=scale(self.minimum_step_size, time_factor),
            simulation_time=scale(self.simulation_time, time_factor),
            time_targets=self.time_targets.scaled(time_factor),
            maximum_absolute_oil_mbe=scale(self.maximum_absolute_oil_mbe, volume_factor),  # type: ignore[arg-type]
            maximum_absolute_water_mbe=scale(self.maximum_absolute_water_mbe, volume_factor),  # type: ignore[arg-type]
            maximum_absolute_gas_mbe=scale(self.maximum_absolute_gas_mbe, volume_factor),  # type: ignore[arg-type]
            maximum_total_absolute_mbe=scale(self.maximum_total_absolute_mbe, volume_factor),  # type: ignore[arg-type]
            unit_system=target,
        )

        converted.elapsed_time = scale(self.elapsed_time, time_factor)
        converted.step_size = scale(self.step_size, time_factor)
        converted.next_step_size = scale(self.next_step_size, time_factor)
        converted.step = self.step
        converted.last_step_failed = self.last_step_failed
        converted.rejection_count = self.rejection_count
        converted.steps_since_last_failure = self.steps_since_last_failure
        converted.last_successful_step_size = scale(self.last_successful_step_size, time_factor)
        converted.failure_ceiling = scale(self.failure_ceiling, time_factor)
        converted._constant_step_warned = self._constant_step_warned
        converted.recent_metrics = deque(
            (
                attrs.evolve(metrics, step_size=scale(metrics.step_size, time_factor))
                for metrics in self.recent_metrics
            ),
            maxlen=self.metrics_history_size,
        )
        converted.failed_step_sizes = deque(
            (scale(failed_size, time_factor) for failed_size in self.failed_step_sizes),
            maxlen=self.failure_memory_window,
        )
        converted._cfl_window = deque(self._cfl_window, maxlen=self.metrics_history_size)
        converted._newton_window = deque(self._newton_window, maxlen=self.metrics_history_size)
        return converted

    def dump_state(self) -> TimerState:
        """
        Serialize the current timer state to a dictionary. `time_targets` are not included,
        since they are rebuilt from the schedule and the deck when a run is restarted.
        """
        return {
            "initial_step_size": self.initial_step_size,
            "maximum_step_size": self.maximum_step_size,
            "minimum_step_size": self.minimum_step_size,
            "simulation_time": self.simulation_time,
            "unit_system": self.unit_system.value,
            **{name: getattr(self, name) for name in TIMER_SETTING_NAMES},  # type: ignore[typeddict-item]
            "elapsed_time": self.elapsed_time,
            "step": self.step,
            "step_size": self.step_size,
            "next_step_size": self.next_step_size,
            "last_step_failed": self.last_step_failed,
            "rejection_count": self.rejection_count,
            "steps_since_last_failure": self.steps_since_last_failure,
            "last_successful_step_size": self.last_successful_step_size,
            "failure_ceiling": self.failure_ceiling,
            "recent_metrics": [
                typing.cast(StepMetricsDict, attrs.asdict(metrics))
                for metrics in self.recent_metrics
            ],
            "failed_step_sizes": list(self.failed_step_sizes),
        }  # type: ignore[return-value]

    @classmethod
    def load_state(cls, state: TimerState, *, time_targets: TimeTargets | None = None) -> Self:
        """
        Reconstruct a timer from a previously saved state dictionary. Settings missing
        from `state` take the class defaults.

        :param state: A dictionary from `dump_state`.
        :param time_targets: The times the restarted run must land on. Rebuild them from the
            schedule and the deck, since they are not part of the saved state.
        :raises ValidationError: If a required key is missing.
        """
        required_keys = {
            "initial_step_size",
            "maximum_step_size",
            "minimum_step_size",
            "simulation_time",
            "elapsed_time",
            "step",
            "step_size",
        }
        missing = required_keys - set(state.keys())
        if missing:
            raise ValidationError(f"Timer state missing required keys: {missing}")

        params: dict[str, typing.Any] = {
            name: state[name]  # type: ignore[literal-required]
            for name in (
                "initial_step_size",
                "maximum_step_size",
                "minimum_step_size",
                "simulation_time",
                *TIMER_SETTING_NAMES,
            )
            if name in state
        }
        params["unit_system"] = UnitSystem(state.get("unit_system", UnitSystem.FIELD.value))
        if time_targets is not None:
            params["time_targets"] = time_targets
        timer = cls(**params)

        timer.elapsed_time = state.get("elapsed_time", 0.0)
        timer.step = state.get("step", 0)
        timer.step_size = state.get("step_size", 0.0)
        timer.next_step_size = state.get("next_step_size", state["step_size"])
        timer.last_step_failed = state.get("last_step_failed", False)
        timer.rejection_count = state.get("rejection_count", 0)
        timer.steps_since_last_failure = state.get(
            "steps_since_last_failure", timer.growth_cooldown_steps
        )
        timer.last_successful_step_size = state.get(
            "last_successful_step_size", state["step_size"]
        )
        timer.failure_ceiling = state.get("failure_ceiling")

        recent_metrics = deque(
            [StepMetrics(**metrics) for metrics in state.get("recent_metrics", [])],
            maxlen=timer.metrics_history_size,
        )
        timer.recent_metrics = recent_metrics
        timer.failed_step_sizes = deque(
            state.get("failed_step_sizes", []), maxlen=timer.failure_memory_window
        )

        for metrics in recent_metrics:
            if metrics.success:
                if metrics.cfl is not None and metrics.cfl > 0.0:
                    timer._cfl_window.append(metrics.cfl)
                if metrics.newton_iterations is not None and metrics.newton_iterations > 0:
                    timer._newton_window.append(metrics.newton_iterations)

        logger.debug(
            "Timer state loaded: step %d, elapsed %.4f, step_size %.6e",
            timer.step,
            timer.elapsed_time,
            timer.step_size,
        )
        return timer
