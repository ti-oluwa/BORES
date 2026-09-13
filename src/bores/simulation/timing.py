"""Simulation time management with smart and adaptive time stepping."""

import logging
import typing
from collections import deque
from datetime import datetime, timedelta

import attrs
from typing_extensions import Self, TypedDict

from bores.constants import c
from bores.errors import TimingError, ValidationError
from bores.serde.stores import StoreSerializable
from bores.types import Number, UnitSystem

__all__ = ["Time", "Timer", "TimerState", "get_current_date", "get_current_time"]

logger = logging.getLogger(__name__)

TIME_UNIT_PER_UNIT_SYSTEM = {
    UnitSystem.FIELD: "days",
    UnitSystem.METRIC: "days",
    UnitSystem.LAB: "hours",
    UnitSystem.SI: "seconds",
}
"""Elapsed-time unit each `UnitSystem` uses, following Eclipse/OPM convention."""


def get_current_date(
    *, start_date: datetime, elapsed_time: Number, unit_system: UnitSystem
) -> datetime:
    """
    Converts an elapsed-time offset into a calendar date.

    :param start_date: The run's start date.
    :param elapsed_time: Time since `start_date`, in `unit_system`'s own
        time unit (days for `FIELD`/`METRIC`, hours for `LAB`, seconds for `SI`).
    :param unit_system: Determines `elapsed_time`'s unit.
    :returns: `start_date` plus `elapsed_time`.
    """
    return start_date + timedelta(**{TIME_UNIT_PER_UNIT_SYSTEM[unit_system]: float(elapsed_time)})


def get_current_time(
    *, start_date: datetime, current_date: datetime, unit_system: UnitSystem
) -> float:
    """
    Converts a calendar date into an elapsed-time offset from `start_date`.

    :param start_date: The run's start date.
    :param current_date: The date to find the elapsed time to.
    :param unit_system: Determines the result's unit (days for
        `FIELD`/`METRIC`, hours for `LAB`, seconds for `SI`).
    :returns: `current_date - start_date`, in `unit_system`'s own time unit.
    """
    total_seconds = (current_date - start_date).total_seconds()
    if unit_system is UnitSystem.SI:
        return total_seconds
    if unit_system is UnitSystem.LAB:
        return total_seconds * c.DAYS_PER_SECOND * c.HOURS_PER_DAY
    return total_seconds * c.DAYS_PER_SECOND


def Time(
    milliseconds: float = 0,
    seconds: float = 0,
    minutes: float = 0,
    hours: float = 0,
    days: float = 0,
    weeks: float = 0,
    months: float = 0,
    years: float = 0,
) -> float:
    """
    Expresses time components as total seconds.

    :param milliseconds: Number of milliseconds.
    :param seconds: Number of seconds.
    :param minutes: Number of minutes.
    :param hours: Number of hours.
    :param days: Number of days.
    :param weeks: Number of weeks.
    :param months: Number of months.
    :param years: Number of years.
    :return: Total time in seconds.
    """
    if years:
        days += years * c.DAYS_PER_YEAR

    if months:
        days += months * (c.DAYS_PER_YEAR / c.MONTHS_PER_YEAR)

    delta = timedelta(
        weeks=weeks,
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        milliseconds=milliseconds,
    )
    return delta.total_seconds()


class StepMetricsDict(TypedDict):
    """Dictionary representation of step metrics."""

    step_number: int
    step_size: float
    cfl: float | None
    newton_iterations: int | None
    success: bool


class TimerState(TypedDict):
    """Complete state of a timer instance for serialization."""

    initial_step_size: float
    maximum_step_size: float
    minimum_step_size: float
    simulation_time: float
    maximum_cfl: float
    cfl_safety_margin: float
    ramp_up_factor: float | None
    backoff_factor: float
    aggressive_backoff_factor: float
    maximum_steps: int | None
    maximum_absolute_oil_mbe: float | None
    maximum_absolute_water_mbe: float | None
    maximum_absolute_gas_mbe: float | None
    maximum_total_absolute_mbe: float | None
    maximum_relative_oil_mbe: float | None
    maximum_relative_water_mbe: float | None
    maximum_relative_gas_mbe: float | None
    maximum_total_relative_mbe: float | None
    use_mbe_for_step_size: bool
    maximum_rejections: int
    maximum_growth_per_step: float
    step_size_smoothing: float
    growth_cooldown_steps: int
    failure_memory_window: int
    metrics_history_size: int
    use_constant_step_size: bool
    elapsed_time: float
    step: int
    step_size: float
    next_step_size: float
    ema_step_size: float
    last_step_failed: bool
    rejection_count: int
    steps_since_last_failure: int
    recent_metrics: list[StepMetricsDict]
    failed_step_sizes: list[float]
    last_successful_step_size: float


@attrs.frozen(slots=True)
class StepMetrics:
    """Metrics for a single time step."""

    step_number: int
    step_size: float
    cfl: float | None = None
    newton_iterations: int | None = None
    success: bool = True


def _utilization(actual: float | None, limit: float | None) -> float | None:
    """Return actual/limit, or None if either is None/zero."""
    if actual is None or limit is None or limit <= 0.0:
        return None
    return actual / limit


@attrs.mutable
class Timer(StoreSerializable):
    """
    Simulation time manager for smart and adaptive time stepping.
    """

    initial_step_size: float
    """Initial time step size in seconds."""
    maximum_step_size: float
    """Maximum allowable time step size in seconds."""
    minimum_step_size: float
    """Minimum allowable time step size in seconds."""
    simulation_time: float
    """Total simulation time in seconds."""
    maximum_cfl: float = 0.9
    """Default CFL limit (max CFL number) for time step adjustments."""
    ramp_up_factor: float | None = None
    """Factor by which to ramp up time step size on successful steps."""
    backoff_factor: float = 0.5
    """
    Default factor by which to reduce time step size on failed steps.

    Only used when there is not enough information to intelligently determine a
    suitable backoff factor.
    """
    aggressive_backoff_factor: float = 0.25
    """Factor by which to aggressively reduce time step size on failed steps."""
    maximum_steps: int | None = None
    """Maximum number of time steps to run for."""

    # Adaptive parameters
    growth_cooldown_steps: int = 5
    """Minimum successful steps required before allowing aggressive growth."""
    maximum_growth_per_step: float = 1.3
    """Maximum multiplicative growth allowed per step (e.g., 1.3 = 30% max growth)."""
    cfl_safety_margin: float = 0.95
    """Safety factor for CFL-based adjustments (target below max CFL)."""
    step_size_smoothing: float = 0.2
    """
    EMA smoothing factor applied to step size changes.

    The EMA is: `ema = (1 - α) * ema_prev + α * new_dt`, so a value of 0.2
    means 20% weight on the new proposal and 80% on the running average
    i.e., *higher values = less smoothing, faster response* (closer to the raw
    proposed size), *lower values = more smoothing, slower adaptation*.
    Set to 1.0 to disable smoothing entirely.
    """
    metrics_history_size: int = 10
    """Number of recent steps to track for performance analysis."""
    failure_memory_window: int = 5
    """Number of recent failures to remember for adaptive behavior."""

    # MBE-based step control
    maximum_absolute_oil_mbe: float | None = None
    """Absolute oil MBE threshold (res ft³). Reject step if exceeded."""
    maximum_absolute_water_mbe: float | None = None
    """Absolute water MBE threshold (res ft³). Reject step if exceeded."""
    maximum_absolute_gas_mbe: float | None = None
    """Absolute gas MBE threshold (res ft³). Reject step if exceeded."""
    maximum_total_absolute_mbe: float | None = None
    """Total absolute MBE threshold (res ft³). Reject step if exceeded."""
    maximum_relative_oil_mbe: float | None = None
    """Relative oil MBE threshold (fraction, e.g. 0.01 = 1%). Reject step if exceeded."""
    maximum_relative_water_mbe: float | None = None
    """Relative water MBE threshold (fraction). Reject step if exceeded."""
    maximum_relative_gas_mbe: float | None = None
    """Relative gas MBE threshold (fraction). Reject step if exceeded."""
    maximum_total_relative_mbe: float | None = None
    """Total relative MBE threshold (fraction). Reject step if exceeded."""
    use_mbe_for_step_size: bool = False
    """
    If True, MBE values modulate the next step size proposal on accepted steps
    (analogous to CFL/saturation-driven growth). When False (default), MBE only
    triggers rejection when a configured limit is exceeded.
    """

    # Runtime state
    elapsed_time: float = attrs.field(init=False, default=0.0)
    """Current simulation time in seconds (sum of all accepted steps)."""
    step_size: float = attrs.field(init=False, default=0.0)
    """The time step size used for the most recently accepted step."""
    next_step_size: float = attrs.field(init=False, default=0.0)
    """Time step size to propose for the next step."""
    ema_step_size: float = attrs.field(init=False, default=0.0)
    """Exponential moving average of accepted step sizes."""
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
    last_successful_step_size: float = attrs.field(init=False, default=0.0)
    """The step size used in the most recently *accepted* step."""

    # Performance tracking
    recent_metrics: deque[StepMetrics] = attrs.field(init=False)
    """Recent step performance metrics (bounded deque, newest at right)."""
    failed_step_sizes: deque[float] = attrs.field(init=False)
    """Recent failed step sizes for failure-zone memory."""

    # Rolling window statistics stored as a fixed-size deque so the sliding
    # window is exact and never corrupted by approximation.
    _cfl_window: deque[float] = attrs.field(init=False)
    """Sliding window of the last N successful CFL values (N = metrics_history_size)."""
    _newton_window: deque[int] = attrs.field(init=False)
    """Sliding window of the last N successful Newton iteration counts."""

    def __attrs_post_init__(self) -> None:
        self.next_step_size = self.initial_step_size
        self.step_size = self.initial_step_size
        self.ema_step_size = self.initial_step_size
        self.last_successful_step_size = self.initial_step_size
        self.use_constant_step_size = (
            self.initial_step_size == self.maximum_step_size
            and self.initial_step_size == self.minimum_step_size
        )
        self.recent_metrics = deque(maxlen=self.metrics_history_size)
        self.failed_step_sizes = deque(maxlen=self.failure_memory_window)
        # Rolling windows share the same bound as metrics_history_size for simplicity.
        self._cfl_window = deque(maxlen=self.metrics_history_size)
        self._newton_window = deque(maxlen=self.metrics_history_size)

    @property
    def next_step(self) -> int:
        """Returns the next time step count (1-indexed)."""
        return self.step + 1

    def done(self) -> bool:
        """True once the simulation has reached its end criteria."""
        if self.elapsed_time >= (self.simulation_time - 1e-9):
            return True
        return self.maximum_steps is not None and self.step >= self.maximum_steps

    @property
    def time_remaining(self) -> float:
        """Remaining simulation time in seconds."""
        return max(self.simulation_time - self.elapsed_time, 0.0)

    @property
    def is_last_step(self) -> bool:
        """True if the latest accepted step completed the simulation."""
        if self.time_remaining <= 0:
            return True
        return self.maximum_steps is not None and self.step >= self.maximum_steps

    @property
    def _avg_cfl(self) -> float | None:
        """Average CFL over the recent window, or None if no data."""
        if not self._cfl_window:
            return None
        return sum(self._cfl_window) / len(self._cfl_window)

    @property
    def _avg_newton(self) -> float | None:
        """Average Newton iterations over the recent window, or None if no data."""
        if not self._newton_window:
            return None
        return sum(self._newton_window) / len(self._newton_window)

    def _is_near_failed_size(self, dt: float, tolerance: float = 0.10) -> bool:
        """
        True if *dt* is within *tolerance* of a recently failed size **from below**.

        We only guard against stepping into the failure zone from the safe side
        (i.e., growing toward a previously failed size). A size that is already
        *smaller* than a failed size is safe by definition.
        """
        for failed_size in self.failed_step_sizes:
            if 0.0 < (failed_size - dt) / failed_size < tolerance:
                return True
        return False

    def _compute_performance_factor(self) -> float:
        """
        Derive a multiplicative adjustment factor from recent solver trends.

        Returns a value in (0.5, 1.0]. Values below 1.0 indicate the solver
        is under stress and growth should be dampened.
        """
        factor = 1.0

        avg_cfl = self._avg_cfl
        if avg_cfl is not None and len(self._cfl_window) >= 3:
            # If we are consistently using >75 % of the CFL budget, be cautious.
            if avg_cfl > 0.75 * self.maximum_cfl:
                factor *= 0.95
            # Upward CFL trend within the window?
            cfl_list = list(self._cfl_window)
            if len(cfl_list) >= 4:
                trend = cfl_list[-1] - cfl_list[0]
                if trend > 0.15:
                    factor *= 0.90

        avg_newton = self._avg_newton
        if avg_newton is not None and len(self._newton_window) >= 3:
            if avg_newton > 8:
                factor *= 0.85
            # Are the last 3 iterations consistently high?
            recent_newtons = list(self._newton_window)[-3:]
            if len(recent_newtons) == 3 and all(n > 10 for n in recent_newtons):
                factor *= 0.75

        return max(factor, 0.5)

    def _check_mbe_violations(
        self,
        absolute_oil_mbe: float | None = None,
        absolute_water_mbe: float | None = None,
        absolute_gas_mbe: float | None = None,
        total_absolute_mbe: float | None = None,
        relative_oil_mbe: float | None = None,
        relative_water_mbe: float | None = None,
        relative_gas_mbe: float | None = None,
        total_relative_mbe: float | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Check if any MBE limits are violated.

        :return: `(any_violated, message_list)`.
        """
        mbe_checks = [
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
            if actual > limit:
                messages.append(f"{label}: |{actual:.3e}| > {limit:.3e}")
        return bool(messages), messages

    def propose_step_size(self) -> float:
        """
        Propose the next time step size without updating any internal state.

        If the proposed size would exceed the remaining simulation time, it is
        clamped to the remaining time. The minimum step size is honoured
        unless the remaining time is already smaller than it.

        :return: Proposed time step size in seconds.
        """
        if self.use_constant_step_size:
            return self.initial_step_size

        dt = self.next_step_size
        remaining = self.time_remaining
        if dt > remaining:
            # Clamp to remaining time; honour minimum only if time allows it.
            return (
                remaining
                if remaining < self.minimum_step_size
                else max(remaining, self.minimum_step_size)
            )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Proposing time step of size %.6e for step %d at elapsed time %.4f",
                dt,
                self.next_step,
                self.elapsed_time,
            )
        return dt

    def reject_step(
        self,
        step_size: float,
        *,
        aggressive: bool = False,
        maximum_cfl_encountered: float | None = None,
        cfl_threshold: float | None = None,
        newton_iterations: int | None = None,
        maximum_saturation_change: float | None = None,
        maximum_allowed_saturation_change: float | None = None,
        maximum_pressure_change: float | None = None,
        maximum_allowed_pressure_change: float | None = None,
        absolute_oil_mbe: float | None = None,
        absolute_water_mbe: float | None = None,
        absolute_gas_mbe: float | None = None,
        total_absolute_mbe: float | None = None,
        relative_oil_mbe: float | None = None,
        relative_water_mbe: float | None = None,
        relative_gas_mbe: float | None = None,
        total_relative_mbe: float | None = None,
    ) -> float:
        """
        Register a rejected time step and compute an intelligently adjusted step
        size based on the specific failure criteria encountered.

        Each diagnostic argument is used to derive a proportional backoff factor.
        The most conservative factor across all active signals is selected and
        applied to the current `next_step_size`.  When the result would fall
        below the last known-good step size and the failure is not severe
        (backoff factor ≥ 0.5), a conservative floor of
        `last_successful_step_size * 0.90` is applied to avoid over-cutting.

        :param step_size: The step size that was rejected (seconds).
        :param aggressive: When *True*, the final factor is additionally capped
            by `aggressive_backoff_factor`.  Intended for use after repeated
            consecutive rejections.
        :param maximum_cfl_encountered: Maximum CFL number observed during the
            attempted step.
        :param cfl_threshold: The CFL threshold that was active during the step.
            Falls back to `maximum_cfl` when *None*.
        :param newton_iterations: Number of Newton iterations attempted before
            the transport solve failed or was deemed too expensive.
        :param maximum_saturation_change: Maximum phase saturation change
            encountered during the attempted step (fraction, 0-1).
        :param maximum_allowed_saturation_change: The configured saturation
            change limit that was violated.
        :param maximum_pressure_change: Maximum absolute pressure change
            encountered during the attempted step (psi).
        :param maximum_allowed_pressure_change: The configured pressure change
            limit that was violated (psi).
        :param absolute_oil_mbe: Absolute oil material balance error (res ft³).
        :param absolute_water_mbe: Absolute water material balance error (res ft³).
        :param absolute_gas_mbe: Absolute gas material balance error (res ft³),
            including dissolved gas contribution.
        :param total_absolute_mbe: Sum of all absolute phase MBEs (res ft³).
        :param relative_oil_mbe: Oil MBE as a fraction of the previous-step oil
            pore volume.
        :param relative_water_mbe: Water MBE as a fraction of the previous-step
            water pore volume.
        :param relative_gas_mbe: Gas MBE as a fraction of the previous-step gas
            pore volume equivalent.
        :param total_relative_mbe: Total MBE as a fraction of the previous-step
            total pore volume.
        :return: The new proposed time step size in seconds.
        :raises TimingError: If the number of consecutive rejections has reached
            `maximum_rejections`.
        """
        if self.rejection_count >= self.maximum_rejections:
            raise TimingError("Maximum number of consecutive time step rejections exceeded")

        rejection_threshold = int(0.7 * self.maximum_rejections)
        if rejection_threshold <= self.rejection_count < self.maximum_rejections:
            logger.warning(
                "Time step rejection count (%d) is approaching maximum allowed (%d).",
                self.rejection_count,
                self.maximum_rejections,
            )

        if self.use_constant_step_size:
            return self.initial_step_size

        self.failed_step_sizes.append(step_size)
        self.recent_metrics.append(
            StepMetrics(
                step_number=self.next_step,
                step_size=step_size,
                cfl=maximum_cfl_encountered,
                newton_iterations=newton_iterations,
                success=False,
            )
        )

        factor = self._compute_backoff_factor(
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
        raw_new_size = self.next_step_size * factor

        # If the backoff brings us *below* the last
        # successful size, don't go lower than a slight discount off that
        # size. The physics at the current state may have changed, so we
        # don't blindly reuse `last_successful_step_size`, but there's no
        # reason to shrink beyond 90 % of it unless the failure was severe
        # (factor < 0.5) or the last successful size itself was the rejected
        # size (which means it is no longer reliable).
        if (
            self.last_successful_step_size > 0.0
            and step_size > self.last_successful_step_size  # we grew and failed
            and factor >= 0.5  # not a severe failure
            and raw_new_size < self.last_successful_step_size
        ):
            # Prefer a slight discount off the last known-good size rather than
            # an over-aggressive cut.
            conservative_floor = self.last_successful_step_size * 0.90
            raw_new_size = max(raw_new_size, conservative_floor)
            logger.debug(
                "Backoff floor applied: raw %.6e raised to conservative floor %.6e "
                "(last successful: %.6e)",
                self.next_step_size * factor,
                raw_new_size,
                self.last_successful_step_size,
            )

        if raw_new_size < self.minimum_step_size:
            logger.warning(
                "Step size %.6e would be below minimum %.6e. Clamping to minimum.",
                raw_new_size,
                self.minimum_step_size,
            )
        self.next_step_size = max(raw_new_size, self.minimum_step_size)

        # Panic-mode detection: repeated failures at/near minimum
        if self.next_step_size <= self.minimum_step_size * 1.01:
            min_failures = sum(
                1 for s in self.failed_step_sizes if s <= self.minimum_step_size * 1.1
            )
            if min_failures >= 3:
                logger.error(
                    "Repeated failures (%d) at or near minimum step size (%.6e). "
                    "Simulation may be unstable.",
                    min_failures,
                    self.minimum_step_size,
                )

        self.ema_step_size = self.next_step_size
        self.last_step_failed = True
        self.rejection_count += 1
        self.steps_since_last_failure = 0

        logger.debug(
            "Step size %.6e rejected (step %d, elapsed %.4fs). "
            "Backoff factor: %.3f -> new size: %.6e",
            step_size,
            self.next_step,
            self.elapsed_time,
            factor,
            self.next_step_size,
        )
        return self.next_step_size

    def _compute_backoff_factor(
        self,
        *,
        maximum_cfl_encountered: float | None = None,
        cfl_threshold: float | None = None,
        newton_iterations: int | None = None,
        maximum_saturation_change: float | None = None,
        maximum_allowed_saturation_change: float | None = None,
        maximum_pressure_change: float | None = None,
        maximum_allowed_pressure_change: float | None = None,
        absolute_oil_mbe: float | None = None,
        absolute_water_mbe: float | None = None,
        absolute_gas_mbe: float | None = None,
        total_absolute_mbe: float | None = None,
        relative_oil_mbe: float | None = None,
        relative_water_mbe: float | None = None,
        relative_gas_mbe: float | None = None,
        total_relative_mbe: float | None = None,
        aggressive: bool = False,
    ) -> float:
        """
        Compute the most conservative (smallest) backoff factor from all
        failure signals.  Returns a value in (0, 1].
        """
        factors: list[float] = []

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
        saturation_utilization = _utilization(
            maximum_saturation_change, maximum_allowed_saturation_change
        )
        if saturation_utilization is not None and saturation_utilization > 1.0:
            if saturation_utilization > 3.0:
                factors.append(0.25)
            elif saturation_utilization > 2.0:
                factors.append(0.40)
            else:
                factors.append(max(1.0 / saturation_utilization, 0.50))

        # Pressure change
        pressure_utilization = _utilization(
            maximum_pressure_change, maximum_allowed_pressure_change
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
            u = _utilization(actual, limit)
            if u is not None and u > 1.0:
                if u > 3.0:
                    factors.append(0.40)
                elif u > 1.5:
                    factors.append(0.60)
                else:
                    factors.append(max(0.90 / u, 0.70))

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
        maximum_cfl_encountered: float | None = None,
        cfl_threshold: float | None = None,
        maximum_saturation_change: float | None = None,
        maximum_allowed_saturation_change: float | None = None,
        maximum_pressure_change: float | None = None,
        maximum_allowed_pressure_change: float | None = None,
        absolute_oil_mbe: float | None = None,
        absolute_water_mbe: float | None = None,
        absolute_gas_mbe: float | None = None,
        total_absolute_mbe: float | None = None,
        relative_oil_mbe: float | None = None,
        relative_water_mbe: float | None = None,
        relative_gas_mbe: float | None = None,
        total_relative_mbe: float | None = None,
        **kwargs: typing.Any,
    ) -> tuple[bool, str]:
        """
        Determine whether a completed time step meets all acceptance criteria.

        Each argument is checked against its corresponding configured limit.
        The first violated criterion causes an immediate `(False, message)`
        return; subsequent criteria are not evaluated. When all criteria pass
        `(True, 'Step acceptable')` is returned.

        :param maximum_cfl_encountered: Maximum CFL number observed during the
            step.  Checked against `cfl_threshold` (or `maximum_cfl` when
            *cfl_threshold* is *None*).
        :param cfl_threshold: The CFL threshold that was active during the step.
        :param maximum_saturation_change: Maximum phase saturation change
            encountered (fraction, 0-1).
        :param maximum_allowed_saturation_change: Configured saturation change
            limit.  Both this and *maximum_saturation_change* must be non-*None*
            for the check to be active.
        :param maximum_pressure_change: Maximum absolute pressure change
            encountered (psi).
        :param maximum_allowed_pressure_change: Configured pressure change limit
            (psi).  Both this and *maximum_pressure_change* must be non-*None*
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
        :return: Tuple of `(acceptable, message)`.  *acceptable* is *True*
            when the step passes all checks.  *message* describes the first
            violated criterion, or `'Step acceptable'` on success.
        """
        cfl_limit = cfl_threshold if cfl_threshold is not None else self.maximum_cfl
        if maximum_cfl_encountered is not None and maximum_cfl_encountered > cfl_limit:
            return (
                False,
                f"Maximum CFL ({cfl_limit}) violated. Encountered: {maximum_cfl_encountered}",
            )

        saturation_utilization = _utilization(
            maximum_saturation_change, maximum_allowed_saturation_change
        )
        if saturation_utilization is not None and saturation_utilization > 1.0:
            return (
                False,
                f"Maximum allowed saturation change ({maximum_allowed_saturation_change}) violated. "
                f"Encountered: {maximum_saturation_change}",
            )

        pressure_utilization = _utilization(
            maximum_pressure_change, maximum_allowed_pressure_change
        )
        if pressure_utilization is not None and pressure_utilization > 1.0:
            return (
                False,
                f"Maximum allowed pressure change ({maximum_allowed_pressure_change}) violated. "
                f"Encountered: {maximum_pressure_change}",
            )

        mbe_violated, mbe_messages = self._check_mbe_violations(
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

    def accept_step(
        self,
        step_size: float,
        *,
        maximum_cfl_encountered: float | None = None,
        cfl_threshold: float | None = None,
        newton_iterations: int | None = None,
        maximum_saturation_change: float | None = None,
        maximum_allowed_saturation_change: float | None = None,
        maximum_pressure_change: float | None = None,
        maximum_allowed_pressure_change: float | None = None,
        absolute_oil_mbe: float | None = None,
        absolute_water_mbe: float | None = None,
        absolute_gas_mbe: float | None = None,
        total_absolute_mbe: float | None = None,
        relative_oil_mbe: float | None = None,
        relative_water_mbe: float | None = None,
        relative_gas_mbe: float | None = None,
        total_relative_mbe: float | None = None,
    ) -> float:
        """
        Register an accepted time step and compute the next proposed step size.

        The next step size is derived by collecting adjustment factors from all
        active diagnostic signals, separating them into *limiting* factors
        (≤ 1.0, all applied) and *growth* factors (> 1.0, only the most
        conservative applied). An additional easy-regime acceleration is
        triggered when all utilization metrics are well below their limits and
        the solver has had several consecutive successes, allowing fast geometric
        growth toward `maximum_step_size`. The result is smoothed via an
        exponential moving average controlled by `step_size_smoothing`.

        :param step_size: The time step size that was just accepted (seconds).
        :param maximum_cfl_encountered: Maximum CFL number observed during the
            step.  Used to compute a CFL-proportional growth or limiting factor.
        :param cfl_threshold: The CFL threshold that was active during the step.
            Falls back to `maximum_cfl` when *None*.
        :param newton_iterations: Number of Newton iterations taken by the
            transport solver. High counts produce a limiting factor; very low
            counts (< 4) after a cooldown period allow additional growth.
        :param maximum_saturation_change: Maximum phase saturation change
            encountered during the step (fraction, 0-1).
        :param maximum_allowed_saturation_change: Configured saturation change
            limit used to compute the utilization ratio.
        :param maximum_pressure_change: Maximum absolute pressure change
            encountered during the step (psi).
        :param maximum_allowed_pressure_change: Configured pressure change limit
            (psi) used to compute the utilization ratio.
        :param absolute_oil_mbe: Absolute oil material balance error (res ft³).
            Only used when `use_mbe_for_step_size` is *True*.
        :param absolute_water_mbe: Absolute water material balance error (res ft³).
            Only used when `use_mbe_for_step_size` is *True*.
        :param absolute_gas_mbe: Absolute gas material balance error (res ft³),
            including dissolved gas. Only used when `use_mbe_for_step_size`
            is *True*.
        :param total_absolute_mbe: Sum of all absolute phase MBEs (res ft³).
            Only used when `use_mbe_for_step_size` is *True*.
        :param relative_oil_mbe: Oil MBE as a fraction of the previous-step oil
            pore volume. Only used when `use_mbe_for_step_size` is *True*.
        :param relative_water_mbe: Water MBE as a fraction of the previous-step
            water pore volume. Only used when `use_mbe_for_step_size` is *True*.
        :param relative_gas_mbe: Gas MBE as a fraction of the previous-step gas
            pore volume equivalent. Only used when `use_mbe_for_step_size`
            is *True*.
        :param total_relative_mbe: Total MBE as a fraction of the previous-step
            total pore volume. Only used when `use_mbe_for_step_size` is *True*.
        :return: The next proposed time step size in seconds.
        :raises TimingError: If *step_size* exceeds the remaining simulation time,
            which indicates a bug in the calling time-stepping logic.
        """
        if step_size > (self.time_remaining + 1e-9):
            raise TimingError(
                f"Step size {step_size} exceeds remaining time {self.time_remaining}."
            )

        if self.use_constant_step_size:
            return self.initial_step_size

        # Advance clocks
        self.elapsed_time += step_size
        self.step_size = step_size
        self.last_successful_step_size = step_size
        self.step += 1
        self.steps_since_last_failure += 1

        # Record metrics
        self.recent_metrics.append(
            StepMetrics(
                step_number=self.step,
                step_size=step_size,
                cfl=maximum_cfl_encountered,
                newton_iterations=newton_iterations,
                success=True,
            )
        )

        # Update exact rolling windows
        if maximum_cfl_encountered is not None and maximum_cfl_encountered > 0.0:
            self._cfl_window.append(maximum_cfl_encountered)
        if newton_iterations is not None and newton_iterations > 0:
            self._newton_window.append(newton_iterations)

        # Build a single consolidated growth factor.
        # Strategy: collect one factor per "signal type", then take the
        # *minimum* of the limiting signals and the *maximum* of the growth
        # signals separately, then combine. This is more physically
        # meaningful than chaining multiplications, which compounds noise.
        limiting_factors: list[float] = []  # ≤ 1.0 must honour all
        growth_factors: list[float] = []  # > 1.0 honour the most conservative

        maximum_cfl = cfl_threshold if cfl_threshold is not None else self.maximum_cfl

        # CFL signal
        if maximum_cfl_encountered is not None and maximum_cfl_encountered > 0.0:
            target_cfl = maximum_cfl * self.cfl_safety_margin
            raw_cfl_factor = target_cfl / maximum_cfl_encountered
            cfl_factor = min(raw_cfl_factor, self.maximum_growth_per_step)
            proximity = maximum_cfl_encountered / maximum_cfl
            if proximity > 0.90:
                cfl_factor = min(cfl_factor, 1.00)
            elif proximity > 0.80:
                cfl_factor = min(cfl_factor, 1.10)
            (limiting_factors if cfl_factor < 1.0 else growth_factors).append(cfl_factor)

        # Saturation signal
        saturation_utilization = _utilization(
            maximum_saturation_change, maximum_allowed_saturation_change
        )
        if saturation_utilization is not None and saturation_utilization > 0.0:
            if saturation_utilization > 0.95:
                limiting_factors.append(0.85)
            elif saturation_utilization > 0.85:
                limiting_factors.append(0.95)
            elif saturation_utilization > 0.70:
                growth_factors.append(1.05)
            elif saturation_utilization < 0.30:
                growth_factors.append(min(1.30, (1.0 / saturation_utilization) * 0.80))
            else:
                growth_factors.append(min(1.15, (1.0 / saturation_utilization) * 0.90))

        # Pressure signal
        pressure_utilization = _utilization(
            maximum_pressure_change, maximum_allowed_pressure_change
        )
        if pressure_utilization is not None and pressure_utilization > 0.0:
            if pressure_utilization > 0.95:
                limiting_factors.append(0.85)
            elif pressure_utilization > 0.85:
                limiting_factors.append(0.95)
            elif pressure_utilization > 0.70:
                growth_factors.append(1.05)
            elif pressure_utilization < 0.30:
                growth_factors.append(min(1.30, (1.0 / pressure_utilization) * 0.80))
            else:
                growth_factors.append(min(1.15, (1.0 / pressure_utilization) * 0.90))

        # Newton signal
        if newton_iterations is not None:
            if newton_iterations > 10:
                limiting_factors.append(0.70)
            elif newton_iterations < 4 and self.steps_since_last_failure >= 3:
                growth_factors.append(1.20)

        # Historical performance signal
        perf = self._compute_performance_factor()
        if perf < 1.0:
            limiting_factors.append(perf)

        # MBE signal (optional)
        if self.use_mbe_for_step_size:
            mbe_accept_pairs = [
                (absolute_oil_mbe, self.maximum_absolute_oil_mbe),
                (absolute_water_mbe, self.maximum_absolute_water_mbe),
                (absolute_gas_mbe, self.maximum_absolute_gas_mbe),
                (total_absolute_mbe, self.maximum_total_absolute_mbe),
                (relative_oil_mbe, self.maximum_relative_oil_mbe),
                (relative_water_mbe, self.maximum_relative_water_mbe),
                (relative_gas_mbe, self.maximum_relative_gas_mbe),
                (total_relative_mbe, self.maximum_total_relative_mbe),
            ]
            for actual, limit in mbe_accept_pairs:
                u = _utilization(actual, limit)
                if u is None:
                    continue
                if u > 0.90:
                    limiting_factors.append(0.85)
                elif u > 0.75:
                    limiting_factors.append(0.95)
                elif u < 0.20:
                    growth_factors.append(min(1.20, (1.0 / max(u, 1e-9)) * 0.50))

        # Combine: all limiting factors must apply; of growth factors only
        # the most conservative (smallest) is used so we don't overshoot.
        dt = self.next_step_size

        # Apply limiting factors (all must apply)
        for f in limiting_factors:
            dt *= f

        # Apply the most conservative growth factor only (avoid compounding)
        if growth_factors and not limiting_factors:
            dt *= min(growth_factors)

        # Ramp-up (only when limits are comfortably met)
        can_ramp = (
            self.ramp_up_factor is not None
            and not self.last_step_failed
            and self.steps_since_last_failure >= self.growth_cooldown_steps
        )
        if can_ramp:
            all_comfy = all(
                u is None or u < 0.70
                for u in (
                    _utilization(maximum_cfl_encountered, maximum_cfl),
                    saturation_utilization,
                    pressure_utilization,
                )
            )
            if all_comfy:
                dt *= self.ramp_up_factor  # type: ignore

        # Easy-regime accelerated growth
        # When *all* utilisation metrics are well below their limits and we
        # have had several consecutive successes, grow more aggressively
        # toward maximum_step_size instead of being held back by the default
        # maximum_growth_per_step cap. This significantly reduces the number
        # of steps in "easy" parts of the simulation.
        easy_regime = (
            self.steps_since_last_failure >= self.growth_cooldown_steps
            and not limiting_factors
            and all(
                u is None or u < 0.30
                for u in (
                    _utilization(maximum_cfl_encountered, maximum_cfl),
                    saturation_utilization,
                    pressure_utilization,
                )
            )
        )
        if easy_regime:
            # Geometric approach toward maximum_step_size
            gap = self.maximum_step_size - dt
            if gap > 0.0:
                dt += gap * 0.50  # close half the gap each step
                logger.debug("Easy regime: geometric growth toward max step size -> %.6e", dt)

        # Hard caps
        dt = min(dt, self.maximum_step_size)
        if not easy_regime:
            # Normal regime: honour growth-rate cap
            dt = min(dt, self.step_size * self.maximum_growth_per_step)
        dt = max(dt, self.minimum_step_size)

        # Failure-zone avoidance
        if self._is_near_failed_size(dt):
            dt *= 0.85
            dt = max(dt, self.minimum_step_size)
            logger.debug("Failure-zone avoidance: reduced proposed size to %.6e", dt)

        # EMA smoothing
        # Formula:  ema = (1 - α) * ema_prev + α * dt
        # α = step_size_smoothing; 0 -> no change (pure old EMA),  1 -> no smoothing.
        # At α = 0.2 the response weight is 20 % new / 80 % history (smooth).
        # At α = 1.0 the EMA equals the raw proposal (no smoothing).
        if self.step == 1:
            self.ema_step_size = dt
        else:
            alpha = self.step_size_smoothing
            self.ema_step_size = (1.0 - alpha) * self.ema_step_size + alpha * dt

        self.next_step_size = self.ema_step_size

        # Reset rejection tracking
        self.last_step_failed = False
        self.rejection_count = 0

        logger.debug(
            "Step %.6e accepted (step %d, elapsed %.4fs). Next proposed: %.6e",
            step_size,
            self.step,
            self.elapsed_time,
            self.next_step_size,
        )
        return self.next_step_size

    def dump_state(self) -> TimerState:
        """Serialize the current timer state to a dictionary."""
        return {
            "initial_step_size": self.initial_step_size,
            "maximum_step_size": self.maximum_step_size,
            "minimum_step_size": self.minimum_step_size,
            "simulation_time": self.simulation_time,
            "maximum_cfl": self.maximum_cfl,
            "cfl_safety_margin": self.cfl_safety_margin,
            "ramp_up_factor": self.ramp_up_factor,
            "backoff_factor": self.backoff_factor,
            "aggressive_backoff_factor": self.aggressive_backoff_factor,
            "maximum_steps": self.maximum_steps,
            "maximum_absolute_oil_mbe": self.maximum_absolute_oil_mbe,
            "maximum_absolute_water_mbe": self.maximum_absolute_water_mbe,
            "maximum_absolute_gas_mbe": self.maximum_absolute_gas_mbe,
            "maximum_total_absolute_mbe": self.maximum_total_absolute_mbe,
            "maximum_relative_oil_mbe": self.maximum_relative_oil_mbe,
            "maximum_relative_water_mbe": self.maximum_relative_water_mbe,
            "maximum_relative_gas_mbe": self.maximum_relative_gas_mbe,
            "maximum_total_relative_mbe": self.maximum_total_relative_mbe,
            "use_mbe_for_step_size": self.use_mbe_for_step_size,
            "maximum_growth_per_step": self.maximum_growth_per_step,
            "maximum_rejections": self.maximum_rejections,
            "step_size_smoothing": self.step_size_smoothing,
            "growth_cooldown_steps": self.growth_cooldown_steps,
            "failure_memory_window": self.failure_memory_window,
            "metrics_history_size": self.metrics_history_size,
            "use_constant_step_size": self.use_constant_step_size,
            "elapsed_time": self.elapsed_time,
            "step": self.step,
            "step_size": self.step_size,
            "next_step_size": self.next_step_size,
            "ema_step_size": self.ema_step_size,
            "last_step_failed": self.last_step_failed,
            "rejection_count": self.rejection_count,
            "steps_since_last_failure": self.steps_since_last_failure,
            "last_successful_step_size": self.last_successful_step_size,
            "recent_metrics": [
                typing.cast(StepMetricsDict, attrs.asdict(m)) for m in self.recent_metrics
            ],
            "failed_step_sizes": list(self.failed_step_sizes),
        }

    @classmethod
    def load_state(cls, state: TimerState) -> Self:
        """Reconstruct a timer from a previously saved state dictionary"""
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

        params = {
            "initial_step_size": state["initial_step_size"],
            "maximum_step_size": state["maximum_step_size"],
            "minimum_step_size": state["minimum_step_size"],
            "simulation_time": state["simulation_time"],
            "maximum_cfl": state.get("maximum_cfl", 1.0),
            "cfl_safety_margin": state.get("cfl_safety_margin", 0.9),
            "ramp_up_factor": state.get("ramp_up_factor"),
            "backoff_factor": state.get("backoff_factor", 0.5),
            "aggressive_backoff_factor": state.get("aggressive_backoff_factor", 0.25),
            "maximum_steps": state.get("maximum_steps"),
            "maximum_absolute_oil_mbe": state.get("maximum_absolute_oil_mbe"),
            "maximum_absolute_water_mbe": state.get("maximum_absolute_water_mbe"),
            "maximum_absolute_gas_mbe": state.get("maximum_absolute_gas_mbe"),
            "maximum_total_absolute_mbe": state.get("maximum_total_absolute_mbe"),
            "maximum_relative_oil_mbe": state.get("maximum_relative_oil_mbe"),
            "maximum_relative_water_mbe": state.get("maximum_relative_water_mbe"),
            "maximum_relative_gas_mbe": state.get("maximum_relative_gas_mbe"),
            "maximum_total_relative_mbe": state.get("maximum_total_relative_mbe"),
            "use_mbe_for_step_size": state.get("use_mbe_for_step_size", False),
            "maximum_rejections": state.get("maximum_rejections", 10),
            "maximum_growth_per_step": state.get("maximum_growth_per_step", 1.3),
            "step_size_smoothing": state.get("step_size_smoothing", 0.2),
            "growth_cooldown_steps": state.get("growth_cooldown_steps", 5),
            "failure_memory_window": state.get("failure_memory_window", 5),
            "metrics_history_size": state.get("metrics_history_size", 10),
        }
        timer = cls(**params)

        for attr, key, default in [
            ("elapsed_time", "elapsed_time", 0.0),
            ("step", "step", 0),
            ("step_size", "step_size", 0.0),
            ("next_step_size", "next_step_size", state["step_size"]),
            ("ema_step_size", "ema_step_size", state["step_size"]),
            ("last_step_failed", "last_step_failed", False),
            ("rejection_count", "rejection_count", 0),
            ("steps_since_last_failure", "steps_since_last_failure", 0),
            (
                "last_successful_step_size",
                "last_successful_step_size",
                state["step_size"],
            ),
        ]:
            object.__setattr__(timer, attr, state.get(key, default))

        recent_metrics = deque(
            [StepMetrics(**m) for m in state.get("recent_metrics", [])],
            maxlen=timer.metrics_history_size,
        )
        object.__setattr__(timer, "recent_metrics", recent_metrics)

        failed_step_sizes = deque(
            state.get("failed_step_sizes", []),
            maxlen=timer.failure_memory_window,
        )
        object.__setattr__(timer, "failed_step_sizes", failed_step_sizes)

        # Rebuild rolling windows from recent_metrics history
        cfl_window: deque = deque(maxlen=timer.metrics_history_size)
        newton_window: deque = deque(maxlen=timer.metrics_history_size)
        for m in recent_metrics:
            if m.success:
                if m.cfl is not None and m.cfl > 0.0:
                    cfl_window.append(m.cfl)
                if m.newton_iterations is not None and m.newton_iterations > 0:
                    newton_window.append(m.newton_iterations)
        object.__setattr__(timer, "_cfl_window", cfl_window)
        object.__setattr__(timer, "_newton_window", newton_window)

        logger.debug(
            "Timer state loaded: step %d, elapsed %.4fs, step_size %.6e",
            timer.step,
            timer.elapsed_time,
            timer.step_size,
        )
        return timer

    def __dump__(self) -> dict[str, typing.Any]:
        return typing.cast(dict[str, typing.Any], self.dump_state())

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        return cls.load_state(typing.cast(TimerState, data))
