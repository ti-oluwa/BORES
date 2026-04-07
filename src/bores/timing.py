"""Simulation time management with smart and adaptive time stepping."""

import logging
import typing
from collections import deque
from datetime import timedelta

import attrs
from typing_extensions import Self, TypedDict

from bores.constants import c
from bores.errors import TimingError, ValidationError
from bores.stores import StoreSerializable

__all__ = ["Time", "Timer", "TimerState"]

logger = logging.getLogger(__name__)


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
    cfl: typing.Optional[float]
    newton_iterations: typing.Optional[int]
    success: bool


class TimerState(TypedDict):
    """Complete state of a timer instance for serialization."""

    initial_step_size: float
    maximum_step_size: float
    minimum_step_size: float
    simulation_time: float
    maximum_cfl_number: float
    cfl_safety_margin: float
    ramp_up_factor: typing.Optional[float]
    backoff_factor: float
    aggressive_backoff_factor: float
    maximum_steps: typing.Optional[int]
    maximum_absolute_oil_mbe: typing.Optional[float]
    maximum_absolute_water_mbe: typing.Optional[float]
    maximum_absolute_gas_mbe: typing.Optional[float]
    maximum_total_absolute_mbe: typing.Optional[float]
    maximum_relative_oil_mbe: typing.Optional[float]
    maximum_relative_water_mbe: typing.Optional[float]
    maximum_relative_gas_mbe: typing.Optional[float]
    maximum_total_relative_mbe: typing.Optional[float]
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
    recent_metrics: typing.List[StepMetricsDict]
    failed_step_sizes: typing.List[float]


@attrs.frozen(slots=True)
class StepMetrics:
    """Metrics for a single time step."""

    step_number: int
    step_size: float
    cfl: typing.Optional[float] = None
    newton_iterations: typing.Optional[int] = None
    success: bool = True


@attrs.define
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
    maximum_cfl_number: float = 0.9
    """Default maximum CFL number for time step adjustments."""
    ramp_up_factor: typing.Optional[float] = None
    """Factor by which to ramp up time step size on successful steps."""
    backoff_factor: float = 0.5
    """Factor by which to reduce time step size on failed steps."""
    aggressive_backoff_factor: float = 0.25
    """Factor by which to aggressively reduce time step size on failed steps."""
    maximum_steps: typing.Optional[int] = None
    """Maximum number of time steps to run for."""

    # Adaptive parameters
    growth_cooldown_steps: int = 5
    """Minimum successful steps required before allowing aggressive growth. Higher values lead to more conservative growth."""
    maximum_growth_per_step: float = 1.3
    """Maximum multiplicative growth allowed per step (e.g., 1.3 = 30% max growth). Lower values lead to much smoother growth."""
    cfl_safety_margin: float = 0.85
    """Safety factor for CFL-based adjustments (target below max CFL)."""
    step_size_smoothing: float = 0.2
    """EMA smoothing factor (0 = no smoothing, 1 = maximum smoothing). Higher values lead to smoother step size changes and higher dampening of fluctuations."""
    metrics_history_size: int = 10
    """Number of recent steps to track for performance analysis."""
    failure_memory_window: int = 5
    """Number of recent failures to remember for adaptive behavior."""

    # MBE-based step control (all optional)
    maximum_absolute_oil_mbe: typing.Optional[float] = None
    """Absolute oil MBE threshold (res ft³). Reject step if exceeded."""
    maximum_absolute_water_mbe: typing.Optional[float] = None
    """Absolute water MBE threshold (res ft³). Reject step if exceeded."""
    maximum_absolute_gas_mbe: typing.Optional[float] = None
    """Absolute gas MBE threshold (res ft³). Reject step if exceeded."""
    maximum_total_absolute_mbe: typing.Optional[float] = None
    """Total absolute MBE threshold (res ft³). Reject step if exceeded."""
    maximum_relative_oil_mbe: typing.Optional[float] = None
    """Relative oil MBE threshold (fraction, e.g. 0.01 = 1%). Reject step if exceeded."""
    maximum_relative_water_mbe: typing.Optional[float] = None
    """Relative water MBE threshold (fraction). Reject step if exceeded."""
    maximum_relative_gas_mbe: typing.Optional[float] = None
    """Relative gas MBE threshold (fraction). Reject step if exceeded."""
    maximum_total_relative_mbe: typing.Optional[float] = None
    """Total relative MBE threshold (fraction). Reject step if exceeded."""
    use_mbe_for_step_size: bool = False
    """
    If True, MBE values are used to modulate the next step size proposal even
    when the step is accepted (analogous to how CFL and saturation change drive
    growth/shrinkage on accepted steps). When False (default), MBE only
    triggers step rejection when a configured limit is exceeded; it has no
    influence on step-size growth on passing steps.
    """

    # State variables
    elapsed_time: float = attrs.field(init=False, default=0.0)
    """Current simulation time in seconds (sum of all accepted steps)."""
    step_size: float = attrs.field(init=False, default=0.0)
    """The time step size (in seconds) that was used for the most recently accepted step."""
    next_step_size: float = attrs.field(init=False, default=0.0)
    """Time step size (in seconds) to propose for the next step."""
    ema_step_size: float = attrs.field(init=False, default=0.0)
    """Exponential moving average of step size for smoothing."""
    step: int = attrs.field(init=False, default=0)
    """Number of accepted time steps completed so far (0-indexed)."""
    last_step_failed: bool = attrs.field(init=False, default=False)
    """Whether the most recent step attempt was rejected."""
    maximum_rejections: int = 10
    """Maximum number of consecutive time step rejections allowed."""
    rejection_count: int = attrs.field(init=False, default=0)
    """Count of consecutive time step rejections."""
    steps_since_last_failure: int = attrs.field(init=False, default=0)
    """Number of successful steps since the last failure."""
    use_constant_step_size: bool = attrs.field(init=False, default=False)
    """Whether to use a constant time step size."""

    # Performance tracking
    recent_metrics: typing.Deque[StepMetrics] = attrs.field(init=False)
    """Recent step performance metrics."""
    failed_step_sizes: typing.Deque[float] = attrs.field(init=False)
    """Recent failed step sizes for memory."""

    def __attrs_post_init__(self) -> None:
        self.next_step_size = self.initial_step_size
        self.step_size = self.initial_step_size
        self.ema_step_size = self.initial_step_size
        self.use_constant_step_size = (
            self.initial_step_size == self.maximum_step_size
            and self.initial_step_size == self.minimum_step_size
        )
        self.recent_metrics = deque(maxlen=self.metrics_history_size)
        self.failed_step_sizes = deque(maxlen=self.failure_memory_window)

    @property
    def next_step(self) -> int:
        """Returns the next time step count."""
        return self.step + 1

    def done(self) -> bool:
        """
        Checks if the simulation has reached its end criteria.

        If True, simulation has reached it end.
        """
        # Use small tolerance for floating-point comparison
        if self.elapsed_time >= (self.simulation_time - 1e-9):
            return True
        return self.maximum_steps is not None and self.step >= self.maximum_steps

    @property
    def time_remaining(self) -> float:
        """Calculates the remaining simulation time in seconds."""
        return max(self.simulation_time - self.elapsed_time, 0.0)

    @property
    def is_last_step(self) -> bool:
        """Determines if the latest accepted step was the last one (simulation is now complete)."""
        if self.time_remaining <= 0:
            return True

        return self.maximum_steps is not None and self.step >= self.maximum_steps

    def _is_near_failed_size(self, dt: float, tolerance: float = 0.15) -> bool:
        """Check if proposed step size is near a recently failed size."""
        for failed_size in self.failed_step_sizes:
            if abs(dt - failed_size) / failed_size < tolerance:
                return True
        return False

    def _compute_performance_factor(self) -> float:
        """
        Analyze recent performance metrics to compute an adaptive factor.

        Returns a factor in (0, 1] where:
        - 1.0 = excellent performance, allow normal growth
        - <1.0 = concerning trends, be more conservative
        """
        if len(self.recent_metrics) < 3:
            return 1.0

        factor = 1.0

        # Check CFL trend (are we pushing limits?)
        recent_cfls = [
            m.cfl
            for m in list(self.recent_metrics)[-5:]
            if m.cfl is not None and m.success
        ]
        if len(recent_cfls) >= 3:
            # If CFL is consistently high or increasing, be conservative
            avg_cfl = sum(recent_cfls) / len(recent_cfls)
            if avg_cfl > 0.75 * self.maximum_cfl_number:
                factor *= 0.95

            # Check if CFL is trending upward
            if len(recent_cfls) >= 4:
                cfl_trend = recent_cfls[-1] - recent_cfls[-4]
                if cfl_trend > 0.15:
                    factor *= 0.9

        # Check Newton iteration trends (is solver struggling?)
        recent_iterations = [
            m.newton_iterations
            for m in list(self.recent_metrics)[-5:]
            if m.newton_iterations is not None and m.success
        ]
        if len(recent_iterations) >= 3:
            avg_iterations = sum(recent_iterations) / len(recent_iterations)
            if avg_iterations > 8:
                factor *= 0.85
            elif all(i > 10 for i in recent_iterations[-3:]):
                factor *= 0.75  # Solver consistently struggling

        return max(factor, 0.5)  # We don't want to be too aggressive in reduction

    def propose_step_size(self) -> float:
        """Proposes the next time step size without updating state."""
        if self.use_constant_step_size:
            return self.initial_step_size

        dt = self.next_step_size
        remaining_time = self.time_remaining
        if dt > remaining_time:
            return (
                max(remaining_time, self.minimum_step_size)
                if remaining_time >= self.minimum_step_size
                else remaining_time
            )

        logger.debug(
            f"Proposing time step of size {dt} for time step {self.next_step} "
            f"at elapsed time {self.elapsed_time}."
        )
        return dt

    def reject_step(
        self,
        step_size: float,
        *,
        aggressive: bool = False,
        maximum_cfl_encountered: typing.Optional[float] = None,
        cfl_threshold: typing.Optional[float] = None,
        newton_iterations: typing.Optional[int] = None,
        maximum_saturation_change: typing.Optional[float] = None,
        maximum_allowed_saturation_change: typing.Optional[float] = None,
        maximum_pressure_change: typing.Optional[float] = None,
        maximum_allowed_pressure_change: typing.Optional[float] = None,
        absolute_oil_mbe: typing.Optional[float] = None,
        absolute_water_mbe: typing.Optional[float] = None,
        absolute_gas_mbe: typing.Optional[float] = None,
        total_absolute_mbe: typing.Optional[float] = None,
        relative_oil_mbe: typing.Optional[float] = None,
        relative_water_mbe: typing.Optional[float] = None,
        relative_gas_mbe: typing.Optional[float] = None,
        total_relative_mbe: typing.Optional[float] = None,
    ) -> float:
        """
        Registers a rejected time step proposal and computes an intelligently adjusted time step size
        based on the specific failure criteria encountered.

        :param step_size: The step size that was rejected.
        :param aggressive: Whether to use aggressive backoff (fallback if no specific info provided).
        :param maximum_cfl_encountered: The maximum CFL number that caused rejection.
        :param cfl_threshold: The CFL threshold that was exceeded.
        :param newton_iterations: Number of Newton iterations attempted before failure.
        :param maximum_saturation_change: The maximum saturation change encountered.
        :param maximum_allowed_saturation_change: The allowed saturation change threshold.
        :param maximum_pressure_change: The maximum pressure change encountered.
        :param maximum_allowed_pressure_change: The allowed pressure change threshold.
        :param absolute_oil_mbe: Absolute oil material balance error for this step (res ft³).
        :param absolute_water_mbe: Absolute water material balance error for this step (res ft³).
        :param absolute_gas_mbe: Absolute gas material balance error for this step (res ft³), including dissolved gas.
        :param total_absolute_mbe: Sum of absolute phase MBEs (res ft³).
        :param relative_oil_mbe: Oil MBE as a fraction of the previous-step oil pore volume.
        :param relative_water_mbe: Water MBE as a fraction of the previous-step water pore volume.
        :param relative_gas_mbe: Gas MBE as a fraction of the previous-step gas pore volume equivalent.
        :param total_relative_mbe: Total MBE as a fraction of the previous-step total pore volume.
        :return: The new/adjusted time step size in seconds.
        """
        if self.rejection_count >= self.maximum_rejections:
            raise TimingError(
                "Maximum number of consecutive time step rejections exceeded"
            )

        # Warn when approaching rejection limit
        rejection_threshold = int(0.7 * self.maximum_rejections)
        if (
            self.rejection_count >= rejection_threshold
            and self.rejection_count < self.maximum_rejections
        ):
            logger.warning(
                f"Time step rejection count ({self.rejection_count}) is approaching "
                f"maximum allowed ({self.maximum_rejections}). Consider adjusting simulation "
                f"parameters or initial conditions."
            )

        if self.use_constant_step_size:
            return self.initial_step_size

        # Store failed step size for memory
        self.failed_step_sizes.append(step_size)

        # Record metrics
        metrics = StepMetrics(
            step_number=self.next_step,
            step_size=step_size,
            cfl=maximum_cfl_encountered,
            newton_iterations=newton_iterations,
            success=False,
        )
        self.recent_metrics.append(metrics)

        # Compute backoff based on failure cause
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
        self.next_step_size *= factor

        # Warn when hitting minimum step size
        if self.next_step_size < self.minimum_step_size:
            logger.warning(
                f"Step size {self.next_step_size:.6e} would be below minimum "
                f"{self.minimum_step_size:.6e}. Clamping to minimum."
            )
        self.next_step_size = max(self.next_step_size, self.minimum_step_size)

        # Check if we're stuck at minimum step size (panic mode detection)
        if self.next_step_size <= self.minimum_step_size * 1.01:
            min_failures = sum(
                1 for s in self.failed_step_sizes if s <= self.minimum_step_size * 1.1
            )
            if min_failures >= 3:
                logger.error(
                    f"Repeated failures ({min_failures}) at or near minimum step size "
                    f"({self.minimum_step_size:.6e}). Simulation may be unstable."
                )

        # Update EMA to reflect the reduction
        self.ema_step_size = self.next_step_size
        self.last_step_failed = True
        self.rejection_count += 1
        self.steps_since_last_failure = 0

        logger.debug(
            f"Time step of size {step_size} rejected for time step {self.next_step} "
            f"at elapsed time {self.elapsed_time}. Backoff factor: {factor:.3f}, "
            f"New size: {self.next_step_size:.6e}"
        )
        return self.next_step_size

    def _compute_backoff_factor(
        self,
        *,
        maximum_cfl_encountered: typing.Optional[float] = None,
        cfl_threshold: typing.Optional[float] = None,
        newton_iterations: typing.Optional[int] = None,
        maximum_saturation_change: typing.Optional[float] = None,
        maximum_allowed_saturation_change: typing.Optional[float] = None,
        maximum_pressure_change: typing.Optional[float] = None,
        maximum_allowed_pressure_change: typing.Optional[float] = None,
        absolute_oil_mbe: typing.Optional[float] = None,
        absolute_water_mbe: typing.Optional[float] = None,
        absolute_gas_mbe: typing.Optional[float] = None,
        total_absolute_mbe: typing.Optional[float] = None,
        relative_oil_mbe: typing.Optional[float] = None,
        relative_water_mbe: typing.Optional[float] = None,
        relative_gas_mbe: typing.Optional[float] = None,
        total_relative_mbe: typing.Optional[float] = None,
        aggressive: bool = False,
    ) -> float:
        """
        Compute an intelligent backoff factor based on the specific failure criteria.

        Returns a factor in (0, 1] to multiply the step size by.
        Smaller factors mean more aggressive reduction.
        """
        factors = []

        # CFL-based backoff
        if maximum_cfl_encountered is not None and cfl_threshold is not None:
            cfl_limit = cfl_threshold if cfl_threshold > 0 else self.maximum_cfl_number
            if maximum_cfl_encountered > cfl_limit:
                # Proportional backoff based on how much we exceeded the limit
                overshoot_ratio = maximum_cfl_encountered / cfl_limit

                if overshoot_ratio > 2.0:
                    # Severe CFL violation
                    cfl_factor = 0.3
                    logger.debug(
                        f"Severe CFL violation: {maximum_cfl_encountered:.3f} > {cfl_limit:.3f} (ratio: {overshoot_ratio:.4f})"
                    )
                elif overshoot_ratio > 1.5:
                    # Moderate CFL violation
                    cfl_factor = 0.5
                    logger.debug(
                        f"Moderate CFL violation: {maximum_cfl_encountered:.3f} > {cfl_limit:.3f} (ratio: {overshoot_ratio:.4f})"
                    )
                else:
                    # Mild CFL violation. Try to target the limit
                    cfl_factor = (cfl_limit * 0.9) / maximum_cfl_encountered
                    cfl_factor = max(cfl_factor, 0.6)  # Don't reduce too much
                    logger.debug(
                        f"Mild CFL violation: {maximum_cfl_encountered:.3f} > {cfl_limit:.3f} (ratio: {overshoot_ratio:.4f})"
                    )
                factors.append(cfl_factor)

        # Saturation change backoff
        if (
            maximum_saturation_change is not None
            and maximum_allowed_saturation_change is not None
            and maximum_saturation_change > maximum_allowed_saturation_change
        ):
            overshoot_ratio = (
                maximum_saturation_change / maximum_allowed_saturation_change
            )

            if overshoot_ratio > 3.0:
                # Very large saturation changes. Apply aggressive reduction
                saturation_factor = 0.25
                logger.debug(
                    f"Severe saturation change: {maximum_saturation_change:.4f} > {maximum_allowed_saturation_change:.4f} (ratio: {overshoot_ratio:.4f})"
                )
            elif overshoot_ratio > 2.0:
                # Large saturation changes
                saturation_factor = 0.4
                logger.debug(
                    f"Large saturation change: {maximum_saturation_change:.4f} > {maximum_allowed_saturation_change:.4f} (ratio: {overshoot_ratio:.4f})"
                )
            else:
                # Moderate overshoot. Apply proportional reduction
                saturation_factor = (
                    maximum_allowed_saturation_change / maximum_saturation_change
                )
                saturation_factor = max(saturation_factor, 0.5)  # Don't reduce too much
                logger.debug(
                    f"Moderate saturation change: {maximum_saturation_change:.4f} > {maximum_allowed_saturation_change:.4f} (ratio: {overshoot_ratio:.4f})"
                )
            factors.append(saturation_factor)

        # Pressure change backoff
        if (
            maximum_pressure_change is not None
            and maximum_allowed_pressure_change is not None
            and maximum_pressure_change > maximum_allowed_pressure_change
        ):
            overshoot_ratio = maximum_pressure_change / maximum_allowed_pressure_change

            if overshoot_ratio > 3.0:
                # Very large pressure changes
                pressure_factor = 0.25
                logger.debug(
                    f"Severe pressure change: {maximum_pressure_change:.4e} > {maximum_allowed_pressure_change:.4e} (ratio: {overshoot_ratio:.4f})"
                )
            elif overshoot_ratio > 2.0:
                # Large pressure changes
                pressure_factor = 0.4
                logger.debug(
                    f"Large pressure change: {maximum_pressure_change:.4e} > {maximum_allowed_pressure_change:.4e} (ratio: {overshoot_ratio:.4f})"
                )
            else:
                # Moderate overshoot
                pressure_factor = (
                    maximum_allowed_pressure_change / maximum_pressure_change
                )
                pressure_factor = max(pressure_factor, 0.5)
                logger.debug(
                    f"Moderate pressure change: {maximum_pressure_change:.4e} > {maximum_allowed_pressure_change:.4e} (ratio: {overshoot_ratio:.4f})"
                )

            factors.append(pressure_factor)

        # Newton iteration failure backoff
        if newton_iterations is not None:
            if newton_iterations > 20:
                # Solver really struggling. Apply aggressive reduction
                newton_iteration_factor = 0.3
                logger.debug(
                    f"Newton solver struggling severely: {newton_iterations} iterations"
                )
                factors.append(newton_iteration_factor)
            elif newton_iterations > 15:
                # Solver struggling. Apply moderate reduction
                newton_iteration_factor = 0.5
                logger.debug(
                    f"Newton solver struggling: {newton_iterations} iterations"
                )
                factors.append(newton_iteration_factor)
            elif newton_iterations > 10:
                # Solver having some difficulty
                newton_iteration_factor = 0.7
                logger.debug(
                    f"Newton solver having difficulty: {newton_iterations} iterations"
                )
                factors.append(newton_iteration_factor)

        # MBE-based backoff
        mbe_pairs = [
            (absolute_oil_mbe, self.maximum_absolute_oil_mbe, "absolute oil MBE"),
            (absolute_water_mbe, self.maximum_absolute_water_mbe, "absolute water MBE"),
            (absolute_gas_mbe, self.maximum_absolute_gas_mbe, "absolute gas MBE"),
            (total_absolute_mbe, self.maximum_total_absolute_mbe, "total absolute MBE"),
            (relative_oil_mbe, self.maximum_relative_oil_mbe, "relative oil MBE"),
            (relative_water_mbe, self.maximum_relative_water_mbe, "relative water MBE"),
            (relative_gas_mbe, self.maximum_relative_gas_mbe, "relative gas MBE"),
            (total_relative_mbe, self.maximum_total_relative_mbe, "total relative MBE"),
        ]
        for actual, limit, label in mbe_pairs:
            if actual is None or limit is None:
                continue
            actual_abs = abs(actual)
            if actual_abs > limit:
                overshoot = actual_abs / limit
                if overshoot > 3.0:
                    mbe_factor = 0.4
                elif overshoot > 1.5:
                    mbe_factor = 0.6
                else:
                    mbe_factor = max(limit / actual_abs * 0.9, 0.7)
                logger.debug(
                    f"MBE violation ({label}): |{actual:.3e}| > {limit:.3e} "
                    f"(ratio {overshoot:.2f}), factor={mbe_factor:.3f}"
                )
                factors.append(mbe_factor)

        # If we have specific information, use the most conservative (smallest) factor
        if factors:
            final_factor = min(factors)
            logger.debug(
                f"Computed backoff factor: {final_factor:.3f} from {len(factors)} criteria"
            )
            return final_factor

        # Fallback to original behavior if no specific information provided
        logger.debug("No specific failure information provided, using fallback backoff")
        return self.aggressive_backoff_factor if aggressive else self.backoff_factor

    def is_acceptable(
        self,
        *,
        maximum_cfl_encountered: typing.Optional[float] = None,
        cfl_threshold: typing.Optional[float] = None,
        maximum_saturation_change: typing.Optional[float] = None,
        maximum_allowed_saturation_change: typing.Optional[float] = None,
        maximum_pressure_change: typing.Optional[float] = None,
        maximum_allowed_pressure_change: typing.Optional[float] = None,
        absolute_oil_mbe: typing.Optional[float] = None,
        absolute_water_mbe: typing.Optional[float] = None,
        absolute_gas_mbe: typing.Optional[float] = None,
        total_absolute_mbe: typing.Optional[float] = None,
        relative_oil_mbe: typing.Optional[float] = None,
        relative_water_mbe: typing.Optional[float] = None,
        relative_gas_mbe: typing.Optional[float] = None,
        total_relative_mbe: typing.Optional[float] = None,
        **kwargs: typing.Any,  # Just to ensure that any other kwargs passed dont cause on error (based on usage)
    ) -> typing.Tuple[bool, str]:
        """
        Determine if a time step is acceptable based on given criteria

        :param maximum_cfl_encountered: The maximum CFL number encountered during the step.
        :param cfl_threshold: The CFL threshold used during the step.
        :param maximum_saturation_change: Maximum saturation change in the accepted step.
        :param maximum_allowed_saturation_change: Maximum allowed saturation change threshold.
        :param maximum_pressure_change: Maximum pressure change in the accepted step.
        :param maximum_allowed_pressure_change: Maximum allowed pressure change threshold.
        :param absolute_oil_mbe: Absolute oil MBE to check against `maximum_absolute_oil_mbe` (res ft³).
        :param absolute_water_mbe: Absolute water MBE to check against `maximum_absolute_water_mbe` (res ft³).
        :param absolute_gas_mbe: Absolute gas MBE to check against `maximum_absolute_gas_mbe` (res ft³).
        :param total_absolute_mbe: Total absolute MBE to check against `maximum_total_absolute_mbe` (res ft³).
        :param relative_oil_mbe: Relative oil MBE fraction to check against `maximum_relative_oil_mbe`.
        :param relative_water_mbe: Relative water MBE fraction to check against `maximum_relative_water_mbe`.
        :param relative_gas_mbe: Relative gas MBE fraction to check against `maximum_relative_gas_mbe`.
        :param total_relative_mbe: Total relative MBE fraction to check against `maximum_total_relative_mbe`.
        :param kwargs: Additional timer kwargs forwarded from `StepResult.timer_kwargs`; ignored.
        :return: Tuple of (acceptable, error/message). The first item is True if the timestep is acceptable. Else, Flase.
        """
        cfl_limit = (
            cfl_threshold if cfl_threshold is not None else self.maximum_cfl_number
        )
        if maximum_cfl_encountered is not None and maximum_cfl_encountered > cfl_limit:
            return (
                False,
                f"Maximum CFL limit ({cfl_limit}) violated. Maximum CFL encountered is {maximum_cfl_encountered}",
            )

        if (
            maximum_saturation_change is not None
            and maximum_allowed_saturation_change is not None
            and maximum_saturation_change > maximum_allowed_saturation_change
        ):
            return (
                False,
                f"Maximum allowed saturation change ({maximum_allowed_saturation_change}) violated. "
                "Maximum saturation change encountered is {maximum_saturation_change}",
            )

        if (
            maximum_pressure_change is not None
            and maximum_allowed_pressure_change is not None
            and maximum_pressure_change > maximum_allowed_pressure_change
        ):
            return (
                False,
                f"Maximum allowed saturation change ({maximum_allowed_saturation_change}) violated. "
                "Maximum saturation change encountered is {maximum_saturation_change}",
            )

        mbe_violated = False
        mbe_message_parts = []
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
        for actual, limit, label in mbe_checks:
            if limit is None or actual is None:
                continue
            if actual > limit:
                mbe_violated = True
                mbe_message_parts.append(f"{label}: |{actual:.3e}| > {limit:.3e}")

        if mbe_violated:
            message = f"MBE limits violated: " + "; ".join(mbe_message_parts)
            return True, message

        return True, "Step acceptable"

    def accept_step(
        self,
        step_size: float,
        *,
        maximum_cfl_encountered: typing.Optional[float] = None,
        cfl_threshold: typing.Optional[float] = None,
        newton_iterations: typing.Optional[int] = None,
        maximum_saturation_change: typing.Optional[float] = None,
        maximum_allowed_saturation_change: typing.Optional[float] = None,
        maximum_pressure_change: typing.Optional[float] = None,
        maximum_allowed_pressure_change: typing.Optional[float] = None,
        absolute_oil_mbe: typing.Optional[float] = None,
        absolute_water_mbe: typing.Optional[float] = None,
        absolute_gas_mbe: typing.Optional[float] = None,
        total_absolute_mbe: typing.Optional[float] = None,
        relative_oil_mbe: typing.Optional[float] = None,
        relative_water_mbe: typing.Optional[float] = None,
        relative_gas_mbe: typing.Optional[float] = None,
        total_relative_mbe: typing.Optional[float] = None,
    ) -> float:
        """
        Registers an accepted time step and computes the next time step size
        based on criteria from the accepted step.

        :param step_size: The time step size that was just accepted.
        :param maximum_cfl_encountered: The maximum CFL number encountered during the step.
        :param cfl_threshold: The CFL threshold used during the step.
        :param newton_iterations: Number of Newton iterations taken (if applicable).
        :param maximum_saturation_change: Maximum saturation change in the accepted step.
        :param maximum_allowed_saturation_change: Maximum allowed saturation change threshold.
        :param maximum_pressure_change: Maximum pressure change in the accepted step.
        :param maximum_allowed_pressure_change: Maximum allowed pressure change threshold.
        :param absolute_oil_mbe: Absolute oil material balance error for this step (res ft³).
        :param absolute_water_mbe: Absolute water material balance error for this step (res ft³).
        :param absolute_gas_mbe: Absolute gas material balance error for this step (res ft³), including dissolved gas.
        :param total_absolute_mbe: Sum of absolute phase MBEs (res ft³).
        :param relative_oil_mbe: Oil MBE as a fraction of the previous-step oil pore volume.
        :param relative_water_mbe: Water MBE as a fraction of the previous-step water pore volume.
        :param relative_gas_mbe: Gas MBE as a fraction of the previous-step gas pore volume equivalent.
        :param total_relative_mbe: Total MBE as a fraction of the previous-step total pore volume.
        :return: The next proposed time step size.
        """
        # Use small tolerance for floating point comparison as step sizes may slightly overshoot
        if step_size > (self.time_remaining + 1e-9):
            raise TimingError(
                f"Step size {step_size} exceeds remaining time {self.time_remaining}. "
                "This indicates a bug in the time stepping logic."
            )

        if self.use_constant_step_size:
            return self.initial_step_size

        # Advance time and step count
        self.elapsed_time += step_size
        self.step_size = step_size
        self.step += 1
        self.steps_since_last_failure += 1

        # Record metrics
        metrics = StepMetrics(
            step_number=self.step,
            step_size=step_size,
            cfl=maximum_cfl_encountered,
            newton_iterations=newton_iterations,
            success=True,
        )
        self.recent_metrics.append(metrics)

        # Start with current step size as base
        dt = self.next_step_size

        # Collect adjustment factors from various criteria
        adjustment_factors = []

        # CFL-based adjustment with safety margin
        maximum_cfl = (
            cfl_threshold if cfl_threshold is not None else self.maximum_cfl_number
        )
        if maximum_cfl_encountered is not None and maximum_cfl_encountered > 0.0:
            target_cfl = maximum_cfl * self.cfl_safety_margin
            cfl_ratio = target_cfl / maximum_cfl_encountered

            # Cap CFL-based growth to prevent wild jumps when CFL is very low
            cfl_ratio = min(cfl_ratio, self.maximum_growth_per_step)

            # Be more conservative if we were close to the limit
            proximity_to_limit = maximum_cfl_encountered / maximum_cfl
            if proximity_to_limit > 0.9:
                # Very close to limit, we only decrease or maintain
                cfl_factor = min(cfl_ratio, 1.0)
                logger.debug(
                    f"CFL very close to limit ({proximity_to_limit:.2%}), conservative growth"
                )
            elif proximity_to_limit > 0.8:
                # Close to limit, we need be cautious
                cfl_factor = min(cfl_ratio, 1.1)
                logger.debug(
                    f"CFL close to limit ({proximity_to_limit:.2%}), cautious growth"
                )
            else:
                # Comfortable margin, we can allow normal growth
                cfl_factor = cfl_ratio
                logger.debug(
                    f"CFL comfortable ({proximity_to_limit:.2%}), normal growth"
                )
            adjustment_factors.append(("CFL", cfl_factor))

        # Saturation change adjustment with intelligent scaling
        if (
            maximum_saturation_change is not None
            and maximum_allowed_saturation_change is not None
            and maximum_saturation_change > 0.0
        ):
            saturation_utilization = (
                maximum_saturation_change / maximum_allowed_saturation_change
            )

            if saturation_utilization > 0.95:
                # Very close to limit. Reduce step size
                saturation_factor = 0.85
                logger.debug(
                    f"Saturation change very high ({saturation_utilization:.2%}), reducing step"
                )
            elif saturation_utilization > 0.85:
                # Getting close, maintain or slightly reduce
                saturation_factor = 0.95
                logger.debug(
                    f"Saturation change high ({saturation_utilization:.2%}), maintaining step"
                )
            elif saturation_utilization > 0.7:
                # Moderate usage, allow modest growth
                saturation_factor = 1.05
                logger.debug(
                    f"Saturation change moderate ({saturation_utilization:.2%}), modest growth"
                )
            elif saturation_utilization < 0.3:
                # Very low usage, could grow more aggressively
                saturation_factor = min(
                    1.3,
                    maximum_allowed_saturation_change / maximum_saturation_change * 0.8,
                )
                logger.debug(
                    f"Saturation change low ({saturation_utilization:.2%}), allowing growth"
                )
            else:
                # Normal range. Apply proportional adjustment
                saturation_factor = min(
                    1.15,
                    maximum_allowed_saturation_change / maximum_saturation_change * 0.9,
                )
                logger.debug(
                    f"Saturation change normal ({saturation_utilization:.2%}), proportional growth"
                )
            adjustment_factors.append(("Saturation", saturation_factor))

        # Pressure change adjustment with intelligent scaling
        if (
            maximum_pressure_change is not None
            and maximum_allowed_pressure_change is not None
            and maximum_pressure_change > 0.0
        ):
            pressure_utilization = (
                maximum_pressure_change / maximum_allowed_pressure_change
            )

            if pressure_utilization > 0.95:
                # Very close to limit. Reduce step size
                pressure_factor = 0.85
                logger.debug(
                    f"Pressure change very high ({pressure_utilization:.2%}), reducing step"
                )
            elif pressure_utilization > 0.85:
                # Getting close. Maintain or slightly reduce
                pressure_factor = 0.95
                logger.debug(
                    f"Pressure change high ({pressure_utilization:.2%}), maintaining step"
                )
            elif pressure_utilization > 0.7:
                # Moderate usage. Allow modest growth
                pressure_factor = 1.05
                logger.debug(
                    f"Pressure change moderate ({pressure_utilization:.2%}), modest growth"
                )
            elif pressure_utilization < 0.3:
                # Very low usage, could grow more aggressively
                pressure_factor = min(
                    1.3, maximum_allowed_pressure_change / maximum_pressure_change * 0.8
                )
                logger.debug(
                    f"Pressure change low ({pressure_utilization:.2%}), allowing growth"
                )
            else:
                # Normal range. Apply proportional adjustment
                pressure_factor = min(
                    1.15,
                    maximum_allowed_pressure_change / maximum_pressure_change * 0.9,
                )
                logger.debug(
                    f"Pressure change normal ({pressure_utilization:.2%}), proportional growth"
                )
            adjustment_factors.append(("Pressure", pressure_factor))

        # Performance-based factor from historical trends
        performance_factor = self._compute_performance_factor()
        if performance_factor < 1.0:
            adjustment_factors.append(("Performance", performance_factor))
            logger.debug(f"Performance trends suggest factor: {performance_factor:.3f}")

        # Newton iteration-based adjustment
        if newton_iterations is not None:
            if newton_iterations > 10:
                newton_iteration_factor = 0.7
                logger.debug(
                    f"High Newton iterations ({newton_iterations}), reducing step"
                )
                adjustment_factors.append(("Newton", newton_iteration_factor))
            elif newton_iterations < 4 and self.steps_since_last_failure >= 3:
                newton_iteration_factor = 1.2
                logger.debug(
                    f"Low Newton iterations ({newton_iterations}), allowing growth"
                )
                adjustment_factors.append(("Newton", newton_iteration_factor))

        # MBE-based step-size adjustment (only when opt-in flag is set)
        if self.use_mbe_for_step_size:
            mbe_accept_pairs = [
                (absolute_oil_mbe, self.maximum_absolute_oil_mbe, "absolute oil MBE"),
                (
                    absolute_water_mbe,
                    self.maximum_absolute_water_mbe,
                    "absolute water MBE",
                ),
                (absolute_gas_mbe, self.maximum_absolute_gas_mbe, "absolute gas MBE"),
                (
                    total_absolute_mbe,
                    self.maximum_total_absolute_mbe,
                    "total absolute MBE",
                ),
                (relative_oil_mbe, self.maximum_relative_oil_mbe, "relative oil MBE"),
                (
                    relative_water_mbe,
                    self.maximum_relative_water_mbe,
                    "relative water MBE",
                ),
                (relative_gas_mbe, self.maximum_relative_gas_mbe, "relative gas MBE"),
                (
                    total_relative_mbe,
                    self.maximum_total_relative_mbe,
                    "total relative MBE",
                ),
            ]
            for actual, limit, label in mbe_accept_pairs:
                if actual is None or limit is None:
                    continue
                actual_abs = abs(actual)
                utilization = actual_abs / limit if limit > 0 else 0.0
                if utilization > 0.9:
                    mbe_factor = 0.85
                elif utilization > 0.75:
                    mbe_factor = 0.95
                elif utilization < 0.2:
                    mbe_factor = min(1.2, limit / max(actual_abs, 1e-30) * 0.5)
                else:
                    mbe_factor = 1.0  # comfortable, no adjustment
                if mbe_factor != 1.0:
                    logger.debug(
                        f"MBE accept adjustment ({label}): utilization={utilization:.2%}, factor={mbe_factor:.3f}"
                    )
                    adjustment_factors.append((f"MBE({label})", mbe_factor))

        # Apply all adjustment factors
        if adjustment_factors:
            logger.debug(f"Applying {len(adjustment_factors)} adjustment factors:")
            for name, factor in adjustment_factors:
                logger.debug(f"  {name}: {factor:.3f}")
                dt *= factor

        # Apply ramp-up factor (only after cooldown period)
        can_ramp_up = (
            self.ramp_up_factor is not None
            and not self.last_step_failed
            and self.steps_since_last_failure >= self.growth_cooldown_steps
        )
        if can_ramp_up:
            # Only apply ramp-up if we're not pushing any limits
            limits_ok = True
            if maximum_cfl_encountered is not None and maximum_cfl > 0:
                limits_ok &= (maximum_cfl_encountered / maximum_cfl) < 0.7
            if (
                maximum_saturation_change is not None
                and maximum_allowed_saturation_change is not None
            ):
                limits_ok &= (
                    maximum_saturation_change / maximum_allowed_saturation_change
                ) < 0.7
            if (
                maximum_pressure_change is not None
                and maximum_allowed_pressure_change is not None
            ):
                limits_ok &= (
                    maximum_pressure_change / maximum_allowed_pressure_change
                ) < 0.7

            if limits_ok:
                dt *= self.ramp_up_factor  # type: ignore
                logger.debug(f"Applying ramp-up factor: {self.ramp_up_factor}")
            else:
                logger.debug("Ramp-up suppressed: too close to limits")

        # Limit growth rate relative to current step
        max_allowed_growth = self.step_size * self.maximum_growth_per_step
        if dt > max_allowed_growth:
            logger.debug(f"Capping growth from {dt:.6e} to {max_allowed_growth:.6e}")
            dt = max_allowed_growth

        # Check if we're approaching a previously failed step size
        if self._is_near_failed_size(dt):
            dt *= 0.8  # Be more conservative near failure zones
            logger.debug(
                f"Step size {dt:.6e} is near a previously failed size, reducing conservatively"
            )

        # Enforce absolute bounds
        dt = min(dt, self.maximum_step_size)
        dt = max(dt, self.minimum_step_size)

        # Apply smoothing via EMA
        # Initialize EMA on first step (more robust than checking == 0.0)
        if self.step == 0:
            self.ema_step_size = dt
        else:
            self.ema_step_size = (
                self.step_size_smoothing * self.ema_step_size
                + (1 - self.step_size_smoothing) * dt
            )

        self.next_step_size = self.ema_step_size

        # Reset rejection tracking
        self.last_step_failed = False
        self.rejection_count = 0

        logger.debug(
            f"Time step of size {step_size:.6e} accepted for time step {self.step} "
            f"at elapsed time {self.elapsed_time:.4f}s. Next size: {self.next_step_size:.6e}"
        )
        return self.next_step_size

    def dump_state(self) -> TimerState:
        """
        Serialize the current timer state to a dictionary.

        Returns all the internal state needed to reconstruct this timer's
        exact state at this point in time. Useful for checkpointing, saving
        simulation progress, or debugging.

        :return: `TimerState` dictionary containing all timer state variables

        Example:
        ```python
        timer = Timer(initial_step_size=0.1, simulation_time=1000.0)
        # ... run simulation for a while ...

        # Save timer state
        timer_state = timer.dump_state()
        save_to_file(timer_state, "timer_state.json")

        # Later, restore timer
        timer_state = load_from_file("timer_state.json")
        timer = Timer.load_state(timer_state)
        ```
        """
        return {
            "initial_step_size": self.initial_step_size,
            "maximum_step_size": self.maximum_step_size,
            "minimum_step_size": self.minimum_step_size,
            "simulation_time": self.simulation_time,
            "maximum_cfl_number": self.maximum_cfl_number,
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
            "recent_metrics": [
                typing.cast(StepMetricsDict, attrs.asdict(metric))
                for metric in self.recent_metrics
            ],
            "failed_step_sizes": list(self.failed_step_sizes),
        }

    @classmethod
    def load_state(cls, state: TimerState) -> Self:
        """
        Reconstruct a timer from a previously saved timer state.

        Creates a new timer instance and restores all internal state from
        the provided dictionary. This is the inverse of `dump_state()`.

        :param state: `TimerState` dictionary containing timer state (from `dump_state()`)
        :return: A new timer instance with the restored state
        :raises `ValidationError`: If state dictionary is invalid or incomplete

        Example:
        ```python
        # Save timer state during simulation
        timer_state = timer.dump_state()

        # Later, restore and continue
        timer = Timer.load_state(timer_state)
        for state in run(model, timer, wells):
            process(state)
        ```
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
        missing_keys = required_keys - set(state.keys())
        if missing_keys:
            raise ValidationError(f"Timer state missing required keys: {missing_keys}")

        config_params = {
            "initial_step_size": state["initial_step_size"],
            "maximum_step_size": state["maximum_step_size"],
            "minimum_step_size": state["minimum_step_size"],
            "simulation_time": state["simulation_time"],
            "maximum_cfl_number": state.get("maximum_cfl_number", 1.0),
            "cfl_safety_margin": state.get("cfl_safety_margin", 0.9),
            "ramp_up_factor": state.get("ramp_up_factor"),
            "backoff_factor": state.get("backoff_factor", 0.5),
            "aggressive_backoff_factor": state.get("aggressive_backoff_factor", 0.25),
            "maximum_steps": state.get("maximum_steps"),
            "maximum_absolute_oil_mbe": state.get("maximum_absolute_oil_mbe", None),
            "maximum_absolute_water_mbe": state.get("maximum_absolute_water_mbe", None),
            "maximum_absolute_gas_mbe": state.get("maximum_absolute_gas_mbe", None),
            "maximum_total_absolute_mbe": state.get("maximum_total_absolute_mbe", None),
            "maximum_relative_oil_mbe": state.get("maximum_relative_oil_mbe", None),
            "maximum_relative_water_mbe": state.get("maximum_relative_water_mbe", None),
            "maximum_relative_gas_mbe": state.get("maximum_relative_gas_mbe", None),
            "maximum_total_relative_mbe": state.get("maximum_total_relative_mbe", None),
            "use_mbe_for_step_size": state.get("use_mbe_for_step_size", False),
            "maximum_rejections": state.get("maximum_rejections", 10),
            "maximum_growth_per_step": state.get("maximum_growth_per_step", 1.5),
            "step_size_smoothing": state.get("step_size_smoothing", 0.7),
            "growth_cooldown_steps": state.get("growth_cooldown_steps", 5),
            "failure_memory_window": state.get("failure_memory_window", 10),
            "metrics_history_size": state.get("metrics_history_size", 20),
        }
        timer = cls(**config_params)  # type: ignore

        # Restore runtime state (use `object.__setattr__` just in case `Timer` is frozen)
        object.__setattr__(timer, "elapsed_time", state["elapsed_time"])
        object.__setattr__(timer, "step", state["step"])
        object.__setattr__(timer, "step_size", state["step_size"])
        object.__setattr__(
            timer, "next_step_size", state.get("next_step_size", state["step_size"])
        )
        object.__setattr__(
            timer, "ema_step_size", state.get("ema_step_size", state["step_size"])
        )
        object.__setattr__(
            timer, "last_step_failed", state.get("last_step_failed", False)
        )
        object.__setattr__(timer, "rejection_count", state.get("rejection_count", 0))
        object.__setattr__(
            timer,
            "steps_since_last_failure",
            state.get("steps_since_last_failure", 0),
        )

        # Restore history
        recent_metrics_data = state.get("recent_metrics", [])
        recent_metrics = deque(
            [StepMetrics(**metric_data) for metric_data in recent_metrics_data],
            maxlen=timer.metrics_history_size,
        )
        object.__setattr__(timer, "recent_metrics", recent_metrics)

        failed_step_sizes_data = state.get("failed_step_sizes", [])
        failed_step_sizes = deque(
            failed_step_sizes_data, maxlen=timer.failure_memory_window
        )
        object.__setattr__(timer, "failed_step_sizes", failed_step_sizes)

        logger.debug(
            f"Timer state loaded: step {timer.step}, "
            f"elapsed time {timer.elapsed_time:.4f}s, "
            f"step size {timer.step_size:.6f}s"
        )
        return timer

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return typing.cast(typing.Dict[str, typing.Any], self.dump_state())

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        data = typing.cast(TimerState, data)
        return cls.load_state(data)
