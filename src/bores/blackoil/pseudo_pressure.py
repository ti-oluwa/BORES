import logging
import threading
import typing
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt
from cachetools import LFUCache
from scipy.integrate import cumulative_trapezoid, quad  # type: ignore[import-untyped]
from scipy.interpolate import PchipInterpolator  # type: ignore[import-untyped]
from scipy.optimize import brentq
from typing_extensions import Self

from bores.constants import UnitConversionTable, c, get_conversion_factors
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.serde.stores import StoreSerializable
from bores.types import (
    FloatArray,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    UnitSystem,
)
from bores.utils import scale

logger = logging.getLogger(__name__)

__all__ = ["PseudoPressureTable", "build_pseudo_pressure_table"]


def compute_gas_pseudo_pressure(
    pressure: Number,
    z_factor_func: typing.Callable[[Number], Number],
    viscosity_func: typing.Callable[[Number], Number],
    reference_pressure: Number = 14.7,
) -> Number:
    """
    Compute the gas pseudo-pressure using Al-Hussainy real-gas potential.

    The pseudo-pressure is defined as:
        m(P) = ∫[P_ref to P] (2*P' / (μ(P') * Z(P'))) dP'

    This formulation accounts for gas compressibility and non-Darcy effects,
    allowing the use of standard liquid-like flow equations for gas.

    Physical Interpretation:
    - m(P) transforms the nonlinear gas diffusivity equation into a linear form
    - At low pressure: m(P) ≈ P² (ideal gas limit)
    - At high pressure: deviations due to Z-factor and viscosity changes

    :param pressure: Current pressure (psi)
    :param z_factor_func: Function returning Z-factor at given pressure Z(P)
    :param viscosity_func: Function returning viscosity at given pressure μ(P) in cP
    :param reference_pressure: Reference pressure for integration (psi), typically 14.7
    :return: Pseudo-pressure m(P) in psi²/cP

    References:
    - Al-Hussainy, R., Ramey, H.J., and Crawford, P.B. (1966).
    - "The Flow of Real Gases Through Porous Media."
    - JPT, May 1966, pp. 624-636.
    """
    if pressure <= 0:
        raise ValidationError(f"Pressure must be positive, got {pressure}")
    if reference_pressure <= 0:
        raise ValidationError(f"Reference pressure must be positive, got {reference_pressure}")

    # If pressure equals reference, pseudo-pressure is zero by definition
    if abs(pressure - reference_pressure) < 1e-6:
        return 0.0

    # Define the integrand: 2*P / (μ*z_factor)
    def integrand(P: Number) -> Number:
        """Integrand for pseudo-pressure calculation."""
        # Clamp pressure to avoid extrapolation issues
        clamped_P = max(1.0, P)  # Don't go below 1 psi

        try:
            z_factor = z_factor_func(clamped_P)
            viscosity = viscosity_func(clamped_P)
        except Exception as exc:
            logger.warning(
                f"Failed to evaluate `z_factor` or μ at P={clamped_P} using ideal gas approximation as fallback: {exc}"
            )
            # Use ideal gas approximation as fallback
            z_factor = 1.0
            viscosity = 0.01  # Typical gas viscosity in cP

        # Protect against division by zero or negative values
        if (
            z_factor <= 0
            or viscosity <= 0
            or not np.isfinite(z_factor)
            or not np.isfinite(viscosity)
        ):
            logger.warning(
                f"Invalid z_factor={z_factor} or μ={viscosity} at P={clamped_P}. Using ideal gas approximation."
            )
            z_factor = max(z_factor, 0.01)  # Minimum reasonable z_factor
            viscosity = max(viscosity, 0.001)  # Minimum reasonable μ (cP)

        result = 2.0 * clamped_P / (viscosity * z_factor)
        # Sanity check on integrand value
        if not np.isfinite(result) or result < 0:
            logger.warning("Invalid integrand %s at P=%.4e", result, clamped_P)
            return 0.0
        return result

    # Perform numerical integration with adaptive strategy
    min_pressure = min(pressure, reference_pressure)
    max_pressure = max(pressure, reference_pressure)

    # Split integration into segments if range is large
    # This helps `quad()` adapt better to different pressure regimes
    if (max_pressure - min_pressure) > 1000:  # Large pressure range
        # Split into low, medium, high pressure segments
        split_points = np.logspace(
            start=np.log10(min_pressure), stop=np.log10(max_pressure), num=5
        )
        total_integral = 0.0

        for i in range(len(split_points) - 1):
            try:
                segment_result, _ = quad(
                    func=integrand,  # type: ignore[arg-type]
                    a=split_points[i],
                    b=split_points[i + 1],
                    epsabs=1e-6,
                    epsrel=1e-4,
                    limit=200,
                )
                total_integral += segment_result
            except Exception as exc:
                logger.warning(
                    f"Integration failed for segment [{split_points[i]:.1f}, {split_points[i + 1]:.1f}]: {exc}. Using trapezoidal approximation"
                )
                # Use trapezoidal approximation for failed segment
                p_seg = np.linspace(split_points[i], split_points[i + 1], 50)
                y_seg = np.array([integrand(p) for p in p_seg])
                total_integral += np.trapezoid(y=y_seg, x=p_seg)

        result = total_integral
    else:
        # Single integration for small range
        try:
            result, _ = quad(
                func=integrand,  # type: ignore[arg-type]
                a=min_pressure,
                b=max_pressure,
                epsabs=1e-6,
                epsrel=1e-4,
                limit=200,
            )
        except Exception as exc:
            logger.warning("Integration failed: %s. Using trapezoidal fallback.", exc)
            # Fallback to simple trapezoidal rule
            p_points = np.linspace(min_pressure, max_pressure, 100)
            y_points = np.array([integrand(p) for p in p_points])
            result = typing.cast(Number, np.trapezoid(y=y_points, x=p_points))

    # Apply sign based on integration direction
    if pressure < reference_pressure:
        result = -result
    return result


def _supports_vectorization(
    z_factor_func: typing.Callable[[NumberOrArray[NDimension]], NumberOrArray[NDimension]],
    viscosity_func: typing.Callable[[NumberOrArray[NDimension]], NumberOrArray[NDimension]],
) -> bool:
    """
    Check if both Z-factor and viscosity functions support vectorized operations.

    :param z_factor_func: Z-factor function
    :param viscosity_func: Viscosity function
    :return: True if both functions support arrays
    """
    z_supports = getattr(z_factor_func, "_supports_arrays", False)
    mu_supports = getattr(viscosity_func, "_supports_arrays", False)
    return z_supports and mu_supports


def compute_pseudo_pressures_vectorized(
    pressures: NumberArray[NDimension],
    z_factor_func: typing.Callable[[NumberOrArray[NDimension]], NumberOrArray[NDimension]],
    viscosity_func: typing.Callable[[NumberOrArray[NDimension]], NumberOrArray[NDimension]],
    reference_pressure: Number,
    dtype: npt.DTypeLike = None,
) -> NumberArray[NDimension]:
    """
    Build entire pseudo-pressure table values using vectorized operations.

    This is much faster than computing each point individually because:
    - Single vectorized call to Z(P_array) and μ(P_array)
    - Vectorized integration using cumulative_trapezoid
    - No threading overhead needed

    :param pressures: Array of pressure points
    :param z_factor_func: Vectorized Z-factor function
    :param viscosity_func: Vectorized viscosity function
    :param reference_pressure: Reference pressure
    :return: Array of pseudo-pressures
    """
    # Clamp pressures
    clamped_pressures = np.maximum(pressures, 1.0)

    # Single vectorized call for all pressures
    z_factor_array = np.asarray(z_factor_func(clamped_pressures))  # type: ignore[arg-type]
    viscosity_array = np.asarray(viscosity_func(clamped_pressures))  # type: ignore[arg-type]

    # Validate shapes
    if z_factor_array.shape != pressures.shape or viscosity_array.shape != pressures.shape:
        raise ValueError(
            f"Shape mismatch: P={pressures.shape}, z_factor={z_factor_array.shape}, μ={viscosity_array.shape}"
        )

    # Handle invalid values
    invalid_z_factor = (z_factor_array <= 0) | ~np.isfinite(z_factor_array)
    invalid_viscosity = (viscosity_array <= 0) | ~np.isfinite(viscosity_array)

    if np.any(invalid_z_factor):
        logger.warning("Clamping %d invalid Z-factor values", np.sum(invalid_z_factor))
        z_factor_array = np.maximum(z_factor_array, 0.01)

    if np.any(invalid_viscosity):
        logger.warning("Clamping %d invalid viscosity values", np.sum(invalid_viscosity))
        viscosity_array = np.maximum(viscosity_array, 0.001)

    # Compute integrand: 2*P / (μ*z_factor)
    integrand_array = 2.0 * clamped_pressures / (viscosity_array * z_factor_array)

    # Handle invalid integrand values
    invalid = ~np.isfinite(integrand_array) | (integrand_array < 0)
    if np.any(invalid):
        logger.warning("Setting %d invalid integrand values to zero", np.sum(invalid))
        integrand_array = np.where(invalid, 0.0, integrand_array)

    # Cumulative integration from reference pressure
    # Find index closest to reference pressure
    reference_idx = np.searchsorted(pressures, reference_pressure)

    if reference_idx == 0:
        # Reference is at or below minimum, hence we integrate forward only
        pseudo_pressures = cumulative_trapezoid(integrand_array, pressures, initial=0)
    elif reference_idx >= len(pressures):
        # Reference is at or above maximum, hence we integrate backward only
        pseudo_pressures = -cumulative_trapezoid(
            integrand_array[::-1], pressures[::-1], initial=0
        )[::-1]
    else:
        # Reference is in the middle so we integrate both directions
        # Backward from ref to start
        backward = -cumulative_trapezoid(
            integrand_array[: reference_idx + 1][::-1],
            pressures[: reference_idx + 1][::-1],
            initial=0,
        )[::-1]

        # Forward from ref to end
        forward = cumulative_trapezoid(
            integrand_array[reference_idx:], pressures[reference_idx:], initial=0
        )
        # Then we combine
        pseudo_pressures = np.concatenate([backward[:-1], forward])

    return np.ascontiguousarray(pseudo_pressures, dtype=dtype)  # type: ignore[return-value]


def compute_pseudo_pressures_scalar(
    pressures: NumberArray[NDimension],
    z_factor_func: typing.Callable[[Number], Number],
    viscosity_func: typing.Callable[[Number], Number],
    reference_pressure: Number,
    max_workers: int,
    dtype: npt.DTypeLike = None,
) -> NumberArray[NDimension]:
    """
    Build pseudo-pressure table values using threaded scalar computation.

    Uses `ThreadPoolExecutor` to parallelize individual integrations.

    :param pressures: Array of pressure points
    :param z_factor_func: Z-factor function
    :param viscosity_func: Viscosity function
    :param reference_pressure: Reference pressure
    :param max_workers: Maximum number of threads to use
    :return: Array of pseudo-pressures
    """

    def compute(pressure: Number) -> Number:
        return compute_gas_pseudo_pressure(
            pressure=pressure,
            z_factor_func=z_factor_func,
            viscosity_func=viscosity_func,
            reference_pressure=reference_pressure,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pseudo_pressures = list(executor.map(compute, pressures))
    return np.ascontiguousarray(pseudo_pressures, dtype=dtype)  # type: ignore[return-value]


def compute_pseudo_pressures(
    pressures: NumberArray[NDimension],
    z_factor_func: typing.Callable[[NumberOrArray[NDimension]], NumberOrArray[NDimension]],
    viscosity_func: typing.Callable[[NumberOrArray[NDimension]], NumberOrArray[NDimension]],
    reference_pressure: Number,
    dtype: npt.DTypeLike = None,
) -> NumberArray[NDimension]:
    """
    Build pseudo-pressure table values with automatic vectorization detection.

    If both z_factor_func and viscosity_func have `_supports_arrays=True`,
    uses fast vectorized computation. Otherwise uses threaded scalar computation.

    :param pressures: Array of pressure points
    :param z_factor_func: Z-factor function
    :param viscosity_func: Viscosity function
    :param reference_pressure: Reference pressure
    :return: Array of pseudo-pressures
    """
    points = len(pressures)
    if _supports_vectorization(z_factor_func, viscosity_func):
        logger.debug(
            f"Building pseudo-pressure table using vectorized computation ({points} points)..."
        )
        try:
            return compute_pseudo_pressures_vectorized(
                pressures=pressures,
                z_factor_func=z_factor_func,
                viscosity_func=viscosity_func,
                reference_pressure=reference_pressure,
                dtype=dtype,
            )
        except Exception as exc:
            logger.warning(
                f"Vectorized computation failed: {exc}, falling back to threaded scalar",
                exc_info=True,
            )

    logger.debug(
        f"Building pseudo-pressure table using threaded scalar computation ({points} points)..."
    )
    max_workers = min(8, points // 50 + 1)
    return compute_pseudo_pressures_scalar(
        pressures=pressures,
        z_factor_func=z_factor_func,  # type: ignore
        viscosity_func=viscosity_func,  # type: ignore
        reference_pressure=reference_pressure,
        max_workers=max_workers,
        dtype=dtype,
    )


def build_pchip_interpolants_from_points(
    x: npt.NDArray,
    values: npt.NDArray,
    number_of_base_points: int,
    number_of_endpoint_extra_points: int,
    minimum_scale_span: Number = 1.0,
    dtype: npt.DTypeLike = np.float64,
) -> tuple[PchipInterpolator, PchipInterpolator]:
    """
    Build a PCHIP interp (and its analytical derivative) from arbitrary
    monotonically increasing x-coordinates and corresponding values.

    When `number_of_base_points > 0` and the raw knot count is smaller than
    `number_of_base_points` and the x-range exceeds `minimum_scale_span`,
    the knot field/array is expanded to `number_of_base_points` base points (plus
    `number_of_endpoint_extra_points` extra knots in each boundary decade)
    before fitting. The expanded field/array is log-spaced when all x values are
    positive (appropriate for pressure), and linearly spaced otherwise.

    :param x: Monotonically increasing x-axis knots (e.g. pressure in psi).
    :param values: Function values at each knot.
    :param number_of_base_points: Target number of base knots for field/array expansion.
        Pass `0` to use the raw knots without expansion.
    :param number_of_endpoint_extra_points: Extra knots injected into the first
        and last 10 % of the x-range during expansion. Pass `0` to disable
        endpoint enrichment.
    :param spacing: Grid spacing mode for the expanded base field/array. `"cosine"`
        clusters points near the boundaries; `"linear"` gives uniform spacing.
    :param minimum_scale_span: Minimum x-range required before field/array expansion is
        attempted. Defaults to `1.0`, which is appropriate for pressure axes
        measured in psi.
    :return: Two-tuple `(interp, d_interp)` where
        `d_interp` is the analytical first derivative of `interp`.
    """
    xs = x
    vals = values

    span = xs[-1] - xs[0]
    should_scale = (
        number_of_base_points > 0 and len(xs) < number_of_base_points and span > minimum_scale_span
    )
    if should_scale:
        # Build expanded base array, log-spaced for positive axes (pressure), linear otherwise
        if xs[0] > 0.0:
            base = np.logspace(
                np.log10(xs[0]),
                np.log10(xs[-1]),
                number_of_base_points,
                dtype=dtype,
            )
        else:
            base = np.linspace(xs[0], xs[-1], number_of_base_points, dtype=dtype)

        # Inject extra knots into each boundary decade
        if number_of_endpoint_extra_points > 0:
            decade_width = 0.10 * span
            lower_refinement = np.linspace(
                xs[0],
                xs[0] + decade_width,
                number_of_endpoint_extra_points + 2,
                dtype=dtype,
            )
            upper_refinement = np.linspace(
                xs[-1] - decade_width,
                xs[-1],
                number_of_endpoint_extra_points + 2,
                dtype=dtype,
            )
            base = np.unique(
                np.concatenate((base, lower_refinement, upper_refinement), dtype=dtype)
            )

        source_interp = PchipInterpolator(xs, vals)
        vals = source_interp(base)
        xs = base

    interp = PchipInterpolator(xs, vals)
    d_interp = interp.derivative(1)
    return interp, d_interp


@typing.final
class PseudoPressureTable(
    StoreSerializable,
    fields={
        "pressures": npt.NDArray,
        "pseudo_pressures": npt.NDArray,
        "reference_pressure": Number | None,
        "number_of_base_points": int,
        "number_of_endpoint_extra_points": int,
        "dtype": npt.DTypeLike | None,
        "unit_system": UnitSystem,
    },
):
    """
    Pre-computed gas pseudo-pressure table for fast lookup during simulation.

    Uses a PCHIP interp for C¹-continuous forward interpolation.

    Two construction modes:

    1. **Function-based**: Provide Z-factor and viscosity functions, and the table is computed
       automatically over a pressure range.
    2. **Value-based**: Provide pre-computed pressure and pseudo-pressure arrays directly
       (e.g., from laboratory data or external calculations).

    These modes are mutually exclusive.

    **Grid scaling** (`number_of_base_points` / `number_of_endpoint_extra_points`):

    When `number_of_base_points > 0` and the raw point count is smaller than
    `number_of_base_points`, the knot grid is expanded before fitting the PCHIP
    interp. Extra knots near the pressure boundaries improve derivative accuracy
    in the low and high-pressure tails where the pseudo-pressure integrand varies
    most rapidly. Pass `number_of_base_points=0` to disable scaling and use the
    raw pressure grid directly.
    """

    @typing.overload
    def __init__(
        self,
        *,
        z_factor_func: typing.Callable[[NumberOrArray[NDimension]], NumberOrArray[NDimension]],
        viscosity_func: typing.Callable[[NumberOrArray[NDimension]], NumberOrArray[NDimension]],
        pressure_range: tuple[Number, Number] | None = ...,
        points: int | None = ...,
        reference_pressure: Number | None = ...,
        number_of_base_points: int = ...,
        number_of_endpoint_extra_points: int = ...,
        dtype: npt.DTypeLike | None = ...,
        unit_system: UnitSystem = ...,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        *,
        pressures: NumberArray[NDimension],
        pseudo_pressures: NumberArray[NDimension],
        reference_pressure: Number | None = ...,
        number_of_base_points: int = ...,
        number_of_endpoint_extra_points: int = ...,
        dtype: npt.DTypeLike | None = ...,
        unit_system: UnitSystem = ...,
    ) -> None: ...

    def __init__(
        self,
        z_factor_func: typing.Callable[[NumberOrArray[NDimension]], NumberOrArray[NDimension]]
        | None = None,
        viscosity_func: typing.Callable[[NumberOrArray[NDimension]], NumberOrArray[NDimension]]
        | None = None,
        pressure_range: tuple[Number, Number] | None = None,
        points: int | None = None,
        pressures: NumberArray[NDimension] | None = None,
        pseudo_pressures: NumberArray[NDimension] | None = None,
        reference_pressure: Number | None = None,
        number_of_base_points: int = 500,
        number_of_endpoint_extra_points: int = 20,
        dtype: npt.DTypeLike = None,
        unit_system: UnitSystem = UnitSystem.FIELD,
    ):
        """
        Build pseudo-pressure lookup table.

        **Function-based mode** (compute from correlations):

        :param z_factor_func: Z-factor correlation Z(P).
        :param viscosity_func: Gas viscosity correlation μ(P) in cP.
        :param pressure_range: (P_min, P_max) for table. Defaults to (c.MINIMUM_VALID_PRESSURE, c.MAXIMUM_VALID_PRESSURE).
        :param points: Number of pressure points to compute before PCHIP fitting.
        :param reference_pressure: Reference pressure (psi), default c.MINIMUM_VALID_PRESSURE.
        :param number_of_base_points: Target knot count for grid expansion before PCHIP fitting. Pass `0` to disable scaling.
        :param number_of_endpoint_extra_points: Extra knots injected into each boundary decade during grid expansion.

        **Value-based mode** (use pre-computed data):

        :param pressures: Array of pressure points (psi), must be sorted ascending.
        :param pseudo_pressures: Array of corresponding pseudo-pressure values (psi²/cP).
        :param reference_pressure: Reference pressure (psi), default c.MINIMUM_VALID_PRESSURE.
        :param number_of_base_points: Same semantics as function-based mode.
        :param number_of_endpoint_extra_points: Same semantics as function-based mode.

        Example (function-based):

        ```python
        table = PseudoPressureTable(
            z_factor_func=my_z_func,
            viscosity_func=my_mu_func,
            pressure_range=(100, 5000),
            points=1000,
        )
        ```

        Example (value-based):

        ```python
        table = PseudoPressureTable(
            pressures=np.array([100, 500, 1000, 2000, 5000]),
            pseudo_pressures=np.array([2.1e4, 5.3e5, 1.2e6, 2.8e6, 8.1e6]),
        )
        ```
        """
        func_mode = z_factor_func is not None or viscosity_func is not None
        value_mode = pressures is not None or pseudo_pressures is not None

        if func_mode and value_mode:
            raise ValidationError(
                "Cannot mix function-based mode (`z_factor_func`, `viscosity_func`) "
                "with value-based mode (`pressures`, `pseudo_pressures`). "
                "Use one or the other, not both."
            )

        if not func_mode and not value_mode:
            raise ValidationError(
                "Must provide either:\n"
                "  1. `z_factor_func` and `viscosity_func` (function-based mode), or\n"
                "  2. `pressures` and `pseudo_pressures` arrays (value-based mode)"
            )

        self.reference_pressure = typing.cast(
            float, reference_pressure or c.MINIMUM_VALID_PRESSURE
        )
        self.number_of_base_points = number_of_base_points
        self.number_of_endpoint_extra_points = number_of_endpoint_extra_points
        self.dtype = np.dtype(dtype if dtype is not None else get_dtype())
        self.unit_system = unit_system

        if value_mode:
            if pressures is None or pseudo_pressures is None:
                raise ValidationError(
                    "Value-based mode requires both 'pressures' and 'pseudo_pressures' arrays"
                )
            if pressures.shape != pseudo_pressures.shape:
                raise ValidationError(
                    f"Pressure and pseudo-pressure arrays must have same shape. "
                    f"Got pressures: {pressures.shape}, pseudo_pressures: {pseudo_pressures.shape}"
                )
            if len(pressures) < 2:
                raise ValidationError(
                    f"Need at least 2 points for interpolation, got {len(pressures)}"
                )

            self.pressures = np.ascontiguousarray(pressures, dtype=dtype)
            self.pseudo_pressures = np.ascontiguousarray(pseudo_pressures, dtype=dtype)
            self.z_factor_func = None  # type: ignore[assignment]
            self.viscosity_func = None  # type: ignore[assignment]

            logger.debug(
                "Built pseudo-pressure table from %d data points: P ∈ [%.4f, %.4f] psi",
                len(pressures),
                pressures.min(),
                pressures.max(),
            )
        else:
            if z_factor_func is None or viscosity_func is None:
                raise ValidationError(
                    "Function-based mode requires both 'z_factor_func' and 'viscosity_func'"
                )

            self.z_factor_func = z_factor_func
            self.viscosity_func = viscosity_func

            min_pressure, max_pressure = pressure_range or (
                c.MINIMUM_VALID_PRESSURE,
                c.MAXIMUM_VALID_PRESSURE,
            )
            n_points = typing.cast(int, points or c.GAS_PSEUDO_PRESSURE_POINTS)
            self.pressures = np.logspace(
                np.log10(min_pressure), np.log10(max_pressure), n_points, dtype=dtype
            )

            logger.info("Building pseudo-pressure table with %d points...", n_points)
            self.pseudo_pressures = compute_pseudo_pressures(
                pressures=self.pressures,
                z_factor_func=self.z_factor_func,  # type: ignore[arg-type]
                viscosity_func=self.viscosity_func,  # type: ignore[arg-type]
                reference_pressure=self.reference_pressure,
                dtype=dtype,
            )
            logger.debug(
                "Pseudo-pressure table built: P ∈ [%.4f, %.4f] psi",
                min_pressure,
                max_pressure,
            )

        # Build interpolants
        self._p_interp, self._dp_interp = build_pchip_interpolants_from_points(
            x=self.pressures,
            values=self.pseudo_pressures,
            number_of_base_points=self.number_of_base_points,
            number_of_endpoint_extra_points=self.number_of_endpoint_extra_points,
            minimum_scale_span=1.0,
            dtype=dtype,
        )

    def interpolate(self, pressure: NumberOrArray[NDimension]) -> NumberOrArray[NDimension]:
        """
        Interpolate pseudo-pressure at given pressure.

        Forward interpolation: pressure to pseudo-pressure.

        Values outside the tabulated range are clamped to the boundary
        pseudo-pressure (constant extrapolation).

        :param pressure: Pressure (psi) - scalar or array.
        :return: Pseudo-pressure m(P) (psi²/cP).
        """
        is_scalar = np.isscalar(pressure)
        dtype = self.dtype
        pressure_arr = np.atleast_1d(pressure)
        min_pressure = self._p_interp.x[0]
        max_pressure = self._p_interp.x[-1]
        result = self._p_interp(np.clip(pressure_arr, min_pressure, max_pressure, dtype=dtype))
        result = np.where(pressure_arr < min_pressure, self.pseudo_pressures[0], result)
        result = np.where(pressure_arr > max_pressure, self.pseudo_pressures[-1], result)
        if is_scalar:
            return typing.cast(Number, dtype.type(result.item()))
        return typing.cast(FloatArray[NDimension], result.reshape(pressure_arr.shape, copy=False))

    def inverse(self, pseudo_pressure: NumberOrArray[NDimension]) -> NumberOrArray[NDimension]:
        """
        Inverse interpolate pressure at given pseudo-pressure.

        Uses numerical inversion of the PCHIP interp (via Brent's method)
        to guarantee consistency with `interpolate`.

        :param pseudo_pressure: Pseudo-pressure m(P) (psi²/cP) - scalar or array.
        :return: Pressure (psi).
        """
        min_pseudo_pressure = self.pseudo_pressures[0]
        max_pseudo_pressure = self.pseudo_pressures[-1]
        min_pressure = self._p_interp.x[0]
        max_pressure = self._p_interp.x[-1]
        dtype = self.dtype

        def invert(pseudo_pressure: Number) -> Number:
            clamped_pseudo_pressure = np.clip(
                pseudo_pressure, min_pseudo_pressure, max_pseudo_pressure, dtype=dtype
            )
            if abs(clamped_pseudo_pressure - min_pseudo_pressure) < 1e-10:
                return min_pressure
            if abs(clamped_pseudo_pressure - max_pseudo_pressure) < 1e-10:
                return max_pressure
            return brentq(
                lambda p: self._p_interp(p) - clamped_pseudo_pressure,
                min_pressure,
                max_pressure,
                xtol=1e-6,
                rtol=1e-8,
            )

        is_scalar = np.isscalar(pseudo_pressure)
        pseudo_pressure_arr = np.atleast_1d(pseudo_pressure)
        result = np.vectorize(invert)(pseudo_pressure_arr)

        if is_scalar:
            return typing.cast(Number, dtype.type(result.item()))
        return typing.cast(
            FloatArray[NDimension],
            result.reshape(pseudo_pressure_arr.shape, copy=False),
        )

    def __call__(self, pressure: NumberOrArray[NDimension]) -> NumberOrArray[NDimension]:
        """
        Fast lookup of pseudo-pressure via PCHIP interpolation.

        :param pressure: Pressure (psi) - scalar or array.
        :return: Pseudo-pressure m(P) (psi²/cP).
        """
        return self.interpolate(pressure)

    def gradient(self, pressure: NumberOrArray[NDimension]) -> NumberOrArray[NDimension]:
        """
        Evaluate dm/dP from the stored analytical PCHIP derivative.

        This is the integrand 2P/(μZ) evaluated consistently with the fitted
        curve rather than from the raw functions, so it is available in both
        construction modes and is always C⁰-continuous.

        The derivative is zero outside the tabulated pressure range (constant
        extrapolation = zero slope).

        :param pressure: Pressure (psi) - scalar or array.
        :return: dm/dP (psi/cP).
        """
        is_scalar = np.isscalar(pressure)
        dtype = self.dtype
        pressure_arr = np.atleast_1d(np.asarray(pressure, dtype=dtype))
        min_pressure = self._dp_interp.x[0]
        max_pressure = self._dp_interp.x[-1]
        result = self._dp_interp(np.clip(pressure_arr, min_pressure, max_pressure, dtype=dtype))
        result = np.where(
            (pressure_arr < min_pressure) | (pressure_arr > max_pressure), 0.0, result
        )
        if is_scalar:
            return typing.cast(Number, dtype.type(result.item()))
        return typing.cast(FloatArray[NDimension], result.reshape(pressure_arr.shape, copy=False))

    dm_dp = gradient  # alias

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `PseudoPressureTable` with pressure and pseudo-pressure axes
        rescaled to *target*.

        :param target: Target `UnitSystem`.
        :returns: New `PseudoPressureTable` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        pressure_factor = factors["pressure"]
        # pseudo-pressure unit is [pressure]² / [viscosity]
        pressure_factor = (pressure_factor**2) / factors["viscosity"]
        return self.__class__(
            pressures=scale(self.pressures, pressure_factor),
            pseudo_pressures=scale(self.pseudo_pressures, pressure_factor),
            reference_pressure=scale(self.reference_pressure, pressure_factor),
            number_of_base_points=self.number_of_base_points,
            number_of_endpoint_extra_points=self.number_of_endpoint_extra_points,
            dtype=self.dtype,
            unit_system=target,
        )


PSEUDO_PRESSURE_TABLE_CACHE: LFUCache[typing.Hashable, PseudoPressureTable] = LFUCache(maxsize=100)
"""Global cache for pseudo-pressure tables"""

_pseudo_pressure_cache_lock = threading.Lock()
"""Thread-safe lock for pseudo-pressure table cache access"""


def build_pseudo_pressure_table(
    z_factor_func: typing.Callable[[NumberOrArray[NDimension]], NumberOrArray[NDimension]],
    viscosity_func: typing.Callable[[NumberOrArray[NDimension]], NumberOrArray[NDimension]],
    reference_pressure: Number | None = None,
    pressure_range: tuple[Number, Number] | None = None,
    points: int | None = None,
    number_of_base_points: int = 500,
    number_of_endpoint_extra_points: int = 20,
    dtype: npt.DTypeLike = None,
    unit_system: UnitSystem = UnitSystem.FIELD,
    cache_key: typing.Hashable | None = None,
) -> PseudoPressureTable:
    """
    Build a gas pseudo-pressure table with optional global caching.

    Creates `PseudoPressureTable` instances with intelligent caching
    to avoid recomputing expensive integrals for identical fluid properties.

    **Thread Safety:**
    This function is thread-safe. Cache access is protected by a lock.

    **Caching Strategy:**
    - If `cache_key` is provided and a table with that key exists, return cached table
    - If `cache_key` is provided but table doesn't exist, compute and cache it
    - If `cache_key` is None, always compute a new table (no caching)

    **Cache Key Construction:**
    The cache key should uniquely identify the table based on:
    - Gas properties (specific gravity, molecular weight)
    - Temperature
    - Pressure range and resolution
    - Whether PVT tables are used

    Example:

    ```python
    # Build cache key from fluid properties
    cache_key = (
        "CH4",  # fluid name
        0.65,  # gas gravity
        16.04,  # molecular weight (g/mol)
        150.0,  # temperature (°F)
        14.7,  # reference pressure (psi)
        (14.7, 5000),  # pressure range
        100,  # points
        None,  # pvt_tables (or hash of tables)
        np.float64,  # dtype
        UnitSystem.METRIC,  # Unit system
    )

    table = build_pseudo_pressure_table(
        z_factor_func=z_func,
        viscosity_func=mu_func,
        cache_key=cache_key,
    )
    ```

    :param z_factor_func: Function to compute Z-factor at a given pressure
    :param viscosity_func: Function to compute viscosity at a given pressure
    :param reference_pressure: Reference pressure (psi), default 14.7
    :param pressure_range: (min, max) pressure range (psi), default (14.7, 5000)
    :param points: Number of pressure points, default 100
    :param interpolation_method: "linear" or "cubic"
    :param cache_key: Optional hashable key for caching. If None, no caching.
    :return: `PseudoPressureTable` instance

    Note:
        The global cache persists for the lifetime of the Python process.
        Use `clear_pseudo_pressure_table_cache()` to free memory if needed.
    """
    # Check cache if key provided
    if cache_key is not None:
        with _pseudo_pressure_cache_lock:
            if cache_key in PSEUDO_PRESSURE_TABLE_CACHE:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Using cached pseudo-pressure table for key: %s", cache_key)
                return PSEUDO_PRESSURE_TABLE_CACHE[cache_key]

    # Build new table outside lock to avoid blocking other threads
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Building new pseudo-pressure table for key: %s", cache_key)

    table = PseudoPressureTable(
        z_factor_func=z_factor_func,
        viscosity_func=viscosity_func,
        reference_pressure=reference_pressure,
        pressure_range=pressure_range,
        points=points,
        number_of_base_points=number_of_base_points,
        number_of_endpoint_extra_points=number_of_endpoint_extra_points,
        dtype=dtype,
        unit_system=unit_system,
    )

    # Cache if key provided
    if cache_key is not None:
        with _pseudo_pressure_cache_lock:
            # Double-check in case another thread built it while we were working
            if cache_key not in PSEUDO_PRESSURE_TABLE_CACHE:
                PSEUDO_PRESSURE_TABLE_CACHE[cache_key] = table
                logger.debug(
                    f"Cached pseudo-pressure table. Cache size: {len(PSEUDO_PRESSURE_TABLE_CACHE)}"
                )
            else:
                # Another thread cached it first, use that one
                table = PSEUDO_PRESSURE_TABLE_CACHE[cache_key]
    return table


def clear_pseudo_pressure_table_cache() -> None:
    """Clear the global pseudo-pressure table cache to free memory."""
    global PSEUDO_PRESSURE_TABLE_CACHE
    with _pseudo_pressure_cache_lock:
        cache_size = len(PSEUDO_PRESSURE_TABLE_CACHE)
        PSEUDO_PRESSURE_TABLE_CACHE.clear()
    logger.info("Cleared %d cached pseudo-pressure tables", cache_size)


def get_pseudo_pressure_table_cache_info() -> dict[str, typing.Any]:
    """Get information about the current cache state."""
    with _pseudo_pressure_cache_lock:
        return {
            "cache_size": len(PSEUDO_PRESSURE_TABLE_CACHE),
            "cached_keys": list(PSEUDO_PRESSURE_TABLE_CACHE.keys()),
        }
