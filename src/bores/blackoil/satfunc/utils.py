import numba
import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator

from bores.types import NDimension, Number, NumberArray, NumberOrArray, Spacing
from bores.utils import atleast_1d


def make_saturation_field(
    n_points: int = 200,
    min_saturation: Number = 0.0,
    max_saturation: Number = 1.0,
    spacing: Spacing = "cosine",
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray:
    """
    Build a 1-D saturation field over `[min_saturation, max_saturation]`.

    :param n_points: Number of field points (>= 2).
    :param min_saturation: Physical saturation range. Must satisfy `0 ≤ min_saturation < max_saturation ≤ 1`.
    :param max_saturation: Physical saturation range. Must satisfy `0 ≤ min_saturation < max_saturation ≤ 1`.
    :param spacing: `"cosine"` (default) - Chebyshev-cosine spacing, denser at the
        endpoints. `"linear"` - uniform spacing.
    :return: NDArray of shape `(n_points,)` with values in `[min_saturation, max_saturation]`,
    monotonically increasing.
    """
    if n_points < 2:
        raise ValueError(f"`n_points` must be >= 2, got {n_points}")
    if not (0.0 <= min_saturation < max_saturation <= 1.0):
        raise ValueError(
            f"Require 0 ≤ min_saturation < max_saturation ≤ 1, got `min_saturation={min_saturation}`, `max_saturation={max_saturation}`"
        )

    if spacing == "cosine":
        i = np.arange(n_points, dtype=dtype)
        unit: npt.NDArray = 0.5 * (1.0 - np.cos(np.pi * i / (n_points - 1)))
    elif spacing == "linear":
        unit: npt.NDArray = np.linspace(0.0, 1.0, n_points)
    else:
        raise ValueError(f"`spacing` must be 'cosine' or 'linspace', got '{spacing}'")

    return (min_saturation + unit * (max_saturation - min_saturation)).astype(dtype, copy=False)


def make_min_span_saturation_field(
    number_of_points: int,
    min_saturation: Number,
    max_saturation: Number,
    spacing: Spacing,
    minimum_span: Number,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray:
    """
    Build a saturation grid, enforcing a floor of `minimum_span` on the
    total range so the grid never collapses to a single point.

    `minimum_span` is a plain positional argument rather than keyword-only
    so that Numba can inline this function without a typing error.

    :param number_of_points: Number of points in the output grid.
    :param min_saturation: Lower bound of the saturation range.
    :param max_saturation: Upper bound of the saturation range.
    :param spacing: Grid spacing strategy (e.g. `'cosine'`, `'linspace'`).
    :param minimum_span: Minimum permitted distance between the lower and upper
        bounds. When the natural span is smaller, the upper bound is extended to
        `min_saturation + minimum_span`.
    :return: 1-D array of saturation values.
    """
    if max_saturation - min_saturation < minimum_span:
        return np.array(
            [
                min_saturation,
                max(min_saturation + minimum_span, max_saturation),
            ],
            dtype=dtype,
        )
    return make_saturation_field(
        n_points=number_of_points,
        min_saturation=min_saturation,
        max_saturation=max_saturation,
        spacing=spacing,
        dtype=dtype,
    )


def pchip_resample(
    source_saturations: npt.NDArray,
    source_values: npt.NDArray,
    number_of_output_points: int,
    spacing: Spacing,
    dtype: npt.DTypeLike = np.float64,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Fit a PCHIP interpolant through (`source_saturations`, `source_values`),
    resample at `number_of_output_points` and return the resampled values.

    :param source_saturations: Original saturation knots, strictly increasing.
    :param source_values: Function values at each knot (kr or Pc).
    :param number_of_output_points: Number of points in the resampled grid.
    :param spacing: Grid spacing strategy for the output grid.
    :return: Two-tuple of `(resampled_saturations, resampled_values)`.
    """
    interpolant = PchipInterpolator(source_saturations, source_values)
    resampled_saturations = make_saturation_field(
        n_points=number_of_output_points,
        min_saturation=source_saturations[0],
        max_saturation=source_saturations[-1],
        spacing=spacing,
        dtype=dtype,
    )
    return resampled_saturations, interpolant(resampled_saturations)


def build_saturation_reference_field(
    number_of_base_points: int,
    saturation_lower_bound: Number,
    saturation_upper_bound: Number,
    spacing: Spacing,
    number_of_endpoint_extra_points: int,
    minimum_grid_span: Number = 1e-6,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray:
    """
    Build a saturation reference field with optional endpoint refinement.

    The base field spans [`saturation_lower_bound`, `saturation_upper_bound`]
    at `number_of_base_points` using `spacing`. When
    `number_of_endpoint_extra_points` > 0, extra knots are injected into the
    first and last 10 % of the range to capture the rapid variation of kr and
    Pc curves near residual saturations.

    :param number_of_base_points: Number of evenly-spaced base knots.
    :param saturation_lower_bound: Left end of the saturation range.
    :param saturation_upper_bound: Right end of the saturation range.
    :param spacing: Grid spacing strategy (e.g. `'cosine'`, `'linspace'`).
    :param number_of_endpoint_extra_points: Number of extra knots added inside
        each boundary decade. Pass `0` to disable endpoint refinement.
    :return: Sorted 1-D array of unique saturation values.
    """
    base_field = make_min_span_saturation_field(
        number_of_base_points,
        saturation_lower_bound,
        saturation_upper_bound,
        spacing,
        minimum_grid_span,
        dtype=dtype,
    )
    saturation_span = saturation_upper_bound - saturation_lower_bound
    if saturation_span < minimum_grid_span or number_of_endpoint_extra_points <= 0:
        return base_field

    endpoint_decade_width = 0.10 * saturation_span
    lower_endpoint_refinement = np.linspace(
        saturation_lower_bound,
        saturation_lower_bound + endpoint_decade_width,
        number_of_endpoint_extra_points + 2,
        dtype=dtype,
    )
    upper_endpoint_refinement = np.linspace(
        saturation_upper_bound - endpoint_decade_width,
        saturation_upper_bound,
        number_of_endpoint_extra_points + 2,
        dtype=dtype,
    )
    return np.unique(
        np.concatenate(
            (
                base_field,
                lower_endpoint_refinement,
                upper_endpoint_refinement,
            ),
            dtype=dtype,
        )
    )


def build_pchip_interpolant(
    reference_saturation: npt.NDArray,
    values: npt.NDArray,
    number_of_base_points: int,
    number_of_endpoint_extra_points: int,
    spacing: Spacing,
    minimum_scale_span: Number = 1e-6,
    dtype: npt.DTypeLike = np.float64,
) -> tuple[PchipInterpolator, PchipInterpolator]:
    """
    Build a PCHIP interpolant (and its derivative) for a two-phase kr or Pc
    curve, optionally after expanding the knot field.

    When `number_of_base_points > 0` *and* the saturation range is wide enough,
    the raw knots are first resampled to a denser field via
    `_pchip_resample_with_derivative`. This is the same field-expansion
    strategy used by `as_three_phase_relperm_table`.

    When `derivative_values` are supplied they are used as the source for
    the initial PCHIP fit (giving a C¹-consistent seed); otherwise the raw
    `values` array is used directly.

    :param reference_saturation: Monotonically increasing saturation knots.
    :param values: Function values (kr or Pc) at each knot.
    :param derivative_values: Optional pre-computed derivative at each knot.
        Used only as the source for the initial PCHIP when present.
    :param number_of_base_points: Target number of base points for field expansion.
        Pass `0` to use the raw knots without expansion.
    :param number_of_endpoint_extra_points: Extra knots in each boundary decade during
        expansion.  Ignored when `number_of_base_points == 0`.
    :param spacing: Grid spacing mode passed to
        `_build_saturation_reference_grid`.
    :param minimum_scale_span: Minimum saturation span required before field scaling is attempted.
    :return: Two-tuple `(interpolant, derivative_interpolant)` where
        `derivative_interpolant` is `interpolant.derivative()`.
    """
    saturation_field = reference_saturation
    vals = values

    span = saturation_field[-1] - saturation_field[0]
    should_scale = (
        number_of_base_points > 0
        and len(saturation_field) < number_of_base_points
        and span > minimum_scale_span
    )
    if should_scale:
        expanded_saturation_field = build_saturation_reference_field(
            number_of_base_points=number_of_base_points,
            saturation_lower_bound=saturation_field[0],
            saturation_upper_bound=saturation_field[-1],
            spacing=spacing,
            number_of_endpoint_extra_points=number_of_endpoint_extra_points,
            dtype=dtype,
        )
        # Fit a temporary PCHIP on the raw knots to resample onto the expanded field
        source_pchip = PchipInterpolator(saturation_field, vals)
        vals = source_pchip(expanded_saturation_field)
        saturation_field = expanded_saturation_field

    interpolant = PchipInterpolator(saturation_field, vals)
    derivative_interpolant: PchipInterpolator = interpolant.derivative(1)
    return interpolant, derivative_interpolant


@numba.njit(cache=True)
def compute_killough_scanning_curve_scalar(
    saturation: Number,
    drainage_curve_value: Number,
    imbibition_curve_value: Number,
    reversal_saturation: Number,
    maximum_historical_saturation: Number,
    is_imbibition: Number,
    scanning_exponent: Number = 1.0,
    numerical_epsilon: Number = 1e-12,
) -> Number:
    """
    Killough scanning-curve interpolation between primary drainage and
    imbibition curves (scalar).

    When the flow direction reverses the reservoir property (kr or Pc)
    follows a scanning curve interpolated between the two primary curves::

        value = value_drain + (value_imb - value_drain) * f(S)

    where the interpolation factor is::

        f(S) = clamp( ((S - S_rev) / (S_max - S_rev))^n , 0, 1 )

    :param saturation: Current saturation.
    :param drainage_curve_value: Value from the primary drainage curve.
    :param imbibition_curve_value: Value from the primary imbibition curve.
    :param reversal_saturation: Saturation at the last reversal point (S_rev).
    :param maximum_historical_saturation: Maximum saturation reached before the reversal (S_max).
    :param is_imbibition: 1.0 if currently on the imbibition path, 0.0 if drainage.
    :param scanning_exponent: Killough interpolation exponent *n* (1 = linear).
    :param numerical_epsilon: Numerical stability tolerance.
    :return: Interpolated scanning-curve value.
    """
    saturation_span = maximum_historical_saturation - reversal_saturation
    if abs(saturation_span) > numerical_epsilon:
        raw_interpolation_factor = (saturation - reversal_saturation) / saturation_span
    else:
        raw_interpolation_factor = 0.0

    interpolation_factor = min(max(raw_interpolation_factor**scanning_exponent, 0.0), 1.0)
    scanning_curve_value = (
        drainage_curve_value
        + (imbibition_curve_value - drainage_curve_value) * interpolation_factor
    )

    on_primary_drainage = (is_imbibition < 0.5) and (
        abs(saturation - maximum_historical_saturation) < numerical_epsilon
    )
    on_primary_imbibition = (is_imbibition >= 0.5) and (
        abs(saturation - reversal_saturation) < numerical_epsilon
    )

    if on_primary_drainage:
        return drainage_curve_value
    if on_primary_imbibition:
        return imbibition_curve_value
    return scanning_curve_value


@numba.njit(cache=True)
def compute_killough_scanning_curve_array(
    saturation: NumberOrArray[NDimension],
    drainage_curve_value: NumberOrArray[NDimension],
    imbibition_curve_value: NumberOrArray[NDimension],
    reversal_saturation: NumberOrArray[NDimension],
    maximum_historical_saturation: NumberOrArray[NDimension],
    is_imbibition: NumberOrArray[NDimension],
    scanning_exponent: Number = 1.0,
    numerical_epsilon: Number = 1e-12,
) -> NumberArray[NDimension]:
    """
    Killough scanning-curve interpolation between primary drainage and
    imbibition curves (array).

    :param saturation: Current saturation array.
    :param drainage_curve_value: Drainage curve values at current saturations.
    :param imbibition_curve_value: Imbibition curve values at current saturations.
    :param reversal_saturation: Saturations at the last reversal points.
    :param maximum_historical_saturation: Maximum saturations reached before the reversals.
    :param is_imbibition: Per-cell imbibition flag (1.0 = imbibition, 0.0 = drainage).
    :param scanning_exponent: Killough interpolation exponent *n* (1 = linear).
    :param numerical_epsilon: Numerical stability tolerance.
    :return: Interpolated scanning-curve values with the same shape as `saturation`.
    """
    saturation_array = atleast_1d(saturation)
    drainage_value_array = atleast_1d(drainage_curve_value)
    imbibition_value_array = atleast_1d(imbibition_curve_value)
    reversal_saturation_array = atleast_1d(reversal_saturation)
    maximum_saturation_array = atleast_1d(maximum_historical_saturation)
    imbibition_flag_array = atleast_1d(is_imbibition)

    # broadcast to common shape
    (
        saturation_array,
        drainage_value_array,
        imbibition_value_array,
        reversal_saturation_array,
        maximum_saturation_array,
        imbibition_flag_array,
    ) = np.broadcast_arrays(
        saturation_array,
        drainage_value_array,
        imbibition_value_array,
        reversal_saturation_array,
        maximum_saturation_array,
        imbibition_flag_array,
    )

    result = np.empty_like(saturation_array)
    for flat_idx in numba.prange(saturation_array.size):  # type: ignore
        saturation_span = (
            maximum_saturation_array.flat[flat_idx] - reversal_saturation_array.flat[flat_idx]
        )
        if abs(saturation_span) > numerical_epsilon:
            raw_interpolation_factor = (
                saturation_array.flat[flat_idx] - reversal_saturation_array.flat[flat_idx]
            ) / saturation_span
        else:
            raw_interpolation_factor = 0.0
        interpolation_factor = min(max(raw_interpolation_factor**scanning_exponent, 0.0), 1.0)
        scanning_curve_value = (
            drainage_value_array.flat[flat_idx]
            + (imbibition_value_array.flat[flat_idx] - drainage_value_array.flat[flat_idx])
            * interpolation_factor
        )

        on_primary_drainage = (imbibition_flag_array.flat[flat_idx] < 0.5) and (
            abs(saturation_array.flat[flat_idx] - maximum_saturation_array.flat[flat_idx])
            < numerical_epsilon
        )
        on_primary_imbibition = (imbibition_flag_array.flat[flat_idx] >= 0.5) and (
            abs(saturation_array.flat[flat_idx] - reversal_saturation_array.flat[flat_idx])
            < numerical_epsilon
        )

        if on_primary_drainage:
            result.flat[flat_idx] = drainage_value_array.flat[flat_idx]
        elif on_primary_imbibition:
            result.flat[flat_idx] = imbibition_value_array.flat[flat_idx]
        else:
            result.flat[flat_idx] = scanning_curve_value
    return result  # type: ignore


def compute_killough_scanning_curve(
    saturation: NumberOrArray[NDimension],
    drainage_curve_value: NumberOrArray[NDimension],
    imbibition_curve_value: NumberOrArray[NDimension],
    reversal_saturation: NumberOrArray[NDimension],
    maximum_historical_saturation: NumberOrArray[NDimension],
    is_imbibition: NumberOrArray[NDimension],
    scanning_exponent: Number = 1.0,
    numerical_epsilon: Number = 1e-12,
) -> NumberOrArray[NDimension]:
    """
    Killough scanning-curve interpolation between primary drainage and
    imbibition curves.

    :param saturation: Current saturation (scalar or array).
    :param drainage_curve_value: Value from the primary drainage curve.
    :param imbibition_curve_value: Value from the primary imbibition curve.
    :param reversal_saturation: Saturation at the last reversal (scalar or array).
    :param maximum_historical_saturation: Maximum saturation before the reversal (scalar or array).
    :param is_imbibition: Imbibition flag - 1.0 for imbibition, 0.0 for drainage (scalar or array).
    :param scanning_exponent: Killough interpolation exponent *n* (1 = linear).
    :param numerical_epsilon: Numerical stability tolerance.
    :return: Interpolated scanning-curve value matching the shape of the input.
    """
    is_scalar = (
        np.isscalar(saturation)
        and np.isscalar(drainage_curve_value)
        and np.isscalar(imbibition_curve_value)
        and np.isscalar(reversal_saturation)
        and np.isscalar(maximum_historical_saturation)
        and np.isscalar(is_imbibition)
    )
    if is_scalar:
        return compute_killough_scanning_curve_scalar(
            saturation=float(saturation),  # type: ignore[arg-type]
            drainage_curve_value=float(drainage_curve_value),  # type: ignore[arg-type]
            imbibition_curve_value=float(imbibition_curve_value),  # type: ignore[arg-type]
            reversal_saturation=float(reversal_saturation),  # type: ignore[arg-type]
            maximum_historical_saturation=float(maximum_historical_saturation),  # type: ignore[arg-type]
            is_imbibition=float(is_imbibition),  # type: ignore[arg-type]
            scanning_exponent=scanning_exponent,
            numerical_epsilon=numerical_epsilon,
        )
    return compute_killough_scanning_curve_array(
        saturation=saturation,  # type: ignore[arg-type]
        drainage_curve_value=drainage_curve_value,  # type: ignore[arg-type]
        imbibition_curve_value=imbibition_curve_value,  # type: ignore[arg-type]
        reversal_saturation=reversal_saturation,  # type: ignore[arg-type]
        maximum_historical_saturation=maximum_historical_saturation,  # type: ignore[arg-type]
        is_imbibition=is_imbibition,  # type: ignore[arg-type]
        scanning_exponent=scanning_exponent,
        numerical_epsilon=numerical_epsilon,
    )


@numba.njit(cache=True, inline="always")
def compute_killough_scanning_curve_derivative_scalar(
    saturation: Number,
    drainage_curve_value: Number,
    imbibition_curve_value: Number,
    drainage_curve_derivative: Number,
    imbibition_curve_derivative: Number,
    reversal_saturation: Number,
    maximum_historical_saturation: Number,
    is_imbibition: Number,
    scanning_exponent: Number = 1.0,
    numerical_epsilon: Number = 1e-12,
) -> Number:
    """
    Analytical derivative of the Killough scanning-curve value with respect
    to the scanning saturation (scalar).

    The scanning curve is:

        V(S) = V_d(S) + [V_i(S) - V_d(S)] * f(S)

    By the product / chain rule:

        dV/dS = dV_d/dS + [dV_i/dS - dV_d/dS] * f
                        + [V_i - V_d] * df/dS

    :param saturation: Current saturation.
    :param drainage_curve_value: Drainage curve value at current saturation.
    :param imbibition_curve_value: Imbibition curve value at current saturation.
    :param drainage_curve_derivative: Derivative of drainage value w.r.t. saturation.
    :param imbibition_curve_derivative: Derivative of imbibition value w.r.t. saturation.
    :param reversal_saturation: Saturation at the last reversal (S_rev).
    :param maximum_historical_saturation: Maximum saturation before the reversal (S_max).
    :param is_imbibition: 1.0 if imbibition, 0.0 if drainage.
    :param scanning_exponent: Killough exponent *n*.
    :param numerical_epsilon: Numerical tolerance.
    :return: Derivative of scanning-curve value w.r.t. saturation.
    """
    saturation_span = maximum_historical_saturation - reversal_saturation

    on_primary_drainage = (is_imbibition < 0.5) and (
        abs(saturation - maximum_historical_saturation) < numerical_epsilon
    )
    on_primary_imbibition = (is_imbibition >= 0.5) and (
        abs(saturation - reversal_saturation) < numerical_epsilon
    )
    if on_primary_drainage:
        return drainage_curve_derivative
    if on_primary_imbibition:
        return imbibition_curve_derivative

    if abs(saturation_span) > numerical_epsilon:
        raw_ratio = (saturation - reversal_saturation) / saturation_span
    else:
        raw_ratio = 0.0

    clamped_ratio = min(max(raw_ratio, 0.0), 1.0)
    interpolation_factor = clamped_ratio**scanning_exponent

    in_active_range = (
        (raw_ratio > 0.0) and (raw_ratio < 1.0) and (abs(saturation_span) > numerical_epsilon)
    )
    if in_active_range:
        if abs(scanning_exponent - 1.0) < 1e-10:
            interpolation_factor_derivative = 1.0 / saturation_span
        else:
            safe_ratio = clamped_ratio if clamped_ratio > 0.0 else 1e-30
            interpolation_factor_derivative = (
                scanning_exponent * (safe_ratio ** (scanning_exponent - 1.0)) / saturation_span
            )
    else:
        interpolation_factor_derivative = 0.0

    return (
        drainage_curve_derivative
        + (imbibition_curve_derivative - drainage_curve_derivative) * interpolation_factor
        + (imbibition_curve_value - drainage_curve_value) * interpolation_factor_derivative
    )


@numba.njit(cache=True)
def compute_killough_scanning_curve_derivative_array(
    saturation: NumberOrArray[NDimension],
    drainage_curve_value: NumberOrArray[NDimension],
    imbibition_curve_value: NumberOrArray[NDimension],
    drainage_curve_derivative: NumberOrArray[NDimension],
    imbibition_curve_derivative: NumberOrArray[NDimension],
    reversal_saturation: NumberOrArray[NDimension],
    maximum_historical_saturation: NumberOrArray[NDimension],
    is_imbibition: NumberOrArray[NDimension],
    scanning_exponent: Number = 1.0,
    numerical_epsilon: Number = 1e-12,
) -> NumberArray[NDimension]:
    """
    Analytical derivative of the Killough scanning-curve value with respect
    to the scanning saturation (array).

    :param saturation: Current saturation array.
    :param drainage_curve_value: Drainage curve values at current saturations.
    :param imbibition_curve_value: Imbibition curve values at current saturations.
    :param drainage_curve_derivative: Derivatives of drainage values w.r.t. saturation.
    :param imbibition_curve_derivative: Derivatives of imbibition values w.r.t. saturation.
    :param reversal_saturation: Saturations at the last reversal points.
    :param maximum_historical_saturation: Maximum saturations before the reversals.
    :param is_imbibition: Per-cell imbibition flags (1.0 / 0.0).
    :param scanning_exponent: Killough exponent *n*.
    :param numerical_epsilon: Numerical tolerance.
    :return: Derivative array with the same shape as `saturation`.
    """
    saturation_array = atleast_1d(saturation)
    drainage_value_array = atleast_1d(drainage_curve_value)
    imbibition_value_array = atleast_1d(imbibition_curve_value)
    drainage_derivative_array = atleast_1d(drainage_curve_derivative)
    imbibition_derivative_array = atleast_1d(imbibition_curve_derivative)
    reversal_saturation_array = atleast_1d(reversal_saturation)
    maximum_saturation_array = atleast_1d(maximum_historical_saturation)
    imbibition_flag_array = atleast_1d(is_imbibition)

    (
        saturation_array,
        drainage_value_array,
        imbibition_value_array,
        drainage_derivative_array,
        imbibition_derivative_array,
        reversal_saturation_array,
        maximum_saturation_array,
        imbibition_flag_array,
    ) = np.broadcast_arrays(
        saturation_array,
        drainage_value_array,
        imbibition_value_array,
        drainage_derivative_array,
        imbibition_derivative_array,
        reversal_saturation_array,
        maximum_saturation_array,
        imbibition_flag_array,
    )

    result = np.empty_like(saturation_array)
    for flat_idx in numba.prange(saturation_array.size):  # type: ignore
        result.flat[flat_idx] = compute_killough_scanning_curve_derivative_scalar(
            saturation=saturation_array.flat[flat_idx],
            drainage_curve_value=drainage_value_array.flat[flat_idx],
            imbibition_curve_value=imbibition_value_array.flat[flat_idx],
            drainage_curve_derivative=drainage_derivative_array.flat[flat_idx],
            imbibition_curve_derivative=imbibition_derivative_array.flat[flat_idx],
            reversal_saturation=reversal_saturation_array.flat[flat_idx],
            maximum_historical_saturation=maximum_saturation_array.flat[flat_idx],
            is_imbibition=imbibition_flag_array.flat[flat_idx],
            scanning_exponent=scanning_exponent,
            numerical_epsilon=numerical_epsilon,
        )
    return result  # type: ignore


def compute_killough_scanning_curve_derivative(
    saturation: NumberOrArray[NDimension],
    drainage_curve_value: NumberOrArray[NDimension],
    imbibition_curve_value: NumberOrArray[NDimension],
    drainage_curve_derivative: NumberOrArray[NDimension],
    imbibition_curve_derivative: NumberOrArray[NDimension],
    reversal_saturation: NumberOrArray[NDimension],
    maximum_historical_saturation: NumberOrArray[NDimension],
    is_imbibition: NumberOrArray[NDimension],
    scanning_exponent: Number = 1.0,
    numerical_epsilon: Number = 1e-12,
) -> NumberOrArray[NDimension]:
    """
    Analytical derivative of the Killough scanning-curve value with respect
    to the scanning saturation.

    :param saturation: Current saturation (scalar or array).
    :param drainage_curve_value: Drainage curve value at current saturation.
    :param imbibition_curve_value: Imbibition curve value at current saturation.
    :param drainage_curve_derivative: Derivative of drainage value w.r.t. saturation.
    :param imbibition_curve_derivative: Derivative of imbibition value w.r.t. saturation.
    :param reversal_saturation: Saturation at the last reversal (scalar or array).
    :param maximum_historical_saturation: Maximum saturation before the reversal (scalar or array).
    :param is_imbibition: Imbibition flag (scalar or array).
    :param scanning_exponent: Killough exponent *n*.
    :param numerical_epsilon: Numerical tolerance.
    :return: Derivative of scanning-curve value w.r.t. saturation, matching the shape of the input.
    """
    is_scalar = (
        np.isscalar(saturation)
        and np.isscalar(drainage_curve_value)
        and np.isscalar(imbibition_curve_value)
        and np.isscalar(drainage_curve_derivative)
        and np.isscalar(imbibition_curve_derivative)
        and np.isscalar(reversal_saturation)
        and np.isscalar(maximum_historical_saturation)
        and np.isscalar(is_imbibition)
    )
    if is_scalar:
        return compute_killough_scanning_curve_derivative_scalar(
            saturation=float(saturation),  # type: ignore[arg-type]
            drainage_curve_value=float(drainage_curve_value),  # type: ignore[arg-type]
            imbibition_curve_value=float(imbibition_curve_value),  # type: ignore[arg-type]
            drainage_curve_derivative=float(drainage_curve_derivative),  # type: ignore[arg-type]
            imbibition_curve_derivative=float(imbibition_curve_derivative),  # type: ignore[arg-type]
            reversal_saturation=float(reversal_saturation),  # type: ignore[arg-type]
            maximum_historical_saturation=float(maximum_historical_saturation),  # type: ignore[arg-type]
            is_imbibition=float(is_imbibition),  # type: ignore[arg-type]
            scanning_exponent=scanning_exponent,
            numerical_epsilon=numerical_epsilon,
        )
    return compute_killough_scanning_curve_derivative_array(
        saturation=saturation,  # type: ignore[arg-type]
        drainage_curve_value=drainage_curve_value,  # type: ignore[arg-type]
        imbibition_curve_value=imbibition_curve_value,  # type: ignore[arg-type]
        drainage_curve_derivative=drainage_curve_derivative,  # type: ignore[arg-type]
        imbibition_curve_derivative=imbibition_curve_derivative,  # type: ignore[arg-type]
        reversal_saturation=reversal_saturation,  # type: ignore[arg-type]
        maximum_historical_saturation=maximum_historical_saturation,  # type: ignore[arg-type]
        is_imbibition=is_imbibition,  # type: ignore[arg-type]
        scanning_exponent=scanning_exponent,
        numerical_epsilon=numerical_epsilon,
    )
