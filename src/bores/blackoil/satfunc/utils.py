import typing

import numba
import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator

from bores.errors import ValidationError
from bores.types import (
    Boolean,
    Integer,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
    Spacing,
)
from bores.utils import atleast_1d

PARALLEL_SIZE_THRESHOLD = 10_000
"""Number of cells from which the satfunc kernels run their parallel variant."""


def make_saturation_field(
    n_points: Integer = 200,
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
    :param dtype: Floating-point dtype of the returned field.
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
        unit = np.linspace(0.0, 1.0, n_points, dtype=dtype)
    else:
        raise ValueError(f"`spacing` must be 'cosine' or 'linear', got '{spacing}'")

    return (min_saturation + unit * (max_saturation - min_saturation)).astype(dtype, copy=False)


def make_min_span_saturation_field(
    number_of_points: Integer,
    min_saturation: Number,
    max_saturation: Number,
    spacing: Spacing,
    minimum_span: Number,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray:
    """
    Build a saturation grid, enforcing a floor of `minimum_span` on the
    total range so the grid never collapses to a single point.

    :param number_of_points: Number of points in the output grid.
    :param min_saturation: Lower bound of the saturation range.
    :param max_saturation: Upper bound of the saturation range.
    :param spacing: Grid spacing strategy (`'cosine'` or `'linear'`).
    :param minimum_span: Minimum permitted distance between the lower and upper
        bounds. When the natural span is smaller, the upper bound is extended to
        `min_saturation + minimum_span` and a two-point grid is returned.
    :param dtype: Floating-point dtype of the returned grid.
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
    number_of_output_points: Integer | None = None,
    spacing: Spacing = "cosine",
    dtype: npt.DTypeLike = np.float64,
    target_saturations: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Fit a PCHIP interpolant through (`source_saturations`, `source_values`),
    evaluate it on a new saturation grid and return both the grid and the
    interpolated values.

    The new grid is `target_saturations` when given, otherwise a field of
    `number_of_output_points` spanning the source range, laid out per `spacing`.

    Targets outside the source range are evaluated at the nearest source end
    (the curve is held constant beyond its first and last knot) rather than
    extrapolated, since PCHIP extrapolation of a kr or Pc curve is unbounded
    and can leave the physical range. The returned saturations are the targets
    as supplied.

    :param source_saturations: Original saturation knots, strictly increasing.
    :param source_values: Function values at each knot (kr or Pc).
    :param number_of_output_points: Number of points in the resampled grid. Required
        unless `target_saturations` is given, in which case it is ignored.
    :param spacing: Grid spacing strategy for the generated output grid. Ignored
        when `target_saturations` is given.
    :param dtype: Floating-point dtype of the resampled saturations.
    :param target_saturations: Explicit 1-D saturation grid to evaluate at.
    :return: Two-tuple of `(resampled_saturations, resampled_values)`.
    :raises ValueError: When neither `target_saturations` nor `number_of_output_points`
        is given.
    """
    interpolant = PchipInterpolator(source_saturations, source_values)
    if target_saturations is not None:
        resampled_saturations = np.asarray(target_saturations, dtype=dtype)
        evaluation_points = np.clip(
            resampled_saturations, source_saturations[0], source_saturations[-1]
        )
        return resampled_saturations, interpolant(evaluation_points)

    if number_of_output_points is None:
        raise ValueError("Provide either `number_of_output_points` or `target_saturations`.")
    resampled_saturations = make_saturation_field(
        n_points=number_of_output_points,
        min_saturation=source_saturations[0],
        max_saturation=source_saturations[-1],
        spacing=spacing,
        dtype=dtype,
    )
    return resampled_saturations, interpolant(resampled_saturations)


def build_saturation_reference_field(
    number_of_base_points: Integer,
    saturation_lower_bound: Number,
    saturation_upper_bound: Number,
    spacing: Spacing,
    number_of_endpoint_extra_points: Integer,
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

    :param number_of_base_points: Number of base knots, laid out per `spacing`.
    :param saturation_lower_bound: Left end of the saturation range.
    :param saturation_upper_bound: Right end of the saturation range.
    :param spacing: Grid spacing strategy (`'cosine'` or `'linear'`).
    :param number_of_endpoint_extra_points: Number of extra knots added inside
        each end region (first and last 10 % of the range). Pass `0` to disable
        endpoint refinement.
    :param minimum_grid_span: Minimum saturation span. Below this the range is
        treated as collapsed and no refinement is added.
    :param dtype: Floating-point dtype of the returned field.
    :return: Sorted 1-D array of saturation values, strictly increasing, with
        knots closer than a few ulps merged.
    """
    base_field = make_min_span_saturation_field(
        number_of_points=number_of_base_points,
        min_saturation=saturation_lower_bound,
        max_saturation=saturation_upper_bound,
        spacing=spacing,
        minimum_span=minimum_grid_span,
        dtype=dtype,
    )
    saturation_span = saturation_upper_bound - saturation_lower_bound
    if saturation_span < minimum_grid_span or number_of_endpoint_extra_points <= 0:
        return base_field

    endpoint_refinement_width = 0.10 * saturation_span
    lower_endpoint_refinement = np.linspace(
        saturation_lower_bound,
        saturation_lower_bound + endpoint_refinement_width,
        number_of_endpoint_extra_points + 2,
        dtype=dtype,
    )
    upper_endpoint_refinement = np.linspace(
        saturation_upper_bound - endpoint_refinement_width,
        saturation_upper_bound,
        number_of_endpoint_extra_points + 2,
        dtype=dtype,
    )
    merged_field = np.unique(
        np.concatenate(
            (
                base_field,
                lower_endpoint_refinement,
                upper_endpoint_refinement,
            ),
            dtype=dtype,
        )
    )
    # `np.unique` only drops bit-identical values. Knots produced by different
    # generators (e.g. a linear base field and an endpoint refinement) can land
    # a few ulps apart, and PCHIP slopes are ill-conditioned across such a tiny
    # interval. Collapse knots closer than a small tolerance, keeping the first
    # of each cluster, then restore the exact upper bound.
    tolerance = max(1e-9, 16.0 * float(np.finfo(dtype).eps)) * saturation_span  # type: ignore
    keep = np.concatenate(([True], np.diff(merged_field) > tolerance))
    deduplicated_field = merged_field[keep]
    deduplicated_field[-1] = merged_field[-1]
    return deduplicated_field


def build_pchip_interpolant(
    reference_saturation: npt.NDArray,
    values: npt.NDArray,
    number_of_base_points: Integer,
    number_of_endpoint_extra_points: Integer,
    spacing: Spacing,
    minimum_scale_span: Number = 1e-6,
    dtype: npt.DTypeLike = np.float64,
) -> tuple[PchipInterpolator, PchipInterpolator]:
    """
    Build a PCHIP interpolant (and its derivative) for a two-phase kr or Pc
    curve, optionally after expanding the knot field.

    The knot field is expanded only when `number_of_base_points > 0`, the raw
    knots number fewer than `number_of_base_points`, *and* the saturation range
    exceeds `minimum_scale_span`. In that case the expanded field is built with
    `build_saturation_reference_field`, a temporary PCHIP fitted on the raw
    knots is evaluated on it, and the returned interpolant is fitted on those
    resampled values. Otherwise the raw knots are used as they are.

    :param reference_saturation: Monotonically increasing saturation knots.
    :param values: Function values (kr or Pc) at each knot.
    :param number_of_base_points: Target number of base knots for field expansion.
        Pass `0` to use the raw knots without expansion.
    :param number_of_endpoint_extra_points: Extra knots in each end region (first and
        last 10 % of the range) during expansion. Ignored when `number_of_base_points == 0`.
    :param spacing: Grid spacing mode (`'cosine'` or `'linear'`) for the expanded field.
    :param minimum_scale_span: Minimum saturation span required before field expansion
        is attempted. Also used as the minimum grid span of the expanded field.
    :param dtype: Floating-point dtype of the expanded field.
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
            minimum_grid_span=minimum_scale_span,
            dtype=dtype,
        )
        # Fit a temporary PCHIP on the raw knots to resample onto the expanded field
        source_pchip = PchipInterpolator(saturation_field, vals)
        vals = source_pchip(expanded_saturation_field)
        saturation_field = expanded_saturation_field

    interpolant = PchipInterpolator(saturation_field, vals)
    derivative_interpolant: PchipInterpolator = interpolant.derivative(1)
    return interpolant, derivative_interpolant


@numba.njit(cache=True, inline="always", error_model="numpy")
def compute_killough_scanning_weight(
    saturation: Number,
    reversal_saturation: Number,
    maximum_historical_saturation: Number,
    is_imbibition: Number,
    scanning_exponent: Number = 1.0,
    numerical_epsilon: Number = 1e-12,
) -> tuple[Number, Number]:
    """
    Weight of the imbibition curve on the Killough scanning curve, and its derivative with
    respect to saturation.

    A scanning curve is `value = drainage + (imbibition - drainage) * weight`, so
    `d value / d S = d drainage + (d imbibition - d drainage) * weight + (imbibition - drainage) * d weight`.
    The weight depends on saturation and history only, so every curve that is scanned over the
    same saturation (for example the water and oil relative permeabilities of the oil-water
    system) can share one weight instead of recomputing it.

    The weight is `clamp((S - S_rev) / (S_max - S_rev), 0, 1) ** n`: it is 0 at the reversal
    saturation and 1 at the historical maximum, whichever way the saturation is moving, so
    the scanning curve is continuous and meets each bounding curve at an end of the range.
    When the range collapses (`S_max == S_rev`) the weight is that of the bounding curve the
    cell is on, 1 while imbibing and 0 while draining, with a zero derivative.

    :param saturation: Current saturation.
    :param reversal_saturation: Saturation at the last reversal point (S_rev).
    :param maximum_historical_saturation: Maximum saturation reached before the reversal (S_max).
    :param is_imbibition: 1.0 if currently on the imbibition path, 0.0 if drainage.
    :param scanning_exponent: Killough interpolation exponent *n* (> 0; 1 = linear).
    :param numerical_epsilon: Numerical stability tolerance.
    :return: Tuple of the weight and its derivative with respect to saturation.
    """
    saturation_span = maximum_historical_saturation - reversal_saturation
    if abs(saturation_span) <= numerical_epsilon:
        return (1.0 if is_imbibition >= 0.5 else 0.0), 0.0

    raw_ratio = (saturation - reversal_saturation) / saturation_span
    if raw_ratio <= 0.0:
        return 0.0, 0.0
    if raw_ratio >= 1.0:
        return 1.0, 0.0

    if abs(scanning_exponent - 1.0) < 1e-10:
        return raw_ratio, 1.0 / saturation_span
    return (
        raw_ratio**scanning_exponent,
        scanning_exponent * raw_ratio ** (scanning_exponent - 1.0) / saturation_span,
    )


@numba.njit(cache=True, error_model="numpy")
def scan_curve_kernel(
    size: Integer,
    saturation: NumberArray[OneDimension],
    reversal_saturation: NumberArray[OneDimension],
    maximum_historical_saturation: NumberArray[OneDimension],
    is_imbibition: NumberArray[OneDimension],
    scanning_exponent: Number,
    numerical_epsilon: Number,
    drainage_value: NumberArray[OneDimension],
    imbibition_value: NumberArray[OneDimension],
    drainage_derivative: NumberArray[OneDimension],
    imbibition_derivative: NumberArray[OneDimension],
    with_derivatives: Boolean,
    out_value: NumberArray[OneDimension],
    out_derivative: NumberArray[OneDimension],
) -> None:
    """
    Scan one curve between its drainage and imbibition bounds for every cell, writing the
    value and (optionally) the derivative in a single pass.

    Every array argument is flat and holds either one value per cell or a single value shared
    by all cells. `out_value` and `out_derivative` must hold one value per cell, except that
    `out_derivative` may be a one-element array when `with_derivatives` is `False`.
    """
    for i in range(size):
        weight, weight_derivative = compute_killough_scanning_weight(
            saturation=element_at(array=saturation, index=i),
            reversal_saturation=element_at(array=reversal_saturation, index=i),
            maximum_historical_saturation=element_at(array=maximum_historical_saturation, index=i),
            is_imbibition=element_at(array=is_imbibition, index=i),
            scanning_exponent=scanning_exponent,
            numerical_epsilon=numerical_epsilon,
        )
        drainage = element_at(array=drainage_value, index=i)
        imbibition = element_at(array=imbibition_value, index=i)
        out_value[i] = drainage + (imbibition - drainage) * weight
        if with_derivatives:
            drainage_slope = element_at(array=drainage_derivative, index=i)
            imbibition_slope = element_at(array=imbibition_derivative, index=i)
            out_derivative[i] = (
                drainage_slope
                + (imbibition_slope - drainage_slope) * weight
                + (imbibition - drainage) * weight_derivative
            )


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

        f(S) = clamp( (S - S_rev) / (S_max - S_rev), 0, 1 ) ^ n

    The ratio is clamped *before* the exponent is applied, so saturations outside
    `[S_rev, S_max]` sit on a primary curve (f = 0 or 1) for any exponent, and
    this function stays consistent with its analytical derivative.

    :param saturation: Current saturation.
    :param drainage_curve_value: Value from the primary drainage curve.
    :param imbibition_curve_value: Value from the primary imbibition curve.
    :param reversal_saturation: Saturation at the last reversal point (S_rev).
    :param maximum_historical_saturation: Maximum saturation reached before the reversal (S_max).
    :param is_imbibition: 1.0 if currently on the imbibition path, 0.0 if drainage.
    :param scanning_exponent: Killough interpolation exponent *n* (> 0; 1 = linear).
    :param numerical_epsilon: Numerical stability tolerance.
    :return: Interpolated scanning-curve value.
    """
    weight, _ = compute_killough_scanning_weight(
        saturation=saturation,
        reversal_saturation=reversal_saturation,
        maximum_historical_saturation=maximum_historical_saturation,
        is_imbibition=is_imbibition,
        scanning_exponent=scanning_exponent,
        numerical_epsilon=numerical_epsilon,
    )
    return drainage_curve_value + (imbibition_curve_value - drainage_curve_value) * weight


def _prepare_killough_arrays(
    *values: NumberOrArray[NDimension],
) -> tuple[NumberArray[OneDimension], ...]:
    """
    Broadcast the inputs of a Killough array function to a common shape and convert them
    to contiguous float64 arrays.

    This happens outside Numba, which cannot broadcast arrays of different dtypes (e.g. a
    boolean imbibition flag against floating point curves).
    """
    broadcast = np.broadcast_arrays(*(np.atleast_1d(np.asarray(value)) for value in values))
    return typing.cast(
        tuple[NumberArray[OneDimension], ...],
        tuple(np.ascontiguousarray(array, dtype=np.float64) for array in broadcast),
    )


@numba.njit(cache=True, inline="always", error_model="numpy")
def element_at(array: NumberArray[OneDimension], index: Integer) -> Number:
    """
    Element `index` of `array`, or its only element when it holds a single value.

    Lets a kernel read a per-cell array and a single value shared by every cell with the
    same code.
    """
    return array[0] if array.size == 1 else array[index]  # type: ignore[return-value]


def prepare_kernel_inputs(
    *values: NumberOrArray[NDimension],
) -> tuple[tuple[Integer, ...], Integer, tuple[NumberArray[OneDimension], ...]]:
    """
    Bring the inputs of a per-cell kernel to flat arrays that the kernel can read by index.

    Inputs are broadcast to a common shape. An input that holds a single value is passed as
    a one-element array, which the kernel reads with `element_at`, so a value shared by
    every cell is never expanded to the full size. Arrays that are already contiguous with
    the common shape are used as they are, without a copy. Inputs that are not floating
    point are converted to float64.

    :param values: Inputs of the kernel, scalars or arrays.
    :return: Tuple of the broadcast shape (at least one-dimensional), the number of cells, and
        the flattened inputs in the order given.
    """
    arrays = [np.asarray(value) for value in values]
    shape = np.broadcast_shapes(*(array.shape for array in arrays)) or (1,)
    size = int(np.prod(shape))
    prepared = []
    for array in arrays:
        if array.dtype.kind != "f":
            array = array.astype(np.float64)
        if array.size == 1:
            prepared.append(array.reshape(1))
        else:
            prepared.append(np.ascontiguousarray(np.broadcast_to(array, shape)).ravel())
    return shape, size, tuple(prepared)


def flatten_kernel_input(
    value: typing.Any, *, shape: tuple[int, ...], as_float: bool = True
) -> NumberArray[OneDimension]:
    """
    Flat array of `value` for a per-cell kernel.

    A single value stays a one-element array that the kernel shares between all cells. Any
    other value is broadcast to `shape` and used as it is when it is already contiguous.
    Non-floating values are converted to float64 when `as_float` is set.
    """
    array = np.asarray(value)
    if as_float and array.dtype.kind != "f":
        array = array.astype(np.float64)
    if array.size == 1:
        return array.reshape(1)
    return np.ascontiguousarray(np.broadcast_to(array, shape)).reshape(-1)


@numba.njit(cache=True, error_model="numpy")
def saturations_within_bounds(
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    size: Integer,
) -> bool:
    """
    Whether every water, oil and gas saturation lies between 0 and 1.

    :param water_saturation: Flat water saturation array, or a single shared value.
    :param oil_saturation: Flat oil saturation array, or a single shared value.
    :param gas_saturation: Flat gas saturation array, or a single shared value.
    :param size: Number of cells.
    :return: `True` if all saturations are within bounds.
    """
    for i in range(size):
        sw = element_at(array=water_saturation, index=i)
        so = element_at(array=oil_saturation, index=i)
        sg = element_at(array=gas_saturation, index=i)
        if sw < 0 or sw > 1 or so < 0 or so > 1 or sg < 0 or sg > 1:
            return False
    return True


def validate_strictly_increasing(*, values: NumberArray[OneDimension], name: str) -> None:
    """
    Check that `values` increase at every step, and say where they do not.

    Interpolating a saturation table needs every saturation to be larger than the one before it,
    so repeated values are rejected as well as decreasing ones.

    :param values: One-dimensional array to check, such as the saturations of a table.
    :param name: Name of the array, used in the error message.
    :raises ValidationError: If any value does not exceed the one before it. The message gives the
        first such position, its value and the value before it, and how many positions fail.
    """
    failing = np.flatnonzero(~(np.diff(values) > 0)) + 1
    if failing.size == 0:
        return
    position = int(failing[0])
    kind = "repeats" if values[position] == values[position - 1] else "is not larger than"
    raise ValidationError(
        f"`{name}` must be strictly increasing, but entry {position} ({values[position]}) "
        f"{kind} the entry before it ({values[position - 1]}); {failing.size} of "
        f"{values.size - 1} steps fail. If the table was read from a deck, check that the "
        "tables of the different regions are each terminated by their own `/`."
    )


def get_floating_dtype(value: NumberOrArray[NDimension]) -> np.dtype:
    """Floating dtype of `value`, or float64 if it is not floating point."""
    dtype = np.asarray(value).dtype
    return dtype if np.issubdtype(dtype, np.floating) else np.dtype(np.float64)


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
    :param scanning_exponent: Killough interpolation exponent *n* (> 0; 1 = linear).
    :param numerical_epsilon: Numerical stability tolerance.
    :return: Interpolated scanning-curve values with the same shape as `saturation`.
    """
    saturation_array = atleast_1d(saturation)
    drainage_value_array = atleast_1d(drainage_curve_value)
    imbibition_value_array = atleast_1d(imbibition_curve_value)
    reversal_saturation_array = atleast_1d(reversal_saturation)
    maximum_saturation_array = atleast_1d(maximum_historical_saturation)
    imbibition_flag_array = atleast_1d(is_imbibition)

    # Curve values are floating point by nature, so the output follows their dtype
    # rather than the saturation dtype (which would truncate if it were an integer array).
    result = np.empty_like(drainage_value_array)
    for flat_idx in range(saturation_array.size):
        result.flat[flat_idx] = compute_killough_scanning_curve_scalar(
            saturation=saturation_array.flat[flat_idx],
            drainage_curve_value=drainage_value_array.flat[flat_idx],
            imbibition_curve_value=imbibition_value_array.flat[flat_idx],
            reversal_saturation=reversal_saturation_array.flat[flat_idx],
            maximum_historical_saturation=maximum_saturation_array.flat[flat_idx],
            is_imbibition=imbibition_flag_array.flat[flat_idx],
            scanning_exponent=scanning_exponent,
            numerical_epsilon=numerical_epsilon,
        )
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
            saturation=saturation,  # type: ignore[arg-type]
            drainage_curve_value=drainage_curve_value,  # type: ignore[arg-type]
            imbibition_curve_value=imbibition_curve_value,  # type: ignore[arg-type]
            reversal_saturation=reversal_saturation,  # type: ignore[arg-type]
            maximum_historical_saturation=maximum_historical_saturation,  # type: ignore[arg-type]
            is_imbibition=is_imbibition,  # type: ignore[arg-type]
            scanning_exponent=scanning_exponent,
            numerical_epsilon=numerical_epsilon,
        )
    (
        saturation_array,
        drainage_array,
        imbibition_array,
        reversal_array,
        maximum_array,
        flag_array,
    ) = _prepare_killough_arrays(
        saturation,  # type: ignore[arg-type]
        drainage_curve_value,  # type: ignore[arg-type]
        imbibition_curve_value,  # type: ignore[arg-type]
        reversal_saturation,  # type: ignore[arg-type]
        maximum_historical_saturation,  # type: ignore[arg-type]
        is_imbibition,  # type: ignore[arg-type]
    )
    result: NumberArray[OneDimension] = compute_killough_scanning_curve_array(
        saturation=saturation_array,  # type: ignore[arg-type]
        drainage_curve_value=drainage_array,  # type: ignore[arg-type]
        imbibition_curve_value=imbibition_array,  # type: ignore[arg-type]
        reversal_saturation=reversal_array,  # type: ignore[arg-type]
        maximum_historical_saturation=maximum_array,  # type: ignore[arg-type]
        is_imbibition=flag_array,  # type: ignore[arg-type]
        scanning_exponent=scanning_exponent,  # type: ignore[arg-type]
        numerical_epsilon=numerical_epsilon,  # type: ignore[arg-type]
    )
    return result.astype(get_floating_dtype(drainage_curve_value), copy=False)  # type: ignore[return-value]


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
    weight, weight_derivative = compute_killough_scanning_weight(
        saturation=saturation,
        reversal_saturation=reversal_saturation,
        maximum_historical_saturation=maximum_historical_saturation,
        is_imbibition=is_imbibition,
        scanning_exponent=scanning_exponent,
        numerical_epsilon=numerical_epsilon,
    )
    return (
        drainage_curve_derivative
        + (imbibition_curve_derivative - drainage_curve_derivative) * weight
        + (imbibition_curve_value - drainage_curve_value) * weight_derivative
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

    # Derivatives are floating point by nature, so the output follows their dtype
    # rather than the saturation dtype (which would truncate if it were an integer array).
    result = np.empty_like(drainage_derivative_array)
    for flat_idx in range(saturation_array.size):
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
    (
        saturation_array,
        drainage_value_array,
        imbibition_value_array,
        drainage_derivative_array,
        imbibition_derivative_array,
        reversal_array,
        maximum_array,
        flag_array,
    ) = _prepare_killough_arrays(
        saturation,  # type: ignore[arg-type]
        drainage_curve_value,  # type: ignore[arg-type]
        imbibition_curve_value,  # type: ignore[arg-type]
        drainage_curve_derivative,  # type: ignore[arg-type]
        imbibition_curve_derivative,  # type: ignore[arg-type]
        reversal_saturation,  # type: ignore[arg-type]
        maximum_historical_saturation,  # type: ignore[arg-type]
        is_imbibition,  # type: ignore[arg-type]
    )
    result: NumberArray[OneDimension] = compute_killough_scanning_curve_derivative_array(
        saturation=saturation_array,  # type: ignore[arg-type]
        drainage_curve_value=drainage_value_array,  # type: ignore[arg-type]
        imbibition_curve_value=imbibition_value_array,  # type: ignore[arg-type]
        drainage_curve_derivative=drainage_derivative_array,  # type: ignore[arg-type]
        imbibition_curve_derivative=imbibition_derivative_array,  # type: ignore[arg-type]
        reversal_saturation=reversal_array,  # type: ignore[arg-type]
        maximum_historical_saturation=maximum_array,  # type: ignore[arg-type]
        is_imbibition=flag_array,  # type: ignore[arg-type]
        scanning_exponent=scanning_exponent,  # type: ignore[arg-type]
        numerical_epsilon=numerical_epsilon,  # type: ignore[arg-type]
    )
    return result.astype(get_floating_dtype(drainage_curve_derivative), copy=False)  # type: ignore[return-value]
