import typing

import numba
import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator

from bores.grids.utils import Spacing, make_saturation_grid


@numba.njit(cache=True, inline="always")
def _make_saturation_grid_with_minimum_span(
    number_of_points: int,
    saturation_minimum: float,
    saturation_maximum: float,
    spacing: Spacing,
    minimum_span: float,
) -> npt.NDArray:
    """
    Build a saturation grid, enforcing a floor of `minimum_span` on the
    total range so the grid never collapses to a single point.

    `minimum_span` is a plain positional argument rather than keyword-only
    so that Numba can inline this function without a typing error.

    :param number_of_points: Number of points in the output grid.
    :param saturation_minimum: Lower bound of the saturation range.
    :param saturation_maximum: Upper bound of the saturation range.
    :param spacing: Grid spacing strategy (e.g. `'cosine'`, `'linspace'`).
    :param minimum_span: Minimum permitted distance between the lower and upper
        bounds. When the natural span is smaller, the upper bound is extended to
        `saturation_minimum + minimum_span`.
    :return: 1-D array of saturation values.
    """
    if saturation_maximum - saturation_minimum < minimum_span:
        return np.array(
            [
                saturation_minimum,
                max(saturation_minimum + minimum_span, saturation_maximum),
            ],
            dtype=np.float64,
        )
    return make_saturation_grid(
        n_points=number_of_points,
        s_min=saturation_minimum,
        s_max=saturation_maximum,
        spacing=spacing,
        dtype=np.float64,
    )


def pchip_resample(
    source_saturations: npt.NDArray,
    source_values: npt.NDArray,
    number_of_output_points: int,
    spacing: Spacing,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
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
    resampled_saturations = make_saturation_grid(
        n_points=number_of_output_points,
        s_min=float(source_saturations[0]),
        s_max=float(source_saturations[-1]),
        spacing=spacing,
        dtype=np.float64,
    )
    return resampled_saturations, interpolant(resampled_saturations)


@numba.njit(cache=True)
def build_saturation_reference_grid(
    number_of_base_points: int,
    saturation_lower_bound: float,
    saturation_upper_bound: float,
    spacing: Spacing,
    number_of_endpoint_extra_points: int,
    minimum_grid_span: float = 1e-6,
) -> npt.NDArray:
    """
    Build a saturation reference grid with optional endpoint refinement.

    The base grid spans [`saturation_lower_bound`, `saturation_upper_bound`]
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
    base_grid = _make_saturation_grid_with_minimum_span(
        number_of_base_points,
        saturation_lower_bound,
        saturation_upper_bound,
        spacing,
        minimum_grid_span,
    )
    saturation_span = saturation_upper_bound - saturation_lower_bound
    if saturation_span < minimum_grid_span or number_of_endpoint_extra_points <= 0:
        return base_grid

    endpoint_decade_width = 0.10 * saturation_span
    lower_endpoint_refinement = np.linspace(
        saturation_lower_bound,
        saturation_lower_bound + endpoint_decade_width,
        number_of_endpoint_extra_points + 2,
    )
    upper_endpoint_refinement = np.linspace(
        saturation_upper_bound - endpoint_decade_width,
        saturation_upper_bound,
        number_of_endpoint_extra_points + 2,
    )
    return np.unique(
        np.concatenate(
            (base_grid, lower_endpoint_refinement, upper_endpoint_refinement)
        )
    )


def build_pchip_interpolant(
    reference_saturation: npt.NDArray,
    values: npt.NDArray,
    number_of_base_points: int,
    number_of_endpoint_extra_points: int,
    spacing: Spacing,
    minimum_scale_span: float = 1e-6,
) -> typing.Tuple[PchipInterpolator, PchipInterpolator]:
    """
    Build a PCHIP interpolant (and its derivative) for a two-phase kr or Pc
    curve, optionally after expanding the knot grid.

    When `number_of_base_points > 0` *and* the saturation range is wide enough,
    the raw knots are first resampled to a denser grid via
    `_pchip_resample_with_derivative`.  This is the same grid-expansion
    strategy used by `as_three_phase_relperm_table`.

    When `derivative_values` are supplied they are used as the source for
    the initial PCHIP fit (giving a C¹-consistent seed); otherwise the raw
    `values` array is used directly.

    :param reference_saturation: Monotonically increasing saturation knots.
    :param values: Function values (kr or Pc) at each knot.
    :param derivative_values: Optional pre-computed derivative at each knot.
        Used only as the source for the initial PCHIP when present.
    :param number_of_base_points: Target number of base points for grid expansion.
        Pass `0` to use the raw knots without expansion.
    :param number_of_endpoint_extra_points: Extra knots in each boundary decade during
        expansion.  Ignored when `number_of_base_points == 0`.
    :param spacing: Grid spacing mode passed to
        `_build_saturation_reference_grid`.
    :param minimum_scale_span: Minimum saturation span required before grid scaling is attempted.
    :return: Two-tuple `(interpolant, derivative_interpolant)` where
        `derivative_interpolant` is `interpolant.derivative()`.
    """
    sat = reference_saturation
    vals = values

    span = float(sat[-1]) - float(sat[0])
    should_scale = (
        number_of_base_points > 0
        and len(sat) < number_of_base_points
        and span > minimum_scale_span
    )
    if should_scale:
        expanded_sat = build_saturation_reference_grid(
            number_of_base_points=number_of_base_points,
            saturation_lower_bound=float(sat[0]),
            saturation_upper_bound=float(sat[-1]),
            spacing=spacing,
            number_of_endpoint_extra_points=number_of_endpoint_extra_points,
        )
        # Fit a temporary PCHIP on the raw knots to resample onto the expanded grid
        source_pchip = PchipInterpolator(sat, vals)
        vals = source_pchip(expanded_sat)
        sat = expanded_sat

    interpolant = PchipInterpolator(sat, vals)
    derivative_interpolant: PchipInterpolator = interpolant.derivative(1)
    return interpolant, derivative_interpolant
