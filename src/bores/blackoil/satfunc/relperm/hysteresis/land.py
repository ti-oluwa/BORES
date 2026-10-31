"""
Land (1968) trapping of the non-wetting phase for relative permeability hysteresis.

- Land, C.S. (1968). "Calculation of Imbibition Relative Permeability for Two- and Three-Phase
  Flow from Rock Properties". SPE 1942.
"""

import attrs
import numba
import numpy as np

from bores.blackoil.satfunc.utils import (
    element_at,
)
from bores.types import (
    Boolean,
    Integer,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
)

__all__ = [
    "LandTrapping",
    "compute_land_residual_saturation",
    "compute_land_residual_saturation_array",
    "compute_land_residual_saturation_scalar",
    "compute_land_trapping_coefficient",
    "land_residual_from_reversal",
    "rescale_scanning_saturation",
]


@numba.njit(cache=True, inline="always", error_model="numpy")
def compute_land_residual_saturation_scalar(
    initial_non_wetting_saturation: Number,
    maximum_residual_saturation: Number,
    land_trapping_coefficient: Number,
    saturation_epsilon: Number = 1e-12,
) -> Number:
    """
    Compute the dynamic residual non-wetting saturation via Land's model (scalar).

    Land (1968) relates the residual saturation that will be trapped when
    imbibition begins at `initial_non_wetting_saturation` to the maximum
    possible residual saturation observed at the drainage endpoint:

        S_r = S_i / (1 + C * S_i),    C = 1 / S_r_max - 1 / S_max

    Higher *C* means less trapping for the same initial saturation, and `C = 0` traps
    everything. The residual never exceeds the saturation the imbibition starts from, nor
    the maximum residual saturation of the drainage endpoint.

    :param initial_non_wetting_saturation: Non-wetting saturation at the
        drainage-imbibition reversal point.
    :param maximum_residual_saturation: Maximum residual saturation from the
        drainage endpoint (S_r_max).
    :param land_trapping_coefficient: Land trapping coefficient *C* (>= 0), see
        `compute_land_trapping_coefficient`.
    :param saturation_epsilon: Small value to guard against division by zero.
    :return: Dynamic residual saturation (scalar).
    """
    safe_maximum_residual = max(maximum_residual_saturation, saturation_epsilon)
    non_negative_initial = max(initial_non_wetting_saturation, 0.0)
    trapped = non_negative_initial / (
        1.0 + max(land_trapping_coefficient, 0.0) * non_negative_initial
    )
    return min(trapped, safe_maximum_residual)


@numba.njit(cache=True, inline="always", error_model="numpy")
def compute_land_trapping_coefficient(
    maximum_residual_saturation: Number,
    maximum_non_wetting_saturation: Number,
    saturation_epsilon: Number = 1e-12,
) -> Number:
    """
    Land trapping coefficient `C = 1 / S_r_max - 1 / S_max`.

    It makes the residual equal `S_r_max` when imbibition starts from the largest
    non-wetting saturation `S_max`.

    :param maximum_residual_saturation: Residual saturation after imbibition from `S_max`.
    :param maximum_non_wetting_saturation: Largest non-wetting saturation, `S_max`.
    :param saturation_epsilon: Small value to guard against division by zero.
    :return: The coefficient, never negative.
    """
    residual = max(maximum_residual_saturation, saturation_epsilon)
    largest = max(maximum_non_wetting_saturation, saturation_epsilon)
    return max(1.0 / residual - 1.0 / largest, 0.0)


@numba.njit(cache=True, parallel=True)
def compute_land_residual_saturation_array(
    initial_non_wetting_saturation: NumberArray[NDimension],
    maximum_residual_saturation: Number,
    land_trapping_coefficient: Number,
    saturation_epsilon: Number = 1e-12,
) -> NumberArray[NDimension]:
    """
    Compute the dynamic residual non-wetting saturation via Land's model (array).

    :param initial_non_wetting_saturation: Non-wetting saturations at the
        drainage-imbibition reversal points (1-D array).
    :param maximum_residual_saturation: Maximum residual saturation from the
        drainage endpoint (scalar).
    :param land_trapping_coefficient: Land trapping coefficient *C* (>= 0).
    :param saturation_epsilon: Small value to guard against division by zero.
    :return: Dynamic residual saturation array with the same shape as the input.
    """
    result = np.empty_like(initial_non_wetting_saturation)
    for index in numba.prange(initial_non_wetting_saturation.size):  # type: ignore
        result.flat[index] = compute_land_residual_saturation_scalar(
            initial_non_wetting_saturation=initial_non_wetting_saturation.flat[index],
            maximum_residual_saturation=maximum_residual_saturation,
            land_trapping_coefficient=land_trapping_coefficient,
            saturation_epsilon=saturation_epsilon,
        )
    return result  # type: ignore


def compute_land_residual_saturation(
    initial_non_wetting_saturation: NumberOrArray[NDimension],
    maximum_residual_saturation: Number,
    land_trapping_coefficient: Number,
    saturation_epsilon: Number = 1e-12,
) -> NumberOrArray[NDimension]:
    """
    Compute the dynamic residual non-wetting saturation via Land's model.

    :param initial_non_wetting_saturation: Non-wetting saturation at the
        drainage-imbibition reversal point (scalar or array).
    :param maximum_residual_saturation: Maximum residual saturation from the
        drainage endpoint (S_r_max).
    :param land_trapping_coefficient: Land trapping coefficient *C* (>= 0).
    :param saturation_epsilon: Small value to guard against division by zero.
    :return: Dynamic residual saturation matching the shape of the input.
    """
    if np.isscalar(initial_non_wetting_saturation):
        return compute_land_residual_saturation_scalar(
            initial_non_wetting_saturation=initial_non_wetting_saturation,  # type: ignore
            maximum_residual_saturation=maximum_residual_saturation,
            land_trapping_coefficient=land_trapping_coefficient,
            saturation_epsilon=saturation_epsilon,
        )
    initial = np.asarray(initial_non_wetting_saturation)
    result = compute_land_residual_saturation_array(
        initial_non_wetting_saturation=np.ascontiguousarray(initial).reshape(-1),
        maximum_residual_saturation=maximum_residual_saturation,
        land_trapping_coefficient=land_trapping_coefficient,
        saturation_epsilon=saturation_epsilon,
    )
    return result.reshape(initial.shape)  # type: ignore


@numba.njit(cache=True, error_model="numpy")
def land_residual_from_reversal(
    size: Integer,
    reversal_saturation: NumberArray[OneDimension],
    other_saturation: NumberArray[OneDimension],
    from_remaining_oil: bool,
    maximum_residual_saturation: Number,
    land_trapping_coefficient: Number,
    saturation_epsilon: Number,
    out: NumberArray[OneDimension],
) -> None:
    """
    Dynamic Land residual for every cell, in one pass and without intermediate arrays.

    When `from_remaining_oil` is set the non-wetting saturation at the reversal is the oil
    that is left, `1 - reversal - other`. Otherwise it is the reversal saturation itself.
    """
    for i in range(size):
        reversal = element_at(array=reversal_saturation, index=i)
        if from_remaining_oil:
            initial = max(0.0, 1.0 - reversal - element_at(array=other_saturation, index=i))
        else:
            initial = reversal
        out[i] = compute_land_residual_saturation_scalar(
            initial_non_wetting_saturation=initial,
            maximum_residual_saturation=maximum_residual_saturation,
            land_trapping_coefficient=land_trapping_coefficient,
            saturation_epsilon=saturation_epsilon,
        )


@numba.njit(cache=True, error_model="numpy")
def rescale_scanning_saturation(
    size: Integer,
    saturation: NumberArray[OneDimension],
    reversal_saturation: NumberArray[OneDimension],
    trapped_saturation: NumberArray[OneDimension],
    on_wetting_axis: Boolean,
    table_start: Number,
    table_end: Number,
    saturation_epsilon: Number,
    out_saturation: NumberArray[OneDimension],
    out_scale: NumberArray[OneDimension],
) -> None:
    """
    Map each cell's saturation onto the saturation axis of a tabular imbibition curve, so the
    curve ends where Land trapping says this cell's imbibition ends.

    A tabular curve has fixed endpoints, so Land's trapped saturation cannot be handed to it the
    way it is to a parametric curve. Instead the axis is stretched linearly, in one pass and
    without intermediate arrays, so that the span the cell actually imbibes over lands on the
    span the table covers:

    - On the wetting axis (water in a water-wet system) imbibition runs from the reversal
      saturation up to `1 - trapped_saturation`, the table from `table_start` (its connate
      saturation) to `table_end` (one minus its residual oil saturation).
    - On a non-wetting axis (gas) imbibition runs from `trapped_saturation` up to the reversal
      saturation, the table from `table_start` (its critical gas saturation) to `table_end` (its
      largest gas saturation).

    A cell whose span has collapsed, because nothing was trapped or the reversal is at the
    trapped saturation, is left unscaled.

    :param size: Number of cells.
    :param saturation: Saturation the table is to be queried at.
    :param reversal_saturation: Saturation at the drainage to imbibition reversal.
    :param trapped_saturation: Land's trapped non-wetting saturation for each cell.
    :param on_wetting_axis: Whether the axis is the wetting phase's saturation.
    :param table_start: Saturation where the table's imbibition curve starts.
    :param table_end: Saturation where the table's imbibition curve ends.
    :param saturation_epsilon: Smallest span that is rescaled.
    :param out_saturation: Receives the saturation to query the table at.
    :param out_scale: Receives the slope of the mapping, which turns a table slope into a slope
        with respect to the cell's own saturation.
    """
    table_span = table_end - table_start
    for i in range(size):
        reversal = element_at(array=reversal_saturation, index=i)
        trapped = element_at(array=trapped_saturation, index=i)
        if on_wetting_axis:
            start = reversal
            span = 1.0 - trapped - reversal
        else:
            start = trapped
            span = reversal - trapped
        current = element_at(array=saturation, index=i)
        if span > saturation_epsilon and table_span > saturation_epsilon:
            scale = table_span / span
            out_saturation[i] = table_start + (current - start) * scale
            out_scale[i] = scale
        else:
            out_saturation[i] = current
            out_scale[i] = 1.0


@attrs.frozen(slots=True)
class LandTrapping:
    """
    Land's (1968) trapping of the non-wetting phase for one displacement.

    It answers one question: how much of the non-wetting phase is left behind when imbibition
    starts from a given saturation. The answer rises with the starting saturation and reaches
    `maximum_trapped_saturation` when imbibition starts from `maximum_saturation`:

        S_t = S_i / (1 + C * S_i),    C = 1 / S_t_max - 1 / S_max
    """

    maximum_trapped_saturation: Number
    """The most that can be trapped, `S_t_max`, reached when imbibition starts from the maximum."""

    maximum_saturation: Number
    """The largest non-wetting saturation, `S_max`, that imbibition can start from."""

    coefficient: Number
    """Land's trapping coefficient `C`. Never negative."""

    @classmethod
    def from_endpoints(
        cls,
        *,
        maximum_trapped_saturation: Number,
        maximum_saturation: Number,
        saturation_epsilon: Number = 1e-12,
    ) -> "LandTrapping":
        """
        Derive the coefficient from the two endpoints, which is how a deck's tables define it.

        :param maximum_trapped_saturation: The imbibition table's critical non-wetting saturation.
        :param maximum_saturation: The largest non-wetting saturation of the drainage table.
        :param saturation_epsilon: Small value to guard against division by zero.
        :returns: The trapping model.
        """
        return cls(
            maximum_trapped_saturation=maximum_trapped_saturation,
            maximum_saturation=maximum_saturation,
            coefficient=compute_land_trapping_coefficient(
                maximum_residual_saturation=maximum_trapped_saturation,
                maximum_non_wetting_saturation=maximum_saturation,
                saturation_epsilon=saturation_epsilon,
            ),
        )

    def trapped_saturation(
        self,
        initial_saturation: NumberOrArray[NDimension],
        *,
        saturation_epsilon: Number = 1e-12,
    ) -> NumberOrArray[NDimension]:
        """
        Non-wetting saturation trapped when imbibition starts from `initial_saturation`.

        It never exceeds the starting saturation or `maximum_trapped_saturation`.

        :param initial_saturation: Non-wetting saturation at the reversal, scalar or array.
        :param saturation_epsilon: Small value to guard against division by zero.
        :returns: The trapped saturation, shaped like the input.
        """
        return compute_land_residual_saturation(
            initial_non_wetting_saturation=initial_saturation,
            maximum_residual_saturation=self.maximum_trapped_saturation,
            land_trapping_coefficient=self.coefficient,
            saturation_epsilon=saturation_epsilon,
        )
