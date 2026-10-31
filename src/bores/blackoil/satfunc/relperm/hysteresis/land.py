"""
Land (1968) trapping of the non-wetting phase for relative permeability hysteresis.

- Land, C.S. (1968). "Calculation of Imbibition Relative Permeability for Two- and Three-Phase
  Flow from Rock Properties". SPE 1942.
"""

import numba
import numpy as np

from bores.blackoil.satfunc.utils import (
    element_at,
)
from bores.types import (
    Integer,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
)

__all__ = [
    "compute_land_residual_saturation",
    "compute_land_residual_saturation_array",
    "compute_land_residual_saturation_scalar",
    "compute_land_trapping_coefficient",
    "land_residual_from_reversal",
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
