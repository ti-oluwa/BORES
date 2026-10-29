"""
Hysteresis models for relative permeability.

Implements the Killough (1976) scanning-curve model with Land's (1968) trapping
for relative permeability hysteresis.

**References**:

- Killough, J.E. (1976). "Reservoir Simulation With History-Dependent
  Saturation Functions". SPE 5106.
- Land, C.S. (1968). "Calculation of Imbibition Relative Permeability for
  Two- and Three-Phase Flow from Rock Properties". SPE 1942.
- Carlson, F.M. (1981). "Simulation of Relative Permeability Hysteresis to
  the Non-Wetting Phase". SPE 10157.
"""

import math
import typing

import attrs
import numba
import numpy as np

from bores.blackoil.satfunc.relperm.mixing_rules import get_builtin_rule_id
from bores.blackoil.satfunc.relperm.models.kernels import rule_derivatives, rule_value
from bores.blackoil.satfunc.relperm.tables import (
    MixingRule,
    RelativePermeabilityTable,
    TwoPhaseRelPermTable,
    get_mixing_rule,
    get_mixing_rule_partial_derivatives,
    query_gas_oil_system,
    query_oil_water_system,
    resolve_residual_oil_saturation_gas,
)
from bores.blackoil.satfunc.utils import (
    PARALLEL_SIZE_THRESHOLD,
    compute_killough_scanning_weight,
    element_at,
    flatten_kernel_input,
)
from bores.constants import c
from bores.errors import ValidationError
from bores.types import (
    Boolean,
    BooleanArray,
    FluidPhase,
    Integer,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
    RelativePermeabilities,
    RelativePermeabilityDerivatives,
)

__all__ = [
    "KilloughLandRelPermTable",
    "compute_land_residual_saturation",
    "compute_land_trapping_coefficient",
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
def saturations_need_normalising(
    size: Integer,
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    saturation_epsilon: Number,
) -> bool:
    """Whether any cell has saturations that do not add up to 1 within `saturation_epsilon`."""
    for i in range(size):
        total = (
            element_at(array=water_saturation, index=i)
            + element_at(array=oil_saturation, index=i)
            + element_at(array=gas_saturation, index=i)
        )
        if abs(total - 1.0) > saturation_epsilon and total > 0.0:
            return True
    return False


@numba.njit(cache=True, error_model="numpy")
def normalise_saturations(
    size: Integer,
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    saturation_epsilon: Number,
) -> tuple[NumberArray[OneDimension], NumberArray[OneDimension], NumberArray[OneDimension]]:
    """Scale the saturations of every cell that does not add up to 1, returning new arrays."""
    new_water = np.empty(size, dtype=water_saturation.dtype)
    new_oil = np.empty(size, dtype=water_saturation.dtype)
    new_gas = np.empty(size, dtype=water_saturation.dtype)
    for i in range(size):
        sw = element_at(array=water_saturation, index=i)
        so = element_at(array=oil_saturation, index=i)
        sg = element_at(array=gas_saturation, index=i)
        total = sw + so + sg
        if abs(total - 1.0) > saturation_epsilon and total > 0.0:
            sw, so, sg = sw / total, so / total, sg / total
        new_water[i] = sw
        new_oil[i] = so
        new_gas[i] = sg
    return new_water, new_oil, new_gas  # type: ignore[return-value]


@numba.njit(cache=True, error_model="numpy", parallel=True)
def normalise_saturations_parallel(
    size: Integer,
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    saturation_epsilon: Number,
) -> tuple[NumberArray[OneDimension], NumberArray[OneDimension], NumberArray[OneDimension]]:
    """Scale the saturations of every cell that does not add up to 1, returning new arrays."""
    new_water = np.empty(size, dtype=water_saturation.dtype)
    new_oil = np.empty(size, dtype=water_saturation.dtype)
    new_gas = np.empty(size, dtype=water_saturation.dtype)
    for i in numba.prange(size):  # type: ignore
        sw = element_at(array=water_saturation, index=i)
        so = element_at(array=oil_saturation, index=i)
        sg = element_at(array=gas_saturation, index=i)
        total = sw + so + sg
        if abs(total - 1.0) > saturation_epsilon and total > 0.0:
            sw, so, sg = sw / total, so / total, sg / total
        new_water[i] = sw
        new_oil[i] = so
        new_gas[i] = sg
    return new_water, new_oil, new_gas  # type: ignore[return-value]


class HistoryArrays(typing.NamedTuple):
    """Flat saturation history of one cell array, for the oil-water and gas-oil systems."""

    max_water_saturation: NumberArray[OneDimension]
    """Historical maximum water saturation per cell, or one shared value."""

    max_gas_saturation: NumberArray[OneDimension]
    """Historical maximum gas saturation per cell, or one shared value."""

    water_imbibition_flag: NumberArray[OneDimension]
    """Water imbibition flag per cell, or one shared value."""

    gas_imbibition_flag: NumberArray[OneDimension]
    """Gas imbibition flag per cell, or one shared value."""

    water_reversal_saturation: NumberArray[OneDimension]
    """Water saturation at the last oil-water reversal per cell, or one shared value."""

    gas_reversal_saturation: NumberArray[OneDimension]
    """Gas saturation at the last gas-oil reversal per cell, or one shared value."""

    gas_trapped_saturation: NumberArray[OneDimension]
    """
    Lowest gas saturation that gas imbibition can reach per cell, or one shared value. It is
    only read while gas is the non-wetting phase.
    """


class CurveSet(typing.NamedTuple):
    """
    Values (or slopes) of the four two-phase curves on both bounding curves.

    The two-phase oil relative permeability of the oil-water system is `kro_water` and that
    of the gas-oil system is `kro_gas`. Each entry holds one value per cell, or a single
    value shared by all cells. When the imbibition curves equal the drainage curves, both
    entries are the same array.
    """

    krw_drainage: NumberArray[OneDimension]
    krw_imbibition: NumberArray[OneDimension]
    kro_water_drainage: NumberArray[OneDimension]
    kro_water_imbibition: NumberArray[OneDimension]
    kro_gas_drainage: NumberArray[OneDimension]
    kro_gas_imbibition: NumberArray[OneDimension]
    krg_drainage: NumberArray[OneDimension]
    krg_imbibition: NumberArray[OneDimension]


class ScanOutputs(typing.NamedTuple):
    """
    Arrays that `scan_relative_permeabilities` fills.

    Entries the requested computation does not need are one-element placeholders.
    """

    krw: NumberArray[OneDimension]
    kro: NumberArray[OneDimension]
    krg: NumberArray[OneDimension]
    kro_gas: NumberArray[OneDimension]
    dkrw_dsw: NumberArray[OneDimension]
    dkro_dsw: NumberArray[OneDimension]
    dkro_dso: NumberArray[OneDimension]
    dkro_dsg: NumberArray[OneDimension]
    dkrg_dsg: NumberArray[OneDimension]


@numba.njit(cache=True, error_model="numpy")
def scan_relative_permeabilities_block(
    start: Integer,
    stop: Integer,
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    history: HistoryArrays,
    use_water_hysteresis: Boolean,
    use_gas_hysteresis: Boolean,
    gas_is_non_wetting: Boolean,
    scanning_exponent: Number,
    numerical_epsilon: Number,
    values: CurveSet,
    slopes: CurveSet,
    with_derivatives: Boolean,
    rule_id: Integer,
    kr_max: Number,
    irreducible_water_saturation: Number,
    residual_oil_saturation_water: Number,
    residual_oil_saturation_gas: Number,
    minimum_mobile_pore_space: Number,
    out: ScanOutputs,
) -> None:
    """
    Scan the four two-phase curves, mix the three-phase oil curve, and differentiate, for the
    cells `start` up to but not including `stop`, in one pass.

    The water curves share one scanning weight, and so do the gas curves, so each weight (and its
    power) is evaluated once per cell instead of once per curve. For a built-in mixing rule
    (`rule_id >= 0`) `out.kro` is the mixed oil relative permeability and the oil derivatives are
    the chain-rule results. For `rule_id < 0` the rule is applied by the caller: `out.kro` then
    holds the scanned oil-water oil relative permeability, `out.kro_gas` the scanned gas-oil one,
    and `out.dkro_dsw` / `out.dkro_dsg` their scanned slopes.

    A system whose history is not in use stays on its primary drainage curves.
    """
    for i in range(start, stop):
        sw = element_at(array=water_saturation, index=i)
        so = element_at(array=oil_saturation, index=i)
        sg = element_at(array=gas_saturation, index=i)

        water_weight = 0.0
        water_weight_derivative = 0.0
        if use_water_hysteresis:
            water_weight, water_weight_derivative = compute_killough_scanning_weight(
                saturation=sw,
                reversal_saturation=element_at(array=history.water_reversal_saturation, index=i),
                maximum_historical_saturation=element_at(
                    array=history.max_water_saturation, index=i
                ),
                is_imbibition=element_at(array=history.water_imbibition_flag, index=i),
                scanning_exponent=scanning_exponent,
                numerical_epsilon=numerical_epsilon,
            )
        gas_weight = 0.0
        gas_weight_derivative = 0.0
        if use_gas_hysteresis:
            gas_reversal = element_at(array=history.gas_reversal_saturation, index=i)
            if gas_is_non_wetting:
                # Gas imbibes by falling from its peak (weight 0) to the trapped saturation
                # (weight 1), so the scanning range runs downward from the reversal.
                gas_range_end = min(
                    element_at(array=history.gas_trapped_saturation, index=i), gas_reversal
                )
            else:
                gas_range_end = element_at(array=history.max_gas_saturation, index=i)
            gas_weight, gas_weight_derivative = compute_killough_scanning_weight(
                saturation=sg,
                reversal_saturation=gas_reversal,
                maximum_historical_saturation=gas_range_end,
                is_imbibition=element_at(array=history.gas_imbibition_flag, index=i),
                scanning_exponent=scanning_exponent,
                numerical_epsilon=numerical_epsilon,
            )

        krw_drainage = element_at(array=values.krw_drainage, index=i)
        krw_imbibition = element_at(array=values.krw_imbibition, index=i)
        kro_water_drainage = element_at(array=values.kro_water_drainage, index=i)
        kro_water_imbibition = element_at(array=values.kro_water_imbibition, index=i)
        kro_gas_drainage = element_at(array=values.kro_gas_drainage, index=i)
        kro_gas_imbibition = element_at(array=values.kro_gas_imbibition, index=i)
        krg_drainage = element_at(array=values.krg_drainage, index=i)
        krg_imbibition = element_at(array=values.krg_imbibition, index=i)

        krw = krw_drainage + (krw_imbibition - krw_drainage) * water_weight
        kro_water = kro_water_drainage + (kro_water_imbibition - kro_water_drainage) * water_weight
        kro_gas = kro_gas_drainage + (kro_gas_imbibition - kro_gas_drainage) * gas_weight
        krg = krg_drainage + (krg_imbibition - krg_drainage) * gas_weight
        out.krw[i] = krw
        out.krg[i] = krg

        dkrw = 0.0
        dkro_water = 0.0
        dkro_gas = 0.0
        dkrg = 0.0
        if with_derivatives:
            dkrw_drainage = element_at(array=slopes.krw_drainage, index=i)
            dkrw_imbibition = element_at(array=slopes.krw_imbibition, index=i)
            dkro_water_drainage = element_at(array=slopes.kro_water_drainage, index=i)
            dkro_water_imbibition = element_at(array=slopes.kro_water_imbibition, index=i)
            dkro_gas_drainage = element_at(array=slopes.kro_gas_drainage, index=i)
            dkro_gas_imbibition = element_at(array=slopes.kro_gas_imbibition, index=i)
            dkrg_drainage = element_at(array=slopes.krg_drainage, index=i)
            dkrg_imbibition = element_at(array=slopes.krg_imbibition, index=i)

            dkrw = (
                dkrw_drainage
                + (dkrw_imbibition - dkrw_drainage) * water_weight
                + (krw_imbibition - krw_drainage) * water_weight_derivative
            )
            dkro_water = (
                dkro_water_drainage
                + (dkro_water_imbibition - dkro_water_drainage) * water_weight
                + (kro_water_imbibition - kro_water_drainage) * water_weight_derivative
            )
            dkro_gas = (
                dkro_gas_drainage
                + (dkro_gas_imbibition - dkro_gas_drainage) * gas_weight
                + (kro_gas_imbibition - kro_gas_drainage) * gas_weight_derivative
            )
            dkrg = (
                dkrg_drainage
                + (dkrg_imbibition - dkrg_drainage) * gas_weight
                + (krg_imbibition - krg_drainage) * gas_weight_derivative
            )
            out.dkrw_dsw[i] = dkrw
            out.dkrg_dsg[i] = dkrg

        if rule_id < 0:
            out.kro[i] = kro_water
            out.kro_gas[i] = kro_gas
            if with_derivatives:
                out.dkro_dsw[i] = dkro_water
                out.dkro_dsg[i] = dkro_gas
            continue

        out.kro[i] = rule_value(
            rule_id=rule_id,
            kro_w=kro_water,
            kro_g=kro_gas,
            krw=krw,
            krg=krg,
            kr_max=kr_max,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
            minimum_mobile_pore_space=minimum_mobile_pore_space,
        )
        if with_derivatives:
            (
                d_kro_water,
                d_kro_gas,
                d_krw,
                d_krg,
                d_sw,
                d_so,
                d_sg,
            ) = rule_derivatives(
                rule_id=rule_id,
                kro_w=kro_water,
                kro_g=kro_gas,
                krw=krw,
                krg=krg,
                kr_max=kr_max,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                irreducible_water_saturation=irreducible_water_saturation,
                residual_oil_saturation_water=residual_oil_saturation_water,
                residual_oil_saturation_gas=residual_oil_saturation_gas,
                minimum_mobile_pore_space=minimum_mobile_pore_space,
            )
            out.dkro_dsw[i] = d_kro_water * dkro_water + d_krw * dkrw + d_sw
            out.dkro_dso[i] = d_so
            out.dkro_dsg[i] = d_kro_gas * dkro_gas + d_krg * dkrg + d_sg


@numba.njit(cache=True, error_model="numpy")
def scan_relative_permeabilities(
    size: Integer,
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    history: HistoryArrays,
    use_water_hysteresis: Boolean,
    use_gas_hysteresis: Boolean,
    gas_is_non_wetting: Boolean,
    scanning_exponent: Number,
    numerical_epsilon: Number,
    values: CurveSet,
    slopes: CurveSet,
    with_derivatives: Boolean,
    rule_id: Integer,
    kr_max: Number,
    irreducible_water_saturation: Number,
    residual_oil_saturation_water: Number,
    residual_oil_saturation_gas: Number,
    minimum_mobile_pore_space: Number,
    out: ScanOutputs,
) -> None:
    """Scan every cell on the calling thread, see `scan_relative_permeabilities_block`."""
    scan_relative_permeabilities_block(
        start=0,
        stop=size,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        history=history,
        use_water_hysteresis=use_water_hysteresis,
        use_gas_hysteresis=use_gas_hysteresis,
        gas_is_non_wetting=gas_is_non_wetting,
        scanning_exponent=scanning_exponent,
        numerical_epsilon=numerical_epsilon,
        values=values,
        slopes=slopes,
        with_derivatives=with_derivatives,
        rule_id=rule_id,
        kr_max=kr_max,
        irreducible_water_saturation=irreducible_water_saturation,
        residual_oil_saturation_water=residual_oil_saturation_water,
        residual_oil_saturation_gas=residual_oil_saturation_gas,
        minimum_mobile_pore_space=minimum_mobile_pore_space,
        out=out,
    )


@numba.njit(cache=True, error_model="numpy", parallel=True)
def scan_relative_permeabilities_parallel(
    size: Integer,
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    history: HistoryArrays,
    use_water_hysteresis: Boolean,
    use_gas_hysteresis: Boolean,
    gas_is_non_wetting: Boolean,
    scanning_exponent: Number,
    numerical_epsilon: Number,
    values: CurveSet,
    slopes: CurveSet,
    with_derivatives: Boolean,
    rule_id: Integer,
    kr_max: Number,
    irreducible_water_saturation: Number,
    residual_oil_saturation_water: Number,
    residual_oil_saturation_gas: Number,
    minimum_mobile_pore_space: Number,
    out: ScanOutputs,
) -> None:
    """Scan the cells in one contiguous block per thread, see `scan_relative_permeabilities_block`."""
    block_count = numba.get_num_threads()
    block_size = (size + block_count - 1) // block_count
    for block in numba.prange(block_count):  # type: ignore
        start = block * block_size
        stop = min(start + block_size, size)
        scan_relative_permeabilities_block(
            start=start,
            stop=stop,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            history=history,
            use_water_hysteresis=use_water_hysteresis,
            use_gas_hysteresis=use_gas_hysteresis,
            gas_is_non_wetting=gas_is_non_wetting,
            scanning_exponent=scanning_exponent,
            numerical_epsilon=numerical_epsilon,
            values=values,
            slopes=slopes,
            with_derivatives=with_derivatives,
            rule_id=rule_id,
            kr_max=kr_max,
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
            minimum_mobile_pore_space=minimum_mobile_pore_space,
            out=out,
        )


def query_oil_water_curves(
    table: RelativePermeabilityTable,
    *,
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    with_derivatives: Boolean,
    **kwargs: typing.Any,
) -> tuple[typing.Any, typing.Any, typing.Any, typing.Any]:
    """
    Water and oil relative permeabilities of an oil-water table, with their slopes with respect to
    water saturation.

    A two-phase table is queried at the oil-water system's own saturations, `So = 1 - Sw`,
    whichever phase its reference axis uses, the same way `ThreePhaseRelPermTable` does.

    :param table: Oil-water relative permeability table (two-phase or three-phase).
    :param water_saturation: Flat water saturation array.
    :param oil_saturation: Flat oil saturation array.
    :param gas_saturation: Flat gas saturation array.
    :param with_derivatives: Whether to compute the slopes. They are `None` otherwise.
    :param kwargs: Extra arguments forwarded to a three-phase table.
    :return: Tuple of `krw`, `kro`, `dkrw/dSw` and `dkro/dSw`.
    """
    if isinstance(table, TwoPhaseRelPermTable):
        krw, kro = query_oil_water_system(table=table, water_saturation=water_saturation)
        if not with_derivatives:
            return krw, kro, None, None
        dkrw, dkro = query_oil_water_system(
            table=table, water_saturation=water_saturation, derivative=True
        )
        return krw, kro, dkrw, dkro

    if with_derivatives:
        values, slopes = table.evaluate_with_derivatives(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        return values["water"], values["oil"], slopes["dkrw_dsw"], slopes["dkro_dsw"]

    values = table.evaluate(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    return values["water"], values["oil"], None, None


def query_gas_oil_curves(
    table: RelativePermeabilityTable,
    *,
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    connate_water_saturation: Number,
    with_derivatives: Boolean,
    **kwargs: typing.Any,
) -> tuple[typing.Any, typing.Any, typing.Any, typing.Any]:
    """
    Oil and gas relative permeabilities of a gas-oil table, with their slopes with respect to gas
    saturation.

    A two-phase table is queried at the gas-oil system's own saturations, `So = 1 - swc - Sg`,
    whichever phase its reference axis uses, the same way `ThreePhaseRelPermTable` does.

    :param table: Gas-oil relative permeability table (two-phase or three-phase).
    :param water_saturation: Flat water saturation array.
    :param oil_saturation: Flat oil saturation array.
    :param gas_saturation: Flat gas saturation array.
    :param connate_water_saturation: Connate water saturation, which places the gas-oil system.
    :param with_derivatives: Whether to compute the slopes. They are `None` otherwise.
    :param kwargs: Extra arguments forwarded to a three-phase table.
    :return: Tuple of `kro`, `krg`, `dkro/dSg` and `dkrg/dSg`.
    """
    if isinstance(table, TwoPhaseRelPermTable):
        krg, kro = query_gas_oil_system(
            table=table,
            gas_saturation=gas_saturation,
            connate_water_saturation=connate_water_saturation,
        )
        if not with_derivatives:
            return kro, krg, None, None
        dkrg, dkro = query_gas_oil_system(
            table=table,
            gas_saturation=gas_saturation,
            connate_water_saturation=connate_water_saturation,
            derivative=True,
        )
        return kro, krg, dkro, dkrg

    if with_derivatives:
        values, slopes = table.evaluate_with_derivatives(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        return values["oil"], values["gas"], slopes["dkro_dsg"], slopes["dkrg_dsg"]

    values = table.evaluate(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    return values["oil"], values["gas"], None, None


@attrs.frozen(slots=True)
class KilloughLandRelPermTable(RelativePermeabilityTable):
    """
    Implements the Killough relative permeability hysteresis model/table with Land trapping.

    During *primary drainage* the relative permeabilities follow the
    `oil_water_drainage_table` and `gas_oil_drainage_table`.

    When the flow reverses (*imbibition*) two things happen simultaneously.

    **Land trapping**: a portion of the non-wetting phase becomes
    disconnected. The dynamic residual saturation depends on the saturation
    at the reversal point via Land's formula:

        S_r(S_i) = S_i / (1 + C * S_i),    C = 1 / S_r_max - 1 / S_max

    The dynamic residual is handed to the imbibition table as an endpoint override, so it
    changes the curve of an analytical (parametric) imbibition table. A tabular table has
    fixed endpoints and ignores it.

    **Killough scanning curves**: between the reversal point and the
    maximum historical saturation, kr follows a scanning curve that
    interpolates between the primary drainage and imbibition bounds.

    Both two-phase (`TwoPhaseRelPermTable`) and full three-phase
    (`RelativePermeabilityTable`) backing tables are supported. Wetting
    and non-wetting phase roles are resolved through the canonical API
    (`get_oil_water_wetting_phase` / `get_gas_oil_wetting_phase`) so the
    model is wettability-agnostic.

    The saturation history is passed as keyword arguments to `evaluate`, `derivatives` and
    `evaluate_with_derivatives`. A sub-system (oil-water or gas-oil) whose history is not
    supplied stays on its primary drainage curves, which is the correct behaviour for the first
    drainage cycle. Imbibition curves are only evaluated for a sub-system that has history and an
    imbibition curve that differs from its drainage curve.
    """

    __type__ = "killough_land_relperm_model"

    oil_water_drainage_table: RelativePermeabilityTable
    """Primary drainage relative permeability table for the oil-water system."""

    gas_oil_drainage_table: RelativePermeabilityTable
    """Primary drainage relative permeability table for the gas-oil system."""

    oil_water_imbibition_table: RelativePermeabilityTable | None = None
    """Primary imbibition table for the oil-water system. Defaults to the drainage table."""

    gas_oil_imbibition_table: RelativePermeabilityTable | None = None
    """Primary imbibition table for the gas-oil system. Defaults to the drainage table."""

    land_trapping_coefficient_water: Number = 1.0
    """
    Land trapping coefficient *C* for the oil-water system (>= 0). Use
    `compute_land_trapping_coefficient` to derive it from the endpoint saturations.
    """

    land_trapping_coefficient_gas: Number = 1.0
    """
    Land trapping coefficient *C* for the gas-oil system (>= 0). Use
    `compute_land_trapping_coefficient` to derive it from the endpoint saturations.
    """

    maximum_residual_oil_saturation_water: float | None = None
    """
    Maximum residual oil saturation S_r_max used by Land's formula for the
    oil-water system. Required when `oil_water_imbibition_table` is set.
    """

    maximum_residual_oil_saturation_gas: float | None = None
    """
    Maximum residual oil saturation S_r_max for the gas-oil system. Required
    when `gas_oil_imbibition_table` is set.
    """

    maximum_residual_gas_saturation: float | None = None
    """
    Maximum residual gas saturation S_r_max used by Land's formula. Required
    when `gas_oil_imbibition_table` is set.
    """

    scanning_interpolation_exponent: Number = 1.0
    """Killough scanning curve interpolation exponent *n* (1 = linear)."""

    mixing_rule: MixingRule | str = "baker_linear_rule"
    """Three-phase oil relative permeability mixing rule."""

    supports_vector: bool = attrs.field(init=False, repr=False, default=True)

    def __attrs_post_init__(self) -> None:
        if isinstance(self.mixing_rule, str):
            object.__setattr__(self, "mixing_rule", get_mixing_rule(name=self.mixing_rule))

        for name, table, phases in (
            (
                "oil_water_drainage_table",
                self.oil_water_drainage_table,
                {FluidPhase.WATER, FluidPhase.OIL},
            ),
            (
                "gas_oil_drainage_table",
                self.gas_oil_drainage_table,
                {FluidPhase.OIL, FluidPhase.GAS},
            ),
            (
                "oil_water_imbibition_table",
                self.oil_water_imbibition_table,
                {FluidPhase.WATER, FluidPhase.OIL},
            ),
            (
                "gas_oil_imbibition_table",
                self.gas_oil_imbibition_table,
                {FluidPhase.OIL, FluidPhase.GAS},
            ),
        ):
            if (
                isinstance(table, TwoPhaseRelPermTable)
                and {
                    table.wetting_phase,
                    table.non_wetting_phase,
                }
                != phases
            ):
                involved = " and ".join(sorted(phase.value for phase in phases))
                raise ValidationError(f"`{name}` must involve {involved} phases.")

    def get_oil_water_wetting_phase(self) -> FluidPhase:
        """
        Return the wetting phase for the oil-water sub-system.

        :return: `FluidPhase.WATER` for water-wet or `FluidPhase.OIL` for
            oil-wet systems, as reported by the drainage table.
        """
        return self.oil_water_drainage_table.get_oil_water_wetting_phase()

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        """
        Return the wetting phase for the gas-oil sub-system.

        :return: `FluidPhase.OIL` for oil-wet or `FluidPhase.GAS` for
            gas-wet systems, as reported by the drainage table.
        """
        return self.gas_oil_drainage_table.get_gas_oil_wetting_phase()

    def get_oil_relperm_endpoint(self) -> Number:
        """Oil relative permeability endpoint from the primary drainage table."""
        return self.oil_water_drainage_table.get_oil_relperm_endpoint()

    def get_water_relperm_endpoint(self) -> Number:
        """Water relative permeability endpoint from the primary drainage table."""
        return self.oil_water_drainage_table.get_water_relperm_endpoint()

    def get_gas_relperm_endpoint(self) -> Number:
        """Gas relative permeability endpoint from the primary drainage table."""
        return self.gas_oil_drainage_table.get_gas_relperm_endpoint()

    def get_connate_water_saturation(self) -> Number:
        """Connate water saturation from the primary drainage table. Hysteresis does not move it."""
        return self.oil_water_drainage_table.get_connate_water_saturation()

    def get_residual_oil_saturation_water(self) -> Number:
        """
        Residual oil saturation to water flooding that the primary drainage table declares.

        This is the static value of the table, not the Land-trapped value used while an
        imbibition scan is active.
        """
        return self.oil_water_drainage_table.get_residual_oil_saturation_water()

    def get_residual_oil_saturation_gas(self) -> Number:
        """
        Residual oil saturation to gas flooding that the primary drainage table declares, taking
        the connate water saturation of the system into account.
        """
        return resolve_residual_oil_saturation_gas(
            gas_oil_table=self.gas_oil_drainage_table,
            connate_water_saturation=self.get_connate_water_saturation(),
        )

    def get_residual_gas_saturation(self) -> Number:
        """Residual (critical) gas saturation that the primary drainage table declares."""
        return self.gas_oil_drainage_table.get_residual_gas_saturation()

    def compute(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        *,
        with_derivatives: Boolean,
        max_water_saturation: NumberOrArray[NDimension] | None = None,
        max_gas_saturation: NumberOrArray[NDimension] | None = None,
        water_imbibition_flag: bool | BooleanArray[NDimension] | None = None,
        gas_imbibition_flag: bool | BooleanArray[NDimension] | None = None,
        water_reversal_saturation: NumberOrArray[NDimension] | None = None,
        gas_reversal_saturation: NumberOrArray[NDimension] | None = None,
        **kwargs: typing.Any,
    ) -> tuple[RelativePermeabilities, RelativePermeabilityDerivatives | None]:
        """
        Relative permeabilities, and optionally their nine derivatives, in one pass.

        This is the one implementation behind `evaluate`, `derivatives` and
        `evaluate_with_derivatives`. Each bounding curve is queried once, the scanning weights
        are shared between the curves that scan over the same saturation, and the mixing rule
        and its derivatives are applied in the same loop for the built-in rules.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param with_derivatives: Whether to also compute the derivatives.
        :param max_water_saturation: Historical maximum water saturation per cell.
        :param max_gas_saturation: Historical maximum gas saturation per cell.
        :param water_imbibition_flag: Per-cell flag, true while water saturation increases.
        :param gas_imbibition_flag: Per-cell flag, true while gas saturation increases.
        :param water_reversal_saturation: Water saturation at the last reversal. Defaults to
            `max_water_saturation`.
        :param gas_reversal_saturation: Gas saturation at the last reversal. Defaults to
            `max_gas_saturation`.
        :param kwargs: Extra arguments forwarded to the backing tables, such as residual
            saturation overrides for analytical models.
        :return: The relative permeabilities and, when requested, their derivatives
            (otherwise `None`).
        """
        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
        )
        use_water_hysteresis = (
            max_water_saturation is not None and water_imbibition_flag is not None
        )
        use_gas_hysteresis = max_gas_saturation is not None and gas_imbibition_flag is not None

        inputs = [water_saturation, oil_saturation, gas_saturation]
        if use_water_hysteresis:
            inputs += [max_water_saturation, water_imbibition_flag]
            if water_reversal_saturation is not None:
                inputs.append(water_reversal_saturation)
        if use_gas_hysteresis:
            inputs += [max_gas_saturation, gas_imbibition_flag]
            if gas_reversal_saturation is not None:
                inputs.append(gas_reversal_saturation)

        shape = np.broadcast_shapes(*(np.shape(value) for value in inputs))
        flat_shape = shape or (1,)
        size = math.prod(flat_shape)

        sw = flatten_kernel_input(value=water_saturation, shape=flat_shape)
        so = flatten_kernel_input(value=oil_saturation, shape=flat_shape)
        sg = flatten_kernel_input(value=gas_saturation, shape=flat_shape)
        saturation_epsilon = c.SATURATION_EPSILON
        if saturations_need_normalising(
            size=size,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            saturation_epsilon=saturation_epsilon,
        ):
            normalise = (
                normalise_saturations_parallel
                if size >= PARALLEL_SIZE_THRESHOLD
                else normalise_saturations
            )
            sw, so, sg = normalise(
                size=size,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                saturation_epsilon=saturation_epsilon,
            )

        placeholder = np.zeros(1, dtype=sw.dtype)
        no_flag = np.zeros(1, dtype=np.bool_)
        if use_water_hysteresis:
            maximum_water = flatten_kernel_input(value=max_water_saturation, shape=flat_shape)
            water_flag = flatten_kernel_input(
                value=water_imbibition_flag, shape=flat_shape, as_float=False
            )
            water_reversal = (
                maximum_water
                if water_reversal_saturation is None
                else flatten_kernel_input(value=water_reversal_saturation, shape=flat_shape)
            )
        else:
            maximum_water, water_flag, water_reversal = placeholder, no_flag, placeholder
        if use_gas_hysteresis:
            maximum_gas = flatten_kernel_input(value=max_gas_saturation, shape=flat_shape)
            gas_flag = flatten_kernel_input(
                value=gas_imbibition_flag, shape=flat_shape, as_float=False
            )
            gas_reversal = (
                maximum_gas
                if gas_reversal_saturation is None
                else flatten_kernel_input(value=gas_reversal_saturation, shape=flat_shape)
            )
        else:
            maximum_gas, gas_flag, gas_reversal = placeholder, no_flag, placeholder

        history = HistoryArrays(
            max_water_saturation=typing.cast(NumberArray[OneDimension], maximum_water),
            max_gas_saturation=typing.cast(NumberArray[OneDimension], maximum_gas),
            water_imbibition_flag=typing.cast(NumberArray[OneDimension], water_flag),
            gas_imbibition_flag=typing.cast(NumberArray[OneDimension], gas_flag),
            water_reversal_saturation=typing.cast(NumberArray[OneDimension], water_reversal),
            gas_reversal_saturation=typing.cast(NumberArray[OneDimension], gas_reversal),
            gas_trapped_saturation=typing.cast(NumberArray[OneDimension], placeholder),
        )
        gas_is_non_wetting = self.get_gas_oil_wetting_phase() != FluidPhase.GAS

        # Bounding curves
        drainage_water_system = self.oil_water_drainage_table
        drainage_gas_system = self.gas_oil_drainage_table
        curve_arguments = dict(
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            with_derivatives=with_derivatives,
        )
        gas_oil_arguments = {
            **curve_arguments,
            "connate_water_saturation": self.get_connate_water_saturation(),
        }

        krw_d, kro_water_d, dkrw_d, dkro_water_d = query_oil_water_curves(
            drainage_water_system, **curve_arguments, **kwargs
        )
        kro_gas_d, krg_d, dkro_gas_d, dkrg_d = query_gas_oil_curves(
            drainage_gas_system, **gas_oil_arguments, **kwargs
        )

        krw_i, kro_water_i, dkrw_i, dkro_water_i = krw_d, kro_water_d, dkrw_d, dkro_water_d
        if use_water_hysteresis:
            imbibition_table = self.oil_water_imbibition_table
            land_kwargs = self.land_override_kwargs_water(
                size=size,
                gas_saturation=sg,
                water_reversal=typing.cast(NumberArray[OneDimension], water_reversal),
                table=imbibition_table or drainage_water_system,
                kwargs=kwargs,
            )
            if imbibition_table is not None and imbibition_table is not drainage_water_system:
                krw_i, kro_water_i, dkrw_i, dkro_water_i = query_oil_water_curves(
                    imbibition_table, **curve_arguments, **land_kwargs
                )
            elif land_kwargs is not kwargs:
                krw_i, kro_water_i, dkrw_i, dkro_water_i = query_oil_water_curves(
                    drainage_water_system, **curve_arguments, **land_kwargs
                )

        kro_gas_i, krg_i, dkro_gas_i, dkrg_i = kro_gas_d, krg_d, dkro_gas_d, dkrg_d
        if use_gas_hysteresis:
            imbibition_table = self.gas_oil_imbibition_table
            land_kwargs = self.land_override_kwargs_gas(
                size=size,
                water_saturation=sw,
                gas_reversal=typing.cast(NumberArray[OneDimension], gas_reversal),
                table=imbibition_table or drainage_gas_system,
                kwargs=kwargs,
            )
            if gas_is_non_wetting:
                if "residual_gas_saturation" in land_kwargs:
                    trapped_gas = land_kwargs["residual_gas_saturation"]
                else:
                    trapped_gas = np.full(
                        1,
                        (imbibition_table or drainage_gas_system).get_residual_gas_saturation(),
                        dtype=sw.dtype,
                    )
                history = history._replace(
                    gas_trapped_saturation=typing.cast(NumberArray[OneDimension], trapped_gas)
                )
            if imbibition_table is not None and imbibition_table is not drainage_gas_system:
                kro_gas_i, krg_i, dkro_gas_i, dkrg_i = query_gas_oil_curves(
                    imbibition_table, **gas_oil_arguments, **land_kwargs
                )
            elif land_kwargs is not kwargs:
                kro_gas_i, krg_i, dkro_gas_i, dkrg_i = query_gas_oil_curves(
                    drainage_gas_system, **gas_oil_arguments, **land_kwargs
                )

        values = CurveSet(
            krw_drainage=krw_d,
            krw_imbibition=krw_i,
            kro_water_drainage=kro_water_d,
            kro_water_imbibition=kro_water_i,
            kro_gas_drainage=kro_gas_d,
            kro_gas_imbibition=kro_gas_i,
            krg_drainage=krg_d,
            krg_imbibition=krg_i,
        )
        slopes = CurveSet(
            krw_drainage=typing.cast(
                NumberArray[OneDimension], dkrw_d if with_derivatives else placeholder
            ),
            krw_imbibition=typing.cast(
                NumberArray[OneDimension], dkrw_i if with_derivatives else placeholder
            ),
            kro_water_drainage=typing.cast(
                NumberArray[OneDimension], dkro_water_d if with_derivatives else placeholder
            ),
            kro_water_imbibition=typing.cast(
                NumberArray[OneDimension], dkro_water_i if with_derivatives else placeholder
            ),
            kro_gas_drainage=typing.cast(
                NumberArray[OneDimension], dkro_gas_d if with_derivatives else placeholder
            ),
            kro_gas_imbibition=typing.cast(
                NumberArray[OneDimension], dkro_gas_i if with_derivatives else placeholder
            ),
            krg_drainage=typing.cast(
                NumberArray[OneDimension], dkrg_d if with_derivatives else placeholder
            ),
            krg_imbibition=typing.cast(
                NumberArray[OneDimension], dkrg_i if with_derivatives else placeholder
            ),
        )
        values = CurveSet(
            *(
                typing.cast(NumberArray[OneDimension], np.asarray(array).reshape(-1))
                for array in values
            )
        )
        slopes = CurveSet(
            *(
                typing.cast(NumberArray[OneDimension], np.asarray(array).reshape(-1))
                for array in slopes
            )
        )

        dtype = values.krw_drainage.dtype
        if dtype.kind != "f":
            dtype = np.dtype(np.float64)
        rule = typing.cast(MixingRule, self.mixing_rule)
        rule_id = get_builtin_rule_id(rule)
        use_rule_in_kernel = rule_id is not None
        needs_gas_oil = not use_rule_in_kernel
        out = ScanOutputs(
            krw=typing.cast(NumberArray[OneDimension], np.empty(size, dtype=dtype)),
            kro=typing.cast(NumberArray[OneDimension], np.empty(size, dtype=dtype)),
            krg=typing.cast(NumberArray[OneDimension], np.empty(size, dtype=dtype)),
            kro_gas=typing.cast(
                NumberArray[OneDimension], np.empty(size if needs_gas_oil else 1, dtype=dtype)
            ),
            dkrw_dsw=typing.cast(
                NumberArray[OneDimension], np.empty(size if with_derivatives else 1, dtype=dtype)
            ),
            dkro_dsw=typing.cast(
                NumberArray[OneDimension], np.empty(size if with_derivatives else 1, dtype=dtype)
            ),
            dkro_dso=typing.cast(
                NumberArray[OneDimension],
                np.empty(size if with_derivatives and use_rule_in_kernel else 1, dtype=dtype),
            ),
            dkro_dsg=typing.cast(
                NumberArray[OneDimension], np.empty(size if with_derivatives else 1, dtype=dtype)
            ),
            dkrg_dsg=typing.cast(
                NumberArray[OneDimension], np.empty(size if with_derivatives else 1, dtype=dtype)
            ),
        )
        kr_max = self.get_oil_relperm_endpoint()
        irreducible_water_saturation = self.get_connate_water_saturation()
        residual_oil_saturation_water = self.get_residual_oil_saturation_water()
        residual_oil_saturation_gas = self.get_residual_oil_saturation_gas()
        scan = (
            scan_relative_permeabilities_parallel
            if size >= PARALLEL_SIZE_THRESHOLD
            else scan_relative_permeabilities
        )
        scan(
            size=size,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            history=history,
            use_water_hysteresis=use_water_hysteresis,
            use_gas_hysteresis=use_gas_hysteresis,
            gas_is_non_wetting=gas_is_non_wetting,
            scanning_exponent=self.scanning_interpolation_exponent,
            numerical_epsilon=1e-12,
            values=values,
            slopes=slopes,
            with_derivatives=with_derivatives,
            rule_id=-1 if rule_id is None else rule_id,
            kr_max=kr_max,
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
            minimum_mobile_pore_space=c.MINIMUM_MOBILE_PORE_SPACE,
            out=out,
        )

        zeros = np.zeros(size, dtype=dtype) if with_derivatives else None
        krw, krg = out.krw, out.krg
        if use_rule_in_kernel:
            kro = out.kro
            dkro_dsw, dkro_dso, dkro_dsg = out.dkro_dsw, out.dkro_dso, out.dkro_dsg
        else:
            kro_water, kro_gas = out.kro, out.kro_gas
            kro = rule(
                kro_w=kro_water,
                kro_g=kro_gas,
                krw=krw,
                krg=krg,
                kr_max=kr_max,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                irreducible_water_saturation=irreducible_water_saturation,
                residual_oil_saturation_water=residual_oil_saturation_water,
                residual_oil_saturation_gas=residual_oil_saturation_gas,
            )
            dkro_dsw = dkro_dso = dkro_dsg = None
            if with_derivatives:
                partials = get_mixing_rule_partial_derivatives(
                    rule=rule,
                    kro_w=kro_water,
                    kro_g=kro_gas,
                    krw=krw,
                    krg=krg,
                    kr_max=kr_max,
                    water_saturation=sw,
                    oil_saturation=so,
                    gas_saturation=sg,
                    irreducible_water_saturation=irreducible_water_saturation,
                    residual_oil_saturation_water=residual_oil_saturation_water,
                    residual_oil_saturation_gas=residual_oil_saturation_gas,
                    epsilon=c.FINITE_DIFFERENCE_EPSILON,
                )
                dkro_dsw = (
                    partials["dkro_dkro_w"] * out.dkro_dsw
                    + partials["dkro_dkrw"] * out.dkrw_dsw
                    + partials["dkro_dsw_explicit"]
                )
                dkro_dso = partials["dkro_dso_explicit"]
                dkro_dsg = (
                    partials["dkro_dkro_g"] * out.dkro_dsg
                    + partials["dkro_dkrg"] * out.dkrg_dsg
                    + partials["dkro_dsg_explicit"]
                )

        def finish(array: typing.Any) -> typing.Any:
            result = np.asarray(array)
            if result.size == 1 and size > 1:
                result = np.broadcast_to(result, (size,))
            result = result.reshape(shape)
            return result.item() if is_scalar else result

        relperms = RelativePermeabilities(
            water=finish(array=krw),
            oil=finish(array=kro),
            gas=finish(array=krg),
        )
        if not with_derivatives:
            return relperms, None
        assert zeros is not None
        zero = finish(array=zeros)
        return relperms, RelativePermeabilityDerivatives(
            dkrw_dsw=finish(array=out.dkrw_dsw),
            dkro_dsw=finish(array=dkro_dsw),
            dkrg_dsw=zero,
            dkrw_dso=zero,
            dkro_dso=finish(array=dkro_dso),
            dkrg_dso=zero,
            dkrw_dsg=zero,
            dkro_dsg=finish(array=dkro_dsg),
            dkrg_dsg=finish(array=out.dkrg_dsg),
        )

    def land_override_kwargs_water(
        self,
        *,
        size: Integer,
        gas_saturation: NumberArray[OneDimension],
        water_reversal: NumberArray[OneDimension],
        table: RelativePermeabilityTable,
        kwargs: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        """
        Keyword arguments for the oil-water imbibition table with the Land residual oil saturation
        in place.

        Land trapping only changes a curve whose table accepts endpoint overrides, so a tabular
        two-phase table and a model without a configured maximum residual get `kwargs` back
        as it is.
        """
        maximum_residual = self.maximum_residual_oil_saturation_water
        if maximum_residual is None or isinstance(table, TwoPhaseRelPermTable):
            return kwargs

        residual = np.empty(size, dtype=gas_saturation.dtype)
        land_residual_from_reversal(
            size=size,
            reversal_saturation=water_reversal,
            other_saturation=gas_saturation,
            from_remaining_oil=True,
            maximum_residual_saturation=maximum_residual,
            land_trapping_coefficient=self.land_trapping_coefficient_water,
            saturation_epsilon=1e-12,
            out=typing.cast(NumberArray[OneDimension], residual),
        )
        return {**kwargs, "residual_oil_saturation_water": residual}

    def land_override_kwargs_gas(
        self,
        *,
        size: Integer,
        water_saturation: NumberArray[OneDimension],
        gas_reversal: NumberArray[OneDimension],
        table: RelativePermeabilityTable,
        kwargs: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        """
        Keyword arguments for the gas-oil imbibition table with the Land residual gas and oil
        saturations in place.

        Same rule as `land_override_kwargs_water`: a table that cannot take endpoint overrides,
        or a model with no configured maximum residual, leaves the entries out.
        """
        if isinstance(table, TwoPhaseRelPermTable):
            return kwargs
        overrides: dict[str, typing.Any] = {}
        if self.maximum_residual_gas_saturation is not None:
            residual_gas = np.empty(size, dtype=water_saturation.dtype)
            land_residual_from_reversal(
                size=size,
                reversal_saturation=gas_reversal,
                other_saturation=water_saturation,
                from_remaining_oil=False,
                maximum_residual_saturation=self.maximum_residual_gas_saturation,
                land_trapping_coefficient=self.land_trapping_coefficient_gas,
                saturation_epsilon=1e-12,
                out=typing.cast(NumberArray[OneDimension], residual_gas),
            )
            overrides["residual_gas_saturation"] = residual_gas
        if self.maximum_residual_oil_saturation_gas is not None:
            residual_oil = np.empty(size, dtype=water_saturation.dtype)
            land_residual_from_reversal(
                size=size,
                reversal_saturation=gas_reversal,
                other_saturation=water_saturation,
                from_remaining_oil=True,
                maximum_residual_saturation=self.maximum_residual_oil_saturation_gas,
                land_trapping_coefficient=self.land_trapping_coefficient_gas,
                saturation_epsilon=1e-12,
                out=typing.cast(NumberArray[OneDimension], residual_oil),
            )
            overrides["residual_oil_saturation_gas"] = residual_oil
        return {**kwargs, **overrides} if overrides else kwargs

    def evaluate(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute three-phase relative permeabilities with Killough/Land hysteresis.

        Without saturation history the result is the primary drainage relative permeability.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param kwargs: History arguments (`max_water_saturation`, `max_gas_saturation`,
            `water_imbibition_flag`, `gas_imbibition_flag`, `water_reversal_saturation`,
            `gas_reversal_saturation`) and any other arguments the backing tables accept.
        :return: `RelativePermeabilities` dictionary with keys `"water"`, `"oil"`, and `"gas"`.
        """
        relperms, _ = self.compute(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            with_derivatives=False,
            **kwargs,
        )
        return relperms

    def derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> RelativePermeabilityDerivatives:
        """
        Compute the nine partial derivatives of the relative permeabilities with
        Killough/Land hysteresis.

        The derivatives with respect to the scanning saturations (Sw for oil-water, Sg for
        gas-oil) follow the chain rule through the scanning curve and the mixing rule. Every other
        cross derivative is zero, as each two-phase sub-system depends on its own saturation only.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param kwargs: Same history and backing table arguments as `evaluate`.
        :return: `RelativePermeabilityDerivatives` dictionary with all nine entries.
        """
        _, derivatives = self.compute(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            with_derivatives=True,
            **kwargs,
        )
        assert derivatives is not None
        return derivatives

    def evaluate_with_derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> tuple[RelativePermeabilities, RelativePermeabilityDerivatives]:
        """
        Compute the relative permeabilities and their derivatives in one pass.

        This evaluates every bounding curve and scanning weight once, where calling `evaluate`
        and `derivatives` separately would do it twice.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param kwargs: Same history and backing table arguments as `evaluate`.
        :return: Tuple of the `RelativePermeabilities` and `RelativePermeabilityDerivatives`.
        """
        relperms, derivatives = self.compute(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            with_derivatives=True,
            **kwargs,
        )
        assert derivatives is not None
        return relperms, derivatives
