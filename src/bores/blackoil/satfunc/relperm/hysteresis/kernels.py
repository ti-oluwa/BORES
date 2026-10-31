"""Numba kernels that scan the bounding relative permeability curves, shared by every hysteresis model."""

import typing

import numba
import numpy as np

from bores.blackoil.satfunc.relperm.models.kernels import rule_derivatives, rule_value
from bores.blackoil.satfunc.utils import (
    compute_killough_scanning_weight,
    element_at,
)
from bores.types import (
    Boolean,
    Integer,
    Number,
    NumberArray,
    OneDimension,
)

__all__ = [
    "CurveSet",
    "HistoryArrays",
    "ScanOutputs",
    "normalise_saturations",
    "normalise_saturations_parallel",
    "saturations_need_normalising",
    "scan_relative_permeabilities",
    "scan_relative_permeabilities_block",
    "scan_relative_permeabilities_parallel",
]


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
