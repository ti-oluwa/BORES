"""
Hysteresis models for relative permeability and capillary pressure.

Implements the Killough (1976) scanning-curve model with Land's (1968) trapping
for relative permeability hysteresis, and a Killough-type scanning-curve model
for capillary pressure hysteresis.

**References**:

- Killough, J.E. (1976). "Reservoir Simulation With History-Dependent
  Saturation Functions". SPE 5106.
- Land, C.S. (1968). "Calculation of Imbibition Relative Permeability for
  Two- and Three-Phase Flow from Rock Properties". SPE 1942.
- Carlson, F.M. (1981). "Simulation of Relative Permeability Hysteresis to
  the Non-Wetting Phase". SPE 10157.
"""

import typing

import attrs
import numba
import numpy as np
import numpy.typing as npt

from bores.constants import c
from bores.errors import ValidationError
from bores.rock_fluid.capillary_pressure import (
    CapillaryPressureTable,
    TwoPhaseCapillaryPressureTable,
    capillary_pressure_table,
)
from bores.rock_fluid.relperm import (
    MixingRule,
    RelativePermeabilityTable,
    TwoPhaseRelPermTable,
    get_mixing_rule,
    get_mixing_rule_partial_derivatives,
    relperm_table,
    serialize_mixing_rule,
)
from bores.types import (
    CapillaryPressureDerivatives,
    CapillaryPressures,
    FloatOrArray,
    FluidPhase,
    RelativePermeabilities,
    RelativePermeabilityDerivatives,
)
from bores.utils import atleast_1d

__all__ = ["KilloughCapillaryPressureModel", "KilloughLandRelPermModel"]


@numba.njit(cache=True)
def _land_residual_saturation_scalar(
    initial_non_wetting_saturation: float,
    max_residual_saturation: float,
    land_coefficient: float,
    saturation_epsilon: float = 1e-12,
) -> float:
    """
    Compute the dynamic residual non-wetting saturation via Land's model (scalar).

    Land (1968) relates the residual saturation that will be trapped when
    imbibition begins at `initial_non_wetting_saturation` to the maximum
    possible residual saturation observed at the drainage endpoint::

        S_r = S_r_max / (1 + C * S_i)

    Higher *C* means more trapping (smaller residual for the same initial
    saturation). *C* = 0 means no dynamic trapping (residual equals
    `max_residual_saturation` regardless of initial saturation).

    :param initial_non_wetting_saturation: Non-wetting saturation at the
        drainage-imbibition reversal point.
    :param max_residual_saturation: Maximum residual saturation from the
        drainage endpoint (S_r_max).
    :param land_coefficient: Land trapping coefficient *C* (≥ 0).
    :param saturation_epsilon: Small value to guard against division by zero.
    :return: Dynamic residual saturation (scalar).
    """
    s_r_max = max(max_residual_saturation, saturation_epsilon)

    if land_coefficient <= 0.0:
        return s_r_max

    s_i = max(initial_non_wetting_saturation, 0.0)
    s_r = s_r_max / (1.0 + land_coefficient * s_i)
    return min(s_r, s_i)


@numba.njit(cache=True, parallel=True)
def _land_residual_saturation_array(
    initial_non_wetting_saturation: npt.NDArray,
    max_residual_saturation: float,
    land_coefficient: float,
    saturation_epsilon: float = 1e-12,
) -> npt.NDArray:
    """
    Compute the dynamic residual non-wetting saturation via Land's model (array).

    Vectorised version of :func:`_land_residual_saturation_scalar`.  All
    elements are processed in a single Numba-compiled loop so no Python
    overhead is incurred per cell.

    :param initial_non_wetting_saturation: Non-wetting saturations at the
        drainage-imbibition reversal points (1-D or N-D array).
    :param max_residual_saturation: Maximum residual saturation from the
        drainage endpoint (scalar).
    :param land_coefficient: Land trapping coefficient *C* (≥ 0).
    :param saturation_epsilon: Small value to guard against division by zero.
    :return: Dynamic residual saturation array with the same shape as the input.
    """
    s_i = atleast_1d(initial_non_wetting_saturation)
    s_r_max = max(max_residual_saturation, saturation_epsilon)

    result = np.empty_like(s_i)
    for idx in numba.prange(s_i.size):  # type: ignore
        s_i_val = max(s_i.flat[idx], 0.0)
        if land_coefficient <= 0.0:
            result.flat[idx] = min(s_r_max, s_i_val)
        else:
            s_r = s_r_max / (1.0 + land_coefficient * s_i_val)
            result.flat[idx] = min(s_r, s_i_val)
    return result


def _land_residual_saturation(
    initial_non_wetting_saturation: FloatOrArray,
    max_residual_saturation: float,
    land_coefficient: float,
    saturation_epsilon: float = 1e-12,
) -> FloatOrArray:
    """
    Dispatch Land residual saturation computation to the scalar or array kernel.

    Routes to :func:`_land_residual_saturation_scalar` when the input is a
    Python scalar and to :func:`_land_residual_saturation_array` otherwise,
    avoiding Numba union-return-type limitations.

    :param initial_non_wetting_saturation: Non-wetting saturation at the
        drainage-imbibition reversal point (scalar or array).
    :param max_residual_saturation: Maximum residual saturation from the
        drainage endpoint (S_r_max).
    :param land_coefficient: Land trapping coefficient *C* (≥ 0).
    :param saturation_epsilon: Small value to guard against division by zero.
    :return: Dynamic residual saturation matching the shape of the input.
    """
    if np.isscalar(initial_non_wetting_saturation):
        return _land_residual_saturation_scalar(
            float(initial_non_wetting_saturation),  # type: ignore[arg-type]
            max_residual_saturation,
            land_coefficient,
            saturation_epsilon,
        )
    return _land_residual_saturation_array(
        np.asarray(initial_non_wetting_saturation, dtype=np.float64),
        max_residual_saturation,
        land_coefficient,
        saturation_epsilon,
    )


@numba.njit(cache=True)
def _killough_interpolation_scalar(
    saturation: float,
    drainage_value: float,
    imbibition_value: float,
    reversal_saturation: float,
    max_saturation: float,
    is_imbibition: float,
    exponent: float = 1.0,
    epsilon: float = 1e-12,
) -> float:
    """
    Killough scanning-curve interpolation between primary drainage and
    imbibition curves (scalar).

    When the flow direction reverses the reservoir property (kr or Pc)
    follows a scanning curve interpolated between the two primary curves::

        value = value_drain + (value_imb - value_drain) * f(S)

    where the interpolation factor is::

        f(S) = clamp( ((S - S_rev) / (S_max - S_rev))^n , 0, 1 )

    :param saturation: Current saturation.
    :param drainage_value: Value from the primary drainage curve.
    :param imbibition_value: Value from the primary imbibition curve.
    :param reversal_saturation: Saturation at the last reversal point (S_rev).
    :param max_saturation: Maximum saturation reached before the reversal (S_max).
    :param is_imbibition: 1.0 if currently on the imbibition path, 0.0 if drainage.
    :param exponent: Killough interpolation exponent *n* (1 = linear).
    :param epsilon: Numerical stability tolerance.
    :return: Interpolated scanning-curve value.
    """
    delta = max_saturation - reversal_saturation
    if abs(delta) > epsilon:
        raw_f = (saturation - reversal_saturation) / delta
    else:
        raw_f = 0.0

    f = min(max(raw_f**exponent, 0.0), 1.0)
    val_scan = drainage_value + (imbibition_value - drainage_value) * f

    on_primary_drain = (is_imbibition < 0.5) and (
        abs(saturation - max_saturation) < epsilon
    )
    on_primary_imb = (is_imbibition >= 0.5) and (
        abs(saturation - reversal_saturation) < epsilon
    )

    if on_primary_drain:
        return drainage_value
    if on_primary_imb:
        return imbibition_value
    return val_scan


@numba.njit(cache=True)
def _killough_interpolation_array(
    saturation: npt.NDArray,
    drainage_value: npt.NDArray,
    imbibition_value: npt.NDArray,
    reversal_saturation: npt.NDArray,
    max_saturation: npt.NDArray,
    is_imbibition: npt.NDArray,
    exponent: float = 1.0,
    epsilon: float = 1e-12,
) -> npt.NDArray:
    """
    Killough scanning-curve interpolation between primary drainage and
    imbibition curves (array).

    Vectorised version of :func:`_killough_interpolation_scalar`.

    :param saturation: Current saturation array.
    :param drainage_value: Drainage curve values at current saturations.
    :param imbibition_value: Imbibition curve values at current saturations.
    :param reversal_saturation: Saturations at the last reversal points.
    :param max_saturation: Maximum saturations reached before the reversals.
    :param is_imbibition: Per-cell imbibition flag (1.0 = imbibition, 0.0 = drainage).
    :param exponent: Killough interpolation exponent *n* (1 = linear).
    :param epsilon: Numerical stability tolerance.
    :return: Interpolated scanning-curve values with the same shape as `saturation`.
    """
    sat = atleast_1d(saturation)
    val_d = atleast_1d(drainage_value)
    val_i = atleast_1d(imbibition_value)
    s_rev = atleast_1d(reversal_saturation)
    s_max = atleast_1d(max_saturation)
    imb = atleast_1d(is_imbibition)

    # broadcast to common shape
    sat, val_d, val_i, s_rev, s_max, imb = np.broadcast_arrays(
        sat, val_d, val_i, s_rev, s_max, imb
    )

    result = np.empty_like(sat)
    for k in numba.prange(sat.size):  # type: ignore
        delta = s_max.flat[k] - s_rev.flat[k]
        if abs(delta) > epsilon:
            raw_f = (sat.flat[k] - s_rev.flat[k]) / delta
        else:
            raw_f = 0.0
        f = min(max(raw_f**exponent, 0.0), 1.0)
        val_scan = val_d.flat[k] + (val_i.flat[k] - val_d.flat[k]) * f

        on_primary_drain = (imb.flat[k] < 0.5) and (
            abs(sat.flat[k] - s_max.flat[k]) < epsilon
        )
        on_primary_imb = (imb.flat[k] >= 0.5) and (
            abs(sat.flat[k] - s_rev.flat[k]) < epsilon
        )

        if on_primary_drain:
            result.flat[k] = val_d.flat[k]
        elif on_primary_imb:
            result.flat[k] = val_i.flat[k]
        else:
            result.flat[k] = val_scan
    return result


def _killough_interpolation(
    saturation: FloatOrArray,
    drainage_value: FloatOrArray,
    imbibition_value: FloatOrArray,
    reversal_saturation: FloatOrArray,
    max_saturation: FloatOrArray,
    is_imbibition: FloatOrArray,
    exponent: float = 1.0,
    epsilon: float = 1e-12,
) -> FloatOrArray:
    """
    Dispatch Killough scanning-curve interpolation to the scalar or array kernel.

    :param saturation: Current saturation (scalar or array).
    :param drainage_value: Value from the primary drainage curve.
    :param imbibition_value: Value from the primary imbibition curve.
    :param reversal_saturation: Saturation at the last reversal (scalar or array).
    :param max_saturation: Maximum saturation before the reversal (scalar or array).
    :param is_imbibition: Imbibition flag — 1.0 for imbibition, 0.0 for drainage
        (scalar or array).
    :param exponent: Killough interpolation exponent *n* (1 = linear).
    :param epsilon: Numerical stability tolerance.
    :return: Interpolated scanning-curve value matching the shape of the input.
    """
    is_scalar = (
        np.isscalar(saturation)
        and np.isscalar(drainage_value)
        and np.isscalar(imbibition_value)
        and np.isscalar(reversal_saturation)
        and np.isscalar(max_saturation)
        and np.isscalar(is_imbibition)
    )
    if is_scalar:
        return _killough_interpolation_scalar(
            float(saturation),  # type: ignore[arg-type]
            float(drainage_value),  # type: ignore[arg-type]
            float(imbibition_value),  # type: ignore[arg-type]
            float(reversal_saturation),  # type: ignore[arg-type]
            float(max_saturation),  # type: ignore[arg-type]
            float(is_imbibition),  # type: ignore[arg-type]
            exponent,
            epsilon,
        )
    return _killough_interpolation_array(
        saturation,  # type: ignore[arg-type]
        drainage_value,  # type: ignore[arg-type]
        imbibition_value,  # type: ignore[arg-type]
        reversal_saturation,  # type: ignore[arg-type]
        max_saturation,  # type: ignore[arg-type]
        is_imbibition,  # type: ignore[arg-type]
        exponent,
        epsilon,
    )


@numba.njit(cache=True, inline="always")
def _killough_derivative_scalar(
    saturation: float,
    drainage_value: float,
    imbibition_value: float,
    d_drainage_d_sat: float,
    d_imbibition_d_sat: float,
    reversal_saturation: float,
    max_saturation: float,
    is_imbibition: float,
    exponent: float = 1.0,
    epsilon: float = 1e-12,
) -> float:
    """
    Analytical derivative of the Killough scanning-curve value with respect
    to the scanning saturation (scalar).

    The scanning curve is::

        V(S) = V_d(S) + [V_i(S) - V_d(S)] * f(S)

    By the product / chain rule::

        dV/dS = dV_d/dS + [dV_i/dS - dV_d/dS] * f
                        + [V_i - V_d] * df/dS

    :param saturation: Current saturation.
    :param drainage_value: Drainage curve value at current saturation.
    :param imbibition_value: Imbibition curve value at current saturation.
    :param d_drainage_d_sat: Derivative of drainage value w.r.t. saturation.
    :param d_imbibition_d_sat: Derivative of imbibition value w.r.t. saturation.
    :param reversal_saturation: Saturation at the last reversal (S_rev).
    :param max_saturation: Maximum saturation before the reversal (S_max).
    :param is_imbibition: 1.0 if imbibition, 0.0 if drainage.
    :param exponent: Killough exponent *n*.
    :param epsilon: Numerical tolerance.
    :return: Derivative of scanning-curve value w.r.t. saturation.
    """
    delta = max_saturation - reversal_saturation

    on_primary_drain = (is_imbibition < 0.5) and (
        abs(saturation - max_saturation) < epsilon
    )
    on_primary_imb = (is_imbibition >= 0.5) and (
        abs(saturation - reversal_saturation) < epsilon
    )
    if on_primary_drain:
        return d_drainage_d_sat
    if on_primary_imb:
        return d_imbibition_d_sat

    if abs(delta) > epsilon:
        raw_ratio = (saturation - reversal_saturation) / delta
    else:
        raw_ratio = 0.0

    ratio_clamped = min(max(raw_ratio, 0.0), 1.0)
    f = ratio_clamped**exponent

    in_range = (raw_ratio > 0.0) and (raw_ratio < 1.0) and (abs(delta) > epsilon)
    if in_range:
        if abs(exponent - 1.0) < 1e-10:
            df_dS = 1.0 / delta
        else:
            safe_ratio = ratio_clamped if ratio_clamped > 0.0 else 1e-30
            df_dS = exponent * (safe_ratio ** (exponent - 1.0)) / delta
    else:
        df_dS = 0.0

    return (
        d_drainage_d_sat
        + (d_imbibition_d_sat - d_drainage_d_sat) * f
        + (imbibition_value - drainage_value) * df_dS
    )


@numba.njit(cache=True)
def _killough_derivative_array(
    saturation: npt.NDArray,
    drainage_value: npt.NDArray,
    imbibition_value: npt.NDArray,
    d_drainage_d_sat: npt.NDArray,
    d_imbibition_d_sat: npt.NDArray,
    reversal_saturation: npt.NDArray,
    max_saturation: npt.NDArray,
    is_imbibition: npt.NDArray,
    exponent: float = 1.0,
    epsilon: float = 1e-12,
) -> npt.NDArray:
    """
    Analytical derivative of the Killough scanning-curve value with respect
    to the scanning saturation (array).

    Vectorised version of :func:`_killough_derivative_scalar`.

    :param saturation: Current saturation array.
    :param drainage_value: Drainage curve values at current saturations.
    :param imbibition_value: Imbibition curve values at current saturations.
    :param d_drainage_d_sat: Derivatives of drainage values w.r.t. saturation.
    :param d_imbibition_d_sat: Derivatives of imbibition values w.r.t. saturation.
    :param reversal_saturation: Saturations at the last reversal points.
    :param max_saturation: Maximum saturations before the reversals.
    :param is_imbibition: Per-cell imbibition flags (1.0 / 0.0).
    :param exponent: Killough exponent *n*.
    :param epsilon: Numerical tolerance.
    :return: Derivative array with the same shape as `saturation`.
    """
    sat = atleast_1d(saturation)
    val_d = atleast_1d(drainage_value)
    val_i = atleast_1d(imbibition_value)
    dv_d = atleast_1d(d_drainage_d_sat)
    dv_i = atleast_1d(d_imbibition_d_sat)
    s_rev = atleast_1d(reversal_saturation)
    s_max = atleast_1d(max_saturation)
    imb = atleast_1d(is_imbibition)

    sat, val_d, val_i, dv_d, dv_i, s_rev, s_max, imb = np.broadcast_arrays(
        sat, val_d, val_i, dv_d, dv_i, s_rev, s_max, imb
    )

    result = np.empty_like(sat)
    for k in numba.prange(sat.size):  # type: ignore
        result.flat[k] = _killough_derivative_scalar(
            sat.flat[k],
            val_d.flat[k],
            val_i.flat[k],
            dv_d.flat[k],
            dv_i.flat[k],
            s_rev.flat[k],
            s_max.flat[k],
            imb.flat[k],
            exponent,
            epsilon,
        )
    return result


def _killough_interpolation_derivative(
    saturation: FloatOrArray,
    drainage_value: FloatOrArray,
    imbibition_value: FloatOrArray,
    d_drainage_d_sat: FloatOrArray,
    d_imbibition_d_sat: FloatOrArray,
    reversal_saturation: FloatOrArray,
    max_saturation: FloatOrArray,
    is_imbibition: FloatOrArray,
    exponent: float = 1.0,
    epsilon: float = 1e-12,
) -> FloatOrArray:
    """
    Dispatch Killough scanning-curve derivative to the scalar or array kernel.

    :param saturation: Current saturation (scalar or array).
    :param drainage_value: Drainage curve value at current saturation.
    :param imbibition_value: Imbibition curve value at current saturation.
    :param d_drainage_d_sat: Derivative of drainage value w.r.t. saturation.
    :param d_imbibition_d_sat: Derivative of imbibition value w.r.t. saturation.
    :param reversal_saturation: Saturation at the last reversal (scalar or array).
    :param max_saturation: Maximum saturation before the reversal (scalar or array).
    :param is_imbibition: Imbibition flag (scalar or array).
    :param exponent: Killough exponent *n*.
    :param epsilon: Numerical tolerance.
    :return: Derivative of scanning-curve value w.r.t. saturation, matching
        the shape of the input.
    """
    is_scalar = (
        np.isscalar(saturation)
        and np.isscalar(drainage_value)
        and np.isscalar(imbibition_value)
        and np.isscalar(d_drainage_d_sat)
        and np.isscalar(d_imbibition_d_sat)
        and np.isscalar(reversal_saturation)
        and np.isscalar(max_saturation)
        and np.isscalar(is_imbibition)
    )
    if is_scalar:
        return _killough_derivative_scalar(
            float(saturation),  # type: ignore[arg-type]
            float(drainage_value),  # type: ignore[arg-type]
            float(imbibition_value),  # type: ignore[arg-type]
            float(d_drainage_d_sat),  # type: ignore[arg-type]
            float(d_imbibition_d_sat),  # type: ignore[arg-type]
            float(reversal_saturation),  # type: ignore[arg-type]
            float(max_saturation),  # type: ignore[arg-type]
            float(is_imbibition),  # type: ignore[arg-type]
            exponent,
            epsilon,
        )
    return _killough_derivative_array(
        saturation,  # type: ignore[arg-type]
        drainage_value,  # type: ignore[arg-type]
        imbibition_value,  # type: ignore[arg-type]
        d_drainage_d_sat,  # type: ignore[arg-type]
        d_imbibition_d_sat,  # type: ignore[arg-type]
        reversal_saturation,  # type: ignore[arg-type]
        max_saturation,  # type: ignore[arg-type]
        is_imbibition,  # type: ignore[arg-type]
        exponent,
        epsilon,
    )


def _ow_kr(
    table: typing.Union[TwoPhaseRelPermTable, RelativePermeabilityTable],
    sw: FloatOrArray,
    so: FloatOrArray,
    sg: FloatOrArray,
    **extra: typing.Any,
) -> typing.Tuple[FloatOrArray, FloatOrArray]:
    """
    Return `(krw, kro_w)` from an oil-water relative permeability table.

    Dispatches saturations correctly for both `TwoPhaseRelPermTable` and
    full `RelativePermeabilityTable` instances by calling
    `table.get_oil_water_wetting_phase()` rather than inspecting any
    internal attribute.

    :param table: Oil-water relative permeability table (two-phase or three-phase).
    :param sw: Water saturation (scalar or array).
    :param so: Oil saturation (scalar or array).
    :param sg: Gas saturation (scalar or array).
    :param extra: Additional keyword arguments forwarded to parametric tables.
    :return: Tuple of `(krw, kro_w)` — water and oil relative permeabilities
        from the oil-water sub-system.
    """
    ow_wetting = table.get_oil_water_wetting_phase()

    if isinstance(table, TwoPhaseRelPermTable):
        wetting_sat = sw if ow_wetting == FluidPhase.WATER else so
        non_wetting_sat = so if ow_wetting == FluidPhase.WATER else sw
        kr_wet = table.get_wetting_phase_relative_permeability(
            wetting_sat, non_wetting_sat
        )
        kr_nwet = table.get_non_wetting_phase_relative_permeability(
            wetting_sat, non_wetting_sat
        )
        if ow_wetting == FluidPhase.WATER:
            return kr_wet, kr_nwet  # krw, kro_w
        return kr_nwet, kr_wet  # krw, kro_w

    # Full three-phase table
    result = table.get_relative_permeabilities(
        water_saturation=sw,
        oil_saturation=so,
        gas_saturation=sg,
        **extra,
    )
    return result["water"], result["oil"]


def _go_kr(
    table: typing.Union[TwoPhaseRelPermTable, RelativePermeabilityTable],
    sw: FloatOrArray,
    so: FloatOrArray,
    sg: FloatOrArray,
    **extra: typing.Any,
) -> typing.Tuple[FloatOrArray, FloatOrArray]:
    """
    Return `(kro_g, krg)` from a gas-oil relative permeability table.

    Dispatches saturations correctly for both `TwoPhaseRelPermTable` and
    full `RelativePermeabilityTable` instances by calling
    `table.get_gas_oil_wetting_phase()` rather than inspecting any internal
    attribute.

    :param table: Gas-oil relative permeability table (two-phase or three-phase).
    :param sw: Water saturation (scalar or array).
    :param so: Oil saturation (scalar or array).
    :param sg: Gas saturation (scalar or array).
    :param extra: Additional keyword arguments forwarded to parametric tables.
    :return: Tuple of `(kro_g, krg)` — oil and gas relative permeabilities
        from the gas-oil sub-system.
    """
    go_wetting = table.get_gas_oil_wetting_phase()

    if isinstance(table, TwoPhaseRelPermTable):
        wetting_sat = so if go_wetting == FluidPhase.OIL else sg
        non_wetting_sat = sg if go_wetting == FluidPhase.OIL else so
        kr_wet = table.get_wetting_phase_relative_permeability(
            wetting_sat, non_wetting_sat
        )
        kr_nwet = table.get_non_wetting_phase_relative_permeability(
            wetting_sat, non_wetting_sat
        )
        if go_wetting == FluidPhase.OIL:
            return kr_wet, kr_nwet  # kro_g, krg
        return kr_nwet, kr_wet  # kro_g, krg

    result = table.get_relative_permeabilities(
        water_saturation=sw,
        oil_saturation=so,
        gas_saturation=sg,
        **extra,
    )
    return result["oil"], result["gas"]


def _ow_kr_deriv(
    table: typing.Union[TwoPhaseRelPermTable, RelativePermeabilityTable],
    sw: FloatOrArray,
    so: FloatOrArray,
    sg: FloatOrArray,
    **extra: typing.Any,
) -> typing.Tuple[FloatOrArray, FloatOrArray]:
    """
    Return `(d_krw/d_ref, d_kro_w/d_ref)` for the oil-water table, where
    *ref* is the table's natural reference saturation (Sw in water-wet, So in
    oil-wet).

    :param table: Oil-water relative permeability table (two-phase or three-phase).
    :param sw: Water saturation (scalar or array).
    :param so: Oil saturation (scalar or array).
    :param sg: Gas saturation (scalar or array).
    :param extra: Additional keyword arguments forwarded to parametric tables.
    :return: Tuple of `(d_krw/d_ref, d_kro_w/d_ref)` where *ref* is the
        reference saturation axis of the oil-water sub-system.
    """
    ow_wetting = table.get_oil_water_wetting_phase()

    if isinstance(table, TwoPhaseRelPermTable):
        wetting_sat = sw if ow_wetting == FluidPhase.WATER else so
        non_wetting_sat = so if ow_wetting == FluidPhase.WATER else sw
        d_wet = table.get_wetting_phase_relative_permeability_derivative(
            wetting_sat, non_wetting_sat
        )
        d_nwet = table.get_non_wetting_phase_relative_permeability_derivative(
            wetting_sat, non_wetting_sat
        )
        if ow_wetting == FluidPhase.WATER:
            return d_wet, d_nwet  # d_krw/d_Sw, d_kro_w/d_Sw
        return d_nwet, d_wet  # d_krw/d_So, d_kro_w/d_So

    derivs = table.get_relative_permeability_derivatives(
        water_saturation=sw,
        oil_saturation=so,
        gas_saturation=sg,
        **extra,
    )
    # The oil-water hysteresis scanning variable is always Sw
    return derivs["dKrw_dSw"], derivs["dKro_dSw"]


def _go_kr_deriv(
    table: typing.Union[TwoPhaseRelPermTable, RelativePermeabilityTable],
    sw: FloatOrArray,
    so: FloatOrArray,
    sg: FloatOrArray,
    **extra: typing.Any,
) -> typing.Tuple[FloatOrArray, FloatOrArray]:
    """
    Return `(d_kro_g/d_ref, d_krg/d_ref)` for the gas-oil table, where
    *ref* is the table's natural reference saturation (So in oil-wet, Sg in
    gas-wet).

    :param table: Gas-oil relative permeability table (two-phase or three-phase).
    :param sw: Water saturation (scalar or array).
    :param so: Oil saturation (scalar or array).
    :param sg: Gas saturation (scalar or array).
    :param extra: Additional keyword arguments forwarded to parametric tables.
    :return: Tuple of `(d_kro_g/d_ref, d_krg/d_ref)` where *ref* is the
        reference saturation axis of the gas-oil sub-system.
    """
    go_wetting = table.get_gas_oil_wetting_phase()

    if isinstance(table, TwoPhaseRelPermTable):
        wetting_sat = so if go_wetting == FluidPhase.OIL else sg
        non_wetting_sat = sg if go_wetting == FluidPhase.OIL else so
        d_wet = table.get_wetting_phase_relative_permeability_derivative(
            wetting_sat, non_wetting_sat
        )
        d_nwet = table.get_non_wetting_phase_relative_permeability_derivative(
            wetting_sat, non_wetting_sat
        )
        if go_wetting == FluidPhase.OIL:
            return d_wet, d_nwet  # d_kro_g/d_So, d_krg/d_So
        return d_nwet, d_wet  # d_kro_g/d_Sg, d_krg/d_Sg

    derivs = table.get_relative_permeability_derivatives(
        water_saturation=sw,
        oil_saturation=so,
        gas_saturation=sg,
        **extra,
    )
    # The gas-oil hysteresis scanning variable is always Sg
    return derivs["dKro_dSg"], derivs["dKrg_dSg"]


def _ow_pc(
    table: typing.Union[TwoPhaseCapillaryPressureTable, CapillaryPressureTable],
    sw: FloatOrArray,
    so: FloatOrArray,
    sg: FloatOrArray,
    **extra: typing.Any,
) -> FloatOrArray:
    """
    Extract Pcow from an oil-water capillary pressure table, dispatching
    saturations correctly via the canonical wetting-phase API.

    :param table: Oil-water capillary pressure table (two-phase or three-phase).
    :param sw: Water saturation (scalar or array).
    :param so: Oil saturation (scalar or array).
    :param sg: Gas saturation (scalar or array).
    :param extra: Additional keyword arguments forwarded to parametric tables.
    :return: Oil-water capillary pressure Pcow = Po - Pw (scalar or array).
    """
    ow_wetting = table.get_oil_water_wetting_phase()

    if isinstance(table, TwoPhaseCapillaryPressureTable):
        wetting_sat = sw if ow_wetting == FluidPhase.WATER else so
        non_wetting_sat = so if ow_wetting == FluidPhase.WATER else sw
        return table.get_capillary_pressure(wetting_sat, non_wetting_sat)

    result = table.get_capillary_pressures(
        water_saturation=sw,
        oil_saturation=so,
        gas_saturation=sg,
        **extra,
    )
    return result["oil_water"]


def _go_pc(
    table: typing.Union[TwoPhaseCapillaryPressureTable, CapillaryPressureTable],
    sw: FloatOrArray,
    so: FloatOrArray,
    sg: FloatOrArray,
    **extra: typing.Any,
) -> FloatOrArray:
    """
    Extract Pcgo from a gas-oil capillary pressure table, dispatching
    saturations correctly via the canonical wetting-phase API.

    :param table: Gas-oil capillary pressure table (two-phase or three-phase).
    :param sw: Water saturation (scalar or array).
    :param so: Oil saturation (scalar or array).
    :param sg: Gas saturation (scalar or array).
    :param extra: Additional keyword arguments forwarded to parametric tables.
    :return: Gas-oil capillary pressure Pcgo = Pg - Po (scalar or array).
    """
    go_wetting = table.get_gas_oil_wetting_phase()

    if isinstance(table, TwoPhaseCapillaryPressureTable):
        wetting_sat = so if go_wetting == FluidPhase.OIL else sg
        non_wetting_sat = sg if go_wetting == FluidPhase.OIL else so
        return table.get_capillary_pressure(wetting_sat, non_wetting_sat)

    result = table.get_capillary_pressures(
        water_saturation=sw,
        oil_saturation=so,
        gas_saturation=sg,
        **extra,
    )
    return result["gas_oil"]


def _ow_pc_deriv(
    table: typing.Union[TwoPhaseCapillaryPressureTable, CapillaryPressureTable],
    sw: FloatOrArray,
    so: FloatOrArray,
    sg: FloatOrArray,
    **extra: typing.Any,
) -> FloatOrArray:
    """
    Return dPcow/d(reference_sat) for the oil-water capillary pressure table.

    For `TwoPhaseCapillaryPressureTable` this is the derivative w.r.t. the
    table's own reference saturation (Sw in water-wet, So in oil-wet).  For
    three-phase tables `dPcow/dSw` is returned to match the scanning variable
    used by the hysteresis layer.

    :param table: Oil-water capillary pressure table (two-phase or three-phase).
    :param sw: Water saturation (scalar or array).
    :param so: Oil saturation (scalar or array).
    :param sg: Gas saturation (scalar or array).
    :param extra: Additional keyword arguments forwarded to parametric tables.
    :return: Derivative of Pcow w.r.t. the reference saturation axis
        (scalar or array).
    """
    ow_wetting = table.get_oil_water_wetting_phase()

    if isinstance(table, TwoPhaseCapillaryPressureTable):
        wetting_sat = sw if ow_wetting == FluidPhase.WATER else so
        non_wetting_sat = so if ow_wetting == FluidPhase.WATER else sw
        return table.get_capillary_pressure_derivative(wetting_sat, non_wetting_sat)

    derivs = table.get_capillary_pressure_derivatives(
        water_saturation=sw,
        oil_saturation=so,
        gas_saturation=sg,
        **extra,
    )
    return derivs["dPcow_dSw"]


def _go_pc_deriv(
    table: typing.Union[TwoPhaseCapillaryPressureTable, CapillaryPressureTable],
    sw: FloatOrArray,
    so: FloatOrArray,
    sg: FloatOrArray,
    **extra: typing.Any,
) -> FloatOrArray:
    """
    Return dPcgo/d(reference_sat) for the gas-oil capillary pressure table.

    For `TwoPhaseCapillaryPressureTable` this is the derivative w.r.t. the
    table's own reference saturation.  For three-phase tables
    `dPcgo/dSg` is returned to match the scanning variable used by the
    hysteresis layer.

    :param table: Gas-oil capillary pressure table (two-phase or three-phase).
    :param sw: Water saturation (scalar or array).
    :param so: Oil saturation (scalar or array).
    :param sg: Gas saturation (scalar or array).
    :param extra: Additional keyword arguments forwarded to parametric tables.
    :return: Derivative of Pcgo w.r.t. the reference saturation axis
        (scalar or array).
    """
    go_wetting = table.get_gas_oil_wetting_phase()

    if isinstance(table, TwoPhaseCapillaryPressureTable):
        wetting_sat = so if go_wetting == FluidPhase.OIL else sg
        non_wetting_sat = sg if go_wetting == FluidPhase.OIL else so
        return table.get_capillary_pressure_derivative(wetting_sat, non_wetting_sat)

    derivs = table.get_capillary_pressure_derivatives(
        water_saturation=sw,
        oil_saturation=so,
        gas_saturation=sg,
        **extra,
    )
    return derivs["dPcgo_dSg"]


@relperm_table
@attrs.frozen
class KilloughLandRelPermModel(
    RelativePermeabilityTable,
    serializers={"mixing_rule": serialize_mixing_rule},
    deserializers={"mixing_rule": get_mixing_rule},
    load_exclude={"supports_arrays"},
    dump_exclude={"supports_arrays"},
):
    """
    Killough relative permeability hysteresis model with Land trapping.

    During *primary drainage* the relative permeabilities follow the
    `oil_water_drainage_table` and `gas_oil_drainage_table`.

    When the flow reverses (*imbibition*) two things happen simultaneously.

    **Land trapping** — a portion of the non-wetting phase becomes
    disconnected.  The dynamic residual saturation depends on the saturation
    at the reversal point via Land's formula::

        S_r(S_i) = S_r_max / (1 + C * S_i)

    **Killough scanning curves** — between the reversal point and the
    maximum historical saturation, kr follows a scanning curve that
    interpolates between the primary drainage and imbibition bounds.

    Both two-phase (`TwoPhaseRelPermTable`) and full three-phase
    (`RelativePermeabilityTable`) backing tables are supported.  Wetting
    and non-wetting phase roles are resolved through the canonical API
    (`get_oil_water_wetting_phase` / `get_gas_oil_wetting_phase`) so the
    model is wettability-agnostic.

    The hysteresis history is passed as additional keyword arguments to
    `get_relative_permeabilities` and
    `get_relative_permeability_derivatives`. When these arguments are
    absent the model degenerates to the primary drainage curves, which is the
    physically correct behaviour for the first drainage cycle (see
    `simulate.py` — the `enable_hysteresis` flag in `Config` controls
    whether history is tracked and passed through).
    """

    __type__ = "killough_land_relperm_model"

    oil_water_drainage_table: typing.Union[
        TwoPhaseRelPermTable, RelativePermeabilityTable
    ]
    """Primary drainage relative permeability table for the oil-water system."""

    gas_oil_drainage_table: typing.Union[
        TwoPhaseRelPermTable, RelativePermeabilityTable
    ]
    """Primary drainage relative permeability table for the gas-oil system."""

    oil_water_imbibition_table: typing.Optional[
        typing.Union[TwoPhaseRelPermTable, RelativePermeabilityTable]
    ] = None
    """Primary imbibition table for the oil-water system. Defaults to the drainage table."""

    gas_oil_imbibition_table: typing.Optional[
        typing.Union[TwoPhaseRelPermTable, RelativePermeabilityTable]
    ] = None
    """Primary imbibition table for the gas-oil system. Defaults to the drainage table."""

    land_coefficient_water: float = 1.0
    """Land trapping coefficient *C* for the oil-water system (≥ 0)."""

    land_coefficient_gas: float = 1.0
    """Land trapping coefficient *C* for the gas-oil system (≥ 0)."""

    residual_oil_saturation_max_water: typing.Optional[float] = None
    """
    Maximum residual oil saturation S_r_max used by Land's formula for the
    oil-water system. Required when `oil_water_imbibition_table` is set.
    """

    residual_oil_saturation_max_gas: typing.Optional[float] = None
    """
    Maximum residual oil saturation S_r_max for the gas-oil system. Required
    when `gas_oil_imbibition_table` is set.
    """

    residual_gas_saturation_max: typing.Optional[float] = None
    """
    Maximum residual gas saturation S_r_max used by Land's formula. Required
    when `gas_oil_imbibition_table` is set.
    """

    scanning_interpolation_exponent: float = 1.0
    """Killough scanning curve interpolation exponent *n* (1 = linear)."""

    mixing_rule: typing.Union[MixingRule, str] = "eclipse_rule"
    """Three-phase oil relative permeability mixing rule."""

    supports_arrays: bool = attrs.field(init=False, repr=False, default=True)

    def __attrs_post_init__(self) -> None:
        if isinstance(self.mixing_rule, str):
            object.__setattr__(self, "mixing_rule", get_mixing_rule(self.mixing_rule))

        if isinstance(self.oil_water_drainage_table, TwoPhaseRelPermTable) and {
            self.oil_water_drainage_table.wetting_phase,
            self.oil_water_drainage_table.non_wetting_phase,
        } != {FluidPhase.WATER, FluidPhase.OIL}:
            raise ValidationError(
                "`oil_water_drainage_table` must involve water and oil phases."
            )

        if isinstance(self.gas_oil_drainage_table, TwoPhaseRelPermTable) and {
            self.gas_oil_drainage_table.wetting_phase,
            self.gas_oil_drainage_table.non_wetting_phase,
        } != {FluidPhase.OIL, FluidPhase.GAS}:
            raise ValidationError(
                "`gas_oil_drainage_table` must involve oil and gas phases."
            )

        if (
            self.oil_water_imbibition_table is not None
            and isinstance(self.oil_water_imbibition_table, TwoPhaseRelPermTable)
            and {
                self.oil_water_imbibition_table.wetting_phase,
                self.oil_water_imbibition_table.non_wetting_phase,
            }
            != {FluidPhase.WATER, FluidPhase.OIL}
        ):
            raise ValidationError(
                "`oil_water_imbibition_table` must involve water and oil phases."
            )

        if (
            self.gas_oil_imbibition_table is not None
            and isinstance(self.gas_oil_imbibition_table, TwoPhaseRelPermTable)
            and {
                self.gas_oil_imbibition_table.wetting_phase,
                self.gas_oil_imbibition_table.non_wetting_phase,
            }
            != {FluidPhase.OIL, FluidPhase.GAS}
        ):
            raise ValidationError(
                "`gas_oil_imbibition_table` must involve oil and gas phases."
            )

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

    def _parse_hysteresis_kwargs(
        self,
        sw: npt.NDArray,
        sg: npt.NDArray,
        max_water_saturation: typing.Optional[FloatOrArray],
        max_gas_saturation: typing.Optional[FloatOrArray],
        water_imbibition_flag: typing.Optional[typing.Union[bool, npt.NDArray]],
        gas_imbibition_flag: typing.Optional[typing.Union[bool, npt.NDArray]],
        water_reversal_saturation: typing.Optional[FloatOrArray],
        gas_reversal_saturation: typing.Optional[FloatOrArray],
    ) -> typing.Tuple[
        npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
    ]:
        """
        Parse and broadcast saturation-history arrays.

        When all history arguments are `None` the method returns arrays that
        replicate the primary-drainage state (no-hysteresis fallback).

        :param sw: Broadcast-ready water saturation array.
        :param sg: Broadcast-ready gas saturation array.
        :param max_water_saturation: Historical maximum water saturation grid or `None`.
        :param max_gas_saturation: Historical maximum gas saturation grid or `None`.
        :param water_imbibition_flag: Per-cell flag: 1 = water imbibition, 0 = drainage,
            or `None`.
        :param gas_imbibition_flag: Per-cell flag: 1 = gas imbibition, 0 = drainage,
            or `None`.
        :param water_reversal_saturation: Water saturation at last oil-water reversal,
            or `None` (defaults to `max_water_saturation`).
        :param gas_reversal_saturation: Gas saturation at last gas-oil reversal, or
            `None` (defaults to `max_gas_saturation`).
        :return: Six broadcast-compatible arrays: `(sw_max, sg_max, sw_imb, sg_imb,
            sw_rev, sg_rev)`.
        """
        use_hysteresis = (
            max_water_saturation is not None
            and max_gas_saturation is not None
            and water_imbibition_flag is not None
            and gas_imbibition_flag is not None
        )
        if use_hysteresis:
            sw_max = np.atleast_1d(np.asarray(max_water_saturation, dtype=np.float64))
            sg_max = np.atleast_1d(np.asarray(max_gas_saturation, dtype=np.float64))
            sw_imb = np.atleast_1d(np.asarray(water_imbibition_flag, dtype=np.float64))
            sg_imb = np.atleast_1d(np.asarray(gas_imbibition_flag, dtype=np.float64))
            sw_rev = (
                np.atleast_1d(np.asarray(water_reversal_saturation, dtype=np.float64))
                if water_reversal_saturation is not None
                else sw_max.copy()
            )
            sg_rev = (
                np.atleast_1d(np.asarray(gas_reversal_saturation, dtype=np.float64))
                if gas_reversal_saturation is not None
                else sg_max.copy()
            )
        else:
            sw_max = sw.copy()
            sg_max = sg.copy()
            sw_imb = np.zeros_like(sw)
            sg_imb = np.zeros_like(sg)
            sw_rev = sw.copy()
            sg_rev = sg.copy()

        sw_max, sg_max, sw_imb, sg_imb, sw_rev, sg_rev = np.broadcast_arrays(
            sw_max, sg_max, sw_imb, sg_imb, sw_rev, sg_rev
        )
        return sw_max, sg_max, sw_imb, sg_imb, sw_rev, sg_rev

    def get_relative_permeabilities(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        max_water_saturation: typing.Optional[FloatOrArray] = None,
        max_gas_saturation: typing.Optional[FloatOrArray] = None,
        water_imbibition_flag: typing.Optional[
            typing.Union[bool, npt.NDArray[np.bool_]]
        ] = None,
        gas_imbibition_flag: typing.Optional[
            typing.Union[bool, npt.NDArray[np.bool_]]
        ] = None,
        water_reversal_saturation: typing.Optional[FloatOrArray] = None,
        gas_reversal_saturation: typing.Optional[FloatOrArray] = None,
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute three-phase relative permeabilities with Killough/Land hysteresis.

        When the saturation-history keyword arguments are absent the method
        returns primary drainage kr values (no hysteresis).

        :param water_saturation: Water saturation (fraction, 0-1) — scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) — scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) — scalar or array.
        :param max_water_saturation: Historical maximum water saturation per cell.
            Enables hysteresis when provided together with the other history args.
        :param max_gas_saturation: Historical maximum gas saturation per cell.
        :param water_imbibition_flag: Boolean / float flag per cell — 1 if water
            saturation is currently increasing (imbibition in the oil-water
            system), 0 if decreasing (drainage).
        :param gas_imbibition_flag: Boolean / float flag per cell — 1 if gas
            saturation is currently increasing, 0 if decreasing.
        :param water_reversal_saturation: Water saturation at the last oil-water
            reversal.  Defaults to `max_water_saturation` when not supplied.
        :param gas_reversal_saturation: Gas saturation at the last gas-oil reversal.
            Defaults to `max_gas_saturation` when not supplied.
        :param kwargs: Additional keyword arguments forwarded to the underlying
            backing tables (e.g. residual saturation overrides for parametric
            models).
        :return: `RelativePermeabilities` dictionary with keys `"water"`,
            `"oil"`, and `"gas"`.
        """
        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
        )
        sw = np.atleast_1d(np.asarray(water_saturation, dtype=np.float64))
        so = np.atleast_1d(np.asarray(oil_saturation, dtype=np.float64))
        sg = np.atleast_1d(np.asarray(gas_saturation, dtype=np.float64))
        sw, so, sg = np.broadcast_arrays(sw, so, sg)

        # Normalise saturations
        total = sw + so + sg
        mask = (np.abs(total - 1.0) > c.SATURATION_EPSILON) & (total > 0.0)
        if np.any(mask):
            sw = np.where(mask, sw / total, sw)
            so = np.where(mask, so / total, so)
            sg = np.where(mask, sg / total, sg)

        sw_max, sg_max, sw_imb, sg_imb, sw_rev, sg_rev = self._parse_hysteresis_kwargs(
            sw,
            sg,
            max_water_saturation,
            max_gas_saturation,
            water_imbibition_flag,
            gas_imbibition_flag,
            water_reversal_saturation,
            gas_reversal_saturation,
        )

        ow_drain = self.oil_water_drainage_table
        ow_imb = self.oil_water_imbibition_table or ow_drain
        go_drain = self.gas_oil_drainage_table
        go_imb = self.gas_oil_imbibition_table or go_drain
        use_hysteresis = (
            max_water_saturation is not None and max_gas_saturation is not None
        )

        # Oil-water system — Land trapping on oil
        so_at_ow_reversal = np.maximum(1.0 - sw_rev - sg, 0.0)
        imb_ow_kwargs = dict(kwargs)
        if use_hysteresis and self.residual_oil_saturation_max_water is not None:
            sor_dyn_water = _land_residual_saturation(
                so_at_ow_reversal,
                self.residual_oil_saturation_max_water,
                self.land_coefficient_water,
            )
            imb_ow_kwargs["residual_oil_saturation_water"] = sor_dyn_water

        krw_drain, kro_w_drain = _ow_kr(ow_drain, sw, so, sg, **kwargs)
        krw_imb, kro_w_imb = _ow_kr(ow_imb, sw, so, sg, **imb_ow_kwargs)

        krw = _killough_interpolation(
            sw,
            krw_drain,
            krw_imb,
            sw_rev,
            sw_max,
            sw_imb,
            exponent=self.scanning_interpolation_exponent,
        )
        kro_w = _killough_interpolation(
            sw,
            kro_w_drain,
            kro_w_imb,
            sw_rev,
            sw_max,
            sw_imb,
            exponent=self.scanning_interpolation_exponent,
        )

        # Gas-oil system — Land trapping on gas and oil
        so_at_go_reversal = np.maximum(1.0 - sg_rev - sw, 0.0)
        imb_go_kwargs = dict(kwargs)
        if use_hysteresis and self.residual_gas_saturation_max is not None:
            sgr_dyn = _land_residual_saturation(
                sg_rev,
                self.residual_gas_saturation_max,
                self.land_coefficient_gas,
            )
            imb_go_kwargs["residual_gas_saturation"] = sgr_dyn
        if use_hysteresis and self.residual_oil_saturation_max_gas is not None:
            sor_dyn_gas = _land_residual_saturation(
                so_at_go_reversal,
                self.residual_oil_saturation_max_gas,
                self.land_coefficient_gas,
            )
            imb_go_kwargs["residual_oil_saturation_gas"] = sor_dyn_gas

        kro_g_drain, krg_drain = _go_kr(go_drain, sw, so, sg, **kwargs)
        kro_g_imb, krg_imb = _go_kr(go_imb, sw, so, sg, **imb_go_kwargs)

        krg = _killough_interpolation(
            sg,
            krg_drain,
            krg_imb,
            sg_rev,
            sg_max,
            sg_imb,
            exponent=self.scanning_interpolation_exponent,
        )
        kro_g = _killough_interpolation(
            sg,
            kro_g_drain,
            kro_g_imb,
            sg_rev,
            sg_max,
            sg_imb,
            exponent=self.scanning_interpolation_exponent,
        )
        # Three-phase oil via mixing rule
        kro = self.mixing_rule(  # type: ignore[operator]
            kro_w=kro_w,
            kro_g=kro_g,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
        )

        if is_scalar:
            return RelativePermeabilities(
                water=float(np.atleast_1d(krw).flat[0]),
                oil=float(np.atleast_1d(kro).flat[0]),
                gas=float(np.atleast_1d(krg).flat[0]),
            )
        return RelativePermeabilities(water=krw, oil=kro, gas=krg)

    def get_relative_permeability_derivatives(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        max_water_saturation: typing.Optional[FloatOrArray] = None,
        max_gas_saturation: typing.Optional[FloatOrArray] = None,
        water_imbibition_flag: typing.Optional[
            typing.Union[bool, npt.NDArray[np.bool_]]
        ] = None,
        gas_imbibition_flag: typing.Optional[
            typing.Union[bool, npt.NDArray[np.bool_]]
        ] = None,
        water_reversal_saturation: typing.Optional[FloatOrArray] = None,
        gas_reversal_saturation: typing.Optional[FloatOrArray] = None,
        **kwargs: typing.Any,
    ) -> RelativePermeabilityDerivatives:
        """
        Compute partial derivatives of three-phase relative permeabilities
        with Killough/Land hysteresis.

        Returns all nine ∂kr_α/∂S_β entries.  Derivatives with respect to
        the scanning variable (Sw for oil-water, Sg for gas-oil) are computed
        analytically via the chain rule through the Killough scanning-curve
        formula; all other cross-derivatives are zero (consistent with the
        assumption that each two-phase sub-system depends only on its own
        reference saturation).

        :param water_saturation: Water saturation (fraction, 0-1) — scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) — scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) — scalar or array.
        :param max_water_saturation: Historical maximum water saturation per cell,
            or `None` to disable hysteresis.
        :param max_gas_saturation: Historical maximum gas saturation per cell,
            or `None` to disable hysteresis.
        :param water_imbibition_flag: Per-cell imbibition flag for the oil-water
            system (1 = imbibition, 0 = drainage), or `None`.
        :param gas_imbibition_flag: Per-cell imbibition flag for the gas-oil
            system (1 = imbibition, 0 = drainage), or `None`.
        :param water_reversal_saturation: Water saturation at the last oil-water
            reversal, or `None` (defaults to `max_water_saturation`).
        :param gas_reversal_saturation: Gas saturation at the last gas-oil reversal,
            or `None` (defaults to `max_gas_saturation`).
        :param kwargs: Additional keyword arguments forwarded to the backing tables.
        :return: `RelativePermeabilityDerivatives` dictionary containing all
            nine ∂kr/∂S entries.
        """
        is_scalar = np.isscalar(water_saturation)
        sw = np.atleast_1d(np.asarray(water_saturation, dtype=np.float64))
        so = np.atleast_1d(np.asarray(oil_saturation, dtype=np.float64))
        sg = np.atleast_1d(np.asarray(gas_saturation, dtype=np.float64))
        sw, so, sg = np.broadcast_arrays(sw, so, sg)
        zeros = np.zeros_like(sw)

        sw_max, sg_max, sw_imb, sg_imb, sw_rev, sg_rev = self._parse_hysteresis_kwargs(
            sw,
            sg,
            max_water_saturation,
            max_gas_saturation,
            water_imbibition_flag,
            gas_imbibition_flag,
            water_reversal_saturation,
            gas_reversal_saturation,
        )

        ow_drain = self.oil_water_drainage_table
        ow_imb = self.oil_water_imbibition_table or ow_drain
        go_drain = self.gas_oil_drainage_table
        go_imb = self.gas_oil_imbibition_table or go_drain
        use_hysteresis = (
            max_water_saturation is not None and max_gas_saturation is not None
        )

        # Build kwargs for imbibition tables (Land trapping)
        so_at_ow_reversal = np.maximum(1.0 - sw_rev - sg, 0.0)
        imb_ow_kwargs = dict(kwargs)
        if use_hysteresis and self.residual_oil_saturation_max_water is not None:
            sor_dyn_water = _land_residual_saturation(
                so_at_ow_reversal,
                self.residual_oil_saturation_max_water,
                self.land_coefficient_water,
            )
            imb_ow_kwargs["residual_oil_saturation_water"] = sor_dyn_water

        so_at_go_reversal = np.maximum(1.0 - sg_rev - sw, 0.0)
        imb_go_kwargs = dict(kwargs)
        if use_hysteresis and self.residual_gas_saturation_max is not None:
            sgr_dyn = _land_residual_saturation(
                sg_rev,
                self.residual_gas_saturation_max,
                self.land_coefficient_gas,
            )
            imb_go_kwargs["residual_gas_saturation"] = sgr_dyn
        if use_hysteresis and self.residual_oil_saturation_max_gas is not None:
            sor_dyn_gas = _land_residual_saturation(
                so_at_go_reversal,
                self.residual_oil_saturation_max_gas,
                self.land_coefficient_gas,
            )
            imb_go_kwargs["residual_oil_saturation_gas"] = sor_dyn_gas

        # Oil-water — values and derivatives
        krw_drain, kro_w_drain = _ow_kr(ow_drain, sw, so, sg, **kwargs)
        krw_imb, kro_w_imb = _ow_kr(ow_imb, sw, so, sg, **imb_ow_kwargs)

        d_krw_drain_d_sw, d_kro_w_drain_d_sw = _ow_kr_deriv(
            ow_drain, sw, so, sg, **kwargs
        )
        d_krw_imb_d_sw, d_kro_w_imb_d_sw = _ow_kr_deriv(
            ow_imb, sw, so, sg, **imb_ow_kwargs
        )

        d_krw_d_sw = _killough_interpolation_derivative(
            sw,
            krw_drain,
            krw_imb,
            d_krw_drain_d_sw,
            d_krw_imb_d_sw,
            sw_rev,
            sw_max,
            sw_imb,
            exponent=self.scanning_interpolation_exponent,
        )
        d_kro_w_d_sw = _killough_interpolation_derivative(
            sw,
            kro_w_drain,
            kro_w_imb,
            d_kro_w_drain_d_sw,
            d_kro_w_imb_d_sw,
            sw_rev,
            sw_max,
            sw_imb,
            exponent=self.scanning_interpolation_exponent,
        )

        krw = _killough_interpolation(
            sw,
            krw_drain,
            krw_imb,
            sw_rev,
            sw_max,
            sw_imb,
            exponent=self.scanning_interpolation_exponent,
        )
        kro_w = _killough_interpolation(
            sw,
            kro_w_drain,
            kro_w_imb,
            sw_rev,
            sw_max,
            sw_imb,
            exponent=self.scanning_interpolation_exponent,
        )

        # Gas-oil — values and derivatives
        kro_g_drain, krg_drain = _go_kr(go_drain, sw, so, sg, **kwargs)
        kro_g_imb, krg_imb = _go_kr(go_imb, sw, so, sg, **imb_go_kwargs)

        d_kro_g_drain_d_sg, d_krg_drain_d_sg = _go_kr_deriv(
            go_drain, sw, so, sg, **kwargs
        )
        d_kro_g_imb_d_sg, d_krg_imb_d_sg = _go_kr_deriv(
            go_imb, sw, so, sg, **imb_go_kwargs
        )

        d_krg_d_sg = _killough_interpolation_derivative(
            sg,
            krg_drain,
            krg_imb,
            d_krg_drain_d_sg,
            d_krg_imb_d_sg,
            sg_rev,
            sg_max,
            sg_imb,
            exponent=self.scanning_interpolation_exponent,
        )
        d_kro_g_d_sg = _killough_interpolation_derivative(
            sg,
            kro_g_drain,
            kro_g_imb,
            d_kro_g_drain_d_sg,
            d_kro_g_imb_d_sg,
            sg_rev,
            sg_max,
            sg_imb,
            exponent=self.scanning_interpolation_exponent,
        )

        krg = _killough_interpolation(
            sg,
            krg_drain,
            krg_imb,
            sg_rev,
            sg_max,
            sg_imb,
            exponent=self.scanning_interpolation_exponent,
        )
        kro_g = _killough_interpolation(
            sg,
            kro_g_drain,
            kro_g_imb,
            sg_rev,
            sg_max,
            sg_imb,
            exponent=self.scanning_interpolation_exponent,
        )

        # Three-phase oil mixing rule — chain rule
        mixing_rule = typing.cast(MixingRule, self.mixing_rule)
        derivs = get_mixing_rule_partial_derivatives(
            rule=mixing_rule,
            kro_w=kro_w,
            kro_g=kro_g,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            epsilon=c.FINITE_DIFFERENCE_EPSILON,
        )

        d_kro_d_sw = (
            derivs["d_kro_d_kro_w"] * d_kro_w_d_sw + derivs["d_kro_d_sw_explicit"]
        )
        d_kro_d_so = derivs["d_kro_d_so_explicit"]
        d_kro_d_sg = (
            derivs["d_kro_d_kro_g"] * d_kro_g_d_sg + derivs["d_kro_d_sg_explicit"]
        )

        if is_scalar:

            def _s(x: FloatOrArray) -> float:
                return float(np.atleast_1d(x).flat[0])

            return RelativePermeabilityDerivatives(
                dKrw_dSw=_s(d_krw_d_sw),
                dKro_dSw=_s(d_kro_d_sw),
                dKrg_dSw=0.0,
                dKrw_dSo=0.0,
                dKro_dSo=_s(d_kro_d_so),
                dKrg_dSo=0.0,
                dKrw_dSg=0.0,
                dKro_dSg=_s(d_kro_d_sg),
                dKrg_dSg=_s(d_krg_d_sg),
            )

        return RelativePermeabilityDerivatives(
            dKrw_dSw=d_krw_d_sw,
            dKro_dSw=d_kro_d_sw,
            dKrg_dSw=zeros.copy(),
            dKrw_dSo=zeros.copy(),
            dKro_dSo=d_kro_d_so,
            dKrg_dSo=zeros.copy(),
            dKrw_dSg=zeros.copy(),
            dKro_dSg=d_kro_d_sg,
            dKrg_dSg=d_krg_d_sg,
        )


@capillary_pressure_table
@attrs.frozen
class KilloughCapillaryPressureModel(
    CapillaryPressureTable,
    load_exclude={"supports_arrays"},
    dump_exclude={"supports_arrays"},
):
    """
    Killough capillary pressure hysteresis model.

    Capillary pressure hysteresis involves no trapping.  When the displacement
    direction reverses, capillary pressure traces a *scanning curve* that
    interpolates between the primary drainage and imbibition bounds.

    Both two-phase (`TwoPhaseCapillaryPressureTable`) and full three-phase
    (`CapillaryPressureTable`) backing tables are supported.  Wetting and
    non-wetting phase roles are resolved through the canonical API
    (`get_oil_water_wetting_phase` / `get_gas_oil_wetting_phase`) so the
    model is wettability-agnostic.

    The oil-water scanning curve scans over *water* saturation; the gas-oil
    scanning curve scans over *gas* saturation.

    Saturation history is passed as additional keyword arguments to
    `get_capillary_pressures` and `get_capillary_pressure_derivatives`.
    When these arguments are absent the model returns primary drainage Pc
    values.
    """

    __type__ = "killough_capillary_pressure_model"

    oil_water_drainage_table: typing.Union[
        TwoPhaseCapillaryPressureTable, CapillaryPressureTable
    ]
    """Primary drainage capillary pressure table for the oil-water system."""

    gas_oil_drainage_table: typing.Union[
        TwoPhaseCapillaryPressureTable, CapillaryPressureTable
    ]
    """Primary drainage capillary pressure table for the gas-oil system."""

    oil_water_imbibition_table: typing.Optional[
        typing.Union[TwoPhaseCapillaryPressureTable, CapillaryPressureTable]
    ] = None
    """Primary imbibition Pc table for the oil-water system. Defaults to the drainage table."""

    gas_oil_imbibition_table: typing.Optional[
        typing.Union[TwoPhaseCapillaryPressureTable, CapillaryPressureTable]
    ] = None
    """Primary imbibition Pc table for the gas-oil system. Defaults to the drainage table."""

    scanning_interpolation_exponent: float = 1.0
    """Killough interpolation exponent *n* (1 = linear)."""

    supports_arrays: bool = attrs.field(init=False, repr=False, default=True)

    def __attrs_post_init__(self) -> None:
        if isinstance(
            self.oil_water_drainage_table, TwoPhaseCapillaryPressureTable
        ) and {
            self.oil_water_drainage_table.wetting_phase,
            self.oil_water_drainage_table.non_wetting_phase,
        } != {FluidPhase.WATER, FluidPhase.OIL}:
            raise ValidationError(
                "`oil_water_drainage_table` must involve water and oil phases."
            )

        if isinstance(self.gas_oil_drainage_table, TwoPhaseCapillaryPressureTable) and {
            self.gas_oil_drainage_table.wetting_phase,
            self.gas_oil_drainage_table.non_wetting_phase,
        } != {FluidPhase.OIL, FluidPhase.GAS}:
            raise ValidationError(
                "`gas_oil_drainage_table` must involve oil and gas phases."
            )

        if (
            self.oil_water_imbibition_table is not None
            and isinstance(
                self.oil_water_imbibition_table, TwoPhaseCapillaryPressureTable
            )
            and {
                self.oil_water_imbibition_table.wetting_phase,
                self.oil_water_imbibition_table.non_wetting_phase,
            }
            != {FluidPhase.WATER, FluidPhase.OIL}
        ):
            raise ValidationError(
                "`oil_water_imbibition_table` must involve water and oil phases."
            )

        if (
            self.gas_oil_imbibition_table is not None
            and isinstance(
                self.gas_oil_imbibition_table, TwoPhaseCapillaryPressureTable
            )
            and {
                self.gas_oil_imbibition_table.wetting_phase,
                self.gas_oil_imbibition_table.non_wetting_phase,
            }
            != {FluidPhase.OIL, FluidPhase.GAS}
        ):
            raise ValidationError(
                "`gas_oil_imbibition_table` must involve oil and gas phases."
            )

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

    def _parse_hysteresis_kwargs(
        self,
        sw: npt.NDArray,
        sg: npt.NDArray,
        max_water_saturation: typing.Optional[FloatOrArray],
        max_gas_saturation: typing.Optional[FloatOrArray],
        water_imbibition_flag: typing.Optional[typing.Union[bool, npt.NDArray]],
        gas_imbibition_flag: typing.Optional[typing.Union[bool, npt.NDArray]],
        water_reversal_saturation: typing.Optional[FloatOrArray],
        gas_reversal_saturation: typing.Optional[FloatOrArray],
    ) -> typing.Tuple[
        npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
    ]:
        """
        Parse and broadcast saturation-history arrays.

        When all history arguments are `None` the method returns arrays that
        replicate the primary-drainage state (no-hysteresis fallback).

        :param sw: Broadcast-ready water saturation array.
        :param sg: Broadcast-ready gas saturation array.
        :param max_water_saturation: Historical maximum water saturation or `None`.
        :param max_gas_saturation: Historical maximum gas saturation or `None`.
        :param water_imbibition_flag: Per-cell oil-water imbibition flag or `None`.
        :param gas_imbibition_flag: Per-cell gas-oil imbibition flag or `None`.
        :param water_reversal_saturation: Water saturation at last oil-water reversal,
            or `None` (defaults to `max_water_saturation`).
        :param gas_reversal_saturation: Gas saturation at last gas-oil reversal, or
            `None` (defaults to `max_gas_saturation`).
        :return: Six broadcast-compatible arrays: `(sw_max, sg_max, sw_imb, sg_imb,
            sw_rev, sg_rev)`.
        """
        use_hysteresis = (
            max_water_saturation is not None
            and max_gas_saturation is not None
            and water_imbibition_flag is not None
            and gas_imbibition_flag is not None
        )
        if use_hysteresis:
            sw_max = np.atleast_1d(np.asarray(max_water_saturation, dtype=np.float64))
            sg_max = np.atleast_1d(np.asarray(max_gas_saturation, dtype=np.float64))
            sw_imb = np.atleast_1d(np.asarray(water_imbibition_flag, dtype=np.float64))
            sg_imb = np.atleast_1d(np.asarray(gas_imbibition_flag, dtype=np.float64))
            sw_rev = (
                np.atleast_1d(np.asarray(water_reversal_saturation, dtype=np.float64))
                if water_reversal_saturation is not None
                else sw_max.copy()
            )
            sg_rev = (
                np.atleast_1d(np.asarray(gas_reversal_saturation, dtype=np.float64))
                if gas_reversal_saturation is not None
                else sg_max.copy()
            )
        else:
            sw_max = sw.copy()
            sg_max = sg.copy()
            sw_imb = np.zeros_like(sw)
            sg_imb = np.zeros_like(sg)
            sw_rev = sw.copy()
            sg_rev = sg.copy()

        sw_max, sg_max, sw_imb, sg_imb, sw_rev, sg_rev = np.broadcast_arrays(
            sw_max, sg_max, sw_imb, sg_imb, sw_rev, sg_rev
        )
        return sw_max, sg_max, sw_imb, sg_imb, sw_rev, sg_rev

    def get_capillary_pressures(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        max_water_saturation: typing.Optional[FloatOrArray] = None,
        max_gas_saturation: typing.Optional[FloatOrArray] = None,
        water_imbibition_flag: typing.Optional[
            typing.Union[bool, npt.NDArray[np.bool_]]
        ] = None,
        gas_imbibition_flag: typing.Optional[
            typing.Union[bool, npt.NDArray[np.bool_]]
        ] = None,
        water_reversal_saturation: typing.Optional[FloatOrArray] = None,
        gas_reversal_saturation: typing.Optional[FloatOrArray] = None,
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute three-phase capillary pressures with Killough hysteresis.

        When the saturation-history keyword arguments are absent the method
        returns primary drainage Pc values.

        :param water_saturation: Water saturation (fraction, 0-1) — scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) — scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) — scalar or array.
        :param max_water_saturation: Historical maximum water saturation per cell.
        :param max_gas_saturation: Historical maximum gas saturation per cell.
        :param water_imbibition_flag: Per-cell oil-water imbibition flag (1 / 0),
            or `None`.
        :param gas_imbibition_flag: Per-cell gas-oil imbibition flag (1 / 0),
            or `None`.
        :param water_reversal_saturation: Water saturation at the last oil-water
            reversal, or `None` (defaults to `max_water_saturation`).
        :param gas_reversal_saturation: Gas saturation at the last gas-oil reversal,
            or `None` (defaults to `max_gas_saturation`).
        :param kwargs: Additional keyword arguments forwarded to the backing tables.
        :return: `CapillaryPressures` dictionary with keys `"oil_water"` and
            `"gas_oil"`.
        """
        is_scalar = np.isscalar(water_saturation)
        sw = np.atleast_1d(np.asarray(water_saturation, dtype=np.float64))
        so = np.atleast_1d(np.asarray(oil_saturation, dtype=np.float64))
        sg = np.atleast_1d(np.asarray(gas_saturation, dtype=np.float64))
        sw, so, sg = np.broadcast_arrays(sw, so, sg)

        sw_max, sg_max, sw_imb, sg_imb, sw_rev, sg_rev = self._parse_hysteresis_kwargs(
            sw,
            sg,
            max_water_saturation,
            max_gas_saturation,
            water_imbibition_flag,
            gas_imbibition_flag,
            water_reversal_saturation,
            gas_reversal_saturation,
        )

        ow_drain = self.oil_water_drainage_table
        ow_imb = self.oil_water_imbibition_table or ow_drain
        go_drain = self.gas_oil_drainage_table
        go_imb = self.gas_oil_imbibition_table or go_drain

        # Oil-water Pc: scan over water saturation
        pcow_drain = _ow_pc(ow_drain, sw, so, sg, **kwargs)
        pcow_imb = _ow_pc(ow_imb, sw, so, sg, **kwargs)
        pcow = _killough_interpolation(
            sw,
            pcow_drain,
            pcow_imb,
            sw_rev,
            sw_max,
            sw_imb,
            exponent=self.scanning_interpolation_exponent,
        )

        # Gas-oil Pc: scan over gas saturation
        pcgo_drain = _go_pc(go_drain, sw, so, sg, **kwargs)
        pcgo_imb = _go_pc(go_imb, sw, so, sg, **kwargs)
        pcgo = _killough_interpolation(
            sg,
            pcgo_drain,
            pcgo_imb,
            sg_rev,
            sg_max,
            sg_imb,
            exponent=self.scanning_interpolation_exponent,
        )

        if is_scalar:
            return CapillaryPressures(
                oil_water=float(np.atleast_1d(pcow).flat[0]),
                gas_oil=float(np.atleast_1d(pcgo).flat[0]),
            )
        return CapillaryPressures(
            oil_water=pcow,  # type: ignore[typeddict-item]
            gas_oil=pcgo,  # type: ignore[typeddict-item]
        )

    def get_capillary_pressure_derivatives(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        max_water_saturation: typing.Optional[FloatOrArray] = None,
        max_gas_saturation: typing.Optional[FloatOrArray] = None,
        water_imbibition_flag: typing.Optional[
            typing.Union[bool, npt.NDArray[np.bool_]]
        ] = None,
        gas_imbibition_flag: typing.Optional[
            typing.Union[bool, npt.NDArray[np.bool_]]
        ] = None,
        water_reversal_saturation: typing.Optional[FloatOrArray] = None,
        gas_reversal_saturation: typing.Optional[FloatOrArray] = None,
        **kwargs: typing.Any,
    ) -> CapillaryPressureDerivatives:
        """
        Compute partial derivatives of capillary pressures with Killough hysteresis.

        Returns the following non-zero entries.

        - `dPcow_dSw`: dPcow/dSw — oil-water Pc scanned over Sw.
        - `dPcow_dSo`: zero — Pcow does not depend directly on So in this model.
        - `dPcgo_dSg`: dPcgo/dSg — gas-oil Pc scanned over Sg.
        - `dPcgo_dSo`: zero — Pcgo does not depend directly on So.

        :param water_saturation: Water saturation (fraction, 0-1) — scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) — scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) — scalar or array.
        :param max_water_saturation: Historical maximum water saturation per cell,
            or `None` to disable hysteresis.
        :param max_gas_saturation: Historical maximum gas saturation per cell,
            or `None` to disable hysteresis.
        :param water_imbibition_flag: Per-cell oil-water imbibition flag or `None`.
        :param gas_imbibition_flag: Per-cell gas-oil imbibition flag or `None`.
        :param water_reversal_saturation: Water saturation at the last oil-water
            reversal, or `None` (defaults to `max_water_saturation`).
        :param gas_reversal_saturation: Gas saturation at the last gas-oil reversal,
            or `None` (defaults to `max_gas_saturation`).
        :param kwargs: Additional keyword arguments forwarded to the backing tables.
        :return: `CapillaryPressureDerivatives` dictionary containing
            `dPcow_dSw`, `dPcow_dSo`, `dPcgo_dSg`, and `dPcgo_dSo`.
        """
        is_scalar = np.isscalar(water_saturation)
        sw = np.atleast_1d(np.asarray(water_saturation, dtype=np.float64))
        so = np.atleast_1d(np.asarray(oil_saturation, dtype=np.float64))
        sg = np.atleast_1d(np.asarray(gas_saturation, dtype=np.float64))
        sw, so, sg = np.broadcast_arrays(sw, so, sg)
        zeros = np.zeros_like(sw)

        sw_max, sg_max, sw_imb, sg_imb, sw_rev, sg_rev = self._parse_hysteresis_kwargs(
            sw,
            sg,
            max_water_saturation,
            max_gas_saturation,
            water_imbibition_flag,
            gas_imbibition_flag,
            water_reversal_saturation,
            gas_reversal_saturation,
        )

        ow_drain = self.oil_water_drainage_table
        ow_imb = self.oil_water_imbibition_table or ow_drain
        go_drain = self.gas_oil_drainage_table
        go_imb = self.gas_oil_imbibition_table or go_drain

        # Oil-water
        pcow_drain = _ow_pc(ow_drain, sw, so, sg, **kwargs)
        pcow_imb = _ow_pc(ow_imb, sw, so, sg, **kwargs)
        d_pcow_drain_d_sw = _ow_pc_deriv(ow_drain, sw, so, sg, **kwargs)
        d_pcow_imb_d_sw = _ow_pc_deriv(ow_imb, sw, so, sg, **kwargs)

        d_pcow_d_sw = _killough_interpolation_derivative(
            sw,
            pcow_drain,
            pcow_imb,
            d_pcow_drain_d_sw,
            d_pcow_imb_d_sw,
            sw_rev,
            sw_max,
            sw_imb,
            exponent=self.scanning_interpolation_exponent,
        )

        # Gas-oil
        pcgo_drain = _go_pc(go_drain, sw, so, sg, **kwargs)
        pcgo_imb = _go_pc(go_imb, sw, so, sg, **kwargs)
        d_pcgo_drain_d_sg = _go_pc_deriv(go_drain, sw, so, sg, **kwargs)
        d_pcgo_imb_d_sg = _go_pc_deriv(go_imb, sw, so, sg, **kwargs)

        d_pcgo_d_sg = _killough_interpolation_derivative(
            sg,
            pcgo_drain,
            pcgo_imb,
            d_pcgo_drain_d_sg,
            d_pcgo_imb_d_sg,
            sg_rev,
            sg_max,
            sg_imb,
            exponent=self.scanning_interpolation_exponent,
        )

        if is_scalar:
            return CapillaryPressureDerivatives(
                dPcow_dSw=float(np.atleast_1d(d_pcow_d_sw).flat[0]),
                dPcow_dSo=0.0,
                dPcgo_dSg=float(np.atleast_1d(d_pcgo_d_sg).flat[0]),
                dPcgo_dSo=0.0,
            )

        return CapillaryPressureDerivatives(
            dPcow_dSw=d_pcow_d_sw,
            dPcow_dSo=zeros.copy(),
            dPcgo_dSg=d_pcgo_d_sg,
            dPcgo_dSo=zeros.copy(),
        )
