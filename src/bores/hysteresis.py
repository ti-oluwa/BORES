"""
Hysteresis models for relative permeability and capillary pressure.

Implements industry-standard Killough model with Land's trapping for relative
permeability hysteresis and Killough-type capillary pressure hysteresis with
scanning curve interpolation.

References:
- Killough, J.E. (1976). "Reservoir Simulation With History-Dependent Saturation Functions"
- Land, C.S. (1968). "Calculation of Imbibition Relative Permeability for Two- and Three-Phase Flow from Rock Properties"
- Carlson, F.M. (2006). "Simulation of Relative Permeability Hysteresis to the Nonwetting Phase"
"""

import typing

import attrs
import numpy as np
import numpy.typing as npt

from bores.capillary_pressures import (
    CapillaryPressureTable,
    TwoPhaseCapillaryPressureTable,
    capillary_pressure_table,
)
from bores.constants import c
from bores.errors import ValidationError
from bores.relperm import (
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

__all__ = [
    "KilloughCapillaryPressureModel",
    "KilloughLandRelPermModel",
]


def _compute_land_trapped_saturation(
    current_saturation: FloatOrArray,
    max_saturation: FloatOrArray,
    residual_saturation_max: float,
    land_coefficient: float,
    saturation_epsilon: float = 1e-12,
) -> FloatOrArray:
    """
    Compute trapped (residual) saturation using Land's trapping model.

    Land's model (1968) relates the residual non-wetting phase saturation to
    the initial saturation at the start of imbibition:

        S_nw_residual = S_nw_residual_max / (1 + C * S_nw_initial)

    where:
    - S_nw_residual_max: maximum residual saturation (endpoint from drainage curve)
    - S_nw_initial: initial non-wetting saturation when imbibition started
    - C: Land trapping coefficient (fitted parameter, typically 0.5 to 5.0)

    Higher C → more trapping (smaller residual).
    Lower C → less trapping (larger residual).
    C = 0 → no dynamic trapping (residual = max).

    The model captures that more initial oil leads to more trapping when water
    invades, which is physically consistent with snap-off and pore-scale
    displacement mechanics.

    :param current_saturation: Current non-wetting saturation (scalar or array).
    :param max_saturation: Historical maximum non-wetting saturation (scalar or array).
        This represents the initial saturation when imbibition started.
    :param residual_saturation_max: Maximum residual saturation from drainage
        curve endpoint.
    :param land_coefficient: Land trapping coefficient C (positive).
    :param saturation_epsilon: Small epsilon for numerical stability.
    :return: Trapped saturation computed via Land's model.
    """
    # Use max_saturation as the "initial saturation" when imbibition started
    # Protect against division by zero when residual is very small
    safe_residual_max = max(residual_saturation_max, saturation_epsilon)

    if land_coefficient <= 0.0:
        # No dynamic trapping, return constant residual
        return np.full_like(current_saturation, safe_residual_max)

    # Land's formula: Sor = Sor_max / (1 + C * So_initial)
    # Clamp max_saturation to avoid division by zero in denominator
    safe_max_sat = np.maximum(max_saturation, saturation_epsilon)
    trapped = safe_residual_max / (1.0 + land_coefficient * safe_max_sat)

    # Ensure trapped saturation doesn't exceed current saturation
    # (can't trap more than what exists)
    trapped = np.minimum(trapped, current_saturation)
    return trapped


def _killough_scanning_curve_interpolation(
    saturation: FloatOrArray,
    drainage_value: FloatOrArray,
    imbibition_value: FloatOrArray,
    reversal_saturation: FloatOrArray,
    max_saturation: FloatOrArray,
    is_imbibition: FloatOrArray,
    interpolation_exponent: float = 1.0,
    saturation_epsilon: float = 1e-12,
) -> FloatOrArray:
    """
    Killough scanning curve interpolation between primary drainage and imbibition curves.

    When saturation reverses direction (e.g., drainage → imbibition → drainage again),
    the property (relative permeability or capillary pressure) follows a scanning curve
    that interpolates between the primary drainage and imbibition bounds.

    Killough's interpolation formula:
        value_scan = value_drain + (value_imb - value_drain) * f(S)

    where f(S) is a saturation-dependent interpolation factor:
        f(S) = ((S - S_reversal) / (S_max - S_reversal))^n

    with:
    - S_reversal: saturation at which direction last reversed
    - S_max: maximum saturation reached before current reversal
    - n: interpolation exponent (typically 1.0 for linear, >1 for sharper transition)

    :param saturation: Current saturation (scalar or array).
    :param drainage_value: Value from primary drainage curve.
    :param imbibition_value: Value from primary imbibition curve.
    :param reversal_saturation: Saturation at which flow direction last reversed.
    :param max_saturation: Maximum saturation reached (for normalization).
    :param is_imbibition: Boolean/float indicating if currently on imbibition (1.0) or drainage (0.0).
    :param interpolation_exponent: Killough interpolation exponent n.
    :param saturation_epsilon: Numerical stability epsilon.
    :return: Interpolated value on the scanning curve.
    """
    is_scalar = np.isscalar(saturation)
    sat = np.atleast_1d(saturation)
    val_drain = np.atleast_1d(drainage_value)
    val_imb = np.atleast_1d(imbibition_value)
    s_rev = np.atleast_1d(reversal_saturation)
    s_max = np.atleast_1d(max_saturation)
    imb_flag = np.atleast_1d(is_imbibition)

    sat, val_drain, val_imb, s_rev, s_max, imb_flag = np.broadcast_arrays(
        sat, val_drain, val_imb, s_rev, s_max, imb_flag
    )

    # Compute interpolation factor
    # f = ((S - S_reversal) / (S_max - S_reversal))^n
    delta_s = sat - s_rev
    delta_s_max = s_max - s_rev

    # Avoid division by zero
    delta_s_max_safe = np.where(
        np.abs(delta_s_max) > saturation_epsilon, delta_s_max, 1.0
    )

    # Interpolation factor (clamped to [0, 1])
    interp_factor = np.clip(
        (delta_s / delta_s_max_safe) ** interpolation_exponent,
        0.0,
        1.0,
    )

    # Linear blend between drainage and imbibition
    val_scan = val_drain + (val_imb - val_drain) * interp_factor

    # Primary curves: use drainage for drainage, imbibition for imbibition
    # Scanning curve for secondary reversals (when not at extremes)
    on_primary_drainage = (imb_flag < 0.5) & (np.abs(sat - s_max) < saturation_epsilon)
    on_primary_imbibition = (imb_flag > 0.5) & (
        np.abs(sat - s_rev) < saturation_epsilon
    )

    val_result = np.where(
        on_primary_drainage,
        val_drain,
        np.where(on_primary_imbibition, val_imb, val_scan),
    )

    if is_scalar:
        return float(val_result[0])
    return val_result


def _compute_scanning_curve_derivatives(
    saturation: FloatOrArray,
    d_val_drain_d_s: FloatOrArray,
    d_val_imb_d_s: FloatOrArray,
    drainage_value: FloatOrArray,
    imbibition_value: FloatOrArray,
    reversal_saturation: FloatOrArray,
    max_saturation: FloatOrArray,
    is_imbibition: FloatOrArray,
    interpolation_exponent: float = 1.0,
    saturation_epsilon: float = 1e-12,
) -> FloatOrArray:
    """
    Analytical derivative of Killough scanning curve with respect to saturation.

    The scanning curve is:
        val_scan(S) = val_drain(S) + [val_imb(S) - val_drain(S)] * f(S)

    where f(S) = ((S - S_rev) / (S_max - S_rev))^n

    The derivative is:
        d(val_scan)/dS = d(val_drain)/dS + [d(val_imb)/dS - d(val_drain)/dS] * f(S)
                        + [val_imb - val_drain] * df/dS

    where:
        df/dS = n * (S - S_rev)^(n-1) / (S_max - S_rev)^n

    :param saturation: Current saturation.
    :param d_val_drain_d_s: Derivative of drainage curve value w.r.t. saturation.
    :param d_val_imb_d_s: Derivative of imbibition curve value w.r.t. saturation.
    :param drainage_value: Drainage curve value.
    :param imbibition_value: Imbibition curve value.
    :param reversal_saturation: Saturation at reversal point.
    :param max_saturation: Maximum saturation reached.
    :param is_imbibition: Imbibition flag.
    :param interpolation_exponent: Killough exponent n.
    :param saturation_epsilon: Numerical epsilon.
    :return: Derivative of scanning curve w.r.t. saturation.
    """
    is_scalar = np.isscalar(saturation)
    sat = np.atleast_1d(saturation)
    s_rev = np.atleast_1d(reversal_saturation)
    s_max = np.atleast_1d(max_saturation)
    imb_flag = np.atleast_1d(is_imbibition)

    sat, s_rev, s_max, imb_flag = np.broadcast_arrays(sat, s_rev, s_max, imb_flag)

    # Compute interpolation factor λ
    delta_s = sat - s_rev
    delta_s_max = s_max - s_rev
    delta_s_max_safe = np.where(
        np.abs(delta_s_max) > saturation_epsilon, delta_s_max, 1.0
    )

    lambda_interp = np.clip(
        (delta_s / delta_s_max_safe) ** interpolation_exponent, 0.0, 1.0
    )

    # Blend derivatives: d(val_scan)/dS = d(drain)/dS + [d(imb)/dS - d(drain)/dS] * λ
    d_val_scan = d_val_drain_d_s + (d_val_imb_d_s - d_val_drain_d_s) * lambda_interp

    # Add contribution from d(λ)/dS * [val_imb - val_drain]
    if abs(interpolation_exponent - 1.0) > 1e-10:
        # d(λ)/dS = n * (S - S_rev)^(n-1) / (S_max - S_rev)^n
        d_lambda_d_s = np.where(
            np.abs(delta_s_max) > saturation_epsilon,
            interpolation_exponent
            * np.power(
                np.maximum(np.abs(delta_s), saturation_epsilon),
                interpolation_exponent - 1.0,
            )
            * np.sign(delta_s)
            / np.power(np.abs(delta_s_max_safe), interpolation_exponent),
            0.0,
        )
        val_diff = imbibition_value - drainage_value
        d_val_scan += val_diff * d_lambda_d_s

    # On primary curves, use their derivatives directly
    on_primary_drainage = (imb_flag < 0.5) & (np.abs(sat - s_max) < saturation_epsilon)
    on_primary_imbibition = (imb_flag > 0.5) & (
        np.abs(sat - s_rev) < saturation_epsilon
    )

    d_val_result = np.where(
        on_primary_drainage,
        d_val_drain_d_s,
        np.where(on_primary_imbibition, d_val_imb_d_s, d_val_scan),
    )

    if is_scalar:
        return float(d_val_result[0])
    return d_val_result


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
    Killough relative permeability hysteresis model with Land's trapping.

    Implements:
    - Primary drainage curves (initial displacement)
    - Primary imbibition curves (reversal displacement)
    - Scanning curves (secondary reversals) via Killough interpolation
    - Dynamic residual saturation via Land's trapping model

    The model uses separate two-phase relative permeability tables (or models)
    for drainage and imbibition. During simulation, it selects the appropriate
    curve based on saturation history and interpolates scanning curves when
    saturation direction reverses.

    **Drainage**: Non-wetting phase displacing wetting phase (e.g., oil into water)
    **Imbibition**: Wetting phase displacing non-wetting phase (e.g., water into oil)
    **Scanning**: Secondary displacement after a reversal

    Requires saturation history tracking via kwargs:
    - `max_water_saturation`: Historical maximum (indicates when water started receding)
    - `max_gas_saturation`: Historical maximum (indicates when gas started receding)
    - `water_imbibition_flag`: True if water currently advancing (imbibition)
    - `gas_imbibition_flag`: True if gas currently advancing (imbibition)
    - `water_reversal_saturation`: Saturation when water last reversed direction
    - `gas_reversal_saturation`: Saturation when gas last reversed direction

    References:
    - Killough, J.E. (1976). SPE 5106
    - Land, C.S. (1968). SPE 1942
    """

    __type__ = "killough_land_relperm_model"

    # Primary drainage curves (initial displacement)
    oil_water_drainage_table: typing.Union[
        TwoPhaseRelPermTable, RelativePermeabilityTable
    ]
    """Relative permeability for oil-water system during primary drainage."""

    gas_oil_drainage_table: typing.Union[
        TwoPhaseRelPermTable, RelativePermeabilityTable
    ]
    """Relative permeability for gas-oil system during primary drainage."""

    # Primary imbibition curves (reversal displacement)
    oil_water_imbibition_table: typing.Optional[
        typing.Union[TwoPhaseRelPermTable, RelativePermeabilityTable]
    ] = None
    """Relative permeability for oil-water system during primary imbibition."""

    gas_oil_imbibition_table: typing.Optional[
        typing.Union[TwoPhaseRelPermTable, RelativePermeabilityTable]
    ] = None
    """Relative permeability for gas-oil system during primary imbibition."""

    land_coefficient_water: float = 1.0
    """Land trapping coefficient C for water→oil imbibition (typical: 0.5-5.0)."""

    land_coefficient_gas: float = 1.0
    """Land trapping coefficient C for gas→oil imbibition (typical: 0.5-5.0)."""

    scanning_interpolation_exponent: float = 1.0
    """Killough scanning curve interpolation exponent n (1.0 = linear)."""

    residual_oil_saturation_max_water: typing.Optional[float] = None
    """Maximum residual oil saturation for Land model (drainage endpoint)."""

    residual_oil_saturation_max_gas: typing.Optional[float] = None
    """Maximum residual oil saturation in gas-oil system."""

    residual_gas_saturation_max: typing.Optional[float] = None
    """Maximum residual gas saturation."""

    mixing_rule: typing.Union[MixingRule, str] = "eclipse_rule"
    """Three-phase oil relative permeability mixing rule."""

    supports_arrays: bool = attrs.field(init=False, repr=False, default=True)
    """Flag indicating support for array inputs."""

    def __attrs_post_init__(self) -> None:
        """Validate configuration."""
        # Validate drainage tables
        if isinstance(self.oil_water_drainage_table, TwoPhaseRelPermTable) and {
            self.oil_water_drainage_table.wetting_phase,
            self.oil_water_drainage_table.non_wetting_phase,
        } != {FluidPhase.WATER, FluidPhase.OIL}:
            raise ValidationError(
                "`oil_water_drainage_table` must be between water and oil phases."
            )

        if isinstance(self.gas_oil_drainage_table, TwoPhaseRelPermTable) and {
            self.gas_oil_drainage_table.wetting_phase,
            self.gas_oil_drainage_table.non_wetting_phase,
        } != {FluidPhase.OIL, FluidPhase.GAS}:
            raise ValidationError(
                "`gas_oil_drainage_table` must be between oil and gas phases."
            )

        # Validate imbibition tables if provided
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
                "`oil_water_imbibition_table` must be between water and oil phases."
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
                "`gas_oil_imbibition_table` must be between oil and gas phases."
            )

        mixing_rule = self.mixing_rule
        if isinstance(mixing_rule, str):
            object.__setattr__(self, "mixing_rule", get_mixing_rule(mixing_rule))

    def _get_two_phase_relperm(
        self,
        table: typing.Union[TwoPhaseRelPermTable, RelativePermeabilityTable],
        wetting_saturation: FloatOrArray,
        non_wetting_saturation: FloatOrArray,
        oil_saturation: typing.Optional[FloatOrArray] = None,
        gas_saturation: typing.Optional[FloatOrArray] = None,
        **kwargs: typing.Any,
    ) -> typing.Tuple[FloatOrArray, FloatOrArray]:
        """Extract two-phase kr from either TwoPhaseRelPermTable or three-phase model."""
        if isinstance(table, TwoPhaseRelPermTable):
            return table.get_relative_permeabilities(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )

        # Three-phase model
        if oil_saturation is None or gas_saturation is None:
            raise ValidationError(
                "Three-phase table requires oil_saturation and gas_saturation."
            )

        water_sat = wetting_saturation
        is_gas_oil_system = np.all(water_sat < 1e-6)
        relperm = table.get_relative_permeabilities(
            water_saturation=water_sat,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )

        if is_gas_oil_system:
            return relperm["oil"], relperm["gas"]
        return relperm["water"], relperm["oil"]

    def _get_two_phase_relperm_derivatives(
        self,
        table: typing.Union[TwoPhaseRelPermTable, RelativePermeabilityTable],
        wetting_saturation: FloatOrArray,
        non_wetting_saturation: FloatOrArray,
        oil_saturation: typing.Optional[FloatOrArray] = None,
        gas_saturation: typing.Optional[FloatOrArray] = None,
        **kwargs: typing.Any,
    ) -> typing.Tuple[FloatOrArray, FloatOrArray]:
        """Extract two-phase kr derivatives."""
        if isinstance(table, TwoPhaseRelPermTable):
            d_krw = table.get_wetting_phase_relative_permeability_derivative(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
            d_krnw = table.get_non_wetting_phase_relative_permeability_derivative(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
            return d_krw, d_krnw

        if oil_saturation is None or gas_saturation is None:
            raise ValidationError(
                "Three-phase table requires oil_saturation and gas_saturation."
            )

        water_sat = wetting_saturation
        is_gas_oil_system = np.all(water_sat < 1e-6)
        derivs = table.get_relative_permeability_derivatives(
            water_saturation=water_sat,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )

        if is_gas_oil_system:
            return derivs["dKro_dSo"], derivs["dKrg_dSo"]
        return derivs["dKrw_dSw"], derivs["dKro_dSw"]

    def get_relative_permeabilities(  # type: ignore[override]
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
        Compute three-phase relative permeabilities with hysteresis.

        **Required kwargs for hysteresis**:
        - `max_water_saturation`: Historical maximum water saturation
        - `max_gas_saturation`: Historical maximum gas saturation
        - `water_imbibition_flag`: Boolean indicating water imbibition
        - `gas_imbibition_flag`: Boolean indicating gas imbibition
        - `water_reversal_saturation`: Saturation when water last reversed
        - `gas_reversal_saturation`: Saturation when gas last reversed

        :param water_saturation: Current water saturation.
        :param oil_saturation: Current oil saturation.
        :param gas_saturation: Current gas saturation.
        :return: RelativePermeabilities with krw, kro, krg.
        """
        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
        )
        sw = np.atleast_1d(water_saturation)
        so = np.atleast_1d(oil_saturation)
        sg = np.atleast_1d(gas_saturation)
        sw, so, sg = np.broadcast_arrays(sw, so, sg)

        # Normalize saturations
        total_sat = sw + so + sg
        needs_norm = (np.abs(total_sat - 1.0) > c.SATURATION_EPSILON) & (
            total_sat > 0.0
        )
        if np.any(needs_norm):
            sw = np.where(needs_norm, sw / total_sat, sw)
            so = np.where(needs_norm, so / total_sat, so)
            sg = np.where(needs_norm, sg / total_sat, sg)

        use_hysteresis = (
            max_water_saturation is not None
            and max_gas_saturation is not None
            and water_imbibition_flag is not None
            and gas_imbibition_flag is not None
        )
        if use_hysteresis:
            sw_max = np.atleast_1d(max_water_saturation)  # type: ignore
            sg_max = np.atleast_1d(max_gas_saturation)  # type: ignore
            water_imb_flag = np.atleast_1d(water_imbibition_flag)  # type: ignore
            gas_imb_flag = np.atleast_1d(gas_imbibition_flag)  # type: ignore

            sw_rev = (
                np.atleast_1d(water_reversal_saturation)
                if water_reversal_saturation is not None
                else sw_max
            )
            sg_rev = (
                np.atleast_1d(gas_reversal_saturation)
                if gas_reversal_saturation is not None
                else sg_max
            )

            sw_max, sg_max, water_imb_flag, gas_imb_flag, sw_rev, sg_rev = (
                np.broadcast_arrays(
                    sw_max, sg_max, water_imb_flag, gas_imb_flag, sw_rev, sg_rev, sw
                )[:6]
            )
        else:
            # No hysteresis: use drainage only
            sw_max = sw
            sg_max = sg
            sw_rev = sw
            sg_rev = sg
            water_imb_flag = np.zeros_like(sw)
            gas_imb_flag = np.zeros_like(sg)

        # Oil-Water system
        ow_table_drain = self.oil_water_drainage_table
        ow_table_imb = self.oil_water_imbibition_table or ow_table_drain

        if isinstance(ow_table_drain, TwoPhaseRelPermTable):
            water_is_wetting = ow_table_drain.wetting_phase == FluidPhase.WATER
        else:
            water_is_wetting = True

        # Drainage kr
        if water_is_wetting:
            krw_drain, kro_w_drain = self._get_two_phase_relperm(
                ow_table_drain, sw, so, so, sg, **kwargs
            )
        else:
            kro_w_drain, krw_drain = self._get_two_phase_relperm(
                ow_table_drain, so, sw, so, sg, **kwargs
            )

        # Imbibition kr
        if water_is_wetting:
            krw_imb, kro_w_imb = self._get_two_phase_relperm(
                ow_table_imb, sw, so, so, sg, **kwargs
            )
        else:
            kro_w_imb, krw_imb = self._get_two_phase_relperm(
                ow_table_imb, so, sw, so, sg, **kwargs
            )

        # Apply scanning curve interpolation
        krw = _killough_scanning_curve_interpolation(
            saturation=sw,
            drainage_value=krw_drain,
            imbibition_value=krw_imb,
            reversal_saturation=sw_rev,
            max_saturation=sw_max,
            is_imbibition=water_imb_flag,
            interpolation_exponent=self.scanning_interpolation_exponent,
        )
        kro_w = _killough_scanning_curve_interpolation(
            saturation=sw,
            drainage_value=kro_w_drain,
            imbibition_value=kro_w_imb,
            reversal_saturation=sw_rev,
            max_saturation=sw_max,
            is_imbibition=water_imb_flag,
            interpolation_exponent=self.scanning_interpolation_exponent,
        )

        # Gas-Oil system
        go_table_drain = self.gas_oil_drainage_table
        go_table_imb = self.gas_oil_imbibition_table or go_table_drain

        if isinstance(go_table_drain, TwoPhaseRelPermTable):
            oil_is_wetting = go_table_drain.wetting_phase == FluidPhase.OIL
        else:
            oil_is_wetting = True

        # Drainage kr
        if oil_is_wetting:
            kro_g_drain, krg_drain = self._get_two_phase_relperm(
                go_table_drain, so, sg, so, sg, **kwargs
            )
        else:
            krg_drain, kro_g_drain = self._get_two_phase_relperm(
                go_table_drain, sg, so, so, sg, **kwargs
            )

        # Imbibition kr
        if oil_is_wetting:
            kro_g_imb, krg_imb = self._get_two_phase_relperm(
                go_table_imb, so, sg, so, sg, **kwargs
            )
        else:
            krg_imb, kro_g_imb = self._get_two_phase_relperm(
                go_table_imb, sg, so, so, sg, **kwargs
            )

        # Apply scanning curve interpolation
        krg = _killough_scanning_curve_interpolation(
            saturation=sg,
            drainage_value=krg_drain,
            imbibition_value=krg_imb,
            reversal_saturation=sg_rev,
            max_saturation=sg_max,
            is_imbibition=gas_imb_flag,
            interpolation_exponent=self.scanning_interpolation_exponent,
        )
        kro_g = _killough_scanning_curve_interpolation(
            saturation=sg,
            drainage_value=kro_g_drain,
            imbibition_value=kro_g_imb,
            reversal_saturation=sg_rev,
            max_saturation=sg_max,
            is_imbibition=gas_imb_flag,
            interpolation_exponent=self.scanning_interpolation_exponent,
        )

        # Three-phase oil kr via mixing rule
        kro = self.mixing_rule(  # type: ignore[operator]
            kro_w=kro_w,
            kro_g=kro_g,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
        )

        if is_scalar:
            return RelativePermeabilities(
                water=krw.item(),  # type: ignore
                oil=kro.item(),  # type: ignore
                gas=krg.item(),  # type: ignore
            )

        return RelativePermeabilities(water=krw, oil=kro, gas=krg)  # type: ignore[typeddict-item]

    def get_relative_permeability_derivatives(  # type: ignore[override]
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
        Compute partial derivatives of three-phase relative permeabilities with hysteresis.

        :param water_saturation: Current water saturation.
        :param oil_saturation: Current oil saturation.
        :param gas_saturation: Current gas saturation.
        :return: RelativePermeabilityDerivatives dictionary.
        """
        is_scalar = np.isscalar(water_saturation)
        sw = np.atleast_1d(water_saturation)
        so = np.atleast_1d(oil_saturation)
        sg = np.atleast_1d(gas_saturation)
        sw, so, sg = np.broadcast_arrays(sw, so, sg)
        zeros = np.zeros_like(sw)

        use_hysteresis = (
            max_water_saturation is not None
            and max_gas_saturation is not None
            and water_imbibition_flag is not None
            and gas_imbibition_flag is not None
        )
        if use_hysteresis:
            sw_max = np.atleast_1d(max_water_saturation)  # type: ignore
            sg_max = np.atleast_1d(max_gas_saturation)  # type: ignore
            water_imb_flag = np.atleast_1d(water_imbibition_flag)  # type: ignore
            gas_imb_flag = np.atleast_1d(gas_imbibition_flag)  # type: ignore

            sw_rev = (
                np.atleast_1d(water_reversal_saturation)
                if water_reversal_saturation is not None
                else sw_max
            )
            sg_rev = (
                np.atleast_1d(gas_reversal_saturation)
                if gas_reversal_saturation is not None
                else sg_max
            )

            sw_max, sg_max, water_imb_flag, gas_imb_flag, sw_rev, sg_rev = (
                np.broadcast_arrays(
                    sw_max, sg_max, water_imb_flag, gas_imb_flag, sw_rev, sg_rev, sw
                )[:6]
            )
        else:
            sw_max = sw
            sg_max = sg
            sw_rev = sw
            sg_rev = sg
            water_imb_flag = np.zeros_like(sw)
            gas_imb_flag = np.zeros_like(sg)

        # Oil-Water derivatives
        ow_table_drain = self.oil_water_drainage_table
        ow_table_imb = self.oil_water_imbibition_table or ow_table_drain

        if isinstance(ow_table_drain, TwoPhaseRelPermTable):
            water_is_wetting = ow_table_drain.wetting_phase == FluidPhase.WATER
        else:
            water_is_wetting = True

        # Get kr values for derivative computation
        if water_is_wetting:
            krw_drain, kro_w_drain = self._get_two_phase_relperm(
                ow_table_drain, sw, so, so, sg, **kwargs
            )
            krw_imb, kro_w_imb = self._get_two_phase_relperm(
                ow_table_imb, sw, so, so, sg, **kwargs
            )
            d_krw_drain, d_kro_w_drain = self._get_two_phase_relperm_derivatives(
                ow_table_drain, sw, so, so, sg, **kwargs
            )
            d_krw_imb, d_kro_w_imb = self._get_two_phase_relperm_derivatives(
                ow_table_imb, sw, so, so, sg, **kwargs
            )
        else:
            kro_w_drain, krw_drain = self._get_two_phase_relperm(
                ow_table_drain, so, sw, so, sg, **kwargs
            )
            kro_w_imb, krw_imb = self._get_two_phase_relperm(
                ow_table_imb, so, sw, so, sg, **kwargs
            )
            d_kro_w_drain, d_krw_drain = self._get_two_phase_relperm_derivatives(
                ow_table_drain, so, sw, so, sg, **kwargs
            )
            d_kro_w_imb, d_krw_imb = self._get_two_phase_relperm_derivatives(
                ow_table_imb, so, sw, so, sg, **kwargs
            )

        # Scanning curve derivatives
        d_krw_d_sw = _compute_scanning_curve_derivatives(
            saturation=sw,
            d_val_drain_d_s=d_krw_drain,
            d_val_imb_d_s=d_krw_imb,
            drainage_value=krw_drain,
            imbibition_value=krw_imb,
            reversal_saturation=sw_rev,
            max_saturation=sw_max,
            is_imbibition=water_imb_flag,
            interpolation_exponent=self.scanning_interpolation_exponent,
        )

        d_kro_w_d_sw = _compute_scanning_curve_derivatives(
            saturation=sw,
            d_val_drain_d_s=d_kro_w_drain,
            d_val_imb_d_s=d_kro_w_imb,
            drainage_value=kro_w_drain,
            imbibition_value=kro_w_imb,
            reversal_saturation=sw_rev,
            max_saturation=sw_max,
            is_imbibition=water_imb_flag,
            interpolation_exponent=self.scanning_interpolation_exponent,
        )

        d_krw_d_so = zeros.copy()
        d_krw_d_sg = zeros.copy()

        # Gas-Oil derivatives
        go_table_drain = self.gas_oil_drainage_table
        go_table_imb = self.gas_oil_imbibition_table or go_table_drain

        if isinstance(go_table_drain, TwoPhaseRelPermTable):
            oil_is_wetting = go_table_drain.wetting_phase == FluidPhase.OIL
        else:
            oil_is_wetting = True

        if oil_is_wetting:
            kro_g_drain, krg_drain = self._get_two_phase_relperm(
                go_table_drain, so, sg, so, sg, **kwargs
            )
            kro_g_imb, krg_imb = self._get_two_phase_relperm(
                go_table_imb, so, sg, so, sg, **kwargs
            )
            d_kro_g_drain, d_krg_drain = self._get_two_phase_relperm_derivatives(
                go_table_drain, so, sg, so, sg, **kwargs
            )
            d_kro_g_imb, d_krg_imb = self._get_two_phase_relperm_derivatives(
                go_table_imb, so, sg, so, sg, **kwargs
            )
        else:
            krg_drain, kro_g_drain = self._get_two_phase_relperm(
                go_table_drain, sg, so, so, sg, **kwargs
            )
            krg_imb, kro_g_imb = self._get_two_phase_relperm(
                go_table_imb, sg, so, so, sg, **kwargs
            )
            d_krg_drain, d_kro_g_drain = self._get_two_phase_relperm_derivatives(
                go_table_drain, sg, so, so, sg, **kwargs
            )
            d_krg_imb, d_kro_g_imb = self._get_two_phase_relperm_derivatives(
                go_table_imb, sg, so, so, sg, **kwargs
            )

        d_krg_d_sg = _compute_scanning_curve_derivatives(
            saturation=sg,
            d_val_drain_d_s=d_krg_drain,
            d_val_imb_d_s=d_krg_imb,
            drainage_value=krg_drain,
            imbibition_value=krg_imb,
            reversal_saturation=sg_rev,
            max_saturation=sg_max,
            is_imbibition=gas_imb_flag,
            interpolation_exponent=self.scanning_interpolation_exponent,
        )

        d_kro_g_d_so = _compute_scanning_curve_derivatives(
            saturation=so,
            d_val_drain_d_s=d_kro_g_drain,
            d_val_imb_d_s=d_kro_g_imb,
            drainage_value=kro_g_drain,
            imbibition_value=kro_g_imb,
            reversal_saturation=1.0 - sg_rev - sw_rev,  # so_reversal
            max_saturation=1.0 - sg_max - sw_max,  # so_max
            is_imbibition=gas_imb_flag,
            interpolation_exponent=self.scanning_interpolation_exponent,
        )

        d_krg_d_sw = zeros.copy()
        d_krg_d_so = zeros.copy()

        # Three-phase oil kr derivatives
        # Forward evaluation for mixing rule
        kro_w = _killough_scanning_curve_interpolation(
            sw,
            kro_w_drain,
            kro_w_imb,
            sw_rev,
            sw_max,
            water_imb_flag,
            self.scanning_interpolation_exponent,
        )
        kro_g = _killough_scanning_curve_interpolation(
            sg,
            kro_g_drain,
            kro_g_imb,
            sg_rev,
            sg_max,
            gas_imb_flag,
            self.scanning_interpolation_exponent,
        )

        # Mixing rule partial derivatives
        derivatives = get_mixing_rule_partial_derivatives(
            rule=self.mixing_rule,  # type: ignore[arg-type]
            kro_w=kro_w,
            kro_g=kro_g,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            epsilon=c.FINITE_DIFFERENCE_EPSILON,
        )
        d_kro_d_kro_w = derivatives["d_kro_d_kro_w"]
        d_kro_d_kro_g = derivatives["d_kro_d_kro_g"]
        d_kro_d_sw_explicit = derivatives["d_kro_d_sw_explicit"]
        d_kro_d_so_explicit = derivatives["d_kro_d_so_explicit"]
        d_kro_d_sg_explicit = derivatives["d_kro_d_sg_explicit"]

        # Chain rule
        d_kro_d_sw = d_kro_d_kro_w * d_kro_w_d_sw + d_kro_d_sw_explicit
        d_kro_d_so = d_kro_d_kro_g * d_kro_g_d_so + d_kro_d_so_explicit
        d_kro_d_sg = d_kro_d_sg_explicit

        if is_scalar:
            return RelativePermeabilityDerivatives(
                dKrw_dSw=d_krw_d_sw.item(),  # type: ignore
                dKro_dSw=d_kro_d_sw.item(),  # type: ignore
                dKrg_dSw=d_krg_d_sw.item(),
                dKrw_dSo=d_krw_d_so.item(),
                dKro_dSo=d_kro_d_so.item(),  # type: ignore
                dKrg_dSo=d_krg_d_so.item(),
                dKrw_dSg=d_krw_d_sg.item(),
                dKro_dSg=d_kro_d_sg.item(),  # type: ignore
                dKrg_dSg=d_krg_d_sg.item(),  # type: ignore
            )

        return RelativePermeabilityDerivatives(
            dKrw_dSw=d_krw_d_sw,
            dKro_dSw=d_kro_d_sw,
            dKrg_dSw=d_krg_d_sw,
            dKrw_dSo=d_krw_d_so,
            dKro_dSo=d_kro_d_so,
            dKrg_dSo=d_krg_d_so,
            dKrw_dSg=d_krw_d_sg,
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

    Implements:
    - Primary drainage capillary pressure curves
    - Primary imbibition capillary pressure curves
    - Scanning curves via Killough interpolation

    Unlike relative permeability hysteresis, capillary pressure hysteresis
    does not typically use Land's trapping model. Instead, it uses scanning
    curve interpolation between drainage and imbibition bounds.

    **Physical basis**:
    - Drainage: Non-wetting phase displaces wetting phase → Pc increases
    - Imbibition: Wetting phase displaces non-wetting phase → Pc decreases
    - Scanning: Intermediate reversals follow interpolated paths

    Requires saturation history tracking via kwargs:
    - `max_water_saturation`: Historical maximum water saturation
    - `max_gas_saturation`: Historical maximum gas saturation
    - `water_imbibition_flag`: True if water currently advancing
    - `gas_imbibition_flag`: True if gas currently advancing
    - `water_reversal_saturation`: Saturation when water last reversed
    - `gas_reversal_saturation`: Saturation when gas last reversed

    Reference:
    - Killough, J.E. (1976). SPE 5106
    """

    __type__ = "killough_capillary_pressure_model"

    oil_water_drainage_table: typing.Union[
        TwoPhaseCapillaryPressureTable, CapillaryPressureTable
    ]
    """Capillary pressure for oil-water system during primary drainage."""

    gas_oil_drainage_table: typing.Union[
        TwoPhaseCapillaryPressureTable, CapillaryPressureTable
    ]
    """Capillary pressure for gas-oil system during primary drainage."""

    oil_water_imbibition_table: typing.Optional[
        typing.Union[TwoPhaseCapillaryPressureTable, CapillaryPressureTable]
    ] = None
    """Capillary pressure for oil-water system during primary imbibition."""

    gas_oil_imbibition_table: typing.Optional[
        typing.Union[TwoPhaseCapillaryPressureTable, CapillaryPressureTable]
    ] = None
    """Capillary pressure for gas-oil system during primary imbibition."""

    scanning_interpolation_exponent: float = 1.0
    """Killough scanning curve interpolation exponent (1.0 = linear)."""

    supports_arrays: bool = attrs.field(init=False, repr=False, default=True)
    """Flag indicating support for array inputs."""

    def __attrs_post_init__(self) -> None:
        """Validate configuration."""
        # Validate drainage tables
        if isinstance(
            self.oil_water_drainage_table, TwoPhaseCapillaryPressureTable
        ) and {
            self.oil_water_drainage_table.wetting_phase,
            self.oil_water_drainage_table.non_wetting_phase,
        } != {FluidPhase.WATER, FluidPhase.OIL}:
            raise ValidationError(
                "`oil_water_drainage_table` must be between water and oil phases."
            )

        if isinstance(self.gas_oil_drainage_table, TwoPhaseCapillaryPressureTable) and {
            self.gas_oil_drainage_table.wetting_phase,
            self.gas_oil_drainage_table.non_wetting_phase,
        } != {FluidPhase.OIL, FluidPhase.GAS}:
            raise ValidationError(
                "`gas_oil_drainage_table` must be between oil and gas phases."
            )

        # Validate imbibition tables if provided
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
                "`oil_water_imbibition_table` must be between water and oil phases."
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
                "`gas_oil_imbibition_table` must be between oil and gas phases."
            )

    def _get_two_phase_pc(
        self,
        table: typing.Union[TwoPhaseCapillaryPressureTable, CapillaryPressureTable],
        wetting_saturation: FloatOrArray,
        non_wetting_saturation: FloatOrArray,
        oil_saturation: typing.Optional[FloatOrArray] = None,
        gas_saturation: typing.Optional[FloatOrArray] = None,
        **kwargs: typing.Any,
    ) -> FloatOrArray:
        """Extract two-phase Pc from either two-phase or three-phase model."""
        if isinstance(table, TwoPhaseCapillaryPressureTable):
            return table.get_capillary_pressure(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )

        # Three-phase model
        if oil_saturation is None or gas_saturation is None:
            raise ValidationError(
                "Three-phase table requires oil_saturation and gas_saturation."
            )

        water_sat = wetting_saturation
        is_gas_oil_system = np.all(water_sat < 1e-6)

        pc_dict = table.get_capillary_pressures(
            water_saturation=water_sat,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )

        if is_gas_oil_system:
            return pc_dict["gas_oil"]
        return pc_dict["oil_water"]

    def _get_two_phase_pc_derivative(
        self,
        table: typing.Union[TwoPhaseCapillaryPressureTable, CapillaryPressureTable],
        wetting_saturation: FloatOrArray,
        non_wetting_saturation: FloatOrArray,
        oil_saturation: typing.Optional[FloatOrArray] = None,
        gas_saturation: typing.Optional[FloatOrArray] = None,
        **kwargs: typing.Any,
    ) -> FloatOrArray:
        """Extract two-phase Pc derivative."""
        if isinstance(table, TwoPhaseCapillaryPressureTable):
            return table.get_capillary_pressure_derivative(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )

        if oil_saturation is None or gas_saturation is None:
            raise ValidationError(
                "Three-phase table requires oil_saturation and gas_saturation."
            )

        water_sat = wetting_saturation
        is_gas_oil_system = np.all(water_sat < 1e-6)

        derivs = table.get_capillary_pressure_derivatives(
            water_saturation=water_sat,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )

        if is_gas_oil_system:
            return derivs["dPcgo_dSo"]
        return derivs["dPcow_dSw"]

    def get_capillary_pressures(  # type: ignore[override]
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
        Compute three-phase capillary pressures with hysteresis.

        **Required kwargs for hysteresis**:
        - `max_water_saturation`: Historical maximum water saturation
        - `max_gas_saturation`: Historical maximum gas saturation
        - `water_imbibition_flag`: Boolean indicating water imbibition
        - `gas_imbibition_flag`: Boolean indicating gas imbibition
        - `water_reversal_saturation`: Saturation when water last reversed
        - `gas_reversal_saturation`: Saturation when gas last reversed

        :param water_saturation: Current water saturation.
        :param oil_saturation: Current oil saturation.
        :param gas_saturation: Current gas saturation.
        :return: CapillaryPressures with Pcow and Pcgo.
        """
        is_scalar = np.isscalar(water_saturation)
        sw = np.atleast_1d(water_saturation)
        so = np.atleast_1d(oil_saturation)
        sg = np.atleast_1d(gas_saturation)
        sw, so, sg = np.broadcast_arrays(sw, so, sg)

        use_hysteresis = (
            max_water_saturation is not None
            and max_gas_saturation is not None
            and water_imbibition_flag is not None
            and gas_imbibition_flag is not None
        )
        if use_hysteresis:
            sw_max = np.atleast_1d(max_water_saturation)  # type: ignore
            sg_max = np.atleast_1d(max_gas_saturation)  # type: ignore
            water_imb_flag = np.atleast_1d(water_imbibition_flag)  # type: ignore
            gas_imb_flag = np.atleast_1d(gas_imbibition_flag)  # type: ignore

            sw_rev = (
                np.atleast_1d(water_reversal_saturation)
                if water_reversal_saturation is not None
                else sw_max
            )
            sg_rev = (
                np.atleast_1d(gas_reversal_saturation)
                if gas_reversal_saturation is not None
                else sg_max
            )
        else:
            sw_max = sw
            sg_max = sg
            sw_rev = sw
            sg_rev = sg
            water_imb_flag = np.zeros_like(sw)
            gas_imb_flag = np.zeros_like(sg)

        # Oil-Water capillary pressure
        ow_table_drain = self.oil_water_drainage_table
        ow_table_imb = self.oil_water_imbibition_table or ow_table_drain

        # Drainage Pc
        pcow_drain = self._get_two_phase_pc(ow_table_drain, sw, so, so, sg, **kwargs)

        # Imbibition Pc
        pcow_imb = self._get_two_phase_pc(ow_table_imb, sw, so, so, sg, **kwargs)

        # Scanning curve
        pcow = _killough_scanning_curve_interpolation(
            saturation=sw,
            drainage_value=pcow_drain,
            imbibition_value=pcow_imb,
            reversal_saturation=sw_rev,
            max_saturation=sw_max,
            is_imbibition=water_imb_flag,
            interpolation_exponent=self.scanning_interpolation_exponent,
        )

        # Gas-Oil capillary pressure
        go_table_drain = self.gas_oil_drainage_table
        go_table_imb = self.gas_oil_imbibition_table or go_table_drain

        # Drainage Pc
        pcgo_drain = self._get_two_phase_pc(go_table_drain, so, sg, so, sg, **kwargs)

        # Imbibition Pc
        pcgo_imb = self._get_two_phase_pc(go_table_imb, so, sg, so, sg, **kwargs)

        # Scanning curve
        pcgo = _killough_scanning_curve_interpolation(
            saturation=sg,
            drainage_value=pcgo_drain,
            imbibition_value=pcgo_imb,
            reversal_saturation=sg_rev,
            max_saturation=sg_max,
            is_imbibition=gas_imb_flag,
            interpolation_exponent=self.scanning_interpolation_exponent,
        )

        if is_scalar:
            return CapillaryPressures(oil_water=pcow.item(), gas_oil=pcgo.item())  # type: ignore
        return CapillaryPressures(oil_water=pcow, gas_oil=pcgo)  # type: ignore[typeddict-item]

    def get_capillary_pressure_derivatives(  # type: ignore[override]
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
        Compute partial derivatives of capillary pressures with hysteresis.

        :param water_saturation: Current water saturation.
        :param oil_saturation: Current oil saturation.
        :param gas_saturation: Current gas saturation.
        :return: CapillaryPressureDerivatives dictionary.
        """
        is_scalar = np.isscalar(water_saturation)
        sw = np.atleast_1d(water_saturation)
        so = np.atleast_1d(oil_saturation)
        sg = np.atleast_1d(gas_saturation)
        sw, so, sg = np.broadcast_arrays(sw, so, sg)
        zeros = np.zeros_like(sw)

        use_hysteresis = (
            max_water_saturation is not None
            and max_gas_saturation is not None
            and water_imbibition_flag is not None
            and gas_imbibition_flag is not None
        )
        if use_hysteresis:
            sw_max = np.atleast_1d(max_water_saturation)  # type: ignore
            sg_max = np.atleast_1d(max_gas_saturation)  # type: ignore
            water_imb_flag = np.atleast_1d(water_imbibition_flag)  # type: ignore
            gas_imb_flag = np.atleast_1d(gas_imbibition_flag)  # type: ignore

            sw_rev = (
                np.atleast_1d(water_reversal_saturation)
                if water_reversal_saturation is not None
                else sw_max
            )
            sg_rev = (
                np.atleast_1d(gas_reversal_saturation)
                if gas_reversal_saturation is not None
                else sg_max
            )
        else:
            sw_max = sw
            sg_max = sg
            sw_rev = sw
            sg_rev = sg
            water_imb_flag = np.zeros_like(sw)
            gas_imb_flag = np.zeros_like(sg)

        # Oil-Water derivatives
        ow_table_drain = self.oil_water_drainage_table
        ow_table_imb = self.oil_water_imbibition_table or ow_table_drain

        pcow_drain = self._get_two_phase_pc(ow_table_drain, sw, so, so, sg, **kwargs)
        pcow_imb = self._get_two_phase_pc(ow_table_imb, sw, so, so, sg, **kwargs)
        d_pcow_drain = self._get_two_phase_pc_derivative(
            ow_table_drain, sw, so, so, sg, **kwargs
        )
        d_pcow_imb = self._get_two_phase_pc_derivative(
            ow_table_imb, sw, so, so, sg, **kwargs
        )

        d_pcow_d_sw = _compute_scanning_curve_derivatives(
            saturation=sw,
            d_val_drain_d_s=d_pcow_drain,
            d_val_imb_d_s=d_pcow_imb,
            drainage_value=pcow_drain,
            imbibition_value=pcow_imb,
            reversal_saturation=sw_rev,
            max_saturation=sw_max,
            is_imbibition=water_imb_flag,
            interpolation_exponent=self.scanning_interpolation_exponent,
        )

        # Gas-Oil derivatives
        go_table_drain = self.gas_oil_drainage_table
        go_table_imb = self.gas_oil_imbibition_table or go_table_drain

        pcgo_drain = self._get_two_phase_pc(go_table_drain, so, sg, so, sg, **kwargs)
        pcgo_imb = self._get_two_phase_pc(go_table_imb, so, sg, so, sg, **kwargs)
        d_pcgo_drain = self._get_two_phase_pc_derivative(
            go_table_drain, so, sg, so, sg, **kwargs
        )
        d_pcgo_imb = self._get_two_phase_pc_derivative(
            go_table_imb, so, sg, so, sg, **kwargs
        )

        d_pcgo_d_so = _compute_scanning_curve_derivatives(
            saturation=so,
            d_val_drain_d_s=d_pcgo_drain,
            d_val_imb_d_s=d_pcgo_imb,
            drainage_value=pcgo_drain,
            imbibition_value=pcgo_imb,
            reversal_saturation=1.0 - sg_rev - sw_rev,
            max_saturation=1.0 - sg_max - sw_max,
            is_imbibition=gas_imb_flag,
            interpolation_exponent=self.scanning_interpolation_exponent,
        )

        if is_scalar:
            return CapillaryPressureDerivatives(
                dPcow_dSw=d_pcow_d_sw.item(),  # type: ignore
                dPcow_dSo=0.0,
                dPcgo_dSo=d_pcgo_d_so.item(),  # type: ignore
                dPcgo_dSg=0.0,
            )

        return CapillaryPressureDerivatives(
            dPcow_dSw=d_pcow_d_sw,
            dPcow_dSo=zeros,
            dPcgo_dSo=d_pcgo_d_so,
            dPcgo_dSg=zeros,
        )
