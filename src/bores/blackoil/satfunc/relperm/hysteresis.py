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

import typing

import attrs
import numba
import numpy as np
import numpy.typing as npt

from bores.blackoil.satfunc.relperm.tables import (
    MixingRule,
    RelativePermeabilityTable,
    TwoPhaseRelPermTable,
    get_mixing_rule,
    get_mixing_rule_partial_derivatives,
)
from bores.blackoil.satfunc.utils import (
    compute_killough_scanning_curve,
    compute_killough_scanning_curve_derivative,
)
from bores.constants import c
from bores.errors import ValidationError
from bores.types import (
    BooleanArray,
    FluidPhase,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    RelativePermeabilities,
    RelativePermeabilityDerivatives,
)
from bores.utils import atleast_1d

__all__ = ["KilloughLandRelPermTable"]


@numba.njit(cache=True)
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

        S_r = S_r_max / (1 + C * S_i)

    Higher *C* means more trapping (smaller residual for the same initial
    saturation). *C* = 0 means no dynamic trapping (residual equals
    `max_residual_saturation` regardless of initial saturation).

    :param initial_non_wetting_saturation: Non-wetting saturation at the
        drainage-imbibition reversal point.
    :param maximum_residual_saturation: Maximum residual saturation from the
        drainage endpoint (S_r_max).
    :param land_trapping_coefficient: Land trapping coefficient *C* (>= 0).
    :param saturation_epsilon: Small value to guard against division by zero.
    :return: Dynamic residual saturation (scalar).
    """
    safe_maximum_residual = max(maximum_residual_saturation, saturation_epsilon)

    if land_trapping_coefficient <= 0.0:
        return safe_maximum_residual

    non_negative_initial = max(initial_non_wetting_saturation, 0.0)
    dynamic_residual = safe_maximum_residual / (
        1.0 + land_trapping_coefficient * non_negative_initial
    )
    return min(dynamic_residual, non_negative_initial)


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
        drainage-imbibition reversal points (1-D or N-D array).
    :param maximum_residual_saturation: Maximum residual saturation from the
        drainage endpoint (scalar).
    :param land_trapping_coefficient: Land trapping coefficient *C* (>= 0).
    :param saturation_epsilon: Small value to guard against division by zero.
    :return: Dynamic residual saturation array with the same shape as the input.
    """
    initial_saturation_array = atleast_1d(initial_non_wetting_saturation)
    safe_maximum_residual = max(maximum_residual_saturation, saturation_epsilon)

    result = np.empty_like(initial_saturation_array)
    for flat_idx in numba.prange(initial_saturation_array.size):  # type: ignore
        non_negative_initial = max(initial_saturation_array.flat[flat_idx], 0.0)
        if land_trapping_coefficient <= 0.0:
            result.flat[flat_idx] = min(safe_maximum_residual, non_negative_initial)
        else:
            dynamic_residual = safe_maximum_residual / (
                1.0 + land_trapping_coefficient * non_negative_initial
            )
            result.flat[flat_idx] = min(dynamic_residual, non_negative_initial)
    return typing.cast(NumberArray[NDimension], result)


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
    return compute_land_residual_saturation_array(
        initial_non_wetting_saturation=typing.cast(
            NumberArray[NDimension], initial_non_wetting_saturation
        ),
        maximum_residual_saturation=maximum_residual_saturation,
        land_trapping_coefficient=land_trapping_coefficient,
        saturation_epsilon=saturation_epsilon,
    )


def get_oil_water_relative_permeabilities(
    oil_water_table: TwoPhaseRelPermTable | RelativePermeabilityTable,
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    **kwargs: typing.Any,
) -> tuple[NumberOrArray[NDimension], NumberOrArray[NDimension]]:
    """
    Return `(krw, kro_w)` from an oil-water relative permeability table.

    Dispatches saturations correctly for both `TwoPhaseRelPermTable` and
    full `RelativePermeabilityTable` instances by calling
    `table.get_oil_water_wetting_phase()` rather than inspecting any internal attribute.

    :param oil_water_table: Oil-water relative permeability table (two-phase or three-phase).
    :param water_saturation: Water saturation (scalar or array).
    :param oil_saturation: Oil saturation (scalar or array).
    :param gas_saturation: Gas saturation (scalar or array).
    :param kwargs: Additional keyword arguments forwarded to parametric tables.
    :return: Tuple of `(krw, kro_w)` - water and oil relative permeabilities
        from the oil-water sub-system.
    """
    oil_water_wetting_phase = oil_water_table.get_oil_water_wetting_phase()

    if isinstance(oil_water_table, TwoPhaseRelPermTable):
        if oil_water_wetting_phase == FluidPhase.WATER:
            wetting_saturation = water_saturation
            non_wetting_saturation = oil_saturation
        else:
            wetting_saturation = oil_saturation
            non_wetting_saturation = water_saturation

        water_relative_permeability = (
            oil_water_table.get_wetting_phase_relative_permeability(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
            if oil_water_wetting_phase == FluidPhase.WATER
            else oil_water_table.get_non_wetting_phase_relative_permeability(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
        )
        oil_relative_permeability = (
            oil_water_table.get_non_wetting_phase_relative_permeability(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
            if oil_water_wetting_phase == FluidPhase.WATER
            else oil_water_table.get_wetting_phase_relative_permeability(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
        )
        return water_relative_permeability, oil_relative_permeability

    # Full three-phase table
    result = oil_water_table.evaluate(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    return result["water"], result["oil"]


def get_gas_oil_relative_permeabilities(
    gas_oil_table: TwoPhaseRelPermTable | RelativePermeabilityTable,
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    **kwargs: typing.Any,
) -> tuple[NumberOrArray[NDimension], NumberOrArray[NDimension]]:
    """
    Return `(kro_g, krg)` from a gas-oil relative permeability table.

    Dispatches saturations correctly for both `TwoPhaseRelPermTable` and
    full `RelativePermeabilityTable` instances by calling
    `table.get_gas_oil_wetting_phase()` rather than inspecting any internal attribute.

    :param gas_oil_table: Gas-oil relative permeability table (two-phase or three-phase).
    :param water_saturation: Water saturation (scalar or array).
    :param oil_saturation: Oil saturation (scalar or array).
    :param gas_saturation: Gas saturation (scalar or array).
    :param kwargs: Additional keyword arguments forwarded to parametric tables.
    :return: Tuple of `(kro_g, krg)` - oil and gas relative permeabilities
        from the gas-oil sub-system.
    """
    gas_oil_wetting_phase = gas_oil_table.get_gas_oil_wetting_phase()

    if isinstance(gas_oil_table, TwoPhaseRelPermTable):
        if gas_oil_wetting_phase == FluidPhase.OIL:
            wetting_saturation = oil_saturation
            non_wetting_saturation = gas_saturation
        else:
            wetting_saturation = gas_saturation
            non_wetting_saturation = oil_saturation

        oil_relative_permeability = (
            gas_oil_table.get_wetting_phase_relative_permeability(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
            if gas_oil_wetting_phase == FluidPhase.OIL
            else gas_oil_table.get_non_wetting_phase_relative_permeability(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
        )
        gas_relative_permeability = (
            gas_oil_table.get_non_wetting_phase_relative_permeability(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
            if gas_oil_wetting_phase == FluidPhase.OIL
            else gas_oil_table.get_wetting_phase_relative_permeability(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
        )
        return oil_relative_permeability, gas_relative_permeability

    result = gas_oil_table.evaluate(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    return result["oil"], result["gas"]


def get_oil_water_relative_permeability_derivatives(
    oil_water_table: TwoPhaseRelPermTable | RelativePermeabilityTable,
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    **kwargs: typing.Any,
) -> tuple[NumberOrArray[NDimension], NumberOrArray[NDimension]]:
    """
    Return `(d_krw/d_ref, d_kro_w/d_ref)` for the oil-water table, where
    *ref* is the table's natural reference saturation (Sw in water-wet, So in
    oil-wet).

    :param oil_water_table: Oil-water relative permeability table (two-phase or three-phase).
    :param water_saturation: Water saturation (scalar or array).
    :param oil_saturation: Oil saturation (scalar or array).
    :param gas_saturation: Gas saturation (scalar or array).
    :param kwargs: Additional keyword arguments forwarded to parametric tables.
    :return: Tuple of `(d_krw/d_ref, d_kro_w/d_ref)` where *ref* is the
        reference saturation axis of the oil-water sub-system.
    """
    oil_water_wetting_phase = oil_water_table.get_oil_water_wetting_phase()

    if isinstance(oil_water_table, TwoPhaseRelPermTable):
        if oil_water_wetting_phase == FluidPhase.WATER:
            wetting_saturation = water_saturation
            non_wetting_saturation = oil_saturation
        else:
            wetting_saturation = oil_saturation
            non_wetting_saturation = water_saturation

        wetting_derivative = oil_water_table.get_wetting_phase_relative_permeability_derivative(
            wetting_saturation=wetting_saturation,
            non_wetting_saturation=non_wetting_saturation,
        )
        non_wetting_derivative = (
            oil_water_table.get_non_wetting_phase_relative_permeability_derivative(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
        )
        if oil_water_wetting_phase == FluidPhase.WATER:
            return (
                wetting_derivative,
                non_wetting_derivative,
            )  # d_krw/d_Sw, d_kro_w/d_Sw
        return non_wetting_derivative, wetting_derivative  # d_krw/d_So, d_kro_w/d_So

    derivatives = oil_water_table.derivatives(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    # The oil-water hysteresis scanning variable is always Sw
    return derivatives["dkrw_dsw"], derivatives["dkro_dsw"]


def get_gas_oil_relative_permeability_derivatives(
    gas_oil_table: TwoPhaseRelPermTable | RelativePermeabilityTable,
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    **kwargs: typing.Any,
) -> tuple[NumberOrArray[NDimension], NumberOrArray[NDimension]]:
    """
    Return `(d_kro_g/d_ref, d_krg/d_ref)` for the gas-oil table, where
    *ref* is the table's natural reference saturation (So in oil-wet, Sg in
    gas-wet).

    :param gas_oil_table: Gas-oil relative permeability table (two-phase or three-phase).
    :param water_saturation: Water saturation (scalar or array).
    :param oil_saturation: Oil saturation (scalar or array).
    :param gas_saturation: Gas saturation (scalar or array).
    :param kwargs: Additional keyword arguments forwarded to parametric tables.
    :return: Tuple of `(d_kro_g/d_ref, d_krg/d_ref)` where *ref* is the
        reference saturation axis of the gas-oil sub-system.
    """
    gas_oil_wetting_phase = gas_oil_table.get_gas_oil_wetting_phase()

    if isinstance(gas_oil_table, TwoPhaseRelPermTable):
        if gas_oil_wetting_phase == FluidPhase.OIL:
            wetting_saturation = oil_saturation
            non_wetting_saturation = gas_saturation
        else:
            wetting_saturation = gas_saturation
            non_wetting_saturation = oil_saturation

        wetting_derivative = gas_oil_table.get_wetting_phase_relative_permeability_derivative(
            wetting_saturation=wetting_saturation,
            non_wetting_saturation=non_wetting_saturation,
        )
        non_wetting_derivative = (
            gas_oil_table.get_non_wetting_phase_relative_permeability_derivative(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
        )
        if gas_oil_wetting_phase == FluidPhase.OIL:
            return (
                wetting_derivative,
                non_wetting_derivative,
            )  # d_kro_g/d_So, d_krg/d_So
        return non_wetting_derivative, wetting_derivative  # d_kro_g/d_Sg, d_krg/d_Sg

    derivatives = gas_oil_table.derivatives(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    # The gas-oil hysteresis scanning variable is always Sg
    return derivatives["dkro_dsg"], derivatives["dkrg_dsg"]


@attrs.frozen(slots=True)
class KilloughLandRelPermTable:
    """
    Implements the Killough relative permeability hysteresis model/table with Land trapping.

    During *primary drainage* the relative permeabilities follow the
    `oil_water_drainage_table` and `gas_oil_drainage_table`.

    When the flow reverses (*imbibition*) two things happen simultaneously.

    **Land trapping**: a portion of the non-wetting phase becomes
    disconnected. The dynamic residual saturation depends on the saturation
    at the reversal point via Land's formula:

        S_r(S_i) = S_r_max / (1 + C * S_i)

    **Killough scanning curves**: between the reversal point and the
    maximum historical saturation, kr follows a scanning curve that
    interpolates between the primary drainage and imbibition bounds.

    Both two-phase (`TwoPhaseRelPermTable`) and full three-phase
    (`RelativePermeabilityTable`) backing tables are supported. Wetting
    and non-wetting phase roles are resolved through the canonical API
    (`get_oil_water_wetting_phase` / `get_gas_oil_wetting_phase`) so the
    model is wettability-agnostic.

    The hysteresis history is passed as additional keyword arguments to
    `evaluate` and `derivatives`. When these arguments are
    absent the model degenerates to the primary drainage curves, which is the
    physically correct behaviour for the first drainage cycle
    (see `simulate.py`, the `enable_hysteresis` flag in `Config` controls
    whether history is tracked and passed through).
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
    """Land trapping coefficient *C* for the oil-water system (>= 0)."""

    land_trapping_coefficient_gas: Number = 1.0
    """Land trapping coefficient *C* for the gas-oil system (>= 0)."""

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

    mixing_rule: MixingRule | str = "eclipse_rule"
    """Three-phase oil relative permeability mixing rule."""

    supports_vector: bool = attrs.field(init=False, repr=False, default=True)

    def __attrs_post_init__(self) -> None:
        if isinstance(self.mixing_rule, str):
            object.__setattr__(self, "mixing_rule", get_mixing_rule(self.mixing_rule))

        if isinstance(self.oil_water_drainage_table, TwoPhaseRelPermTable) and {
            self.oil_water_drainage_table.wetting_phase,
            self.oil_water_drainage_table.non_wetting_phase,
        } != {FluidPhase.WATER, FluidPhase.OIL}:
            raise ValidationError("`oil_water_drainage_table` must involve water and oil phases.")

        if isinstance(self.gas_oil_drainage_table, TwoPhaseRelPermTable) and {
            self.gas_oil_drainage_table.wetting_phase,
            self.gas_oil_drainage_table.non_wetting_phase,
        } != {FluidPhase.OIL, FluidPhase.GAS}:
            raise ValidationError("`gas_oil_drainage_table` must involve oil and gas phases.")

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
            raise ValidationError("`gas_oil_imbibition_table` must involve oil and gas phases.")

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
        """Connate water saturation from the primary drainage table (static; hysteresis doesn't move it)."""
        return self.oil_water_drainage_table.get_connate_water_saturation()

    def get_residual_oil_saturation_water(self) -> Number:
        """
        Static (drainage-table) residual oil to water, from the primary
        drainage table. Note this is the *static* sorw the table declares,
        not the dynamically Land-trapped value used during an active
        imbibition scan (see `_parse_hysteresis_kwargs`/`compute_land_residual_saturation`).
        """
        return self.oil_water_drainage_table.get_residual_oil_saturation_water()

    def get_residual_oil_saturation_gas(self) -> Number:
        """Static (drainage-table) residual oil to gas; see `get_residual_oil_saturation_water`."""
        return self.gas_oil_drainage_table.get_residual_oil_saturation_gas()

    def get_residual_gas_saturation(self) -> Number:
        """Static (drainage-table) residual/critical gas saturation; see `get_residual_oil_saturation_water`."""
        return self.gas_oil_drainage_table.get_residual_gas_saturation()

    def _parse_hysteresis_kwargs(
        self,
        water_saturation: npt.NDArray,
        gas_saturation: npt.NDArray,
        max_water_saturation: NumberOrArray[NDimension] | None,
        max_gas_saturation: NumberOrArray[NDimension] | None,
        water_imbibition_flag: bool | BooleanArray[NDimension] | None,
        gas_imbibition_flag: bool | BooleanArray[NDimension] | None,
        water_reversal_saturation: NumberOrArray[NDimension] | None,
        gas_reversal_saturation: NumberOrArray[NDimension] | None,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Parse and broadcast saturation-history arrays.

        The oil-water and gas-oil sub-systems are resolved **independently**:
        each falls back to its own primary-drainage state (no scanning curve,
        no Land trapping) if its own history is incomplete, rather than
        requiring both sub-systems' history to be present before either can
        engage hysteresis. Supplying `max_water_saturation`/
        `water_imbibition_flag` alone is enough to activate oil-water
        hysteresis even if the gas-oil history is entirely absent, and vice
        versa.

        :param water_saturation: Broadcast-ready water saturation array.
        :param gas_saturation: Broadcast-ready gas saturation array.
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
        use_water_hysteresis = (
            max_water_saturation is not None and water_imbibition_flag is not None
        )
        use_gas_hysteresis = max_gas_saturation is not None and gas_imbibition_flag is not None

        if use_water_hysteresis:
            maximum_water_saturation = np.atleast_1d(max_water_saturation)  # type: ignore
            water_imbibition_flag = np.atleast_1d(water_imbibition_flag)  # type: ignore
            water_reversal_saturation = typing.cast(
                NumberArray[NDimension],
                np.atleast_1d(water_reversal_saturation)
                if water_reversal_saturation is not None
                else maximum_water_saturation.copy(),
            )
        else:
            maximum_water_saturation = water_saturation.copy()
            water_imbibition_flag = typing.cast(
                BooleanArray[NDimension],
                np.zeros_like(water_saturation, dtype=np.bool_),
            )
            water_reversal_saturation = typing.cast(
                NumberArray[NDimension], water_saturation.copy()
            )

        if use_gas_hysteresis:
            maximum_gas_saturation = np.atleast_1d(max_gas_saturation)  # type: ignore
            gas_imbibition_flag = np.atleast_1d(gas_imbibition_flag)  # type: ignore
            gas_reversal_saturation = typing.cast(
                NumberArray[NDimension],
                np.atleast_1d(gas_reversal_saturation)
                if gas_reversal_saturation is not None
                else maximum_gas_saturation.copy(),
            )
        else:
            maximum_gas_saturation = gas_saturation.copy()
            gas_imbibition_flag = typing.cast(
                BooleanArray[NDimension], np.zeros_like(gas_saturation, dtype=np.bool_)
            )
            gas_reversal_saturation = typing.cast(NumberArray[NDimension], gas_saturation.copy())

        (
            maximum_water_saturation,
            maximum_gas_saturation,
            water_imbibition_flag,
            gas_imbibition_flag,
            water_reversal_saturation,
            gas_reversal_saturation,
        ) = np.broadcast_arrays(
            maximum_water_saturation,
            maximum_gas_saturation,
            water_imbibition_flag,  # type: ignore
            gas_imbibition_flag,  # type: ignore
            water_reversal_saturation,
            gas_reversal_saturation,
        )
        return (  # type: ignore[return-value]
            maximum_water_saturation,
            maximum_gas_saturation,
            water_imbibition_flag,
            gas_imbibition_flag,
            water_reversal_saturation,
            gas_reversal_saturation,
        )

    def evaluate(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        max_water_saturation: NumberOrArray[NDimension] | None = None,
        max_gas_saturation: NumberOrArray[NDimension] | None = None,
        water_imbibition_flag: bool | BooleanArray[NDimension] | None = None,
        gas_imbibition_flag: bool | BooleanArray[NDimension] | None = None,
        water_reversal_saturation: NumberOrArray[NDimension] | None = None,
        gas_reversal_saturation: NumberOrArray[NDimension] | None = None,
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute three-phase relative permeabilities with Killough/Land hysteresis.

        When the saturation-history keyword arguments are absent the method
        returns primary drainage kr values (no hysteresis).

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param max_water_saturation: Historical maximum water saturation per cell.
            Enables hysteresis when provided together with the other history args.
        :param max_gas_saturation: Historical maximum gas saturation per cell.
        :param water_imbibition_flag: Boolean / float flag per cell - 1 if water
            saturation is currently increasing (imbibition in the oil-water
            system), 0 if decreasing (drainage).
        :param gas_imbibition_flag: Boolean / float flag per cell - 1 if gas
            saturation is currently increasing, 0 if decreasing.
        :param water_reversal_saturation: Water saturation at the last oil-water
            reversal.  Defaults to `max_water_saturation` when not supplied.
        :param gas_reversal_saturation: Gas saturation at the last gas-oil reversal.
            Defaults to `max_gas_saturation` when not supplied.
        :param kwargs: Additional keyword arguments forwarded to the underlying
            backing tables (e.g. residual saturation overrides for parametric
            models).
        :return: `RelativePermeabilities` dictionary with keys `"water"`, `"oil"`, and `"gas"`.
        """
        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
        )
        water_saturation = np.atleast_1d(water_saturation)  # type: ignore
        oil_saturation = np.atleast_1d(oil_saturation)  # type: ignore
        gas_saturation = np.atleast_1d(gas_saturation)  # type: ignore
        water_saturation, oil_saturation, gas_saturation = np.broadcast_arrays(  # type: ignore
            water_saturation, oil_saturation, gas_saturation
        )

        # Normalise saturations
        total_saturation = water_saturation + oil_saturation + gas_saturation
        normalisation_mask = (np.abs(total_saturation - 1.0) > c.SATURATION_EPSILON) & (
            total_saturation > 0.0
        )
        if np.any(normalisation_mask):
            water_saturation = np.where(  # type: ignore
                normalisation_mask,
                water_saturation / total_saturation,
                water_saturation,
            )
            oil_saturation = np.where(  # type: ignore
                normalisation_mask,
                oil_saturation / total_saturation,
                oil_saturation,
            )
            gas_saturation = np.where(  # type: ignore
                normalisation_mask,
                gas_saturation / total_saturation,
                gas_saturation,
            )

        # Captured before the tuple-unpack below shadows `water_imbibition_flag`/
        # `gas_imbibition_flag` with their resolved (always non-None) arrays;
        # matches _parse_hysteresis_kwargs's own per-sub-system gate exactly,
        # so Land trapping only engages when the same sub-system's scanning
        # curve does.
        use_water_hysteresis = (
            max_water_saturation is not None and water_imbibition_flag is not None
        )
        use_gas_hysteresis = max_gas_saturation is not None and gas_imbibition_flag is not None

        (
            maximum_water_saturation,
            maximum_gas_saturation,
            water_imbibition_flag,
            gas_imbibition_flag,
            water_reversal_saturation,
            gas_reversal_saturation,
        ) = self._parse_hysteresis_kwargs(  # type: ignore
            water_saturation=water_saturation,  # type: ignore
            gas_saturation=gas_saturation,  # type: ignore
            max_water_saturation=max_water_saturation,
            max_gas_saturation=max_gas_saturation,
            water_imbibition_flag=water_imbibition_flag,
            gas_imbibition_flag=gas_imbibition_flag,
            water_reversal_saturation=water_reversal_saturation,
            gas_reversal_saturation=gas_reversal_saturation,
        )

        oil_water_drainage_table = self.oil_water_drainage_table
        oil_water_imbibition_table = self.oil_water_imbibition_table or oil_water_drainage_table
        gas_oil_drainage_table = self.gas_oil_drainage_table
        gas_oil_imbibition_table = self.gas_oil_imbibition_table or gas_oil_drainage_table

        # Oil-water system - Land trapping on oil
        oil_saturation_at_oil_water_reversal = np.maximum(
            0.0,
            1.0 - water_reversal_saturation - gas_saturation,  # type: ignore
        )
        imbibition_oil_water_kwargs = dict(kwargs)
        if use_water_hysteresis and self.maximum_residual_oil_saturation_water is not None:
            dynamic_residual_oil_saturation_water = compute_land_residual_saturation(
                initial_non_wetting_saturation=oil_saturation_at_oil_water_reversal,
                maximum_residual_saturation=self.maximum_residual_oil_saturation_water,
                land_trapping_coefficient=self.land_trapping_coefficient_water,
            )
            imbibition_oil_water_kwargs["residual_oil_saturation_water"] = (
                dynamic_residual_oil_saturation_water
            )

        (
            water_relative_permeability_drainage,
            oil_relative_permeability_water_drainage,
        ) = get_oil_water_relative_permeabilities(
            oil_water_table=oil_water_drainage_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        (
            water_relative_permeability_imbibition,
            oil_relative_permeability_water_imbibition,
        ) = get_oil_water_relative_permeabilities(
            oil_water_table=oil_water_imbibition_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **imbibition_oil_water_kwargs,
        )

        water_relative_permeability = compute_killough_scanning_curve(
            saturation=water_saturation,
            drainage_curve_value=water_relative_permeability_drainage,
            imbibition_curve_value=water_relative_permeability_imbibition,
            reversal_saturation=water_reversal_saturation,  # type: ignore
            maximum_historical_saturation=maximum_water_saturation,
            is_imbibition=water_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )
        oil_relative_permeability_water = compute_killough_scanning_curve(
            saturation=water_saturation,
            drainage_curve_value=oil_relative_permeability_water_drainage,
            imbibition_curve_value=oil_relative_permeability_water_imbibition,
            reversal_saturation=water_reversal_saturation,  # type: ignore
            maximum_historical_saturation=maximum_water_saturation,
            is_imbibition=water_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )

        # Gas-oil system - Land trapping on gas and oil
        oil_saturation_at_gas_oil_reversal = np.maximum(
            0.0,
            1.0 - gas_reversal_saturation - water_saturation,  # type: ignore
        )
        imbibition_gas_oil_kwargs = dict(kwargs)
        if use_gas_hysteresis and self.maximum_residual_gas_saturation is not None:
            dynamic_residual_gas_saturation = compute_land_residual_saturation(
                initial_non_wetting_saturation=gas_reversal_saturation,  # type: ignore
                maximum_residual_saturation=self.maximum_residual_gas_saturation,
                land_trapping_coefficient=self.land_trapping_coefficient_gas,
            )
            imbibition_gas_oil_kwargs["residual_gas_saturation"] = dynamic_residual_gas_saturation
        if use_gas_hysteresis and self.maximum_residual_oil_saturation_gas is not None:
            dynamic_residual_oil_saturation_gas = compute_land_residual_saturation(
                initial_non_wetting_saturation=oil_saturation_at_gas_oil_reversal,
                maximum_residual_saturation=self.maximum_residual_oil_saturation_gas,
                land_trapping_coefficient=self.land_trapping_coefficient_gas,
            )
            imbibition_gas_oil_kwargs["residual_oil_saturation_gas"] = (
                dynamic_residual_oil_saturation_gas
            )

        oil_relative_permeability_gas_drainage, gas_relative_permeability_drainage = (
            get_gas_oil_relative_permeabilities(
                gas_oil_table=gas_oil_drainage_table,
                water_saturation=water_saturation,
                oil_saturation=oil_saturation,
                gas_saturation=gas_saturation,
                **kwargs,
            )
        )
        (
            oil_relative_permeability_gas_imbibition,
            gas_relative_permeability_imbibition,
        ) = get_gas_oil_relative_permeabilities(
            gas_oil_table=gas_oil_imbibition_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **imbibition_gas_oil_kwargs,
        )

        gas_relative_permeability = compute_killough_scanning_curve(
            saturation=gas_saturation,
            drainage_curve_value=gas_relative_permeability_drainage,
            imbibition_curve_value=gas_relative_permeability_imbibition,
            reversal_saturation=gas_reversal_saturation,  # type: ignore
            maximum_historical_saturation=maximum_gas_saturation,
            is_imbibition=gas_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )
        oil_relative_permeability_gas = compute_killough_scanning_curve(
            saturation=gas_saturation,
            drainage_curve_value=oil_relative_permeability_gas_drainage,
            imbibition_curve_value=oil_relative_permeability_gas_imbibition,
            reversal_saturation=gas_reversal_saturation,  # type: ignore
            maximum_historical_saturation=maximum_gas_saturation,
            is_imbibition=gas_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )
        oil_relperm_endpoint = self.get_oil_relperm_endpoint()

        # Three-phase oil via mixing rule
        three_phase_mixing_rule = typing.cast(MixingRule, self.mixing_rule)
        oil_relative_permeability = three_phase_mixing_rule(
            kro_w=oil_relative_permeability_water,
            kro_g=oil_relative_permeability_gas,
            krw=water_relative_permeability,
            krg=gas_relative_permeability,
            kr_max=oil_relperm_endpoint,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
        )

        if is_scalar:
            return RelativePermeabilities(
                water=water_relative_permeability.item(),  # type: ignore
                oil=oil_relative_permeability.item(),  # type: ignore
                gas=gas_relative_permeability.item(),  # type: ignore
            )
        return RelativePermeabilities(
            water=water_relative_permeability,
            oil=oil_relative_permeability,
            gas=gas_relative_permeability,
        )

    def derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        max_water_saturation: NumberOrArray[NDimension] | None = None,
        max_gas_saturation: NumberOrArray[NDimension] | None = None,
        water_imbibition_flag: bool | BooleanArray[NDimension] | None = None,
        gas_imbibition_flag: bool | BooleanArray[NDimension] | None = None,
        water_reversal_saturation: NumberOrArray[NDimension] | None = None,
        gas_reversal_saturation: NumberOrArray[NDimension] | None = None,
        **kwargs: typing.Any,
    ) -> RelativePermeabilityDerivatives:
        """
        Compute partial derivatives of three-phase relative permeabilities
        with Killough/Land hysteresis.

        Returns all nine ∂kr_α/∂S_β entries. Derivatives with respect to
        the scanning variable (Sw for oil-water, Sg for gas-oil) are computed
        analytically via the chain rule through the Killough scanning-curve
        formula; all other cross-derivatives are zero (consistent with the
        assumption that each two-phase sub-system depends only on its own
        reference saturation).

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
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
        water_saturation = np.atleast_1d(water_saturation)  # type: ignore
        oil_saturation = np.atleast_1d(oil_saturation)  # type: ignore
        gas_saturation = np.atleast_1d(gas_saturation)  # type: ignore
        water_saturation, oil_saturation, gas_saturation = np.broadcast_arrays(  # type: ignore
            water_saturation, oil_saturation, gas_saturation
        )
        zeros = np.zeros_like(water_saturation)

        # Captured before the tuple-unpack below shadows `water_imbibition_flag`/
        # `gas_imbibition_flag` with their resolved (always non-None) arrays;
        # matches _parse_hysteresis_kwargs's own per-sub-system gate exactly,
        # so Land trapping only engages when the same sub-system's scanning
        # curve does.
        use_water_hysteresis = (
            max_water_saturation is not None and water_imbibition_flag is not None
        )
        use_gas_hysteresis = max_gas_saturation is not None and gas_imbibition_flag is not None

        (
            maximum_water_saturation,
            maximum_gas_saturation,
            water_imbibition_flag,
            gas_imbibition_flag,
            water_reversal_saturation,
            gas_reversal_saturation,
        ) = self._parse_hysteresis_kwargs(  # type: ignore
            water_saturation=water_saturation,  # type: ignore
            gas_saturation=gas_saturation,  # type: ignore
            max_water_saturation=max_water_saturation,
            max_gas_saturation=max_gas_saturation,
            water_imbibition_flag=water_imbibition_flag,
            gas_imbibition_flag=gas_imbibition_flag,
            water_reversal_saturation=water_reversal_saturation,
            gas_reversal_saturation=gas_reversal_saturation,
        )

        oil_water_drainage_table = self.oil_water_drainage_table
        oil_water_imbibition_table = self.oil_water_imbibition_table or oil_water_drainage_table
        gas_oil_drainage_table = self.gas_oil_drainage_table
        gas_oil_imbibition_table = self.gas_oil_imbibition_table or gas_oil_drainage_table

        # Build imbibition kwargs with Land trapping
        oil_saturation_at_oil_water_reversal = np.maximum(
            0.0,
            1.0 - water_reversal_saturation - gas_saturation,  # type: ignore
        )
        imbibition_oil_water_kwargs = dict(kwargs)
        if use_water_hysteresis and self.maximum_residual_oil_saturation_water is not None:
            dynamic_residual_oil_saturation_water = compute_land_residual_saturation(
                initial_non_wetting_saturation=oil_saturation_at_oil_water_reversal,
                maximum_residual_saturation=self.maximum_residual_oil_saturation_water,
                land_trapping_coefficient=self.land_trapping_coefficient_water,
            )
            imbibition_oil_water_kwargs["residual_oil_saturation_water"] = (
                dynamic_residual_oil_saturation_water
            )

        oil_saturation_at_gas_oil_reversal = np.maximum(
            0.0,
            1.0 - gas_reversal_saturation - water_saturation,  # type: ignore
        )
        imbibition_gas_oil_kwargs = dict(kwargs)
        if use_gas_hysteresis and self.maximum_residual_gas_saturation is not None:
            dynamic_residual_gas_saturation = compute_land_residual_saturation(
                initial_non_wetting_saturation=gas_reversal_saturation,  # type: ignore
                maximum_residual_saturation=self.maximum_residual_gas_saturation,
                land_trapping_coefficient=self.land_trapping_coefficient_gas,
            )
            imbibition_gas_oil_kwargs["residual_gas_saturation"] = dynamic_residual_gas_saturation
        if use_gas_hysteresis and self.maximum_residual_oil_saturation_gas is not None:
            dynamic_residual_oil_saturation_gas = compute_land_residual_saturation(
                initial_non_wetting_saturation=oil_saturation_at_gas_oil_reversal,
                maximum_residual_saturation=self.maximum_residual_oil_saturation_gas,
                land_trapping_coefficient=self.land_trapping_coefficient_gas,
            )
            imbibition_gas_oil_kwargs["residual_oil_saturation_gas"] = (
                dynamic_residual_oil_saturation_gas
            )

        # Oil-water - values and derivatives
        (
            water_relative_permeability_drainage,
            oil_relative_permeability_water_drainage,
        ) = get_oil_water_relative_permeabilities(
            oil_water_table=oil_water_drainage_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        (
            water_relative_permeability_imbibition,
            oil_relative_permeability_water_imbibition,
        ) = get_oil_water_relative_permeabilities(
            oil_water_table=oil_water_imbibition_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **imbibition_oil_water_kwargs,
        )

        (
            water_relative_permeability_drainage_derivative,
            oil_relative_permeability_water_drainage_derivative,
        ) = get_oil_water_relative_permeability_derivatives(
            oil_water_table=oil_water_drainage_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        (
            water_relative_permeability_imbibition_derivative,
            oil_relative_permeability_water_imbibition_derivative,
        ) = get_oil_water_relative_permeability_derivatives(
            oil_water_table=oil_water_imbibition_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **imbibition_oil_water_kwargs,
        )

        water_relative_permeability_derivative = compute_killough_scanning_curve_derivative(
            saturation=water_saturation,
            drainage_curve_value=water_relative_permeability_drainage,
            imbibition_curve_value=water_relative_permeability_imbibition,
            drainage_curve_derivative=water_relative_permeability_drainage_derivative,
            imbibition_curve_derivative=water_relative_permeability_imbibition_derivative,
            reversal_saturation=water_reversal_saturation,  # type: ignore
            maximum_historical_saturation=maximum_water_saturation,
            is_imbibition=water_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )
        oil_relative_permeability_water_derivative = compute_killough_scanning_curve_derivative(
            saturation=water_saturation,
            drainage_curve_value=oil_relative_permeability_water_drainage,
            imbibition_curve_value=oil_relative_permeability_water_imbibition,
            drainage_curve_derivative=oil_relative_permeability_water_drainage_derivative,
            imbibition_curve_derivative=oil_relative_permeability_water_imbibition_derivative,
            reversal_saturation=water_reversal_saturation,  # type: ignore
            maximum_historical_saturation=maximum_water_saturation,
            is_imbibition=water_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )

        water_relative_permeability = compute_killough_scanning_curve(
            saturation=water_saturation,
            drainage_curve_value=water_relative_permeability_drainage,
            imbibition_curve_value=water_relative_permeability_imbibition,
            reversal_saturation=water_reversal_saturation,  # type: ignore
            maximum_historical_saturation=maximum_water_saturation,
            is_imbibition=water_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )
        oil_relative_permeability_water = compute_killough_scanning_curve(
            saturation=water_saturation,
            drainage_curve_value=oil_relative_permeability_water_drainage,
            imbibition_curve_value=oil_relative_permeability_water_imbibition,
            reversal_saturation=water_reversal_saturation,  # type: ignore
            maximum_historical_saturation=maximum_water_saturation,
            is_imbibition=water_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )

        # Gas-oil - values and derivatives
        oil_relative_permeability_gas_drainage, gas_relative_permeability_drainage = (
            get_gas_oil_relative_permeabilities(
                gas_oil_table=gas_oil_drainage_table,
                water_saturation=water_saturation,
                oil_saturation=oil_saturation,
                gas_saturation=gas_saturation,
                **kwargs,
            )
        )
        (
            oil_relative_permeability_gas_imbibition,
            gas_relative_permeability_imbibition,
        ) = get_gas_oil_relative_permeabilities(
            gas_oil_table=gas_oil_imbibition_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **imbibition_gas_oil_kwargs,
        )

        (
            oil_relative_permeability_gas_drainage_derivative,
            gas_relative_permeability_drainage_derivative,
        ) = get_gas_oil_relative_permeability_derivatives(
            gas_oil_table=gas_oil_drainage_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        (
            oil_relative_permeability_gas_imbibition_derivative,
            gas_relative_permeability_imbibition_derivative,
        ) = get_gas_oil_relative_permeability_derivatives(
            gas_oil_table=gas_oil_imbibition_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **imbibition_gas_oil_kwargs,
        )

        gas_relative_permeability_derivative = compute_killough_scanning_curve_derivative(
            saturation=gas_saturation,
            drainage_curve_value=gas_relative_permeability_drainage,
            imbibition_curve_value=gas_relative_permeability_imbibition,
            drainage_curve_derivative=gas_relative_permeability_drainage_derivative,
            imbibition_curve_derivative=gas_relative_permeability_imbibition_derivative,
            reversal_saturation=gas_reversal_saturation,  # type: ignore
            maximum_historical_saturation=maximum_gas_saturation,
            is_imbibition=gas_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )
        oil_relative_permeability_gas_derivative = compute_killough_scanning_curve_derivative(
            saturation=gas_saturation,
            drainage_curve_value=oil_relative_permeability_gas_drainage,
            imbibition_curve_value=oil_relative_permeability_gas_imbibition,
            drainage_curve_derivative=oil_relative_permeability_gas_drainage_derivative,
            imbibition_curve_derivative=oil_relative_permeability_gas_imbibition_derivative,
            reversal_saturation=gas_reversal_saturation,  # type: ignore
            maximum_historical_saturation=maximum_gas_saturation,
            is_imbibition=gas_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )

        gas_relative_permeability = compute_killough_scanning_curve(
            saturation=gas_saturation,
            drainage_curve_value=gas_relative_permeability_drainage,
            imbibition_curve_value=gas_relative_permeability_imbibition,
            reversal_saturation=gas_reversal_saturation,  # type: ignore
            maximum_historical_saturation=maximum_gas_saturation,
            is_imbibition=gas_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )
        oil_relative_permeability_gas = compute_killough_scanning_curve(
            saturation=gas_saturation,
            drainage_curve_value=oil_relative_permeability_gas_drainage,
            imbibition_curve_value=oil_relative_permeability_gas_imbibition,
            reversal_saturation=gas_reversal_saturation,  # type: ignore
            maximum_historical_saturation=maximum_gas_saturation,
            is_imbibition=gas_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )
        oil_relperm_endpoint = self.get_oil_relperm_endpoint()

        # Three-phase oil mixing rule - chain rule
        three_phase_mixing_rule = typing.cast(MixingRule, self.mixing_rule)
        mixing_rule_partial_derivatives = get_mixing_rule_partial_derivatives(
            rule=three_phase_mixing_rule,
            kro_w=oil_relative_permeability_water,
            kro_g=oil_relative_permeability_gas,
            krw=water_relative_permeability,
            krg=gas_relative_permeability,
            kr_max=oil_relperm_endpoint,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            epsilon=c.FINITE_DIFFERENCE_EPSILON,
        )

        oil_relative_permeability_derivative_water = (
            mixing_rule_partial_derivatives["dkro_dkro_w"]
            * oil_relative_permeability_water_derivative
            + mixing_rule_partial_derivatives["dkro_dsw_explicit"]
        )
        oil_relative_permeability_derivative_oil = mixing_rule_partial_derivatives[
            "dkro_dso_explicit"
        ]
        oil_relative_permeability_derivative_gas = (
            mixing_rule_partial_derivatives["dkro_dkro_g"]
            * oil_relative_permeability_gas_derivative
            + mixing_rule_partial_derivatives["dkro_dsg_explicit"]
        )

        if is_scalar:
            return RelativePermeabilityDerivatives(
                dkrw_dsw=water_relative_permeability_derivative.item(),  # type: ignore
                dkro_dsw=oil_relative_permeability_derivative_water.item(),  # type: ignore
                dkrg_dsw=0.0,
                dkrw_dso=0.0,
                dkro_dso=oil_relative_permeability_derivative_oil.item(),  # type: ignore
                dkrg_dso=0.0,
                dkrw_dsg=0.0,
                dkro_dsg=oil_relative_permeability_derivative_gas.item(),  # type: ignore
                dkrg_dsg=gas_relative_permeability_derivative.item(),  # type: ignore
            )

        return RelativePermeabilityDerivatives(
            dkrw_dsw=water_relative_permeability_derivative,
            dkro_dsw=oil_relative_permeability_derivative_water,
            dkrg_dsw=zeros.copy(),
            dkrw_dso=zeros.copy(),
            dkro_dso=oil_relative_permeability_derivative_oil,
            dkrg_dso=zeros.copy(),
            dkrw_dsg=zeros.copy(),
            dkro_dsg=oil_relative_permeability_derivative_gas,
            dkrg_dsg=gas_relative_permeability_derivative,
        )
