"""Base class and curve queries shared by the relative permeability hysteresis models."""

import math
import typing

import attrs
import numpy as np

from bores.blackoil.satfunc.relperm.hysteresis.kernels import (
    CurveSet,
    HistoryArrays,
    ScanOutputs,
    normalise_saturations,
    normalise_saturations_parallel,
    saturations_need_normalising,
    scan_relative_permeabilities,
    scan_relative_permeabilities_parallel,
)
from bores.blackoil.satfunc.relperm.hysteresis.land import (
    compute_land_residual_from_reversal,
)
from bores.blackoil.satfunc.relperm.mixing_rules import get_builtin_rule_id
from bores.blackoil.satfunc.relperm.tables import (
    MixingRule,
    RelativePermeabilityTable,
    ThreePhaseRelPermTable,
    TwoPhaseRelPermTable,
    get_mixing_rule,
    get_mixing_rule_partial_derivatives,
    query_gas_oil_system,
    query_oil_water_system,
    resolve_residual_oil_saturation_gas,
)
from bores.blackoil.satfunc.utils import (
    PARALLEL_SIZE_THRESHOLD,
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
    "HysteresisRelPermTable",
    "TwoPhaseRelPerms",
    "query_gas_oil_curves",
    "query_oil_water_curves",
]


class TwoPhaseRelPerms(typing.NamedTuple):
    """
    Relative permeabilities of the two phases of one two-phase system, and their slopes.

    The wetting phase comes first. The slopes are with respect to the saturation the system is
    read over: water saturation for the oil-water system, gas saturation for the gas-oil system.
    """

    wetting_relperm: NumberOrArray[OneDimension]
    """Relative permeability of the wetting phase: water, or oil in the gas-oil system."""

    non_wetting_relperm: NumberOrArray[OneDimension]
    """Relative permeability of the non-wetting phase: oil, or gas in the gas-oil system."""

    wetting_relperm_slope: NumberOrArray[OneDimension] | None
    """Slope of `wetting_relperm`, or `None` when slopes were not asked for."""

    non_wetting_relperm_slope: NumberOrArray[OneDimension] | None
    """Slope of `non_wetting_relperm`, or `None` when slopes were not asked for."""


def query_oil_water_curves(
    table: RelativePermeabilityTable,
    *,
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    with_derivatives: Boolean,
    **kwargs: typing.Any,
) -> TwoPhaseRelPerms:
    """
    Returns water and oil relative permeabilities of an oil-water table, with their slopes with respect to
    water saturation.

    A two-phase table is queried at the oil-water system's own saturations, `So = 1 - Sw`,
    whichever phase its reference axis uses, the same way `ThreePhaseRelPermTable` does.

    :param table: Oil-water relative permeability table (two-phase or three-phase).
    :param water_saturation: Flat water saturation array.
    :param oil_saturation: Flat oil saturation array.
    :param gas_saturation: Flat gas saturation array.
    :param with_derivatives: Whether to compute the slopes. They are `None` otherwise.
    :param kwargs: Extra arguments forwarded to a three-phase table.
    :return: `krw`, `kro`, `dkrw/dSw` and `dkro/dSw`, wetting phase first.
    """
    if isinstance(table, TwoPhaseRelPermTable):
        krw, kro = query_oil_water_system(table=table, water_saturation=water_saturation)
        if not with_derivatives:
            return TwoPhaseRelPerms(krw, kro, None, None)

        dkrw, dkro = query_oil_water_system(
            table=table, water_saturation=water_saturation, derivative=True
        )
        return TwoPhaseRelPerms(krw, kro, dkrw, dkro)

    if with_derivatives:
        values, slopes = table.evaluate_with_derivatives(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        return TwoPhaseRelPerms(
            values["water"],
            values["oil"],
            typing.cast("NumberOrArray[OneDimension]", slopes["dkrw_dsw"]),
            typing.cast("NumberOrArray[OneDimension]", slopes["dkro_dsw"]),
        )

    values = table.evaluate(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    return TwoPhaseRelPerms(values["water"], values["oil"], None, None)


def query_gas_oil_curves(
    table: RelativePermeabilityTable,
    *,
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    connate_water_saturation: Number,
    with_derivatives: Boolean,
    **kwargs: typing.Any,
) -> TwoPhaseRelPerms:
    """
    Returns oil and gas relative permeabilities of a gas-oil table, with their slopes with respect to gas
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
    :return: `kro`, `krg`, `dkro/dSg` and `dkrg/dSg`, wetting phase first.
    """
    if isinstance(table, TwoPhaseRelPermTable):
        krg, kro = query_gas_oil_system(
            table=table,
            gas_saturation=gas_saturation,
            connate_water_saturation=connate_water_saturation,
        )
        if not with_derivatives:
            return TwoPhaseRelPerms(kro, krg, None, None)

        dkrg, dkro = query_gas_oil_system(
            table=table,
            gas_saturation=gas_saturation,
            connate_water_saturation=connate_water_saturation,
            derivative=True,
        )
        return TwoPhaseRelPerms(kro, krg, dkro, dkrg)

    if with_derivatives:
        values, slopes = table.evaluate_with_derivatives(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        return TwoPhaseRelPerms(
            values["oil"],
            values["gas"],
            typing.cast("NumberOrArray[OneDimension]", slopes["dkro_dsg"]),
            typing.cast("NumberOrArray[OneDimension]", slopes["dkrg_dsg"]),
        )

    values = table.evaluate(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    return TwoPhaseRelPerms(values["oil"], values["gas"], None, None)


class ScanInputs(typing.NamedTuple):
    """Flat, normalised saturations and history of one `compute` call, ready for the curve queries."""

    is_scalar: Boolean
    """Whether every saturation given was a scalar, so the results are returned as scalars."""

    shape: tuple[int, ...]
    """Broadcast shape of the inputs."""

    size: Integer
    """Number of cells."""

    sw: NumberArray[OneDimension]
    """Flat, normalized water saturations."""

    so: NumberArray[OneDimension]
    """Flat, normalized oil saturations."""

    sg: NumberArray[OneDimension]
    """Flat, normalized gas saturations."""

    placeholder: NumberArray[OneDimension]
    """One-element array standing in for anything a pass does not need."""

    use_water_hysteresis: Boolean
    """Whether water saturation history enables an oil-water scan."""

    use_gas_hysteresis: Boolean
    """Whether gas saturation history enables a gas-oil scan."""

    gas_is_non_wetting: Boolean
    """Whether gas is the non-wetting phase of the gas-oil system."""

    water_reversal: NumberArray[OneDimension]
    """Water saturation at the last reversal, or the maximum when unspecified."""

    gas_reversal: NumberArray[OneDimension]
    """Gas saturation at the last reversal, or the maximum when unspecified."""

    minimum_water: NumberArray[OneDimension]
    """Historical minimum water saturation per cell, or the placeholder when it was not given."""


class BoundingCurves(typing.NamedTuple):
    """
    Values and slopes of the two-phase curves on both bounding curves, as the table queries return
    them. The slopes are `None` when derivatives were not requested.
    """

    krw_d: NumberOrArray[OneDimension]
    """Water relative permeability on the oil-water drainage curve."""

    krw_i: NumberOrArray[OneDimension]
    """Water relative permeability on the oil-water imbibition curve."""

    kro_water_d: NumberOrArray[OneDimension]
    """Oil relative permeability on the oil-water drainage curve."""

    kro_water_i: NumberOrArray[OneDimension]
    """Oil relative permeability on the oil-water imbibition curve."""

    kro_gas_d: NumberOrArray[OneDimension]
    """Oil relative permeability on the gas-oil drainage curve."""

    kro_gas_i: NumberOrArray[OneDimension]
    """Oil relative permeability on the gas-oil imbibition curve."""

    krg_d: NumberOrArray[OneDimension]
    """Gas relative permeability on the gas-oil drainage curve."""

    krg_i: NumberOrArray[OneDimension]
    """Gas relative permeability on the gas-oil imbibition curve."""

    dkrw_d: NumberOrArray[OneDimension] | None
    """Water drainage-curve slope with respect to water saturation, or `None` if not requested."""

    dkrw_i: NumberOrArray[OneDimension] | None
    """Water imbibition-curve slope with respect to water saturation, or `None` if not requested."""

    dkro_water_d: NumberOrArray[OneDimension] | None
    """Oil-water drainage-curve slope with respect to water saturation, or `None` if not requested."""

    dkro_water_i: NumberOrArray[OneDimension] | None
    """Oil-water imbibition-curve slope with respect to water saturation, or `None` if not requested."""

    dkro_gas_d: NumberOrArray[OneDimension] | None
    """Gas-oil drainage-curve oil slope with respect to gas saturation, or `None` if not requested."""

    dkro_gas_i: NumberOrArray[OneDimension] | None
    """Gas-oil imbibition-curve oil slope with respect to gas saturation, or `None` if not requested."""

    dkrg_d: NumberOrArray[OneDimension] | None
    """Gas drainage-curve slope with respect to gas saturation, or `None` if not requested."""

    dkrg_i: NumberOrArray[OneDimension] | None
    """Gas imbibition-curve slope with respect to gas saturation, or `None` if not requested."""


def uses_tabular_endpoints(table: RelativePermeabilityTable) -> bool:
    """
    Checks whether a table's curves have fixed endpoints, so Land's trapped saturation cannot be handed to
    it as an override and its saturation axis has to be rescaled instead.
    """
    return isinstance(table, (TwoPhaseRelPermTable, ThreePhaseRelPermTable))


@attrs.frozen(slots=True)
class HysteresisRelPermTable(RelativePermeabilityTable):
    """
    The plumbing every relative permeability hysteresis model shares: the drainage and imbibition
    tables of both sub-systems, Land trapping of the non-wetting phase, the endpoints, and the
    `evaluate`, `derivatives` and `evaluate_with_derivatives` entry points.

    A model supplies `compute`, which returns the relative permeabilities and optionally their
    derivatives for the saturation history passed as keyword arguments.
    """

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

    maximum_residual_oil_saturation_water: Number | None = None
    """
    Maximum residual oil saturation S_r_max used by Land's formula for the
    oil-water system. Required when `oil_water_imbibition_table` is set.
    """

    maximum_residual_oil_saturation_gas: Number | None = None
    """
    Maximum residual oil saturation S_r_max for the gas-oil system. Required
    when `gas_oil_imbibition_table` is set.
    """

    maximum_residual_gas_saturation: Number | None = None
    """
    Maximum residual gas saturation S_r_max used by Land's formula. Required
    when `gas_oil_imbibition_table` is set.
    """

    tabular_land_trapping: Boolean = True
    """
    Whether Land trapping also reaches tabular imbibition curves, in addition to parametric ones.

    A parametric curve takes the trapped saturation as an endpoint. A tabular curve has fixed
    endpoints, so its saturation axis is rescaled instead, until the curve ends at the trapped
    saturation and starts from the reversal saturation. When this is off a tabular imbibition
    curve is used as tabulated, whatever the maximum residual saturations are.
    """

    mixing_rule: MixingRule | str = "baker_linear_rule"
    """Three-phase oil relative permeability mixing rule."""

    supports_vector: Boolean = attrs.field(init=False, repr=False, default=True)

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
        Returns the wetting phase for the oil-water sub-system.

        :return: `FluidPhase.WATER` for water-wet or `FluidPhase.OIL` for
            oil-wet systems, as reported by the drainage table.
        """
        return self.oil_water_drainage_table.get_oil_water_wetting_phase()

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        """
        Returns the wetting phase for the gas-oil sub-system.

        :return: `FluidPhase.OIL` for oil-wet or `FluidPhase.GAS` for
            gas-wet systems, as reported by the drainage table.
        """
        return self.gas_oil_drainage_table.get_gas_oil_wetting_phase()

    def get_oil_relperm_endpoint(self) -> Number:
        """Returns oil relative permeability endpoint from the primary drainage table."""
        return self.oil_water_drainage_table.get_oil_relperm_endpoint()

    def get_water_relperm_endpoint(self) -> Number:
        """Returns water relative permeability endpoint from the primary drainage table."""
        return self.oil_water_drainage_table.get_water_relperm_endpoint()

    def get_gas_relperm_endpoint(self) -> Number:
        """Returns gas relative permeability endpoint from the primary drainage table."""
        return self.gas_oil_drainage_table.get_gas_relperm_endpoint()

    def get_connate_water_saturation(self) -> Number:
        """Returns connate water saturation from the primary drainage table. Hysteresis does not move it."""
        return self.oil_water_drainage_table.get_connate_water_saturation()

    def get_residual_oil_saturation_water(self) -> Number:
        """
        Returns residual oil saturation to water flooding that the primary drainage table declares.

        This is the static value of the table, not the Land-trapped value used while an
        imbibition scan is active.
        """
        return self.oil_water_drainage_table.get_residual_oil_saturation_water()

    def get_residual_oil_saturation_gas(self) -> Number:
        """
        Returns residual oil saturation to gas flooding that the primary drainage table declares, taking
        the connate water saturation of the system into account.
        """
        return resolve_residual_oil_saturation_gas(
            gas_oil_table=self.gas_oil_drainage_table,
            connate_water_saturation=self.get_connate_water_saturation(),
        )

    def get_residual_gas_saturation(self) -> Number:
        """Returns residual (critical) gas saturation that the primary drainage table declares."""
        return self.gas_oil_drainage_table.get_residual_gas_saturation()

    def compute(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        *,
        with_derivatives: Boolean,
        **kwargs: typing.Any,
    ) -> tuple[RelativePermeabilities, RelativePermeabilityDerivatives | None]:
        """
        Returns relative permeabilities, and optionally their nine derivatives, in one pass.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param with_derivatives: Whether to also compute the derivatives.
        :param kwargs: The saturation history and any arguments the backing tables accept.
        :return: The relative permeabilities and, when requested, their derivatives
            (otherwise `None`).
        """
        raise NotImplementedError

    def prepare_scan_inputs(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        *,
        max_water_saturation: NumberOrArray[NDimension] | None,
        max_gas_saturation: NumberOrArray[NDimension] | None,
        water_imbibition_flag: Boolean | BooleanArray[NDimension] | None,
        gas_imbibition_flag: Boolean | BooleanArray[NDimension] | None,
        water_reversal_saturation: NumberOrArray[NDimension] | None,
        gas_reversal_saturation: NumberOrArray[NDimension] | None,
        min_water_saturation: NumberOrArray[NDimension] | None = None,
    ) -> tuple[ScanInputs, HistoryArrays]:
        """
        Broadcasts, flatten and normalise the saturations and history of one `compute` call.

        :returns: The flat inputs, and the flat saturation history the scan kernel reads.
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
            if min_water_saturation is not None:
                inputs.append(min_water_saturation)
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

        minimum_water = (
            flatten_kernel_input(value=min_water_saturation, shape=flat_shape)
            if use_water_hysteresis and min_water_saturation is not None
            else placeholder
        )

        history = HistoryArrays(
            max_water_saturation=typing.cast("NumberArray[OneDimension]", maximum_water),
            max_gas_saturation=typing.cast("NumberArray[OneDimension]", maximum_gas),
            water_imbibition_flag=typing.cast("NumberArray[OneDimension]", water_flag),
            gas_imbibition_flag=typing.cast("NumberArray[OneDimension]", gas_flag),
            water_reversal_saturation=typing.cast("NumberArray[OneDimension]", water_reversal),
            gas_reversal_saturation=typing.cast("NumberArray[OneDimension]", gas_reversal),
            gas_trapped_saturation=typing.cast("NumberArray[OneDimension]", placeholder),
        )
        gas_is_non_wetting = self.get_gas_oil_wetting_phase() != FluidPhase.GAS
        inputs = ScanInputs(
            is_scalar=is_scalar,
            shape=shape,
            size=size,
            sw=sw,
            so=so,
            sg=sg,
            placeholder=typing.cast("NumberArray[OneDimension]", placeholder),
            use_water_hysteresis=use_water_hysteresis,
            use_gas_hysteresis=use_gas_hysteresis,
            gas_is_non_wetting=gas_is_non_wetting,
            water_reversal=typing.cast("NumberArray[OneDimension]", water_reversal),
            gas_reversal=typing.cast("NumberArray[OneDimension]", gas_reversal),
            minimum_water=typing.cast("NumberArray[OneDimension]", minimum_water),
        )
        return inputs, history

    def scan_bounding_curves(
        self,
        *,
        inputs: ScanInputs,
        history: HistoryArrays,
        curves: BoundingCurves,
        scanning_exponent: Number,
        gas_translated_scanning: Boolean,
        water_translated_scanning: Boolean = False,
        with_derivatives: Boolean,
    ) -> tuple[RelativePermeabilities, RelativePermeabilityDerivatives | None]:
        """
        Blends the bounding curves cell by cell and applies the three-phase mixing rule, in one
        kernel pass, then shape the results like the inputs.

        :param inputs: The flat inputs from `prepare_scan_inputs`.
        :param history: The flat saturation history the weights are read from.
        :param curves: Both bounding curves of every two-phase system, with their slopes.
        :param scanning_exponent: Exponent of the Killough scanning weight. Unused when
            `gas_translated_scanning` is set.
        :param gas_translated_scanning: Whether the gas weight is a step, 1 below the reversal
            saturation and 0 at and above it, as the Carlson translated curve needs.
        :param water_translated_scanning: Whether the water weight, which also weighs the oil of
            the oil-water system, is a step, 1 above the reversal saturation (the minimum water
            saturation) and 0 at and below it, as the Carlson translated curve needs.
        :param with_derivatives: Whether to also compute the derivatives.
        :returns: The relative permeabilities and, when requested, their derivatives.
        """
        is_scalar, shape, size = inputs.is_scalar, inputs.shape, inputs.size
        sw, so, sg = inputs.sw, inputs.so, inputs.sg
        placeholder = inputs.placeholder
        use_water_hysteresis = inputs.use_water_hysteresis
        use_gas_hysteresis = inputs.use_gas_hysteresis
        gas_is_non_wetting = inputs.gas_is_non_wetting
        (
            krw_d,
            krw_i,
            kro_water_d,
            kro_water_i,
            kro_gas_d,
            kro_gas_i,
            krg_d,
            krg_i,
            dkrw_d,
            dkrw_i,
            dkro_water_d,
            dkro_water_i,
            dkro_gas_d,
            dkro_gas_i,
            dkrg_d,
            dkrg_i,
        ) = curves

        values = CurveSet(
            krw_drainage=typing.cast("NumberArray[OneDimension]", krw_d),
            krw_imbibition=typing.cast("NumberArray[OneDimension]", krw_i),
            kro_water_drainage=typing.cast("NumberArray[OneDimension]", kro_water_d),
            kro_water_imbibition=typing.cast("NumberArray[OneDimension]", kro_water_i),
            kro_gas_drainage=typing.cast("NumberArray[OneDimension]", kro_gas_d),
            kro_gas_imbibition=typing.cast("NumberArray[OneDimension]", kro_gas_i),
            krg_drainage=typing.cast("NumberArray[OneDimension]", krg_d),
            krg_imbibition=typing.cast("NumberArray[OneDimension]", krg_i),
        )
        if with_derivatives:
            assert all(
                slope is not None
                for slope in (
                    dkrw_d,
                    dkrw_i,
                    dkro_water_d,
                    dkro_water_i,
                    dkro_gas_d,
                    dkro_gas_i,
                    dkrg_d,
                    dkrg_i,
                )
            )
        slopes = CurveSet(
            krw_drainage=(
                typing.cast("NumberArray[OneDimension]", dkrw_d)
                if with_derivatives
                else placeholder
            ),
            krw_imbibition=(
                typing.cast("NumberArray[OneDimension]", dkrw_i)
                if with_derivatives
                else placeholder
            ),
            kro_water_drainage=(
                typing.cast("NumberArray[OneDimension]", dkro_water_d)
                if with_derivatives
                else placeholder
            ),
            kro_water_imbibition=(
                typing.cast("NumberArray[OneDimension]", dkro_water_i)
                if with_derivatives
                else placeholder
            ),
            kro_gas_drainage=(
                typing.cast("NumberArray[OneDimension]", dkro_gas_d)
                if with_derivatives
                else placeholder
            ),
            kro_gas_imbibition=(
                typing.cast("NumberArray[OneDimension]", dkro_gas_i)
                if with_derivatives
                else placeholder
            ),
            krg_drainage=(
                typing.cast("NumberArray[OneDimension]", dkrg_d)
                if with_derivatives
                else placeholder
            ),
            krg_imbibition=(
                typing.cast("NumberArray[OneDimension]", dkrg_i)
                if with_derivatives
                else placeholder
            ),
        )

        dtype = values.krw_drainage.dtype
        if dtype.kind != "f":
            dtype = np.dtype(np.float64)

        rule = typing.cast(MixingRule, self.mixing_rule)
        rule_id = get_builtin_rule_id(rule)
        use_rule_in_kernel = rule_id is not None
        needs_gas_oil = not use_rule_in_kernel
        out = ScanOutputs(
            krw=typing.cast("NumberArray[OneDimension]", np.empty(size, dtype=dtype)),
            kro=typing.cast("NumberArray[OneDimension]", np.empty(size, dtype=dtype)),
            krg=typing.cast("NumberArray[OneDimension]", np.empty(size, dtype=dtype)),
            kro_gas=typing.cast(
                "NumberArray[OneDimension]", np.empty(size if needs_gas_oil else 1, dtype=dtype)
            ),
            dkrw_dsw=typing.cast(
                "NumberArray[OneDimension]", np.empty(size if with_derivatives else 1, dtype=dtype)
            ),
            dkro_dsw=typing.cast(
                "NumberArray[OneDimension]", np.empty(size if with_derivatives else 1, dtype=dtype)
            ),
            dkro_dso=typing.cast(
                "NumberArray[OneDimension]",
                np.empty(size if with_derivatives and use_rule_in_kernel else 1, dtype=dtype),
            ),
            dkro_dsg=typing.cast(
                "NumberArray[OneDimension]", np.empty(size if with_derivatives else 1, dtype=dtype)
            ),
            dkrg_dsg=typing.cast(
                "NumberArray[OneDimension]", np.empty(size if with_derivatives else 1, dtype=dtype)
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
            scanning_exponent=scanning_exponent,
            gas_translated_scanning=gas_translated_scanning,
            water_translated_scanning=water_translated_scanning,
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

        zeros = (
            typing.cast("NumberArray[OneDimension]", np.zeros(size, dtype=dtype))
            if with_derivatives
            else None
        )
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
                dkro_dsw = typing.cast(
                    "NumberArray[OneDimension]",
                    partials["dkro_dkro_w"] * out.dkro_dsw
                    + partials["dkro_dkrw"] * out.dkrw_dsw
                    + partials["dkro_dsw_explicit"],
                )
                dkro_dso = typing.cast("NumberArray[OneDimension]", partials["dkro_dso_explicit"])
                dkro_dsg = typing.cast(
                    "NumberArray[OneDimension]",
                    partials["dkro_dkro_g"] * out.dkro_dsg
                    + partials["dkro_dkrg"] * out.dkrg_dsg
                    + partials["dkro_dsg_explicit"],
                )

        def finish(array: NumberOrArray[OneDimension] | None) -> NumberOrArray[NDimension]:
            assert array is not None
            result = np.asarray(array)
            if result.size == 1 and size > 1:
                result = np.broadcast_to(result, (int(size),))
            result = result.reshape(shape)
            if is_scalar:
                return typing.cast("Number", result.item())
            return typing.cast("NumberArray[NDimension]", result)

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

    def get_land_residual_oil_water(
        self,
        *,
        size: Integer,
        gas_saturation: NumberArray[OneDimension],
        water_reversal: NumberArray[OneDimension],
    ) -> NumberArray[OneDimension] | None:
        """
        Returns Land residual oil saturation of the oil-water system for every cell, or `None` when no
        maximum residual oil saturation is configured for it.
        """
        maximum_residual = self.maximum_residual_oil_saturation_water
        if maximum_residual is None:
            return None

        residual = np.empty(size, dtype=gas_saturation.dtype)
        compute_land_residual_from_reversal(
            size=size,
            reversal_saturation=water_reversal,
            other_saturation=gas_saturation,
            from_remaining_oil=True,
            maximum_residual_saturation=maximum_residual,
            land_trapping_coefficient=self.land_trapping_coefficient_water,
            saturation_epsilon=1e-12,
            out=typing.cast("NumberArray[OneDimension]", residual),
        )
        return typing.cast("NumberArray[OneDimension]", residual)

    def get_land_residual_gas(
        self,
        *,
        size: Integer,
        water_saturation: NumberArray[OneDimension],
        gas_reversal: NumberArray[OneDimension],
    ) -> NumberArray[OneDimension] | None:
        """
        Returns Land residual (trapped) gas saturation for every cell, or `None` when no maximum residual
        gas saturation is configured.
        """
        maximum_residual = self.maximum_residual_gas_saturation
        if maximum_residual is None:
            return None

        residual = np.empty(size, dtype=water_saturation.dtype)
        compute_land_residual_from_reversal(
            size=size,
            reversal_saturation=gas_reversal,
            other_saturation=water_saturation,
            from_remaining_oil=False,
            maximum_residual_saturation=maximum_residual,
            land_trapping_coefficient=self.land_trapping_coefficient_gas,
            saturation_epsilon=1e-12,
            out=typing.cast("NumberArray[OneDimension]", residual),
        )
        return typing.cast("NumberArray[OneDimension]", residual)

    def get_land_residual_oil_gas(
        self,
        *,
        size: Integer,
        water_saturation: NumberArray[OneDimension],
        gas_reversal: NumberArray[OneDimension],
    ) -> NumberArray[OneDimension] | None:
        """
        Returns Land residual oil saturation of the gas-oil system for every cell, or `None` when no
        maximum residual oil saturation is configured for it.
        """
        maximum_residual = self.maximum_residual_oil_saturation_gas
        if maximum_residual is None:
            return None

        residual = np.empty(size, dtype=water_saturation.dtype)
        compute_land_residual_from_reversal(
            size=size,
            reversal_saturation=gas_reversal,
            other_saturation=water_saturation,
            from_remaining_oil=True,
            maximum_residual_saturation=maximum_residual,
            land_trapping_coefficient=self.land_trapping_coefficient_gas,
            saturation_epsilon=1e-12,
            out=typing.cast("NumberArray[OneDimension]", residual),
        )
        return typing.cast("NumberArray[OneDimension]", residual)

    def get_land_override_kwargs_water(
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

        Only a parametric curve takes its endpoints as overrides, so a tabular table and a model
        without a configured maximum residual get `kwargs` back as it is. A tabular table has its
        axis rescaled instead, see `tabular_land_trapping`.
        """
        if uses_tabular_endpoints(table):
            return kwargs

        residual = self.get_land_residual_oil_water(
            size=size, gas_saturation=gas_saturation, water_reversal=water_reversal
        )
        if residual is None:
            return kwargs
        return {**kwargs, "residual_oil_saturation_water": residual}

    def get_land_override_kwargs_gas(
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

        Same rule as `get_land_override_kwargs_water`: a tabular table, or a model with no configured
        maximum residual, leaves the entries out.
        """
        if uses_tabular_endpoints(table):
            return kwargs

        overrides: dict[str, typing.Any] = {}
        residual_gas = self.get_land_residual_gas(
            size=size, water_saturation=water_saturation, gas_reversal=gas_reversal
        )
        if residual_gas is not None:
            overrides["residual_gas_saturation"] = residual_gas

        residual_oil = self.get_land_residual_oil_gas(
            size=size, water_saturation=water_saturation, gas_reversal=gas_reversal
        )
        if residual_oil is not None:
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
        Computes three-phase relative permeabilities with Killough/Land hysteresis.

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
        Computes the nine partial derivatives of the relative permeabilities with
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
        Computes the relative permeabilities and their derivatives in one pass.

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
