"""The base class and curve queries shared by the relative permeability hysteresis models."""

import typing

import attrs
import numpy as np

from bores.blackoil.satfunc.relperm.hysteresis.land import (
    land_residual_from_reversal,
)
from bores.blackoil.satfunc.relperm.tables import (
    MixingRule,
    RelativePermeabilityTable,
    ThreePhaseRelPermTable,
    TwoPhaseRelPermTable,
    get_mixing_rule,
    query_gas_oil_system,
    query_oil_water_system,
    resolve_residual_oil_saturation_gas,
)
from bores.errors import ValidationError
from bores.types import (
    Boolean,
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

__all__ = ["HysteresisRelPermTable", "query_gas_oil_curves", "query_oil_water_curves"]


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


def uses_tabular_endpoints(table: RelativePermeabilityTable) -> bool:
    """
    Whether a table's curves have fixed endpoints, so Land's trapped saturation cannot be handed to
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

    tabular_land_trapping: bool = True
    """
    Whether Land trapping also reaches tabular imbibition curves, in addition to parametric ones.

    A parametric curve takes the trapped saturation as an endpoint. A tabular curve has fixed
    endpoints, so its saturation axis is rescaled instead, until the curve ends at the trapped
    saturation and starts from the reversal saturation. When this is off a tabular imbibition
    curve is used as tabulated, whatever the maximum residual saturations are.
    """

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
        **kwargs: typing.Any,
    ) -> tuple[RelativePermeabilities, RelativePermeabilityDerivatives | None]:
        """
        Relative permeabilities, and optionally their nine derivatives, in one pass.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param with_derivatives: Whether to also compute the derivatives.
        :param kwargs: The saturation history and any arguments the backing tables accept.
        :return: The relative permeabilities and, when requested, their derivatives
            (otherwise `None`).
        """
        raise NotImplementedError

    def land_residual_oil_water(
        self,
        *,
        size: Integer,
        gas_saturation: NumberArray[OneDimension],
        water_reversal: NumberArray[OneDimension],
    ) -> NumberArray[OneDimension] | None:
        """
        Land residual oil saturation of the oil-water system for every cell, or `None` when no
        maximum residual oil saturation is configured for it.
        """
        maximum_residual = self.maximum_residual_oil_saturation_water
        if maximum_residual is None:
            return None
        residual = np.empty(size, dtype=gas_saturation.dtype)
        land_residual_from_reversal(
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

    def land_residual_gas(
        self,
        *,
        size: Integer,
        water_saturation: NumberArray[OneDimension],
        gas_reversal: NumberArray[OneDimension],
    ) -> NumberArray[OneDimension] | None:
        """
        Land residual (trapped) gas saturation for every cell, or `None` when no maximum residual
        gas saturation is configured.
        """
        maximum_residual = self.maximum_residual_gas_saturation
        if maximum_residual is None:
            return None
        residual = np.empty(size, dtype=water_saturation.dtype)
        land_residual_from_reversal(
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

    def land_residual_oil_gas(
        self,
        *,
        size: Integer,
        water_saturation: NumberArray[OneDimension],
        gas_reversal: NumberArray[OneDimension],
    ) -> NumberArray[OneDimension] | None:
        """
        Land residual oil saturation of the gas-oil system for every cell, or `None` when no
        maximum residual oil saturation is configured for it.
        """
        maximum_residual = self.maximum_residual_oil_saturation_gas
        if maximum_residual is None:
            return None
        residual = np.empty(size, dtype=water_saturation.dtype)
        land_residual_from_reversal(
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

        Only a parametric curve takes its endpoints as overrides, so a tabular table and a model
        without a configured maximum residual get `kwargs` back as it is. A tabular table has its
        axis rescaled instead, see `tabular_land_trapping`.
        """
        if uses_tabular_endpoints(table):
            return kwargs
        residual = self.land_residual_oil_water(
            size=size, gas_saturation=gas_saturation, water_reversal=water_reversal
        )
        if residual is None:
            return kwargs
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

        Same rule as `land_override_kwargs_water`: a tabular table, or a model with no configured
        maximum residual, leaves the entries out.
        """
        if uses_tabular_endpoints(table):
            return kwargs
        overrides: dict[str, typing.Any] = {}
        residual_gas = self.land_residual_gas(
            size=size, water_saturation=water_saturation, gas_reversal=gas_reversal
        )
        if residual_gas is not None:
            overrides["residual_gas_saturation"] = residual_gas
        residual_oil = self.land_residual_oil_gas(
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
