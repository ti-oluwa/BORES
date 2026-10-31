"""The base class and curve queries shared by the capillary pressure hysteresis models."""

import typing

import attrs

from bores.blackoil.satfunc.capillary_pressure.tables import (
    CapillaryPressureTable,
    ThreePhaseCapillaryPressureTable,
    TwoPhaseCapillaryPressureTable,
    query_gas_oil_capillary_pressure,
    query_oil_water_capillary_pressure,
)
from bores.errors import ValidationError
from bores.types import (
    Boolean,
    CapillaryPressureDerivatives,
    CapillaryPressures,
    FluidPhase,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
)

__all__ = [
    "AnyCapillaryTable",
    "HysteresisCapillaryPressureTable",
    "evaluate_three_phase_table",
    "query_gas_oil_curve",
    "query_oil_water_curve",
]

AnyCapillaryTable = TwoPhaseCapillaryPressureTable | CapillaryPressureTable


def query_oil_water_curve(
    table: AnyCapillaryTable,
    *,
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    with_derivatives: Boolean,
    memo: dict[int, typing.Any],
    **kwargs: typing.Any,
) -> tuple[typing.Any, typing.Any]:
    """
    Oil-water capillary pressure of a table and its slope with respect to water saturation.

    A two-phase table is queried at the oil-water system's own saturations (`So = 1 - Sw`),
    whichever phase its reference axis uses. A three-phase table is evaluated at the three-phase
    point, and `memo` lets a table that serves both systems be evaluated once.

    :param table: Two-phase or three-phase capillary pressure table.
    :param water_saturation: Flat water saturation array.
    :param oil_saturation: Flat oil saturation array.
    :param gas_saturation: Flat gas saturation array.
    :param with_derivatives: Whether to compute the slope. It is `None` otherwise.
    :param memo: Evaluations of three-phase tables already made in this call, by table id.
    :param kwargs: Extra arguments forwarded to a three-phase table.
    :return: Tuple of `Pcow` and `dPcow/dSw`.
    """
    if isinstance(table, TwoPhaseCapillaryPressureTable):
        value = query_oil_water_capillary_pressure(table=table, water_saturation=water_saturation)
        if not with_derivatives:
            return value, None
        return value, query_oil_water_capillary_pressure(
            table=table, water_saturation=water_saturation, derivative=True
        )
    values, slopes = evaluate_three_phase_table(
        table=table,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        with_derivatives=with_derivatives,
        memo=memo,
        **kwargs,
    )
    return values["oil_water"], None if slopes is None else slopes["dpcow_dsw"]


def query_gas_oil_curve(
    table: AnyCapillaryTable,
    *,
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    connate_water_saturation: Number,
    with_derivatives: Boolean,
    memo: dict[int, typing.Any],
    **kwargs: typing.Any,
) -> tuple[typing.Any, typing.Any]:
    """
    Gas-oil capillary pressure of a table and its slope with respect to gas saturation.

    A two-phase table is queried at the gas-oil system's own saturations
    (`So = 1 - swc - Sg`), whichever phase its reference axis uses. A three-phase table is
    evaluated at the three-phase point, and `memo` lets a table that serves both systems be
    evaluated once.

    :param table: Two-phase or three-phase capillary pressure table.
    :param water_saturation: Flat water saturation array.
    :param oil_saturation: Flat oil saturation array.
    :param gas_saturation: Flat gas saturation array.
    :param connate_water_saturation: Connate water saturation, which places the gas-oil system.
    :param with_derivatives: Whether to compute the slope. It is `None` otherwise.
    :param memo: Evaluations of three-phase tables already made in this call, by table id.
    :param kwargs: Extra arguments forwarded to a three-phase table.
    :return: Tuple of `Pcgo` and `dPcgo/dSg`.
    """
    if isinstance(table, TwoPhaseCapillaryPressureTable):
        value = query_gas_oil_capillary_pressure(
            table=table,
            gas_saturation=gas_saturation,
            connate_water_saturation=connate_water_saturation,
        )
        if not with_derivatives:
            return value, None
        return value, query_gas_oil_capillary_pressure(
            table=table,
            gas_saturation=gas_saturation,
            connate_water_saturation=connate_water_saturation,
            derivative=True,
        )
    values, slopes = evaluate_three_phase_table(
        table=table,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        with_derivatives=with_derivatives,
        memo=memo,
        **kwargs,
    )
    return values["gas_oil"], None if slopes is None else slopes["dpcgo_dsg"]


def evaluate_three_phase_table(
    table: CapillaryPressureTable,
    *,
    water_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    with_derivatives: Boolean,
    memo: dict[int, typing.Any],
    **kwargs: typing.Any,
) -> tuple[CapillaryPressures, CapillaryPressureDerivatives | None]:
    """
    Evaluate a three-phase capillary pressure table, once per call for a table used by both systems.

    The values and derivatives come from a single `evaluate_with_derivatives` call when the
    derivatives are wanted, which shares work inside models that can.
    """
    cached = memo.get(id(table), None)
    if cached is not None and (cached[1] is not None or not with_derivatives):
        return cached
    arguments = dict(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    if with_derivatives:
        result = table.evaluate_with_derivatives(**arguments)
    else:
        result = (table.evaluate(**arguments), None)
    memo[id(table)] = result
    return result


@attrs.frozen(slots=True)
class HysteresisCapillaryPressureTable(CapillaryPressureTable):
    """
    The plumbing every capillary pressure hysteresis model shares: the drainage and imbibition
    tables of both sub-systems, the wetting phase roles, and the `evaluate`, `derivatives` and
    `evaluate_with_derivatives` entry points.

    A model supplies `compute`, which returns the capillary pressures and optionally their
    derivatives for the saturation history passed as keyword arguments.
    """

    oil_water_drainage_table: TwoPhaseCapillaryPressureTable | CapillaryPressureTable
    """Primary drainage capillary pressure table for the oil-water system."""

    gas_oil_drainage_table: TwoPhaseCapillaryPressureTable | CapillaryPressureTable
    """Primary drainage capillary pressure table for the gas-oil system."""

    oil_water_imbibition_table: TwoPhaseCapillaryPressureTable | CapillaryPressureTable | None = (
        None
    )
    """Primary imbibition Pc table for the oil-water system. Defaults to the drainage table."""

    gas_oil_imbibition_table: TwoPhaseCapillaryPressureTable | CapillaryPressureTable | None = None
    """Primary imbibition Pc table for the gas-oil system. Defaults to the drainage table."""

    supports_vector: bool = attrs.field(init=False, repr=False, default=True)

    def __attrs_post_init__(self) -> None:
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
                isinstance(table, TwoPhaseCapillaryPressureTable)
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

    def get_connate_water_saturation(self) -> Number:
        """
        Connate water saturation of the system, read off the oil-water drainage table. It
        places the gas-oil system (`So = 1 - swc - Sg`) when the gas-oil table is a
        two-phase table.

        :return: Connate water saturation.
        """
        table = self.oil_water_drainage_table
        if isinstance(table, TwoPhaseCapillaryPressureTable | ThreePhaseCapillaryPressureTable):
            return table.get_connate_water_saturation()
        return table.irreducible_water_saturation  # type: ignore[attr-defined]

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        """
        Return the wetting phase for the gas-oil sub-system.

        :return: `FluidPhase.OIL` for oil-wet or `FluidPhase.GAS` for
            gas-wet systems, as reported by the drainage table.
        """
        return self.gas_oil_drainage_table.get_gas_oil_wetting_phase()

    def compute(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        *,
        with_derivatives: Boolean,
        **kwargs: typing.Any,
    ) -> tuple[CapillaryPressures, CapillaryPressureDerivatives | None]:
        """
        Capillary pressures, and optionally their derivatives, in one pass.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param with_derivatives: Whether to also compute the derivatives.
        :param kwargs: The saturation history and any arguments the backing tables accept.
        :return: The capillary pressures and, when requested, their derivatives
            (otherwise `None`).
        """
        raise NotImplementedError

    @staticmethod
    def is_distinct_imbibition(
        imbibition_table: AnyCapillaryTable | None, drainage_table: AnyCapillaryTable
    ) -> bool:
        """Whether the imbibition curve can differ from the drainage curve."""
        return imbibition_table is not None and imbibition_table is not drainage_table

    def evaluate(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute three-phase capillary pressures with Killough hysteresis.

        Without saturation history the result is the primary drainage capillary pressure.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param kwargs: History arguments (`max_water_saturation`, `max_gas_saturation`,
            `water_imbibition_flag`, `gas_imbibition_flag`, `water_reversal_saturation`,
            `gas_reversal_saturation`) and any other arguments the backing tables accept.
        :return: `CapillaryPressures` dictionary with keys `"oil_water"` and `"gas_oil"`.
        """
        pressures, _ = self.compute(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            with_derivatives=False,
            **kwargs,
        )
        return pressures

    def derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> CapillaryPressureDerivatives:
        """
        Compute partial derivatives of capillary pressures with Killough hysteresis.

        Returns the following non-zero entries.

        - `dpcow_dsw`: dpcow/dsw - oil-water Pc scanned over Sw.
        - `dpcow_dso`: zero - Pcow does not depend directly on So in this model.
        - `dpcgo_dsg`: dpcgo/dsg - gas-oil Pc scanned over Sg.
        - `dpcgo_dso`: zero - Pcgo does not depend directly on So.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param kwargs: Same history and backing table arguments as `evaluate`.
        :return: `CapillaryPressureDerivatives` dictionary containing
            `dpcow_dsw`, `dpcow_dso`, `dpcgo_dsg`, and `dpcgo_dso`.
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
    ) -> tuple[CapillaryPressures, CapillaryPressureDerivatives]:
        """
        Compute the capillary pressures and their derivatives in one pass.

        Every bounding curve is evaluated once, where calling `evaluate` and `derivatives`
        separately would evaluate each twice.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param kwargs: Same history and backing table arguments as `evaluate`.
        :return: Tuple of the `CapillaryPressures` and `CapillaryPressureDerivatives`.
        """
        pressures, derivatives = self.compute(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            with_derivatives=True,
            **kwargs,
        )
        assert derivatives is not None
        return pressures, derivatives
