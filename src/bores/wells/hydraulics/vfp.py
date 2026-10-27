"""
VFP (vertical flow performance) tables.

Interpolates bottomhole pressure at a well's datum depth as a function of flow rate,
tubing head pressure, water fraction, gas fraction, and artificial lift quantity.

Each table keeps the definitions of its own flow, water and gas axes, and the values it
is looked up with are computed from a well's rates according to them.
This is the datum-to-surface leg of well hydraulics only; connection-level
pressure distribution below the datum is unaffected by whether a table is assigned.
"""

import enum
import logging
import math
import typing

import attrs
import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator  # type: ignore[import-untyped]
from typing_extensions import Self

from bores.deck.file import DeckFile
from bores.deck.keywords.schedule import VFPInjectorDeckTable, VFPProducerDeckTable
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.serde.base import Serializable
from bores.serde.stores.base import StoreSerializable
from bores.types import (
    FiveDimensions,
    FluidPhase,
    Integer,
    NDimension,
    Number,
    NumberArray,
    OneDimension,
    TableQuery,
    TableResult,
    UnitSystem,
)
from bores.units import UnitConversionTable, get_conversion_factors
from bores.utils import scale
from bores.wells.base import WellType
from bores.wells.hydraulics.base import SurfaceSample, WellBoreModel
from bores.wells.hydraulics.dispatch import (
    compute_tubing_head_pressure,
    compute_tubing_head_pressure_from_table,
)
from bores.wells.state import PhaseValues

__all__ = [
    "VFPData",
    "VFPFlowType",
    "VFPGasFractionType",
    "VFPLookupVariables",
    "VFPTable",
    "VFPTables",
    "VFPWaterFractionType",
    "as_vfp_table",
]

logger = logging.getLogger(__name__)

DECK_UNIT_SYSTEMS: typing.Mapping[str, UnitSystem] = {
    "METRIC": UnitSystem.METRIC,
    "FIELD": UnitSystem.FIELD,
    "LAB": UnitSystem.LAB,
    "SI": UnitSystem.SI,
    "PVT-M": UnitSystem.METRIC,
}
"""Deck unit keyword to the unit system a `VFPPROD` / `VFPINJ` table is written in."""


class VFPFlowType(enum.Enum):
    """What a VFP table's flow axis measures, in surface-condition volumes per time."""

    OIL = "oil"
    """Oil rate."""

    LIQUID = "liquid"
    """Oil plus water rate."""

    WATER = "water"
    """Water rate. Only for injector tables."""

    GAS = "gas"
    """Gas rate."""

    def __str__(self) -> str:
        return self.value


class VFPWaterFractionType(enum.Enum):
    """What a producer table's water axis measures."""

    WATER_CUT = "water_cut"
    """Water rate over liquid rate."""

    WATER_OIL_RATIO = "water_oil_ratio"
    """Water rate over oil rate."""

    WATER_GAS_RATIO = "water_gas_ratio"
    """Water rate over gas rate."""

    def __str__(self) -> str:
        return self.value


class VFPGasFractionType(enum.Enum):
    """What a producer table's gas axis measures."""

    GAS_OIL_RATIO = "gas_oil_ratio"
    """Gas rate over oil rate."""

    GAS_LIQUID_RATIO = "gas_liquid_ratio"
    """Gas rate over liquid rate."""

    OIL_GAS_RATIO = "oil_gas_ratio"
    """Oil rate over gas rate."""

    def __str__(self) -> str:
        return self.value


PRODUCER_FLOW_TYPE_MAP: typing.Mapping[str, VFPFlowType] = {
    "OIL": VFPFlowType.OIL,
    "LIQ": VFPFlowType.LIQUID,
    "GAS": VFPFlowType.GAS,
}
"""Deck `VFPPROD` flow axis definition to `VFPFlowType`."""

INJECTOR_FLOW_TYPE_MAP: typing.Mapping[str, VFPFlowType] = {
    "OIL": VFPFlowType.OIL,
    "WAT": VFPFlowType.WATER,
    "GAS": VFPFlowType.GAS,
}
"""Deck `VFPINJ` flow axis definition to `VFPFlowType`."""

WATER_FRACTION_TYPE_MAP: typing.Mapping[str, VFPWaterFractionType] = {
    "WCT": VFPWaterFractionType.WATER_CUT,
    "WOR": VFPWaterFractionType.WATER_OIL_RATIO,
    "WGR": VFPWaterFractionType.WATER_GAS_RATIO,
}
"""Deck water axis definition to `VFPWaterFractionType`."""

GAS_FRACTION_TYPE_MAP: typing.Mapping[str, VFPGasFractionType] = {
    "GOR": VFPGasFractionType.GAS_OIL_RATIO,
    "GLR": VFPGasFractionType.GAS_LIQUID_RATIO,
    "OGR": VFPGasFractionType.OIL_GAS_RATIO,
}
"""Deck gas axis definition to `VFPGasFractionType`."""

AXIS_NAMES = ("flow_rate", "thp", "water_fraction", "gas_fraction", "artificial_lift_quantity")
"""Order `VFPData`'s five axes are always addressed in, internally."""


class VFPLookupVariables(typing.NamedTuple):
    """The values a `VFPTable` is looked up with, computed from a well's rates."""

    flow_rate: Number
    """Value on the table's flow axis."""

    water_fraction: Number
    """Value on the table's water axis."""

    gas_fraction: Number
    """Value on the table's gas axis."""


@attrs.frozen(kw_only=True, slots=True)
class VFPData(Serializable):
    """
    Raw axes and bottomhole-pressure grid for one VFP table.

    A producer table typically varies over all five axes. An injector
    table usually varies only with `flow_rates` and `thps`; leave
    `water_fractions`, `gas_fractions`, and `artificial_lift_quantities` at
    their single-value defaults in that case.

    `flow_type`, `water_fraction_type` and `gas_fraction_type` state what the
    corresponding axes measure. A table is never re-expressed in other terms;
    `VFPTable.get_lookup_variables` computes the values to look it up with from
    a well's rates.
    """

    table_number: Integer
    """
    Table number, matching deck `VFPPROD`/`VFPINJ` item 1. A well
    selects this table through `WCONPROD`/`WCONINJE`'s `vfp_table` item.
    """

    well_type: WellType
    """Whether this is a `VFPPROD` (producer) or `VFPINJ` (injector) table."""

    datum_depth: Number
    """
    Reference depth this table's bottomhole pressure is reported at,
    matching the owning well's `reference_depth`.
    """

    flow_type: VFPFlowType = attrs.field(
        default=attrs.Factory(
            lambda self: (
                VFPFlowType.GAS if self.well_type == WellType.INJECTOR else VFPFlowType.LIQUID
            ),
            takes_self=True,
        )
    )
    """
    What `flow_rates` measures. Oil, liquid or gas for a producer table; for an
    injector table, the injected phase. Liquid for a producer and gas for an
    injector if not given.
    """

    water_fraction_type: VFPWaterFractionType = VFPWaterFractionType.WATER_CUT
    """What `water_fractions` measures. Unused by an injector table."""

    gas_fraction_type: VFPGasFractionType = VFPGasFractionType.GAS_OIL_RATIO
    """What `gas_fractions` measures. Unused by an injector table."""

    flow_rates: NumberArray[OneDimension]
    """Flow rate axis, strictly ascending, in the terms of `flow_type`."""

    thps: NumberArray[OneDimension]
    """Tubing head pressure axis, strictly ascending."""

    water_fractions: NumberArray[OneDimension] = attrs.field(
        factory=lambda: typing.cast(NumberArray[OneDimension], np.array([0.0]))
    )
    """
    Water axis, strictly ascending, in the terms of `water_fraction_type`. A single
    value (the default) for a table with no water dependence.
    """

    gas_fractions: NumberArray[OneDimension] = attrs.field(
        factory=lambda: typing.cast(NumberArray[OneDimension], np.array([0.0]))
    )
    """
    Gas axis, strictly ascending, in the terms of `gas_fraction_type`. A single value
    (the default) for a table with no gas dependence.
    """

    artificial_lift_quantities: NumberArray[OneDimension] = attrs.field(
        factory=lambda: typing.cast(NumberArray[OneDimension], np.array([0.0]))
    )
    """
    Artificial lift quantity axis (gas-lift injection rate, pump
    power, etc.), strictly ascending. A single value (the default) for
    a table with no artificial-lift dependence.
    """

    bhps: NumberArray[FiveDimensions]
    """
    Bottomhole pressure at `datum_depth`, shaped `(len(flow_rates),
    len(thps), len(water_fractions), len(gas_fractions),
    len(artificial_lift_quantities))`.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system every dimensioned field above is expressed in."""

    def __attrs_post_init__(self) -> None:
        axes = (
            self.flow_rates,
            self.thps,
            self.water_fractions,
            self.gas_fractions,
            self.artificial_lift_quantities,
        )
        valid_flow_types = (
            (VFPFlowType.OIL, VFPFlowType.WATER, VFPFlowType.GAS)
            if self.well_type == WellType.INJECTOR
            else (VFPFlowType.OIL, VFPFlowType.LIQUID, VFPFlowType.GAS)
        )
        if self.flow_type not in valid_flow_types:
            raise ValidationError(
                f"A {self.well_type.value} table can't have flow type {self.flow_type}; "
                f"expected one of {[str(flow_type) for flow_type in valid_flow_types]}."
            )

        expected_shape = tuple(len(axis) for axis in axes)
        if self.bhps.shape != expected_shape:
            raise ValidationError(
                f"`bhps` shape {self.bhps.shape} doesn't match the axes "
                f"{dict(zip(AXIS_NAMES, expected_shape, strict=True))}."
            )
        for name, axis in zip(AXIS_NAMES, axes, strict=True):
            if len(axis) == 0:
                raise ValidationError(f"`{name}s` must have at least one value.")
            if len(axis) > 1 and not np.all(np.diff(axis) > 0):
                raise ValidationError(f"`{name}s` must be strictly ascending.")

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Converts every dimensioned axis and `bhps` to a different unit system.

        Each axis is converted according to what it measures: a flow axis by its
        phase, a water-oil ratio or water cut not at all, a water-gas or oil-gas ratio
        by the ratio of surface volumes, and a gas-oil or gas-liquid ratio by the
        gas-oil ratio factor.

        `artificial_lift_quantities` is left unconverted, since its
        physical quantity depends on the lift method and this data
        structure doesn't know which. Build or load `VFPData` already
        in the target unit system if artificial lift matters across a
        unit-system boundary.

        :param target: Target unit system.
        :param table: Optional custom unit-conversion table.
        :returns: This data, with every axis except
            `artificial_lift_quantities`, and `bhps`, converted to `target`.
        """
        if target == self.unit_system:
            return self
        factors = get_conversion_factors(self.unit_system, target, table=table)
        volume_ratio_factor = factors["liquid_surface_volume"] / factors["gas_surface_volume"]
        if self.flow_type == VFPFlowType.GAS:
            flow_factor = factors["gas_surface_rate"]
        else:
            flow_factor = factors["liquid_surface_rate"]

        if self.water_fraction_type == VFPWaterFractionType.WATER_GAS_RATIO:
            water_factor = volume_ratio_factor
        else:
            water_factor = 1.0

        if self.gas_fraction_type == VFPGasFractionType.OIL_GAS_RATIO:
            gas_factor = volume_ratio_factor
        else:
            gas_factor = factors["gas_oil_ratio"]
        return attrs.evolve(
            self,
            flow_rates=scale(self.flow_rates, flow_factor),
            thps=scale(self.thps, factors["pressure"]),
            water_fractions=scale(self.water_fractions, water_factor),
            gas_fractions=scale(self.gas_fractions, gas_factor),
            datum_depth=scale(self.datum_depth, factors["length"]),
            bhps=scale(self.bhps, factors["pressure"]),
            unit_system=target,
        )

    @typing.overload
    @classmethod
    def from_deck(
        cls, deck_file: DeckFile, *, table_number: Integer, well_type: WellType | None = None
    ) -> Self: ...
    @typing.overload
    @classmethod
    def from_deck(
        cls, deck_file: DeckFile, *, table_number: None = None, well_type: WellType | None = None
    ) -> list[Self]: ...

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        table_number: Integer | None = None,
        well_type: WellType | None = None,
    ) -> Self | list[Self]:
        """
        Construct one or all `VFPData` objects from a parsed `DeckFile`.

        Reads the `VFPPROD` and `VFPINJ` keywords (see `load_vfp_data`).

        :param deck_file: Parsed `bores.deck.file.DeckFile`.
        :param table_number: Number of a specific table to extract, or `None` for all.
        :param well_type: Read only producer (`VFPPROD`) or injector (`VFPINJ`) tables. `None`
            reads both, and is then ambiguous only if `table_number` exists under both.
        :returns: A single `VFPData` if `table_number` is given, otherwise a list.
        :raises ValidationError: If the deck has no matching table, `table_number` is not found
            or is ambiguous, or a table cannot be mapped.
        """
        return typing.cast(
            Self | list[Self],
            load_vfp_data(deck_file, table_number=table_number, well_type=well_type),
        )


class VFPTable(StoreSerializable):
    """
    A `VFPData` grid with a pre-built interpolator for bottomhole-pressure lookup.

    Any axis with only one value is collapsed out at construction; the
    interpolator is only built over axes that actually vary.
    """

    __abstract_serializable__ = True

    def __init__(
        self,
        data: VFPData,
        *,
        warn_on_extrapolation: bool = False,
        dtype: npt.DTypeLike = None,
    ) -> None:
        """
        Builds a `VFPTable` from raw `VFPData`.

        :param data: The table's axes and BHP grid.
        :param warn_on_extrapolation: Log a warning when a query falls
            outside the table's bounds on any axis that varies.
        :param dtype: Output array dtype. `bores.precision.get_dtype()`
            if not given.
        :raises ValidationError: If every axis is single-valued.
        """
        self._data = data
        self.warn_on_extrapolation = warn_on_extrapolation
        self.dtype = np.dtype(dtype) if dtype is not None else get_dtype()

        axes = (
            data.flow_rates,
            data.thps,
            data.water_fractions,
            data.gas_fractions,
            data.artificial_lift_quantities,
        )
        self._active = tuple(len(axis) > 1 for axis in axes)
        if not any(self._active):
            raise ValidationError(
                "This table's every axis is single-valued; there's nothing "
                "to interpolate. A VFP table needs at least `flow_rates` to vary."
            )

        active_axes = tuple(
            axis for axis, active in zip(axes, self._active, strict=True) if active
        )
        squeeze_axes = tuple(i for i, active in enumerate(self._active) if not active)
        values = np.asarray(data.bhps, dtype=self.dtype)
        if squeeze_axes:
            values = np.squeeze(values, axis=squeeze_axes)

        self._interpolator = RegularGridInterpolator(
            points=active_axes,
            values=values,  # type: ignore[arg-type]
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        self._extrapolation_bounds: dict[str, tuple[Number, Number]] = {
            name: (axis[0], axis[-1])
            for name, axis, active in zip(AXIS_NAMES, axes, self._active, strict=True)
            if active
        }

    def __dump__(self) -> dict[str, typing.Any]:
        return {
            "data": self._data.dump(),
            "warn_on_extrapolation": self.warn_on_extrapolation,
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        return cls(
            data=VFPData.load(data["data"]),
            warn_on_extrapolation=data.get("warn_on_extrapolation", False),
        )

    @property
    def table_number(self) -> Integer:
        """This table's number, matching deck `VFPPROD`/`VFPINJ` item 1."""
        return self._data.table_number

    @property
    def well_type(self) -> WellType:
        """Whether this is a `VFPPROD` (producer) or `VFPINJ` (injector) table."""
        return self._data.well_type

    @property
    def datum_depth(self) -> Number:
        """Reference depth this table's bottomhole pressure is reported at."""
        return self._data.datum_depth

    @property
    def unit_system(self) -> UnitSystem:
        """Unit system of the underlying table data."""
        return self._data.unit_system

    @property
    def flow_type(self) -> VFPFlowType:
        """What this table's flow axis measures."""
        return self._data.flow_type

    @property
    def water_fraction_type(self) -> VFPWaterFractionType:
        """What this table's water axis measures."""
        return self._data.water_fraction_type

    @property
    def gas_fraction_type(self) -> VFPGasFractionType:
        """What this table's gas axis measures."""
        return self._data.gas_fraction_type

    def get_lookup_variables(
        self,
        *,
        oil_rate: Number,
        water_rate: Number,
        gas_rate: Number,
    ) -> VFPLookupVariables:
        """
        Computes the flow, water and gas values to look this table up with from a well's rates.

        Each value is built from the surface-condition phase rates according to what the
        table's own axes measure. Rates are taken as magnitudes. A ratio whose denominator
        is zero is zero if its numerator is also zero, and otherwise the largest value of
        that axis, so the interpolator never sees an infinite value. An injector table has
        no water or gas dependence, so both are zero.

        :param oil_rate: Surface oil rate.
        :param water_rate: Surface water rate.
        :param gas_rate: Surface gas rate.
        :returns: The `flow_rate`, `water_fraction` and `gas_fraction` to query with.
        """
        oil, water, gas = abs(oil_rate), abs(water_rate), abs(gas_rate)
        liquid = oil + water

        if self.flow_type == VFPFlowType.OIL:
            flow_rate = oil
        elif self.flow_type == VFPFlowType.LIQUID:
            flow_rate = liquid
        elif self.flow_type == VFPFlowType.WATER:
            flow_rate = water
        else:
            flow_rate = gas

        if self.well_type == WellType.INJECTOR:
            return VFPLookupVariables(flow_rate=flow_rate, water_fraction=0.0, gas_fraction=0.0)

        if self.water_fraction_type == VFPWaterFractionType.WATER_CUT:
            water_numerator, water_denominator = water, liquid
        elif self.water_fraction_type == VFPWaterFractionType.WATER_OIL_RATIO:
            water_numerator, water_denominator = water, oil
        else:
            water_numerator, water_denominator = water, gas

        if self.gas_fraction_type == VFPGasFractionType.GAS_OIL_RATIO:
            gas_numerator, gas_denominator = gas, oil
        elif self.gas_fraction_type == VFPGasFractionType.GAS_LIQUID_RATIO:
            gas_numerator, gas_denominator = gas, liquid
        else:
            gas_numerator, gas_denominator = oil, gas

        return VFPLookupVariables(
            flow_rate=flow_rate,
            water_fraction=self.safe_ratio(
                numerator=water_numerator,
                denominator=water_denominator,
                axis=self._data.water_fractions,
            ),
            gas_fraction=self.safe_ratio(
                numerator=gas_numerator,
                denominator=gas_denominator,
                axis=self._data.gas_fractions,
            ),
        )

    @staticmethod
    def safe_ratio(
        *, numerator: Number, denominator: Number, axis: NumberArray[OneDimension]
    ) -> Number:
        """
        A ratio for a table lookup: zero for `0/0`, and the axis maximum for a nonzero
        numerator over a zero denominator.

        :param numerator: Ratio numerator.
        :param denominator: Ratio denominator.
        :param axis: The table axis the ratio is looked up on.
        :returns: The ratio.
        """
        if denominator > 0.0:
            return numerator / denominator
        if numerator > 0.0:
            return axis[-1]
        return 0.0

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Converts this table to a different unit system, rebuilding the interpolator.

        :param target: Target unit system.
        :param table: Optional custom unit-conversion table.
        :returns: A new `VFPTable` in `target` units.
        """
        if target == self.unit_system:
            return self
        return self.__class__(
            data=self._data.convert(target, table=table),
            warn_on_extrapolation=self.warn_on_extrapolation,
            dtype=self.dtype,
        )

    def _warn_extrapolation(
        self,
        flow_rate: TableQuery[NDimension],
        thp: TableQuery[NDimension],
        water_fraction: TableQuery[NDimension],
        gas_fraction: TableQuery[NDimension],
        artificial_lift_quantity: TableQuery[NDimension],
    ) -> None:
        if not self.warn_on_extrapolation:
            return

        values = (flow_rate, thp, water_fraction, gas_fraction, artificial_lift_quantity)
        for name, value in zip(AXIS_NAMES, values, strict=True):
            bounds = self._extrapolation_bounds.get(name)
            if bounds is None:
                continue

            min_value, max_value = bounds
            value_array = np.atleast_1d(value)
            if np.any(value_array < min_value) or np.any(value_array > max_value):
                logger.warning(
                    "%s extrapolation: queried %s ∈ [%.4g, %.4g], table range [%.4g, %.4g]",
                    name,
                    name,
                    value_array.min(),
                    value_array.max(),
                    min_value,
                    max_value,
                )

    def query(
        self,
        *,
        flow_rate: TableQuery[NDimension],
        thp: TableQuery[NDimension],
        water_fraction: TableQuery[NDimension] = 0.0,
        gas_fraction: TableQuery[NDimension] = 0.0,
        artificial_lift_quantity: TableQuery[NDimension] = 0.0,
    ) -> TableResult[NDimension]:
        """
        Interpolates bottomhole pressure at a point.

        A value given for an axis this table doesn't vary over is accepted but ignored.

        :param flow_rate: Flow rate.
        :param thp: Tubing head pressure.
        :param water_fraction: Water axis value. Ignored if `water_fractions` is single-valued.
        :param gas_fraction: Gas axis value. Ignored if `gas_fractions` is single-valued.
        :param artificial_lift_quantity: Artificial lift quantity.
            Ignored if `artificial_lift_quantities` is single-valued.
        :returns: Interpolated bottomhole pressure, matching the input shape.
        """
        self._warn_extrapolation(
            flow_rate=flow_rate,
            thp=thp,
            water_fraction=water_fraction,
            gas_fraction=gas_fraction,
            artificial_lift_quantity=artificial_lift_quantity,
        )

        full_point = (flow_rate, thp, water_fraction, gas_fraction, artificial_lift_quantity)
        active_point = tuple(
            value for value, active in zip(full_point, self._active, strict=True) if active
        )
        is_scalar = all(np.isscalar(value) for value in active_point)

        arrays = [np.atleast_1d(value) for value in active_point]
        broadcast_shape = np.broadcast_shapes(*(array.shape for array in arrays))
        arrays = [np.broadcast_to(array, broadcast_shape) for array in arrays]
        points = np.column_stack([array.ravel() for array in arrays])
        result = self._interpolator(points).reshape(broadcast_shape)

        dtype = self.dtype
        if is_scalar:
            return typing.cast(Number, result.astype(dtype, copy=False).item())
        return typing.cast(NumberArray[NDimension], result.astype(dtype, copy=False))

    get_bhp = query

    def get_thp(
        self,
        *,
        flow_rate: Number,
        bhp: Number,
        water_fraction: Number = 0.0,
        gas_fraction: Number = 0.0,
        artificial_lift_quantity: Number = 0.0,
    ) -> Number:
        """
        Finds the tubing head pressure that goes with a bottom-hole pressure.

        The table gives BHP from THP, so this inverts it at fixed flow rate, water fraction,
        gas fraction and artificial lift quantity: the BHP is evaluated at every point of the
        THP axis, and the THP is interpolated linearly between the two points that bracket
        `bhp`, or extended linearly past the first or last point.

        :param flow_rate: Flow rate, in the table's own definition of it.
        :param bhp: The bottom-hole pressure to find the THP of.
        :param water_fraction: Water axis value. Ignored if `water_fractions` is single-valued.
        :param gas_fraction: Gas axis value. Ignored if `gas_fractions` is single-valued.
        :param artificial_lift_quantity: Artificial lift quantity.
            Ignored if `artificial_lift_quantities` is single-valued.
        :returns: The tubing head pressure.
        :raises ValidationError: If the table has fewer than two THP points, or its BHP does not
            increase strictly with THP at these conditions, so there is no unique THP.
        """
        thps = self._data.thps
        if len(thps) < 2:
            raise ValidationError(
                f"VFP table {self.table_number} has a single THP point, so a THP can't be "
                "found from a BHP."
            )

        bhps = np.asarray(
            self.query(
                flow_rate=flow_rate,
                thp=thps,
                water_fraction=water_fraction,
                gas_fraction=gas_fraction,
                artificial_lift_quantity=artificial_lift_quantity,
            ),
            dtype=np.float64,
        )
        if not (np.diff(bhps) > 0.0).all():
            raise ValidationError(
                f"VFP table {self.table_number}: BHP does not increase strictly with THP at "
                f"flow rate {flow_rate}, water fraction {water_fraction}, gas fraction {gas_fraction} and "
                f"artificial lift quantity {artificial_lift_quantity}, so a THP can't be found "
                "from a BHP."
            )

        if bhp <= bhps[0]:
            lower, upper = 0, 1
        elif bhp >= bhps[-1]:
            lower, upper = len(thps) - 2, len(thps) - 1
        else:
            return np.interp(bhp, bhps, thps)

        slope = (thps[upper] - thps[lower]) / (bhps[upper] - bhps[lower])
        return thps[lower] + slope * (bhp - bhps[lower])

    @typing.overload
    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        table_number: Integer,
        well_type: WellType | None = None,
        warn_on_extrapolation: bool = False,
        dtype: npt.DTypeLike = None,
    ) -> Self: ...
    @typing.overload
    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        table_number: None = None,
        well_type: WellType | None = None,
        warn_on_extrapolation: bool = False,
        dtype: npt.DTypeLike = None,
    ) -> list[Self]: ...

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        table_number: Integer | None = None,
        well_type: WellType | None = None,
        warn_on_extrapolation: bool = False,
        dtype: npt.DTypeLike = None,
    ) -> Self | list[Self]:
        """
        Construct one or all `VFPTable` objects from a parsed `DeckFile`.

        Reads the `VFPPROD` and `VFPINJ` keywords (see `load_vfp_table`).

        :param deck_file: Parsed `bores.deck.file.DeckFile`.
        :param table_number: Number of a specific table to extract, or `None` for all.
        :param well_type: Read only producer or injector tables, or both if `None`.
        :param warn_on_extrapolation: Log a warning when a query falls outside a table's bounds.
        :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
        :returns: A single `VFPTable` if `table_number` is given, otherwise a list.
        :raises ValidationError: As for `VFPData.from_deck`.
        """
        return typing.cast(
            Self | list[Self],
            load_vfp_table(
                deck_file,
                table_number=table_number,
                well_type=well_type,
                warn_on_extrapolation=warn_on_extrapolation,
                dtype=dtype,
            ),
        )


@attrs.frozen(kw_only=True, slots=True)
class VFPTables(StoreSerializable):
    """
    Number-indexed collections of producer and injector `VFPTable`s.

    Wells select a table by number, via `WCONPROD`/`WCONINJE`'s `vfp_table` item. Several wells
    commonly share one table. Producer and injector tables are numbered independently, as in a
    deck, so the same number can exist in both.
    """

    producers: typing.Mapping[Integer, VFPTable] = attrs.field(factory=dict)
    """Producer (`VFPPROD`) tables, keyed by their own `table_number`."""

    injectors: typing.Mapping[Integer, VFPTable] = attrs.field(factory=dict)
    """Injector (`VFPINJ`) tables, keyed by their own `table_number`."""

    def __attrs_post_init__(self) -> None:
        for well_type, tables in (
            (WellType.PRODUCER, self.producers),
            (WellType.INJECTOR, self.injectors),
        ):
            for table_number, table in tables.items():
                if table.table_number != table_number:
                    raise ValidationError(
                        f"`{well_type.value}` table keyed {table_number!r} has `table_number` "
                        f"{table.table_number!r}."
                    )
                if table.well_type != well_type:
                    raise ValidationError(
                        f"VFP table {table_number!r} is a {table.well_type.value} table but is "
                        f"held with the {well_type.value} tables."
                    )

    @property
    def unit_system(self) -> UnitSystem | None:
        """Unit system shared by every table, or `None` if empty."""
        return next(
            (table.unit_system for table in (*self.producers.values(), *self.injectors.values())),
            None,
        )

    def table(self, table_number: Integer, /, *, well_type: WellType) -> VFPTable:
        """
        Gets a table by number.

        :param table_number: The table's number, deck `VFPPROD`/`VFPINJ` item 1.
        :param well_type: Whether to look among producer or injector tables.
        :returns: The matching `VFPTable`.
        :raises ValidationError: If no such table exists.
        """
        tables = self.producers if well_type == WellType.PRODUCER else self.injectors
        if table_number not in tables:
            raise ValidationError(
                f"No {well_type.value} VFP table numbered {table_number!r}. "
                f"Available: {sorted(tables.keys())}."
            )
        return tables[table_number]

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Converts every table to a different unit system.

        :param target: Target unit system.
        :param table: Optional custom unit-conversion table.
        :returns: New `VFPTables` with every table converted to `target`.
        """
        if target == self.unit_system:
            return self
        return attrs.evolve(
            self,
            producers={
                number: vfp_table.convert(target, table=table)
                for number, vfp_table in self.producers.items()
            },
            injectors={
                number: vfp_table.convert(target, table=table)
                for number, vfp_table in self.injectors.items()
            },
        )

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        well_type: WellType | None = None,
        warn_on_extrapolation: bool = False,
        dtype: npt.DTypeLike = None,
    ) -> Self:
        """
        Construct a `VFPTables` collection from a parsed `DeckFile`.

        Reads the `VFPPROD` and `VFPINJ` keywords (see `load_vfp_tables`).

        :param deck_file: Parsed `bores.deck.file.DeckFile`.
        :param well_type: Read only producer or injector tables, or both if `None`.
        :param warn_on_extrapolation: Log a warning when a query falls outside a table's bounds.
        :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
        :returns: The collection.
        :raises ValidationError: If the deck has no matching table or a table cannot be mapped.
        """
        return typing.cast(
            Self,
            load_vfp_tables(
                deck_file,
                well_type=well_type,
                warn_on_extrapolation=warn_on_extrapolation,
                dtype=dtype,
            ),
        )


VFP_KEYWORDS: typing.Mapping[WellType, str] = {
    WellType.PRODUCER: "VFPPROD",
    WellType.INJECTOR: "VFPINJ",
}
"""Deck keyword that defines the VFP tables of each well type."""


def map_deck_definition(
    *,
    label: str,
    axis: str,
    declared: str,
    mapping: typing.Mapping[str, typing.Any],
) -> typing.Any:
    """
    Maps the definition a deck declares for a table axis to its enum.

    :param label: The table's name, for the error message.
    :param axis: The axis's name, for the error message.
    :param declared: The deck's definition of the axis, such as `LIQ` or `WCT`.
    :param mapping: Deck definition to enum.
    :returns: The enum member.
    :raises ValidationError: If the deck's definition is not one of `mapping`'s keys.
    """
    member = mapping.get(declared.upper())
    if member is None:
        raise ValidationError(
            f"{label}: the {axis} axis is {declared!r}; supported: {sorted(mapping)}."
        )
    return member


def load_vfp_data_from_record(
    record: VFPProducerDeckTable | VFPInjectorDeckTable,
    *,
    well_type: WellType,
    unit_system: UnitSystem,
) -> VFPData:
    """
    Build a `VFPData` from one parsed `VFPPROD` / `VFPINJ` table.

    The table keeps the definitions its deck gives its axes. Only the units are brought to
    the ones used internally: a FIELD table writes gas rates and gas-oil and gas-liquid
    ratios per thousand standard cubic feet, and water-gas and oil-gas ratios per
    thousand standard cubic feet in the denominator, so those are rescaled to standard
    cubic feet. METRIC and LAB tables need no rescaling.

    :param record: One parsed `VFPPROD` / `VFPINJ` table.
    :param well_type: Whether the record is a producer or an injector table.
    :param unit_system: Unit system of the deck, used when the table does not state one.
    :returns: The table data.
    :raises ValidationError: If the table has no datum depth, uses a pressure definition
        other than `THP` and `BHP`, declares an axis definition that is not supported
        (including the `WG` and `TM` flow definitions), or its axes are invalid.
    """
    units = (record["units"] or "").upper()
    unit_system = DECK_UNIT_SYSTEMS.get(units, unit_system) if units else unit_system
    table_number = record["table_number"]
    label = f"VFP table {table_number}"
    datum_depth = record["datum_depth"]
    if datum_depth is None:
        raise ValidationError(f"{label}: the datum depth is required.")
    if record["thp_type"] != "THP" or record["bhp_type"] != "BHP":
        raise ValidationError(
            f"{label}: only THP pressures and BHP values are supported; got "
            f"{record['thp_type']!r} / {record['bhp_type']!r}."
        )

    is_field = unit_system == UnitSystem.FIELD
    if well_type == WellType.INJECTOR:
        injector = typing.cast(VFPInjectorDeckTable, record)
        flow_type = map_deck_definition(
            label=label,
            axis="flow",
            declared=injector["flow_type"],
            mapping=INJECTOR_FLOW_TYPE_MAP,
        )
        gas_flow_scale = 1000.0 if is_field and flow_type == VFPFlowType.GAS else 1.0
        return VFPData(
            table_number=table_number,
            well_type=WellType.INJECTOR,
            datum_depth=datum_depth,
            flow_type=flow_type,
            flow_rates=typing.cast(NumberArray[OneDimension], injector["flow"] * gas_flow_scale),
            thps=typing.cast(NumberArray[OneDimension], injector["thp"]),
            bhps=typing.cast(
                NumberArray[FiveDimensions], injector["bhps"][:, :, None, None, None]
            ),
            unit_system=unit_system,
        )

    producer = typing.cast(VFPProducerDeckTable, record)
    flow_type = map_deck_definition(
        label=label,
        axis="flow",
        declared=producer["flow_type"],
        mapping=PRODUCER_FLOW_TYPE_MAP,
    )
    water_fraction_type = map_deck_definition(
        label=label,
        axis="water",
        declared=producer["water_fraction_type"],
        mapping=WATER_FRACTION_TYPE_MAP,
    )
    gas_fraction_type = map_deck_definition(
        label=label,
        axis="gas",
        declared=producer["gas_fraction_type"],
        mapping=GAS_FRACTION_TYPE_MAP,
    )

    flow_scale = 1000.0 if is_field and flow_type == VFPFlowType.GAS else 1.0
    water_scale = (
        1e-3 if is_field and water_fraction_type == VFPWaterFractionType.WATER_GAS_RATIO else 1.0
    )
    if not is_field:
        gas_scale = 1.0
    elif gas_fraction_type == VFPGasFractionType.OIL_GAS_RATIO:
        gas_scale = 1e-3
    else:
        gas_scale = 1000.0
    return VFPData(
        table_number=table_number,
        well_type=WellType.PRODUCER,
        datum_depth=datum_depth,
        flow_type=flow_type,
        water_fraction_type=water_fraction_type,
        gas_fraction_type=gas_fraction_type,
        flow_rates=typing.cast(NumberArray[OneDimension], producer["flow"] * flow_scale),
        thps=typing.cast(NumberArray[OneDimension], producer["thp"]),
        water_fractions=typing.cast(
            NumberArray[OneDimension], producer["water_fraction"] * water_scale
        ),
        gas_fractions=typing.cast(NumberArray[OneDimension], producer["gas_fraction"] * gas_scale),
        artificial_lift_quantities=typing.cast(NumberArray[OneDimension], producer["alq"]),
        bhps=typing.cast(NumberArray[FiveDimensions], producer["bhps"]),
        unit_system=unit_system,
    )


def select_vfp_records(
    deck_file: DeckFile,
    *,
    table_number: Integer | None,
    well_type: WellType | None,
) -> list[tuple[WellType, VFPProducerDeckTable | VFPInjectorDeckTable]]:
    """
    The `VFPPROD` / `VFPINJ` tables of a deck that match a selection.

    A table number defined more than once within one keyword keeps its last definition, as in a
    deck. Producer and injector tables are numbered independently.

    :param deck_file: Parsed deck.
    :param table_number: The table to select, or `None` for every table.
    :param well_type: Select only producer or injector tables, or both if `None`.
    :returns: `(well_type, record)` pairs in deck order.
    :raises ValidationError: If the deck has no matching keyword, `table_number` is not found,
        or it exists as both a producer and an injector table and `well_type` was not given.
    """
    kinds = (well_type,) if well_type is not None else tuple(VFP_KEYWORDS)
    latest: dict[tuple[WellType, Integer], VFPProducerDeckTable | VFPInjectorDeckTable] = {}
    for kind in kinds:
        for record in deck_file.get(VFP_KEYWORDS[kind]) or []:
            latest[kind, record["table_number"]] = record

    if not latest:
        names = " or ".join(f"`{VFP_KEYWORDS[kind]}`" for kind in kinds)
        raise ValidationError(f"No {names} keyword found in the provided deck.")

    selected = [(kind, record) for (kind, _), record in latest.items()]
    if table_number is None:
        return selected

    matching = [
        (kind, record) for kind, record in selected if record["table_number"] == table_number
    ]
    if not matching:
        available = sorted({number for _, number in latest})
        raise ValidationError(f"VFP table {table_number!r} not found. Available: {available}.")

    if len(matching) > 1:
        raise ValidationError(
            f"VFP table {table_number!r} is defined by both `VFPPROD` and `VFPINJ`; pass "
            "`well_type` to choose one."
        )
    return matching


@typing.overload
def load_vfp_data(
    deck_file: DeckFile, *, table_number: Integer, well_type: WellType | None = None
) -> VFPData: ...
@typing.overload
def load_vfp_data(
    deck_file: DeckFile, *, table_number: None = None, well_type: WellType | None = None
) -> list[VFPData]: ...


def load_vfp_data(
    deck_file: DeckFile,
    *,
    table_number: Integer | None = None,
    well_type: WellType | None = None,
) -> VFPData | list[VFPData]:
    """
    Load one or all `VFPPROD` / `VFPINJ` tables of a deck as `VFPData`.

    :param deck_file: Parsed `bores.deck.file.DeckFile`.
    :param table_number: Number of a specific table to load, or `None` for all.
    :param well_type: Load only producer or injector tables, or both if `None`.
    :returns: A single `VFPData` if `table_number` is given, otherwise a list in deck order.
    :raises ValidationError: If the deck has no matching table, `table_number` is not found or
        is ambiguous, or a table cannot be mapped (see `load_vfp_data_from_record`).
    """
    loaded = [
        load_vfp_data_from_record(record, well_type=kind, unit_system=deck_file.unit_system)
        for kind, record in select_vfp_records(
            deck_file, table_number=table_number, well_type=well_type
        )
    ]
    return loaded[0] if table_number is not None else loaded


@typing.overload
def load_vfp_table(
    deck_file: DeckFile,
    *,
    table_number: Integer,
    well_type: WellType | None = None,
    warn_on_extrapolation: bool = False,
    dtype: npt.DTypeLike = None,
) -> VFPTable: ...
@typing.overload
def load_vfp_table(
    deck_file: DeckFile,
    *,
    table_number: None = None,
    well_type: WellType | None = None,
    warn_on_extrapolation: bool = False,
    dtype: npt.DTypeLike = None,
) -> list[VFPTable]: ...


def load_vfp_table(
    deck_file: DeckFile,
    *,
    table_number: Integer | None = None,
    well_type: WellType | None = None,
    warn_on_extrapolation: bool = False,
    dtype: npt.DTypeLike = None,
) -> VFPTable | list[VFPTable]:
    """
    Load one or all `VFPPROD` / `VFPINJ` tables of a deck as `VFPTable`s.

    :param deck_file: Parsed `bores.deck.file.DeckFile`.
    :param table_number: Number of a specific table to load, or `None` for all.
    :param well_type: Load only producer or injector tables, or both if `None`.
    :param warn_on_extrapolation: Log a warning when a query falls outside a table's bounds.
    :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
    :returns: A single `VFPTable` if `table_number` is given, otherwise a list in deck order.
    :raises ValidationError: As for `load_vfp_data`.
    """
    data = load_vfp_data(deck_file, table_number=table_number, well_type=well_type)
    if table_number is not None:
        return VFPTable(
            typing.cast(VFPData, data), warn_on_extrapolation=warn_on_extrapolation, dtype=dtype
        )
    return [
        VFPTable(item, warn_on_extrapolation=warn_on_extrapolation, dtype=dtype)
        for item in typing.cast(list[VFPData], data)
    ]


def load_vfp_tables(
    deck_file: DeckFile,
    *,
    well_type: WellType | None = None,
    warn_on_extrapolation: bool = False,
    dtype: npt.DTypeLike = None,
) -> VFPTables:
    """
    Load the `VFPPROD` / `VFPINJ` tables of a deck into one `VFPTables` collection.

    Producer and injector tables are held separately, so the same number can exist in both.

    :param deck_file: Parsed `bores.deck.file.DeckFile`.
    :param well_type: Load only producer or injector tables, or both if `None`.
    :param warn_on_extrapolation: Log a warning when a query falls outside a table's bounds.
    :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
    :returns: The collection.
    :raises ValidationError: If the deck has no matching table or a table cannot be mapped.
    """
    producers: dict[Integer, VFPTable] = {}
    injectors: dict[Integer, VFPTable] = {}
    for kind, record in select_vfp_records(deck_file, table_number=None, well_type=well_type):
        data = load_vfp_data_from_record(record, well_type=kind, unit_system=deck_file.unit_system)
        vfp_table = VFPTable(data, warn_on_extrapolation=warn_on_extrapolation, dtype=dtype)
        (producers if kind == WellType.PRODUCER else injectors)[data.table_number] = vfp_table
    return VFPTables(producers=producers, injectors=injectors)


FLOW_ROW: typing.Mapping[VFPFlowType, tuple[float, float, float]] = {
    VFPFlowType.OIL: (1.0, 0.0, 0.0),
    VFPFlowType.LIQUID: (1.0, 1.0, 0.0),
    VFPFlowType.WATER: (0.0, 1.0, 0.0),
    VFPFlowType.GAS: (0.0, 0.0, 1.0),
}
"""Coefficients of `(oil, water, gas)` rates that equal the flow axis value, by flow type."""


def compute_surface_rates(
    *,
    flow_type: VFPFlowType,
    water_fraction_type: VFPWaterFractionType,
    gas_fraction_type: VFPGasFractionType,
    flow_rate: Number,
    water_fraction: Number,
    gas_fraction: Number,
) -> PhaseValues:
    """
    Surface oil, water and gas rates that a table's lookup values describe.

    This is the inverse of `VFPTable.get_lookup_variables`: the three definitions each give
    one linear equation in the three rates, which are solved together.

    :param flow_type: What `flow_rate` measures.
    :param water_fraction_type: What `water_fraction` measures.
    :param gas_fraction_type: What `gas_fraction` measures.
    :param flow_rate: Value on the flow axis.
    :param water_fraction: Value on the water axis.
    :param gas_fraction: Value on the gas axis.
    :returns: Surface-condition phase rates.
    :raises ValidationError: If the three values and definitions don't determine a single
        non-negative set of rates, such as a water cut of 1 against an oil rate.
    """
    if water_fraction_type == VFPWaterFractionType.WATER_CUT:
        water_row = (-water_fraction, 1.0 - water_fraction, 0.0)
    elif water_fraction_type == VFPWaterFractionType.WATER_OIL_RATIO:
        water_row = (-water_fraction, 1.0, 0.0)
    else:
        water_row = (0.0, 1.0, -water_fraction)

    if gas_fraction_type == VFPGasFractionType.GAS_OIL_RATIO:
        gas_row = (-gas_fraction, 0.0, 1.0)
    elif gas_fraction_type == VFPGasFractionType.GAS_LIQUID_RATIO:
        gas_row = (-gas_fraction, -gas_fraction, 1.0)
    else:
        gas_row = (1.0, 0.0, -gas_fraction)

    matrix = np.array([FLOW_ROW[flow_type], water_row, gas_row], dtype=np.float64)
    try:
        oil, water, gas = np.linalg.solve(matrix, np.array([flow_rate, 0.0, 0.0]))
    except np.linalg.LinAlgError:
        oil = water = gas = np.nan
    if not (np.isfinite([oil, water, gas]).all() and min(oil, water, gas) >= -1e-9):
        raise ValidationError(
            f"A {flow_type} flow of {flow_rate}, {water_fraction_type} of {water_fraction} and "
            f"{gas_fraction_type} of {gas_fraction} don't describe one set of non-negative "
            "phase rates."
        )
    return PhaseValues(oil=max(oil, 0.0), water=max(water, 0.0), gas=max(gas, 0.0))


def compute_reservoir_rates(
    *,
    surface_rates: PhaseValues,
    formation_volume_factors: PhaseValues,
) -> PhaseValues:
    """
    Reservoir-condition phase rates from surface-condition ones.

    :param surface_rates: Surface-condition oil, water and gas rates.
    :param formation_volume_factors: Oil, water and gas formation volume factors.
    :returns: Reservoir-condition phase rates.
    """
    return PhaseValues(
        oil=surface_rates.oil * formation_volume_factors.oil,
        water=surface_rates.water * formation_volume_factors.water,
        gas=surface_rates.gas * formation_volume_factors.gas,
    )


def bisect_bhp_for_thp(
    *,
    wellbore: WellBoreModel,
    reference_depth: Number,
    rates: PhaseValues,
    surface_rates: PhaseValues,
    surface_sample: SurfaceSample | None,
    is_injector: bool,
    target_thp: Number,
    min_bhp: Number,
    max_bhp: Number,
    max_iterations: Integer,
    convergence_tolerance: Number,
) -> Number:
    low, high = min_bhp, max_bhp
    bhp = 0.5 * (low + high)
    for _ in range(max_iterations):
        bhp = 0.5 * (low + high)
        if wellbore.tubing_fluid_table is not None:
            thp = compute_tubing_head_pressure_from_table(
                wellbore=wellbore,
                reference_depth=reference_depth,
                reference_pressure=bhp,
                surface_rates=surface_rates,
                is_injector=is_injector,
                static_density=surface_sample.density if surface_sample is not None else math.nan,
            )
        else:
            assert surface_sample is not None
            thp = compute_tubing_head_pressure(
                wellbore=wellbore,
                reference_depth=reference_depth,
                reference_pressure=bhp,
                rates=rates,
                surface_sample=surface_sample,
                is_injector=is_injector,
            )
        if abs(thp - target_thp) <= convergence_tolerance * max(abs(target_thp), 1.0):
            break
        # Higher BHP means higher THP, for both well types.
        if thp < target_thp:
            low = bhp
        else:
            high = bhp
    return bhp


def as_vfp_table(
    wellbore: WellBoreModel,
    *,
    table_number: Integer,
    well_type: WellType,
    reference_depth: Number,
    flow_rates: NumberArray[OneDimension],
    thps: NumberArray[OneDimension],
    surface_sample: SurfaceSample | None = None,
    flow_type: VFPFlowType | None = None,
    water_fraction_type: VFPWaterFractionType = VFPWaterFractionType.WATER_CUT,
    gas_fraction_type: VFPGasFractionType = VFPGasFractionType.GAS_OIL_RATIO,
    water_fractions: NumberArray[OneDimension] | None = None,
    gas_fractions: NumberArray[OneDimension] | None = None,
    formation_volume_factors: PhaseValues | None = None,
    injected_phase: FluidPhase | None = None,
    min_bhp: Number,
    max_bhp: Number,
    max_bisection_iterations: Integer = 60,
    bhp_convergence_tolerance: Number = 1e-4,
    unit_system: UnitSystem = UnitSystem.FIELD,
    dtype: npt.DTypeLike = None,
    warn_on_extrapolation: bool = False,
) -> VFPTable:
    """
    Samples a `WellBoreModel` correlation into a `VFPTable`.

    The table's axes are written in whatever terms are declared by `flow_type`,
    `water_fraction_type` and `gas_fraction_type`. For each grid point, the surface
    oil, water and gas rates those values describe are found, converted to reservoir
    conditions, and the BHP that makes the correlation's THP match that point's `thps`
    value is found by bisection. This resamples the same datum-to-surface relationship
    `compute_tubing_head_pressure` computes analytically, once per grid point, so later
    lookups are a single interpolation instead of a bisection.

    `formation_volume_factors` and `surface_sample` are held
    fixed across the whole grid; this doesn't vary either with pressure
    or with the `water_fractions`/`gas_fractions` sweep. Build separate
    tables if a wider composition range needs its own PVT behavior.

    No artificial-lift correlation exists in `wells.hydraulics` yet, so
    the resulting table's `artificial_lift_quantities` axis is always
    single-valued.

    :param wellbore: The correlation to sample.
    :param table_number: Table number for the resulting `VFPData`.
    :param well_type: Producer or injector.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param flow_rates: Flow rate axis, at surface conditions, in the terms of `flow_type`.
    :param thps: Tubing head pressure axis to solve BHP for.
    :param surface_sample: Fixed surface fluid properties used for every grid point. Not needed
        if `wellbore` has `tubing_fluid_table`, which then give the fluids at every pressure, and
        the grid is sampled from surface-condition rates directly.
    :param flow_type: What `flow_rates` measures. For a producer, liquid rate if not given.
        For an injector, the phase of `injected_phase`.
    :param water_fraction_type: What `water_fractions` measures. Ignored for an injector.
    :param gas_fraction_type: What `gas_fractions` measures. Ignored for an injector.
    :param water_fractions: Water axis, in the terms of `water_fraction_type`. Ignored for an
        injector. `[0.0]` if not given.
    :param gas_fractions: Gas axis, in the terms of `gas_fraction_type`. Ignored for an
        injector. `[0.0]` if not given.
    :param formation_volume_factors: Fixed oil/water/gas FVFs, converting
        the surface-condition rates the axes describe to the reservoir-condition rates
        `wellbore`'s correlation needs.
        `PhaseValues(oil=1.0, water=1.0, gas=1.0)` (no conversion) if not given.
    :param injected_phase: The injected phase, for an injector. Used to choose `flow_type` if
        that is not given.
    :param min_bhp: Lower bisection bracket bound, covering every grid point.
    :param max_bhp: Upper bisection bracket bound, covering every grid point.
    :param max_bisection_iterations: Per grid point.
    :param bhp_convergence_tolerance: Relative tolerance on `thps` for
        the bisection to stop early.
    :param unit_system: Unit system `flow_rates`/`thps`/etc. are already
        expressed in.
    :param dtype: Output array dtype for the resulting `VFPTable`.
    :param warn_on_extrapolation: Forwarded to `VFPTable`.
    :returns: A `VFPTable` covering `flow_rates` x `thps` x `water_fractions` x `gas_fractions`.
    :raises ValidationError: If an injector's flow is not water or gas, or a grid point's values
        don't describe non-negative phase rates.
    """
    is_injector = well_type == WellType.INJECTOR
    if surface_sample is None and wellbore.tubing_fluid_table is None:
        raise ValidationError(
            "`surface_sample` is needed when `wellbore` has no `tubing_fluid_table`."
        )

    resolved_formation_volume_factors = (
        formation_volume_factors
        if formation_volume_factors is not None
        else PhaseValues(oil=1.0, water=1.0, gas=1.0)
    )
    resolved_water_fractions = water_fractions if water_fractions is not None else np.array([0.0])
    resolved_gas_fractions = gas_fractions if gas_fractions is not None else np.array([0.0])
    if is_injector:
        resolved_water_fractions = np.array([0.0])
        resolved_gas_fractions = np.array([0.0])
        water_fraction_type = VFPWaterFractionType.WATER_CUT
        gas_fraction_type = VFPGasFractionType.GAS_OIL_RATIO
        if flow_type is None:
            if injected_phase == FluidPhase.WATER:
                flow_type = VFPFlowType.WATER
            elif injected_phase == FluidPhase.GAS:
                flow_type = VFPFlowType.GAS

        if flow_type not in (VFPFlowType.WATER, VFPFlowType.GAS):
            raise ValidationError(
                "An injector table's `flow_type` must be water or gas, or `injected_phase` must "
                f"be `FluidPhase.WATER` or `FluidPhase.GAS`; got {flow_type!r} and "
                f"{injected_phase!r}."
            )
    elif flow_type is None:
        flow_type = VFPFlowType.LIQUID

    bhps = np.empty((
        len(flow_rates),
        len(thps),
        len(resolved_water_fractions),
        len(resolved_gas_fractions),
        1,
    ))
    for i, flow_rate in enumerate(flow_rates):
        for k, water_fraction in enumerate(resolved_water_fractions):
            for m, gas_fraction in enumerate(resolved_gas_fractions):
                if is_injector:
                    surface_rates = PhaseValues(
                        oil=0.0,
                        water=flow_rate if flow_type == VFPFlowType.WATER else 0.0,
                        gas=flow_rate if flow_type == VFPFlowType.GAS else 0.0,
                    )
                else:
                    surface_rates = compute_surface_rates(
                        flow_type=flow_type,
                        water_fraction_type=water_fraction_type,
                        gas_fraction_type=gas_fraction_type,
                        flow_rate=flow_rate,
                        water_fraction=water_fraction,
                        gas_fraction=gas_fraction,
                    )

                rates = compute_reservoir_rates(
                    surface_rates=surface_rates,
                    formation_volume_factors=resolved_formation_volume_factors,
                )
                for j, thp in enumerate(thps):
                    bhps[i, j, k, m, 0] = bisect_bhp_for_thp(
                        wellbore=wellbore,
                        reference_depth=reference_depth,
                        rates=rates,
                        surface_rates=surface_rates,
                        surface_sample=surface_sample,
                        is_injector=is_injector,
                        target_thp=thp,
                        min_bhp=min_bhp,
                        max_bhp=max_bhp,
                        max_iterations=max_bisection_iterations,
                        convergence_tolerance=bhp_convergence_tolerance,
                    )

    data = VFPData(
        table_number=table_number,
        well_type=well_type,
        datum_depth=reference_depth,
        flow_type=flow_type,
        water_fraction_type=water_fraction_type,
        gas_fraction_type=gas_fraction_type,
        flow_rates=flow_rates,
        thps=thps,
        water_fractions=typing.cast(NumberArray[OneDimension], resolved_water_fractions),
        gas_fractions=typing.cast(NumberArray[OneDimension], resolved_gas_fractions),
        bhps=bhps,
        unit_system=unit_system,
    )
    return VFPTable(data, warn_on_extrapolation=warn_on_extrapolation, dtype=dtype)
