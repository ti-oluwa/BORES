"""
VFP (vertical flow performance) tables.

Interpolates bottomhole pressure at a well's datum depth as a function of flow rate,
tubing head pressure, water cut, gas-oil ratio, and artificial lift quantity.
This is the datum-to-surface leg of well hydraulics only; connection-level
pressure distribution below the datum is unaffected by whether a table is assigned.
"""

import logging
import typing
import warnings

import attrs
import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator  # type: ignore[import-untyped]
from typing_extensions import Self

from bores.constants import UnitConversionTable, get_conversion_factors
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
from bores.utils import scale
from bores.wells.base import WellType
from bores.wells.hydraulics.base import (
    SurfaceFluidProperties,
    WellBoreModel,
    compute_tubing_head_pressure,
)
from bores.wells.state import PhaseValues

__all__ = ["VFPData", "VFPTable", "VFPTables", "as_vfp_table"]

logger = logging.getLogger(__name__)

DECK_UNIT_SYSTEMS: typing.Mapping[str, UnitSystem] = {
    "METRIC": UnitSystem.METRIC,
    "FIELD": UnitSystem.FIELD,
    "LAB": UnitSystem.LAB,
    "SI": UnitSystem.SI,
    "PVT-M": UnitSystem.METRIC,
}
"""Deck unit keyword to the unit system a `VFPPROD` / `VFPINJ` table is written in."""

AXIS_NAMES = ("flow_rate", "thp", "water_cut", "gas_oil_ratio", "artificial_lift_quantity")
"""Order `VFPData`'s five axes are always addressed in, internally."""


@attrs.frozen(kw_only=True, slots=True)
class VFPData(Serializable):
    """
    Raw axes and bottomhole-pressure grid for one VFP table.

    A producer table typically varies over all five axes. An injector
    table usually varies only with `flow_rates` and `thps`; leave
    `water_cuts`, `gas_oil_ratios`, and `artificial_lift_quantities` at
    their single-value defaults in that case.
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

    flow_rates: NumberArray[OneDimension]
    """
    Flow rate axis, strictly ascending. Liquid rate for most producer
    tables; the injected phase's rate for an injector table.
    """

    thps: NumberArray[OneDimension]
    """Tubing head pressure axis, strictly ascending."""

    water_cuts: NumberArray[OneDimension] = attrs.field(
        factory=lambda: typing.cast(NumberArray[OneDimension], np.array([0.0]))
    )
    """
    Water cut axis, strictly ascending. A single value (the default)
    for a table with no water-cut dependence.
    """

    gas_oil_ratios: NumberArray[OneDimension] = attrs.field(
        factory=lambda: typing.cast(NumberArray[OneDimension], np.array([0.0]))
    )
    """
    Gas-oil ratio axis, strictly ascending. A single value (the
    default) for a table with no GOR dependence.
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
    len(thps), len(water_cuts), len(gas_oil_ratios),
    len(artificial_lift_quantities))`.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system every dimensioned field above is expressed in."""

    def __attrs_post_init__(self) -> None:
        axes = (
            self.flow_rates,
            self.thps,
            self.water_cuts,
            self.gas_oil_ratios,
            self.artificial_lift_quantities,
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
        return attrs.evolve(
            self,
            flow_rates=scale(
                self.flow_rates,
                factors[
                    "gas_surface_rate"
                    if self.well_type == WellType.INJECTOR
                    else "liquid_surface_rate"
                ],
            ),
            thps=scale(self.thps, factors["pressure"]),
            gas_oil_ratios=scale(self.gas_oil_ratios, factors["gas_oil_ratio"]),
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
            data.water_cuts,
            data.gas_oil_ratios,
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
        water_cut: TableQuery[NDimension],
        gas_oil_ratio: TableQuery[NDimension],
        artificial_lift_quantity: TableQuery[NDimension],
    ) -> None:
        if not self.warn_on_extrapolation:
            return
        values = (flow_rate, thp, water_cut, gas_oil_ratio, artificial_lift_quantity)
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
        water_cut: TableQuery[NDimension] = 0.0,
        gas_oil_ratio: TableQuery[NDimension] = 0.0,
        artificial_lift_quantity: TableQuery[NDimension] = 0.0,
    ) -> TableResult[NDimension]:
        """
        Interpolates bottomhole pressure at a point.

        A value given for an axis this table doesn't vary over is
        accepted but ignored.

        :param flow_rate: Flow rate.
        :param thp: Tubing head pressure.
        :param water_cut: Water cut. Ignored if `water_cuts` is single-valued.
        :param gas_oil_ratio: Gas-oil ratio. Ignored if `gas_oil_ratios` is single-valued.
        :param artificial_lift_quantity: Artificial lift quantity.
            Ignored if `artificial_lift_quantities` is single-valued.
        :returns: Interpolated bottomhole pressure, matching the input shape.
        """
        self._warn_extrapolation(
            flow_rate=flow_rate,
            thp=thp,
            water_cut=water_cut,
            gas_oil_ratio=gas_oil_ratio,
            artificial_lift_quantity=artificial_lift_quantity,
        )

        full_point = (flow_rate, thp, water_cut, gas_oil_ratio, artificial_lift_quantity)
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


def regrid_oil_flow_to_liquid_flow(
    flow_rates: NumberArray[OneDimension],
    water_cuts: NumberArray[OneDimension],
    bhps: NumberArray[FiveDimensions],
) -> tuple[NumberArray[OneDimension], NumberArray[FiveDimensions]]:
    """
    Re-express a table written against oil rate as one written against liquid rate.

    At water cut `w`, an oil rate `q` is a liquid rate `q / (1 - w)`, so every water cut has its
    own native liquid-rate axis. The result uses the union of all of them as its flow axis. Each
    water cut's BHP values are exact at its own native points and are interpolated linearly
    between them (and extended linearly beyond its ends) at the other points, which makes the
    result an approximation away from the original grid points.

    :param flow_rates: Oil rate axis.
    :param water_cuts: Water cut axis (fraction).
    :param bhps: BHP values shaped `(flow, thp, water_cut, gas_oil_ratio, alq)`.
    :returns: `(liquid_rate_axis, bhps)`, with the BHPs shaped against the new axis.
    :raises ValidationError: If a water cut is 1 or more, or the flow axis has a single value.
    """
    if (water_cuts >= 1.0).any():
        raise ValidationError(
            "A water cut of 1 or more has no finite liquid rate for a given oil rate, so the "
            "table cannot be re-expressed against liquid rate."
        )
    if len(flow_rates) < 2:
        raise ValidationError("Re-gridding needs at least two values on the flow axis.")

    native = flow_rates[:, None] / (1.0 - water_cuts[None, :])
    liquid_rates = np.unique(native.ravel())
    regridded = np.empty((len(liquid_rates), *bhps.shape[1:]), dtype=np.float64)
    for index in range(len(water_cuts)):
        axis = native[:, index]
        upper = np.clip(np.searchsorted(axis, liquid_rates), 1, len(axis) - 1)
        lower = upper - 1
        fraction = (liquid_rates - axis[lower]) / (axis[upper] - axis[lower])
        values = bhps[:, :, index, :, :]
        regridded[:, :, index, :, :] = values[lower] + fraction[:, None, None, None] * (
            values[upper] - values[lower]
        )
    return typing.cast(NumberArray[OneDimension], liquid_rates), typing.cast(
        NumberArray[FiveDimensions], regridded
    )


def load_vfp_data_from_record(
    record: VFPProducerDeckTable | VFPInjectorDeckTable,
    *,
    well_type: WellType,
    unit_system: UnitSystem,
) -> VFPData:
    """
    Build a `VFPData` from one parsed `VFPPROD` / `VFPINJ` table.

    A producer table maps exactly when it is written in terms of liquid rate (`LIQ`), water cut
    (`WCT`) or water-oil ratio (`WOR`, converted to water cut) and gas-oil ratio (`GOR`), with
    `THP` pressures and `BHP` values. An oil-rate table (`OIL`) is re-expressed against liquid
    rate, which is approximate (see `regrid_oil_flow_to_liquid_flow`) and warns. An injector
    table maps with any injected phase. Gas-oil ratios and injected gas rates are rescaled from
    the deck's thousands of standard cubic feet to standard cubic feet for FIELD tables.

    :param record: One parsed `VFPPROD` / `VFPINJ` table.
    :param well_type: Whether the record is a producer or an injector table.
    :param unit_system: Unit system of the deck, used when the table does not state one.
    :returns: The table data.
    :raises ValidationError: If the table uses an axis definition that cannot be mapped, has no
        datum depth, or its axes are invalid.
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

    if well_type == WellType.INJECTOR:
        injector = typing.cast(VFPInjectorDeckTable, record)
        scale_gas = unit_system == UnitSystem.FIELD and injector["flow_type"] == "GAS"
        return VFPData(
            table_number=table_number,
            well_type=WellType.INJECTOR,
            datum_depth=datum_depth,
            flow_rates=typing.cast(
                NumberArray[OneDimension], injector["flow"] * (1000.0 if scale_gas else 1.0)
            ),
            thps=typing.cast(NumberArray[OneDimension], injector["thp"]),
            bhps=typing.cast(
                NumberArray[FiveDimensions], injector["bhps"][:, :, None, None, None]
            ),
            unit_system=unit_system,
        )

    producer = typing.cast(VFPProducerDeckTable, record)
    water_fraction = producer["water_fraction"]
    if producer["water_fraction_type"] == "WCT":
        water_cuts = water_fraction
    elif producer["water_fraction_type"] == "WOR":
        water_cuts = typing.cast(
            NumberArray[OneDimension], water_fraction / (1.0 + water_fraction)
        )
    else:
        raise ValidationError(
            f"{label}: the water axis is {producer['water_fraction_type']!r}; only `WCT` and "
            "`WOR` are supported."
        )

    if producer["gas_fraction_type"] != "GOR":
        raise ValidationError(
            f"{label}: the gas axis is {producer['gas_fraction_type']!r}; only `GOR` is supported."
        )

    flow_rates = producer["flow"]
    bhps = producer["bhps"]
    if producer["flow_type"] == "OIL":
        warnings.warn(
            f"{label}: the oil-rate flow axis was re-expressed against liquid rate; BHPs away "
            "from the original grid points are interpolated.",
            stacklevel=3,
        )
        flow_rates, bhps = regrid_oil_flow_to_liquid_flow(flow_rates, water_cuts, bhps)
    elif producer["flow_type"] != "LIQ":
        raise ValidationError(
            f"{label}: the flow axis is {producer['flow_type']!r}; only liquid rate (`LIQ`) and "
            "oil rate (`OIL`) are supported."
        )

    gas_oil_ratios = producer["gas_fraction"] * (
        1000.0 if unit_system == UnitSystem.FIELD else 1.0
    )
    return VFPData(
        table_number=table_number,
        well_type=WellType.PRODUCER,
        datum_depth=datum_depth,
        flow_rates=typing.cast(NumberArray[OneDimension], flow_rates),
        thps=typing.cast(NumberArray[OneDimension], producer["thp"]),
        water_cuts=typing.cast(NumberArray[OneDimension], water_cuts),
        gas_oil_ratios=typing.cast(NumberArray[OneDimension], gas_oil_ratios),
        artificial_lift_quantities=typing.cast(NumberArray[OneDimension], producer["alq"]),
        bhps=typing.cast(NumberArray[FiveDimensions], bhps),
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


def compute_reservoir_phase_rates(
    *,
    flow_rate: Number,
    water_cut: Number,
    gas_oil_ratio: Number,
    well_type: WellType,
    injected_phase: FluidPhase | None,
    formation_volume_factors: PhaseValues,
) -> PhaseValues:
    if well_type == WellType.INJECTOR:
        if injected_phase == FluidPhase.WATER:
            return PhaseValues(oil=0.0, water=flow_rate * formation_volume_factors.water, gas=0.0)
        if injected_phase == FluidPhase.GAS:
            return PhaseValues(oil=0.0, water=0.0, gas=flow_rate * formation_volume_factors.gas)
        raise ValidationError(
            f"`injected_phase` must be `FluidPhase.WATER` or `FluidPhase.GAS` "
            f"for an injector table, got {injected_phase!r}."
        )
    water_rate = flow_rate * water_cut
    oil_rate = flow_rate - water_rate
    gas_rate = oil_rate * gas_oil_ratio
    return PhaseValues(
        oil=oil_rate * formation_volume_factors.oil,
        water=water_rate * formation_volume_factors.water,
        gas=gas_rate * formation_volume_factors.gas,
    )


def bisect_bhp_for_thp(
    *,
    wellbore: WellBoreModel,
    reference_depth: Number,
    phase_rates: PhaseValues,
    surface_fluid_properties: SurfaceFluidProperties,
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
        thp = compute_tubing_head_pressure(
            wellbore=wellbore,
            reference_depth=reference_depth,
            reference_pressure=bhp,
            phase_rates=phase_rates,
            surface_fluid_properties=surface_fluid_properties,
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
    surface_fluid_properties: SurfaceFluidProperties,
    water_cuts: NumberArray[OneDimension] | None = None,
    gas_oil_ratios: NumberArray[OneDimension] | None = None,
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

    For each grid point, finds the BHP that makes the correlation's THP
    match that point's `thps` value, by bisection. This resamples the
    same datum-to-surface relationship `compute_tubing_head_pressure`
    computes analytically, once per grid point, so later lookups are a
    single interpolation instead of a bisection.

    `formation_volume_factors` and `surface_fluid_properties` are held
    fixed across the whole grid; this doesn't vary either with pressure
    or with the `water_cuts`/`gas_oil_ratios` sweep. Build separate
    tables if a wider composition range needs its own PVT behavior.

    No artificial-lift correlation exists in `wells.hydraulics` yet, so
    the resulting table's `artificial_lift_quantities` axis is always
    single-valued.

    :param wellbore: The correlation to sample.
    :param table_number: Table number for the resulting `VFPData`.
    :param well_type: Producer or injector.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param flow_rates: Flow rate axis, at surface conditions.
    :param thps: Tubing head pressure axis to solve BHP for.
    :param surface_fluid_properties: Fixed surface fluid properties used
        for every grid point.
    :param water_cuts: Water cut axis. Ignored for an injector. `[0.0]` if not given.
    :param gas_oil_ratios: Gas-oil ratio axis. Ignored for an injector. `[0.0]` if not given.
    :param formation_volume_factors: Fixed oil/water/gas FVFs, converting
        `flow_rates`/`water_cuts`/`gas_oil_ratios` (surface conditions) to
        the reservoir-condition rates `wellbore`'s correlation needs.
        `PhaseValues(oil=1.0, water=1.0, gas=1.0)` (no conversion) if not given.
    :param injected_phase: Which phase `flow_rates` is, for an injector.
        Required if `well_type` is `WellType.INJECTOR`.
    :param min_bhp: Lower bisection bracket bound, covering every grid point.
    :param max_bhp: Upper bisection bracket bound, covering every grid point.
    :param max_bisection_iterations: Per grid point.
    :param bhp_convergence_tolerance: Relative tolerance on `thps` for
        the bisection to stop early.
    :param unit_system: Unit system `flow_rates`/`thps`/etc. are already
        expressed in.
    :param dtype: Output array dtype for the resulting `VFPTable`.
    :param warn_on_extrapolation: Forwarded to `VFPTable`.
    :returns: A `VFPTable` covering `flow_rates` x `thps` x `water_cuts` x `gas_oil_ratios`.
    :raises ValidationError: If `well_type` is `WellType.INJECTOR` and
        `injected_phase` isn't `FluidPhase.WATER` or `FluidPhase.GAS`.
    """
    is_injector = well_type == WellType.INJECTOR
    resolved_formation_volume_factors = (
        formation_volume_factors
        if formation_volume_factors is not None
        else PhaseValues(oil=1.0, water=1.0, gas=1.0)
    )
    resolved_water_cuts = water_cuts if water_cuts is not None else np.array([0.0])
    resolved_gas_oil_ratios = gas_oil_ratios if gas_oil_ratios is not None else np.array([0.0])
    if is_injector:
        resolved_water_cuts = np.array([0.0])
        resolved_gas_oil_ratios = np.array([0.0])

    bhps = np.empty((
        len(flow_rates),
        len(thps),
        len(resolved_water_cuts),
        len(resolved_gas_oil_ratios),
        1,
    ))
    for i, flow_rate in enumerate(flow_rates):
        for k, water_cut in enumerate(resolved_water_cuts):
            for m, gas_oil_ratio in enumerate(resolved_gas_oil_ratios):
                phase_rates = compute_reservoir_phase_rates(
                    flow_rate=flow_rate,
                    water_cut=water_cut,
                    gas_oil_ratio=gas_oil_ratio,
                    well_type=well_type,
                    injected_phase=injected_phase,
                    formation_volume_factors=resolved_formation_volume_factors,
                )
                for j, thp in enumerate(thps):
                    bhps[i, j, k, m, 0] = bisect_bhp_for_thp(
                        wellbore=wellbore,
                        reference_depth=reference_depth,
                        phase_rates=phase_rates,
                        surface_fluid_properties=surface_fluid_properties,
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
        flow_rates=flow_rates,
        thps=thps,
        water_cuts=typing.cast(NumberArray[OneDimension], resolved_water_cuts),
        gas_oil_ratios=typing.cast(NumberArray[OneDimension], resolved_gas_oil_ratios),
        bhps=bhps,
        unit_system=unit_system,
    )
    return VFPTable(data, warn_on_extrapolation=warn_on_extrapolation, dtype=dtype)
