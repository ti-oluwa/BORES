"""
Eclipse-style VFP (vertical flow performance) tables: bottomhole
pressure at a well's datum depth as a function of flow rate, tubing
head pressure, water cut, gas-oil ratio, and artificial lift quantity.
This is the datum-to-surface leg of well hydraulics only; connection-level
pressure distribution below the datum is unaffected by whether a table
is assigned.

Real VFPPROD/VFPINJ deck parsing isn't implemented here yet. This
module covers the table representation, in-memory construction, and
lookup.
"""

import logging
import typing

import attrs
import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator  # type: ignore[import-untyped]
from typing_extensions import Self

from bores.constants import UnitConversionTable, get_conversion_factors
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.serde.base import Serializable
from bores.serde.stores.base import StoreSerializable
from bores.types import (
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

__all__ = ["VFPData", "VFPTable", "VFPTables"]

logger = logging.getLogger(__name__)

FiveDimensions: typing.TypeAlias = tuple[int, int, int, int, int]

_AXIS_NAMES = ("flow_rate", "thp", "water_cut", "gas_oil_ratio", "artificial_lift_quantity")
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
    """Table number, matching deck `VFPPROD`/`VFPINJ` item 1. A well
    selects this table through `WCONPROD`/`WCONINJE`'s `vfp_table` item."""

    well_type: WellType
    """Whether this is a `VFPPROD` (producer) or `VFPINJ` (injector) table."""

    datum_depth: Number
    """Reference depth this table's bottomhole pressure is reported at,
    matching the owning well's `reference_depth`."""

    flow_rates: NumberArray[OneDimension]
    """Flow rate axis, strictly ascending. Liquid rate for most producer
    tables; the injected phase's rate for an injector table."""

    thps: NumberArray[OneDimension]
    """Tubing head pressure axis, strictly ascending."""

    water_cuts: NumberArray[OneDimension] = attrs.field(
        factory=lambda: typing.cast(NumberArray[OneDimension], np.array([0.0]))
    )
    """Water cut axis, strictly ascending. A single value (the default)
    for a table with no water-cut dependence."""

    gas_oil_ratios: NumberArray[OneDimension] = attrs.field(
        factory=lambda: typing.cast(NumberArray[OneDimension], np.array([0.0]))
    )
    """Gas-oil ratio axis, strictly ascending. A single value (the
    default) for a table with no GOR dependence."""

    artificial_lift_quantities: NumberArray[OneDimension] = attrs.field(
        factory=lambda: typing.cast(NumberArray[OneDimension], np.array([0.0]))
    )
    """Artificial lift quantity axis (gas-lift injection rate, pump
    power, etc.), strictly ascending. A single value (the default) for
    a table with no artificial-lift dependence. Not unit-converted by
    `convert()`; see its docstring."""

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
                f"{dict(zip(_AXIS_NAMES, expected_shape, strict=True))}."
            )
        for name, axis in zip(_AXIS_NAMES, axes, strict=True):
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
        self.dtype = np.dtype(dtype) if dtype is not None else np.dtype(get_dtype())

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
            for name, axis, active in zip(_AXIS_NAMES, axes, self._active, strict=True)
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
        for name, value in zip(_AXIS_NAMES, values, strict=True):
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
                    float(value_array.min()),
                    float(value_array.max()),
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
            flow_rate, thp, water_cut, gas_oil_ratio, artificial_lift_quantity
        )

        full_point = (flow_rate, thp, water_cut, gas_oil_ratio, artificial_lift_quantity)
        active_point = tuple(
            value for value, active in zip(full_point, self._active, strict=True) if active
        )
        is_scalar = all(np.isscalar(value) for value in active_point)

        arrays = [np.atleast_1d(value) for value in active_point]
        broadcast_shape = np.broadcast_shapes(*(arr.shape for arr in arrays))
        arrays = [np.broadcast_to(arr, broadcast_shape) for arr in arrays]
        points = np.column_stack([arr.ravel() for arr in arrays])
        result = self._interpolator(points).reshape(broadcast_shape)

        dtype = self.dtype
        if is_scalar:
            return typing.cast(Number, result.astype(dtype, copy=False).item())
        return typing.cast(NumberArray[NDimension], result.astype(dtype, copy=False))


@attrs.frozen(kw_only=True, slots=True)
class VFPTables(StoreSerializable):
    """
    Number-indexed collection of `VFPTable`s.

    Wells select a table by number, via `WCONPROD`/`WCONINJE`'s
    `vfp_table` item, not by well name; several wells commonly share
    one table.
    """

    tables: typing.Mapping[Integer, VFPTable] = attrs.field(factory=dict)
    """VFP table, keyed by its own `table_number`."""

    def __attrs_post_init__(self) -> None:
        for table_number, table in self.tables.items():
            if table.table_number != table_number:
                raise ValidationError(
                    f"`tables[{table_number!r}].table_number` is "
                    f"{table.table_number!r}, not {table_number!r}."
                )

    @property
    def unit_system(self) -> UnitSystem | None:
        """Unit system shared by every table, or `None` if empty."""
        return next((table.unit_system for table in self.tables.values()), None)

    def table(self, table_number: Integer) -> VFPTable:
        """
        Gets a table by number.

        :param table_number: The table's number, deck `VFPPROD`/`VFPINJ` item 1.
        :returns: The matching `VFPTable`.
        :raises ValidationError: If no table with that number exists.
        """
        if table_number not in self.tables:
            raise ValidationError(
                f"No VFP table numbered {table_number!r}. Available: {sorted(self.tables.keys())}."
            )
        return self.tables[table_number]

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
            tables={
                number: vfp_table.convert(target, table=table)
                for number, vfp_table in self.tables.items()
            },
        )
