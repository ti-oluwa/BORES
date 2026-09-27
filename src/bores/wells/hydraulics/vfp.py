"""
VFP (vertical flow performance) tables.

Interpolates bottomhole pressure at a well's datum depth as a function of flow rate,
tubing head pressure, water cut, gas-oil ratio, and artificial lift quantity.
This is the datum-to-surface leg of well hydraulics only; connection-level
pressure distribution below the datum is unaffected by whether a table is assigned.
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

FiveDimensions: typing.TypeAlias = tuple[int, int, int, int, int]

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


@attrs.frozen(kw_only=True, slots=True)
class VFPTables(StoreSerializable):
    """
    Number-indexed collection of `VFPTable`s.

    Wells select a table by number, via `WCONPROD`/`WCONINJE`'s
    `vfp_table` item. Several wells commonly share one table.
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

    def table(self, table_number: Integer, /) -> VFPTable:
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
