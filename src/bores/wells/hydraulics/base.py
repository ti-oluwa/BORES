"""Shared wellbore hydraulics primitives."""

import math
import typing

import numba
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.constants import c
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.types import (
    Number,
    NumberArray,
    OneDimension,
    PhaseValues,
    UnitConversionTable,
    UnitSystem,
)
from bores.units import get_conversion_factors
from bores.wells.state import ConnectionSample

if typing.TYPE_CHECKING:
    from bores.wells.hydraulics.tubing_fluids import TubingFluidTable

__all__ = [
    "UNIT_SYSTEMS_BY_TAG",
    "UNIT_SYSTEM_TAGS",
    "ConnectionFluids",
    "ConnectionWalk",
    "PressureDrop",
    "SurfaceSample",
    "WellBoreModel",
    "WellBoreModelOptions",
    "compile_connection_fluids",
    "compute_friction_factor",
    "compute_hydrostatic_pressure",
    "compute_mixture_density",
    "compute_mixture_velocity",
    "compute_mixture_viscosity",
    "compute_segment_pressure_drop",
    "compute_static_hydrostatic_drop",
    "compute_static_mixture_density",
    "compute_superficial_velocity",
    "compute_surface_mixture_density",
    "compute_surface_mixture_viscosity",
    "get_gravitational_acceleration",
    "get_hydrostatic_divisor",
    "get_unit_system_constant",
    "plan_connection_walk",
    "require_static_column_density",
    "require_two_phase_surface_properties",
    "split_liquid_gas",
]

WellBoreModelOptions = typing.TypeVar("WellBoreModelOptions")
"""A wellbore models's own config type"""


class PressureDrop(typing.NamedTuple):
    """Pressure drop across one tubing segment, split by physical mechanism."""

    hydrostatic: Number
    """Pressure change from the weight of the fluid column."""

    friction: Number
    """Pressure loss from friction against the tubing wall."""

    acceleration: Number
    """Pressure change from the fluid speeding up or slowing down along the segment. Zero for a
    correlation that neglects it, which is every one except Beggs & Brill."""

    @property
    def total(self) -> Number:
        """The combined pressure drop across the segment."""
        return self.hydrostatic + self.friction + self.acceleration


class SurfaceSample(typing.NamedTuple):
    """
    Fluid properties at surface conditions, for computing tubing head pressure.

    Anything not given is `NaN`.
    """

    density: Number = math.nan
    """Mixture density at surface conditions. Required unless `phase_densities` is given."""

    viscosity: Number = math.nan
    """Mixture viscosity at surface conditions. Required unless `phase_viscosities` is given."""

    phase_densities: PhaseValues = PhaseValues(oil=math.nan, water=math.nan, gas=math.nan)
    """Density of each phase at surface conditions. Required for a
    two-phase slip correlation such as Beggs & Brill."""

    phase_viscosities: PhaseValues = PhaseValues(oil=math.nan, water=math.nan, gas=math.nan)
    """
    Viscosity of each phase at surface conditions. Required for a
    two-phase slip correlation such as Beggs & Brill.
    """

    gas_liquid_surface_tension: Number = math.nan
    """
    Surface tension between the gas and liquid phases. Required for a
    two-phase slip correlation such as Beggs & Brill.
    """


class WellBoreModel(typing.NamedTuple):
    """
    A wellbore hydraulics model: which correlation to use, and its configuration.
    """

    name: str
    """Which correlation this is: `"homogeneous"`, `"beggs_and_brill"`, or `"hagedorn_brown"`."""

    options: typing.Any
    """Wellbore model's/correlation's configuration."""

    tubing_fluid_table: "TubingFluidTable | None" = None
    """
    The fluid properties of the tubing against pressure. With them, the tubing head pressure is
    found by marching up the tubing with the fluids of each segment at its own pressure, from
    surface-condition rates. Without them, the tubing is taken as one segment, with the surface
    fluid properties of a `SurfaceSample`.
    """

    tubing_fluids_in_walk: bool = False
    """
    Whether the pressure at each connection is also found with the fluids read from
    `tubing_fluid_table` at the pressure in the wellbore, rather than at the pressure of the connected
    cell. Gas that expands and gas that comes out of solution in the wellbore then count in the
    velocities and densities between connections. Off by default, because the rates and the
    pressures then depend on each other more strongly, and a well probed at a very low flowing
    pressure can fail to converge.
    """

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Converts this model to a different unit system.

        :param target: Target unit system.
        :param table: Optional custom unit-conversion table, forwarded to `options.convert`.
        :returns: This model, with `options` converted to `target`.
        :raises ValueError: If `options` has no `convert` method.
        :raises ValidationError: If the model has `tubing_fluid_table` in another unit system. Build
            them again for `target` with `build_tubing_fluid_table`.
        """
        if not hasattr(self.options, "convert"):
            raise ValueError(
                f"{self.__class__.__name__} {self.name!r}'s options ({type(self.options).__name__}) "
                "has no convert method."
            )
        if self.tubing_fluid_table is not None and self.tubing_fluid_table.unit_system != target:
            raise ValidationError(
                f"This model's tubing fluids are in {self.tubing_fluid_table.unit_system.value} units. "
                f"Build them again for {target.value} units with `build_tubing_fluid_table`."
            )
        return self._replace(options=self.options.convert(target, table=table))  # type: ignore[attr-defined]


@numba.njit(cache=True)
def compute_mixture_density(rates: PhaseValues, phase_densities: PhaseValues) -> Number:
    """
    Computes the no-slip, rate-weighted mixture density of a multiphase stream.

    :param rates: Rate of each phase.
    :param phase_densities: Density of each phase, at the same condition as `rates`.
    :returns: Mixture density.
    :raises ValueError: If `rates` sums to zero. Use
        `compute_static_hydrostatic_drop` for a well with no flow.
    """
    total_rate = rates.oil + rates.water + rates.gas
    if total_rate == 0:
        raise ValueError("`rates` sums to zero; use the static no-flow case instead")
    return (
        rates.oil * phase_densities.oil
        + rates.water * phase_densities.water
        + rates.gas * phase_densities.gas
    ) / total_rate


@numba.njit(cache=True)
def compute_mixture_viscosity(rates: PhaseValues, phase_viscosities: PhaseValues) -> Number:
    """
    Computes the no-slip, rate-weighted mixture viscosity of a multiphase stream.

    :param rates: Rate of each phase.
    :param phase_viscosities: Viscosity of each phase, at the same condition as `rates`.
    :returns: Mixture viscosity.
    :raises ValueError: If `rates` sums to zero. Use
        `compute_static_hydrostatic_drop` for a well with no flow.
    """
    total_rate = rates.oil + rates.water + rates.gas
    if total_rate == 0:
        raise ValueError("`rates` sums to zero; use the static no-flow case instead")
    return (
        rates.oil * phase_viscosities.oil
        + rates.water * phase_viscosities.water
        + rates.gas * phase_viscosities.gas
    ) / total_rate


@numba.njit(cache=True)
def compute_superficial_velocity(
    rate: Number, tubing_inner_diameter: Number, rate_time_factor: Number
) -> Number:
    """
    Converts a single phase's rate to a superficial velocity in tubing.

    :param rate: Phase rate at reservoir conditions, in the model's own
        unit system's own reporting time unit (day for FIELD/METRIC,
        hour for LAB, second for SI - see `UnitConversionFactors.time`).
    :param tubing_inner_diameter: Tubing inner diameter, in the model's
        own unit system's length unit.
    :param rate_time_factor: `c.RATE_TIME_FACTOR_<unit_system>` -
        seconds per that system's own time unit. Converts `rate` to a
        per-second rate, so the resulting velocity is per second,
        matching the per-second `gravitational_acceleration` every
        wellbore correlation's hydrostatic term already uses, in every
        unit system (FIELD: ft/s; METRIC/SI: m/s; LAB: cm/s).
    :returns: Superficial velocity, per second, in the model's own
        unit system's length unit.
    :raises ValueError: If `tubing_inner_diameter` isn't positive.
    """
    if tubing_inner_diameter <= 0.0:
        raise ValueError("`tubing_inner_diameter` must be positive")
    cross_sectional_area = math.pi * (tubing_inner_diameter / 2.0) ** 2
    return (rate * rate_time_factor) / cross_sectional_area


@numba.njit(cache=True)
def compute_mixture_velocity(
    rates: PhaseValues, tubing_inner_diameter: Number, rate_time_factor: Number
) -> Number:
    """
    Computes the no-slip superficial velocity of a multiphase stream in tubing.

    :param rates: Rate of each phase at reservoir conditions, in
        the model's own unit system's own reporting time unit.
    :param tubing_inner_diameter: Tubing inner diameter.
    :param rate_time_factor: `c.RATE_TIME_FACTOR_<unit_system>` - see `compute_superficial_velocity`.
    :returns: Mixture velocity, per second.
    :raises ValueError: If `tubing_inner_diameter` isn't positive.
    """
    total_rate = rates.oil + rates.water + rates.gas
    return compute_superficial_velocity(
        rate=total_rate,
        tubing_inner_diameter=tubing_inner_diameter,
        rate_time_factor=rate_time_factor,
    )


@numba.njit(cache=True)
def compute_surface_mixture_density(
    density: Number, phase_densities: PhaseValues, rates: PhaseValues
) -> Number:
    """
    Gets the surface mixture density, working it out from the phase densities if no mixture
    density was given.

    :param density: Mixture density at surface conditions, or `NaN` if not given.
    :param phase_densities: Density of each phase at surface conditions, `NaN` if not given.
    :param rates: Rate of each phase, at surface conditions.
    :returns: Mixture density at surface conditions.
    :raises ValueError: If neither a mixture density nor all three phase densities were given.
    """
    if not math.isnan(density):
        return density
    if (
        math.isnan(phase_densities.oil)
        or math.isnan(phase_densities.water)
        or math.isnan(phase_densities.gas)
    ):
        raise ValueError("`SurfaceSample` needs density or `phase_densities`")
    return compute_mixture_density(rates=rates, phase_densities=phase_densities)


@numba.njit(cache=True)
def compute_surface_mixture_viscosity(
    viscosity: Number, phase_viscosities: PhaseValues, rates: PhaseValues
) -> Number:
    """
    Gets the surface mixture viscosity, working it out from the phase viscosities if no mixture
    viscosity was given.

    :param viscosity: Mixture viscosity at surface conditions, or `NaN` if not given.
    :param phase_viscosities: Viscosity of each phase at surface conditions, `NaN` if not given.
    :param rates: Rate of each phase, at surface conditions.
    :returns: Mixture viscosity at surface conditions.
    :raises ValueError: If neither a mixture viscosity nor all three phase viscosities were given.
    """
    if not math.isnan(viscosity):
        return viscosity
    if (
        math.isnan(phase_viscosities.oil)
        or math.isnan(phase_viscosities.water)
        or math.isnan(phase_viscosities.gas)
    ):
        raise ValueError("`SurfaceSample` needs viscosity or `phase_viscosities`")
    return compute_mixture_viscosity(rates=rates, phase_viscosities=phase_viscosities)


@numba.njit(cache=True)
def require_two_phase_surface_properties(
    phase_densities: PhaseValues,
    phase_viscosities: PhaseValues,
    gas_liquid_surface_tension: Number,
) -> None:
    """
    Checks that the surface properties a two-phase slip correlation needs were given.

    :param phase_densities: Density of each phase at surface conditions.
    :param phase_viscosities: Viscosity of each phase at surface conditions.
    :param gas_liquid_surface_tension: Surface tension between the gas and liquid phases.
    :raises ValueError: If any of them is `NaN`.
    """
    if (
        math.isnan(phase_densities.oil)
        or math.isnan(phase_densities.water)
        or math.isnan(phase_densities.gas)
    ):
        raise ValueError(
            "`SurfaceSample.phase_densities` is required for a two-phase wellbore model."
        )
    if (
        math.isnan(phase_viscosities.oil)
        or math.isnan(phase_viscosities.water)
        or math.isnan(phase_viscosities.gas)
    ):
        raise ValueError(
            "`SurfaceSample.phase_viscosities` is required for a two-phase wellbore model."
        )
    if math.isnan(gas_liquid_surface_tension):
        raise ValueError(
            "`SurfaceSample.gas_liquid_surface_tension` is required for a two-phase wellbore model."
        )


@numba.njit(cache=True)
def require_static_column_density(density: Number) -> Number:
    """
    Gets the density of the fluid column in a well with no flow.

    With no flow there are no rates to weight the phases by, so the column's density has to be
    given directly.

    :param density: Mixture density at surface conditions, `NaN` if not given.
    :returns: The same density.
    :raises ValueError: If no density was given.
    """
    if math.isnan(density):
        raise ValueError(
            "A well with no flow needs `SurfaceSample.density` to give the density of the "
            "fluid column."
        )
    return density


@numba.njit(cache=True)
def compute_friction_factor(
    reynolds_number: Number,
    relative_roughness: Number,
    method: int,
    laminar_reynolds_limit: Number,
    turbulent_reynolds_limit: Number,
    friction_max_iterations: int,
    friction_tolerance: Number,
) -> Number:
    """
    Computes the Darcy friction factor for flow in tubing.

    :param reynolds_number: Reynolds number of the flow. Must be positive.
    :param relative_roughness: Tubing roughness divided by tubing inner
        diameter. `0.0` for a smooth pipe.
    :param method: `0` for the simplified correlation, `1` for
        Colebrook iteration.
    :param laminar_reynolds_limit: Reynolds number below which flow is
        treated as laminar. Only used by the simplified correlation.
    :param turbulent_reynolds_limit: Reynolds number above which flow is
        treated as fully turbulent. Only used by the simplified correlation.
    :param friction_max_iterations: Maximum iterations. Only used by Colebrook.
    :param friction_tolerance: Convergence tolerance. Only used by Colebrook.
    :returns: Darcy friction factor.
    :raises ValueError: If `reynolds_number` isn't positive.
    """
    if reynolds_number <= 0.0:
        raise ValueError("`reynolds_number` must be positive")

    if method == 0:
        if reynolds_number < laminar_reynolds_limit:
            return 64.0 / reynolds_number
        if reynolds_number < turbulent_reynolds_limit:
            return 0.316 * reynolds_number**-0.25
        return 0.25 / (math.log10(relative_roughness / 3.7 + 5.74 / reynolds_number**0.9)) ** 2

    friction_factor = 0.02
    for _ in range(friction_max_iterations):
        rhs = -2.0 * math.log10(
            relative_roughness / 3.7 + 2.51 / (reynolds_number * math.sqrt(friction_factor))
        )
        updated = 1.0 / rhs**2
        if abs(updated - friction_factor) < friction_tolerance:
            friction_factor = updated
            break
        friction_factor = updated
    return friction_factor


def get_unit_system_constant(prefix: str, unit_system: UnitSystem) -> Number:
    """
    Looks up a per-unit-system constant from `bores.constants`.

    :param prefix: The constant's name, without the unit system suffix -
        e.g. `"GRAVITATIONAL_FACTOR"`.
    :param unit_system: Selects which per-system constant to return.
    :returns: `f"{prefix}_{unit_system.name}"` from `bores.constants.c`.
    """
    return getattr(c, f"{prefix}_{unit_system.name.upper()}")


UNIT_SYSTEM_TAGS: typing.Mapping[UnitSystem, int] = {
    unit_system: tag for tag, unit_system in enumerate(UnitSystem)
}
"""
Integer tag of each unit system. A wellbore model keeps its unit system as this tag so that the
tuple passed to jitted functions holds only numbers, which numba reads far faster than an enum.
"""

UNIT_SYSTEMS_BY_TAG: tuple[UnitSystem, ...] = tuple(UnitSystem)
"""Unit system of each tag in `UNIT_SYSTEM_TAGS`."""


def get_gravitational_acceleration(unit_system: UnitSystem) -> Number:
    """
    Standard gravity in a unit system's length and time units.

    :param unit_system: The unit system to express it in.
    :returns: Acceleration due to gravity.
    """
    gravity = typing.cast(Number, c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE)
    if unit_system != UnitSystem.FIELD:
        gravity = gravity * get_conversion_factors(UnitSystem.FIELD, unit_system)["length"]
    return gravity


def compute_hydrostatic_pressure(
    *,
    density: Number,
    gravitational_acceleration: Number,
    length: Number,
    unit_system: UnitSystem,
    gravitational_factor: Number | None = None,
    hydrostatic_area_factor: Number | None = None,
) -> Number:
    """
    Computes the hydrostatic pressure of a fluid column.

    :param density: Column density.
    :param gravitational_acceleration: Acceleration due to gravity, in `unit_system`'s units.
    :param length: Column length.
    :param unit_system: The unit system `density`/`gravitational_acceleration`/`length` are in.
    :param gravitational_factor: `c.GRAVITATIONAL_FACTOR_<unit_system>` if not given.
    :param hydrostatic_area_factor: `c.HYDROSTATIC_AREA_FACTOR_<unit_system>` if not given.
    :returns: Hydrostatic pressure, in `unit_system`'s pressure unit.
    """
    resolved_gravitational_constant = (
        gravitational_factor
        if gravitational_factor is not None
        else get_unit_system_constant(prefix="GRAVITATIONAL_FACTOR", unit_system=unit_system)
    )
    resolved_area_factor = (
        hydrostatic_area_factor
        if hydrostatic_area_factor is not None
        else get_unit_system_constant(prefix="HYDROSTATIC_AREA_FACTOR", unit_system=unit_system)
    )
    return (density * gravitational_acceleration * length) / (
        resolved_gravitational_constant * resolved_area_factor
    )


class ConnectionFluids(typing.NamedTuple):
    """
    Reservoir conditions at every connection of a well, one array per property, in the order of
    the well's connections.

    This is what a well is resolved against. `compile_connection_fluids` builds it from
    `ConnectionSample` objects, one per connection, when that is how the conditions are at hand.
    """

    pressures: NumberArray[OneDimension]
    """Pressure at each connection."""

    gas_liquid_surface_tensions: NumberArray[OneDimension]
    """Gas-liquid surface tension at each connection."""

    oil_densities: NumberArray[OneDimension]
    """Oil density at each connection."""

    water_densities: NumberArray[OneDimension]
    """Water density at each connection."""

    gas_densities: NumberArray[OneDimension]
    """Gas density at each connection."""

    oil_viscosities: NumberArray[OneDimension]
    """Oil viscosity at each connection."""

    water_viscosities: NumberArray[OneDimension]
    """Water viscosity at each connection."""

    gas_viscosities: NumberArray[OneDimension]
    """Gas viscosity at each connection."""

    oil_saturations: NumberArray[OneDimension]
    """Oil saturation at each connection."""

    water_saturations: NumberArray[OneDimension]
    """Water saturation at each connection."""

    gas_saturations: NumberArray[OneDimension]
    """Gas saturation at each connection."""

    oil_mobilities: NumberArray[OneDimension]
    """Reservoir-condition oil mobility at each connection."""

    water_mobilities: NumberArray[OneDimension]
    """Reservoir-condition water mobility at each connection."""

    gas_mobilities: NumberArray[OneDimension]
    """Reservoir-condition gas mobility at each connection."""

    oil_formation_volume_factors: NumberArray[OneDimension]
    """Oil formation volume factor at each connection's current pressure."""

    water_formation_volume_factors: NumberArray[OneDimension]
    """Water formation volume factor at each connection's current pressure."""

    gas_formation_volume_factors: NumberArray[OneDimension]
    """Gas formation volume factor at each connection's current pressure."""

    solution_gas_oil_ratios: NumberArray[OneDimension]
    """Gas the oil holds in solution at each connection, per stock-tank barrel of oil."""


class ConnectionWalk(typing.NamedTuple):
    """
    The segments of the walk outward from a well's reference depth to every connection, one entry
    per segment, downward walk first.
    """

    connection_indices: NumberArray[OneDimension]
    """Index of the connection each segment ends at."""

    previous_indices: NumberArray[OneDimension]
    """Index of the connection each segment starts at, or `-1` if it starts at the reference depth."""

    vertical_distances: NumberArray[OneDimension]
    """Vertical extent of each segment."""

    segment_lengths: NumberArray[OneDimension]
    """Length of each segment along the wellbore. Never less than its vertical extent."""

    geometric_signs: NumberArray[OneDimension]
    """`1` where a segment ends deeper than it starts, `-1` where shallower."""

    carried_oil_rates: NumberArray[OneDimension]
    """Oil rate flowing through each segment, at reservoir conditions."""

    carried_water_rates: NumberArray[OneDimension]
    """Water rate flowing through each segment, at reservoir conditions."""

    carried_gas_rates: NumberArray[OneDimension]
    """Gas rate flowing through each segment, at reservoir conditions."""


def compile_connection_fluids(
    *, connection_samples: typing.Sequence[ConnectionSample], dtype: npt.DTypeLike = None
) -> ConnectionFluids:
    """
    Compiles a well's connection samples into the arrays it is resolved against.

    :param connection_samples: One `ConnectionSample` per connection.
    :param dtype: Array data type. `bores.precision.get_dtype()` if not given.
    :returns: The conditions at every connection.
    """
    resolved_dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    count = len(connection_samples)

    def gather(read: typing.Callable[[ConnectionSample], Number]) -> NumberArray[OneDimension]:
        return typing.cast(
            NumberArray[OneDimension],
            np.fromiter((read(sample) for sample in connection_samples), resolved_dtype, count),
        )

    return ConnectionFluids(
        pressures=gather(lambda sample: sample.pressure),
        gas_liquid_surface_tensions=gather(lambda sample: sample.gas_liquid_surface_tension),
        oil_densities=gather(lambda sample: sample.phase_densities.oil),
        water_densities=gather(lambda sample: sample.phase_densities.water),
        gas_densities=gather(lambda sample: sample.phase_densities.gas),
        oil_viscosities=gather(lambda sample: sample.phase_viscosities.oil),
        water_viscosities=gather(lambda sample: sample.phase_viscosities.water),
        gas_viscosities=gather(lambda sample: sample.phase_viscosities.gas),
        oil_saturations=gather(lambda sample: sample.phase_saturations.oil),
        water_saturations=gather(lambda sample: sample.phase_saturations.water),
        gas_saturations=gather(lambda sample: sample.phase_saturations.gas),
        oil_mobilities=gather(lambda sample: sample.phase_mobilities.oil),
        water_mobilities=gather(lambda sample: sample.phase_mobilities.water),
        gas_mobilities=gather(lambda sample: sample.phase_mobilities.gas),
        oil_formation_volume_factors=gather(lambda sample: sample.phase_fvfs.oil),
        water_formation_volume_factors=gather(lambda sample: sample.phase_fvfs.water),
        gas_formation_volume_factors=gather(lambda sample: sample.phase_fvfs.gas),
        solution_gas_oil_ratios=gather(lambda sample: sample.solution_gas_oil_ratio),
    )


def get_hydrostatic_divisor(unit_system: UnitSystem) -> Number:
    """
    The divisor that turns `density * gravity * length` into a pressure in `unit_system`: the
    product of its gravitational factor and hydrostatic area factor, as the constants are set
    now.

    :param unit_system: The unit system the column is in.
    :returns: The divisor.
    """
    return get_unit_system_constant(
        prefix="GRAVITATIONAL_FACTOR", unit_system=unit_system
    ) * get_unit_system_constant(prefix="HYDROSTATIC_AREA_FACTOR", unit_system=unit_system)


@numba.njit(cache=True)
def plan_connection_walk(
    reference_depth: Number,
    reference_measured_depth: Number,
    representative_depths: NumberArray[OneDimension],
    representative_measured_depths: NumberArray[OneDimension],
    connection_oil_rates: NumberArray[OneDimension],
    connection_water_rates: NumberArray[OneDimension],
    connection_gas_rates: NumberArray[OneDimension],
) -> ConnectionWalk:
    """
    Plans the walk outward from a well's reference depth to every connection, both down and up.

    Each segment carries the flow of every connection beneath it: a producer's flow rises past
    them, and an injector's falls to them. So moving down from the reference depth, a segment
    carries the connection it ends at and every deeper one. Moving up, it carries every
    connection at or below the reference depth, plus every connection above it that the walk has
    already passed. A connection exactly at the reference depth belongs to the downward walk.
    Connections at the same depth are visited in index order.

    The reference depths are taken in the data type of the depth arrays, so a connection at the
    reference depth is at it exactly, whatever precision the reference was given in.

    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_measured_depth: Measured depth along the wellbore of `reference_depth`.
    :param representative_depths: One vertical depth per connection.
    :param representative_measured_depths: One measured depth along the wellbore per connection.
    :param connection_oil_rates: Each connection's own oil rate, at reservoir conditions.
    :param connection_water_rates: Each connection's own water rate, at reservoir conditions.
    :param connection_gas_rates: Each connection's own gas rate, at reservoir conditions.
    :returns: The segments.
    """
    reference_depth = representative_depths.dtype.type(reference_depth)
    reference_measured_depth = representative_measured_depths.dtype.type(reference_measured_depth)
    count = representative_depths.shape[0]
    ascending = np.argsort(representative_depths, kind="mergesort")
    descending = np.argsort(-representative_depths, kind="mergesort")
    below_count = 0
    for k in range(count):
        if representative_depths[k] >= reference_depth:
            below_count += 1

    connection_indices = np.empty(count, dtype=np.int64)
    previous_indices = np.empty(count, dtype=np.int64)
    vertical_distances = np.empty(count, dtype=representative_depths.dtype)
    segment_lengths = np.empty(count, dtype=representative_depths.dtype)
    geometric_signs = np.empty(count, dtype=representative_depths.dtype)
    carried_oil_rates = np.empty(count, dtype=connection_oil_rates.dtype)
    carried_water_rates = np.empty(count, dtype=connection_water_rates.dtype)
    carried_gas_rates = np.empty(count, dtype=connection_gas_rates.dtype)

    below_start = count - below_count
    oil_beneath = connection_oil_rates.dtype.type(0.0)
    water_beneath = connection_water_rates.dtype.type(0.0)
    gas_beneath = connection_gas_rates.dtype.type(0.0)
    for step in range(below_count - 1, -1, -1):
        idx = ascending[below_start + step]
        oil_beneath = oil_beneath + connection_oil_rates[idx]
        water_beneath = water_beneath + connection_water_rates[idx]
        gas_beneath = gas_beneath + connection_gas_rates[idx]
        carried_oil_rates[step] = oil_beneath
        carried_water_rates[step] = water_beneath
        carried_gas_rates[step] = gas_beneath

    previous_index = -1
    previous_depth = reference_depth
    previous_measured_depth = reference_measured_depth
    for step in range(below_count):
        idx = ascending[below_start + step]
        vertical_distance = abs(representative_depths[idx] - previous_depth)
        connection_indices[step] = idx
        previous_indices[step] = previous_index
        vertical_distances[step] = vertical_distance
        segment_lengths[step] = max(
            abs(representative_measured_depths[idx] - previous_measured_depth), vertical_distance
        )
        geometric_signs[step] = 1.0
        previous_index = idx
        previous_depth = representative_depths[idx]
        previous_measured_depth = representative_measured_depths[idx]

    oil_carried = oil_beneath
    water_carried = water_beneath
    gas_carried = gas_beneath
    previous_index = -1
    previous_depth = reference_depth
    previous_measured_depth = reference_measured_depth
    for step in range(below_count, count):
        idx = descending[step]
        vertical_distance = abs(representative_depths[idx] - previous_depth)
        connection_indices[step] = idx
        previous_indices[step] = previous_index
        vertical_distances[step] = vertical_distance
        segment_lengths[step] = max(
            abs(representative_measured_depths[idx] - previous_measured_depth), vertical_distance
        )
        geometric_signs[step] = -1.0
        carried_oil_rates[step] = oil_carried
        carried_water_rates[step] = water_carried
        carried_gas_rates[step] = gas_carried
        oil_carried = oil_carried + connection_oil_rates[idx]
        water_carried = water_carried + connection_water_rates[idx]
        gas_carried = gas_carried + connection_gas_rates[idx]
        previous_index = idx
        previous_depth = representative_depths[idx]
        previous_measured_depth = representative_measured_depths[idx]

    return ConnectionWalk(
        connection_indices=connection_indices,
        previous_indices=previous_indices,
        vertical_distances=vertical_distances,  # type: ignore[arg-type]
        segment_lengths=segment_lengths,  # type: ignore[arg-type]
        geometric_signs=geometric_signs,  # type: ignore[arg-type]
        carried_oil_rates=carried_oil_rates,  # type: ignore[arg-type]
        carried_water_rates=carried_water_rates,  # type: ignore[arg-type]
        carried_gas_rates=carried_gas_rates,  # type: ignore[arg-type]
    )


@numba.njit(cache=True)
def compute_segment_pressure_drop(
    length: Number,
    inclination_from_vertical: Number,
    tubing_inner_diameter: Number,
    tubing_roughness: Number,
    mixture_density: Number,
    mixture_viscosity: Number,
    mixture_velocity: Number,
    gravitational_acceleration: Number,
    hydrostatic_scale: Number,
    viscosity_scale: Number,
    method: int,
    laminar_reynolds_limit: Number,
    turbulent_reynolds_limit: Number,
    friction_max_iterations: int,
    friction_tolerance: Number,
) -> PressureDrop:
    """
    Computes the pressure drop across one tubing segment, using no-slip
    mixture properties throughout.

    :param length: Along-wellbore segment length.
    :param inclination_from_vertical: Segment inclination, in radians. `0` is vertical.
    :param tubing_inner_diameter: Tubing inner diameter.
    :param tubing_roughness: Absolute pipe roughness. `NaN` for a smooth pipe.
    :param mixture_density: No-slip mixture density for this segment.
    :param mixture_viscosity: No-slip mixture viscosity for this segment.
    :param mixture_velocity: Superficial mixture velocity in the segment.
    :param gravitational_acceleration: Acceleration due to gravity, already unit-resolved.
    :param hydrostatic_scale: Unit-conversion factor converting a
        `density * velocity-squared` or `density * gravitational_acceleration
        * length` term into the caller's own pressure unit -
        `1 / (gravitational_factor * hydrostatic_area_factor)` for that
        unit system. Applied to both the hydrostatic and the friction
        term, since both are that same kind of quantity before
        conversion; without it, friction comes out several thousand
        times too large relative to hydrostatic in field units.
    :param viscosity_scale: `c.VISCOSITY_SCALE_<unit_system>` - converts
        `mixture_viscosity` from that unit system's reporting unit (cP
        for FIELD/METRIC/LAB, already Pa*s for SI) to the coherent unit
        the Reynolds number needs to come out dimensionless, matching
        `mixture_density`/velocity/`tubing_inner_diameter`.
    :param method: Forwarded to `compute_friction_factor`.
    :param laminar_reynolds_limit: Forwarded to `compute_friction_factor`.
    :param turbulent_reynolds_limit: Forwarded to `compute_friction_factor`.
    :param friction_max_iterations: Forwarded to `compute_friction_factor`.
    :param friction_tolerance: Forwarded to `compute_friction_factor`.
    :returns: Pressure drop for this segment.
    """
    vertical_length = length * math.cos(inclination_from_vertical)
    hydrostatic_drop = (
        mixture_density * gravitational_acceleration * vertical_length * hydrostatic_scale
    )

    mean_velocity = mixture_velocity
    relative_roughness = (
        0.0 if math.isnan(tubing_roughness) else tubing_roughness / tubing_inner_diameter
    )
    reynolds_number = (
        mixture_density
        * abs(mean_velocity)
        * tubing_inner_diameter
        / (mixture_viscosity * viscosity_scale)
    )
    if reynolds_number <= 0.0:
        friction_drop = 0.0
    else:
        friction_factor = compute_friction_factor(
            reynolds_number=reynolds_number,
            relative_roughness=relative_roughness,
            method=method,
            laminar_reynolds_limit=laminar_reynolds_limit,
            turbulent_reynolds_limit=turbulent_reynolds_limit,
            friction_max_iterations=friction_max_iterations,
            friction_tolerance=friction_tolerance,
        )
        friction_drop = (
            friction_factor
            * (length / tubing_inner_diameter)
            * (mixture_density * mean_velocity**2 / 2.0)
            * hydrostatic_scale
        )

    # The acceleration (kinetic energy) term is neglected. Evaluating it needs the fluid's velocity at
    # both ends of the segment, which means fluid properties at the flowing pressure at each end, and
    # the connection fluids are only known at reservoir conditions. It is small for oil wells, but a
    # high gas rate at a low wellhead pressure can make it matter.
    return PressureDrop(
        hydrostatic=hydrostatic_drop,
        friction=friction_drop,
        acceleration=0.0,
    )


@numba.njit(cache=True)
def split_liquid_gas(
    rates: PhaseValues,
    phase_densities: PhaseValues,
    phase_viscosities: PhaseValues,
) -> tuple[Number, Number, Number, Number, Number, Number]:
    """
    Pools oil and water into a combined liquid phase, for a two-phase
    correlation such as Beggs & Brill or Hagedorn & Brown that treats
    liquid and gas as the only two phases and doesn't distinguish oil from water.

    :param rates: Rate of each phase.
    :param phase_densities: Density of each phase, at the same condition as `rates`.
    :param phase_viscosities: Viscosity of each phase, at the same condition as `rates`.
    :returns: `(liquid_rate, gas_rate, liquid_density, gas_density, liquid_viscosity, gas_viscosity)`.
    """
    liquid_rate = rates.oil + rates.water
    gas_rate = rates.gas
    if liquid_rate > 0.0:
        liquid_density = (
            rates.oil * phase_densities.oil + rates.water * phase_densities.water
        ) / liquid_rate
        liquid_viscosity = (
            rates.oil * phase_viscosities.oil + rates.water * phase_viscosities.water
        ) / liquid_rate
    else:
        liquid_density = phase_densities.oil
        liquid_viscosity = phase_viscosities.oil
    return (
        liquid_rate,
        gas_rate,
        liquid_density,
        phase_densities.gas,
        liquid_viscosity,
        phase_viscosities.gas,
    )


def compute_static_hydrostatic_drop(
    mixture_density: Number,
    length: Number,
    gravitational_acceleration: Number,
    *,
    unit_system: UnitSystem,
    gravitational_factor: Number | None = None,
    hydrostatic_area_factor: Number | None = None,
) -> PressureDrop:
    """
    Computes the pressure drop across a static (no-flow) tubing column.

    :param mixture_density: Static column density.
    :param length: Vertical column length.
    :param gravitational_acceleration: Acceleration due to gravity, already unit-resolved.
    :param unit_system: Forwarded to `compute_hydrostatic_pressure`.
    :param gravitational_factor: Forwarded to `compute_hydrostatic_pressure`.
    :param hydrostatic_area_factor: Forwarded to `compute_hydrostatic_pressure`.
    :returns: Pressure drop with `friction=0` and `acceleration=0`.
    """
    return PressureDrop(
        hydrostatic=compute_hydrostatic_pressure(
            density=mixture_density,
            gravitational_acceleration=gravitational_acceleration,
            length=length,
            unit_system=unit_system,
            gravitational_factor=gravitational_factor,
            hydrostatic_area_factor=hydrostatic_area_factor,
        ),
        friction=0.0,
        acceleration=0.0,
    )


@numba.njit(cache=True)
def compute_static_mixture_density(
    phase_saturations: PhaseValues, phase_densities: PhaseValues
) -> Number:
    """
    Computes the saturation-weighted mixture density of a static (no-flow) column.

    :param phase_saturations: Saturation of each phase.
    :param phase_densities: Density of each phase.
    :returns: Saturation-weighted mixture density.
    :raises ValueError: If `phase_saturations` sums to zero.
    """
    total_saturation = phase_saturations.oil + phase_saturations.water + phase_saturations.gas
    if total_saturation == 0:
        raise ValueError("`phase_saturations` sums to zero; cannot derive a static density")
    return (
        phase_saturations.oil * phase_densities.oil
        + phase_saturations.water * phase_densities.water
        + phase_saturations.gas * phase_densities.gas
    ) / total_saturation
