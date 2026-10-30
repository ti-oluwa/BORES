"""No-slip mixture wellbore hydraulics."""

import typing

import numba
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.constants import c
from bores.types import (
    Boolean,
    FrictionMethod,
    Integer,
    Number,
    NumberArray,
    OneDimension,
    PhaseValues,
    UnitConversionTable,
    UnitSystem,
)
from bores.units import get_conversion_factors
from bores.utils import scale
from bores.wells.hydraulics.base import (
    UNIT_SYSTEM_TAGS,
    UNIT_SYSTEMS_BY_TAG,
    ConnectionFluids,
    PressureDrop,
    SurfaceSample,
    WellBoreModel,
    compute_mixture_density,
    compute_mixture_velocity,
    compute_mixture_viscosity,
    compute_segment_pressure_drop,
    compute_static_mixture_density,
    compute_surface_mixture_density,
    compute_surface_mixture_viscosity,
    get_hydrostatic_divisor,
    get_unit_system_constant,
    plan_connection_walk,
    require_static_column_density,
    validate_tubing_fluid_options,
)
from bores.wells.hydraulics.tubing_fluids import (
    MARCH_ITERATIONS,
    TubingFluidTable,
    compute_connection_surface_rates,
    compute_no_slip_mixture,
)

__all__ = [
    "HomogeneousWellbore",
    "compute_perforation_pressures",
    "compute_segment_drop",
    "compute_tubing_head_pressure",
    "compute_tubing_head_pressure_from_table",
    "homogeneous_wellbore",
    "march_segment_from_table",
    "walk_perforation_pressures_from_table",
]


class HomogeneousWellbore(typing.NamedTuple):
    """Configuration for the homogeneous (no-slip) wellbore hydraulics model."""

    tubing_inner_diameter: Number
    """Tubing inner diameter."""

    tubing_roughness: Number
    """Absolute pipe roughness. `NaN` for a smooth pipe."""

    friction_method: int
    """
    Which friction-factor correlation to use: `0` for the simplified
    correlation, `1` for Colebrook.
    """

    gravitational_acceleration: Number
    """Acceleration due to gravity, in this model's unit system."""

    laminar_reynolds_limit: Number
    """Reynolds number below which flow is treated as laminar."""

    turbulent_reynolds_limit: Number
    """Reynolds number above which flow is treated as fully turbulent."""

    friction_max_iterations: int
    """Maximum iterations for the Colebrook friction-factor calculation."""

    friction_tolerance: Number
    """Convergence tolerance for the Colebrook friction-factor calculation."""

    rate_time_factor: Number
    """
    Multiplier converting a phase rate, in this model's own unit system,
    to a per-second rate. Turns a rate into a velocity that matches the
    per-second `gravitational_acceleration`.
    """

    viscosity_scale: Number
    """
    `c.VISCOSITY_SCALE_<unit_system>` - converts a viscosity from this
    unit system's reporting unit (cP for FIELD/METRIC/LAB, already Pa*s
    for SI) to the coherent unit the Reynolds number needs.
    """

    hydrostatic_scale: Number
    """
    Unit-conversion factor converting a `density * velocity-squared` or
    `density * gravitational_acceleration * length` term into this
    model's own pressure unit. Applied to both the hydrostatic and the
    friction term, since both are that same kind of quantity before
    conversion.
    """

    unit_system_tag: int
    """
    Tag of this model's unit system in `UNIT_SYSTEM_TAGS`. Held as a number so the model stays
    cheap to pass to jitted functions; read it as `unit_system`.
    """

    @property
    def unit_system(self) -> UnitSystem:
        """This model's unit system."""
        return UNIT_SYSTEMS_BY_TAG[self.unit_system_tag]

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
        :param table: Optional custom unit-conversion table.
        :returns: This model, converted to `target`.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        length_factor = factors["length"]
        return self._replace(
            tubing_inner_diameter=scale(self.tubing_inner_diameter, length_factor),
            tubing_roughness=scale(self.tubing_roughness, length_factor),
            gravitational_acceleration=scale(self.gravitational_acceleration, length_factor),
            rate_time_factor=get_unit_system_constant(
                prefix="RATE_TIME_FACTOR", unit_system=target
            ),
            viscosity_scale=get_unit_system_constant(prefix="VISCOSITY_SCALE", unit_system=target),
            hydrostatic_scale=1.0
            / (
                get_unit_system_constant(prefix="GRAVITATIONAL_FACTOR", unit_system=target)
                * get_unit_system_constant(prefix="HYDROSTATIC_AREA_FACTOR", unit_system=target)
            ),
            unit_system_tag=UNIT_SYSTEM_TAGS[target],
        )


def homogeneous_wellbore(
    *,
    tubing_inner_diameter: Number,
    tubing_roughness: Number | None = None,
    friction_method: FrictionMethod = "simplified",
    unit_system: UnitSystem = UnitSystem.FIELD,
    gravitational_acceleration: Number | None = None,
    laminar_reynolds_limit: Number | None = None,
    turbulent_reynolds_limit: Number | None = None,
    friction_max_iterations: int | None = None,
    friction_tolerance: Number | None = None,
    tubing_fluid_table: TubingFluidTable | None = None,
    use_tubing_fluid_table: bool = False,
) -> WellBoreModel:
    """
    Builds a `WellBoreModel` wrapping a fully configured `HomogeneousWellbore`.

    This is a no-slip model. It treats oil, water, and gas as one mixed
    fluid moving at a single shared velocity, with no allowance for gas
    slipping past liquid. It doesn't identify flow regimes or use holdup
    correlations the way Beggs & Brill or Hagedorn & Brown do. It's the
    simplest of the three correlations here, and works reasonably well at
    high flow rates where the phases are well mixed, but is less reliable
    at low rates where slip matters more.

    :param tubing_inner_diameter: Tubing inner diameter.
    :param tubing_roughness: Absolute pipe roughness. `None` for a smooth pipe.
    :param friction_method: Which friction-factor correlation to use.
    :param unit_system: This model's unit system.
    :param gravitational_acceleration: Acceleration due to gravity. Resolved
        from `unit_system`'s standard gravity if not given.
    :param laminar_reynolds_limit: Reynolds number below which flow is
        treated as laminar. `c.WELLBORE_LAMINAR_REYNOLDS_LIMIT` if not given.
    :param turbulent_reynolds_limit: Reynolds number above which flow is
        treated as fully turbulent. `c.WELLBORE_TURBULENT_REYNOLDS_LIMIT`
        if not given.
    :param friction_max_iterations: Maximum Colebrook iterations.
        `c.COLEBROOK_MAX_ITERATIONS` if not given.
    :param friction_tolerance: Colebrook convergence tolerance.
        `c.COLEBROOK_TOLERANCE` if not given.
    :param tubing_fluid_table: The fluid properties of the tubing against pressure, from
        `build_tubing_fluid_table`. With it, the tubing head pressure is found by marching up the
        tubing from surface-condition rates, with the fluids of each segment at its own pressure.
        Without it, the tubing is taken as one segment with the fluids of a `SurfaceSample`. Must
        be in `unit_system`.
    :param use_tubing_fluid_table: Whether the pressure at each connection is also found with the
        fluids read from `tubing_fluid_table` at the pressure in the wellbore, rather than at the
        pressure of the connected cell. Needs `tubing_fluid_table`.
    :raises ValidationError: If `tubing_fluid_table` is not in `unit_system`, or
        `use_tubing_fluid_table` is set without a `tubing_fluid_table`.
    :returns: `WellBoreModel(name="homogeneous", options=<HomogeneousWellbore>)`.
    """
    validate_tubing_fluid_options(
        tubing_fluid_table=tubing_fluid_table,
        use_tubing_fluid_table=use_tubing_fluid_table,
        unit_system=unit_system,
    )
    if gravitational_acceleration is None:
        gravitational_acceleration = typing.cast(
            Number, c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE
        )
        if unit_system != UnitSystem.FIELD:
            factors = get_conversion_factors(UnitSystem.FIELD, unit_system)
            gravitational_acceleration = gravitational_acceleration * factors["length"]

    options = HomogeneousWellbore(
        tubing_inner_diameter=tubing_inner_diameter,
        tubing_roughness=tubing_roughness if tubing_roughness is not None else float("nan"),
        friction_method=1 if friction_method == "colebrook" else 0,
        gravitational_acceleration=typing.cast(Number, gravitational_acceleration),
        laminar_reynolds_limit=(
            laminar_reynolds_limit
            if laminar_reynolds_limit is not None
            else c.WELLBORE_LAMINAR_REYNOLDS_LIMIT
        ),
        turbulent_reynolds_limit=(
            turbulent_reynolds_limit
            if turbulent_reynolds_limit is not None
            else c.WELLBORE_TURBULENT_REYNOLDS_LIMIT
        ),
        friction_max_iterations=(
            friction_max_iterations
            if friction_max_iterations is not None
            else c.COLEBROOK_MAX_ITERATIONS
        ),
        friction_tolerance=(
            friction_tolerance if friction_tolerance is not None else c.COLEBROOK_TOLERANCE
        ),
        rate_time_factor=get_unit_system_constant(
            prefix="RATE_TIME_FACTOR", unit_system=unit_system
        ),
        viscosity_scale=get_unit_system_constant(
            prefix="VISCOSITY_SCALE", unit_system=unit_system
        ),
        hydrostatic_scale=1.0
        / (
            get_unit_system_constant(prefix="GRAVITATIONAL_FACTOR", unit_system=unit_system)
            * get_unit_system_constant(prefix="HYDROSTATIC_AREA_FACTOR", unit_system=unit_system)
        ),
        unit_system_tag=UNIT_SYSTEM_TAGS[unit_system],
    )
    return WellBoreModel(
        name="homogeneous",
        options=options,
        tubing_fluid_table=tubing_fluid_table,
        use_tubing_fluid_table=use_tubing_fluid_table,
    )


@numba.njit(cache=True)
def compute_segment_drop(
    model: HomogeneousWellbore,
    length: Number,
    inclination_from_vertical: Number,
    mixture_density: Number,
    mixture_viscosity: Number,
    mixture_velocity: Number,
) -> PressureDrop:
    """
    Computes the pressure drop across one tubing segment.

    :param model: This well's `HomogeneousWellbore`.
    :param length: Along-wellbore segment length.
    :param inclination_from_vertical: Segment inclination, in radians. `0` is vertical.
    :param mixture_density: No-slip mixture density for this segment.
    :param mixture_viscosity: No-slip mixture viscosity for this segment.
    :param mixture_velocity: Superficial mixture velocity in the segment.
    :returns: Pressure drop for this segment.
    """
    return compute_segment_pressure_drop(
        length=length,
        inclination_from_vertical=inclination_from_vertical,
        tubing_inner_diameter=model.tubing_inner_diameter,
        tubing_roughness=model.tubing_roughness,
        mixture_density=mixture_density,
        mixture_viscosity=mixture_viscosity,
        mixture_velocity=mixture_velocity,
        gravitational_acceleration=model.gravitational_acceleration,
        hydrostatic_scale=model.hydrostatic_scale,
        viscosity_scale=model.viscosity_scale,
        method=model.friction_method,
        laminar_reynolds_limit=model.laminar_reynolds_limit,
        turbulent_reynolds_limit=model.turbulent_reynolds_limit,
        friction_max_iterations=model.friction_max_iterations,
        friction_tolerance=model.friction_tolerance,
    )


@numba.njit(cache=True)
def walk_perforation_pressures(
    model: HomogeneousWellbore,
    reference_depth: Number,
    reference_measured_depth: Number,
    reference_pressure: Number,
    connection_oil_rates: NumberArray[OneDimension],
    connection_water_rates: NumberArray[OneDimension],
    connection_gas_rates: NumberArray[OneDimension],
    representative_depths: NumberArray[OneDimension],
    representative_measured_depths: NumberArray[OneDimension],
    inclinations_from_vertical: NumberArray[OneDimension],
    connection_fluids: ConnectionFluids,
    is_injector: Boolean,
    hydrostatic_divisor: Number,
    pressures: NumberArray[OneDimension],
) -> None:
    """
    Writes the flowing pressure at each connection into `pressures`, walking outward from
    `reference_depth`.

    The wellbore is walked first down through the connections at or below `reference_depth`, then
    up through the connections above it, nearest connection first. Each segment carries the
    combined rate of every connection beneath it, since a producer's flow rises past them and an
    injector's falls to them. Friction acts against the flow, so it adds to the pressure going
    against the flow and subtracts going with it. Hydrostatic head follows each segment's vertical
    extent, and friction its length along the wellbore. A segment with no flow holds the
    saturation-weighted reservoir fluid of the connection it ends at.

    :param model: This well's `HomogeneousWellbore`.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_measured_depth: Measured depth along the wellbore of `reference_depth`.
    :param reference_pressure: Pressure at `reference_depth`.
    :param connection_oil_rates: Each connection's own oil rate, at reservoir conditions.
    :param connection_water_rates: Each connection's own water rate, at reservoir conditions.
    :param connection_gas_rates: Each connection's own gas rate, at reservoir conditions.
    :param representative_depths: One vertical depth per connection.
    :param representative_measured_depths: One measured depth along the wellbore per connection.
    :param inclinations_from_vertical: One inclination per connection, in radians.
    :param connection_fluids: Reservoir conditions at each connection.
    :param is_injector: Whether this well is an injector.
    :param hydrostatic_divisor: The model's unit system's gravitational factor times its
        hydrostatic area factor, which turns the weight of a static column into a pressure.
    :param pressures: Output array, one entry per connection.
    :raises ValueError: If a connection with no flow beneath it has saturations that sum to zero.
    """
    walk = plan_connection_walk(
        reference_depth=reference_depth,
        reference_measured_depth=reference_measured_depth,
        representative_depths=representative_depths,
        representative_measured_depths=representative_measured_depths,
        connection_oil_rates=connection_oil_rates,
        connection_water_rates=connection_water_rates,
        connection_gas_rates=connection_gas_rates,
    )
    friction_sign = -1.0 if is_injector else 1.0
    for step in range(walk.connection_indices.shape[0]):
        idx = walk.connection_indices[step]
        previous = walk.previous_indices[step]
        previous_pressure = reference_pressure if previous < 0 else pressures[previous]
        geometric_sign = walk.geometric_signs[step]
        remaining_rates = PhaseValues(
            oil=walk.carried_oil_rates[step],
            water=walk.carried_water_rates[step],
            gas=walk.carried_gas_rates[step],
        )
        phase_densities = PhaseValues(
            oil=connection_fluids.oil_densities[idx],  # type: ignore[arg-type]
            water=connection_fluids.water_densities[idx],  # type: ignore[arg-type]
            gas=connection_fluids.gas_densities[idx],  # type: ignore[arg-type]
        )
        phase_viscosities = PhaseValues(
            oil=connection_fluids.oil_viscosities[idx],  # type: ignore[arg-type]
            water=connection_fluids.water_viscosities[idx],  # type: ignore[arg-type]
            gas=connection_fluids.gas_viscosities[idx],  # type: ignore[arg-type]
        )

        if remaining_rates.oil + remaining_rates.water + remaining_rates.gas == 0:
            mixture_density = compute_static_mixture_density(
                phase_saturations=PhaseValues(
                    oil=connection_fluids.oil_saturations[idx],  # type: ignore[arg-type]
                    water=connection_fluids.water_saturations[idx],  # type: ignore[arg-type]
                    gas=connection_fluids.gas_saturations[idx],  # type: ignore[arg-type]
                ),
                phase_densities=phase_densities,
            )
            hydrostatic = (
                mixture_density * model.gravitational_acceleration * walk.vertical_distances[step]
            ) / hydrostatic_divisor
            pressures[idx] = previous_pressure + geometric_sign * hydrostatic
        else:
            mixture_density = compute_mixture_density(
                rates=remaining_rates, phase_densities=phase_densities
            )
            mixture_viscosity = compute_mixture_viscosity(
                rates=remaining_rates, phase_viscosities=phase_viscosities
            )
            velocity = compute_mixture_velocity(
                rates=remaining_rates,
                tubing_inner_diameter=model.tubing_inner_diameter,
                rate_time_factor=model.rate_time_factor,
            )
            drop = compute_segment_drop(
                model=model,
                length=walk.segment_lengths[step],
                inclination_from_vertical=inclinations_from_vertical[idx],  # type: ignore[arg-type]
                mixture_density=mixture_density,
                mixture_viscosity=mixture_viscosity,
                mixture_velocity=velocity,
            )
            pressures[idx] = (
                previous_pressure
                + geometric_sign * (drop.hydrostatic + drop.acceleration)
                + friction_sign * walk.flow_directions[step] * geometric_sign * drop.friction
            )


def compute_perforation_pressures(
    model: HomogeneousWellbore,
    reference_depth: Number,
    reference_measured_depth: Number,
    reference_pressure: Number,
    connection_oil_rates: NumberArray[OneDimension],
    connection_water_rates: NumberArray[OneDimension],
    connection_gas_rates: NumberArray[OneDimension],
    representative_depths: NumberArray[OneDimension],
    representative_measured_depths: NumberArray[OneDimension],
    inclinations_from_vertical: NumberArray[OneDimension],
    connection_fluids: ConnectionFluids,
    is_injector: bool,
    out: NumberArray[OneDimension] | None = None,
    dtype: npt.DTypeLike = None,
) -> NumberArray[OneDimension]:
    """
    Computes the flowing pressure at each connection.

    See `walk_perforation_pressures` for how the wellbore is walked.

    :param model: This well's `HomogeneousWellbore`.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_measured_depth: Measured depth along the wellbore of `reference_depth`.
    :param reference_pressure: Pressure at `reference_depth`.
    :param connection_oil_rates: Each connection's own oil rate, at reservoir conditions.
    :param connection_water_rates: Each connection's own water rate, at reservoir conditions.
    :param connection_gas_rates: Each connection's own gas rate, at reservoir conditions.
    :param representative_depths: One vertical depth per connection.
    :param representative_measured_depths: One measured depth along the wellbore per connection.
    :param inclinations_from_vertical: One inclination per connection, in radians.
    :param connection_fluids: Reservoir conditions at each connection.
    :param is_injector: Whether this well is an injector.
    :param out: Optional preallocated output array, one entry per connection.
    :param dtype: Optional output array data type. Ignored if `out` is given. The data type of
        `representative_depths` if not given.
    :returns: Pressure at each connection.
    :raises ValueError: If the per-connection inputs don't all have the same length, or `out`
        is the wrong length.
    """
    count = len(representative_depths)
    if not (
        len(representative_measured_depths)
        == len(inclinations_from_vertical)
        == len(connection_oil_rates)
        == len(connection_water_rates)
        == len(connection_gas_rates)
        == len(connection_fluids.pressures)
        == count
    ):
        raise ValueError("All per-connection inputs must have the same length.")
    if out is not None and len(out) != count:
        raise ValueError("If given, `out` must have one entry per connection.")

    pressures: NumberArray[OneDimension]
    if out is not None:
        pressures = typing.cast("NumberArray[OneDimension]", out)
    else:
        pressures = typing.cast(
            "NumberArray[OneDimension]",
            np.empty(
                count,
                dtype=np.dtype(dtype)
                if dtype is not None
                else np.asarray(representative_depths).dtype,
            ),
        )

    walk_perforation_pressures(
        model=model,
        reference_depth=reference_depth,
        reference_measured_depth=reference_measured_depth,
        reference_pressure=reference_pressure,
        connection_oil_rates=connection_oil_rates,
        connection_water_rates=connection_water_rates,
        connection_gas_rates=connection_gas_rates,
        representative_depths=representative_depths,
        representative_measured_depths=representative_measured_depths,
        inclinations_from_vertical=inclinations_from_vertical,
        connection_fluids=connection_fluids,
        is_injector=is_injector,
        hydrostatic_divisor=get_hydrostatic_divisor(model.unit_system),
        pressures=pressures,
    )
    return pressures


@numba.njit(cache=True)
def compute_tubing_head_pressure(
    model: HomogeneousWellbore,
    reference_depth: Number,
    reference_pressure: Number,
    rates: PhaseValues,
    surface_sample: SurfaceSample,
    is_injector: Boolean,
) -> Number:
    """
    Computes tubing head pressure at surface.

    :param model: This well's `HomogeneousWellbore`.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_pressure: Pressure at `reference_depth`.
    :param rates: Rate of each phase, at reservoir conditions.
    :param surface_sample: Fluid properties at surface conditions. Needs a mixture density and
        viscosity, or all three phase densities and viscosities. A well with no flow needs the
        mixture density.
    :param is_injector: Whether this well is an injector.
    :returns: Tubing head pressure.
    :raises ValueError: If neither a mixture density nor all three phase densities, or neither a
        mixture viscosity nor all three phase viscosities, were given, or if the well has no flow
        and no mixture density was given.
    """
    dz = 0.0 - reference_depth
    total_rate = rates.oil + rates.water + rates.gas
    friction_sign = -1.0 if is_injector else 1.0

    if total_rate == 0:
        static_density = require_static_column_density(density=surface_sample.density)
        return (
            reference_pressure
            - static_density * model.gravitational_acceleration * abs(dz) * model.hydrostatic_scale
        )

    mixture_density = compute_surface_mixture_density(
        density=surface_sample.density,
        phase_densities=surface_sample.phase_densities,
        rates=rates,
    )
    mixture_viscosity = compute_surface_mixture_viscosity(
        viscosity=surface_sample.viscosity,
        phase_viscosities=surface_sample.phase_viscosities,
        rates=rates,
    )
    velocity = compute_mixture_velocity(
        rates=rates,
        tubing_inner_diameter=model.tubing_inner_diameter,
        rate_time_factor=model.rate_time_factor,
    )
    drop = compute_segment_drop(
        model=model,
        length=abs(dz),
        inclination_from_vertical=0.0,
        mixture_density=mixture_density,
        mixture_viscosity=mixture_viscosity,
        mixture_velocity=velocity,
    )
    return (
        reference_pressure - (drop.hydrostatic + drop.acceleration) - friction_sign * drop.friction
    )


@numba.njit(cache=True)
def march_segment_from_table(
    model: HomogeneousWellbore,
    pressure: Number,
    length: Number,
    inclination_from_vertical: Number,
    geometric_sign: Number,
    flow_direction: Number,
    surface_rates: PhaseValues,
    tubing_fluid_table: TubingFluidTable,
    is_injector: Boolean,
    sub_segment_count: Integer,
) -> Number:
    """
    Finds the pressure at the far end of a stretch of the wellbore, by marching along it in
    equal sub-segments with the fluids of each read from `tubing_fluid_table` at its own pressure.

    The mixture velocity, density and viscosity of a sub-segment are taken at its mean pressure,
    from the same volumes of oil, water and free gas. Acceleration is the mass flux times the
    change in mixture velocity between the two ends of the sub-segment.

    :param model: This well's `HomogeneousWellbore`.
    :param pressure: Pressure at the near end of the stretch.
    :param length: Length of the stretch along the wellbore.
    :param inclination_from_vertical: Inclination of the stretch from vertical, in radians.
    :param geometric_sign: One if the far end is deeper than the near end, negative one if it is
        shallower.
    :param surface_rates: Rate of each phase carried along the stretch, at surface
        conditions. The gas rate is the total, free and dissolved.
    :param tubing_fluid_table: The well's fluid properties against pressure.
    :param flow_direction: `1` if the stretch carries flow the way the well flows, `-1` if
        connections that flow against the well have turned it around.
    :param is_injector: Whether this well is an injector.
    :param sub_segment_count: Number of sub-segments to march in.
    :returns: Pressure at the far end.
    """
    oil_rate = surface_rates.oil
    water_rate = surface_rates.water
    gas_rate = surface_rates.gas
    friction_sign = -1.0 if is_injector else 1.0
    sub_length = length / sub_segment_count
    previous_change = 0.0
    for _ in range(sub_segment_count):
        start = compute_no_slip_mixture(
            table=tubing_fluid_table,
            pressure=pressure,
            surface_oil_rate=oil_rate,
            surface_water_rate=water_rate,
            surface_gas_rate=gas_rate,
            tubing_inner_diameter=model.tubing_inner_diameter,
            rate_time_factor=model.rate_time_factor,
        )
        end_pressure = pressure + previous_change
        for _ in range(MARCH_ITERATIONS):
            middle = compute_no_slip_mixture(
                table=tubing_fluid_table,
                pressure=0.5 * (pressure + end_pressure),
                surface_oil_rate=oil_rate,
                surface_water_rate=water_rate,
                surface_gas_rate=gas_rate,
                tubing_inner_diameter=model.tubing_inner_diameter,
                rate_time_factor=model.rate_time_factor,
            )
            end = compute_no_slip_mixture(
                table=tubing_fluid_table,
                pressure=end_pressure,
                surface_oil_rate=oil_rate,
                surface_water_rate=water_rate,
                surface_gas_rate=gas_rate,
                tubing_inner_diameter=model.tubing_inner_diameter,
                rate_time_factor=model.rate_time_factor,
            )
            drop = compute_segment_drop(
                model=model,
                length=sub_length,
                inclination_from_vertical=inclination_from_vertical,
                mixture_density=middle.density,
                mixture_viscosity=middle.viscosity,
                mixture_velocity=middle.velocity,
            )
            mass_flux = 0.5 * (start.density * start.velocity + end.density * end.velocity)
            kinetic = mass_flux * (end.velocity - start.velocity) * model.hydrostatic_scale
            next_end_pressure = (
                pressure
                + geometric_sign * (drop.hydrostatic + drop.acceleration)
                + friction_sign * flow_direction * geometric_sign * drop.friction
                - kinetic
            )
            end_pressure = next_end_pressure
        previous_change = end_pressure - pressure
        pressure = end_pressure
    return pressure


@numba.njit(cache=True)
def compute_tubing_head_pressure_from_table(
    model: HomogeneousWellbore,
    reference_depth: Number,
    reference_pressure: Number,
    surface_rates: PhaseValues,
    tubing_fluid_table: TubingFluidTable,
    is_injector: Boolean,
    static_density: Number,
) -> Number:
    """
    Computes tubing head pressure by marching up the tubing in segments, with the fluids of each
    segment read from `tubing_fluid_table` at that segment's own pressure.

    :param model: This well's `HomogeneousWellbore`.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_pressure: Pressure at `reference_depth`.
    :param surface_rates: Rate of each phase at surface conditions. The gas rate is the
        total, free and dissolved.
    :param tubing_fluid_table: The well's fluid properties against pressure.
    :param is_injector: Whether this well is an injector.
    :param static_density: Density of the column when the well has no flow. `NaN` if unknown.
    :returns: Tubing head pressure.
    :raises ValueError: If the well has no flow and `static_density` is `NaN`.
    """
    length = abs(0.0 - reference_depth)
    if surface_rates.oil + surface_rates.water + surface_rates.gas == 0:
        density = require_static_column_density(density=static_density)
        return (
            reference_pressure
            - density * model.gravitational_acceleration * length * model.hydrostatic_scale
        )
    return march_segment_from_table(
        model=model,
        pressure=reference_pressure,
        length=length,
        inclination_from_vertical=0.0,
        geometric_sign=-1.0,
        flow_direction=1.0,
        surface_rates=surface_rates,
        tubing_fluid_table=tubing_fluid_table,
        is_injector=is_injector,
        sub_segment_count=tubing_fluid_table.segment_count,
    )


@numba.njit(cache=True)
def walk_perforation_pressures_from_table(
    model: HomogeneousWellbore,
    reference_depth: Number,
    reference_measured_depth: Number,
    reference_pressure: Number,
    connection_oil_rates: NumberArray[OneDimension],
    connection_water_rates: NumberArray[OneDimension],
    connection_gas_rates: NumberArray[OneDimension],
    representative_depths: NumberArray[OneDimension],
    representative_measured_depths: NumberArray[OneDimension],
    inclinations_from_vertical: NumberArray[OneDimension],
    connection_fluids: ConnectionFluids,
    tubing_fluid_table: TubingFluidTable,
    is_injector: Boolean,
    pressures: NumberArray[OneDimension],
) -> None:
    """
    Writes the flowing pressure at each connection into `pressures`, walking outward from
    `reference_depth` as `walk_perforation_pressures` does, but with the fluids in every stretch
    of the wellbore read from `tubing_fluid_table` at the pressure there.

    The rates are turned into surface-condition rates first, so that the gas that expands and the
    gas that comes out of solution at the lower pressure inside the wellbore is counted, rather
    than the volumes at the pressure of the connected cell. A stretch with no flow holds the
    saturation-weighted reservoir fluid of the connection it ends at.

    :param model: This well's `HomogeneousWellbore`.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_measured_depth: Measured depth along the wellbore of `reference_depth`.
    :param reference_pressure: Pressure at `reference_depth`.
    :param connection_oil_rates: Each connection's own oil rate, at reservoir conditions.
    :param connection_water_rates: Each connection's own water rate, at reservoir conditions.
    :param connection_gas_rates: Each connection's own gas rate, at reservoir conditions.
    :param representative_depths: One vertical depth per connection.
    :param representative_measured_depths: One measured depth along the wellbore per connection.
    :param inclinations_from_vertical: One inclination per connection, in radians.
    :param connection_fluids: Reservoir conditions at each connection.
    :param tubing_fluid_table: The well's fluid properties against pressure.
    :param is_injector: Whether this well is an injector.
    :param pressures: Output array, one entry per connection.
    :raises ValueError: If a connection with no flow beneath it has saturations that sum to zero.
    """
    surface_oil_rates, surface_water_rates, surface_gas_rates = compute_connection_surface_rates(
        connection_oil_rates=connection_oil_rates,
        connection_water_rates=connection_water_rates,
        connection_gas_rates=connection_gas_rates,
        oil_formation_volume_factors=connection_fluids.oil_formation_volume_factors,
        water_formation_volume_factors=connection_fluids.water_formation_volume_factors,
        gas_formation_volume_factors=connection_fluids.gas_formation_volume_factors,
        solution_gas_oil_ratios=connection_fluids.solution_gas_oil_ratios,
        vaporized_oil_gas_ratios=connection_fluids.vaporized_oil_gas_ratios,
        stb_to_volume_factor=tubing_fluid_table.stb_to_volume_factor,
        is_injector=is_injector,
    )
    walk = plan_connection_walk(
        reference_depth=reference_depth,
        reference_measured_depth=reference_measured_depth,
        representative_depths=representative_depths,
        representative_measured_depths=representative_measured_depths,
        connection_oil_rates=surface_oil_rates,
        connection_water_rates=surface_water_rates,
        connection_gas_rates=surface_gas_rates,
    )
    target_length = np.sum(walk.segment_lengths) / tubing_fluid_table.segment_count
    for step in range(walk.connection_indices.shape[0]):
        idx = walk.connection_indices[step]
        previous = walk.previous_indices[step]
        previous_pressure = reference_pressure if previous < 0 else pressures[previous]
        geometric_sign = walk.geometric_signs[step]
        remaining_rates = PhaseValues(
            oil=walk.carried_oil_rates[step],
            water=walk.carried_water_rates[step],
            gas=walk.carried_gas_rates[step],
        )
        if remaining_rates.oil + remaining_rates.water + remaining_rates.gas == 0:
            mixture_density = compute_static_mixture_density(
                phase_saturations=PhaseValues(
                    oil=connection_fluids.oil_saturations[idx],  # type: ignore
                    water=connection_fluids.water_saturations[idx],  # type: ignore
                    gas=connection_fluids.gas_saturations[idx],  # type: ignore
                ),
                phase_densities=PhaseValues(
                    oil=connection_fluids.oil_densities[idx],  # type: ignore
                    water=connection_fluids.water_densities[idx],  # type: ignore
                    gas=connection_fluids.gas_densities[idx],  # type: ignore
                ),
            )
            pressures[idx] = (
                previous_pressure
                + geometric_sign
                * mixture_density
                * model.gravitational_acceleration
                * walk.vertical_distances[step]
                * model.hydrostatic_scale
            )
        else:
            sub_segment_count = 1
            if target_length > 0.0:
                sub_segment_count = max(
                    1, int(np.ceil(walk.segment_lengths[step] / target_length))
                )
            pressures[idx] = march_segment_from_table(
                model=model,
                pressure=previous_pressure,  # type: ignore[arg-type]
                length=walk.segment_lengths[step],
                inclination_from_vertical=inclinations_from_vertical[idx],  # type: ignore
                geometric_sign=geometric_sign,
                flow_direction=walk.flow_directions[step],
                surface_rates=remaining_rates,
                tubing_fluid_table=tubing_fluid_table,
                is_injector=is_injector,
                sub_segment_count=sub_segment_count,
            )
