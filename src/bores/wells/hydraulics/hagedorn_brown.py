"""Hagedorn & Brown (1965) two-phase slip wellbore hydraulics, with the Griffith bubble-flow correction."""

import math
import typing

import numba
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.constants import c
from bores.types import (
    Boolean,
    FrictionMethod,
    Number,
    NumberArray,
    OneDimension,
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
    compute_friction_factor,
    compute_static_hydrostatic_drop,
    compute_static_mixture_density,
    compute_superficial_velocity,
    get_hydrostatic_divisor,
    get_unit_system_constant,
    plan_connection_walk,
    split_liquid_gas,
)
from bores.wells.state import PhaseValues

__all__ = [
    "HagedornBrownWellbore",
    "compute_griffith_holdup",
    "compute_hagedorn_brown_holdup",
    "compute_perforation_pressures",
    "compute_segment_drop",
    "compute_tubing_head_pressure",
    "hagedorn_brown_wellbore",
    "is_griffith_bubble_flow",
]


class HagedornBrownWellbore(typing.NamedTuple):
    """Configuration for the Hagedorn & Brown two-phase slip wellbore hydraulics model."""

    tubing_inner_diameter: Number
    """Tubing inner diameter."""

    tubing_roughness: Number
    """Absolute pipe roughness. `NaN` for a smooth pipe."""

    friction_method: int
    """
    Which single-phase friction-factor correlation is used: `0` for the
    simplified correlation, `1` for Colebrook.
    """

    gravitational_acceleration: Number
    """Acceleration due to gravity, in this model's unit system."""

    griffith_slip_velocity: Number
    """
    Bubble rise velocity used by the Griffith correlation, for the
    bubble-flow regime. In this model's unit system.
    """

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

    field_length_factor: Number
    """
    Multiplies a length in this model's own unit system to get feet.
    `1.0` for a FIELD model. Used only by `is_griffith_bubble_flow`,
    whose empirical flow-regime boundary is a dimensional fit valid
    only in ft/s and ft.
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
            griffith_slip_velocity=scale(self.griffith_slip_velocity, length_factor),
            rate_time_factor=get_unit_system_constant(
                prefix="RATE_TIME_FACTOR", unit_system=target
            ),
            viscosity_scale=get_unit_system_constant(prefix="VISCOSITY_SCALE", unit_system=target),
            field_length_factor=get_conversion_factors(target, UnitSystem.FIELD, table=table)[
                "length"
            ],
            hydrostatic_scale=1.0
            / (
                get_unit_system_constant(prefix="GRAVITATIONAL_FACTOR", unit_system=target)
                * get_unit_system_constant(prefix="HYDROSTATIC_AREA_FACTOR", unit_system=target)
            ),
            unit_system_tag=UNIT_SYSTEM_TAGS[target],
        )


def hagedorn_brown_wellbore(
    *,
    tubing_inner_diameter: Number,
    tubing_roughness: Number | None = None,
    friction_method: FrictionMethod = "simplified",
    unit_system: UnitSystem = UnitSystem.FIELD,
    gravitational_acceleration: Number | None = None,
    griffith_slip_velocity: Number | None = None,
    laminar_reynolds_limit: Number | None = None,
    turbulent_reynolds_limit: Number | None = None,
    friction_max_iterations: int | None = None,
    friction_tolerance: Number | None = None,
) -> WellBoreModel:
    """
    Builds a `WellBoreModel` wrapping a fully configured `HagedornBrownWellbore`.

    Hagedorn & Brown was built from a vertical test well and doesn't
    correct its liquid holdup for pipe inclination the way Beggs & Brill
    does, so it is best suited to wells that are vertical or close to it.
    It is generally regarded as one of the more reliable correlations for
    oil wells in that range.

    :param tubing_inner_diameter: Tubing inner diameter.
    :param tubing_roughness: Absolute pipe roughness. `None` for a smooth pipe.
    :param friction_method: Which single-phase friction-factor correlation to use.
    :param unit_system: This model's unit system.
    :param gravitational_acceleration: Acceleration due to gravity. Resolved
        from `unit_system`'s standard gravity if not given.
    :param griffith_slip_velocity: Bubble rise velocity for the Griffith
        bubble-flow correction. Resolved from a standard value if not given.
    :param laminar_reynolds_limit: Reynolds number below which flow is
        treated as laminar. `c.WELLBORE_LAMINAR_REYNOLDS_LIMIT` if not given.
    :param turbulent_reynolds_limit: Reynolds number above which flow is
        treated as fully turbulent. `c.WELLBORE_TURBULENT_REYNOLDS_LIMIT`
        if not given.
    :param friction_max_iterations: Maximum Colebrook iterations.
        `c.COLEBROOK_MAX_ITERATIONS` if not given.
    :param friction_tolerance: Colebrook convergence tolerance.
        `c.COLEBROOK_TOLERANCE` if not given.
    :returns: `WellBoreModel(name="hagedorn_brown", options=<HagedornBrownWellbore>)`.
    """
    if unit_system != UnitSystem.FIELD:
        factors = get_conversion_factors(UnitSystem.FIELD, unit_system)
        length_factor = factors["length"]
    else:
        length_factor = 1.0

    if gravitational_acceleration is None:
        gravitational_acceleration = (
            c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE * length_factor
        )
    if griffith_slip_velocity is None:
        griffith_slip_velocity = c.GRIFFITH_BUBBLE_SLIP_VELOCITY_FEET_PER_SECOND * length_factor

    options = HagedornBrownWellbore(
        tubing_inner_diameter=tubing_inner_diameter,
        tubing_roughness=tubing_roughness if tubing_roughness is not None else float("nan"),
        friction_method=1 if friction_method == "colebrook" else 0,
        gravitational_acceleration=typing.cast(Number, gravitational_acceleration),
        griffith_slip_velocity=typing.cast(Number, griffith_slip_velocity),
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
        field_length_factor=get_conversion_factors(unit_system, UnitSystem.FIELD)["length"],
        hydrostatic_scale=1.0
        / (
            get_unit_system_constant(prefix="GRAVITATIONAL_FACTOR", unit_system=unit_system)
            * get_unit_system_constant(prefix="HYDROSTATIC_AREA_FACTOR", unit_system=unit_system)
        ),
        unit_system_tag=UNIT_SYSTEM_TAGS[unit_system],
    )
    return WellBoreModel(name="hagedorn_brown", options=options)


@numba.njit(cache=True)
def compute_liquid_velocity_number(
    superficial_liquid_velocity: Number, liquid_density: Number, liquid_surface_tension: Number
) -> Number:
    """
    Computes Hagedorn & Brown's dimensionless liquid velocity number, NLV.

    Field units throughout: velocity in ft/s, density in lbm/ft3, surface
    tension in dyne/cm.

    :param superficial_liquid_velocity: Liquid rate divided by cross-sectional area.
    :param liquid_density: Combined oil-and-water density.
    :param liquid_surface_tension: Gas-liquid surface tension.
    :returns: NLV.
    """
    return 1.938 * superficial_liquid_velocity * (liquid_density / liquid_surface_tension) ** 0.25


@numba.njit(cache=True)
def compute_gas_velocity_number(
    superficial_gas_velocity: Number, liquid_density: Number, liquid_surface_tension: Number
) -> Number:
    """
    Computes Hagedorn & Brown's dimensionless gas velocity number, NGV.

    Field units throughout, same as `compute_liquid_velocity_number`.

    :param superficial_gas_velocity: Gas rate divided by cross-sectional area.
    :param liquid_density: Combined oil-and-water density.
    :param liquid_surface_tension: Gas-liquid surface tension.
    :returns: NGV.
    """
    return 1.938 * superficial_gas_velocity * (liquid_density / liquid_surface_tension) ** 0.25


@numba.njit(cache=True)
def compute_pipe_diameter_number(
    tubing_inner_diameter: Number, liquid_density: Number, liquid_surface_tension: Number
) -> Number:
    """
    Computes Hagedorn & Brown's dimensionless pipe diameter number, ND.

    Field units throughout: diameter in ft, density in lbm/ft3, surface
    tension in dyne/cm.

    :param tubing_inner_diameter: Tubing inner diameter.
    :param liquid_density: Combined oil-and-water density.
    :param liquid_surface_tension: Gas-liquid surface tension.
    :returns: ND.
    """
    return 120.872 * tubing_inner_diameter * (liquid_density / liquid_surface_tension) ** 0.5


@numba.njit(cache=True)
def compute_liquid_viscosity_number(
    liquid_viscosity: Number, liquid_density: Number, liquid_surface_tension: Number
) -> Number:
    """
    Computes Hagedorn & Brown's dimensionless liquid viscosity number, NL.

    Field units throughout: viscosity in cP, density in lbm/ft3, surface
    tension in dyne/cm.

    :param liquid_viscosity: Combined oil-and-water viscosity.
    :param liquid_density: Combined oil-and-water density.
    :param liquid_surface_tension: Gas-liquid surface tension.
    :returns: NL.
    """
    return (
        0.15726 * liquid_viscosity * (1.0 / (liquid_density * liquid_surface_tension**3)) ** 0.25
    )


@numba.njit(cache=True)
def compute_hagedorn_brown_holdup(
    superficial_liquid_velocity: Number,
    superficial_gas_velocity: Number,
    liquid_density: Number,
    liquid_viscosity: Number,
    liquid_surface_tension: Number,
    tubing_inner_diameter: Number,
    pressure: Number,
) -> Number:
    """
    Computes in-situ liquid holdup per Hagedorn & Brown (1965), using the
    curve-fit form of the original correlation charts.

    Field units throughout: velocities in ft/s, density in lbm/ft3,
    viscosity in cP, surface tension in dyne/cm, diameter in ft, pressure
    in psia.

    :param superficial_liquid_velocity: Liquid rate divided by cross-sectional area.
    :param superficial_gas_velocity: Gas rate divided by cross-sectional area.
    :param liquid_density: Combined oil-and-water density.
    :param liquid_viscosity: Combined oil-and-water viscosity.
    :param liquid_surface_tension: Gas-liquid surface tension.
    :param tubing_inner_diameter: Tubing inner diameter.
    :param pressure: Local pressure, for the H group's pressure term.
    :returns: In-situ liquid holdup, always at least the no-slip holdup.
    """
    mixture_velocity = superficial_liquid_velocity + superficial_gas_velocity
    no_slip_holdup = superficial_liquid_velocity / mixture_velocity

    nlv = compute_liquid_velocity_number(
        superficial_liquid_velocity, liquid_density, liquid_surface_tension
    )
    ngv = compute_gas_velocity_number(
        superficial_gas_velocity, liquid_density, liquid_surface_tension
    )
    nd = compute_pipe_diameter_number(
        tubing_inner_diameter, liquid_density, liquid_surface_tension
    )
    nl = compute_liquid_viscosity_number(liquid_viscosity, liquid_density, liquid_surface_tension)

    viscosity_number_coefficient = 0.061 * nl**3 - 0.0929 * nl**2 + 0.0505 * nl + 0.0019
    primary_group = (
        (nlv / ngv**0.575) * (pressure / 14.7) ** 0.1 * viscosity_number_coefficient / nd
    )
    secondary_group = (ngv * nlv**0.38) / nd**2.14

    if secondary_group <= 0.025:
        psi = (
            27170.0 * secondary_group**3
            - 317.52 * secondary_group**2
            + 0.5472 * secondary_group
            + 0.9999
        )
    elif secondary_group <= 0.055:
        psi = -533.33 * secondary_group**2 + 58.524 * secondary_group + 0.1171
    else:
        psi = 2.5714 * secondary_group + 1.5962

    holdup_ratio = math.sqrt(
        (0.0047 + 1123.32 * primary_group + 729489.64 * primary_group**2)
        / (1.0 + 1097.1566 * primary_group + 722153.97 * primary_group**2)
    )
    holdup = holdup_ratio * psi
    return max(holdup, no_slip_holdup)


@numba.njit(cache=True)
def is_griffith_bubble_flow(
    superficial_liquid_velocity: Number,
    superficial_gas_velocity: Number,
    tubing_inner_diameter: Number,
    field_length_factor: Number,
) -> Boolean:
    """
    Checks whether flow falls in the bubble-flow regime Griffith's
    correlation applies to, per the boundary Hagedorn & Brown adopted for it.

    The `1.071`/`0.2218` boundary is a dimensional empirical fit from the
    original paper, only valid with velocity in ft/s and diameter in ft -
    unlike every other quantity in this module, it isn't something a
    unit-system scale *factor* can fix, since the coefficients themselves
    carry implicit units. `field_length_factor` converts the velocity and
    diameter to ft/s and ft for this one check; the boolean result needs
    no converting back.

    :param superficial_liquid_velocity: Liquid rate divided by
        cross-sectional area, in this model's own unit system.
    :param superficial_gas_velocity: Gas rate divided by cross-sectional
        area, in this model's own unit system.
    :param tubing_inner_diameter: Tubing inner diameter, in this model's
        own unit system.
    :param field_length_factor: `HagedornBrownWellbore.field_length_factor` -
        multiplies a length (or, here, a velocity - both scale the same
        way under a pure length-unit change) in this model's own unit
        system to get feet (or ft/s). `1.0` for a FIELD model.
    :returns: Whether the bubble-flow correction applies.
    """
    liquid_velocity_fps = superficial_liquid_velocity * field_length_factor
    gas_velocity_fps = superficial_gas_velocity * field_length_factor
    diameter_ft = tubing_inner_diameter * field_length_factor
    mixture_velocity_fps = liquid_velocity_fps + gas_velocity_fps
    no_slip_gas_fraction = gas_velocity_fps / mixture_velocity_fps
    bubble_flow_boundary = max(1.071 - 0.2218 * mixture_velocity_fps**2 / diameter_ft, 0.13)
    return no_slip_gas_fraction < bubble_flow_boundary


@numba.njit(cache=True)
def compute_griffith_holdup(
    superficial_liquid_velocity: Number,
    superficial_gas_velocity: Number,
    griffith_slip_velocity: Number,
) -> Number:
    """
    Computes liquid holdup in the bubble-flow regime, per Griffith's
    correlation as adopted by Hagedorn & Brown.

    :param superficial_liquid_velocity: Liquid rate divided by cross-sectional area.
    :param superficial_gas_velocity: Gas rate divided by cross-sectional area.
    :param griffith_slip_velocity: Bubble rise velocity, `HagedornBrownWellbore.griffith_slip_velocity`.
    :returns: In-situ liquid holdup for bubble flow.
    """
    mixture_velocity = superficial_liquid_velocity + superficial_gas_velocity
    velocity_ratio = mixture_velocity / griffith_slip_velocity
    return 1.0 - 0.5 * (
        1.0
        + velocity_ratio
        - math.sqrt(
            (1.0 + velocity_ratio) ** 2 - 4.0 * superficial_gas_velocity / griffith_slip_velocity
        )
    )


@numba.njit(cache=True)
def compute_segment_drop(
    model: HagedornBrownWellbore,
    length: Number,
    inclination_from_vertical: Number,
    superficial_liquid_velocity: Number,
    superficial_gas_velocity: Number,
    liquid_density: Number,
    gas_density: Number,
    liquid_viscosity: Number,
    gas_viscosity: Number,
    liquid_surface_tension: Number,
    pressure: Number,
    is_injector: Boolean,
) -> PressureDrop:
    """
    Computes the pressure drop across one tubing segment.

    Follows the original Hagedorn & Brown (1965) pressure gradient
    equation exactly: the same in-situ (slip) mixture density is used for
    the gravity, friction, and acceleration terms, and the Reynolds number
    uses that same density with a holdup-weighted viscosity. Some later
    software revisions use a different density for the friction term;
    this follows the original paper's own derivation, not those revisions.

    In the bubble-flow regime (per `is_griffith_bubble_flow`), holdup
    switches to the Griffith correlation, and the friction and Reynolds
    number terms switch to using the liquid phase alone rather than the mixture.

    :param model: This well's `HagedornBrownWellbore`.
    :param length: Along-wellbore segment length.
    :param inclination_from_vertical: Segment inclination, in radians. `0`
        is vertical. Hagedorn & Brown's own holdup correlation has no
        inclination correction, so this only affects the vertical
        projection of the segment length, the same as a purely vertical
        correlation would.
    :param superficial_liquid_velocity: Liquid rate divided by cross-sectional area.
    :param superficial_gas_velocity: Gas rate divided by cross-sectional area.
    :param liquid_density: Combined oil-and-water density.
    :param gas_density: Gas density.
    :param liquid_viscosity: Combined oil-and-water viscosity.
    :param gas_viscosity: Gas viscosity.
    :param liquid_surface_tension: Gas-liquid surface tension.
    :param pressure: Local pressure, needed by the holdup correlation.
    :param is_injector: Whether this well is an injector.
    :returns: Pressure drop for this segment.
    :raises OverflowError: If the holdup-weighted viscosity overflows.
    """
    mixture_velocity = superficial_liquid_velocity + superficial_gas_velocity

    bubble_flow = is_griffith_bubble_flow(
        superficial_liquid_velocity=superficial_liquid_velocity,
        superficial_gas_velocity=superficial_gas_velocity,
        tubing_inner_diameter=model.tubing_inner_diameter,
        field_length_factor=model.field_length_factor,
    )
    if bubble_flow:
        in_situ_holdup = compute_griffith_holdup(
            superficial_liquid_velocity=superficial_liquid_velocity,
            superficial_gas_velocity=superficial_gas_velocity,
            griffith_slip_velocity=model.griffith_slip_velocity,
        )
    else:
        in_situ_holdup = compute_hagedorn_brown_holdup(
            superficial_liquid_velocity=superficial_liquid_velocity,
            superficial_gas_velocity=superficial_gas_velocity,
            liquid_density=liquid_density,
            liquid_viscosity=liquid_viscosity,
            liquid_surface_tension=liquid_surface_tension,
            tubing_inner_diameter=model.tubing_inner_diameter,
            pressure=pressure,
        )

    in_situ_density = liquid_density * in_situ_holdup + gas_density * (1.0 - in_situ_holdup)
    vertical_length = length * math.cos(inclination_from_vertical)
    hydrostatic_drop = (
        in_situ_density
        * model.gravitational_acceleration
        * vertical_length
        * model.hydrostatic_scale
    )

    relative_roughness = (
        0.0
        if math.isnan(model.tubing_roughness)
        else model.tubing_roughness / model.tubing_inner_diameter
    )

    if bubble_flow:
        # Griffith's own treatment: friction and the Reynolds number use
        # the liquid phase alone, not the mixture.
        friction_density = liquid_density
        friction_velocity = superficial_liquid_velocity
        reynolds_number = (
            liquid_density
            * superficial_liquid_velocity
            * model.tubing_inner_diameter
            / (liquid_viscosity * model.viscosity_scale)
        )
    else:
        # The original paper's own friction term: the same in-situ
        # mixture density used above for gravity, not a separate
        # no-slip value.
        friction_density = in_situ_density
        friction_velocity = mixture_velocity
        holdup_weighted_viscosity = liquid_viscosity**in_situ_holdup * gas_viscosity ** (
            1.0 - in_situ_holdup
        )
        if math.isinf(holdup_weighted_viscosity):
            raise OverflowError(
                "The holdup-weighted viscosity overflowed; check the liquid and gas viscosities "
                "and the holdup."
            )
        reynolds_number = (
            in_situ_density
            * mixture_velocity
            * model.tubing_inner_diameter
            / (holdup_weighted_viscosity * model.viscosity_scale)
        )

    if reynolds_number <= 0.0:
        friction_drop = 0.0
    else:
        friction_factor = compute_friction_factor(
            reynolds_number=reynolds_number,
            relative_roughness=relative_roughness,
            method=model.friction_method,
            laminar_reynolds_limit=model.laminar_reynolds_limit,
            turbulent_reynolds_limit=model.turbulent_reynolds_limit,
            friction_max_iterations=model.friction_max_iterations,
            friction_tolerance=model.friction_tolerance,
        )
        friction_drop = (
            friction_factor
            * (length / model.tubing_inner_diameter)
            * (friction_density * friction_velocity**2 / 2.0)
            * model.hydrostatic_scale
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
def walk_perforation_pressures(
    model: HagedornBrownWellbore,
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

    :param model: This well's `HagedornBrownWellbore`.
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
            (
                liquid_rate,
                gas_rate,
                liquid_density,
                gas_density,
                liquid_viscosity,
                gas_viscosity,
            ) = split_liquid_gas(
                phase_rates=remaining_rates,
                phase_densities=phase_densities,
                phase_viscosities=phase_viscosities,
            )
            drop = compute_segment_drop(
                model=model,
                length=walk.segment_lengths[step],
                inclination_from_vertical=inclinations_from_vertical[idx],  # type: ignore[arg-type]
                superficial_liquid_velocity=compute_superficial_velocity(
                    rate=liquid_rate,
                    tubing_inner_diameter=model.tubing_inner_diameter,
                    rate_time_factor=model.rate_time_factor,
                ),
                superficial_gas_velocity=compute_superficial_velocity(
                    rate=gas_rate,
                    tubing_inner_diameter=model.tubing_inner_diameter,
                    rate_time_factor=model.rate_time_factor,
                ),
                liquid_density=liquid_density,
                gas_density=gas_density,
                liquid_viscosity=liquid_viscosity,
                gas_viscosity=gas_viscosity,
                liquid_surface_tension=connection_fluids.gas_liquid_surface_tensions[idx],  # type: ignore[arg-type]
                pressure=connection_fluids.pressures[idx],  # type: ignore[arg-type]
                is_injector=is_injector,
            )
            pressures[idx] = (
                previous_pressure
                + geometric_sign * (drop.hydrostatic + drop.acceleration)
                + friction_sign * geometric_sign * drop.friction
            )


def compute_perforation_pressures(
    model: HagedornBrownWellbore,
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

    :param model: This well's `HagedornBrownWellbore`.
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
        pressures = typing.cast(NumberArray[OneDimension], out)
    else:
        pressures = typing.cast(
            NumberArray[OneDimension],
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


def compute_tubing_head_pressure(
    model: HagedornBrownWellbore,
    reference_depth: Number,
    reference_pressure: Number,
    phase_rates: PhaseValues,
    surface_sample: SurfaceSample,
    is_injector: bool,
) -> Number:
    """
    Computes tubing head pressure at surface.

    :param model: This well's `HagedornBrownWellbore`.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_pressure: Pressure at `reference_depth`.
    :param phase_rates: Rate of each phase, at reservoir conditions.
    :param surface_sample: Fluid properties at surface conditions.
        `phase_densities`, `phase_viscosities`, and `gas_liquid_surface_tension`
        are all required, as a two-phase correlation needs a real liquid/gas
        split, not just a single mixture value.
    :param is_injector: Whether this well is an injector.
    :returns: Tubing head pressure.
    :raises ValueError: If `phase_densities`, `phase_viscosities`, or
        `gas_liquid_surface_tension` isn't set on `surface_sample`.
    """
    if surface_sample.phase_densities is None:
        raise ValueError(
            "`SurfaceSample.phase_densities` is required for the Hagedorn & Brown wellbore model."
        )
    if surface_sample.phase_viscosities is None:
        raise ValueError(
            "`SurfaceSample.phase_viscosities` is required for the Hagedorn & Brown "
            "wellbore model."
        )
    if surface_sample.gas_liquid_surface_tension is None:
        raise ValueError(
            "`SurfaceSample.gas_liquid_surface_tension` is required for the "
            "Hagedorn & Brown wellbore model."
        )

    dz = 0.0 - reference_depth
    total_rate = phase_rates.oil + phase_rates.water + phase_rates.gas
    friction_sign = -1.0 if is_injector else 1.0

    if total_rate == 0:
        drop = compute_static_hydrostatic_drop(
            mixture_density=compute_static_mixture_density(
                phase_saturations=PhaseValues(
                    oil=phase_rates.oil, water=phase_rates.water, gas=phase_rates.gas
                ),
                phase_densities=surface_sample.phase_densities,
            ),
            length=abs(dz),
            gravitational_acceleration=model.gravitational_acceleration,
            unit_system=model.unit_system,
        )
        return reference_pressure - drop.total

    (
        liquid_rate,
        gas_rate,
        liquid_density,
        gas_density,
        liquid_viscosity,
        gas_viscosity,
    ) = split_liquid_gas(
        phase_rates=phase_rates,
        phase_densities=surface_sample.phase_densities,
        phase_viscosities=surface_sample.phase_viscosities,
    )
    drop = compute_segment_drop(
        model=model,
        length=abs(dz),
        inclination_from_vertical=0.0,
        superficial_liquid_velocity=compute_superficial_velocity(
            rate=liquid_rate,
            tubing_inner_diameter=model.tubing_inner_diameter,
            rate_time_factor=model.rate_time_factor,
        ),
        superficial_gas_velocity=compute_superficial_velocity(
            rate=gas_rate,
            tubing_inner_diameter=model.tubing_inner_diameter,
            rate_time_factor=model.rate_time_factor,
        ),
        liquid_density=liquid_density,
        gas_density=gas_density,
        liquid_viscosity=liquid_viscosity,
        gas_viscosity=gas_viscosity,
        liquid_surface_tension=surface_sample.gas_liquid_surface_tension,
        pressure=reference_pressure,
        is_injector=is_injector,
    )
    return (
        reference_pressure - (drop.hydrostatic + drop.acceleration) - friction_sign * drop.friction
    )
