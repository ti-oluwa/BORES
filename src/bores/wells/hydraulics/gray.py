"""Gray (1974) wellbore hydraulics, API 14B form."""

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
    "GrayWellbore",
    "compute_gray_effective_roughness",
    "compute_gray_holdup",
    "compute_perforation_pressures",
    "compute_segment_drop",
    "compute_tubing_head_pressure",
    "gray_wellbore",
]


class GrayWellbore(typing.NamedTuple):
    """Configuration for the Gray (1974) wellbore hydraulics model."""

    tubing_inner_diameter: Number
    """Tubing inner diameter."""

    tubing_roughness: Number
    """Absolute dry-pipe roughness. `NaN` for a smooth pipe."""

    friction_method: int
    """
    Which single-phase friction-factor correlation to apply, using
    Gray's own effective roughness in place of `tubing_roughness`: `0`
    for the simplified correlation, `1` for Colebrook.
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
    conversion; without it, friction comes out several thousand times
    too large relative to hydrostatic in field units.
    """

    field_length_factor: Number
    """
    Multiplies a length in this model's own unit system to get feet.
    `1.0` for a FIELD model. Used only by `compute_gray_holdup`/
    `compute_gray_effective_roughness`, whose empirical correlations are
    dimensional fits valid only in ft/s, lbm/ft3, and ft.
    """

    field_density_factor: Number
    """
    Multiplies a density in this model's own unit system to get lbm/ft3.
    `1.0` for a FIELD model. Same reason as `field_length_factor`.
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
        field_factors = get_conversion_factors(target, UnitSystem.FIELD, table=table)
        return self._replace(
            tubing_inner_diameter=self.tubing_inner_diameter * length_factor,
            tubing_roughness=self.tubing_roughness * length_factor,
            gravitational_acceleration=self.gravitational_acceleration * length_factor,
            rate_time_factor=get_unit_system_constant(
                prefix="RATE_TIME_FACTOR", unit_system=target
            ),
            viscosity_scale=get_unit_system_constant(prefix="VISCOSITY_SCALE", unit_system=target),
            hydrostatic_scale=1.0
            / (
                get_unit_system_constant(prefix="GRAVITATIONAL_FACTOR", unit_system=target)
                * get_unit_system_constant(prefix="HYDROSTATIC_AREA_FACTOR", unit_system=target)
            ),
            field_length_factor=field_factors["length"],
            field_density_factor=field_factors["density"],
            unit_system_tag=UNIT_SYSTEM_TAGS[target],
        )


def gray_wellbore(
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
) -> WellBoreModel:
    """
    Builds a `WellBoreModel` wrapping a fully configured `GrayWellbore`.

    Gray (1974) is an empirical correlation for vertical gas and gas
    condensate wells carrying a light liquid load, developed as part of
    API 14B and widely used for mist-flow gas wells. Its holdup and
    effective-roughness correlations are dimensional fits, calibrated
    for velocities in ft/s, densities in lbm/ft3, surface tension in
    dyne/cm, and diameter in ft, the same limitation Hagedorn & Brown's
    Griffith correction has elsewhere in this package - for any other
    `unit_system`, the relevant inputs are converted to those units for
    just those two correlations, and the (already unit-invariant)
    results used directly; nothing else in this model needs converting
    back.

    Unlike Beggs & Brill or Hagedorn & Brown, Gray does not carry
    gas/liquid slip into the friction term. Friction uses the no-slip
    mixture density and viscosity, together with an effective pipe
    roughness that Gray's own method derives from how much liquid is
    being carried, standing in for the wall-wetting effect of a liquid
    film in a high-velocity gas stream. Gray's own correlations have no
    inclination correction either; they were developed for vertical wells.

    :param tubing_inner_diameter: Tubing inner diameter.
    :param tubing_roughness: Absolute dry-pipe roughness. `None` for a smooth pipe.
    :param friction_method: Which single-phase friction-factor correlation
        to apply, using Gray's own effective roughness.
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
    :returns: `WellBoreModel(name="gray", options=<GrayWellbore>)`.
    """
    if gravitational_acceleration is None:
        gravitational_acceleration = typing.cast(
            Number, c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE
        )
        if unit_system != UnitSystem.FIELD:
            factors = get_conversion_factors(UnitSystem.FIELD, unit_system)
            gravitational_acceleration = gravitational_acceleration * factors["length"]

    options = GrayWellbore(
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
        field_length_factor=get_conversion_factors(unit_system, UnitSystem.FIELD)["length"],
        field_density_factor=get_conversion_factors(unit_system, UnitSystem.FIELD)["density"],
        unit_system_tag=UNIT_SYSTEM_TAGS[unit_system],
    )
    return WellBoreModel(name="gray", options=options)


@numba.njit(cache=True)
def compute_gray_holdup(
    superficial_liquid_velocity: Number,
    superficial_gas_velocity: Number,
    liquid_density: Number,
    gas_density: Number,
    gas_liquid_surface_tension: Number,
    tubing_inner_diameter: Number,
) -> Number:
    """
    Computes in-situ liquid holdup per Gray (1974), API 14B form.

    Field units throughout: velocities in ft/s, densities in lbm/ft3,
    surface tension in dyne/cm, diameter in ft. `453.592` converts
    surface tension from dyne/cm to lbm/s2 so it lines up with the other
    terms; `32.174` is standard gravity in ft/s2. Both are fixed parts of
    the published correlation, not this model's own configured
    `gravitational_acceleration`.

    :param superficial_liquid_velocity: Liquid rate divided by cross-sectional area.
    :param superficial_gas_velocity: Gas rate divided by cross-sectional area.
    :param liquid_density: Combined oil-and-water density.
    :param gas_density: Gas density.
    :param gas_liquid_surface_tension: Gas-liquid surface tension.
    :param tubing_inner_diameter: Tubing inner diameter.
    :returns: In-situ liquid holdup, always at least the no-slip holdup.
    """
    mixture_velocity = superficial_liquid_velocity + superficial_gas_velocity
    no_slip_holdup = superficial_liquid_velocity / mixture_velocity
    if no_slip_holdup <= 0.0:
        return 0.0
    if no_slip_holdup >= 1.0 or superficial_gas_velocity <= 0.0:
        return 1.0
    if gas_liquid_surface_tension <= 0.0:
        return no_slip_holdup

    velocity_ratio = superficial_liquid_velocity / superficial_gas_velocity
    density_difference = max(liquid_density - gas_density, 0.1)
    no_slip_density = liquid_density * no_slip_holdup + gas_density * (1.0 - no_slip_holdup)

    velocity_number = (
        453.592
        * no_slip_density**2
        * mixture_velocity**4
        / (32.174 * gas_liquid_surface_tension * density_difference)
    )
    diameter_number = (
        453.592
        * 32.174
        * density_difference
        * tubing_inner_diameter**2
        / gas_liquid_surface_tension
    )
    exponent = 0.0814 * (
        1.0 - 0.0554 * math.log(1.0 + 730.0 * velocity_ratio / (velocity_ratio + 1.0))
    )
    holdup_exponent = -2.314 * (velocity_number * (1.0 + 205.0 / diameter_number)) ** exponent
    gas_holdup = (1.0 - math.exp(holdup_exponent)) / (velocity_ratio + 1.0)
    return min(max(1.0 - gas_holdup, no_slip_holdup), 1.0)


@numba.njit(cache=True)
def compute_gray_effective_roughness(
    tubing_roughness: Number,
    gas_liquid_surface_tension: Number,
    no_slip_density: Number,
    superficial_liquid_velocity: Number,
    superficial_gas_velocity: Number,
) -> Number:
    """
    Computes Gray's effective (wet-film) pipe roughness, API 14B form.

    Gray's method leaves gas/liquid slip out of the friction term
    entirely. Instead, it derives an effective roughness from how much
    liquid is being carried, standing in for the wall-wetting effect of
    a liquid film in a high-velocity gas stream: below a liquid loading
    of `velocity_ratio = 0.007`, it interpolates between the dry-pipe
    roughness and the fully wet-film value; at or above it, the wet-film
    value applies outright. Field units throughout, matching
    `compute_gray_holdup`.

    :param tubing_roughness: Absolute dry-pipe roughness. `NaN` is
        treated as a smooth pipe (`0.0`).
    :param gas_liquid_surface_tension: Gas-liquid surface tension.
    :param no_slip_density: No-slip mixture density.
    :param superficial_liquid_velocity: Liquid rate divided by cross-sectional area.
    :param superficial_gas_velocity: Gas rate divided by cross-sectional area.
    :returns: Effective roughness, at least `2.77e-5` ft (API 14B's own floor).
    """
    dry_roughness = 0.0 if math.isnan(tubing_roughness) else tubing_roughness
    mixture_velocity = superficial_liquid_velocity + superficial_gas_velocity
    if (
        mixture_velocity <= 0.0
        or no_slip_density <= 0.0
        or gas_liquid_surface_tension <= 0.0
        or superficial_gas_velocity <= 0.0
    ):
        return max(dry_roughness, 2.77e-5)

    velocity_ratio = superficial_liquid_velocity / superficial_gas_velocity
    wet_film_roughness = (
        28.5 * gas_liquid_surface_tension / (453.592 * no_slip_density * mixture_velocity**2)
    )
    if velocity_ratio >= 0.007:
        effective_roughness = wet_film_roughness
    else:
        effective_roughness = (
            dry_roughness + velocity_ratio * (wet_film_roughness - dry_roughness) / 0.007
        )
    return max(effective_roughness, 2.77e-5)


@numba.njit(cache=True)
def compute_segment_drop(
    model: GrayWellbore,
    length: Number,
    inclination_from_vertical: Number,
    superficial_liquid_velocity: Number,
    superficial_gas_velocity: Number,
    liquid_density: Number,
    gas_density: Number,
    liquid_viscosity: Number,
    gas_viscosity: Number,
    gas_liquid_surface_tension: Number,
) -> PressureDrop:
    """
    Computes the pressure drop across one tubing segment.

    Gravity uses the in-situ (slip) mixture density, from
    `compute_gray_holdup`. Friction uses the no-slip mixture density and
    viscosity, together with `compute_gray_effective_roughness` in place
    of the tubing's own dry-pipe roughness.

    :param model: This well's `GrayWellbore`.
    :param length: Along-wellbore segment length.
    :param inclination_from_vertical: Segment inclination, in radians.
        `0` is vertical. Gray's own correlations have no inclination
        term, so this only affects the vertical projection of the
        segment length, the same as a purely vertical correlation would.
    :param superficial_liquid_velocity: Liquid rate divided by cross-sectional area.
    :param superficial_gas_velocity: Gas rate divided by cross-sectional area.
    :param liquid_density: Combined oil-and-water density.
    :param gas_density: Gas density.
    :param liquid_viscosity: Combined oil-and-water viscosity.
    :param gas_viscosity: Gas viscosity.
    :param gas_liquid_surface_tension: Gas-liquid surface tension.
    :returns: Pressure drop for this segment.
    """
    mixture_velocity = superficial_liquid_velocity + superficial_gas_velocity
    no_slip_holdup = superficial_liquid_velocity / mixture_velocity
    no_slip_density = liquid_density * no_slip_holdup + gas_density * (1.0 - no_slip_holdup)
    no_slip_viscosity = liquid_viscosity * no_slip_holdup + gas_viscosity * (1.0 - no_slip_holdup)

    # compute_gray_holdup's own correlation is a dimensional fit (ft/s,
    # lbm/ft3, dyne/cm, ft); convert just its inputs to those units for
    # this one call. gas_liquid_surface_tension is always dyne/cm
    # already, regardless of unit_system. The result (a 0-1 holdup
    # fraction) needs no converting back.
    in_situ_holdup = compute_gray_holdup(
        superficial_liquid_velocity=superficial_liquid_velocity * model.field_length_factor,
        superficial_gas_velocity=superficial_gas_velocity * model.field_length_factor,
        liquid_density=liquid_density * model.field_density_factor,
        gas_density=gas_density * model.field_density_factor,
        gas_liquid_surface_tension=gas_liquid_surface_tension,
        tubing_inner_diameter=model.tubing_inner_diameter * model.field_length_factor,
    )
    in_situ_density = liquid_density * in_situ_holdup + gas_density * (1.0 - in_situ_holdup)

    vertical_length = length * math.cos(inclination_from_vertical)
    hydrostatic_drop = (
        in_situ_density
        * model.gravitational_acceleration
        * vertical_length
        * model.hydrostatic_scale
    )

    # Same dimensional-fit situation as compute_gray_holdup above.
    # effective_roughness comes back in feet; rather than converting it
    # back to this model's own length unit, divide it by the diameter
    # in the same feet (also already needed above), since only the
    # (unit-invariant) ratio is used from here on.
    effective_roughness_ft = compute_gray_effective_roughness(
        tubing_roughness=model.tubing_roughness * model.field_length_factor,
        gas_liquid_surface_tension=gas_liquid_surface_tension,
        no_slip_density=no_slip_density * model.field_density_factor,
        superficial_liquid_velocity=superficial_liquid_velocity * model.field_length_factor,
        superficial_gas_velocity=superficial_gas_velocity * model.field_length_factor,
    )
    relative_roughness = effective_roughness_ft / (
        model.tubing_inner_diameter * model.field_length_factor
    )
    no_slip_reynolds_number = (
        no_slip_density
        * mixture_velocity
        * model.tubing_inner_diameter
        / (no_slip_viscosity * model.viscosity_scale)
    )
    if no_slip_reynolds_number <= 0.0:
        friction_drop = 0.0
    else:
        friction_factor = compute_friction_factor(
            reynolds_number=no_slip_reynolds_number,
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
            * (no_slip_density * mixture_velocity**2 / 2.0)
            * model.hydrostatic_scale
        )

    # Velocity is only ever set at a connection, where a perforation's own
    # rate joins or leaves the flow (see compute_perforation_pressures).
    # Within one segment there is no other source of velocity change, so
    # this term is always zero today, matching every other correlation
    # in this package.
    acceleration_drop = in_situ_density * (mixture_velocity**2 - mixture_velocity**2) / 2.0
    return PressureDrop(
        hydrostatic=hydrostatic_drop,
        friction=friction_drop,
        acceleration=acceleration_drop,
    )


@numba.njit(cache=True)
def walk_perforation_pressures(
    model: GrayWellbore,
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

    :param model: This well's `GrayWellbore`.
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
            oil=connection_fluids.oil_densities[idx],  # type: ignore
            water=connection_fluids.water_densities[idx],  # type: ignore
            gas=connection_fluids.gas_densities[idx],  # type: ignore
        )
        phase_viscosities = PhaseValues(
            oil=connection_fluids.oil_viscosities[idx],  # type: ignore
            water=connection_fluids.water_viscosities[idx],  # type: ignore
            gas=connection_fluids.gas_viscosities[idx],  # type: ignore
        )

        if remaining_rates.oil + remaining_rates.water + remaining_rates.gas == 0:
            mixture_density = compute_static_mixture_density(
                phase_saturations=PhaseValues(
                    oil=connection_fluids.oil_saturations[idx],  # type: ignore
                    water=connection_fluids.water_saturations[idx],  # type: ignore
                    gas=connection_fluids.gas_saturations[idx],  # type: ignore
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
                inclination_from_vertical=inclinations_from_vertical[idx],  # type: ignore
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
                gas_liquid_surface_tension=connection_fluids.gas_liquid_surface_tensions[idx],  # type: ignore
            )
            pressures[idx] = (
                previous_pressure
                + geometric_sign * (drop.hydrostatic + drop.acceleration)
                + friction_sign * geometric_sign * drop.friction
            )


def compute_perforation_pressures(
    model: GrayWellbore,
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

    :param model: This well's `GrayWellbore`.
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
    model: GrayWellbore,
    reference_depth: Number,
    reference_pressure: Number,
    phase_rates: PhaseValues,
    surface_sample: SurfaceSample,
    is_injector: bool,
) -> Number:
    """
    Computes tubing head pressure at surface.

    :param model: This well's `GrayWellbore`.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_pressure: Pressure at `reference_depth`.
    :param phase_rates: Rate of each phase, at reservoir conditions.
    :param surface_sample: Fluid properties at surface
        conditions. `phase_densities`, `phase_viscosities`, and
        `gas_liquid_surface_tension` are all required, as Gray needs a
        real liquid/gas split, not just a single mixture value.
    :param is_injector: Whether this well is an injector.
    :returns: Tubing head pressure.
    :raises ValueError: If `phase_densities`, `phase_viscosities`, or
        `gas_liquid_surface_tension` isn't set on `surface_sample`.
    """
    if surface_sample.phase_densities is None:
        raise ValueError(
            "`SurfaceSample.phase_densities` is required for the Gray wellbore model."
        )
    if surface_sample.phase_viscosities is None:
        raise ValueError(
            "`SurfaceSample.phase_viscosities` is required for the Gray wellbore model."
        )
    if surface_sample.gas_liquid_surface_tension is None:
        raise ValueError(
            "`SurfaceSample.gas_liquid_surface_tension` is required for the Gray wellbore model."
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
        gas_liquid_surface_tension=surface_sample.gas_liquid_surface_tension,
    )
    return (
        reference_pressure - (drop.hydrostatic + drop.acceleration) - friction_sign * drop.friction
    )
