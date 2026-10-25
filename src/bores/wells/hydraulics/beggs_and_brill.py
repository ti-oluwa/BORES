"""Beggs & Brill (1973) two-phase slip wellbore hydraulics."""

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
    CompiledSurfaceFluids,
    ConnectionFluids,
    PressureDrop,
    SurfaceFluids,
    WellBoreModel,
    check_two_phase_surface_fluids,
    compile_surface_fluids,
    compute_static_mixture_density,
    compute_superficial_velocity,
    get_hydrostatic_divisor,
    get_two_phase_surface_properties,
    get_unit_system_constant,
    plan_connection_walk,
    split_liquid_gas,
)
from bores.wells.state import PhaseValues

__all__ = [
    "BeggsAndBrillWellbore",
    "beggs_and_brill_wellbore",
    "compute_beggs_brill_holdup",
    "compute_horizontal_holdup",
    "compute_perforation_pressures",
    "compute_segment_drop",
    "compute_tubing_head_pressure",
    "compute_two_phase_friction_factor",
    "evaluate_tubing_head_pressure",
    "get_flow_pattern",
]


class BeggsAndBrillWellbore(typing.NamedTuple):
    """Configuration for the Beggs & Brill two-phase slip wellbore hydraulics model."""

    tubing_inner_diameter: Number
    """Tubing inner diameter."""

    tubing_roughness: Number
    """Absolute pipe roughness. `NaN` for a smooth pipe."""

    friction_method: int
    """
    Which single-phase friction-factor correlation the two-phase
    correction is applied to: `0` for the simplified correlation, `1` for
    Colebrook.
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


def beggs_and_brill_wellbore(
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
    Builds a `WellBoreModel` wrapping a fully configured `BeggsAndBrillWellbore`.

    :param tubing_inner_diameter: Tubing inner diameter.
    :param tubing_roughness: Absolute pipe roughness. `None` for a smooth pipe.
    :param friction_method: Which single-phase friction-factor correlation
        the two-phase correction is applied to.
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
    :returns: `WellBoreModel(name="beggs_and_brill", options=<BeggsAndBrillWellbore>)`.
    """
    if gravitational_acceleration is None:
        gravitational_acceleration = typing.cast(
            Number, c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE
        )
        if unit_system != UnitSystem.FIELD:
            factors = get_conversion_factors(UnitSystem.FIELD, unit_system)
            gravitational_acceleration = gravitational_acceleration * factors["length"]

    options = BeggsAndBrillWellbore(
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
    return WellBoreModel(name="beggs_and_brill", options=options)


@numba.njit(cache=True)
def get_flow_pattern(no_slip_holdup: Number, froude_number: Number) -> int:
    """
    Classifies two-phase flow pattern per Beggs & Brill (1973).

    :param no_slip_holdup: No-slip liquid holdup.
    :param froude_number: Mixture Froude number.
    :returns: `0` for segregated, `1` for transition, `2` for intermittent, `3` for distributed.
    """
    l1 = 316.0 * no_slip_holdup**0.302
    l2 = 0.0009252 * no_slip_holdup**-2.4684
    l3 = 0.10 * no_slip_holdup**-1.4516
    l4 = 0.5 * no_slip_holdup**-6.738

    if (no_slip_holdup < 0.01 and froude_number < l1) or (
        no_slip_holdup >= 0.01 and froude_number < l2
    ):
        return 0  # segregated
    if no_slip_holdup >= 0.01 and l2 <= froude_number <= l3:
        return 1  # transition
    if (0.01 <= no_slip_holdup < 0.4 and l3 < froude_number <= l1) or (
        no_slip_holdup >= 0.4 and l3 < froude_number <= l4
    ):
        return 2  # intermittent
    return 3  # distributed


@numba.njit(cache=True)
def compute_horizontal_holdup(
    pattern: int, no_slip_holdup: Number, froude_number: Number
) -> Number:
    """
    Computes horizontal (zero-inclination) liquid holdup for a given flow pattern.

    A `pattern` of `1` (transition) interpolates between the
    segregated and intermittent correlations.

    :param pattern: Flow pattern, from `get_flow_pattern`.
    :param no_slip_holdup: No-slip liquid holdup.
    :param froude_number: Mixture Froude number.
    :returns: Horizontal liquid holdup.
    """
    if pattern == 1:
        l2 = 0.0009252 * no_slip_holdup**-2.4684
        l3 = 0.10 * no_slip_holdup**-1.4516
        hl_segregated = 0.98 * no_slip_holdup**0.4846 / froude_number**0.0868
        hl_intermittent = 0.845 * no_slip_holdup**0.5351 / froude_number**0.0173
        interpolation_weight = (l3 - froude_number) / (l3 - l2)
        holdup = (
            interpolation_weight * hl_segregated + (1.0 - interpolation_weight) * hl_intermittent
        )
    elif pattern == 0:
        holdup = 0.98 * no_slip_holdup**0.4846 / froude_number**0.0868
    elif pattern == 2:
        holdup = 0.845 * no_slip_holdup**0.5351 / froude_number**0.0173
    else:
        holdup = 1.065 * no_slip_holdup**0.5824 / froude_number**0.0609

    return max(holdup, no_slip_holdup)


@numba.njit(cache=True)
def compute_beggs_brill_holdup(
    superficial_liquid_velocity: Number,
    superficial_gas_velocity: Number,
    liquid_density: Number,
    liquid_surface_tension: Number,
    tubing_inner_diameter: Number,
    inclination_from_vertical: Number,
    is_injector: Boolean,
    gravitational_acceleration: Number,
) -> Number:
    """
    Computes in-situ liquid holdup per Beggs & Brill (1973), including the
    inclination correction. A well in the transition flow pattern uses the
    intermittent pattern's inclination-correction coefficients.

    :param superficial_liquid_velocity: Liquid rate divided by cross-sectional area.
    :param superficial_gas_velocity: Gas rate divided by cross-sectional area.
    :param liquid_density: Combined oil-and-water density.
    :param liquid_surface_tension: Gas-liquid surface tension.
    :param tubing_inner_diameter: Tubing inner diameter.
    :param inclination_from_vertical: Segment inclination, in radians. `0` is vertical.
    :param is_injector: Whether this well is an injector.
    :param gravitational_acceleration: Acceleration due to gravity, already unit-resolved.
    :returns: In-situ liquid holdup.
    """
    mixture_velocity = superficial_liquid_velocity + superficial_gas_velocity
    no_slip_holdup = superficial_liquid_velocity / mixture_velocity
    froude_number = mixture_velocity**2 / (gravitational_acceleration * tubing_inner_diameter)
    liquid_velocity_number = (
        1.938 * superficial_liquid_velocity * (liquid_density / liquid_surface_tension) ** 0.25
    )

    pattern = get_flow_pattern(no_slip_holdup=no_slip_holdup, froude_number=froude_number)
    holdup_at_horizontal = compute_horizontal_holdup(
        pattern=pattern,
        no_slip_holdup=no_slip_holdup,
        froude_number=froude_number,
    )
    inclination_from_horizontal = (math.pi / 2.0) - inclination_from_vertical

    if is_injector:
        d_coef, e_coef, f_coef, g_coef = 4.7, -0.3692, 0.1244, -0.5056
        correction_argument = (
            d_coef
            * no_slip_holdup**e_coef
            * liquid_velocity_number**f_coef
            * froude_number**g_coef
        )
        correction_coefficient = max(0.0, (1.0 - no_slip_holdup) * math.log(correction_argument))
    else:
        if pattern == 3:  # distributed: no correction
            correction_coefficient = 0.0
        else:
            if pattern == 0:  # segregated
                d_coef, e_coef, f_coef, g_coef = 0.011, -3.768, 3.539, -1.614
            else:  # intermittent or transition
                d_coef, e_coef, f_coef, g_coef = 2.96, 0.305, -0.4473, 0.0978
            correction_argument = (
                d_coef
                * no_slip_holdup**e_coef
                * liquid_velocity_number**f_coef
                * froude_number**g_coef
            )
            correction_coefficient = max(
                0.0, (1.0 - no_slip_holdup) * math.log(correction_argument)
            )

    psi = 1.0 + correction_coefficient * (
        math.sin(1.8 * inclination_from_horizontal)
        - (1.0 / 3.0) * math.sin(1.8 * inclination_from_horizontal) ** 3
    )
    in_situ_holdup = holdup_at_horizontal * psi
    return min(max(in_situ_holdup, no_slip_holdup), 1.0)


@numba.njit(cache=True)
def compute_two_phase_friction_factor(
    no_slip_holdup: Number,
    in_situ_holdup: Number,
    no_slip_reynolds_number: Number,
    relative_roughness: Number,
    method: int,
    laminar_reynolds_limit: Number,
    turbulent_reynolds_limit: Number,
    friction_max_iterations: int,
    friction_tolerance: Number,
) -> Number:
    """
    Computes the two-phase Darcy friction factor. The single-phase
    (no-slip) friction factor, scaled by Beggs & Brill's holdup-ratio correction.

    :param no_slip_holdup: No-slip liquid holdup.
    :param in_situ_holdup: In-situ liquid holdup, from `compute_beggs_brill_holdup`.
    :param no_slip_reynolds_number: No-slip Reynolds number of the mixture.
    :param relative_roughness: Tubing roughness divided by tubing inner diameter.
    :param method: `0` for the simplified correlation, `1` for Colebrook.
    :param laminar_reynolds_limit: Reynolds number below which flow is
        treated as laminar. Only used by the simplified correlation.
    :param turbulent_reynolds_limit: Reynolds number above which flow is
        treated as fully turbulent. Only used by the simplified correlation.
    :param friction_max_iterations: Maximum iterations. Only used by Colebrook.
    :param friction_tolerance: Convergence tolerance. Only used by Colebrook.
    :returns: Two-phase Darcy friction factor.
    """
    if method == 0:
        if no_slip_reynolds_number < laminar_reynolds_limit:
            no_slip_friction_factor = 64.0 / no_slip_reynolds_number
        elif no_slip_reynolds_number < turbulent_reynolds_limit:
            no_slip_friction_factor = 0.316 * no_slip_reynolds_number**-0.25
        else:
            no_slip_friction_factor = (
                0.25
                / (math.log10(relative_roughness / 3.7 + 5.74 / no_slip_reynolds_number**0.9)) ** 2
            )
    else:
        no_slip_friction_factor = 0.02
        for _ in range(friction_max_iterations):
            rhs = -2.0 * math.log10(
                relative_roughness / 3.7
                + 2.51 / (no_slip_reynolds_number * math.sqrt(no_slip_friction_factor))
            )
            updated = 1.0 / rhs**2
            if abs(updated - no_slip_friction_factor) < friction_tolerance:
                no_slip_friction_factor = updated
                break
            no_slip_friction_factor = updated

    y = no_slip_holdup / in_situ_holdup**2
    if 1.0 < y < 1.2:
        s = math.log(2.2 * y - 1.2)
    else:
        ln_y = math.log(y)
        s = ln_y / (-0.0523 + 3.182 * ln_y - 0.8725 * ln_y**2 + 0.01853 * ln_y**4)
    return no_slip_friction_factor * math.exp(s)


@numba.njit(cache=True)
def compute_segment_drop(
    model: BeggsAndBrillWellbore,
    length: Number,
    inclination_from_vertical: Number,
    superficial_liquid_velocity: Number,
    superficial_gas_velocity: Number,
    liquid_density: Number,
    gas_density: Number,
    liquid_viscosity: Number,
    gas_viscosity: Number,
    liquid_surface_tension: Number,
    local_pressure: Number,
    is_injector: Boolean,
) -> PressureDrop:
    """
    Computes the pressure drop across one tubing segment.

    The pressure gradient is the sum of the hydrostatic and friction gradients divided by
    `1 - Ek`, where `Ek = slip density * mixture velocity * gas velocity / pressure` is the
    correction for the gas speeding up as the pressure falls along the flow. The extra
    pressure this adds is reported as the acceleration drop.

    :param model: This well's `BeggsAndBrillWellbore`.
    :param length: Along-wellbore segment length.
    :param inclination_from_vertical: Segment inclination, in radians. `0` is vertical.
    :param superficial_liquid_velocity: Liquid rate divided by cross-sectional area.
    :param superficial_gas_velocity: Gas rate divided by cross-sectional area.
    :param liquid_density: Combined oil-and-water density.
    :param gas_density: Gas density.
    :param liquid_viscosity: Combined oil-and-water viscosity.
    :param gas_viscosity: Gas viscosity.
    :param liquid_surface_tension: Gas-liquid surface tension.
    :param local_pressure: Pressure the velocities and densities refer to, used for the
        acceleration correction. `NaN` leaves the correction out.
    :param is_injector: Whether this well is an injector.
    :returns: Pressure drop for this segment.
    """
    in_situ_holdup = compute_beggs_brill_holdup(
        superficial_liquid_velocity=superficial_liquid_velocity,
        superficial_gas_velocity=superficial_gas_velocity,
        liquid_density=liquid_density,
        liquid_surface_tension=liquid_surface_tension,
        tubing_inner_diameter=model.tubing_inner_diameter,
        inclination_from_vertical=inclination_from_vertical,
        is_injector=is_injector,
        gravitational_acceleration=model.gravitational_acceleration,
    )
    in_situ_density = liquid_density * in_situ_holdup + gas_density * (1.0 - in_situ_holdup)
    vertical_length = length * math.cos(inclination_from_vertical)
    hydrostatic_drop = (
        in_situ_density
        * model.gravitational_acceleration
        * vertical_length
        * model.hydrostatic_scale
    )

    mixture_velocity = superficial_liquid_velocity + superficial_gas_velocity
    no_slip_holdup = superficial_liquid_velocity / mixture_velocity
    no_slip_density = liquid_density * no_slip_holdup + gas_density * (1.0 - no_slip_holdup)
    no_slip_viscosity = liquid_viscosity * no_slip_holdup + gas_viscosity * (1.0 - no_slip_holdup)
    relative_roughness = (
        0.0
        if math.isnan(model.tubing_roughness)
        else model.tubing_roughness / model.tubing_inner_diameter
    )
    no_slip_reynolds_number = (
        no_slip_density
        * mixture_velocity
        * model.tubing_inner_diameter
        / (no_slip_viscosity * model.viscosity_scale)
    )
    friction_factor = compute_two_phase_friction_factor(
        no_slip_holdup=no_slip_holdup,
        in_situ_holdup=in_situ_holdup,
        no_slip_reynolds_number=no_slip_reynolds_number,
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

    acceleration_drop = 0.0
    if not math.isnan(local_pressure):
        # The correction factor stays below 1, as it reaches 1 where the flow would choke.
        kinetic_ratio = min(
            in_situ_density
            * mixture_velocity
            * superficial_gas_velocity
            * model.hydrostatic_scale
            / local_pressure,
            0.95,
        )
        friction_sign = -1.0 if is_injector else 1.0
        acceleration_drop = (
            (hydrostatic_drop + friction_sign * friction_drop)
            * kinetic_ratio
            / (1.0 - kinetic_ratio)
        )
    return PressureDrop(
        hydrostatic=hydrostatic_drop,
        friction=friction_drop,
        acceleration=acceleration_drop,
    )


@numba.njit(cache=True)
def walk_perforation_pressures(
    model: BeggsAndBrillWellbore,
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

    :param model: This well's `BeggsAndBrillWellbore`.
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
                local_pressure=connection_fluids.pressures[idx],  # type: ignore[arg-type]
                is_injector=is_injector,
            )
            pressures[idx] = (
                previous_pressure
                + geometric_sign * (drop.hydrostatic + drop.acceleration)
                + friction_sign * geometric_sign * drop.friction
            )


def compute_perforation_pressures(
    model: BeggsAndBrillWellbore,
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
    out: NumberArray[OneDimension] | None = None,
    dtype: npt.DTypeLike = None,
) -> NumberArray[OneDimension]:
    """
    Computes the flowing pressure at each connection.

    See `walk_perforation_pressures` for how the wellbore is walked.

    :param model: This well's `BeggsAndBrillWellbore`.
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

    if out is not None:
        pressures = out
    else:
        pressures = np.empty(
            count,
            dtype=np.dtype(dtype)
            if dtype is not None
            else np.asarray(representative_depths).dtype,
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
        pressures=typing.cast(NumberArray[OneDimension], pressures),
    )
    return typing.cast(NumberArray[OneDimension], pressures)


@numba.njit(cache=True)
def evaluate_tubing_head_pressure(
    model: BeggsAndBrillWellbore,
    reference_depth: Number,
    reference_pressure: Number,
    phase_rates: PhaseValues,
    surface_fluids: CompiledSurfaceFluids,
    is_injector: Boolean,
) -> Number:
    """
    Computes tubing head pressure at surface from compiled surface fluid properties.

    :param model: This well's `BeggsAndBrillWellbore`.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_pressure: Pressure at `reference_depth`.
    :param phase_rates: Rate of each phase, at reservoir conditions.
    :param surface_fluids: Fluid properties at surface conditions. Needs the phase densities,
        phase viscosities and gas-liquid surface tension.
    :param is_injector: Whether this well is an injector.
    :returns: Tubing head pressure.
    :raises ValueError: If a surface property the correlation needs is `NaN`.
    """
    phase_densities, phase_viscosities, gas_liquid_surface_tension = (
        get_two_phase_surface_properties(surface_fluids=surface_fluids)
    )
    dz = 0.0 - reference_depth
    total_rate = phase_rates.oil + phase_rates.water + phase_rates.gas
    friction_sign = -1.0 if is_injector else 1.0

    if total_rate == 0:
        static_density = compute_static_mixture_density(
            phase_saturations=PhaseValues(
                oil=phase_rates.oil, water=phase_rates.water, gas=phase_rates.gas
            ),
            phase_densities=phase_densities,
        )
        return (
            reference_pressure
            - static_density * model.gravitational_acceleration * abs(dz) * model.hydrostatic_scale
        )

    (
        liquid_rate,
        gas_rate,
        liquid_density,
        gas_density,
        liquid_viscosity,
        gas_viscosity,
    ) = split_liquid_gas(
        phase_rates=phase_rates,
        phase_densities=phase_densities,
        phase_viscosities=phase_viscosities,
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
        liquid_surface_tension=gas_liquid_surface_tension,
        local_pressure=math.nan,
        is_injector=is_injector,
    )
    return (
        reference_pressure - (drop.hydrostatic + drop.acceleration) - friction_sign * drop.friction
    )


def compute_tubing_head_pressure(
    model: BeggsAndBrillWellbore,
    reference_depth: Number,
    reference_pressure: Number,
    phase_rates: PhaseValues,
    surface_sample: SurfaceFluids,
    is_injector: Boolean,
) -> Number:
    """
    Computes tubing head pressure at surface.

    :param model: This well's `BeggsAndBrillWellbore`.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_pressure: Pressure at `reference_depth`.
    :param phase_rates: Rate of each phase, at reservoir conditions.
    :param surface_sample: Fluid properties at surface conditions.
        `phase_densities`, `phase_viscosities`, and `gas_liquid_surface_tension`
        are all required, as a two-phase correlation needs a real liquid/gas
        split, not just a single mixture value.
    :param is_injector: Whether this well is an injector.
    :returns: Tubing head pressure. The acceleration correction is not applied here, since the tubing
        is one segment at surface-sample properties, with no pressure along it to evaluate it at.
    :raises ValueError: If `phase_densities`, `phase_viscosities`, or
        `gas_liquid_surface_tension` isn't set on `surface_sample`.
    """
    surface_fluids = compile_surface_fluids(surface_sample=surface_sample)
    check_two_phase_surface_fluids(surface_fluids=surface_fluids, model_label="Beggs & Brill")
    return evaluate_tubing_head_pressure(
        model=model,
        reference_depth=reference_depth,
        reference_pressure=reference_pressure,
        phase_rates=phase_rates,
        surface_fluids=surface_fluids,
        is_injector=is_injector,
    )
