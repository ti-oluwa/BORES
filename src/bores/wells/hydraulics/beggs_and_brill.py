"""Beggs & Brill (1973) two-phase slip wellbore hydraulics."""

import math
import typing

import numba
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.constants import c, get_conversion_factors
from bores.precision import get_dtype
from bores.types import (
    FrictionMethod,
    Number,
    NumberArray,
    OneDimension,
    UnitConversionTable,
    UnitSystem,
)
from bores.utils import scale
from bores.wells.hydraulics.base import (
    PressureDrop,
    SurfaceFluidProperties,
    WellBoreModel,
    compute_static_hydrostatic_drop,
    compute_static_mixture_density,
    get_unit_system_constant,
    split_liquid_gas,
)
from bores.wells.states import ConnectionSample, PhaseValues

__all__ = [
    "BeggsAndBrillWellbore",
    "beggs_and_brill_wellbore",
    "compute_beggs_brill_holdup",
    "compute_horizontal_holdup",
    "compute_perforation_pressures",
    "compute_segment_drop",
    "compute_tubing_head_pressure",
    "compute_two_phase_friction_factor",
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

    hydrostatic_scale: Number
    """Unit-conversion factor converting a `density * velocity-squared` or
    `density * gravitational_acceleration * length` term into this
    model's own pressure unit. Applied to both the hydrostatic and the
    friction term, since both are that same kind of quantity before
    conversion."""

    unit_system: UnitSystem
    """This model's unit system."""

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
            hydrostatic_scale=1.0
            / (
                get_unit_system_constant(prefix="GRAVITATIONAL_FACTOR", unit_system=target)
                * get_unit_system_constant(prefix="HYDROSTATIC_AREA_FACTOR", unit_system=target)
            ),
            unit_system=target,
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
        hydrostatic_scale=1.0
        / (
            get_unit_system_constant(prefix="GRAVITATIONAL_FACTOR", unit_system=unit_system)
            * get_unit_system_constant(prefix="HYDROSTATIC_AREA_FACTOR", unit_system=unit_system)
        ),
        unit_system=unit_system,
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
    is_injector: bool,
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
    theta_from_horizontal = (math.pi / 2.0) - inclination_from_vertical

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
        math.sin(1.8 * theta_from_horizontal)
        - (1.0 / 3.0) * math.sin(1.8 * theta_from_horizontal) ** 3
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
    is_injector: bool,
) -> PressureDrop:
    """
    Computes the pressure drop across one tubing segment.

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
        no_slip_density * mixture_velocity * model.tubing_inner_diameter / no_slip_viscosity
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

    # TODO: Look into this and if this is valid
    # Velocity is only ever set at a connection, where a perforation's own
    # rate joins or leaves the flow (see compute_perforation_pressures).
    # Within one segment there is no other source of velocity change
    # (constant PVT properties are assumed along the segment), so this
    # term is always zero today. The density convention below still
    # follows the correlation's own spec (slip density, matching the
    # gravity term) for whenever that stops being true.
    acceleration_drop = in_situ_density * (mixture_velocity**2 - mixture_velocity**2) / 2.0
    return PressureDrop(
        hydrostatic=hydrostatic_drop,
        friction=friction_drop,
        acceleration=acceleration_drop,
    )


def compute_perforation_pressures(
    model: BeggsAndBrillWellbore,
    reference_depth: Number,
    reference_pressure: Number,
    connection_phase_rates: typing.Sequence[PhaseValues],
    representative_depths: NumberArray[OneDimension],
    inclinations_from_vertical: NumberArray[OneDimension],
    connection_samples: typing.Sequence[ConnectionSample],
    is_injector: bool,
    out: NumberArray[OneDimension] | None = None,
    dtype: npt.DTypeLike = None,
) -> NumberArray[OneDimension]:
    """
    Computes flowing pressure at each perforation connection, integrating
    the wellbore sequentially from `reference_depth` rather than treating
    each connection as an independent path from the reference.

    The wellbore is split at `reference_depth` into up to two branches,
    connections at or below it, and connections above it, each walked
    independently outward from the reference, nearest connection first.
    The segment feeding into a connection carries the combined rate of
    that connection and every connection beyond it on the same branch
    (not yet joined/still to be added to the branch's cumulative flow);
    once a connection is passed, its own rate is removed from the running
    total for the next segment. This holds for both production (rate
    accumulates as segments approach the reference) and injection (rate
    depletes as segments move away from the reference) under the same
    walk, since both describe a monotonically decreasing carried rate
    with distance from the reference.

    :param model: This well's `BeggsAndBrillWellbore`.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_pressure: Pressure at `reference_depth`.
    :param connection_phase_rates: Each connection's own rate of each
        phase, at reservoir conditions, not the well total in same order
        as `connection_samples`.
    :param representative_depths: One depth per connection, same order as `connection_samples`.
    :param inclinations_from_vertical: One inclination per connection, in
        radians, same order as `connection_samples`.
    :param connection_samples: Reservoir conditions at each connection.
    :param is_injector: Whether this well is an injector.
    :param out: Optional preallocated output array. If given, must have the same
        length as `connection_samples`.
    :param dtype: Optional output array data type. Ignored if `out` is given.
    :returns: Pressure at each connection, same order as `connection_samples`.
    :raises ValueError: If `representative_depths`, `inclinations_from_vertical`,
        `connection_phase_rates`, and `connection_samples` don't all have the same length.
    """
    n_samples = len(connection_samples)
    if out is not None and len(out) != n_samples:
        raise ValueError("If given, `out` must have the same length as `connection_samples`.")
    if not (
        len(representative_depths)
        == len(inclinations_from_vertical)
        == len(connection_phase_rates)
        == n_samples
    ):
        raise ValueError(
            "`representative_depths`, `inclinations_from_vertical`, "
            "`connection_phase_rates`, and `connection_samples` must all have the same length."
        )

    if out is not None:
        pressures = out
    else:
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        pressures = np.empty(n_samples, dtype=dtype)

    friction_sign = -1.0 if is_injector else 1.0
    cross_sectional_area = math.pi * (model.tubing_inner_diameter / 2.0) ** 2

    below = sorted(
        (i for i in range(n_samples) if representative_depths[i] >= reference_depth),
        key=lambda i: representative_depths[i],
    )
    above = sorted(
        (i for i in range(n_samples) if representative_depths[i] < reference_depth),
        key=lambda i: -representative_depths[i],
    )

    for branch in (below, above):
        if not branch:
            continue

        remaining_rates = PhaseValues(
            oil=sum(connection_phase_rates[i].oil for i in branch),
            water=sum(connection_phase_rates[i].water for i in branch),
            gas=sum(connection_phase_rates[i].gas for i in branch),
        )
        current_depth = reference_depth
        current_pressure = reference_pressure

        for i in branch:
            length = abs(representative_depths[i] - current_depth)
            geometric_sign = 1.0 if representative_depths[i] >= current_depth else -1.0
            sample = connection_samples[i]
            remaining_total = remaining_rates.oil + remaining_rates.water + remaining_rates.gas

            if remaining_total == 0:
                drop = compute_static_hydrostatic_drop(
                    mixture_density=compute_static_mixture_density(
                        phase_saturations=sample.phase_saturations,
                        phase_densities=sample.phase_densities,
                    ),
                    length=length,
                    gravitational_acceleration=model.gravitational_acceleration,
                    unit_system=model.unit_system,
                )
                current_pressure = current_pressure + geometric_sign * drop.total
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
                    phase_densities=sample.phase_densities,
                    phase_viscosities=sample.phase_viscosities,
                )
                drop = compute_segment_drop(
                    model=model,
                    length=length,
                    inclination_from_vertical=inclinations_from_vertical[i],
                    superficial_liquid_velocity=liquid_rate / cross_sectional_area,
                    superficial_gas_velocity=gas_rate / cross_sectional_area,
                    liquid_density=liquid_density,
                    gas_density=gas_density,
                    liquid_viscosity=liquid_viscosity,
                    gas_viscosity=gas_viscosity,
                    liquid_surface_tension=sample.gas_liquid_surface_tension,
                    is_injector=is_injector,
                )
                current_pressure = (
                    current_pressure
                    + geometric_sign * (drop.hydrostatic + drop.acceleration)
                    + friction_sign * drop.friction
                )

            pressures[i] = current_pressure
            current_depth = representative_depths[i]
            remaining_rates = PhaseValues(
                oil=remaining_rates.oil - connection_phase_rates[i].oil,
                water=remaining_rates.water - connection_phase_rates[i].water,
                gas=remaining_rates.gas - connection_phase_rates[i].gas,
            )

    return pressures


def compute_tubing_head_pressure(
    model: BeggsAndBrillWellbore,
    reference_depth: Number,
    reference_pressure: Number,
    phase_rates: PhaseValues,
    surface_fluid_properties: SurfaceFluidProperties,
    is_injector: bool,
) -> Number:
    """
    Computes tubing head pressure at surface.

    :param model: This well's `BeggsAndBrillWellbore`.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_pressure: Pressure at `reference_depth`.
    :param phase_rates: Rate of each phase, at reservoir conditions.
    :param surface_fluid_properties: Fluid properties at surface conditions.
        `phase_densities`, `phase_viscosities`, and `gas_liquid_surface_tension`
        are all required, as a two-phase correlation needs a real liquid/gas
        split, not just a single mixture value.
    :param is_injector: Whether this well is an injector.
    :returns: Tubing head pressure.
    :raises ValueError: If `phase_densities`, `phase_viscosities`, or
        `gas_liquid_surface_tension` isn't set on `surface_fluid_properties`.
    """
    if surface_fluid_properties.phase_densities is None:
        raise ValueError(
            "`SurfaceFluidProperties.phase_densities` is required for the Beggs & Brill "
            "wellbore model."
        )
    if surface_fluid_properties.phase_viscosities is None:
        raise ValueError(
            "`SurfaceFluidProperties.phase_viscosities` is required for the Beggs & Brill "
            "wellbore model."
        )
    if surface_fluid_properties.gas_liquid_surface_tension is None:
        raise ValueError(
            "`SurfaceFluidProperties.gas_liquid_surface_tension` is required for the "
            "Beggs & Brill wellbore model."
        )

    dz = 0.0 - reference_depth
    total_rate = phase_rates.oil + phase_rates.water + phase_rates.gas
    friction_sign = -1.0 if is_injector else 1.0
    cross_sectional_area = math.pi * (model.tubing_inner_diameter / 2.0) ** 2

    if total_rate == 0:
        drop = compute_static_hydrostatic_drop(
            mixture_density=compute_static_mixture_density(
                phase_saturations=PhaseValues(
                    oil=phase_rates.oil, water=phase_rates.water, gas=phase_rates.gas
                ),
                phase_densities=surface_fluid_properties.phase_densities,
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
        phase_densities=surface_fluid_properties.phase_densities,
        phase_viscosities=surface_fluid_properties.phase_viscosities,
    )
    drop = compute_segment_drop(
        model=model,
        length=abs(dz),
        inclination_from_vertical=0.0,
        superficial_liquid_velocity=liquid_rate / cross_sectional_area,
        superficial_gas_velocity=gas_rate / cross_sectional_area,
        liquid_density=liquid_density,
        gas_density=gas_density,
        liquid_viscosity=liquid_viscosity,
        gas_viscosity=gas_viscosity,
        liquid_surface_tension=surface_fluid_properties.gas_liquid_surface_tension,
        is_injector=is_injector,
    )
    return (
        reference_pressure - (drop.hydrostatic + drop.acceleration) - friction_sign * drop.friction
    )
