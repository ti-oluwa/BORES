"""Gray (1974) wellbore hydraulics, API 14B form."""

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
from bores.wells.hydraulics.base import (
    PressureDrop,
    SurfaceFluidProperties,
    WellBoreModel,
    compute_friction_factor,
    compute_static_hydrostatic_drop,
    compute_static_mixture_density,
    get_unit_system_constant,
    split_liquid_gas,
)
from bores.wells.states import ConnectionSample, PhaseValues

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
    """Which single-phase friction-factor correlation to apply, using
    Gray's own effective roughness in place of `tubing_roughness`: `0`
    for the simplified correlation, `1` for Colebrook."""

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
    conversion; without it, friction comes out several thousand times
    too large relative to hydrostatic in field units."""

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
            tubing_inner_diameter=self.tubing_inner_diameter * length_factor,
            tubing_roughness=self.tubing_roughness * length_factor,
            gravitational_acceleration=self.gravitational_acceleration * length_factor,
            hydrostatic_scale=1.0
            / (
                get_unit_system_constant(prefix="GRAVITATIONAL_FACTOR", unit_system=target)
                * get_unit_system_constant(prefix="HYDROSTATIC_AREA_FACTOR", unit_system=target)
            ),
            unit_system=target,
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
    API 14B and widely used for mist-flow gas wells. Field units
    throughout: the holdup and effective-roughness correlations are
    calibrated for velocities in ft/s, densities in lbm/ft3, surface
    tension in dyne/cm, and diameter in ft, the same assumption Hagedorn
    & Brown makes elsewhere in this package.

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
        hydrostatic_scale=1.0
        / (
            get_unit_system_constant(prefix="GRAVITATIONAL_FACTOR", unit_system=unit_system)
            * get_unit_system_constant(prefix="HYDROSTATIC_AREA_FACTOR", unit_system=unit_system)
        ),
        unit_system=unit_system,
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

    in_situ_holdup = compute_gray_holdup(
        superficial_liquid_velocity=superficial_liquid_velocity,
        superficial_gas_velocity=superficial_gas_velocity,
        liquid_density=liquid_density,
        gas_density=gas_density,
        gas_liquid_surface_tension=gas_liquid_surface_tension,
        tubing_inner_diameter=model.tubing_inner_diameter,
    )
    in_situ_density = liquid_density * in_situ_holdup + gas_density * (1.0 - in_situ_holdup)

    vertical_length = length * math.cos(inclination_from_vertical)
    hydrostatic_drop = (
        in_situ_density
        * model.gravitational_acceleration
        * vertical_length
        * model.hydrostatic_scale
    )

    effective_roughness = compute_gray_effective_roughness(
        tubing_roughness=model.tubing_roughness,
        gas_liquid_surface_tension=gas_liquid_surface_tension,
        no_slip_density=no_slip_density,
        superficial_liquid_velocity=superficial_liquid_velocity,
        superficial_gas_velocity=superficial_gas_velocity,
    )
    relative_roughness = effective_roughness / model.tubing_inner_diameter
    no_slip_reynolds_number = (
        no_slip_density * mixture_velocity * model.tubing_inner_diameter / no_slip_viscosity
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


def compute_perforation_pressures(
    model: GrayWellbore,
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

    The wellbore is split at `reference_depth` into up to two branches -
    connections at or below it, and connections above it - each walked
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

    :param model: This well's `GrayWellbore`.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_pressure: Pressure at `reference_depth`.
    :param connection_phase_rates: Each connection's own rate of each
        phase, at reservoir conditions - not the well total. Same order
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
    n = len(connection_samples)
    if out is not None and len(out) != n:
        raise ValueError("If given, `out` must have the same length as `connection_samples`.")
    if not (
        len(representative_depths)
        == len(inclinations_from_vertical)
        == len(connection_phase_rates)
        == n
    ):
        raise ValueError(
            "`representative_depths`, `inclinations_from_vertical`, "
            "`connection_phase_rates`, and `connection_samples` must all have the same length."
        )

    if out is not None:
        pressures = out
    else:
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        pressures = np.empty(n, dtype=dtype)

    friction_sign = -1.0 if is_injector else 1.0
    cross_sectional_area = math.pi * (model.tubing_inner_diameter / 2.0) ** 2

    below = sorted(
        (i for i in range(n) if representative_depths[i] >= reference_depth),
        key=lambda i: representative_depths[i],
    )
    above = sorted(
        (i for i in range(n) if representative_depths[i] < reference_depth),
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
                    gas_liquid_surface_tension=sample.gas_liquid_surface_tension,
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
    model: GrayWellbore,
    reference_depth: Number,
    reference_pressure: Number,
    phase_rates: PhaseValues,
    surface_fluid_properties: SurfaceFluidProperties,
    is_injector: bool,
) -> Number:
    """
    Computes tubing head pressure at surface.

    :param model: This well's `GrayWellbore`.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_pressure: Pressure at `reference_depth`.
    :param phase_rates: Rate of each phase, at reservoir conditions.
    :param surface_fluid_properties: Fluid properties at surface
        conditions. `phase_densities`, `phase_viscosities`, and
        `gas_liquid_surface_tension` are all required, as Gray needs a
        real liquid/gas split, not just a single mixture value.
    :param is_injector: Whether this well is an injector.
    :returns: Tubing head pressure.
    :raises ValueError: If `phase_densities`, `phase_viscosities`, or
        `gas_liquid_surface_tension` isn't set on `surface_fluid_properties`.
    """
    if surface_fluid_properties.phase_densities is None:
        raise ValueError(
            "SurfaceFluidProperties.phase_densities is required for the Gray wellbore model."
        )
    if surface_fluid_properties.phase_viscosities is None:
        raise ValueError(
            "SurfaceFluidProperties.phase_viscosities is required for the Gray wellbore model."
        )
    if surface_fluid_properties.gas_liquid_surface_tension is None:
        raise ValueError(
            "SurfaceFluidProperties.gas_liquid_surface_tension is required for the "
            "Gray wellbore model."
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
        gas_liquid_surface_tension=surface_fluid_properties.gas_liquid_surface_tension,
    )
    return (
        reference_pressure - (drop.hydrostatic + drop.acceleration) - friction_sign * drop.friction
    )
