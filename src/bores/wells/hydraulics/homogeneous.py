"""No-slip mixture wellbore hydraulics."""

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
    compute_mixture_density,
    compute_mixture_velocity,
    compute_mixture_viscosity,
    compute_segment_pressure_drop,
    compute_static_hydrostatic_drop,
    compute_static_mixture_density,
    compute_surface_mixture_density,
    compute_surface_mixture_viscosity,
    get_unit_system_constant,
)
from bores.wells.states import ConnectionSample, PhaseValues

__all__ = [
    "HomogeneousWellbore",
    "compute_perforation_pressures",
    "compute_segment_drop",
    "compute_tubing_head_pressure",
    "homogeneous_wellbore",
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

    hydrostatic_scale: Number
    """
    Unit-conversion factor converting a `density * velocity-squared` or
    `density * gravitational_acceleration * length` term into this
    model's own pressure unit. Applied to both the hydrostatic and the
    friction term, since both are that same kind of quantity before
    conversion.
    """

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
    :returns: `WellBoreModel(name="homogeneous", options=<HomogeneousWellbore>)`.
    """
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
        hydrostatic_scale=1.0
        / (
            get_unit_system_constant(prefix="GRAVITATIONAL_FACTOR", unit_system=unit_system)
            * get_unit_system_constant(prefix="HYDROSTATIC_AREA_FACTOR", unit_system=unit_system)
        ),
        unit_system=unit_system,
    )
    return WellBoreModel(name="homogeneous", options=options)


@numba.njit(cache=True)
def compute_segment_drop(
    model: HomogeneousWellbore,
    length: Number,
    inclination_from_vertical: Number,
    mixture_density: Number,
    mixture_viscosity: Number,
    mixture_velocity_in: Number,
    mixture_velocity_out: Number,
) -> PressureDrop:
    """
    Computes the pressure drop across one tubing segment.

    :param model: This well's `HomogeneousWellbore`.
    :param length: Along-wellbore segment length.
    :param inclination_from_vertical: Segment inclination, in radians. `0` is vertical.
    :param mixture_density: No-slip mixture density for this segment.
    :param mixture_viscosity: No-slip mixture viscosity for this segment.
    :param mixture_velocity_in: Superficial mixture velocity entering the segment.
    :param mixture_velocity_out: Superficial mixture velocity leaving the segment.
    :returns: Pressure drop for this segment.
    """
    return compute_segment_pressure_drop(
        length=length,
        inclination_from_vertical=inclination_from_vertical,
        tubing_inner_diameter=model.tubing_inner_diameter,
        tubing_roughness=model.tubing_roughness,
        mixture_density=mixture_density,
        mixture_viscosity=mixture_viscosity,
        mixture_velocity_in=mixture_velocity_in,
        mixture_velocity_out=mixture_velocity_out,
        gravitational_acceleration=model.gravitational_acceleration,
        hydrostatic_scale=model.hydrostatic_scale,
        method=model.friction_method,
        laminar_reynolds_limit=model.laminar_reynolds_limit,
        turbulent_reynolds_limit=model.turbulent_reynolds_limit,
        friction_max_iterations=model.friction_max_iterations,
        friction_tolerance=model.friction_tolerance,
    )


def compute_perforation_pressures(
    model: HomogeneousWellbore,
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

    :param model: This well's `HomogeneousWellbore`.
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
                mixture_density = compute_mixture_density(
                    phase_rates=remaining_rates, phase_densities=sample.phase_densities
                )
                mixture_viscosity = compute_mixture_viscosity(
                    phase_rates=remaining_rates,
                    phase_viscosities=sample.phase_viscosities,
                )
                velocity = compute_mixture_velocity(
                    phase_rates=remaining_rates,
                    tubing_inner_diameter=model.tubing_inner_diameter,
                )
                # Velocity is only ever set at a connection, where a
                # perforation's own rate joins or leaves the flow (see the
                # walk above). Within one segment there is no other
                # source of velocity change (constant PVT properties are
                # assumed along the segment), so passing the same value
                # in and out here means the acceleration term always
                # comes out to zero.
                drop = compute_segment_drop(
                    model=model,
                    length=length,
                    inclination_from_vertical=inclinations_from_vertical[i],
                    mixture_density=mixture_density,
                    mixture_viscosity=mixture_viscosity,
                    mixture_velocity_in=velocity,
                    mixture_velocity_out=velocity,
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
    model: HomogeneousWellbore,
    reference_depth: Number,
    reference_pressure: Number,
    phase_rates: PhaseValues,
    surface_fluid_properties: SurfaceFluidProperties,
    is_injector: bool,
) -> Number:
    """
    Computes tubing head pressure at surface.

    :param model: This well's `HomogeneousWellbore`.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_pressure: Pressure at `reference_depth`.
    :param phase_rates: Rate of each phase, at reservoir conditions.
    :param surface_fluid_properties: Fluid properties at surface conditions.
    :param is_injector: Whether this well is an injector.
    :returns: Tubing head pressure.
    """
    dz = 0.0 - reference_depth
    total_rate = phase_rates.oil + phase_rates.water + phase_rates.gas
    friction_sign = -1.0 if is_injector else 1.0
    mixture_density = compute_surface_mixture_density(
        properties=surface_fluid_properties, phase_rates=phase_rates
    )

    if total_rate == 0:
        drop = compute_static_hydrostatic_drop(
            mixture_density=mixture_density,
            length=abs(dz),
            gravitational_acceleration=model.gravitational_acceleration,
            unit_system=model.unit_system,
        )
        return reference_pressure - drop.total

    mixture_viscosity = compute_surface_mixture_viscosity(
        properties=surface_fluid_properties, phase_rates=phase_rates
    )
    velocity = compute_mixture_velocity(
        phase_rates=phase_rates, tubing_inner_diameter=model.tubing_inner_diameter
    )
    # Same simplification as compute_perforation_pressures: velocity is
    # held constant across this tubing segment, so the acceleration term
    # always comes out to zero here too.
    drop = compute_segment_drop(
        model=model,
        length=abs(dz),
        inclination_from_vertical=0.0,
        mixture_density=mixture_density,
        mixture_viscosity=mixture_viscosity,
        mixture_velocity_in=velocity,
        mixture_velocity_out=velocity,
    )
    return (
        reference_pressure - (drop.hydrostatic + drop.acceleration) - friction_sign * drop.friction
    )
