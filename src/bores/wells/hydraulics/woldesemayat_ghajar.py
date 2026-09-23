"""Woldesemayat and Ghajar (2007) wellbore hydraulics."""

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
from bores.wells.state import ConnectionSample, PhaseValues

__all__ = [
    "WoldesemayatGhajarWellbore",
    "compute_perforation_pressures",
    "compute_segment_drop",
    "compute_tubing_head_pressure",
    "compute_woldesemayat_ghajar_void_fraction",
    "woldesemayat_ghajar_wellbore",
]


class WoldesemayatGhajarWellbore(typing.NamedTuple):
    """Configuration for the Woldesemayat and Ghajar (2007) wellbore hydraulics model."""

    tubing_inner_diameter: Number
    """Tubing inner diameter."""

    tubing_roughness: Number
    """Absolute pipe roughness. `NaN` for a smooth pipe."""

    friction_method: int
    """
    Which single-phase friction-factor correlation to apply: `0` for
    the simplified correlation, `1` for Colebrook.
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

    si_length_factor: Number
    """
    Multiplies a length or velocity in this model's unit system to get
    metres or metres per second. The void-fraction correlation's
    drift-velocity term is only valid in SI, unlike this model's other
    calculations; this converts into SI for that one step and back out.
    """

    si_density_factor: Number
    """
    Multiplies a density in this model's unit system to get kilograms
    per cubic metre, for the same reason as `si_length_factor`.
    """

    si_pressure_factor: Number
    """
    Multiplies a pressure in this model's unit system to get pascals,
    for the same reason as `si_length_factor`.
    """

    standard_gravity_si: Number
    """
    `c.ACCELERATION_DUE_TO_GRAVITY_METER_PER_SECONDS_SQUARE`, resolved
    here rather than as a literal in the njit'd functions that need it
    (`bores.constants`'s proxy can't be resolved inside a numba `njit`
    function), for the drift-velocity term's own SI calculation. Not the
    same thing as `gravitational_acceleration`, which is in this model's
    own unit system and used for the actual hydrostatic term.
    """

    standard_atmosphere_si: Number
    """`c.STANDARD_PRESSURE_PASCAL`, resolved here for the same reason as
    `standard_gravity_si` - the drift-velocity term's own atmospheric-to-local
    pressure ratio."""

    dyne_per_cm_to_newton_per_m: Number
    """`c.DYNE_PER_CENTIMETER_TO_NEWTON_PER_METER`, resolved here for the
    same reason as `standard_gravity_si`. `gas_liquid_surface_tension` is
    always in dyne/cm in this package, regardless of this model's own
    unit system, the same assumption Hagedorn & Brown makes elsewhere."""

    centipoise_to_pascal_second: Number
    """`c.CENTIPOISE_TO_PASCAL_SECONDS`, resolved here for the same reason
    as `standard_gravity_si`. Viscosity is always in cP in this package,
    regardless of this model's own unit system, the same assumption
    Hagedorn & Brown makes elsewhere. Used only for `compute_segment_drop`'s
    own Reynolds number, which must be genuinely dimensionless; the
    friction and hydrostatic terms themselves stay in this model's native
    units throughout, converted by `hydrostatic_scale` as usual."""

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
        si_factors = get_conversion_factors(target, UnitSystem.SI)
        return self._replace(
            tubing_inner_diameter=self.tubing_inner_diameter * length_factor,
            tubing_roughness=self.tubing_roughness * length_factor,
            gravitational_acceleration=self.gravitational_acceleration * length_factor,
            hydrostatic_scale=1.0
            / (
                get_unit_system_constant(prefix="GRAVITATIONAL_FACTOR", unit_system=target)
                * get_unit_system_constant(prefix="HYDROSTATIC_AREA_FACTOR", unit_system=target)
            ),
            si_length_factor=si_factors["length"],
            si_density_factor=si_factors["density"],
            si_pressure_factor=si_factors["pressure"],
            unit_system=target,
        )


def woldesemayat_ghajar_wellbore(
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
    Builds a `WellBoreModel` wrapping a fully configured `WoldesemayatGhajarWellbore`.

    Woldesemayat and Ghajar (2007) is a drift-flux void-fraction
    correlation built from a wide range of experimental data spanning
    horizontal, upward-inclined, and vertical flow, and is not tied to a
    single flow pattern the way Beggs & Brill's holdup correction is.

    The original paper covers void fraction only, not a full pressure
    gradient method. Following this package's own primary-source
    convention (see `hagedorn_brown.py`'s own note on this), the
    in-situ (void-fraction-weighted) mixture density and viscosity are
    used for both gravity and friction, with a single-phase Darcy
    friction factor from this package's own shared correlation, rather
    than a separate two-phase friction multiplier.

    Unlike Gray or Hagedorn & Brown, whose own dimensionless groups are
    calibrated to field units specifically, this correlation's
    drift-velocity term is dimensionally consistent only in SI. This
    model converts to SI for that one calculation and back, so it works
    correctly under any of this package's unit systems, not field units alone.

    :param tubing_inner_diameter: Tubing inner diameter.
    :param tubing_roughness: Absolute pipe roughness. `None` for a smooth pipe.
    :param friction_method: Which single-phase friction-factor correlation to apply.
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
    :returns: `WellBoreModel(name="woldesemayat_ghajar", options=<WoldesemayatGhajarWellbore>)`.
    """
    if gravitational_acceleration is None:
        gravitational_acceleration = typing.cast(
            Number, c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE
        )
        if unit_system != UnitSystem.FIELD:
            factors = get_conversion_factors(UnitSystem.FIELD, unit_system)
            gravitational_acceleration = gravitational_acceleration * factors["length"]

    si_factors = get_conversion_factors(unit_system, UnitSystem.SI)
    options = WoldesemayatGhajarWellbore(
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
        si_length_factor=si_factors["length"],
        si_density_factor=si_factors["density"],
        si_pressure_factor=si_factors["pressure"],
        standard_gravity_si=c.ACCELERATION_DUE_TO_GRAVITY_METER_PER_SECONDS_SQUARE,
        standard_atmosphere_si=c.STANDARD_PRESSURE_PASCAL,
        dyne_per_cm_to_newton_per_m=c.DYNE_PER_CENTIMETER_TO_NEWTON_PER_METER,
        centipoise_to_pascal_second=c.CENTIPOISE_TO_PASCAL_SECONDS,
        unit_system=unit_system,
    )
    return WellBoreModel(name="woldesemayat_ghajar", options=options)


@numba.njit(cache=True)
def compute_woldesemayat_ghajar_void_fraction(
    superficial_liquid_velocity: Number,
    superficial_gas_velocity: Number,
    liquid_density: Number,
    gas_density: Number,
    gas_liquid_surface_tension: Number,
    tubing_inner_diameter: Number,
    inclination_from_vertical: Number,
    pressure: Number,
    si_length_factor: Number,
    si_density_factor: Number,
    si_pressure_factor: Number,
    standard_gravity_si: Number,
    standard_atmosphere_si: Number,
    dyne_per_cm_to_newton_per_m: Number,
) -> Number:
    """
    Computes void fraction (in-situ gas fraction) per Woldesemayat and Ghajar (2007).

    Every argument is in this model's own unit system; `si_length_factor`,
    `si_density_factor`, and `si_pressure_factor` convert into SI
    internally, since the drift-velocity term is only dimensionally
    consistent there. `gas_liquid_surface_tension` is always dyne/cm,
    converted to N/m directly via `dyne_per_cm_to_newton_per_m`.

    The original paper's inclination angle is measured from horizontal,
    with `0` horizontal and increasing upward; this package measures
    `inclination_from_vertical` from vertical instead, with `0` vertical.
    The two relate by `angle_from_horizontal = pi/2 - inclination_from_vertical`,
    so `cos(angle_from_horizontal) = sin(inclination_from_vertical)` and
    `sin(angle_from_horizontal) = cos(inclination_from_vertical)`; both
    are applied below rather than converting the angle itself.

    `standard_gravity_si`/`standard_atmosphere_si`/`dyne_per_cm_to_newton_per_m`
    are `c.<NAME>` constants, resolved once in `woldesemayat_ghajar_wellbore`
    and passed in here rather than referenced directly, since
    `bores.constants`'s proxy can't be resolved inside a numba `njit` function.

    :param superficial_liquid_velocity: Liquid rate divided by cross-sectional area.
    :param superficial_gas_velocity: Gas rate divided by cross-sectional area.
    :param liquid_density: Combined oil-and-water density.
    :param gas_density: Gas density.
    :param gas_liquid_surface_tension: Gas-liquid surface tension, in dyne/cm.
    :param tubing_inner_diameter: Tubing inner diameter.
    :param inclination_from_vertical: Segment inclination, in radians. `0` is vertical.
    :param pressure: Local pressure, for the drift-velocity term's own
        atmospheric-to-local pressure ratio.
    :param si_length_factor: Multiplies a length or velocity in this
        model's unit system to get metres or metres per second.
    :param si_density_factor: Multiplies a density in this model's unit
        system to get kilograms per cubic metre.
    :param si_pressure_factor: Multiplies a pressure in this model's unit
        system to get pascals.
    :param standard_gravity_si: `c.ACCELERATION_DUE_TO_GRAVITY_METER_PER_SECONDS_SQUARE`.
    :param standard_atmosphere_si: `c.STANDARD_PRESSURE_PASCAL`.
    :param dyne_per_cm_to_newton_per_m: `c.DYNE_PER_CENTIMETER_TO_NEWTON_PER_METER`.
    :returns: Void fraction, between `0` and `1`.
    """
    mixture_velocity = superficial_liquid_velocity + superficial_gas_velocity
    if superficial_gas_velocity <= 0.0:
        return 0.0
    no_slip_gas_fraction = superficial_gas_velocity / mixture_velocity
    if superficial_liquid_velocity <= 0.0:
        return 1.0
    if liquid_density <= gas_density or gas_liquid_surface_tension <= 0.0:
        return no_slip_gas_fraction

    density_ratio_exponent = (gas_density / liquid_density) ** 0.1
    distribution_coefficient = no_slip_gas_fraction * (
        1.0 + (superficial_liquid_velocity / superficial_gas_velocity) ** density_ratio_exponent
    )

    diameter_si = tubing_inner_diameter * si_length_factor
    liquid_density_si = liquid_density * si_density_factor
    gas_density_si = gas_density * si_density_factor
    surface_tension_si = gas_liquid_surface_tension * dyne_per_cm_to_newton_per_m
    pressure_si = max(pressure * si_pressure_factor, 1.0)

    buoyancy_term = max(
        standard_gravity_si
        * diameter_si
        * surface_tension_si
        * (1.0 + math.sin(inclination_from_vertical))
        * (liquid_density_si - gas_density_si)
        / liquid_density_si**2,
        0.0,
    )
    inclination_factor = (1.22 + 1.22 * math.cos(inclination_from_vertical)) ** (
        standard_atmosphere_si / pressure_si
    )
    drift_velocity_si = 2.9 * buoyancy_term**0.25 * inclination_factor
    drift_velocity = drift_velocity_si / si_length_factor if si_length_factor > 0.0 else 0.0

    void_fraction = superficial_gas_velocity / (
        distribution_coefficient * mixture_velocity + drift_velocity
    )
    return min(max(void_fraction, 0.0), 1.0)


@numba.njit(cache=True)
def compute_segment_drop(
    model: WoldesemayatGhajarWellbore,
    length: Number,
    inclination_from_vertical: Number,
    superficial_liquid_velocity: Number,
    superficial_gas_velocity: Number,
    liquid_density: Number,
    gas_density: Number,
    liquid_viscosity: Number,
    gas_viscosity: Number,
    gas_liquid_surface_tension: Number,
    pressure: Number,
) -> PressureDrop:
    """
    Computes the pressure drop across one tubing segment.

    Both gravity and friction use the same in-situ (void-fraction-weighted)
    mixture density and viscosity, following this correlation's own
    primary-source recommendation rather than mixing a no-slip friction
    term with a slip gravity term the way Beggs & Brill does.

    :param model: This well's `WoldesemayatGhajarWellbore`.
    :param length: Along-wellbore segment length.
    :param inclination_from_vertical: Segment inclination, in radians. `0` is vertical.
    :param superficial_liquid_velocity: Liquid rate divided by cross-sectional area.
    :param superficial_gas_velocity: Gas rate divided by cross-sectional area.
    :param liquid_density: Combined oil-and-water density.
    :param gas_density: Gas density.
    :param liquid_viscosity: Combined oil-and-water viscosity.
    :param gas_viscosity: Gas viscosity.
    :param gas_liquid_surface_tension: Gas-liquid surface tension.
    :param pressure: Local pressure, passed through to the void-fraction calculation.
    :returns: Pressure drop for this segment.
    """
    mixture_velocity = superficial_liquid_velocity + superficial_gas_velocity
    void_fraction = compute_woldesemayat_ghajar_void_fraction(
        superficial_liquid_velocity=superficial_liquid_velocity,
        superficial_gas_velocity=superficial_gas_velocity,
        liquid_density=liquid_density,
        gas_density=gas_density,
        gas_liquid_surface_tension=gas_liquid_surface_tension,
        tubing_inner_diameter=model.tubing_inner_diameter,
        inclination_from_vertical=inclination_from_vertical,
        pressure=pressure,
        si_length_factor=model.si_length_factor,
        si_density_factor=model.si_density_factor,
        si_pressure_factor=model.si_pressure_factor,
        standard_gravity_si=model.standard_gravity_si,
        standard_atmosphere_si=model.standard_atmosphere_si,
        dyne_per_cm_to_newton_per_m=model.dyne_per_cm_to_newton_per_m,
    )
    in_situ_holdup = 1.0 - void_fraction
    in_situ_density = liquid_density * in_situ_holdup + gas_density * void_fraction
    in_situ_viscosity = liquid_viscosity * in_situ_holdup + gas_viscosity * void_fraction

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
    # Reynolds number must come out the same regardless of unit system, since it's
    # what compute_friction_factor's laminar/turbulent thresholds and Darcy/Colebrook
    # formulas are calibrated against - none of that has any unit-system slack built
    # in. density/velocity/diameter are already correctly rescaled between unit
    # systems by model construction, but viscosity is always cP (see
    # compute_woldesemayat_ghajar_void_fraction's own docstring on this), so computing
    # the ratio directly in native units would silently pick up a spurious unit-system
    # dependence: only the mass/length units of density*velocity*diameter change
    # between systems, cP does not, so the ratio isn't actually dimensionless unless
    # everything is converted to one consistent system first. SI is the one already
    # available via si_length_factor/si_density_factor; centipoise_to_pascal_second
    # (cP -> Pa*s) is the same fixed, unit-system-independent conversion
    # dyne_per_cm_to_newton_per_m already applies to surface tension.
    reynolds_number = (
        (in_situ_density * model.si_density_factor)
        * (mixture_velocity * model.si_length_factor)
        * (model.tubing_inner_diameter * model.si_length_factor)
        / (in_situ_viscosity * model.centipoise_to_pascal_second)
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
            * (in_situ_density * mixture_velocity**2 / 2.0)
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
    model: WoldesemayatGhajarWellbore,
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

    :param model: This well's `WoldesemayatGhajarWellbore`.
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
                    gas_liquid_surface_tension=sample.gas_liquid_surface_tension,
                    pressure=sample.pressure,
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
    model: WoldesemayatGhajarWellbore,
    reference_depth: Number,
    reference_pressure: Number,
    phase_rates: PhaseValues,
    surface_fluid_properties: SurfaceFluidProperties,
    is_injector: bool,
) -> Number:
    """
    Computes tubing head pressure at surface.

    :param model: This well's `WoldesemayatGhajarWellbore`.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_pressure: Pressure at `reference_depth`.
    :param phase_rates: Rate of each phase, at reservoir conditions.
    :param surface_fluid_properties: Fluid properties at surface
        conditions. `phase_densities`, `phase_viscosities`, and
        `gas_liquid_surface_tension` are all required, as this
        correlation needs a real liquid/gas split, not just a single
        mixture value.
    :param is_injector: Whether this well is an injector.
    :returns: Tubing head pressure.
    :raises ValueError: If `phase_densities`, `phase_viscosities`, or
        `gas_liquid_surface_tension` isn't set on `surface_fluid_properties`.
    """
    if surface_fluid_properties.phase_densities is None:
        raise ValueError(
            "SurfaceFluidProperties.phase_densities is required for the "
            "Woldesemayat and Ghajar wellbore model."
        )
    if surface_fluid_properties.phase_viscosities is None:
        raise ValueError(
            "SurfaceFluidProperties.phase_viscosities is required for the "
            "Woldesemayat and Ghajar wellbore model."
        )
    if surface_fluid_properties.gas_liquid_surface_tension is None:
        raise ValueError(
            "SurfaceFluidProperties.gas_liquid_surface_tension is required for the "
            "Woldesemayat and Ghajar wellbore model."
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
        pressure=reference_pressure,
    )
    return (
        reference_pressure - (drop.hydrostatic + drop.acceleration) - friction_sign * drop.friction
    )
