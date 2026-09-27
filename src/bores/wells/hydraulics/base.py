"""Shared wellbore hydraulics primitives."""

import math
import typing

import numba
from typing_extensions import Self

from bores.constants import c
from bores.errors import ValidationError
from bores.types import Number, NumberArray, OneDimension, UnitConversionTable, UnitSystem
from bores.wells.state import PhaseValues

__all__ = [
    "PressureDrop",
    "SurfaceFluidProperties",
    "WellBoreModel",
    "WellBoreModelOptions",
    "compute_friction_factor",
    "compute_hydrostatic_pressure",
    "compute_mixture_density",
    "compute_mixture_velocity",
    "compute_mixture_viscosity",
    "compute_perforation_pressures",
    "compute_segment_pressure_drop",
    "compute_static_hydrostatic_drop",
    "compute_static_mixture_density",
    "compute_superficial_velocity",
    "compute_surface_mixture_density",
    "compute_surface_mixture_viscosity",
    "compute_tubing_head_pressure",
    "get_unit_system_constant",
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
    """Pressure change from the fluid speeding up or slowing down along the segment."""

    @property
    def total(self) -> Number:
        """The combined pressure drop across the segment."""
        return self.hydrostatic + self.friction + self.acceleration


class SurfaceFluidProperties(typing.NamedTuple):
    """Fluid properties at surface conditions, for computing tubing head pressure."""

    density: Number | None = None
    """Mixture density at surface conditions. Required unless `phase_densities` is given."""

    viscosity: Number | None = None
    """Mixture viscosity at surface conditions. Required unless `phase_viscosities` is given."""

    phase_densities: PhaseValues | None = None
    """Density of each phase at surface conditions. Required for a
    two-phase slip correlation such as Beggs & Brill."""

    phase_viscosities: PhaseValues | None = None
    """
    Viscosity of each phase at surface conditions. Required for a
    two-phase slip correlation such as Beggs & Brill.
    """

    gas_liquid_surface_tension: Number | None = None
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
    """
    This correlation's own configuration. `HomogeneousWellbore`,
    `BeggsAndBrillWellbore`, `HagedornBrownWellbore`, or any other correlation's own `NamedTuple`.
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
        """
        if not hasattr(self.options, "convert"):
            raise ValueError(
                f"{self.__class__.__name__} {self.name!r}'s options ({type(self.options).__name__}) "
                "has no convert method."
            )
        return self._replace(options=self.options.convert(target, table=table))  # type: ignore[attr-defined]


@numba.njit(cache=True)
def compute_mixture_density(phase_rates: PhaseValues, phase_densities: PhaseValues) -> Number:
    """
    Computes the no-slip, rate-weighted mixture density of a multiphase stream.

    :param phase_rates: Rate of each phase.
    :param phase_densities: Density of each phase, at the same condition as `phase_rates`.
    :returns: Mixture density.
    :raises ValueError: If `phase_rates` sums to zero. Use
        `compute_static_hydrostatic_drop` for a well with no flow.
    """
    total_rate = phase_rates.oil + phase_rates.water + phase_rates.gas
    if total_rate == 0:
        raise ValueError("`phase_rates` sums to zero; use the static no-flow case instead")
    return (
        phase_rates.oil * phase_densities.oil
        + phase_rates.water * phase_densities.water
        + phase_rates.gas * phase_densities.gas
    ) / total_rate


@numba.njit(cache=True)
def compute_mixture_viscosity(phase_rates: PhaseValues, phase_viscosities: PhaseValues) -> Number:
    """
    Computes the no-slip, rate-weighted mixture viscosity of a multiphase stream.

    :param phase_rates: Rate of each phase.
    :param phase_viscosities: Viscosity of each phase, at the same condition as `phase_rates`.
    :returns: Mixture viscosity.
    :raises ValueError: If `phase_rates` sums to zero. Use
        `compute_static_hydrostatic_drop` for a well with no flow.
    """
    total_rate = phase_rates.oil + phase_rates.water + phase_rates.gas
    if total_rate == 0:
        raise ValueError("`phase_rates` sums to zero; use the static no-flow case instead")
    return (
        phase_rates.oil * phase_viscosities.oil
        + phase_rates.water * phase_viscosities.water
        + phase_rates.gas * phase_viscosities.gas
    ) / total_rate


def compute_superficial_velocity(rate: Number, tubing_inner_diameter: Number) -> Number:
    """
    Converts a single phase's day-rate to a superficial velocity in tubing.

    :param rate: Phase rate, at reservoir conditions, per day.
    :param tubing_inner_diameter: Tubing inner diameter.
    :returns: Superficial velocity, per second (matching the per-second
        `gravitational_acceleration`/`hydrostatic_scale` convention every
        correlation's hydrostatic term already uses).
    :raises ValueError: If `tubing_inner_diameter` isn't positive.
    """
    if tubing_inner_diameter <= 0.0:
        raise ValueError("`tubing_inner_diameter` must be positive")
    cross_sectional_area = math.pi * (tubing_inner_diameter / 2.0) ** 2
    return (rate * c.DAYS_PER_SECOND) / cross_sectional_area


def compute_mixture_velocity(phase_rates: PhaseValues, tubing_inner_diameter: Number) -> Number:
    """
    Computes the no-slip superficial velocity of a multiphase stream in tubing.

    :param phase_rates: Rate of each phase, at reservoir conditions, per day.
    :param tubing_inner_diameter: Tubing inner diameter.
    :returns: Mixture velocity, per second.
    :raises ValueError: If `tubing_inner_diameter` isn't positive.
    """
    total_rate = phase_rates.oil + phase_rates.water + phase_rates.gas
    return compute_superficial_velocity(total_rate, tubing_inner_diameter)


def compute_surface_mixture_density(
    properties: SurfaceFluidProperties, phase_rates: PhaseValues
) -> Number:
    """
    Gets the surface mixture density from `properties`, computing it from
    per-phase data if it wasn't supplied directly.

    :param properties: Surface fluid properties for the well.
    :param phase_rates: Rate of each phase, at surface conditions.
    :returns: Mixture density at surface conditions.
    :raises ValueError: If neither `properties.density` nor
        `properties.phase_densities` is set.
    """
    if properties.density is not None:
        return properties.density
    if properties.phase_densities is None:
        raise ValueError("`SurfaceFluidProperties` needs density or `phase_densities`")
    return compute_mixture_density(
        phase_rates=phase_rates, phase_densities=properties.phase_densities
    )


def compute_surface_mixture_viscosity(
    properties: SurfaceFluidProperties, phase_rates: PhaseValues
) -> Number:
    """
    Gets the surface mixture viscosity from `properties`, computing it
    from per-phase data if it wasn't supplied directly.

    :param properties: Surface fluid properties for the well.
    :param phase_rates: Rate of each phase, at surface conditions.
    :returns: Mixture viscosity at surface conditions.
    :raises ValueError: If neither `properties.viscosity` nor
        `properties.phase_viscosities` is set.
    """
    if properties.viscosity is not None:
        return properties.viscosity
    if properties.phase_viscosities is None:
        raise ValueError("`SurfaceFluidProperties` needs viscosity or `phase_viscosities`")
    return compute_mixture_viscosity(
        phase_rates=phase_rates, phase_viscosities=properties.phase_viscosities
    )


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


@numba.njit(cache=True)
def compute_segment_pressure_drop(
    length: Number,
    inclination_from_vertical: Number,
    tubing_inner_diameter: Number,
    tubing_roughness: Number,
    mixture_density: Number,
    mixture_viscosity: Number,
    mixture_velocity_in: Number,
    mixture_velocity_out: Number,
    gravitational_acceleration: Number,
    hydrostatic_scale: Number,
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
    :param mixture_velocity_in: Superficial mixture velocity entering the segment.
    :param mixture_velocity_out: Superficial mixture velocity leaving the segment.
    :param gravitational_acceleration: Acceleration due to gravity, already unit-resolved.
    :param hydrostatic_scale: Unit-conversion factor converting a
        `density * velocity-squared` or `density * gravitational_acceleration
        * length` term into the caller's own pressure unit -
        `1 / (gravitational_factor * hydrostatic_area_factor)` for that
        unit system. Applied to both the hydrostatic and the friction
        term, since both are that same kind of quantity before
        conversion; without it, friction comes out several thousand
        times too large relative to hydrostatic in field units.
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

    mean_velocity = 0.5 * (mixture_velocity_in + mixture_velocity_out)
    relative_roughness = (
        0.0 if math.isnan(tubing_roughness) else tubing_roughness / tubing_inner_diameter
    )
    reynolds_number = (
        mixture_density * abs(mean_velocity) * tubing_inner_diameter / mixture_viscosity
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

    acceleration_drop = mixture_density * (mixture_velocity_out**2 - mixture_velocity_in**2) / 2.0
    return PressureDrop(
        hydrostatic=hydrostatic_drop,
        friction=friction_drop,
        acceleration=acceleration_drop,
    )


def split_liquid_gas(
    phase_rates: PhaseValues,
    phase_densities: PhaseValues,
    phase_viscosities: PhaseValues,
) -> tuple[Number, Number, Number, Number, Number, Number]:
    """
    Pools oil and water into a combined liquid phase, for a two-phase
    correlation such as Beggs & Brill or Hagedorn & Brown that treats
    liquid and gas as the only two phases and doesn't distinguish oil from water.

    :param phase_rates: Rate of each phase.
    :param phase_densities: Density of each phase, at the same condition as `phase_rates`.
    :param phase_viscosities: Viscosity of each phase, at the same condition as `phase_rates`.
    :returns: `(liquid_rate, gas_rate, liquid_density, gas_density, liquid_viscosity, gas_viscosity)`.
    """
    liquid_rate = phase_rates.oil + phase_rates.water
    gas_rate = phase_rates.gas
    if liquid_rate > 0.0:
        liquid_density = (
            phase_rates.oil * phase_densities.oil + phase_rates.water * phase_densities.water
        ) / liquid_rate
        liquid_viscosity = (
            phase_rates.oil * phase_viscosities.oil + phase_rates.water * phase_viscosities.water
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


def compute_perforation_pressures(
    *, wellbore: WellBoreModel, **kwargs: typing.Any
) -> NumberArray[OneDimension]:
    """
    Dispatches to the `compute_perforation_pressures` of whichever module
    `wellbore.name` selects.

    :param wellbore: `WellBoreModel` naming a hydraulics correlation.
    :param kwargs: Forwarded to that correlation's `compute_perforation_pressures`.
    :returns: Pressure at each connection.
    :raises ValidationError: If `wellbore.name` isn't recognized.
    """
    if wellbore.name == "homogeneous":
        from bores.wells.hydraulics.homogeneous import (
            compute_perforation_pressures as compute,
        )

        return compute(model=wellbore.options, **kwargs)
    if wellbore.name in ("beggs_brill", "beggs_and_brill"):
        from bores.wells.hydraulics.beggs_and_brill import (
            compute_perforation_pressures as compute,
        )

        return compute(model=wellbore.options, **kwargs)
    if wellbore.name == "hagedorn_brown":
        from bores.wells.hydraulics.hagedorn_brown import (
            compute_perforation_pressures as compute,
        )

        return compute(model=wellbore.options, **kwargs)
    if wellbore.name == "gray":
        from bores.wells.hydraulics.gray import compute_perforation_pressures as compute

        return compute(model=wellbore.options, **kwargs)
    if wellbore.name == "woldesemayat_ghajar":
        from bores.wells.hydraulics.woldesemayat_ghajar import (
            compute_perforation_pressures as compute,
        )

        return compute(model=wellbore.options, **kwargs)
    raise ValidationError(f"Unknown `WellBoreModel` name: {wellbore.name!r}")


def compute_tubing_head_pressure(
    *, wellbore: WellBoreModel | None, **kwargs: typing.Any
) -> Number:
    """
    Dispatches to the `compute_tubing_head_pressure` of whichever module
    `wellbore.name` selects.

    Unlike the connection-distribution leg, this has no fallback for a
    missing `wellbore`; converting between BHP and THP is meaningless
    without a real hydraulics model or VFP table, so `ConnectionPressureMode`
    doesn't apply here. Also used directly by `wells.hydraulics.vfp.as_vfp_table`
    to sample a correlation while building a `VFPTable`.

    :param wellbore: `WellBoreModel` naming a hydraulics correlation.
        Required; `None` always raises.
    :param kwargs: Forwarded to that correlation's `compute_tubing_head_pressure`.
    :returns: Tubing head pressure.
    :raises ValidationError: If `wellbore` is `None`, or `wellbore.name`
        isn't recognized.
    """
    if wellbore is None:
        raise ValidationError(
            "Tubing head pressure requires a `WellBoreModel` or VFP table "
            "assigned to this well; THP control, a `THPLimit`, and THP "
            "reporting all need one. `ConnectionPressureMode.UNIFORM_BHP` "
            "only covers connection-to-connection pressure distribution, "
            "not the datum-to-surface relationship."
        )
    if wellbore.name == "homogeneous":
        from bores.wells.hydraulics.homogeneous import (
            compute_tubing_head_pressure as compute,
        )

        return compute(model=wellbore.options, **kwargs)
    if wellbore.name in ("beggs_brill", "beggs_and_brill"):
        from bores.wells.hydraulics.beggs_and_brill import (
            compute_tubing_head_pressure as compute,
        )

        return compute(model=wellbore.options, **kwargs)
    if wellbore.name == "hagedorn_brown":
        from bores.wells.hydraulics.hagedorn_brown import (
            compute_tubing_head_pressure as compute,
        )

        return compute(model=wellbore.options, **kwargs)
    if wellbore.name == "gray":
        from bores.wells.hydraulics.gray import compute_tubing_head_pressure as compute

        return compute(model=wellbore.options, **kwargs)
    if wellbore.name == "woldesemayat_ghajar":
        from bores.wells.hydraulics.woldesemayat_ghajar import (
            compute_tubing_head_pressure as compute,
        )

        return compute(model=wellbore.options, **kwargs)
    raise ValidationError(f"Unknown `WellBoreModel` name: {wellbore.name!r}")
