"""
Fluid properties against pressure for the tubing of a well.

A tubing head pressure is found by marching up the tubing in segments. The fluid in each segment
has to be described at that segment's own pressure, since gas expands and oil gives up dissolved
gas as the pressure falls. `TubingFluidTable` holds the properties against pressure, built once
from the PVT tables, and `get_tubing_fluid_sample` reads them at any pressure.
"""

import logging
import math
import typing

import numba
import numpy as np
import numpy.typing as npt

from bores.constants import c
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.types import (
    AnyDimension,
    Boolean,
    Integer,
    Number,
    NumberArray,
    OneDimension,
    PhaseValues,
    UnitSystem,
)
from bores.wells.hydraulics.base import (
    UNIT_SYSTEM_TAGS,
    UNIT_SYSTEMS_BY_TAG,
    compute_mixture_density,
    compute_mixture_velocity,
    compute_mixture_viscosity,
    compute_superficial_velocity,
    split_liquid_gas,
)

if typing.TYPE_CHECKING:
    from bores.blackoil.pvt.tables import PVTTable, PVTTables

__all__ = [
    "MARCH_ITERATIONS",
    "NoSlipMixture",
    "TubingFluidSample",
    "TubingFluidTable",
    "TubingState",
    "TwoPhaseFlow",
    "build_tubing_fluid_table",
    "compute_connection_surface_rates",
    "compute_no_slip_mixture",
    "compute_two_phase_flow",
    "evaluate_tubing_state",
    "get_stb_to_volume_factor",
    "get_tubing_fluid_sample",
]

logger = logging.getLogger(__name__)

MARCH_ITERATIONS = 3
"""
Times a segment's end pressure is refined when marching up the tubing. A fixed count keeps the
pressure a smooth function of the rates, which the rate iterations of a well rely on.
"""


class TubingFluidTable(typing.NamedTuple):
    """
    Fluid properties of a well's tubing against pressure.

    Every column is given at the same ascending pressures. Oil is described along its saturated
    curve, that is, oil that holds as much dissolved gas as the pressure allows. Oil that holds
    less than that, because the well's produced gas-oil ratio is lower than the saturated
    solution gas-oil ratio, is described by the same curve at its own bubble point pressure, and
    then carried to the actual pressure with the compressibility of the oil above its bubble point.
    The same applies above the highest bubble point pressure.
    """

    pressures: NumberArray[OneDimension]
    """Pressures the table is given at, ascending."""

    oil_formation_volume_factors: NumberArray[OneDimension]
    """Saturated oil formation volume factor at each pressure."""

    water_formation_volume_factors: NumberArray[OneDimension]
    """Water formation volume factor at each pressure."""

    gas_formation_volume_factors: NumberArray[OneDimension]
    """Gas formation volume factor at each pressure."""

    solution_gas_oil_ratios: NumberArray[OneDimension]
    """Saturated solution gas-oil ratio at each pressure, rising with pressure."""

    oil_densities: NumberArray[OneDimension]
    """Saturated oil density at each pressure."""

    water_densities: NumberArray[OneDimension]
    """Water density at each pressure."""

    gas_densities: NumberArray[OneDimension]
    """Gas density at each pressure."""

    oil_viscosities: NumberArray[OneDimension]
    """Saturated oil viscosity at each pressure."""

    water_viscosities: NumberArray[OneDimension]
    """Water viscosity at each pressure."""

    gas_viscosities: NumberArray[OneDimension]
    """Gas viscosity at each pressure."""

    gas_liquid_surface_tensions: NumberArray[OneDimension]
    """Gas-liquid surface tension at each pressure."""

    vaporized_oil_gas_ratios: NumberArray[OneDimension]
    """
    Saturated vaporized oil-gas ratio at each pressure, the oil the gas carries as vapor, in
    stock-tank barrels per standard cubic foot. Zero for dry gas.
    """

    oil_density_compressibilities: NumberArray[OneDimension]
    """
    How fast the density of oil rises with pressure above its bubble point, as a fraction of its
    density per unit of pressure, for oil that is saturated at each pressure.
    """

    oil_formation_volume_factor_compressibilities: NumberArray[OneDimension]
    """
    How fast the formation volume factor of oil falls with pressure above its bubble point, as a
    fraction of its value per unit of pressure, for oil that is saturated at each pressure.
    """

    oil_viscosity_pressure_coefficients: NumberArray[OneDimension]
    """
    How fast the viscosity of oil rises with pressure above its bubble point, as a fraction of its
    viscosity per unit of pressure, for oil that is saturated at each pressure.
    """

    highest_bubble_point_pressure: Number
    """The highest bubble point pressure that the saturated oil columns reach."""

    oil_surface_density: Number
    """Density of the oil at surface conditions."""

    gas_surface_density: Number
    """Density of the gas at surface conditions."""

    stb_to_volume_factor: Number
    """
    Volume of one stock-tank barrel of oil, in the volume unit of rates in the wellbore. Oil and
    water rates at the surface are in stock-tank barrels per day, and the rates in the wellbore
    are in volume per day, so a surface rate times a formation volume factor times this gives a
    wellbore rate. One in the unit systems where a stock-tank volume is a plain volume.
    """

    segment_count: Integer
    """Number of equal segments the tubing is divided into when marching up it."""

    unit_system_tag: Integer
    """Tag of the unit system of the table, in `UNIT_SYSTEM_TAGS`. Read it as `unit_system`."""

    @property
    def unit_system(self) -> UnitSystem:
        """The unit system the table is in."""
        return UNIT_SYSTEMS_BY_TAG[self.unit_system_tag]


class TubingFluidSample(typing.NamedTuple):
    """The fluids of a well's tubing at one pressure."""

    pressure: Number
    """Pressure the fluids are at."""

    formation_volume_factors: PhaseValues
    """Formation volume factor of each phase."""

    dissolved_gas_oil_ratio: Number
    """Gas-oil ratio the oil actually holds in solution at this pressure."""

    phase_densities: PhaseValues
    """Density of each phase."""

    phase_viscosities: PhaseValues
    """Viscosity of each phase."""

    gas_liquid_surface_tension: Number
    """Gas-liquid surface tension."""


def get_stb_to_volume_factor(unit_system: UnitSystem) -> Number:
    """
    The volume of one stock-tank barrel in the volume unit that surface rates are in.

    :param unit_system: The unit system.
    :returns: The factor, one where gas-oil ratios are volume per volume.
    """
    from bores.blackoil.pvt.data import get_stb_to_volume_factor as resolve_factor

    return resolve_factor(unit_system)


@numba.njit(cache=True)
def interpolate_column(
    axis: NumberArray[OneDimension], values: NumberArray[OneDimension], x: Number
) -> Number:
    """
    Reads a column at `x` by linear interpolation, holding the end values outside the axis.

    :param axis: Ascending axis.
    :param values: Column given at `axis`.
    :param x: Where to read the column.
    :returns: The interpolated value.
    """
    last = axis.shape[0] - 1
    if x <= axis[0]:
        return values[0]
    if x >= axis[last]:
        return values[last]
    lower = np.searchsorted(axis, x, side="right") - 1
    weight = (x - axis[lower]) / (axis[lower + 1] - axis[lower])
    return values[lower] + weight * (values[lower + 1] - values[lower])


@numba.njit(cache=True)
def find_bubble_point_pressure(
    pressures: NumberArray[OneDimension],
    solution_gas_oil_ratios: NumberArray[OneDimension],
    gas_oil_ratio: Number,
) -> Number:
    """
    Finds the pressure at which oil holding `gas_oil_ratio` in solution is just saturated.

    :param pressures: Ascending pressures.
    :param solution_gas_oil_ratios: Saturated solution gas-oil ratio at each pressure.
    :param gas_oil_ratio: The gas-oil ratio of the oil.
    :returns: The bubble point pressure, held to the ends of `pressures`.
    """
    last = pressures.shape[0] - 1
    if gas_oil_ratio <= solution_gas_oil_ratios[0]:
        return pressures[0]
    if gas_oil_ratio >= solution_gas_oil_ratios[last]:
        return pressures[last]
    upper = np.searchsorted(solution_gas_oil_ratios, gas_oil_ratio, side="left")
    lower = upper - 1
    span = solution_gas_oil_ratios[upper] - solution_gas_oil_ratios[lower]
    weight = (gas_oil_ratio - solution_gas_oil_ratios[lower]) / span
    return pressures[lower] + weight * (pressures[upper] - pressures[lower])


@numba.njit(cache=True)
def locate_in_axis(axis: NumberArray[OneDimension], x: Number) -> tuple[Integer, Number]:
    """
    Finds where `x` sits in an ascending axis.

    :param axis: Ascending axis.
    :param x: The value to place, held to the ends of `axis`.
    :returns: Index of the lower neighbour, and the fraction of the way to the next point.
    """
    last = axis.shape[0] - 1
    if x <= axis[0]:
        return 0, 0.0
    if x >= axis[last]:
        return last - 1, 1.0
    lower = np.searchsorted(axis, x, side="right") - 1
    return lower, (x - axis[lower]) / (axis[lower + 1] - axis[lower])


@numba.njit(cache=True)
def read_column(values: NumberArray[OneDimension], lower: Integer, fraction: Number) -> Number:
    """
    Reads a column at a position found by `locate_in_axis`.

    :param values: Column given at the axis the position was found in.
    :param lower: Index of the lower neighbour.
    :param fraction: Fraction of the way to the next point.
    :returns: The interpolated value.
    """
    return values[lower] + fraction * (values[lower + 1] - values[lower])


@numba.njit(cache=True)
def get_tubing_fluid_sample(
    table: TubingFluidTable, pressure: Number, produced_gas_oil_ratio: Number
) -> TubingFluidSample:
    """
    Reads the properties of fluids of a well's tubing at a given pressure.

    The oil holds the lower of the produced gas-oil ratio and the saturated solution gas-oil
    ratio at this pressure. Oil that is not saturated is read from the table at its own bubble
    point pressure, then carried up to this pressure with its compressibility above the bubble
    point.

    :param table: The well's `TubingFluidTable`.
    :param pressure: Pressure to read the fluids at.
    :param produced_gas_oil_ratio: The gas the oil carries, per stock-tank barrel. Infinite or
        `NaN` for oil that is saturated.
    :returns: The fluids at `pressure`.
    """
    lower, fraction = locate_in_axis(table.pressures, pressure)
    saturated_ratio = read_column(table.solution_gas_oil_ratios, lower, fraction)
    ratio = produced_gas_oil_ratio
    oil_lower, oil_fraction = lower, fraction
    excess_pressure = 0.0
    if math.isnan(ratio) or ratio >= saturated_ratio:
        dissolved_ratio = saturated_ratio
        if pressure > table.highest_bubble_point_pressure:
            excess_pressure = pressure - table.highest_bubble_point_pressure
            oil_lower, oil_fraction = locate_in_axis(
                table.pressures, table.highest_bubble_point_pressure
            )
    else:
        dissolved_ratio = max(ratio, 0.0)
        bubble_point_pressure = find_bubble_point_pressure(
            pressures=table.pressures,
            solution_gas_oil_ratios=table.solution_gas_oil_ratios,
            gas_oil_ratio=dissolved_ratio,
        )
        excess_pressure = max(pressure - bubble_point_pressure, 0.0)
        oil_lower, oil_fraction = locate_in_axis(table.pressures, bubble_point_pressure)

    oil_density = read_column(table.oil_densities, oil_lower, oil_fraction)
    oil_formation_volume_factor = read_column(
        table.oil_formation_volume_factors, oil_lower, oil_fraction
    )
    oil_viscosity = read_column(table.oil_viscosities, oil_lower, oil_fraction)
    if excess_pressure > 0.0:
        oil_density *= math.exp(
            read_column(table.oil_density_compressibilities, oil_lower, oil_fraction)
            * excess_pressure
        )
        oil_formation_volume_factor *= math.exp(
            -read_column(
                table.oil_formation_volume_factor_compressibilities, oil_lower, oil_fraction
            )
            * excess_pressure
        )
        oil_viscosity *= math.exp(
            read_column(table.oil_viscosity_pressure_coefficients, oil_lower, oil_fraction)
            * excess_pressure
        )
    return TubingFluidSample(
        pressure=pressure,
        formation_volume_factors=PhaseValues(
            oil=oil_formation_volume_factor,
            water=read_column(table.water_formation_volume_factors, lower, fraction),
            gas=read_column(table.gas_formation_volume_factors, lower, fraction),
        ),
        dissolved_gas_oil_ratio=dissolved_ratio,
        phase_densities=PhaseValues(
            oil=oil_density,
            water=read_column(table.water_densities, lower, fraction),
            gas=read_column(table.gas_densities, lower, fraction),
        ),
        phase_viscosities=PhaseValues(
            oil=oil_viscosity,
            water=read_column(table.water_viscosities, lower, fraction),
            gas=read_column(table.gas_viscosities, lower, fraction),
        ),
        gas_liquid_surface_tension=read_column(table.gas_liquid_surface_tensions, lower, fraction),
    )


class TubingState(typing.NamedTuple):
    """What a well's surface rates make at one pressure in its tubing."""

    sample: TubingFluidSample
    """The properties of each phase at the pressure."""

    phase_rates: PhaseValues
    """Volume rate of the oil, water and free gas at the pressure, per day."""


@numba.njit(cache=True)
def evaluate_tubing_state(
    table: TubingFluidTable,
    pressure: Number,
    surface_oil_rate: Number,
    surface_water_rate: Number,
    surface_gas_rate: Number,
) -> TubingState:
    """
    Finds the phases a well's surface rates make at one pressure in its tubing, and their volumes.

    The surface oil rate is the total, oil that flows as liquid plus the condensate the gas
    carries, and the surface gas rate is the total as well, free gas plus gas dissolved in the
    oil. At a pressure where both phases are present, the oil is saturated with gas and the gas
    with oil. When the oil can hold all of the gas, there is no free gas and the oil is not
    saturated. When the gas can hold all of the oil, there is no liquid oil and the gas is not
    saturated.

    :param table: The well's fluid properties against pressure.
    :param pressure: Pressure in the tubing.
    :param surface_oil_rate: Total oil rate at surface conditions, in stock-tank barrels per day.
    :param surface_water_rate: Water rate at surface conditions, in stock-tank barrels per day.
    :param surface_gas_rate: Total gas rate at surface conditions, in standard cubic feet per day.
    :returns: The properties and volumes of the phases at `pressure`.
    """
    lower, fraction = locate_in_axis(table.pressures, pressure)
    saturated_solution_ratio = read_column(table.solution_gas_oil_ratios, lower, fraction)
    saturated_vaporized_ratio = read_column(table.vaporized_oil_gas_ratios, lower, fraction)

    free_gas_rate = (surface_gas_rate - saturated_solution_ratio * surface_oil_rate) / (
        1.0 - saturated_solution_ratio * saturated_vaporized_ratio
    )
    liquid_oil_rate = surface_oil_rate - saturated_vaporized_ratio * free_gas_rate
    gas_density_override = math.nan
    if free_gas_rate < 0.0:
        free_gas_rate = 0.0
        liquid_oil_rate = surface_oil_rate
        sample = get_tubing_fluid_sample(
            table=table,
            pressure=pressure,
            produced_gas_oil_ratio=surface_gas_rate / surface_oil_rate,
        )
    else:
        sample = get_tubing_fluid_sample(
            table=table, pressure=pressure, produced_gas_oil_ratio=math.inf
        )
        if liquid_oil_rate < 0.0:
            liquid_oil_rate = 0.0
            free_gas_rate = surface_gas_rate
            if surface_gas_rate > 0.0:
                vaporized_ratio = surface_oil_rate / surface_gas_rate
                gas_density_override = (
                    table.gas_surface_density
                    + vaporized_ratio * table.oil_surface_density * table.stb_to_volume_factor
                ) / sample.formation_volume_factors.gas

    if not math.isnan(gas_density_override):
        sample = TubingFluidSample(
            pressure=sample.pressure,
            formation_volume_factors=sample.formation_volume_factors,
            dissolved_gas_oil_ratio=sample.dissolved_gas_oil_ratio,
            phase_densities=PhaseValues(
                oil=sample.phase_densities.oil,
                water=sample.phase_densities.water,
                gas=gas_density_override,
            ),
            phase_viscosities=sample.phase_viscosities,
            gas_liquid_surface_tension=sample.gas_liquid_surface_tension,
        )
    return TubingState(
        sample=sample,
        phase_rates=PhaseValues(
            oil=liquid_oil_rate * sample.formation_volume_factors.oil * table.stb_to_volume_factor,
            water=surface_water_rate
            * sample.formation_volume_factors.water
            * table.stb_to_volume_factor,
            gas=free_gas_rate * sample.formation_volume_factors.gas,
        ),
    )


class NoSlipMixture(typing.NamedTuple):
    """The no-slip mixture of a well's flow at one pressure."""

    density: Number
    """Rate-weighted density of the mixture."""

    viscosity: Number
    """Rate-weighted viscosity of the mixture."""

    velocity: Number
    """Superficial velocity of the whole mixture, per second."""


@numba.njit(cache=True)
def compute_no_slip_mixture(
    table: TubingFluidTable,
    pressure: Number,
    surface_oil_rate: Number,
    surface_water_rate: Number,
    surface_gas_rate: Number,
    tubing_inner_diameter: Number,
    rate_time_factor: Number,
) -> NoSlipMixture:
    """
    Finds the no-slip mixture a well's surface rates make at a pressure in its tubing.

    :param table: The well's fluid properties against pressure.
    :param pressure: Pressure in the tubing.
    :param surface_oil_rate: Total oil rate at surface conditions, in stock-tank barrels per day.
    :param surface_water_rate: Water rate at surface conditions, in stock-tank barrels per day.
    :param surface_gas_rate: Total gas rate at surface conditions, in standard cubic feet per day.
    :param tubing_inner_diameter: Tubing inner diameter.
    :param rate_time_factor: Converts a rate to a per-second rate.
    :returns: The mixture at `pressure`.
    """
    state = evaluate_tubing_state(
        table=table,
        pressure=pressure,
        surface_oil_rate=surface_oil_rate,
        surface_water_rate=surface_water_rate,
        surface_gas_rate=surface_gas_rate,
    )
    sample = state.sample
    rates = state.phase_rates
    return NoSlipMixture(
        density=compute_mixture_density(rates=rates, phase_densities=sample.phase_densities),
        viscosity=compute_mixture_viscosity(
            rates=rates, phase_viscosities=sample.phase_viscosities
        ),
        velocity=compute_mixture_velocity(
            rates=rates,
            tubing_inner_diameter=tubing_inner_diameter,
            rate_time_factor=rate_time_factor,
        ),
    )


class TwoPhaseFlow(typing.NamedTuple):
    """A well's flow at one pressure in its tubing, as a liquid stream and a gas stream."""

    superficial_liquid_velocity: Number
    """Liquid rate, oil and water together, divided by the cross-sectional area, per second."""

    superficial_gas_velocity: Number
    """Free gas rate divided by the cross-sectional area, per second."""

    liquid_density: Number
    """Rate-weighted density of the oil and water."""

    gas_density: Number
    """Density of the gas."""

    liquid_viscosity: Number
    """Rate-weighted viscosity of the oil and water."""

    gas_viscosity: Number
    """Viscosity of the gas."""

    gas_liquid_surface_tension: Number
    """Gas-liquid surface tension."""


@numba.njit(cache=True)
def compute_two_phase_flow(
    table: TubingFluidTable,
    pressure: Number,
    surface_oil_rate: Number,
    surface_water_rate: Number,
    surface_gas_rate: Number,
    tubing_inner_diameter: Number,
    rate_time_factor: Number,
) -> TwoPhaseFlow:
    """
    Finds the liquid and gas a well's surface rates make at a pressure in its tubing.

    :param table: The well's fluid properties against pressure.
    :param pressure: Pressure in the tubing.
    :param surface_oil_rate: Total oil rate at surface conditions, in stock-tank barrels per day.
    :param surface_water_rate: Water rate at surface conditions, in stock-tank barrels per day.
    :param surface_gas_rate: Total gas rate at surface conditions, in standard cubic feet per day.
    :param tubing_inner_diameter: Tubing inner diameter.
    :param rate_time_factor: Converts a rate to a per-second rate.
    :returns: The liquid and gas at `pressure`.
    """
    state = evaluate_tubing_state(
        table=table,
        pressure=pressure,
        surface_oil_rate=surface_oil_rate,
        surface_water_rate=surface_water_rate,
        surface_gas_rate=surface_gas_rate,
    )
    sample = state.sample
    rates = state.phase_rates
    (
        liquid_rate,
        gas_rate,
        liquid_density,
        gas_density,
        liquid_viscosity,
        gas_viscosity,
    ) = split_liquid_gas(
        rates=rates,
        phase_densities=sample.phase_densities,
        phase_viscosities=sample.phase_viscosities,
    )
    return TwoPhaseFlow(
        superficial_liquid_velocity=compute_superficial_velocity(
            rate=liquid_rate,
            tubing_inner_diameter=tubing_inner_diameter,
            rate_time_factor=rate_time_factor,
        ),
        superficial_gas_velocity=compute_superficial_velocity(
            rate=gas_rate,
            tubing_inner_diameter=tubing_inner_diameter,
            rate_time_factor=rate_time_factor,
        ),
        liquid_density=liquid_density,
        gas_density=gas_density,
        liquid_viscosity=liquid_viscosity,
        gas_viscosity=gas_viscosity,
        gas_liquid_surface_tension=sample.gas_liquid_surface_tension,
    )


@numba.njit(cache=True)
def compute_connection_surface_rates(
    connection_oil_rates: NumberArray[OneDimension],
    connection_water_rates: NumberArray[OneDimension],
    connection_gas_rates: NumberArray[OneDimension],
    oil_formation_volume_factors: NumberArray[OneDimension],
    water_formation_volume_factors: NumberArray[OneDimension],
    gas_formation_volume_factors: NumberArray[OneDimension],
    solution_gas_oil_ratios: NumberArray[OneDimension],
    vaporized_oil_gas_ratios: NumberArray[OneDimension],
    stb_to_volume_factor: Number,
    is_injector: Boolean,
) -> tuple[NumberArray[OneDimension], NumberArray[OneDimension], NumberArray[OneDimension]]:
    """
    Turns each connection's rates at reservoir conditions into rates at surface conditions.

    The oil and gas rates of a producing connection are totals. The gas is the gas that flows
    free plus the gas that its oil carries in solution, and the oil is the oil that flows as
    liquid plus the condensate that its gas carries.

    :param connection_oil_rates: Each connection's oil rate at reservoir conditions.
    :param connection_water_rates: Each connection's water rate at reservoir conditions.
    :param connection_gas_rates: Each connection's gas rate at reservoir conditions.
    :param oil_formation_volume_factors: Oil formation volume factor at each connection.
    :param water_formation_volume_factors: Water formation volume factor at each connection.
    :param gas_formation_volume_factors: Gas formation volume factor at each connection.
    :param solution_gas_oil_ratios: Gas the oil holds in solution at each connection, per
        stock-tank barrel.
    :param vaporized_oil_gas_ratios: Oil the gas carries as vapor at each connection, in
        stock-tank barrels per standard cubic foot.
    :param stb_to_volume_factor: `TubingFluidTable.stb_to_volume_factor`.
    :param is_injector: Whether the well is an injector, whose fluids carry nothing out.
    :returns: Oil and water rates in stock-tank barrels per day and gas rate in standard cubic
        feet per day, at surface conditions, one entry per connection.
    """
    count = connection_oil_rates.shape[0]
    oil = np.empty(count, dtype=connection_oil_rates.dtype)
    water = np.empty(count, dtype=connection_oil_rates.dtype)
    gas = np.empty(count, dtype=connection_oil_rates.dtype)
    for idx in range(count):
        liquid_oil = connection_oil_rates[idx] / (
            oil_formation_volume_factors[idx] * stb_to_volume_factor
        )
        free_gas = connection_gas_rates[idx] / gas_formation_volume_factors[idx]
        oil[idx] = liquid_oil
        gas[idx] = free_gas
        water[idx] = connection_water_rates[idx] / (
            water_formation_volume_factors[idx] * stb_to_volume_factor
        )
        if not is_injector:
            oil[idx] += free_gas * vaporized_oil_gas_ratios[idx]
            gas[idx] += liquid_oil * solution_gas_oil_ratios[idx]
    return oil, water, gas  # type: ignore[return-value]


def get_pressure_axis(
    *, min_pressure: Number, max_pressure: Number, point_count: Integer, dtype: npt.DTypeLike
) -> NumberArray[OneDimension]:
    """
    Builds an ascending pressure axis, closer together at low pressure where gas changes fastest.

    :param min_pressure: Lowest pressure.
    :param max_pressure: Highest pressure.
    :param point_count: Number of pressures.
    :param dtype: Data type of the axis.
    :returns: The pressures.
    """
    fractions = np.linspace(0.0, 1.0, point_count) ** 2
    return typing.cast(
        NumberArray[OneDimension],
        (min_pressure + (max_pressure - min_pressure) * fractions).astype(dtype, copy=False),
    )


def build_tubing_fluid_table(
    *,
    pvt_tables: "PVTTables",
    temperature: Number,
    gas_liquid_surface_tension: Number,
    pressure_range: tuple[Number, Number] | None = None,
    point_count: Integer = 200,
    segment_count: Integer | None = None,
    salinity: Number | None = None,
    unit_system: UnitSystem | None = None,
    dtype: npt.DTypeLike | None = None,
) -> TubingFluidTable:
    """
    Builds the fluid table of a well's tubing from PVT tables.

    The table depends only on the PVT tables and the temperature, so one table serves a well for
    a whole run. The whole tubing is taken to be at `temperature`. Gas that carries oil as vapor is
    handled through the vaporized oil-gas ratio of the PVT tables.

    :param pvt_tables: PVT tables of the region around the well's reference depth, with oil,
        gas and water.
    :param temperature: Temperature of the fluids in the tubing, in the unit system of the table.
    :param gas_liquid_surface_tension: Gas-liquid surface tension, in dyne/cm in every unit
        system.
    :param pressure_range: Lowest and highest pressure of the table, in the unit system of the
        table. The range of the oil and gas PVT tables if not given. It should reach from
        surface pressure to the highest pressure at the bottom of the tubing.
    :param point_count: Number of pressures. Closely spaced at low pressure.
    :param segment_count: Number of equal segments to march up the tubing in.
        `c.TUBING_SEGMENT_COUNT` if not given.
    :param salinity: Salinity of the water, in ppm. The default of the water table if not given.
    :param unit_system: Unit system to build the table in. That of `pvt_tables` if not given.
    :param dtype: Data type of the columns. `bores.precision.get_dtype()` if not given.
    :returns: The table.
    :raises ValidationError: If the PVT tables lack a phase, or the pressure range is empty.
    """
    target = unit_system if unit_system is not None else pvt_tables.unit_system
    tables = pvt_tables.convert(target) if target != pvt_tables.unit_system else pvt_tables
    oil_table, gas_table, water_table = tables.oil, tables.gas, tables.water
    if oil_table is None or gas_table is None or water_table is None:
        raise ValidationError(
            "`pvt_tables` needs oil, gas and water tables to build tubing fluids."
        )

    dataset = tables.dataset
    assert dataset.oil is not None and dataset.gas is not None
    if pressure_range is None:
        min_pressure = max(dataset.oil.pressures.min(), dataset.gas.pressures.min())
        max_pressure = min(dataset.oil.pressures.max(), dataset.gas.pressures.max())
    else:
        min_pressure, max_pressure = pressure_range[0], pressure_range[1]

    if not max_pressure > min_pressure:
        raise ValidationError(f"The pressure range ({min_pressure}, {max_pressure}) is empty.")
    if point_count < 3:
        raise ValidationError("`point_count` must be at least 3.")
    if segment_count is not None and segment_count < 1:
        raise ValidationError("`segment_count` must be at least 1.")

    resolved_dtype = np.dtype(dtype) if dtype is not None else np.dtype(get_dtype())
    pressures = get_pressure_axis(
        min_pressure=min_pressure,
        max_pressure=max_pressure,
        point_count=point_count,
        dtype=np.float64,
    )

    bubble_points = np.asarray(dataset.oil.bubble_point_pressures, dtype=np.float64)
    highest_bubble_point = bubble_points.max()
    oil_pressures = np.minimum(pressures, highest_bubble_point)
    if highest_bubble_point < max_pressure:
        logger.info(
            "The oil is held at its properties at its highest bubble point pressure %s above "
            "that pressure.",
            highest_bubble_point,
        )

    def read_table(
        table: "PVTTable", name: str, *args: typing.Any, **kwargs: typing.Any
    ) -> NumberArray[AnyDimension]:
        result = getattr(table, name)(*args, **kwargs)
        if result is None:
            raise ValidationError(f"The PVT table of {table.phase.value} has no {name}.")
        return np.asarray(result, dtype=np.float64)

    oil_kwargs = {"bubble_point_pressure": oil_pressures}
    water_kwargs = {} if salinity is None else {"salinity": salinity}
    columns = {
        "oil_formation_volume_factors": read_table(
            oil_table, "formation_volume_factor", oil_pressures, temperature, **oil_kwargs
        ),
        "water_formation_volume_factors": read_table(
            water_table, "formation_volume_factor", pressures, temperature, **water_kwargs
        ),
        "gas_formation_volume_factors": read_table(
            gas_table, "formation_volume_factor", pressures, temperature
        ),
        "solution_gas_oil_ratios": read_table(
            oil_table, "solution_gas_to_oil_ratio", oil_pressures, temperature, **oil_kwargs
        ),
        "oil_densities": read_table(
            oil_table, "density", oil_pressures, temperature, **oil_kwargs
        ),
        "water_densities": read_table(
            water_table, "density", pressures, temperature, **water_kwargs
        ),
        "gas_densities": read_table(gas_table, "density", pressures, temperature),
        "oil_viscosities": read_table(
            oil_table, "viscosity", oil_pressures, temperature, **oil_kwargs
        ),
        "water_viscosities": read_table(
            water_table, "viscosity", pressures, temperature, **water_kwargs
        ),
        "gas_viscosities": read_table(gas_table, "viscosity", pressures, temperature),
    }
    vaporized_ratios = gas_table.vaporized_oil_gas_ratio(pressures, temperature)
    columns["vaporized_oil_gas_ratios"] = (
        np.zeros_like(pressures)
        if vaporized_ratios is None
        else np.maximum(np.asarray(vaporized_ratios, dtype=np.float64), 0.0)
    )
    oil_surface_density = oil_table.stock_tank_oil_density
    gas_surface_density = oil_table.stock_tank_gas_density
    if oil_surface_density is None or gas_surface_density is None:
        raise ValidationError(
            "The PVT tables need stock-tank oil and gas densities to build tubing fluids."
        )
    solution_ratios = np.maximum.accumulate(columns["solution_gas_oil_ratios"])
    columns["solution_gas_oil_ratios"] = solution_ratios

    # How the oil changes above its bubble point, averaged over a wide step above each node. The
    # undersaturated curves of the PVT tables are not straight just above the bubble point, so a
    # narrow step would give a compressibility that is too high.
    step = 0.1 * (max_pressure - min_pressure)
    undersaturated_kwargs = {
        "solution_gor": solution_ratios,
        "bubble_point_pressure": oil_pressures,
    }
    stepped_pressures = oil_pressures + step
    stepped_density = read_table(
        oil_table, "density", stepped_pressures, temperature, **undersaturated_kwargs
    )
    stepped_formation_volume_factor = read_table(
        oil_table,
        "formation_volume_factor",
        stepped_pressures,
        temperature,
        **undersaturated_kwargs,
    )
    stepped_viscosity = read_table(
        oil_table, "viscosity", stepped_pressures, temperature, **undersaturated_kwargs
    )
    columns["oil_density_compressibilities"] = (
        np.log(stepped_density / columns["oil_densities"]) / step
    )
    columns["oil_formation_volume_factor_compressibilities"] = (
        -np.log(stepped_formation_volume_factor / columns["oil_formation_volume_factors"]) / step
    )
    columns["oil_viscosity_pressure_coefficients"] = (
        np.log(stepped_viscosity / columns["oil_viscosities"]) / step
    )

    for name, column in columns.items():
        if not np.all(np.isfinite(column)):
            raise ValidationError(f"The tubing fluid column {name!r} has non-finite values.")

    return TubingFluidTable(
        pressures=typing.cast(NumberArray[OneDimension], pressures.astype(resolved_dtype)),
        gas_liquid_surface_tensions=typing.cast(
            NumberArray[OneDimension],
            np.full(point_count, gas_liquid_surface_tension).astype(resolved_dtype),
        ),
        stb_to_volume_factor=get_stb_to_volume_factor(target),
        oil_surface_density=oil_surface_density,
        gas_surface_density=gas_surface_density,
        highest_bubble_point_pressure=min(highest_bubble_point, max_pressure),
        segment_count=segment_count if segment_count is not None else int(c.TUBING_SEGMENT_COUNT),
        unit_system_tag=UNIT_SYSTEM_TAGS[target],
        **{
            name: typing.cast(NumberArray[OneDimension], column.astype(resolved_dtype))
            for name, column in columns.items()
        },
    )
