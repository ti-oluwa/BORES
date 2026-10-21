"""Connection pressures from the weight of the fluid between connections."""

import typing

import numba
import numpy as np
import numpy.typing as npt

from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.types import Integer, Number, NumberArray, OneDimension, UnitSystem
from bores.wells.hydraulics.base import (
    ConnectionFluids,
    PhaseValues,
    build_connection_fluids,
    compute_hydrostatic_pressure,
    compute_static_mixture_density,
    get_gravitational_acceleration,
)
from bores.wells.state import ConnectionSample
from bores.wells.workspace import ConnectionRates

__all__ = [
    "compute_hydrostatic_connection_pressures",
    "compute_hydrostatic_pressures_from_fluids",
    "get_pressure_per_density_and_depth",
    "walk_hydrostatic_pressures",
]


@numba.njit(cache=True)
def get_segment_density(
    first_beneath: Integer,
    static_density: NumberArray[OneDimension],
    mass_rate_beneath: NumberArray[OneDimension],
    volume_rate_beneath: NumberArray[OneDimension],
    static_rate_beneath: NumberArray[OneDimension],
) -> Number:
    """
    Computes the density of the fluid carried past connection `first_beneath` and every one
    deeper, with the connections ordered by depth.

    :param first_beneath: Index of the shallowest connection the segment carries the flow of.
    :param static_density: Saturation-weighted reservoir fluid density of each connection.
    :param mass_rate_beneath: Mass rate of this connection and every deeper one.
    :param volume_rate_beneath: Volume rate of this connection and every deeper one.
    :param static_rate_beneath: Sum of the static densities of this connection and every deeper
        one.
    :returns: The segment's density.
    """
    count = static_density.shape[0]
    if first_beneath >= count:
        return static_density[count - 1]
    if volume_rate_beneath[first_beneath] > 0.0:
        return mass_rate_beneath[first_beneath] / volume_rate_beneath[first_beneath]
    return static_rate_beneath[first_beneath] / (count - first_beneath)


@numba.njit(cache=True)
def walk_hydrostatic_pressures(
    reference_depth: Number,
    reference_pressure: Number,
    representative_depths: NumberArray[OneDimension],
    connection_fluids: ConnectionFluids,
    connection_oil_rates: NumberArray[OneDimension],
    connection_water_rates: NumberArray[OneDimension],
    connection_gas_rates: NumberArray[OneDimension],
    pressure_per_density_and_depth: Number,
    pressures: NumberArray[OneDimension],
) -> None:
    """
    Writes the flowing pressure at each connection into `pressures`, from the weight of the fluid
    between connections, working outward from `reference_depth`, with no friction.

    The scalar inputs are taken in the data type of `pressures`.

    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_pressure: Pressure at `reference_depth`.
    :param representative_depths: One vertical depth per connection.
    :param connection_fluids: Reservoir conditions at each connection.
    :param connection_oil_rates: Each connection's own oil rate, at reservoir conditions.
    :param connection_water_rates: Each connection's own water rate, at reservoir conditions.
    :param connection_gas_rates: Each connection's own gas rate, at reservoir conditions.
    :param pressure_per_density_and_depth: Pressure of a column of density one and length one,
        in the unit system of the depths and densities.
    :param pressures: Output array, one entry per connection.
    :raises ValueError: If a connection with no flow beneath it has saturations that sum to zero.
    """
    count = representative_depths.shape[0]
    if count == 0:
        return
    zero = pressures.dtype.type(0.0)
    scale = pressures.dtype.type(pressure_per_density_and_depth)
    order = np.argsort(representative_depths, kind="mergesort")

    depths = np.empty(count, dtype=representative_depths.dtype)
    static_density = np.empty(count, dtype=pressures.dtype)
    mass_rate = np.empty(count, dtype=pressures.dtype)
    volumetric_rate = np.empty(count, dtype=pressures.dtype)
    for k in range(count):
        i = order[k]
        depths[k] = representative_depths[i]
        oil_rate = abs(connection_oil_rates[i])
        water_rate = abs(connection_water_rates[i])
        gas_rate = abs(connection_gas_rates[i])
        mass_rate[k] = (
            oil_rate * connection_fluids.oil_densities[i]
            + water_rate * connection_fluids.water_densities[i]
            + gas_rate * connection_fluids.gas_densities[i]
        )
        volumetric_rate[k] = oil_rate + water_rate + gas_rate
        static_density[k] = compute_static_mixture_density(
            phase_saturations=PhaseValues(
                connection_fluids.oil_saturations[i],
                connection_fluids.water_saturations[i],
                connection_fluids.gas_saturations[i],
            ),
            phase_densities=PhaseValues(
                connection_fluids.oil_densities[i],
                connection_fluids.water_densities[i],
                connection_fluids.gas_densities[i],
            ),
        )

    mass_rate_beneath = np.empty(count, dtype=pressures.dtype)
    volume_rate_beneath = np.empty(count, dtype=pressures.dtype)
    static_rate_beneath = np.empty(count, dtype=pressures.dtype)
    mass_sum, volume_sum, static_sum = zero, zero, zero
    for k in range(count - 1, -1, -1):
        mass_sum = mass_sum + mass_rate[k]
        volume_sum = volume_sum + volumetric_rate[k]
        static_sum = static_sum + static_density[k]
        mass_rate_beneath[k] = mass_sum
        volume_rate_beneath[k] = volume_sum
        static_rate_beneath[k] = static_sum

    first_below = np.searchsorted(depths, representative_depths.dtype.type(reference_depth))

    pressure = pressures.dtype.type(reference_pressure)
    previous_depth = representative_depths.dtype.type(reference_depth)
    for k in range(first_below, count):
        pressure = pressure + (
            get_segment_density(
                k, static_density, mass_rate_beneath, volume_rate_beneath, static_rate_beneath
            )
            * (depths[k] - previous_depth)
            * scale
        )
        pressures[order[k]] = pressure
        previous_depth = depths[k]

    pressure = pressures.dtype.type(reference_pressure)
    previous_depth = representative_depths.dtype.type(reference_depth)
    for k in range(first_below - 1, -1, -1):
        pressure = pressure - (
            get_segment_density(
                k + 1, static_density, mass_rate_beneath, volume_rate_beneath, static_rate_beneath
            )
            * (previous_depth - depths[k])
            * scale
        )
        pressures[order[k]] = pressure
        previous_depth = depths[k]


def get_pressure_per_density_and_depth(unit_system: UnitSystem) -> Number:
    """
    Pressure of a column of density one and length one, in a unit system, as the constants are set
    now.

    :param unit_system: The unit system of the depths and densities.
    :returns: The pressure.
    """
    return compute_hydrostatic_pressure(
        density=1.0,
        gravitational_acceleration=get_gravitational_acceleration(unit_system),
        length=1.0,
        unit_system=unit_system,
    )


def compute_hydrostatic_pressures_from_fluids(
    reference_depth: Number,
    reference_pressure: Number,
    representative_depths: NumberArray[OneDimension],
    connection_fluids: ConnectionFluids,
    connection_rates: ConnectionRates,
    unit_system: UnitSystem,
    out: NumberArray[OneDimension] | None = None,
    dtype: npt.DTypeLike = None,
    pressure_per_density_and_depth: Number | None = None,
) -> NumberArray[OneDimension]:
    """
    Computes the flowing pressure at each connection from the weight of the fluid between
    connections, working outward from the well's reference depth, with no friction, from
    per-connection arrays.

    See `compute_hydrostatic_connection_pressures` for how the fluid in each segment is found.

    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_pressure: Pressure at `reference_depth`.
    :param representative_depths: One depth per connection.
    :param connection_fluids: Reservoir conditions at each connection.
    :param connection_rates: Each connection's own reservoir-condition rate of each phase.
    :param unit_system: The unit system the depths, densities and pressure are in.
    :param out: Optional preallocated output array, one entry per connection.
    :param dtype: Optional output array data type. Ignored if `out` is given. The data type of
        `representative_depths` if not given.
    :param pressure_per_density_and_depth: As `get_pressure_per_density_and_depth` gives it. Read
        from the constants if not given; pass it when calling repeatedly for the same well.
    :returns: Pressure at each connection.
    :raises ValidationError: If the per-connection inputs don't all have the same length, or `out`
        is the wrong length.
    """
    count = len(representative_depths)
    if not (
        len(connection_fluids.pressures)
        == len(connection_rates.oil)
        == len(connection_rates.water)
        == len(connection_rates.gas)
        == count
    ):
        raise ValidationError("All per-connection inputs must have the same length.")
    if out is not None and len(out) != count:
        raise ValidationError("If given, `out` must have one entry per connection.")

    if out is not None:
        pressures = out
    else:
        pressures = np.empty(
            count, dtype=np.dtype(dtype) if dtype is not None else representative_depths.dtype
        )
    walk_hydrostatic_pressures(
        reference_depth,
        reference_pressure,
        representative_depths,
        connection_fluids,
        connection_rates.oil,
        connection_rates.water,
        connection_rates.gas,
        (
            pressure_per_density_and_depth
            if pressure_per_density_and_depth is not None
            else get_pressure_per_density_and_depth(unit_system)
        ),
        pressures,
    )
    return pressures


def compute_hydrostatic_connection_pressures(
    *,
    reference_depth: Number,
    reference_pressure: Number,
    representative_depths: NumberArray[OneDimension],
    connection_samples: typing.Sequence[ConnectionSample],
    connection_rates: ConnectionRates,
    unit_system: UnitSystem,
    out: NumberArray[OneDimension] | None = None,
    dtype: npt.DTypeLike = None,
) -> NumberArray[OneDimension]:
    """
    Computes the flowing pressure at each connection from the weight of the fluid between
    connections, working outward from the well's reference depth, with no friction.

    The fluid in a segment is the commingled mixture of every connection beneath it, whether
    the well produces (it flows up past them) or injects (it flows down to them): the total mass
    rate of those connections over their total volume rate, at reservoir conditions. A
    segment with no flow beneath it holds the saturation-weighted reservoir fluid of the
    connections beneath it instead, or of the nearest connection if none is beneath it.
    Only vertical depth difference matters, so connection inclination plays no part.

    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_pressure: Pressure at `reference_depth`.
    :param representative_depths: One depth per connection, same order as `connection_samples`.
    :param connection_samples: Reservoir conditions at each connection.
    :param connection_rates: Each connection's own reservoir-condition rate of each phase.
        Same order as `connection_samples`.
    :param unit_system: The unit system the depths, densities and pressure are in.
    :param out: Optional preallocated output array, the length of `connection_samples`.
    :param dtype: Output array data type. Ignored if `out` is given.
    :returns: Pressure at each connection, same order as `connection_samples`.
    :raises ValidationError: If the lengths of `representative_depths`, `connection_samples`
        and `connection_rates` differ, or `out` is the wrong length.
    """
    count = len(connection_samples)
    if len(representative_depths) != count:
        raise ValidationError(
            "`representative_depths`, `connection_samples` and `connection_rates` must all "
            "have the same length."
        )
    if out is not None and len(out) != count:
        raise ValidationError("If given, `out` must have the same length as `connection_samples`.")

    if out is not None:
        array_dtype = out.dtype
    elif dtype is not None:
        array_dtype = np.dtype(dtype)
    else:
        array_dtype = get_dtype()
    return compute_hydrostatic_pressures_from_fluids(
        reference_depth,
        reference_pressure,
        representative_depths,
        build_connection_fluids(connection_samples=connection_samples, dtype=array_dtype),
        connection_rates,
        unit_system,
        out=out,
        dtype=array_dtype,
    )
