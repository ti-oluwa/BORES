"""Connection pressures from the weight of the fluid between connections."""

import typing

import numpy as np
import numpy.typing as npt

from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.types import Integer, Number, NumberArray, OneDimension, UnitSystem
from bores.wells.hydraulics.base import (
    compute_hydrostatic_pressure,
    compute_static_mixture_density,
    get_gravitational_acceleration,
)
from bores.wells.state import ConnectionSample
from bores.wells.workspace import ConnectionRates

__all__ = ["compute_hydrostatic_connection_pressures"]


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
    if not (
        len(representative_depths)
        == len(connection_rates.oil)
        == len(connection_rates.water)
        == len(connection_rates.gas)
        == count
    ):
        raise ValidationError(
            "`representative_depths`, `connection_samples` and `connection_rates` must all "
            "have the same length."
        )
    if out is not None and len(out) != count:
        raise ValidationError("If given, `out` must have the same length as `connection_samples`.")

    pressures = out if out is not None else np.empty(count, dtype=dtype or get_dtype())
    if count == 0:
        return pressures

    working_dtype = pressures.dtype
    order = np.argsort(representative_depths, kind="stable")
    depths = representative_depths[order]
    samples = [connection_samples[i] for i in order]
    oil = np.abs(connection_rates.oil[order])
    water = np.abs(connection_rates.water[order])
    gas = np.abs(connection_rates.gas[order])

    mass = (
        oil * np.fromiter((sample.phase_densities.oil for sample in samples), working_dtype, count)
        + water
        * np.fromiter((sample.phase_densities.water for sample in samples), working_dtype, count)
        + gas
        * np.fromiter((sample.phase_densities.gas for sample in samples), working_dtype, count)
    )
    volume = oil + water + gas
    static = np.fromiter(
        (
            compute_static_mixture_density(
                phase_saturations=sample.phase_saturations, phase_densities=sample.phase_densities
            )
            for sample in samples
        ),
        working_dtype,
        count,
    )
    mass_beneath = np.cumsum(mass[::-1])[::-1]
    volume_beneath = np.cumsum(volume[::-1])[::-1]
    static_beneath = np.cumsum(static[::-1])[::-1]

    def get_segment_density(first_beneath: Integer) -> Number:
        """Compute the density of the fluid carried past connection `first_beneath` and every one deeper."""
        if first_beneath >= count:
            return static[-1]
        if volume_beneath[first_beneath] > 0.0:
            return mass_beneath[first_beneath] / volume_beneath[first_beneath]
        return static_beneath[first_beneath] / (count - first_beneath)

    pressure_per_density_and_depth = compute_hydrostatic_pressure(
        density=1.0,
        gravitational_acceleration=get_gravitational_acceleration(unit_system),
        length=1.0,
        unit_system=unit_system,
    )
    first_below = int(np.searchsorted(depths, reference_depth, side="left"))

    pressure = reference_pressure
    previous_depth = reference_depth
    for k in range(first_below, count):
        pressure += (
            get_segment_density(k) * (depths[k] - previous_depth) * pressure_per_density_and_depth
        )
        pressures[order[k]] = pressure
        previous_depth = depths[k]

    pressure = reference_pressure
    previous_depth = reference_depth
    for k in range(first_below - 1, -1, -1):
        pressure -= (
            get_segment_density(k + 1)
            * (previous_depth - depths[k])
            * pressure_per_density_and_depth
        )
        pressures[order[k]] = pressure
        previous_depth = depths[k]
    return pressures
