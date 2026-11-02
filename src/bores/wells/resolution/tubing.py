"""
What relates a well's bottom-hole pressure to its tubing head pressure, as the resolver sees it.

A well's tubing performance is either a hydraulics correlation (`WellBoreModel`) or the deck's
vertical flow performance tables (`VFPTables`). `WellTubing` is the single object the resolver
passes around for one well, and `compute_tubing_head_pressure` is the single place that decides
which of the two answers a THP question.
"""

import math
import typing

import numba
import numpy as np

from bores.errors import ValidationError
from bores.types import Boolean, Integer, Number, NumberArray, OneDimension, PhaseValues
from bores.wells.base import WellType
from bores.wells.compile import CompiledWellSystem, WellKind
from bores.wells.hydraulics.base import (
    SurfaceSample,
    WellBoreModel,
    compute_hydrostatic_pressure,
    get_gravitational_acceleration,
)
from bores.wells.hydraulics.dispatch import (
    compute_tubing_head_pressure as compute_hydraulic_tubing_head_pressure,
)
from bores.wells.hydraulics.dispatch import (
    compute_tubing_head_pressure_from_table as compute_hydraulic_tubing_head_pressure_from_table,
)
from bores.wells.hydraulics.vfp import VFPTable, VFPTables
from bores.wells.workspace import ConnectionRates, PerforationWorkspace

__all__ = [
    "FlowingFluid",
    "WellTubing",
    "attach_flowing_fluid",
    "compute_flowing_density",
    "compute_tubing_head_pressure",
    "resolve_well_tubing",
]


class FlowingFluid(typing.NamedTuple):
    """What a well's connections hold, characterised well enough to weigh a column of what it flows."""

    oil_densities: NumberArray[OneDimension]
    """Reservoir-condition oil density at each connection."""

    water_densities: NumberArray[OneDimension]
    """Reservoir-condition water density at each connection."""

    gas_densities: NumberArray[OneDimension]
    """Reservoir-condition gas density at each connection."""

    zero_flow_density: Number
    """
    Density of the column when the well flows nothing: each flowing phase's density, weighted by
    that phase's share of the well's total deliverability.
    """


class WellTubing(typing.NamedTuple):
    """The tubing performance of one well, resolved for the current moment of a run."""

    wellbore: WellBoreModel | None
    """Hydraulics correlation. Also distributes pressure across the well's connections."""

    vfp_table: VFPTable | None
    """The VFP table the well's control currently selects."""

    artificial_lift_quantity: Number
    """The artificial lift quantity the well currently operates at."""

    flowing_fluid: FlowingFluid | None = None
    """
    The fluid the well flows. Needed to move a VFP table's BHP between its datum depth
    and the well's reference depth, when they differ.
    """

    @property
    def uses_tubing_fluid_table(self) -> bool:
        """Whether the tubing head pressure is found by marching up the tubing with fluid tables."""
        return self.wellbore is not None and self.wellbore.tubing_fluid_table is not None

    @property
    def needs_surface_sample(self) -> bool:
        """Whether the tubing head pressure needs the surface fluid properties of a sample."""
        return (
            self.vfp_table is None
            and self.wellbore is not None
            and self.wellbore.tubing_fluid_table is None
        )

    @property
    def can_evaluate_thp(self) -> bool:
        """Whether a THP can be found from a BHP, by a correlation or a VFP table."""
        return self.wellbore is not None or self.vfp_table is not None


def resolve_well_tubing(
    *,
    tubing_performance: WellBoreModel | VFPTables | None,
    compiled_system: CompiledWellSystem,
    well_row: Integer,
) -> WellTubing:
    """
    Picks a well's tubing performance, reading its VFP table number and ALQ from the compiled
    controls so a change made by a schedule action takes effect on the next solve.

    :param tubing_performance: A correlation, a collection of VFP tables, or `None`.
    :param compiled_system: The compiled well system holding the well's controls.
    :param well_row: The well's row.
    :returns: The well's `WellTubing`. A collection of tables yields no table for a well that
        has no table number.
    :raises ValidationError: If the well's table number is not in the collection.
    """
    controls = compiled_system.controls
    artificial_lift_quantity = controls.artificial_lift_quantities[well_row]
    if isinstance(tubing_performance, VFPTables):
        table_number = controls.get_vfp_table_number(well_row=well_row)
        if table_number is None:
            return WellTubing(None, None, artificial_lift_quantity)

        well_type = (
            WellType.INJECTOR
            if controls.well_kinds[well_row] == WellKind.INJECTOR
            else WellType.PRODUCER
        )
        return WellTubing(
            None,
            tubing_performance.table(table_number, well_type=well_type),
            artificial_lift_quantity,
        )
    return WellTubing(tubing_performance, None, artificial_lift_quantity)


def attach_flowing_fluid(
    tubing: WellTubing,
    workspace: PerforationWorkspace,
    relevant_phases: PhaseValues,
) -> WellTubing:
    """
    Records the fluids at a well's connections, for a well whose tubing is a VFP table or has
    tubing fluid tables.

    A well with a correlation and no tubing fluid tables is returned unchanged, since the
    correlation works from the connection samples itself.

    :param tubing: The well's tubing performance.
    :param workspace: The well's perforation workspace.
    :param relevant_phases: Mask of the phases the well flows: all three for a producer, only
        the injected phase for an injector.
    :returns: `tubing`, with its `flowing_fluid` set if it has a VFP table or tubing fluids.
    """
    if tubing.vfp_table is None and not tubing.uses_tubing_fluid_table:
        return tubing

    fluids = workspace.connection_fluids
    oil_flows = relevant_phases.oil > 0.0
    water_flows = relevant_phases.water > 0.0
    gas_flows = relevant_phases.gas > 0.0
    zero_flow_density = compute_zero_flow_density(
        well_indices=workspace.well_indices,
        oil_mobilities=fluids.oil_mobilities,
        water_mobilities=fluids.water_mobilities,
        gas_mobilities=fluids.gas_mobilities,
        oil_densities=fluids.oil_densities,
        water_densities=fluids.water_densities,
        gas_densities=fluids.gas_densities,
        oil_flows=oil_flows,
        water_flows=water_flows,
        gas_flows=gas_flows,
    )
    return tubing._replace(
        flowing_fluid=FlowingFluid(
            oil_densities=fluids.oil_densities,
            water_densities=fluids.water_densities,
            gas_densities=fluids.gas_densities,
            zero_flow_density=zero_flow_density,
        )
    )


@numba.njit(cache=True)
def compute_zero_flow_density(
    well_indices: NumberArray[OneDimension],
    oil_mobilities: NumberArray[OneDimension],
    water_mobilities: NumberArray[OneDimension],
    gas_mobilities: NumberArray[OneDimension],
    oil_densities: NumberArray[OneDimension],
    water_densities: NumberArray[OneDimension],
    gas_densities: NumberArray[OneDimension],
    oil_flows: Boolean,
    water_flows: Boolean,
    gas_flows: Boolean,
) -> Number:
    oil_weights = well_indices * oil_mobilities
    water_weights = well_indices * water_mobilities
    gas_weights = well_indices * gas_mobilities

    oil_total = oil_weights.sum()
    water_total = water_weights.sum()
    gas_total = gas_weights.sum()

    oil_average = (
        np.dot(oil_weights, oil_densities) / oil_total if oil_total > 0.0 else oil_densities.mean()
    )
    water_average = (
        np.dot(water_weights, water_densities) / water_total
        if water_total > 0.0
        else water_densities.mean()
    )
    gas_average = (
        np.dot(gas_weights, gas_densities) / gas_total if gas_total > 0.0 else gas_densities.mean()
    )

    oil_deliverability = oil_total if oil_flows else 0.0
    water_deliverability = water_total if water_flows else 0.0
    gas_deliverability = gas_total if gas_flows else 0.0
    total_deliverability = oil_deliverability + water_deliverability + gas_deliverability

    if total_deliverability > 0.0:
        oil_fraction = oil_deliverability / total_deliverability
        water_fraction = water_deliverability / total_deliverability
        gas_fraction = gas_deliverability / total_deliverability
    else:
        flowing = int(oil_flows) + int(water_flows) + int(gas_flows)
        oil_fraction = 1.0 / flowing if oil_flows else 0.0
        water_fraction = 1.0 / flowing if water_flows else 0.0
        gas_fraction = 1.0 / flowing if gas_flows else 0.0

    return oil_fraction * oil_average + water_fraction * water_average + gas_fraction * gas_average


@numba.njit(cache=True)
def compute_flowing_density(
    flowing_fluid: FlowingFluid, connection_rates: ConnectionRates
) -> Number:
    """
    Computes the density of the mixture a well flows as its total mass rate over its total volume rate,
    both at reservoir conditions, summed over every phase at every connection.

    A well flowing nothing is given `flowing_fluid.zero_flow_density`.

    :param flowing_fluid: What the well's connections hold.
    :param connection_rates: Reservoir-condition rate of each phase at each connection.
    :returns: The mixture density.
    """
    oil_rate, water_rate, gas_rate = (np.abs(rates) for rates in connection_rates)
    volumetric_rate = oil_rate.sum() + water_rate.sum() + gas_rate.sum()
    if volumetric_rate <= 0.0:
        return flowing_fluid.zero_flow_density

    mass_rate = (
        np.dot(oil_rate, flowing_fluid.oil_densities)
        + np.dot(water_rate, flowing_fluid.water_densities)
        + np.dot(gas_rate, flowing_fluid.gas_densities)
    )
    return mass_rate / volumetric_rate


def compute_datum_pressure_shift(
    *,
    table: VFPTable,
    reference_depth: Number,
    flowing_fluid: FlowingFluid | None = None,
    connection_rates: ConnectionRates | None = None,
) -> Number:
    """
    Computes the pressure to add to the BHP at the well's reference depth to get the BHP at a VFP
    table's datum depth.

    The fluid column between the two depths is the flowing mixture, with no friction, so the
    shift is its weight. It is positive when the datum is deeper than the reference depth, negative
    when it is shallower.

    :param table: The VFP table.
    :param reference_depth: The well's reference depth.
    :param flowing_fluid: What the well's connections hold. Not needed if the depths match.
    :param connection_rates: Reservoir-condition phase rates at each connection, at the BHP
        being evaluated. Not needed if the depths match.
    :returns: The pressure shift, in the table's unit system.
    :raises ValidationError: If the depths differ and `flowing_fluid` or `connection_rates` is
        `None`.
    """
    depth_difference = table.datum_depth - reference_depth
    if math.isclose(depth_difference, 0.0, abs_tol=1e-6):
        return 0.0

    if flowing_fluid is None or connection_rates is None:
        raise ValidationError(
            f"VFP table {table.table_number} refers to a datum depth of {table.datum_depth}, but "
            f"this well's reference depth is {reference_depth}, and the fluid it flows isn't "
            "known at this point, so the table's BHP can't be moved between the two depths."
        )

    unit_system = table.unit_system
    return compute_hydrostatic_pressure(
        density=compute_flowing_density(
            flowing_fluid=flowing_fluid, connection_rates=connection_rates
        ),
        gravitational_acceleration=get_gravitational_acceleration(unit_system),
        length=depth_difference,
        unit_system=unit_system,
    )


def compute_tubing_head_pressure(
    *,
    tubing: WellTubing,
    reference_depth: Number,
    reference_pressure: Number,
    rates: PhaseValues,
    surface_rates: PhaseValues,
    surface_sample: SurfaceSample | None,
    is_injector: Boolean,
    connection_rates: ConnectionRates | None = None,
) -> Number:
    """
    Finds the tubing head pressure for a bottom-hole pressure and the rates it produces.

    A VFP table is used if the well has one, otherwise its hydraulics correlation. A table's
    BHP is at its own datum depth, so if that differs from `reference_depth` the BHP is first
    moved to the datum by the weight of the flowing fluid column between them.

    :param tubing: The well's tubing performance.
    :param reference_depth: Depth the bottom-hole pressure refers to.
    :param reference_pressure: The bottom-hole pressure.
    :param rates: Reservoir-condition phase rates (used by a correlation).
    :param surface_rates: Surface-condition phase rates (used by a VFP table).
    :param surface_sample: Surface fluid properties. Only a correlation needs them.
    :param is_injector: Whether the well is an injector.
    :param connection_rates: Reservoir-condition phase rates at each connection, at
        `reference_pressure`. Needed only for a VFP table whose datum depth differs from
        `reference_depth`, to weigh the fluid column between them.
    :returns: The tubing head pressure.
    :raises ValidationError: If the well has no tubing performance, a correlation is needed but
        `surface_sample` is `None`, or a VFP table's datum depth differs from
        `reference_depth` and the well's `flowing_fluid` or `connection_rates` isn't given.
    """
    if tubing.vfp_table is not None:
        table = tubing.vfp_table
        datum_pressure = reference_pressure + compute_datum_pressure_shift(
            table=table,
            reference_depth=reference_depth,
            flowing_fluid=tubing.flowing_fluid,
            connection_rates=connection_rates,
        )
        lookup = table.get_lookup_variables(
            oil_rate=surface_rates.oil,
            water_rate=surface_rates.water,
            gas_rate=surface_rates.gas,
        )
        return table.get_thp(
            flow_rate=lookup.flow_rate,
            bhp=datum_pressure,
            water_fraction=lookup.water_fraction,
            gas_fraction=lookup.gas_fraction,
            artificial_lift_quantity=tubing.artificial_lift_quantity,
        )

    if tubing.wellbore is None:
        raise ValidationError(
            "This well has no `WellBoreModel` or VFP table to find a tubing head pressure with."
        )
    if tubing.uses_tubing_fluid_table:
        flowing_fluid = tubing.flowing_fluid
        static_density = (
            flowing_fluid.zero_flow_density
            if flowing_fluid is not None
            else (surface_sample.density if surface_sample is not None else math.nan)
        )
        return compute_hydraulic_tubing_head_pressure_from_table(
            wellbore=tubing.wellbore,
            reference_depth=reference_depth,
            reference_pressure=reference_pressure,
            surface_rates=surface_rates,
            is_injector=is_injector,
            static_density=static_density,
        )

    if surface_sample is None:
        raise ValidationError(
            "A hydraulics correlation needs `surface_sample` to find a tubing head pressure."
        )
    return compute_hydraulic_tubing_head_pressure(
        wellbore=tubing.wellbore,
        reference_depth=reference_depth,
        reference_pressure=reference_pressure,
        rates=rates,
        surface_sample=surface_sample,
        is_injector=is_injector,
    )
