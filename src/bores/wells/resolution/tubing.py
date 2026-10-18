"""
What relates a well's bottom-hole pressure to its tubing head pressure, as the resolver sees it.

A well's tubing performance is either a hydraulics correlation (`WellBoreModel`) or the deck's
vertical flow performance tables (`VFPTables`). `WellTubing` is the single object the resolver
passes around for one well, and `compute_tubing_head_pressure` is the single place that decides
which of the two answers a THP question.
"""

import math
import typing

import numpy as np

from bores.constants import c, get_conversion_factors
from bores.errors import ValidationError
from bores.types import Integer, Number, UnitSystem
from bores.wells.base import WellType
from bores.wells.compile import CompiledWellSystem, WellKind
from bores.wells.hydraulics.base import (
    PhaseValues,
    SurfaceFluidProperties,
    WellBoreModel,
    compute_hydrostatic_pressure,
)
from bores.wells.hydraulics.base import compute_tubing_head_pressure as compute_correlation_thp
from bores.wells.hydraulics.vfp import VFPTable, VFPTables
from bores.wells.state import ConnectionSample
from bores.wells.workspace import PerforationWorkspace

__all__ = [
    "FlowingFluid",
    "WellTubing",
    "attach_flowing_fluid",
    "compute_flowing_density",
    "compute_tubing_head_pressure",
    "resolve_well_tubing",
]


class FlowingFluid(typing.NamedTuple):
    """The fluid a well flows, characterised well enough to find the weight of a fluid column."""

    phase_densities: PhaseValues
    """
    Reservoir-condition density of each phase, averaged over the well's connections
    in proportion to how much of that phase each connection can deliver.
    """

    fallback_fractions: PhaseValues
    """
    Volume fraction of each phase, used for the column's density when the well flows
    nothing. Proportional to each flowing phase's total deliverability.
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
    artificial_lift_quantity = float(controls.artificial_lift_quantities[well_row])
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
    *,
    tubing: WellTubing,
    workspace: PerforationWorkspace,
    connection_samples: typing.Sequence[ConnectionSample],
    relevant_phases: PhaseValues,
) -> WellTubing:
    """
    Characterises the fluid a well flows, for a well whose tubing is a VFP table.

    Each phase's density is the average over the connections, weighted by the connection's
    well index times that phase's mobility, which is each connection's share of that phase's
    deliverability. A well with a correlation is returned unchanged, since the correlation
    works from the connection samples itself.

    :param tubing: The well's tubing performance.
    :param workspace: The well's perforation workspace.
    :param connection_samples: The well's connection samples, in the workspace's order.
    :param relevant_phases: Mask of the phases the well flows: all three for a producer, only
        the injected phase for an injector.
    :returns: `tubing`, with its `flowing_fluid` set if it has a VFP table.
    """
    if tubing.vfp_table is None:
        return tubing

    mobilities = (workspace.oil_mobilities, workspace.water_mobilities, workspace.gas_mobilities)
    flags = (relevant_phases.oil > 0.0, relevant_phases.water > 0.0, relevant_phases.gas > 0.0)
    sample_densities = (
        np.array([sample.phase_densities.oil for sample in connection_samples]),
        np.array([sample.phase_densities.water for sample in connection_samples]),
        np.array([sample.phase_densities.gas for sample in connection_samples]),
    )
    densities = []
    deliverabilities = []
    for mobility, sample_density, flag in zip(mobilities, sample_densities, flags, strict=True):
        weights = workspace.well_indices * mobility
        total = float(weights.sum())
        densities.append(
            float(np.dot(weights, sample_density) / total)
            if total > 0.0
            else float(sample_density.mean())
        )
        deliverabilities.append(total if flag else 0.0)

    total_deliverability = sum(deliverabilities)
    if total_deliverability > 0.0:
        fractions = [value / total_deliverability for value in deliverabilities]
    else:
        count = sum(flags)
        fractions = [1.0 / count if flag else 0.0 for flag in flags]
    return tubing._replace(
        flowing_fluid=FlowingFluid(
            phase_densities=PhaseValues(*densities),
            fallback_fractions=PhaseValues(*fractions),
        )
    )


def compute_flowing_density(*, flowing_fluid: FlowingFluid, phase_rates: PhaseValues) -> Number:
    """
    Density of the mixture a well flows: its total mass rate over its total volume rate, both
    at reservoir conditions.

    A well flowing nothing is given the density of its fallback phase fractions instead.

    :param flowing_fluid: The fluid the well flows.
    :param phase_rates: Reservoir-condition phase rates.
    :returns: The mixture density.
    """
    rates = (abs(phase_rates.oil), abs(phase_rates.water), abs(phase_rates.gas))
    densities = flowing_fluid.phase_densities
    volume = sum(rates)
    if volume > 0.0:
        mass = sum(rate * density for rate, density in zip(rates, densities, strict=True))
        return mass / volume
    return sum(
        fraction * density
        for fraction, density in zip(flowing_fluid.fallback_fractions, densities, strict=True)
    )


def compute_datum_pressure_shift(
    *,
    table: VFPTable,
    reference_depth: Number,
    flowing_fluid: FlowingFluid | None,
    phase_rates: PhaseValues,
) -> Number:
    """
    The pressure to add to the BHP at the well's reference depth to get the BHP at a VFP
    table's datum depth.

    The fluid column between the two depths is the flowing mixture, with no friction, so the
    shift is its weight: positive when the datum is deeper than the reference depth, negative
    when it is shallower.

    :param table: The VFP table.
    :param reference_depth: The well's reference depth.
    :param flowing_fluid: The fluid the well flows. Not needed if the depths match.
    :param phase_rates: Reservoir-condition phase rates.
    :returns: The pressure shift, in the table's unit system.
    :raises ValidationError: If the depths differ and `flowing_fluid` is `None`.
    """
    depth_difference = table.datum_depth - reference_depth
    if math.isclose(depth_difference, 0.0, abs_tol=1e-6):
        return 0.0
    if flowing_fluid is None:
        raise ValidationError(
            f"VFP table {table.table_number} refers to a datum depth of {table.datum_depth}, but "
            f"this well's reference depth is {reference_depth}, and the fluid the well flows isn't "
            "known, so the table's BHP can't be moved between the two depths."
        )

    unit_system = table.unit_system
    gravity = typing.cast(Number, c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE)
    if unit_system != UnitSystem.FIELD:
        gravity = gravity * get_conversion_factors(UnitSystem.FIELD, unit_system)["length"]
    return compute_hydrostatic_pressure(
        density=compute_flowing_density(flowing_fluid=flowing_fluid, phase_rates=phase_rates),
        gravitational_acceleration=gravity,
        length=depth_difference,
        unit_system=unit_system,
    )


def compute_tubing_head_pressure(
    *,
    tubing: WellTubing,
    reference_depth: Number,
    reference_pressure: Number,
    phase_rates: PhaseValues,
    surface_phase_rates: PhaseValues,
    surface_fluid_properties: SurfaceFluidProperties | None,
    is_injector: bool,
) -> Number:
    """
    Finds the tubing head pressure for a bottom-hole pressure and the rates it produces.

    A VFP table is used if the well has one, otherwise its hydraulics correlation. A table's
    BHP is at its own datum depth, so if that differs from `reference_depth` the BHP is first
    moved to the datum by the weight of the flowing fluid column between them.

    :param tubing: The well's tubing performance.
    :param reference_depth: Depth the bottom-hole pressure refers to.
    :param reference_pressure: The bottom-hole pressure.
    :param phase_rates: Reservoir-condition phase rates (used by a correlation).
    :param surface_phase_rates: Surface-condition phase rates (used by a VFP table).
    :param surface_fluid_properties: Surface fluid properties. Only a correlation needs them.
    :param is_injector: Whether the well is an injector.
    :returns: The tubing head pressure.
    :raises ValidationError: If the well has no tubing performance, a correlation is needed but
        `surface_fluid_properties` is `None`, or a VFP table's datum depth differs from
        `reference_depth` and the well's `flowing_fluid` isn't set.
    """
    if tubing.vfp_table is not None:
        table = tubing.vfp_table
        datum_pressure = reference_pressure + compute_datum_pressure_shift(
            table=table,
            reference_depth=reference_depth,
            flowing_fluid=tubing.flowing_fluid,
            phase_rates=phase_rates,
        )
        lookup = table.get_lookup_variables(
            oil_rate=surface_phase_rates.oil,
            water_rate=surface_phase_rates.water,
            gas_rate=surface_phase_rates.gas,
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
    if surface_fluid_properties is None:
        raise ValidationError(
            "A hydraulics correlation needs `surface_fluid_properties` to find a tubing head "
            "pressure."
        )
    return compute_correlation_thp(
        wellbore=tubing.wellbore,
        reference_depth=reference_depth,
        reference_pressure=reference_pressure,
        phase_rates=phase_rates,
        surface_fluid_properties=surface_fluid_properties,
        is_injector=is_injector,
    )
