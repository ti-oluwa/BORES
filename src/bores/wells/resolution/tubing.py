"""
What relates a well's bottom-hole pressure to its tubing head pressure, as the resolver sees it.

A well's tubing performance is either a hydraulics correlation (`WellBoreModel`) or the deck's
vertical flow performance tables (`VFPTables`). `WellTubing` is the single object the resolver
passes around for one well, and `compute_tubing_head_pressure` is the single place that decides
which of the two answers a THP question.
"""

import math
import typing

from bores.errors import ValidationError
from bores.types import Integer, Number
from bores.wells.base import WellType
from bores.wells.compile import CompiledWellSystem, WellKind
from bores.wells.hydraulics.base import PhaseValues, SurfaceFluidProperties, WellBoreModel
from bores.wells.hydraulics.base import compute_tubing_head_pressure as compute_correlation_thp
from bores.wells.hydraulics.vfp import VFPTable, VFPTables

__all__ = ["WellTubing", "compute_tubing_head_pressure", "resolve_well_tubing"]


class WellTubing(typing.NamedTuple):
    """The tubing performance of one well, resolved for the current moment of a run."""

    wellbore: WellBoreModel | None
    """Hydraulics correlation. Also distributes pressure across the well's connections."""

    vfp_table: VFPTable | None
    """The VFP table the well's control currently selects."""

    artificial_lift_quantity: Number
    """The artificial lift quantity the well currently operates at."""

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

    A VFP table is used if the well has one, otherwise its hydraulics correlation.

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
        `reference_depth`.
    """
    if tubing.vfp_table is not None:
        table = tubing.vfp_table
        if not math.isclose(table.datum_depth, reference_depth, rel_tol=1e-6, abs_tol=1e-6):
            raise ValidationError(
                f"VFP table {table.table_number} refers to a datum depth of {table.datum_depth}, "
                f"but this well's reference depth is {reference_depth}. Shifting a table's BHP "
                "between depths isn't supported, so they must match."
            )

        if is_injector:
            flow_rate = abs(surface_phase_rates.water) + abs(surface_phase_rates.gas)
            water_cut, gas_oil_ratio = 0.0, 0.0
        else:
            oil_rate, water_rate = abs(surface_phase_rates.oil), abs(surface_phase_rates.water)
            flow_rate = oil_rate + water_rate
            water_cut = water_rate / flow_rate if flow_rate > 0.0 else 0.0
            gas_oil_ratio = abs(surface_phase_rates.gas) / oil_rate if oil_rate > 0.0 else 0.0
        return table.get_thp(
            flow_rate=flow_rate,
            bhp=reference_pressure,
            water_cut=water_cut,
            gas_oil_ratio=gas_oil_ratio,
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
