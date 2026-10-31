"""`CompiledWellSystem` resolution for one timestep."""

import math
import typing

from bores.errors import ValidationError
from bores.types import Integer
from bores.wells.compile import UNSET_INT, CompiledWellSystem
from bores.wells.hydraulics.base import (
    ConnectionDerivatives,
    ConnectionFluids,
    SurfaceSample,
    WellBoreModel,
)
from bores.wells.hydraulics.vfp import VFPTables
from bores.wells.resolution.derivatives import compute_well_derivatives
from bores.wells.resolution.engine import resolve_well_control
from bores.wells.resolution.groups.allocation import (
    allocate_group_targets,
    redistribute_group_shortfall,
)
from bores.wells.resolution.groups.limits import (
    GroupEconomicLimitOutcome,
    enforce_group_economic_limits,
)
from bores.wells.resolution.spec import WellControlSpec
from bores.wells.workspace import WellsWorkspace

__all__ = ["resolve_wells"]


def resolve_wells(
    *,
    compiled_system: CompiledWellSystem,
    workspace: WellsWorkspace,
    spec: WellControlSpec,
    get_tubing_performance: typing.Callable[[Integer], WellBoreModel | VFPTables | None],
    get_connection_fluids: typing.Callable[[Integer], ConnectionFluids],
    get_surface_sample: typing.Callable[[Integer], SurfaceSample | None] | None = None,
    get_connection_derivatives: typing.Callable[[Integer], ConnectionDerivatives] | None = None,
) -> dict[str, GroupEconomicLimitOutcome]:
    """
    Resolves every well in `compiled_system` for one timestep, in the
    order a `GRUP`-mode well and a group economic limit both need:

    1. Allocates every group's own target across its `GRUP`-mode members
       (`allocate_group_targets`), giving each a concrete mode and target for this pass.
       A group whose own control mode has no directly allocatable target
       (`FLD`/`NONE`/`VREP`/`REIN`) is skipped. Its members are
       controlled some other way, not this group's own target.
    2. Resolves every active well's control (`resolve_well_control`), now
       that none are left in `GRUP` mode.
    3. Redistributes any shortfall left in a group's own target across
       whichever of its members have headroom
       (`redistribute_group_shortfall`), re-resolving each one boosted.
    4. Enforces every group's own `GECON` limits
       (`enforce_group_economic_limits`), which may shut a member well,
       cut a group's target, or both with each followed by its own
       reallocation and re-resolution of the wells it affects, so the
       workspace is fully consistent before this function returns.
    5. If `get_connection_derivatives` is given, computes every well's derivatives
       (`compute_well_derivatives`) from the settled solution.

    :param compiled_system: The compiled well system to resolve.
    :param workspace: This run's `WellsWorkspace`, updated in place.
    :param spec: Solver tunables, shared by every well and group.
    :param get_tubing_performance: Given a well row, what relates its BHP to its THP
        (a `WellBoreModel` or a `VFPTables` collection), or `None` if it has none.
    :param get_connection_fluids: Given a well row, its active, open
        connections' current reservoir conditions.
    :param get_surface_sample: Given a well row, its surface
        fluid properties, or `None` if it has no THP control/limit to
        check. Omit if no well in the system ever needs this.
    :param get_connection_derivatives: Given a well row, how its connections' fluid properties
        change with their cells' unknowns. Omit to skip computing the well derivatives, which
        need a workspace built with `well_offsets`.
    :returns: Each group's own `enforce_group_economic_limits` outcome,
        keyed by group name. Empty if `compiled_system.group_controls` is `None`.
    """
    group_controls = compiled_system.group_controls
    workspace.group_control_modes[:] = UNSET_INT
    workspace.group_target_rates[:] = math.nan
    allocated_wells_by_group: dict[str, tuple[str, ...]] = {}

    if group_controls is not None:
        for group_name in group_controls.names:
            try:
                allocated_wells_by_group[group_name] = allocate_group_targets(
                    group_name, well_system=compiled_system, workspace=workspace
                )
            except ValidationError:
                # This group's own control mode has no directly
                # allocatable rate target. Its members are controlled
                # some other way, not this group's own target.
                continue

    for well_row in range(len(compiled_system.names)):
        if compiled_system.schedule_statuses[well_row] == 0:
            continue
        resolve_well_control(
            compiled_system=compiled_system,
            well_row=well_row,
            tubing_performance=get_tubing_performance(well_row),
            connection_fluids=get_connection_fluids(well_row),
            workspace=workspace,
            spec=spec,
            surface_sample=(get_surface_sample(well_row) if get_surface_sample else None),
        )

    outcomes: dict[str, GroupEconomicLimitOutcome] = {}
    if group_controls is not None:
        for group_name, allocated_wells in allocated_wells_by_group.items():
            if not allocated_wells:
                continue
            redistribute_group_shortfall(
                group_name=group_name,
                allocated_wells=allocated_wells,
                well_system=compiled_system,
                workspace=workspace,
                spec=spec,
                get_tubing_performance=get_tubing_performance,
                get_connection_fluids=get_connection_fluids,
                get_surface_sample=get_surface_sample,
            )

        group_limits = group_controls.limits
        for group_row, group_name in enumerate(group_controls.names):
            has_limits = (
                group_limits.group_offsets[group_row] != group_limits.group_offsets[group_row + 1]
            )
            if not has_limits:
                continue
            outcomes[group_name] = enforce_group_economic_limits(
                group_name=group_name,
                well_system=compiled_system,
                workspace=workspace,
                spec=spec,
                get_tubing_performance=get_tubing_performance,
                get_connection_fluids=get_connection_fluids,
                get_surface_sample=get_surface_sample,
            )

    if get_connection_derivatives is not None:
        for well_row in range(len(compiled_system.names)):
            compute_well_derivatives(
                compiled_system=compiled_system,
                well_row=well_row,
                tubing_performance=get_tubing_performance(well_row),
                connection_fluids=get_connection_fluids(well_row),
                connection_derivatives=get_connection_derivatives(well_row),
                workspace=workspace,
                spec=spec,
                surface_sample=(get_surface_sample(well_row) if get_surface_sample else None),
            )
    return outcomes
