"""
Complete `CompiledWellSystem` resolution for one timestep.

`resolution.engine.resolve_well_control` resolves one well; it raises if
that well is still in `GRUP` mode, since a `GRUP`-mode well has no
concrete target until its group's own allocation runs. This module is
the orchestrator that sequences group allocation, well resolution, and
group economic-limit enforcement correctly, so a caller never has to get
that ordering right by hand.
"""

import typing

from bores.errors import ValidationError
from bores.types import Integer
from bores.wells.compile import CompiledWellSystem
from bores.wells.hydraulics.base import SurfaceFluidProperties, WellBoreModel
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
from bores.wells.state import ConnectionSample
from bores.wells.workspace import WellsWorkspace

__all__ = ["resolve_wells"]


def resolve_wells(
    *,
    compiled_system: CompiledWellSystem,
    workspace: WellsWorkspace,
    control_spec: WellControlSpec,
    get_wellbore: typing.Callable[[Integer], WellBoreModel | None],
    get_connection_samples: typing.Callable[[Integer], typing.Sequence[ConnectionSample]],
    get_surface_fluid_properties: typing.Callable[[Integer], SurfaceFluidProperties | None]
    | None = None,
) -> dict[str, GroupEconomicLimitOutcome]:
    """
    Resolves every well in `compiled_system` for one timestep, in the
    order a `GRUP`-mode well and a group economic limit both need:

    1. Allocates every group's own target across its `GRUP`-mode members
       (`allocate_group_targets`), converting each to a concrete mode.
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

    :param compiled_system: The compiled well system to resolve.
    :param workspace: This run's `WellsWorkspace`, updated in place.
    :param control_spec: Solver tunables, shared by every well and group.
    :param get_wellbore: Given a well row, its hydraulics correlation, or
        `None` if it has none.
    :param get_connection_samples: Given a well row, its active, open
        connections' current reservoir samples.
    :param get_surface_fluid_properties: Given a well row, its surface
        fluid properties, or `None` if it has no THP control/limit to
        check. Omit if no well in the system ever needs this.
    :returns: Each group's own `enforce_group_economic_limits` outcome,
        keyed by group name. Empty if `compiled_system.group_controls` is `None`.
    """
    group_controls = compiled_system.group_controls
    allocated_wells_by_group: dict[str, tuple[str, ...]] = {}

    if group_controls is not None:
        for group_name in group_controls.names:
            try:
                allocated_wells_by_group[group_name] = allocate_group_targets(
                    group_name, compiled_system, workspace
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
            wellbore=get_wellbore(well_row),
            connection_samples=get_connection_samples(well_row),
            workspace=workspace,
            control_spec=control_spec,
            surface_fluid_properties=(
                get_surface_fluid_properties(well_row) if get_surface_fluid_properties else None
            ),
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
                control_spec=control_spec,
                get_wellbore=get_wellbore,
                get_connection_samples=get_connection_samples,
                get_surface_fluid_properties=get_surface_fluid_properties,
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
                control_spec=control_spec,
                get_wellbore=get_wellbore,
                get_connection_samples=get_connection_samples,
                get_surface_fluid_properties=get_surface_fluid_properties,
            )
    return outcomes
