"""Top-level well-control resolution. One well's row per call."""

import typing

from bores.errors import StopSimulation, ValidationError
from bores.types import Integer, NumberArray, OneDimension
from bores.wells.compile import (
    CompiledWellSystem,
    InjectorControlModeTag,
    ProducerControlModeTag,
    WellKind,
)
from bores.wells.hydraulics.base import SurfaceFluidProperties, WellBoreModel
from bores.wells.resolution.limits import apply_limits
from bores.wells.resolution.solvers import (
    ALL_PHASES,
    bisect_bhp,
    compute_phase_rates,
    compute_tubing_head_pressure,
    get_default_pressure_bracket,
    phase_mask,
    solve_injector_bhp_mode,
    solve_injector_rate_mode,
    solve_producer_bhp_mode,
    solve_producer_rate_mode,
)
from bores.wells.resolution.spec import WellControlSpec
from bores.wells.state import ConnectionSample
from bores.wells.workspace import WellsWorkspace, build_perforation_workspace

__all__ = ["resolve_well_control"]


def resolve_well_control(
    *,
    compiled_system: CompiledWellSystem,
    well_row: Integer,
    wellbore: WellBoreModel,
    connection_samples: typing.Sequence[ConnectionSample],
    workspace: WellsWorkspace,
    control_spec: WellControlSpec,
    surface_fluid_properties: SurfaceFluidProperties | None = None,
) -> None:
    """
    Resolves one well's control and writes the result into `resolution`'s
    row at `well_row`.

    A `PENDING` well (`compiled_system.schedule_statuses[well_row] == 0`)
    or an `UNSET`-mode well is left untouched (still `NaN`/`UNSET_INT`/`0`
    from `compile_well_resolution`) as there's nothing to resolve.

    :param compiled_system: The compiled well system.
    :param well_row: Which well, by row index into `compiled_system`.
    :param wellbore: Hydraulics correlation for this well.
    :param connection_samples: Reservoir samples for this well's active,
        open connections, in the same order `compiled_system.perforations`'
        rows for this well appear (after filtering to
        `completion_statuses == 1` and `schedule_statuses == 1`).
    :param workspace: The system-wide `WellsWorkspace` to update in place.
    :param control_spec: Solver tunables.
    :param surface_fluid_properties: Required to compute THP, or to
        resolve a `THP`-mode control, or to check a `THPLimit`. Omit if
        none of those apply to this well.
    :raises ValidationError: If `control_modes[well_row]` isn't a
        recognized tag for this well's `well_kinds[well_row]`.
    """
    if compiled_system.schedule_statuses[well_row] == 0:
        return

    controls = compiled_system.controls
    control_mode = controls.control_modes[well_row]
    well_kind = controls.well_kinds[well_row]
    is_injector = well_kind == WellKind.INJECTOR
    reference_depth = compiled_system.reference_depths[well_row]

    perforations = compiled_system.perforations
    perforation_start, perforation_end = (
        perforations.well_offsets[well_row],
        perforations.well_offsets[well_row + 1],
    )
    active_open = [
        i
        for i in range(perforation_start, perforation_end)
        if perforations.completion_statuses[i] == 1 and perforations.schedule_statuses[i] == 1
    ]
    if not active_open or len(active_open) != len(connection_samples):
        raise ValidationError(
            f"`resolve_well_control`: `well_row` {well_row} has {len(active_open)} "
            f"active/open connections but was given {len(connection_samples)} "
            "`connection_samples`. These must match 1:1."
        )

    perforation_workspace = build_perforation_workspace(
        well_indices=typing.cast(
            NumberArray[OneDimension], perforations.well_indices[active_open]
        ),
        representative_depths=typing.cast(
            NumberArray[OneDimension], perforations.representative_depths[active_open]
        ),
        inclinations_from_vertical=typing.cast(
            NumberArray[OneDimension],
            perforations.inclinations_from_vertical[active_open],
        ),
        connection_samples=connection_samples,
        connection_conductivities=typing.cast(
            NumberArray[OneDimension],
            perforations.connection_conductivities[active_open],
        ),
        d_factor=compiled_system.d_factors[well_row],
    )

    if is_injector:
        if control_mode == InjectorControlModeTag.UNSET:
            return
        injected_phase = controls.injected_phases[well_row]
        relevant_phases = phase_mask(injected_phase)
        if control_mode in (
            InjectorControlModeTag.RATE,
            InjectorControlModeTag.RESERVOIR_VOLUME_RATE,
        ):
            bhp, connection_pressures, phase_rates, surface_phase_rates = solve_injector_rate_mode(
                control_mode=control_mode,
                target_rate=controls.target_rates[well_row],
                injected_phase=injected_phase,
                wellbore=wellbore,
                reference_depth=reference_depth,
                workspace=perforation_workspace,
                connection_samples=connection_samples,
                control_spec=control_spec,
            )
        elif control_mode == InjectorControlModeTag.BHP:
            bhp, connection_pressures, phase_rates, surface_phase_rates = solve_injector_bhp_mode(
                target_bhp=controls.target_bhps[well_row],
                injected_phase=injected_phase,
                wellbore=wellbore,
                reference_depth=reference_depth,
                workspace=perforation_workspace,
                connection_samples=connection_samples,
                control_spec=control_spec,
            )
        elif control_mode == InjectorControlModeTag.THP:
            if surface_fluid_properties is None:
                raise ValidationError("A THP-mode injector requires `surface_fluid_properties`.")
            min_pressure, max_pressure = get_default_pressure_bracket(
                connection_samples, is_injector=True, control_spec=control_spec
            )
            bhp, _, _ = bisect_bhp(
                wellbore=wellbore,
                reference_depth=reference_depth,
                workspace=perforation_workspace,
                connection_samples=connection_samples,
                relevant_phases=relevant_phases,
                is_injector=True,
                target=controls.target_thps[well_row],
                min_pressure=min_pressure,
                max_pressure=max_pressure,
                control_spec=control_spec,
                metric="thp",
                surface_fluid_properties=surface_fluid_properties,
            )
            connection_pressures, phase_rates, surface_phase_rates = compute_phase_rates(
                wellbore=wellbore,
                reference_depth=reference_depth,
                workspace=perforation_workspace,
                connection_samples=connection_samples,
                reference_pressure=bhp,
                relevant_phases=relevant_phases,
                is_injector=True,
                control_spec=control_spec,
            )
        elif control_mode == InjectorControlModeTag.GROUP:
            raise ValidationError(
                f"Well row {well_row} is under `GRUP` control. Resolve group "
                "allocation (wells.resolution.groups.allocation) into a concrete "
                "rate/BHP target before calling `resolve_well_control`."
            )
        else:
            raise ValidationError(f"Unknown InjectorControlModeTag: {control_mode!r}.")
    else:
        if control_mode == ProducerControlModeTag.UNSET:
            return

        relevant_phases = ALL_PHASES
        if control_mode in (
            ProducerControlModeTag.OIL_RATE,
            ProducerControlModeTag.WATER_RATE,
            ProducerControlModeTag.GAS_RATE,
            ProducerControlModeTag.LIQUID_RATE,
            ProducerControlModeTag.RESERVOIR_VOLUME_RATE,
        ):
            bhp, connection_pressures, phase_rates, surface_phase_rates = solve_producer_rate_mode(
                control_mode=control_mode,
                target_rate=controls.target_rates[well_row],
                wellbore=wellbore,
                reference_depth=reference_depth,
                workspace=perforation_workspace,
                connection_samples=connection_samples,
                control_spec=control_spec,
            )
        elif control_mode == ProducerControlModeTag.BHP:
            bhp, connection_pressures, phase_rates, surface_phase_rates = solve_producer_bhp_mode(
                target_bhp=controls.target_bhps[well_row],
                wellbore=wellbore,
                reference_depth=reference_depth,
                workspace=perforation_workspace,
                connection_samples=connection_samples,
                control_spec=control_spec,
            )
        elif control_mode == ProducerControlModeTag.THP:
            if surface_fluid_properties is None:
                raise ValidationError("A THP-mode producer requires `surface_fluid_properties`.")
            min_pressure, max_pressure = get_default_pressure_bracket(
                connection_samples, is_injector=False, control_spec=control_spec
            )
            bhp, _, _ = bisect_bhp(
                wellbore=wellbore,
                reference_depth=reference_depth,
                workspace=perforation_workspace,
                connection_samples=connection_samples,
                relevant_phases=relevant_phases,
                is_injector=False,
                target=controls.target_thps[well_row],
                min_pressure=min_pressure,
                max_pressure=max_pressure,
                control_spec=control_spec,
                metric="thp",
                surface_fluid_properties=surface_fluid_properties,
            )
            connection_pressures, phase_rates, surface_phase_rates = compute_phase_rates(
                wellbore=wellbore,
                reference_depth=reference_depth,
                workspace=perforation_workspace,
                connection_samples=connection_samples,
                reference_pressure=bhp,
                relevant_phases=relevant_phases,
                is_injector=False,
                control_spec=control_spec,
            )
        elif control_mode == ProducerControlModeTag.GROUP:
            raise ValidationError(
                f"Well row {well_row} is under `GRUP` control. Resolve group "
                "allocation (wells.resolution.groups.allocation) into a concrete "
                "rate/BHP target before calling `resolve_well_control`."
            )
        else:
            raise ValidationError(f"Unknown `ProducerControlModeTag`: {control_mode!r}.")

    limits = controls.limits
    limits_start, limits_end = (
        limits.well_offsets[well_row],
        limits.well_offsets[well_row + 1],
    )
    min_pressure, max_pressure = get_default_pressure_bracket(
        connection_samples, is_injector=is_injector, control_spec=control_spec
    )
    (
        bhp,
        connection_pressures,
        phase_rates,
        surface_phase_rates,
        active_limit_row,
        economic_shutin,
    ) = apply_limits(
        limits=limits,
        limits_start=limits_start,
        limits_end=limits_end,
        wellbore=wellbore,
        reference_depth=reference_depth,
        workspace=perforation_workspace,
        connection_samples=connection_samples,
        relevant_phases=relevant_phases,
        is_injector=is_injector,
        bhp=bhp,
        connection_pressures=connection_pressures,
        phase_rates=phase_rates,
        surface_phase_rates=surface_phase_rates,
        min_pressure=min_pressure,
        max_pressure=max_pressure,
        control_spec=control_spec,
        surface_fluid_properties=surface_fluid_properties,
    )

    workspace.bhps[well_row] = bhp
    workspace.oil_rates[well_row] = phase_rates.oil
    workspace.water_rates[well_row] = phase_rates.water
    workspace.gas_rates[well_row] = phase_rates.gas
    workspace.surface_oil_rates[well_row] = surface_phase_rates.oil
    workspace.surface_water_rates[well_row] = surface_phase_rates.water
    workspace.surface_gas_rates[well_row] = surface_phase_rates.gas
    workspace.active_limit_rows[well_row] = active_limit_row
    workspace.economic_shutins[well_row] = 1 if economic_shutin else 0
    workspace.connection_pressures[active_open] = connection_pressures
    if economic_shutin:
        workspace.connection_oil_rates[active_open] = 0.0
        workspace.connection_water_rates[active_open] = 0.0
        workspace.connection_gas_rates[active_open] = 0.0
    else:
        # workspace's own per-connection scratch arrays hold whatever the
        # last solve call for this well wrote into them which is the governing
        # one, whether that was the nominal resolution or `apply_limits`'
        # own recompute at a limit-adjusted BHP, since every such call
        # reuses (overwrites) the same buffers. Safe to read here only
        # because nothing else touches this workspace after this point.
        workspace.connection_oil_rates[active_open] = perforation_workspace.connection_oil_rates
        workspace.connection_water_rates[active_open] = (
            perforation_workspace.connection_water_rates
        )
        workspace.connection_gas_rates[active_open] = perforation_workspace.connection_gas_rates

    if economic_shutin and limits.end_run_flags[active_limit_row]:
        raise StopSimulation(
            f"Well {compiled_system.names[well_row]!r} breached an economic limit "
            "flagged to end the run."
        )

    if surface_fluid_properties is not None:
        workspace.thps[well_row] = compute_tubing_head_pressure(
            wellbore=wellbore,
            reference_depth=reference_depth,
            reference_pressure=bhp,
            phase_rates=phase_rates,
            surface_fluid_properties=surface_fluid_properties,
            is_injector=is_injector,
        )
    return None
