"""Top-level well-control resolution. One well's row per call."""

import functools
import typing

import numpy as np

from bores.errors import StopSimulation, ValidationError, WellConvergenceError
from bores.types import Integer, NumberArray, OneDimension, P, R
from bores.wells.compile import (
    UNSET_INT,
    CompiledWellSystem,
    InjectorControlModeTag,
    ProducerControlModeTag,
    WellKind,
)
from bores.wells.hydraulics.base import ConnectionFluids, SurfaceSample, WellBoreModel
from bores.wells.hydraulics.vfp import VFPTables
from bores.wells.resolution.limits import apply_limits
from bores.wells.resolution.solvers import (
    ALL_PHASES,
    get_pressure_bracket,
    phase_mask,
    solve_bhp,
    solve_bhp_injector,
    solve_bhp_producer,
    solve_connection_pressures_and_rates,
    solve_rate_injector,
    solve_rate_producer,
)
from bores.wells.resolution.spec import WellControlSpec
from bores.wells.resolution.tubing import (
    attach_flowing_fluid,
    compute_tubing_head_pressure,
    resolve_well_tubing,
)
from bores.wells.workspace import (
    ConnectionRates,
    WellsWorkspace,
    build_perforation_workspace,
    get_connection_rates,
)

__all__ = ["record_shut_well", "resolve_well_control"]


def record_shut_well(
    *,
    compiled_system: CompiledWellSystem,
    well_row: Integer,
    workspace: WellsWorkspace,
) -> None:
    """
    Records a well that has no open connection as flowing nothing.

    Every rate of the well, at reservoir and surface conditions, is zero. Its bottom-hole and
    tubing-head pressures are `NaN`, since a well that is not connected to the reservoir has no
    flowing pressure. Its connections carry zero rate and no pressure, and no limit binds.

    :param compiled_system: The compiled well system.
    :param well_row: Which well, by row index into `compiled_system`.
    :param workspace: The system-wide `WellsWorkspace` to update in place.
    """
    workspace.bhps[well_row] = np.nan
    workspace.thps[well_row] = np.nan
    workspace.oil_rates[well_row] = 0.0
    workspace.water_rates[well_row] = 0.0
    workspace.gas_rates[well_row] = 0.0
    workspace.surface_oil_rates[well_row] = 0.0
    workspace.surface_water_rates[well_row] = 0.0
    workspace.surface_gas_rates[well_row] = 0.0
    workspace.active_limit_rows[well_row] = UNSET_INT
    workspace.economic_shutins[well_row] = 0
    perforations = compiled_system.perforations
    start, end = perforations.well_offsets[well_row], perforations.well_offsets[well_row + 1]
    workspace.connection_pressures[start:end] = np.nan
    workspace.connection_oil_rates[start:end] = 0.0
    workspace.connection_water_rates[start:end] = 0.0
    workspace.connection_gas_rates[start:end] = 0.0


def name_well_in_convergence_errors(func: typing.Callable[P, R]) -> typing.Callable[P, R]:
    """
    Wraps a keyword-only well resolution function so that a `WellConvergenceError` it raises
    names the well it was resolving.

    :param func: A function taking `compiled_system` and `well_row` as keyword arguments.
    :returns: The wrapped function.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except WellConvergenceError as exc:
            name = kwargs["compiled_system"].names[kwargs["well_row"]]  # type: ignore[attr-defined]
            raise WellConvergenceError(f"Well {name!r}: {exc}") from exc

    return wrapper


@name_well_in_convergence_errors
def resolve_well_control(
    *,
    compiled_system: CompiledWellSystem,
    well_row: Integer,
    tubing_performance: WellBoreModel | VFPTables | None,
    connection_fluids: ConnectionFluids,
    workspace: WellsWorkspace,
    control_spec: WellControlSpec,
    surface_sample: SurfaceSample | None = None,
) -> None:
    """
    Resolves one well's control and writes the result into `resolution`'s
    row at `well_row`.

    A `PENDING` well (`compiled_system.schedule_statuses[well_row] == 0`)
    or an `UNSET`-mode well is left untouched (still `NaN`/`UNSET_INT`/`0`
    from `compile_well_resolution`) as there's nothing to resolve.

    An active well with no open connection, because every one is shut, flows nothing, and
    `record_shut_well` writes that. On every call, the connections of an active well that are
    not open, whether shut or yet to come into effect, are given zero rate and no pressure, so
    nothing from an earlier call lingers on them.

    :param compiled_system: The compiled well system.
    :param well_row: Which well, by row index into `compiled_system`.
    :param tubing_performance: What relates this well's BHP to its THP: a `WellBoreModel`,
        a `VFPTables` collection (the well's control selects its table by number), or `None`.
        THP control, a `THPLimit`, and THP reporting require one, regardless of
        `control_spec.connection_pressure_mode`. A `WellBoreModel` distributes pressure across
        connections; a well without one follows `control_spec.connection_pressure_mode`.
    :param connection_fluids: Reservoir conditions at this well's active, open connections,
        in the same order `compiled_system.perforations`' rows for this well appear (after
        filtering to `completion_statuses == 1` and `schedule_statuses == 1`).
    :param workspace: The system-wide `WellsWorkspace` to update in place.
    :param control_spec: Solver tunables.
    :param surface_sample: Required to compute THP with a `WellBoreModel`, or to
        resolve a `THP`-mode control or check a `THPLimit` with one. A VFP table doesn't need
        it. Omit if none of those apply to this well.
    :raises ValidationError: If `control_modes[well_row]` isn't a
        recognized tag for this well's `well_kinds[well_row]`, if this
        well is under THP control or has a `THPLimit`/THP reporting
        requested and it has neither a `WellBoreModel` nor a VFP table, or if it has no
        `WellBoreModel` and `control_spec.connection_pressure_mode` is `HYDRAULIC`
        while this well still needs connection-pressure distribution.
    """
    if compiled_system.schedule_statuses[well_row] == 0:
        return

    controls = compiled_system.controls
    tubing = resolve_well_tubing(
        tubing_performance=tubing_performance,
        compiled_system=compiled_system,
        well_row=well_row,
    )
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
    if len(active_open) != len(connection_fluids.pressures):
        raise ValidationError(
            f"`resolve_well_control`: `well_row` {well_row} has {len(active_open)} "
            f"active/open connections but was given {len(connection_fluids.pressures)} "
            "connections in `connection_fluids`. These must match 1:1."
        )
    if not active_open:
        if control_mode != (
            InjectorControlModeTag.UNSET if is_injector else ProducerControlModeTag.UNSET
        ):
            record_shut_well(
                compiled_system=compiled_system, well_row=well_row, workspace=workspace
            )
        return

    is_open = np.zeros(perforation_end - perforation_start, dtype=np.bool_)
    is_open[np.asarray(active_open) - perforation_start] = True
    not_open = np.flatnonzero(~is_open) + perforation_start
    workspace.connection_pressures[not_open] = np.nan
    workspace.connection_oil_rates[not_open] = 0.0
    workspace.connection_water_rates[not_open] = 0.0
    workspace.connection_gas_rates[not_open] = 0.0

    perforation_workspace = build_perforation_workspace(
        well_indices=typing.cast(
            NumberArray[OneDimension], perforations.well_indices[active_open]
        ),
        representative_depths=typing.cast(
            NumberArray[OneDimension], perforations.representative_depths[active_open]
        ),
        representative_measured_depths=typing.cast(
            NumberArray[OneDimension], perforations.representative_measured_depths[active_open]
        ),
        inclinations_from_vertical=typing.cast(
            NumberArray[OneDimension],
            perforations.inclinations_from_vertical[active_open],
        ),
        connection_fluids=connection_fluids,
        reference_measured_depth=compiled_system.reference_measured_depths[well_row],
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
        tubing = attach_flowing_fluid(
            tubing=tubing,
            workspace=perforation_workspace,
            relevant_phases=relevant_phases,
        )
        if control_mode in (
            InjectorControlModeTag.RATE,
            InjectorControlModeTag.RESERVOIR_VOLUME_RATE,
        ):
            bhp, connection_pressures, phase_rates, surface_phase_rates = solve_rate_injector(
                control_mode=control_mode,
                target_rate=controls.target_rates[well_row],
                injected_phase=injected_phase,
                tubing=tubing,
                reference_depth=reference_depth,
                workspace=perforation_workspace,
                control_spec=control_spec,
            )
        elif control_mode == InjectorControlModeTag.BHP:
            bhp, connection_pressures, phase_rates, surface_phase_rates = solve_bhp_injector(
                target_bhp=controls.target_bhps[well_row],
                injected_phase=injected_phase,
                tubing=tubing,
                reference_depth=reference_depth,
                workspace=perforation_workspace,
                control_spec=control_spec,
            )
        elif control_mode == InjectorControlModeTag.THP:
            if surface_sample is None:
                raise ValidationError("A THP-mode injector requires `surface_sample`.")
            if not tubing.can_evaluate_thp:
                raise ValidationError(
                    "A THP-mode injector requires a `WellBoreModel` or VFP table, "
                    "regardless of `control_spec.connection_pressure_mode`."
                )
            min_pressure, max_pressure = get_pressure_bracket(
                workspace=perforation_workspace, is_injector=True, control_spec=control_spec
            )
            bhp, _, _ = solve_bhp(
                tubing=tubing,
                reference_depth=reference_depth,
                workspace=perforation_workspace,
                relevant_phases=relevant_phases,
                is_injector=True,
                target=controls.target_thps[well_row],
                min_pressure=min_pressure,
                max_pressure=max_pressure,
                control_spec=control_spec,
                metric="thp",
                surface_sample=surface_sample,
            )
            connection_pressures, phase_rates, surface_phase_rates = (
                solve_connection_pressures_and_rates(
                    tubing=tubing,
                    reference_depth=reference_depth,
                    workspace=perforation_workspace,
                    reference_pressure=bhp,
                    relevant_phases=relevant_phases,
                    is_injector=True,
                    control_spec=control_spec,
                )
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
        tubing = attach_flowing_fluid(
            tubing=tubing,
            workspace=perforation_workspace,
            relevant_phases=relevant_phases,
        )
        if control_mode in (
            ProducerControlModeTag.OIL_RATE,
            ProducerControlModeTag.WATER_RATE,
            ProducerControlModeTag.GAS_RATE,
            ProducerControlModeTag.LIQUID_RATE,
            ProducerControlModeTag.RESERVOIR_VOLUME_RATE,
        ):
            bhp, connection_pressures, phase_rates, surface_phase_rates = solve_rate_producer(
                control_mode=control_mode,
                target_rate=controls.target_rates[well_row],
                tubing=tubing,
                reference_depth=reference_depth,
                workspace=perforation_workspace,
                control_spec=control_spec,
            )
        elif control_mode == ProducerControlModeTag.BHP:
            bhp, connection_pressures, phase_rates, surface_phase_rates = solve_bhp_producer(
                target_bhp=controls.target_bhps[well_row],
                tubing=tubing,
                reference_depth=reference_depth,
                workspace=perforation_workspace,
                control_spec=control_spec,
            )
        elif control_mode == ProducerControlModeTag.THP:
            if surface_sample is None:
                raise ValidationError("A THP-mode producer requires `surface_sample`.")
            if not tubing.can_evaluate_thp:
                raise ValidationError(
                    "A THP-mode producer requires a `WellBoreModel` or VFP table, "
                    "regardless of `control_spec.connection_pressure_mode`."
                )
            min_pressure, max_pressure = get_pressure_bracket(
                workspace=perforation_workspace, is_injector=False, control_spec=control_spec
            )
            bhp, _, _ = solve_bhp(
                tubing=tubing,
                reference_depth=reference_depth,
                workspace=perforation_workspace,
                relevant_phases=relevant_phases,
                is_injector=False,
                target=controls.target_thps[well_row],
                min_pressure=min_pressure,
                max_pressure=max_pressure,
                control_spec=control_spec,
                metric="thp",
                surface_sample=surface_sample,
            )
            connection_pressures, phase_rates, surface_phase_rates = (
                solve_connection_pressures_and_rates(
                    tubing=tubing,
                    reference_depth=reference_depth,
                    workspace=perforation_workspace,
                    reference_pressure=bhp,
                    relevant_phases=relevant_phases,
                    is_injector=False,
                    control_spec=control_spec,
                )
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
    min_pressure, max_pressure = get_pressure_bracket(
        workspace=perforation_workspace, is_injector=is_injector, control_spec=control_spec
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
        tubing=tubing,
        reference_depth=reference_depth,
        workspace=perforation_workspace,
        relevant_phases=relevant_phases,
        is_injector=is_injector,
        bhp=bhp,
        connection_pressures=connection_pressures,
        phase_rates=phase_rates,
        surface_phase_rates=surface_phase_rates,
        min_pressure=min_pressure,
        max_pressure=max_pressure,
        control_spec=control_spec,
        surface_sample=surface_sample,
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

    if tubing.vfp_table is not None or (
        tubing.wellbore is not None and surface_sample is not None
    ):
        workspace.thps[well_row] = compute_tubing_head_pressure(
            tubing=tubing,
            reference_depth=reference_depth,
            reference_pressure=bhp,
            phase_rates=phase_rates,
            surface_phase_rates=surface_phase_rates,
            surface_sample=surface_sample,
            is_injector=is_injector,
            connection_rates=ConnectionRates(
                oil=np.zeros_like(perforation_workspace.connection_oil_rates),
                water=np.zeros_like(perforation_workspace.connection_water_rates),
                gas=np.zeros_like(perforation_workspace.connection_gas_rates),
            )
            if economic_shutin
            else get_connection_rates(workspace=perforation_workspace),
        )
    return None
