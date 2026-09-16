"""Mode-specific well-control solvers: target -> nominal (bhp, phase_rates)."""

import typing

from bores.errors import ValidationError
from bores.types import Integer, Number, NumberArray, OneDimension
from bores.wells.compile import (
    FluidPhaseTag,
    InjectorControlModeTag,
    ProducerControlModeTag,
    RateQuantityTag,
)
from bores.wells.hydraulics.base import SurfaceFluidProperties, WellBoreModel
from bores.wells.resolution.compile import (
    PerforationWorkspace,
    accumulate_phase_rates,
    build_connection_phase_rates,
)
from bores.wells.resolution.spec import ControlResolverSpec
from bores.wells.states import ConnectionSample, PhaseValues

__all__ = [
    "PRODUCER_RATE_MODE_PHASES",
    "RATE_QUANTITY_PHASES",
    "bisect_bhp",
    "compute_phase_rates",
    "get_default_pressure_bracket",
    "phase_mask",
    "solve_connection_pressures_and_rates",
    "solve_injector_bhp_mode",
    "solve_injector_rate_mode",
    "solve_producer_bhp_mode",
    "solve_producer_rate_mode",
]


PRODUCER_RATE_MODE_PHASES: dict[Integer, PhaseValues] = {
    ProducerControlModeTag.OIL_RATE: PhaseValues(oil=1.0, water=0.0, gas=0.0),
    ProducerControlModeTag.WATER_RATE: PhaseValues(oil=0.0, water=1.0, gas=0.0),
    ProducerControlModeTag.GAS_RATE: PhaseValues(oil=0.0, water=0.0, gas=1.0),
    ProducerControlModeTag.LIQUID_RATE: PhaseValues(oil=1.0, water=1.0, gas=0.0),
    ProducerControlModeTag.RESERVOIR_VOLUME_RATE: PhaseValues(oil=1.0, water=1.0, gas=1.0),
}
"""Which phases count toward a producer's rate target, by `ProducerControlModeTag`."""

RATE_QUANTITY_PHASES: dict[Integer, PhaseValues] = {
    RateQuantityTag.OIL: PhaseValues(oil=1.0, water=0.0, gas=0.0),
    RateQuantityTag.WATER: PhaseValues(oil=0.0, water=1.0, gas=0.0),
    RateQuantityTag.GAS: PhaseValues(oil=0.0, water=0.0, gas=1.0),
    RateQuantityTag.LIQUID: PhaseValues(oil=1.0, water=1.0, gas=0.0),
    RateQuantityTag.RESERVOIR: PhaseValues(oil=1.0, water=1.0, gas=1.0),
}
"""Which phases a `RateLimit` sums over, by `RateQuantityTag`."""

ALL_PHASES = PhaseValues(oil=1.0, water=1.0, gas=1.0)
"""Every phase relevant - a producer's rate report includes oil, water, and gas."""


def phase_mask(phase_tag: Integer) -> PhaseValues:
    """
    Builds a single-phase relevant-phases mask from a `FluidPhaseTag`.

    :param phase_tag: `FluidPhaseTag` value, e.g. an injector's `injected_phase`.
    :returns: `PhaseValues` with `1.0` for `phase_tag`'s phase, `0.0` for the other two.
    """
    return PhaseValues(
        oil=1.0 if phase_tag == FluidPhaseTag.OIL else 0.0,
        water=1.0 if phase_tag == FluidPhaseTag.WATER else 0.0,
        gas=1.0 if phase_tag == FluidPhaseTag.GAS else 0.0,
    )


def compute_perforation_pressures(
    *, wellbore: WellBoreModel, **kwargs: typing.Any
) -> NumberArray[OneDimension]:
    """
    Dispatches to the `compute_perforation_pressures` of whichever module
    `wellbore.name` selects.

    :param wellbore: `WellBoreModel` naming a hydraulics correlation.
    :param kwargs: Forwarded to that correlation's `compute_perforation_pressures`.
    :returns: Pressure at each connection.
    :raises ValidationError: If `wellbore.name` isn't recognized.
    """
    if wellbore.name == "homogeneous":
        from bores.wells.hydraulics.homogeneous import (
            compute_perforation_pressures as compute,
        )

        return compute(model=wellbore.options, **kwargs)
    if wellbore.name in ("beggs_brill", "beggs_and_brill"):
        from bores.wells.hydraulics.beggs_and_brill import (
            compute_perforation_pressures as compute,
        )

        return compute(model=wellbore.options, **kwargs)
    if wellbore.name == "hagedorn_brown":
        from bores.wells.hydraulics.hagedorn_brown import (
            compute_perforation_pressures as compute,
        )

        return compute(model=wellbore.options, **kwargs)
    raise ValidationError(f"Unknown `WellBoreModel` name: {wellbore.name!r}")


def compute_tubing_head_pressure(*, wellbore: WellBoreModel, **kwargs: typing.Any) -> Number:
    """
    Dispatches to the `compute_tubing_head_pressure` of whichever module
    `wellbore.name` selects.

    :param wellbore: `WellBoreModel` naming a hydraulics correlation.
    :param kwargs: Forwarded to that correlation's `compute_tubing_head_pressure`.
    :returns: Tubing head pressure.
    :raises ValidationError: If `wellbore.name` isn't recognized.
    """
    if wellbore.name == "homogeneous":
        from bores.wells.hydraulics.homogeneous import (
            compute_tubing_head_pressure as compute,
        )

        return compute(model=wellbore.options, **kwargs)
    if wellbore.name in ("beggs_brill", "beggs_and_brill"):
        from bores.wells.hydraulics.beggs_and_brill import (
            compute_tubing_head_pressure as compute,
        )

        return compute(model=wellbore.options, **kwargs)
    if wellbore.name == "hagedorn_brown":
        from bores.wells.hydraulics.hagedorn_brown import (
            compute_tubing_head_pressure as compute,
        )

        return compute(model=wellbore.options, **kwargs)
    raise ValidationError(f"Unknown `WellBoreModel` name: {wellbore.name!r}")


def get_default_pressure_bracket(
    connection_samples: typing.Sequence[ConnectionSample],
    *,
    is_injector: bool,
    resolver_spec: ControlResolverSpec,
) -> tuple[Number, Number]:
    """
    Builds a default `(min_pressure, max_pressure)` BHP bisection bracket.

    A producer's BHP can't usefully exceed the highest connected-cell
    pressure as there's no drawdown left to give a rate above that. An
    injector's upper bound is reservoir pressure scaled by
    `resolver_spec.injector_bhp_bracket_multiplier`, giving room above
    reservoir pressure to actually inject.

    :param connection_samples: Reservoir samples for every connection.
    :param is_injector: Selects which bracket shape to use.
    :param resolver_spec: Supplies `producer_bhp_floor`/`injector_bhp_bracket_multiplier`.
    :returns: `(min_pressure, max_pressure)`.
    """
    cell_pressures = [sample.pressure for sample in connection_samples]
    if is_injector:
        return min(cell_pressures), resolver_spec.injector_bhp_bracket_multiplier * max(
            cell_pressures
        )
    return resolver_spec.producer_bhp_floor, max(cell_pressures)


def solve_connection_pressures_and_rates(
    *,
    wellbore: WellBoreModel,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    connection_samples: typing.Sequence[ConnectionSample],
    reference_pressure: Number,
    relevant_phases: PhaseValues,
    is_injector: bool,
    resolver_spec: ControlResolverSpec,
) -> tuple[NumberArray[OneDimension], PhaseValues, PhaseValues]:
    """
    Does fixed-point iterations of connection flowing pressures against IPR-derived
    rates at a fixed `reference_pressure` (BHP), until the well-total
    relevant-phase rate stabilizes.

    Starts from static (zero-rate) connection pressures, derives IPR rates
    from those, feeds the resulting `phase_rates` back through the
    hydraulics correlation for updated flowing pressures, and repeats
    until the relative change in total rate is within
    `resolver_spec.rate_convergence_tolerance`, or
    `resolver_spec.max_fixed_point_iterations` is reached (best-effort).
    `workspace.connection_pressures` is reused as the output buffer on
    every call rather than reallocated.

    :param wellbore: Hydraulics correlation for this well.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: This well's `PerforationWorkspace`.
    :param connection_samples: Reservoir samples, same order as `workspace`'s arrays.
    :param reference_pressure: BHP held fixed while iterating.
    :param relevant_phases: Mask of which phases to accumulate rates for.
    :param is_injector: Selects the drawdown sign convention.
    :param resolver_spec: Supplies the iteration cap and convergence tolerance.
    :returns: `(connection_pressures, phase_rates, surface_phase_rates)`.
    """
    zero_rates = PhaseValues(oil=0.0, water=0.0, gas=0.0)
    n_connections = len(connection_samples)
    zero_connection_phase_rates = [zero_rates] * n_connections
    connection_pressures = compute_perforation_pressures(
        wellbore=wellbore,
        reference_depth=reference_depth,
        reference_pressure=reference_pressure,
        connection_phase_rates=zero_connection_phase_rates,
        representative_depths=workspace.representative_depths,
        inclinations_from_vertical=workspace.inclinations_from_vertical,
        connection_samples=connection_samples,
        is_injector=is_injector,
        out=workspace.connection_pressures,
    )
    total_rate = 0.0
    phase_rates = zero_rates
    surface_phase_rates = zero_rates

    relevant_oil = relevant_phases.oil > 0.0
    relevant_water = relevant_phases.water > 0.0
    relevant_gas = relevant_phases.gas > 0.0

    for _ in range(resolver_spec.max_fixed_point_iterations):
        (
            oil_rate,
            water_rate,
            gas_rate,
            surface_oil_rate,
            surface_water_rate,
            surface_gas_rate,
        ) = accumulate_phase_rates(
            connection_pressures=connection_pressures,
            well_indices=workspace.well_indices,
            reservoir_pressures=workspace.reservoir_pressures,
            oil_mobilities=workspace.oil_mobilities,
            water_mobilities=workspace.water_mobilities,
            gas_mobilities=workspace.gas_mobilities,
            oil_formation_volume_factors=workspace.oil_formation_volume_factors,
            water_formation_volume_factors=workspace.water_formation_volume_factors,
            gas_formation_volume_factors=workspace.gas_formation_volume_factors,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            out_connection_oil_rates=workspace.connection_oil_rates,
            out_connection_water_rates=workspace.connection_water_rates,
            out_connection_gas_rates=workspace.connection_gas_rates,
        )
        phase_rates = PhaseValues(oil=oil_rate, water=water_rate, gas=gas_rate)
        surface_phase_rates = PhaseValues(
            oil=surface_oil_rate,
            water=surface_water_rate,
            gas=surface_gas_rate,
        )
        new_total_rate = oil_rate + water_rate + gas_rate
        connection_phase_rates = build_connection_phase_rates(
            connection_oil_rates=workspace.connection_oil_rates,
            connection_water_rates=workspace.connection_water_rates,
            connection_gas_rates=workspace.connection_gas_rates,
        )
        connection_pressures = compute_perforation_pressures(
            wellbore=wellbore,
            reference_depth=reference_depth,
            reference_pressure=reference_pressure,
            connection_phase_rates=connection_phase_rates,
            representative_depths=workspace.representative_depths,
            inclinations_from_vertical=workspace.inclinations_from_vertical,
            connection_samples=connection_samples,
            is_injector=is_injector,
            out=workspace.connection_pressures,
        )
        if abs(new_total_rate - total_rate) <= resolver_spec.rate_convergence_tolerance * max(
            abs(new_total_rate), 1.0
        ):
            total_rate = new_total_rate
            break
        total_rate = new_total_rate

    return connection_pressures, phase_rates, surface_phase_rates


def compute_phase_rates(
    *,
    wellbore: WellBoreModel,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    connection_samples: typing.Sequence[ConnectionSample],
    reference_pressure: Number,
    relevant_phases: PhaseValues,
    is_injector: bool,
    resolver_spec: ControlResolverSpec,
) -> tuple[NumberArray[OneDimension], PhaseValues, PhaseValues]:
    """
    `solve_connection_pressures_and_rates` at a fixed BHP.

    For a producer, callers pass `relevant_phases=ALL_PHASES` - a producer
    can legitimately flow oil, water, and gas simultaneously. For an
    injector, `relevant_phases` must be `phase_mask(injected_phase)` only -
    an injector displaces reservoir fluid rather than having an
    IPR-driven rate of its own for the other two phases.

    :param wellbore: Hydraulics correlation for this well.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: This well's `PerforationWorkspace`.
    :param connection_samples: Reservoir samples, same order as `workspace`'s arrays.
    :param reference_pressure: BHP to report rates at.
    :param relevant_phases: Mask of which phases to compute a real rate for.
    :param is_injector: Selects the drawdown sign convention.
    :param resolver_spec: Forwarded to `solve_connection_pressures_and_rates`.
    :returns: `(connection_pressures, phase_rates, surface_phase_rates)` -
        flowing pressure at each connection, reservoir-condition phase
        rates, and surface-condition phase rates.
    """
    return solve_connection_pressures_and_rates(
        wellbore=wellbore,
        reference_depth=reference_depth,
        workspace=workspace,
        connection_samples=connection_samples,
        reference_pressure=reference_pressure,
        relevant_phases=relevant_phases,
        is_injector=is_injector,
        resolver_spec=resolver_spec,
    )


def bisect_bhp(
    *,
    wellbore: WellBoreModel,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    connection_samples: typing.Sequence[ConnectionSample],
    relevant_phases: PhaseValues,
    is_injector: bool,
    target: Number,
    min_pressure: Number,
    max_pressure: Number,
    resolver_spec: ControlResolverSpec,
    metric: typing.Literal["rate", "thp"] = "rate",
    target_rate_condition: typing.Literal["surface", "reservoir"] = "surface",
    surface_fluid_properties: SurfaceFluidProperties | None = None,
) -> tuple[Number, NumberArray[OneDimension], PhaseValues]:
    """
    Bisects `reference_pressure` in `[min_pressure, max_pressure]` until
    the resulting metric (`relevant_phases` total rate, or THP) matches `target`.

    Both metrics are monotonic in BHP, in a direction that depends on the
    metric and (for rate) the well type. For `metric="rate"`, lower BHP means
    a higher producer rate, higher BHP means a higher injector rate;
    For `metric="thp"`, higher BHP means higher THP, for both well types.

    `min_pressure`/`max_pressure` must bracket the target. Callers should pick
    these via `get_default_pressure_bracket` (or their own). Best-effort
    (no exception) if `target` isn't achievable within the bracket.
    Returns the closest bound reached after `resolver_spec.max_bisection_iterations`.

    :param wellbore: Hydraulics correlation for this well.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: This well's `PerforationWorkspace`.
    :param connection_samples: Reservoir samples, same order as `workspace`'s arrays.
    :param relevant_phases: Mask the `"rate"` metric sums over.
    :param is_injector: Selects the drawdown sign convention and the
        direction `"rate"` moves in as BHP changes.
    :param target: Value the chosen metric is bisected toward.
    :param min_pressure: Lower bracket bound.
    :param max_pressure: Upper bracket bound.
    :param resolver_spec: Supplies the iteration cap and convergence tolerance.
    :param metric: `"rate"` (default) matches `relevant_phases`' total
        rate against `target`; `"thp"` matches tubing head pressure
        against `target` instead (requires `surface_fluid_properties`).
    :param target_rate_condition: `"surface"` (default) bisects against
        the surface-condition total; `"reservoir"` bisects against the
        raw reservoir-condition total instead. Ignored when `metric="thp"`.
    :param surface_fluid_properties: Required if `metric="thp"`.
    :returns: `(bhp, connection_pressures, phase_rates)`, `phase_rates`
        always reservoir-condition regardless of `target_rate_condition`.
    :raises ValidationError: If `metric="thp"` and `surface_fluid_properties` is `None`.
    """
    if metric == "thp" and surface_fluid_properties is None:
        raise ValidationError("bisect_bhp(..., metric='thp') requires `surface_fluid_properties`.")

    low, high = min_pressure, max_pressure
    average_pressure = 0.5 * (low + high)
    pressures = None
    phase_rates = PhaseValues(oil=0.0, water=0.0, gas=0.0)

    for _ in range(resolver_spec.max_bisection_iterations):
        average_pressure = 0.5 * (low + high)
        pressures, phase_rates, surface_phase_rates = solve_connection_pressures_and_rates(
            wellbore=wellbore,
            reference_depth=reference_depth,
            workspace=workspace,
            connection_samples=connection_samples,
            reference_pressure=average_pressure,
            relevant_phases=relevant_phases,
            is_injector=is_injector,
            resolver_spec=resolver_spec,
        )

        if metric == "rate":
            target_rates = (
                phase_rates if target_rate_condition == "reservoir" else surface_phase_rates
            )
            value = target_rates.oil + target_rates.water + target_rates.gas
            increasing_with_bhp = is_injector
        else:
            assert surface_fluid_properties is not None
            value = compute_tubing_head_pressure(
                wellbore=wellbore,
                reference_depth=reference_depth,
                reference_pressure=average_pressure,
                phase_rates=phase_rates,
                surface_fluid_properties=surface_fluid_properties,
                is_injector=is_injector,
            )
            increasing_with_bhp = True

        if abs(value - target) <= resolver_spec.rate_convergence_tolerance * max(abs(target), 1.0):
            break

        if (value < target) == increasing_with_bhp:
            low = average_pressure
        else:
            high = average_pressure

    assert pressures is not None
    return average_pressure, pressures, phase_rates


def solve_producer_rate_mode(
    *,
    control_mode: Integer,
    target_rate: Number,
    wellbore: WellBoreModel,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    connection_samples: typing.Sequence[ConnectionSample],
    resolver_spec: ControlResolverSpec,
) -> tuple[Number, NumberArray[OneDimension], PhaseValues, PhaseValues]:
    """
    Resolves a producer under `control_mode in {ORAT, WRAT, GRAT, LRAT, RESV}`.

    Nominal resolution only. Does not check limits; the caller applies
    those afterward via `wells.resolution.limits.apply_limits`.

    :param control_mode: `ProducerControlModeTag` value.
    :param target_rate: The rate target.
    :param wellbore: Hydraulics correlation for this well.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: This well's `PerforationWorkspace`.
    :param connection_samples: Reservoir samples, same order as `workspace`'s arrays.
    :param resolver_spec: Solver tunables.
    :returns: `(bhp, connection_pressures, phase_rates, surface_phase_rates)`,
        reservoir- and surface-condition rates for every phase.
    """
    relevant_phases = PRODUCER_RATE_MODE_PHASES[control_mode]
    min_pressure, max_pressure = get_default_pressure_bracket(
        connection_samples, is_injector=False, resolver_spec=resolver_spec
    )
    bhp, _, _ = bisect_bhp(
        wellbore=wellbore,
        reference_depth=reference_depth,
        workspace=workspace,
        connection_samples=connection_samples,
        relevant_phases=relevant_phases,
        is_injector=False,
        target=target_rate,
        min_pressure=min_pressure,
        max_pressure=max_pressure,
        resolver_spec=resolver_spec,
        target_rate_condition="reservoir"
        if control_mode == ProducerControlModeTag.RESERVOIR_VOLUME_RATE
        else "surface",
    )
    connection_pressures, phase_rates, surface_phase_rates = compute_phase_rates(
        wellbore=wellbore,
        reference_depth=reference_depth,
        workspace=workspace,
        connection_samples=connection_samples,
        reference_pressure=bhp,
        relevant_phases=ALL_PHASES,
        is_injector=False,
        resolver_spec=resolver_spec,
    )
    return bhp, connection_pressures, phase_rates, surface_phase_rates


def solve_producer_bhp_mode(
    *,
    target_bhp: Number,
    wellbore: WellBoreModel,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    connection_samples: typing.Sequence[ConnectionSample],
    resolver_spec: ControlResolverSpec,
) -> tuple[Number, NumberArray[OneDimension], PhaseValues, PhaseValues]:
    """
    Resolves a producer held at a fixed BHP; rates are the output.

    Nominal resolution only. See `solve_producer_rate_mode`.
    """
    connection_pressures, phase_rates, surface_phase_rates = compute_phase_rates(
        wellbore=wellbore,
        reference_depth=reference_depth,
        workspace=workspace,
        connection_samples=connection_samples,
        reference_pressure=target_bhp,
        relevant_phases=ALL_PHASES,
        is_injector=False,
        resolver_spec=resolver_spec,
    )
    return target_bhp, connection_pressures, phase_rates, surface_phase_rates


def solve_injector_rate_mode(
    *,
    control_mode: Integer,
    target_rate: Number,
    injected_phase: Integer,
    wellbore: WellBoreModel,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    connection_samples: typing.Sequence[ConnectionSample],
    resolver_spec: ControlResolverSpec,
) -> tuple[Number, NumberArray[OneDimension], PhaseValues, PhaseValues]:
    """
    Injector analogue of `solve_producer_rate_mode` - `injected_phase`
    selects the single phase being allocated.

    Nominal resolution only. See `solve_producer_rate_mode`.

    :param control_mode: `InjectorControlModeTag` value.
    :param target_rate: The rate target.
    :param injected_phase: `FluidPhaseTag` value of the injected phase.
    :param wellbore: Hydraulics correlation for this well.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: This well's `PerforationWorkspace`.
    :param connection_samples: Reservoir samples, same order as `workspace`'s arrays.
    :param resolver_spec: Solver tunables.
    :returns: `(bhp, connection_pressures, phase_rates, surface_phase_rates)` -
        reservoir- and surface-condition rates, `injected_phase` only nonzero.
    """
    relevant_phases = phase_mask(injected_phase)
    min_pressure, max_pressure = get_default_pressure_bracket(
        connection_samples, is_injector=True, resolver_spec=resolver_spec
    )
    bhp, _, _ = bisect_bhp(
        wellbore=wellbore,
        reference_depth=reference_depth,
        workspace=workspace,
        connection_samples=connection_samples,
        relevant_phases=relevant_phases,
        is_injector=True,
        target=target_rate,
        min_pressure=min_pressure,
        max_pressure=max_pressure,
        resolver_spec=resolver_spec,
        target_rate_condition="reservoir"
        if control_mode == InjectorControlModeTag.RESERVOIR_VOLUME_RATE
        else "surface",
    )
    connection_pressures, phase_rates, surface_phase_rates = compute_phase_rates(
        wellbore=wellbore,
        reference_depth=reference_depth,
        workspace=workspace,
        connection_samples=connection_samples,
        reference_pressure=bhp,
        relevant_phases=relevant_phases,
        is_injector=True,
        resolver_spec=resolver_spec,
    )
    return bhp, connection_pressures, phase_rates, surface_phase_rates


def solve_injector_bhp_mode(
    *,
    target_bhp: Number,
    injected_phase: Integer,
    wellbore: WellBoreModel,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    connection_samples: typing.Sequence[ConnectionSample],
    resolver_spec: ControlResolverSpec,
) -> tuple[Number, NumberArray[OneDimension], PhaseValues, PhaseValues]:
    """
    Injector analogue of `solve_producer_bhp_mode`.

    Nominal resolution only. See `solve_producer_rate_mode`.
    """
    relevant_phases = phase_mask(injected_phase)
    connection_pressures, phase_rates, surface_phase_rates = compute_phase_rates(
        wellbore=wellbore,
        reference_depth=reference_depth,
        workspace=workspace,
        connection_samples=connection_samples,
        reference_pressure=target_bhp,
        relevant_phases=relevant_phases,
        is_injector=True,
        resolver_spec=resolver_spec,
    )
    return target_bhp, connection_pressures, phase_rates, surface_phase_rates
