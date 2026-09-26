"""Secondary-limit enforcement for a resolved well control."""

import math
import typing

from bores.errors import ValidationError
from bores.types import Integer, Number, NumberArray, OneDimension
from bores.wells.compile import (
    UNSET_INT,
    CompiledGroupLimits,
    CompiledLimits,
    EconomicQuantityTag,
    LimitKind,
    RateQuantityTag,
)
from bores.wells.hydraulics.base import SurfaceFluidProperties, WellBoreModel
from bores.wells.resolution.solvers import (
    RATE_QUANTITY_PHASES,
    bisect_bhp,
    compute_phase_rates,
    compute_tubing_head_pressure,
    solve_connection_pressures_and_rates,
)
from bores.wells.resolution.spec import WellControlSpec
from bores.wells.state import ConnectionSample, PhaseValues
from bores.wells.workspace import PerforationWorkspace

__all__ = ["apply_limits"]


def get_bhp_bound(
    *, min_value: Number, max_value: Number, bhp: Number, is_injector: bool
) -> Number | None:
    """
    Gets the bounding BHP for a `BHPLimit` row, if violated.

    :param min_value: The limit row's `min_value`. `NaN` means no floor.
    :param max_value: The limit row's `max_value`. `NaN` means no ceiling.
    :param bhp: The nominal resolution's BHP.
    :param is_injector: Whether this well is an injector.
    :returns: The violated bound's value, or `None` if not violated.
    """
    if is_injector:
        if not math.isnan(max_value) and bhp > max_value:
            return max_value
        return None
    if not math.isnan(min_value) and bhp < min_value:
        return min_value
    return None


def get_rate_bound(
    *,
    quantity: Integer,
    max_value: Number,
    bhp: Number,
    wellbore: WellBoreModel | None,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    connection_samples: typing.Sequence[ConnectionSample],
    is_injector: bool,
    min_pressure: Number,
    max_pressure: Number,
    control_spec: WellControlSpec,
) -> Number | None:
    """
    Gets the bounding BHP for a `RateLimit` row, if violated.

    Found by bisecting BHP against `max_value` as a rate target, against
    the reservoir-condition total for a `RESERVOIR`-quantity limit, the
    surface-condition total otherwise.

    The violation check itself recomputes whichever total is relevant at `bhp`
    rather than relying on an already-computed reservoir-condition total, since
    a `RateLimit` on a surface quantity needs the surface-condition total to
    check correctly.

    :param quantity: `RateQuantityTag` value.
    :param max_value: The limit row's `max_value`.
    :param bhp: The nominal resolution's BHP.
    :param wellbore: Hydraulics correlation for this well. May be `None`.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: This well's `PerforationWorkspace`.
    :param connection_samples: Reservoir samples, same order as `workspace`'s arrays.
    :param is_injector: Whether this well is an injector.
    :param min_pressure: Lower bisection bracket bound.
    :param max_pressure: Upper bisection bracket bound.
    :param control_spec: Solver tunables.
    :returns: Bounding BHP, or `None` if not violated.
    """
    quantity_phases = RATE_QUANTITY_PHASES[quantity]
    target_rate_condition: typing.Literal["surface", "reservoir"] = (
        "reservoir" if quantity == RateQuantityTag.RESERVOIR else "surface"
    )

    _, reservoir_condition_rates, surface_condition_rates = solve_connection_pressures_and_rates(
        wellbore=wellbore,
        reference_depth=reference_depth,
        workspace=workspace,
        connection_samples=connection_samples,
        reference_pressure=bhp,
        relevant_phases=quantity_phases,
        is_injector=is_injector,
        control_spec=control_spec,
    )
    rates_to_check = (
        reservoir_condition_rates
        if target_rate_condition == "reservoir"
        else surface_condition_rates
    )
    current = rates_to_check.oil + rates_to_check.water + rates_to_check.gas
    if current <= max_value:
        return None

    bound_bhp, _, _ = bisect_bhp(
        wellbore=wellbore,
        reference_depth=reference_depth,
        workspace=workspace,
        connection_samples=connection_samples,
        relevant_phases=quantity_phases,
        is_injector=is_injector,
        target=max_value,
        min_pressure=min_pressure,
        max_pressure=max_pressure,
        control_spec=control_spec,
        metric="rate",
        target_rate_condition=target_rate_condition,
    )
    return bound_bhp


def get_thp_bound(
    *,
    min_value: Number,
    max_value: Number,
    bhp: Number,
    phase_rates: PhaseValues,
    wellbore: WellBoreModel,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    connection_samples: typing.Sequence[ConnectionSample],
    relevant_phases: PhaseValues,
    is_injector: bool,
    min_pressure: Number,
    max_pressure: Number,
    control_spec: WellControlSpec,
    surface_fluid_properties: SurfaceFluidProperties,
) -> Number | None:
    """
    Gets the bounding BHP for a `THPLimit` row, if violated.

    Checked in the forward direction only (candidate BHP -> tubing head
    pressure), bisected against `min_value`/`max_value` as a target.

    :param min_value: The limit row's `min_value`. `NaN` means no floor.
    :param max_value: The limit row's `max_value`. `NaN` means no ceiling.
    :param bhp: The nominal resolution's BHP.
    :param phase_rates: The nominal resolution's reservoir-condition phase rates.
    :param wellbore: Hydraulics correlation for this well. Required as a
        THP limit can't be evaluated without one, regardless of
        `control_spec.connection_pressure_mode`.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: This well's `PerforationWorkspace`.
    :param connection_samples: Reservoir samples, same order as `workspace`'s arrays.
    :param relevant_phases: Mask the `"rate"` metric would sum over
        (unused for the THP check itself; forwarded to `bisect_bhp`).
    :param is_injector: Whether this well is an injector.
    :param min_pressure: Lower bisection bracket bound.
    :param max_pressure: Upper bisection bracket bound.
    :param control_spec: Solver tunables.
    :param surface_fluid_properties: Fluid properties at surface conditions.
    :returns: Bounding BHP, or `None` if not violated.
    :raises ValidationError: If `wellbore` is `None`.
    """
    current_thp = compute_tubing_head_pressure(
        wellbore=wellbore,
        reference_depth=reference_depth,
        reference_pressure=bhp,
        phase_rates=phase_rates,
        surface_fluid_properties=surface_fluid_properties,
        is_injector=is_injector,
    )
    if is_injector:
        violated = not math.isnan(max_value) and current_thp > max_value
        target = max_value
    else:
        violated = not math.isnan(min_value) and current_thp < min_value
        target = min_value
    if not violated:
        return None

    bound_bhp, _, _ = bisect_bhp(
        wellbore=wellbore,
        reference_depth=reference_depth,
        workspace=workspace,
        connection_samples=connection_samples,
        relevant_phases=relevant_phases,
        is_injector=is_injector,
        target=target,
        min_pressure=min_pressure,
        max_pressure=max_pressure,
        control_spec=control_spec,
        metric="thp",
        surface_fluid_properties=surface_fluid_properties,
    )
    return bound_bhp


def check_economic_violation(
    *,
    limits: CompiledLimits | CompiledGroupLimits,
    limits_start: Integer,
    limits_end: Integer,
    phase_rates: PhaseValues,
) -> Integer:
    """
    Finds the first `ECONOMIC` limit row in `[limits_start, limits_end)`
    that's currently violated.

    `WATER_CUT`, `GOR`, and `WATER_GAS_RATIO` are violated once their
    ratio rises above `max_values[row]`. `OIL_RATE` and `GAS_RATE` are
    violated once that rate falls below `min_values[row]`, since they
    represent a minimum economic rate rather than a ceiling.

    :param limits: The full system's `CompiledLimits`.
    :param limits_start: First row of this well's limit range.
    :param limits_end: One past the last row of this well's limit range.
    :param phase_rates: Well-total phase rates to check.
    :returns: The violated row's index, or `UNSET_INT` if none is violated.
    """
    oil, water, gas = phase_rates.oil, phase_rates.water, phase_rates.gas

    for row in range(limits_start, limits_end):
        if limits.kinds[row] != LimitKind.ECONOMIC:
            continue
        quantity = limits.quantities[row]
        if quantity == EconomicQuantityTag.OIL_RATE:
            if oil < limits.min_values[row]:
                return row
            continue
        if quantity == EconomicQuantityTag.GAS_RATE:
            if gas < limits.min_values[row]:
                return row
            continue
        if quantity == EconomicQuantityTag.WATER_CUT:
            ratio = water / (oil + water) if (oil + water) > 0 else 0.0
        elif quantity == EconomicQuantityTag.GOR:
            ratio = (gas / oil) if oil > 0 else (math.inf if gas > 0 else 0.0)
        else:
            ratio = (water / gas) if gas > 0 else (math.inf if water > 0 else 0.0)
        if ratio > limits.max_values[row]:
            return row
    return UNSET_INT


def apply_limits(
    *,
    limits: CompiledLimits,
    limits_start: Integer,
    limits_end: Integer,
    wellbore: WellBoreModel | None,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    connection_samples: typing.Sequence[ConnectionSample],
    relevant_phases: PhaseValues,
    is_injector: bool,
    bhp: Number,
    connection_pressures: NumberArray[OneDimension],
    phase_rates: PhaseValues,
    surface_phase_rates: PhaseValues,
    min_pressure: Number,
    max_pressure: Number,
    control_spec: WellControlSpec,
    surface_fluid_properties: SurfaceFluidProperties | None = None,
) -> tuple[Number, NumberArray[OneDimension], PhaseValues, PhaseValues, Integer, bool]:
    """
    Evaluates every limit row in `[limits_start, limits_end)` against a
    nominal resolution. If one or more are violated, re-resolves at
    whichever implies the most restrictive BHP and returns that as the
    governing resolution.

    "Most restrictive wins", not "first in list wins": every limit kind
    here reduces to "the BHP that exactly satisfies this limit", a scalar
    on the same axis every mode solver already searches. Because total
    rate and THP are both monotonic in BHP, the single most extreme
    required BHP among every violated limit automatically satisfies every
    other violated limit too so no outer iteration needed.

    Returns `(bhp, connection_pressures, phase_rates, surface_phase_rates,
    UNSET_INT, False)` unchanged if nothing is violated.

    Called for every control mode (rate and BHP alike) because a well held at a
    fixed BHP can still violate a `RateLimit`/`THPLimit` configured
    alongside that BHP target, so this isn't skipped for BHP-mode resolutions.

    :param limits: The full system's `CompiledLimits`.
    :param limits_start: First row of this well's limit range.
    :param limits_end: One past the last row of this well's limit range.
    :param wellbore: Hydraulics correlation for this well. May be `None`. 
        A `THPLimit` row in range still requires one regardless.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: This well's `PerforationWorkspace`.
    :param connection_samples: Reservoir samples, same order as `workspace`'s arrays.
    :param relevant_phases: `solvers.ALL_PHASES` for a producer;
        `solvers.phase_mask(injected_phase)` for an injector.
    :param is_injector: Whether this well is an injector.
    :param bhp: The nominal resolution's BHP.
    :param connection_pressures: The nominal resolution's flowing pressure
        at each connection, same order as `workspace`'s arrays.
    :param phase_rates: The nominal resolution's reservoir-condition phase rates.
    :param surface_phase_rates: The nominal resolution's surface-condition phase rates.
    :param min_pressure: Lower bisection bracket bound.
    :param max_pressure: Upper bisection bracket bound.
    :param control_spec: Solver tunables.
    :param surface_fluid_properties: Required if any row in range is a `THPLimit`.
    :returns: `(bhp, connection_pressures, phase_rates, surface_phase_rates,
        active_limit_row, economic_shutin)`. `active_limit_row` is
        `UNSET_INT` if nothing is binding. `phase_rates`/`surface_phase_rates`
        are zeroed if `economic_shutin` is `True`; `connection_pressures`
        is not, as a shut-in well still has a real wellbore pressure profile.
    :raises ValidationError: If a `THPLimit` row is present but
        `surface_fluid_properties` wasn't supplied, or `wellbore` is `None`.
    """
    candidates: list[tuple[Number, Integer]] = []

    for row in range(limits_start, limits_end):
        kind = limits.kinds[row]
        if kind == LimitKind.ECONOMIC:
            continue

        if kind == LimitKind.BHP:
            bound = get_bhp_bound(
                min_value=limits.min_values[row],
                max_value=limits.max_values[row],
                bhp=bhp,
                is_injector=is_injector,
            )
        elif kind == LimitKind.RATE:
            bound = get_rate_bound(
                quantity=limits.quantities[row],
                max_value=limits.max_values[row],
                bhp=bhp,
                wellbore=wellbore,
                reference_depth=reference_depth,
                workspace=workspace,
                connection_samples=connection_samples,
                is_injector=is_injector,
                min_pressure=min_pressure,
                max_pressure=max_pressure,
                control_spec=control_spec,
            )
        elif kind == LimitKind.THP:
            if surface_fluid_properties is None:
                raise ValidationError(
                    "A THP limit is present but no `surface_fluid_properties` "
                    "was supplied to `apply_limits`. THP limits can't be "
                    "evaluated without it."
                )
            if wellbore is None:
                raise ValidationError(
                    "A THP limit is present but no `WellBoreModel` or VFP table "
                    "is assigned to this well. THP limits can't be evaluated "
                    "without one, regardless of "
                    "`control_spec.connection_pressure_mode`."
                )
            bound = get_thp_bound(
                min_value=limits.min_values[row],
                max_value=limits.max_values[row],
                bhp=bhp,
                phase_rates=phase_rates,
                wellbore=wellbore,
                reference_depth=reference_depth,
                workspace=workspace,
                connection_samples=connection_samples,
                relevant_phases=relevant_phases,
                is_injector=is_injector,
                min_pressure=min_pressure,
                max_pressure=max_pressure,
                control_spec=control_spec,
                surface_fluid_properties=surface_fluid_properties,
            )
        else:
            raise ValidationError(f"Unknown `LimitKind`: {kind!r}.")

        if bound is not None:
            candidates.append((bound, row))

    if not candidates:
        governing_bhp, governing_row = bhp, UNSET_INT
        governing_connection_pressures = connection_pressures
        governing_phase_rates = phase_rates
        governing_surface_phase_rates = surface_phase_rates
    else:
        # Most restrictive: highest bounding BHP for a producer (lowest
        # resulting rate), lowest bounding BHP for an injector (lowest
        # resulting injection rate).
        governing_bhp, governing_row = (
            min(candidates, key=lambda pair: pair[0])
            if is_injector
            else max(candidates, key=lambda pair: pair[0])
        )
        governing_connection_pressures, governing_phase_rates, governing_surface_phase_rates = (
            compute_phase_rates(
                wellbore=wellbore,
                reference_depth=reference_depth,
                workspace=workspace,
                connection_samples=connection_samples,
                reference_pressure=governing_bhp,
                relevant_phases=relevant_phases,
                is_injector=is_injector,
                control_spec=control_spec,
            )
        )

    # Always checked, even when nothing else in range produced a
    # candidate bound. Beacause a well can be perfectly on target for BHP/rate/THP
    # and still be economically unviable, so this isn't conditional on
    # another limit having been the one to settle `governing_bhp`.
    violated_row = check_economic_violation(
        limits=limits,
        limits_start=limits_start,
        limits_end=limits_end,
        phase_rates=governing_phase_rates,
    )
    if violated_row != UNSET_INT:
        zero = PhaseValues(oil=0.0, water=0.0, gas=0.0)
        return (
            governing_bhp,
            governing_connection_pressures,
            zero,
            zero,
            violated_row,
            True,
        )
    return (
        governing_bhp,
        governing_connection_pressures,
        governing_phase_rates,
        governing_surface_phase_rates,
        governing_row,
        False,
    )
