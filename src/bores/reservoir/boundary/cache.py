"""Per-iteration boundary condition evaluation and per-timestep commit."""

import typing

import numba
import numpy as np
import numpy.typing as npt

from bores.precision import get_dtype
from bores.reservoir.boundary.aquifers import carter_tracy, fetkovich
from bores.reservoir.boundary.compile import (
    CARTER_TRACY_AQUIFER_KIND,
    CompiledAquifers,
    CompiledBoundaryConditions,
    CompiledProductivityIndices,
)
from bores.reservoir.boundary.workspace import AquiferWorkspace
from bores.types import BooleanArray, CellArray, Integer, Number, NumberArray, OneDimension

__all__ = ["BoundaryCache", "commit_boundary_conditions", "compute_boundary_cache"]


class BoundaryCache(typing.NamedTuple):
    """One iteration's worth of per-boundary-face solver input."""

    pressure_values: NumberArray[OneDimension]
    """Shape `(n_boundary_faces,)`. Prescribed pressure at Dirichlet faces; `0.0` elsewhere."""

    flux_values: NumberArray[OneDimension]
    """Shape `(n_boundary_faces,)`. Prescribed flux at Neumann/Robin faces; `0.0` elsewhere."""

    is_dirichlet: BooleanArray[OneDimension]
    """Shape `(n_boundary_faces,)`. `True` where a `PRESSURE`-type condition is active."""

    flux_derivatives: NumberArray[OneDimension]
    """
    Shape `(n_boundary_faces,)`. Derivative of each face's flux with respect to the pressure of
    the cell that owns it, for the Jacobian diagonal. `-PI` at a `ProductivityIndexBoundary`
    face, `w_i² · aquifer_rate_derivatives[row]` at an aquifer face, `0.0` elsewhere.
    """

    aquifer_rate_derivatives: NumberArray[OneDimension]
    """
    Shape `(n_aquifers,)`. Derivative of each aquifer's total rate with respect to its boundary
    pressure (the face-weighted mean of its owner cells' pressures). An aquifer's face `i`
    depends on every owner cell `j` of that aquifer through
    `d flux_i / d p_j = w_i · aquifer_rate_derivatives[row] · w_j`, which is the coupling
    between cells that `flux_derivatives` leaves out.
    """


@numba.njit(cache=True, parallel=True)
def apply_productivity_index_values(
    productivity_indices: CompiledProductivityIndices,
    pressure: CellArray,
    out_flux_values: NumberArray[OneDimension],
    out_flux_derivatives: NumberArray[OneDimension],
) -> None:
    """
    Writes every `ProductivityIndexBoundary` face's flux and its derivative directly into
    the output arrays, in place.

    :param productivity_indices: Every PI region's faces and parameters, whole.
    :param pressure: Current (trial or accepted) cell pressures.
    :param out_flux_values: The full-length flux array, written into at
        `productivity_indices.face_positions`.
    :param out_flux_derivatives: The full-length flux derivative array, written the same way.
    """
    owner_cells = productivity_indices.owner_cells
    face_positions = productivity_indices.face_positions
    row_of_face = productivity_indices.row_of_face
    pressure_boundary = productivity_indices.pressure_boundary
    productivity_index = productivity_indices.productivity_index

    n_faces = owner_cells.shape[0]
    for i in numba.prange(n_faces):
        row = row_of_face[i]
        out_flux_values[face_positions[i]] = productivity_index[row] * (
            pressure_boundary[row] - pressure[owner_cells[i]]
        )
        out_flux_derivatives[face_positions[i]] = -productivity_index[row]


@numba.njit(cache=True)
def evaluate_aquifer_step(
    aquifers: CompiledAquifers,
    workspace: AquiferWorkspace,
    row: Integer,
    boundary_pressure: Number,
    time: Number,
) -> tuple[Number, Number, Number, Number]:
    """
    One aquifer's recursion step from its last committed state to `time`, for a given
    boundary pressure. Reads the workspace and writes nothing, so it serves both the trial
    evaluation inside an iteration and the commit of an accepted time step.

    :param aquifers: Every aquifer's static parameters, whole.
    :param workspace: This run's `AquiferWorkspace`, read only.
    :param row: The aquifer's row.
    :param boundary_pressure: The aquifer's current boundary pressure.
    :param time: Time being solved for or committed to.
    :returns: `(cumulative_influx, aquifer_pressure, dimensionless_time, influx_derivative)`,
        where the derivative is `d cumulative_influx / d boundary_pressure`. No time has
        elapsed since the last committed state when `time` is not past it, and then the
        influx is unchanged and the derivative is zero.
    """
    previous_cumulative_influx = workspace.previous_cumulative_influx[row]
    previous_aquifer_pressure = workspace.previous_aquifer_pressure[row]
    previous_dimensionless_time = workspace.previous_dimensionless_time[row]
    elapsed_time = time - workspace.previous_time[row]
    if elapsed_time <= 0.0:
        return (
            previous_cumulative_influx,
            previous_aquifer_pressure,
            previous_dimensionless_time,
            0.0,
        )

    if aquifers.kinds[row] == CARTER_TRACY_AQUIFER_KIND:
        start = aquifers.influence_table_offsets[row]
        end = aquifers.influence_table_offsets[row + 1]
        current_dimensionless_time = aquifers.dimensionless_time_scales[row] * (
            time - workspace.origin_time[row]
        )
        pressure_drop = aquifers.initial_pressures[row] - boundary_pressure
        new_cumulative_influx, influx_per_pressure_drop = carter_tracy.compute_incremental_influx(
            previous_cumulative_influx=previous_cumulative_influx,
            previous_dimensionless_time=previous_dimensionless_time,
            current_dimensionless_time=current_dimensionless_time,
            current_pressure_drop=pressure_drop,
            aquifer_constant=aquifers.aquifer_constants[row],
            bounded=aquifers.bounded[row],
            dimensionless_radius_ratio=aquifers.dimensionless_radius_ratios[row],
            betas=aquifers.bessel_roots[row],
            pd_coefficients=aquifers.pd_coefficients[row],
            pd_prime_coefficients=aquifers.pd_prime_coefficients[row],
            linear_coefficient=aquifers.linear_coefficients[row],
            constant_coefficient=aquifers.constant_coefficients[row],
            influence_times=aquifers.influence_table_times[start:end],  # type: ignore[arg-type]
            influence_pressures=aquifers.influence_table_pressures[start:end],  # type: ignore[arg-type]
        )
        return (
            new_cumulative_influx,
            previous_aquifer_pressure,
            current_dimensionless_time,
            -influx_per_pressure_drop,
        )

    new_cumulative_influx, new_aquifer_pressure, influx_derivative = (
        fetkovich.compute_incremental_influx(
            previous_cumulative_influx=previous_cumulative_influx,
            previous_aquifer_pressure=previous_aquifer_pressure,
            boundary_pressure=boundary_pressure,
            productivity_index=aquifers.productivity_indices[row],
            initial_encroachable_water=aquifers.encroachable_waters[row],
            initial_pressure=aquifers.initial_pressures[row],
            elapsed_time=elapsed_time,
        )
    )
    return (
        new_cumulative_influx,
        new_aquifer_pressure,
        previous_dimensionless_time,
        influx_derivative,
    )


@numba.njit(cache=True)
def compute_aquifer_boundary_pressure(
    aquifers: CompiledAquifers, pressure: CellArray, row: Integer
) -> Number:
    """
    COmputes an aquifer's boundary pressure: its owner cells' pressures averaged with each face's
    share of the aquifer's influx as the weight.

    :param aquifers: Every aquifer's static parameters, whole.
    :param pressure: Current (trial or accepted) cell pressures.
    :param row: The aquifer's row.
    :returns: The boundary pressure.
    """
    total = 0.0
    for i in range(aquifers.region_offsets[row], aquifers.region_offsets[row + 1]):
        total += aquifers.face_weights[i] * pressure[aquifers.owner_cells[i]]
    return total


@numba.njit(cache=True, parallel=True)
def apply_aquifer_rates(
    aquifers: CompiledAquifers,
    workspace: AquiferWorkspace,
    pressure: CellArray,
    time: Number,
    out_flux_values: NumberArray[OneDimension],
    out_flux_derivatives: NumberArray[OneDimension],
    out_rate_derivatives: NumberArray[OneDimension],
) -> None:
    """
    Computes every aquifer's current rate and broadcasts it across its own faces, directly
    into the output arrays. The rate is the influx over the step from the last committed time
    to `time`, divided by that step's length.

    :param aquifers: Every aquifer's static parameters, whole.
    :param workspace: This run's `AquiferWorkspace`. Read only.
    :param pressure: Current (trial or accepted) cell pressures.
    :param time: The time being solved for, the end of the step in progress.
    :param out_flux_values: The full-length flux array, written into at each aquifer's own
        `face_positions` with that face's share of the aquifer's rate.
    :param out_flux_derivatives: The full-length flux derivative array, written the same way.
    :param out_rate_derivatives: Shape `(n_aquifers,)`, each aquifer's rate derivative with
        respect to its boundary pressure.
    """
    region_offsets = aquifers.region_offsets
    face_positions = aquifers.face_positions
    face_weights = aquifers.face_weights

    n_aquifers = region_offsets.shape[0] - 1
    for row in numba.prange(n_aquifers):
        boundary_pressure = compute_aquifer_boundary_pressure(aquifers, pressure, row)
        elapsed_time = time - workspace.previous_time[row]
        rate = 0.0
        rate_derivative = 0.0
        if elapsed_time > 0.0:
            new_cumulative_influx, _, _, influx_derivative = evaluate_aquifer_step(
                aquifers, workspace, row, boundary_pressure, time
            )
            rate = (new_cumulative_influx - workspace.previous_cumulative_influx[row]) / (
                elapsed_time
            )
            rate_derivative = influx_derivative / elapsed_time
        out_rate_derivatives[row] = rate_derivative
        for i in range(region_offsets[row], region_offsets[row + 1]):
            out_flux_values[face_positions[i]] = rate * face_weights[i]
            out_flux_derivatives[face_positions[i]] = (
                rate_derivative * face_weights[i] * face_weights[i]
            )


@numba.njit(cache=True, parallel=True)
def advance_aquifer_workspace(
    aquifers: CompiledAquifers,
    workspace: AquiferWorkspace,
    pressure: CellArray,
    time: Number,
) -> None:
    """
    Advances every aquifer row of `workspace` to `time`, in place, from
    `pressure`. This is the commit step.

    :param aquifers: Every aquifer's static parameters, whole.
    :param workspace: Updated in place.
    :param pressure: The accepted cell pressures for `time`.
    :param time: Time being committed to.
    """
    n_aquifers = aquifers.region_offsets.shape[0] - 1
    for row in numba.prange(n_aquifers):
        if time - workspace.previous_time[row] <= 0.0:
            continue
        boundary_pressure = compute_aquifer_boundary_pressure(aquifers, pressure, row)
        new_cumulative_influx, new_aquifer_pressure, new_dimensionless_time, _ = (
            evaluate_aquifer_step(aquifers, workspace, row, boundary_pressure, time)
        )
        workspace.previous_cumulative_influx[row] = new_cumulative_influx
        workspace.previous_aquifer_pressure[row] = new_aquifer_pressure
        workspace.previous_dimensionless_time[row] = new_dimensionless_time
        workspace.previous_boundary_pressure[row] = boundary_pressure
        workspace.previous_time[row] = time


def compute_boundary_cache(
    compiled: CompiledBoundaryConditions,
    aquifer_workspace: AquiferWorkspace,
    pressure: CellArray,
    time: Number,
    dtype: npt.DTypeLike = None,
    out: BoundaryCache | None = None,
) -> BoundaryCache:
    """
    Builds one iteration's `BoundaryCache`.

    :param compiled: The compiled boundary condition set.
    :param aquifer_workspace: This run's `AquiferWorkspace`. Read only.
    :param pressure: Current (trial or accepted) cell pressures.
    :param time: The time being solved for, the end of the step in progress. An aquifer's rate
        is its influx since the last committed time divided by the length of that step.
    :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
    :param out: Reuse an existing `BoundaryCache`'s buffers instead of allocating new ones.
    :returns: This iteration's `BoundaryCache`.
    """
    resolved_dtype = np.dtype(dtype if dtype is not None else get_dtype())
    n_aquifers = compiled.aquifers.kinds.shape[0]
    if out is not None:
        pressure_values, flux_values, is_dirichlet, flux_derivatives, rate_derivatives = out
        pressure_values[:] = compiled.static_pressure_values
        flux_values[:] = compiled.static_flux_values
        is_dirichlet[:] = compiled.static_is_dirichlet
        flux_derivatives[:] = 0.0
        rate_derivatives[:] = 0.0
    else:
        pressure_values = compiled.static_pressure_values.astype(resolved_dtype, copy=True)
        flux_values = compiled.static_flux_values.astype(resolved_dtype, copy=True)
        is_dirichlet = compiled.static_is_dirichlet.copy()
        flux_derivatives = np.zeros(compiled.n_boundary_faces, dtype=resolved_dtype)
        rate_derivatives = np.zeros(n_aquifers, dtype=resolved_dtype)

    apply_productivity_index_values(
        productivity_indices=compiled.productivity_indices,
        pressure=pressure,
        out_flux_values=typing.cast(NumberArray[OneDimension], flux_values),
        out_flux_derivatives=typing.cast(NumberArray[OneDimension], flux_derivatives),
    )
    apply_aquifer_rates(
        aquifers=compiled.aquifers,
        workspace=aquifer_workspace,
        pressure=pressure,
        time=time,
        out_flux_values=typing.cast(NumberArray[OneDimension], flux_values),
        out_flux_derivatives=typing.cast(NumberArray[OneDimension], flux_derivatives),
        out_rate_derivatives=typing.cast(NumberArray[OneDimension], rate_derivatives),
    )
    return BoundaryCache(
        pressure_values=typing.cast(NumberArray[OneDimension], pressure_values),
        flux_values=typing.cast(NumberArray[OneDimension], flux_values),
        is_dirichlet=is_dirichlet,
        flux_derivatives=typing.cast(NumberArray[OneDimension], flux_derivatives),
        aquifer_rate_derivatives=typing.cast(NumberArray[OneDimension], rate_derivatives),
    )


def commit_boundary_conditions(
    compiled: CompiledBoundaryConditions,
    aquifer_workspace: AquiferWorkspace,
    pressure: CellArray,
    time: Number,
) -> None:
    """
    Advances `aquifer_workspace` to `time`, in place, from `pressure`.

    To be called exactly once per accepted timestep, with the accepted pressure
    field, never from inside a Newton/Picard iteration.

    Stateless conditions (`ConstantFluxBoundary`, `ConstantPressureBoundary`,
    `ProductivityIndexBoundary`) have nothing to commit so only `aquifer_workspace`'s
    rows change here.

    :param compiled: The compiled boundary condition set.
    :param aquifer_workspace: Updated in place.
    :param pressure: The accepted cell pressures for `time`.
    :param time: Time being committed to.
    """
    advance_aquifer_workspace(
        aquifers=compiled.aquifers,
        workspace=aquifer_workspace,
        pressure=pressure,
        time=time,
    )
