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
from bores.types import BooleanArray, CellArray, Number, NumberArray, OneDimension

__all__ = ["BoundaryCache", "commit_boundary_conditions", "compute_boundary_cache"]


class BoundaryCache(typing.NamedTuple):
    """One iteration's worth of per-boundary-face solver input."""

    pressure_values: NumberArray[OneDimension]
    """Shape `(n_boundary_faces,)`. Prescribed pressure at Dirichlet faces; `0.0` elsewhere."""

    flux_values: NumberArray[OneDimension]
    """Shape `(n_boundary_faces,)`. Prescribed flux at Neumann/Robin faces; `0.0` elsewhere."""

    is_dirichlet: BooleanArray[OneDimension]
    """Shape `(n_boundary_faces,)`. `True` where a `PRESSURE`-type condition is active."""


@numba.njit(cache=True, parallel=True)
def apply_productivity_index_values(
    productivity_indices: CompiledProductivityIndices,
    pressure: CellArray,
    out_flux_values: NumberArray[OneDimension],
) -> None:
    """
    Writes every `ProductivityIndexBoundary` face's flux directly into
    `out_flux_values`, in place .

    :param productivity_indices: Every PI region's faces and parameters, whole.
    :param pressure: Current (trial or accepted) cell pressures.
    :param out_flux_values: The full-length flux array, written into at
        `productivity_indices.face_positions`.
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


@numba.njit(cache=True, parallel=True)
def apply_aquifer_rates(
    aquifers: CompiledAquifers,
    workspace: AquiferWorkspace,
    pressure: CellArray,
    time: Number,
    out_flux_values: NumberArray[OneDimension],
) -> None:
    """
    Computes every aquifer's current rate and broadcasts it across its
    own faces, directly into `out_flux_values`.

    :param aquifers: Every aquifer's static parameters, whole.
    :param workspace: This run's `AquiferWorkspace`. Read only.
    :param pressure: Current (trial or accepted) cell pressures.
    :param time: Current simulation time.
    :param out_flux_values: The full-length flux array, written into at
        each aquifer's own `face_positions`.
    """
    region_offsets = aquifers.region_offsets
    owner_cells = aquifers.owner_cells
    face_positions = aquifers.face_positions
    kinds = aquifers.kinds
    initial_pressures = aquifers.initial_pressures
    aquifer_constants = aquifers.aquifer_constants
    dimensionless_time_scales = aquifers.dimensionless_time_scales
    bounded = aquifers.bounded
    dimensionless_radius_ratios = aquifers.dimensionless_radius_ratios
    bessel_roots = aquifers.bessel_roots
    pd_coefficients = aquifers.pd_coefficients
    pd_prime_coefficients = aquifers.pd_prime_coefficients
    linear_coefficients = aquifers.linear_coefficients
    constant_coefficients = aquifers.constant_coefficients
    productivity_indices = aquifers.productivity_indices
    encroachable_waters = aquifers.encroachable_waters

    previous_time = workspace.previous_time
    previous_cumulative_influx = workspace.previous_cumulative_influx
    previous_dimensionless_time = workspace.previous_dimensionless_time
    previous_aquifer_pressure = workspace.previous_aquifer_pressure

    n_aquifers = region_offsets.shape[0] - 1
    for row in numba.prange(n_aquifers):
        start = region_offsets[row]
        end = region_offsets[row + 1]

        total_pressure = 0.0
        for i in range(start, end):
            total_pressure += pressure[owner_cells[i]]
        boundary_pressure = total_pressure / (end - start)

        elapsed_time = time - previous_time[row]
        if elapsed_time <= 0.0:
            rate = 0.0
        else:
            if kinds[row] == CARTER_TRACY_AQUIFER_KIND:
                current_dimensionless_time = dimensionless_time_scales[row] * time
                pressure_drop = initial_pressures[row] - boundary_pressure
                new_cumulative_influx = carter_tracy.compute_incremental_influx(
                    previous_cumulative_influx=previous_cumulative_influx[row],
                    previous_dimensionless_time=previous_dimensionless_time[row],
                    current_dimensionless_time=current_dimensionless_time,
                    current_pressure_drop=pressure_drop,
                    aquifer_constant=aquifer_constants[row],
                    bounded=bounded[row],
                    dimensionless_radius_ratio=dimensionless_radius_ratios[row],
                    betas=bessel_roots[row],
                    pd_coefficients=pd_coefficients[row],
                    pd_prime_coefficients=pd_prime_coefficients[row],
                    linear_coefficient=linear_coefficients[row],
                    constant_coefficient=constant_coefficients[row],
                )
            else:
                new_cumulative_influx, _ = fetkovich.compute_incremental_influx(
                    previous_cumulative_influx=previous_cumulative_influx[row],
                    previous_aquifer_pressure=previous_aquifer_pressure[row],
                    boundary_pressure=boundary_pressure,
                    productivity_index=productivity_indices[row],
                    initial_encroachable_water=encroachable_waters[row],
                    initial_pressure=initial_pressures[row],
                    elapsed_time=elapsed_time,
                )
            rate = (new_cumulative_influx - previous_cumulative_influx[row]) / elapsed_time

        for i in range(start, end):
            out_flux_values[face_positions[i]] = rate


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
    region_offsets = aquifers.region_offsets
    owner_cells = aquifers.owner_cells
    kinds = aquifers.kinds
    initial_pressures = aquifers.initial_pressures
    aquifer_constants = aquifers.aquifer_constants
    dimensionless_time_scales = aquifers.dimensionless_time_scales
    bounded = aquifers.bounded
    dimensionless_radius_ratios = aquifers.dimensionless_radius_ratios
    bessel_roots = aquifers.bessel_roots
    pd_coefficients = aquifers.pd_coefficients
    pd_prime_coefficients = aquifers.pd_prime_coefficients
    linear_coefficients = aquifers.linear_coefficients
    constant_coefficients = aquifers.constant_coefficients
    productivity_indices = aquifers.productivity_indices
    encroachable_waters = aquifers.encroachable_waters

    previous_time = workspace.previous_time
    previous_cumulative_influx = workspace.previous_cumulative_influx
    previous_boundary_pressure = workspace.previous_boundary_pressure
    previous_dimensionless_time = workspace.previous_dimensionless_time
    previous_aquifer_pressure = workspace.previous_aquifer_pressure

    n_aquifers = region_offsets.shape[0] - 1
    for row in numba.prange(n_aquifers):
        start = region_offsets[row]
        end = region_offsets[row + 1]

        total_pressure = 0.0
        for i in range(start, end):
            total_pressure += pressure[owner_cells[i]]
        boundary_pressure = total_pressure / (end - start)

        elapsed_time = time - previous_time[row]
        if elapsed_time <= 0.0:
            continue

        if kinds[row] == CARTER_TRACY_AQUIFER_KIND:
            current_dimensionless_time = dimensionless_time_scales[row] * time
            pressure_drop = initial_pressures[row] - boundary_pressure
            new_cumulative_influx = carter_tracy.compute_incremental_influx(
                previous_cumulative_influx=previous_cumulative_influx[row],
                previous_dimensionless_time=previous_dimensionless_time[row],
                current_dimensionless_time=current_dimensionless_time,
                current_pressure_drop=pressure_drop,
                aquifer_constant=aquifer_constants[row],
                bounded=bounded[row],
                dimensionless_radius_ratio=dimensionless_radius_ratios[row],
                betas=bessel_roots[row],
                pd_coefficients=pd_coefficients[row],
                pd_prime_coefficients=pd_prime_coefficients[row],
                linear_coefficient=linear_coefficients[row],
                constant_coefficient=constant_coefficients[row],
            )
            previous_dimensionless_time[row] = current_dimensionless_time
        else:
            new_cumulative_influx, new_aquifer_pressure = fetkovich.compute_incremental_influx(
                previous_cumulative_influx=previous_cumulative_influx[row],
                previous_aquifer_pressure=previous_aquifer_pressure[row],
                boundary_pressure=boundary_pressure,
                productivity_index=productivity_indices[row],
                initial_encroachable_water=encroachable_waters[row],
                initial_pressure=initial_pressures[row],
                elapsed_time=elapsed_time,
            )
            previous_aquifer_pressure[row] = new_aquifer_pressure

        previous_cumulative_influx[row] = new_cumulative_influx
        previous_time[row] = time
        previous_boundary_pressure[row] = boundary_pressure


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
    :param time: Current simulation time.
    :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
    :param out: Reuse an existing `BoundaryCache`'s buffers instead of allocating new ones.
    :returns: This iteration's `BoundaryCache`.
    """
    resolved_dtype = np.dtype(dtype) if dtype is not None else get_dtype()

    if out is not None:
        pressure_values, flux_values, is_dirichlet = out
        pressure_values[:] = compiled.static_pressure_values
        flux_values[:] = compiled.static_flux_values
        is_dirichlet[:] = compiled.static_is_dirichlet
    else:
        pressure_values = compiled.static_pressure_values.astype(resolved_dtype, copy=True)
        flux_values = compiled.static_flux_values.astype(resolved_dtype, copy=True)
        is_dirichlet = compiled.static_is_dirichlet.copy()

    apply_productivity_index_values(compiled.productivity_indices, pressure, flux_values)
    apply_aquifer_rates(compiled.aquifers, aquifer_workspace, pressure, time, flux_values)
    return BoundaryCache(
        pressure_values=pressure_values,
        flux_values=flux_values,
        is_dirichlet=is_dirichlet,
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
    advance_aquifer_workspace(compiled.aquifers, aquifer_workspace, pressure, time)
