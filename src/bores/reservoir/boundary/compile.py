"""Compiled, array-based boundary condition structures."""

import enum
import typing

import numpy as np
import numpy.typing as npt

from bores.constants import c
from bores.precision import get_dtype
from bores.reservoir.boundary.aquifers.carter_tracy import CarterTracyAquifer
from bores.reservoir.boundary.aquifers.fetkovich import FetkovichAquifer
from bores.reservoir.boundary.conditions import BoundaryConditions
from bores.reservoir.boundary.types import (
    ConstantFluxBoundary,
    ConstantPressureBoundary,
    ProductivityIndexBoundary,
)
from bores.reservoir.model import Reservoir
from bores.types import (
    BooleanArray,
    IntArray,
    Integer,
    Number,
    NumberArray,
    OneDimension,
    TwoDimensions,
)

__all__ = [
    "CARTER_TRACY_AQUIFER_KIND",
    "FETKOVICH_AQUIFER_KIND",
    "AquiferKind",
    "CompiledAquifers",
    "CompiledBoundaryConditions",
    "CompiledProductivityIndices",
    "compile_boundary_conditions",
]


class AquiferKind(enum.IntEnum):
    """Tag value for `CompiledAquifers.kinds`."""

    CARTER_TRACY = 0
    FETKOVICH = 1


# Plain-int mirrors of AquiferKind's own values, for use inside `@numba.njit`
# code, where comparing against an IntEnum member directly is best avoided.
CARTER_TRACY_AQUIFER_KIND: Integer = AquiferKind.CARTER_TRACY.value
FETKOVICH_AQUIFER_KIND: Integer = AquiferKind.FETKOVICH.value


class CompiledProductivityIndices(typing.NamedTuple):
    """Every `ProductivityIndexBoundary` region's faces, flattened row-per-face."""

    owner_cells: IntArray[OneDimension]
    """Shape `(n_faces,)`. Each face's owner cell, precomputed once so no grid lookup on the hot path."""

    face_positions: IntArray[OneDimension]
    """Shape `(n_faces,)`. Positions into `Grid.boundary_face_indices`, same order as `owner_cells`."""

    row_of_face: IntArray[OneDimension]
    """Shape `(n_faces,)`. Which region (row into `pressure_boundary`/`productivity_index`) owns this face."""

    pressure_boundary: NumberArray[OneDimension]
    """Shape `(n_regions,)`. Each region's own reference boundary pressure."""

    productivity_index: NumberArray[OneDimension]
    """Shape `(n_regions,)`. Each region's own uniform productivity index."""


class CompiledAquifers(typing.NamedTuple):
    """
    Every analytic aquifer region (`CarterTracyAquifer`/`FetkovichAquifer`),
    flattened row-per-face for geometry and row-per-aquifer for parameters.

    Static parameters only. The evolving recursive state lives in `AquiferWorkspace`.

    Every field below is present for every row regardless of `kinds[row]`;
    a row's fields for the other kind are `NaN`/zero-filled and unused.
    """

    kinds: IntArray[OneDimension]
    """Shape `(n_aquifers,)`. An `AquiferKind` (`AQUIFER_KIND_*`) per row."""

    region_offsets: IntArray[OneDimension]
    """Shape `(n_aquifers + 1,)`."""

    owner_cells: IntArray[OneDimension]
    """Shape `(n_faces,)`. Each face's owner cell, precomputed once so no grid lookup on the hot path."""

    face_positions: IntArray[OneDimension]
    """Shape `(n_faces,)`. Positions into `Grid.boundary_face_indices`, same order as `owner_cells`."""

    initial_pressures: NumberArray[OneDimension]
    """Shape `(n_aquifers,)`. Both kinds' own `initial_pressure`."""

    # Carter-Tracy fields

    aquifer_constants: NumberArray[OneDimension]
    """Shape `(n_aquifers,)`. Carter-Tracy only."""

    dimensionless_time_scales: NumberArray[OneDimension]
    """
    Shape `(n_aquifers,)`. Carter-Tracy only. `tD = dimensionless_time_scales[row] * time`,
    already resolved from whichever of `hydraulic_diffusivity/inner_radius^2` (physical mode) 
    or `dimensionless_time_scale` (calibrated mode, `1.0` if that was left unset) applies.
    """

    bounded: BooleanArray[OneDimension]
    """Shape `(n_aquifers,)`. Carter-Tracy only."""

    dimensionless_radius_ratios: NumberArray[OneDimension]
    """Shape `(n_aquifers,)`. Carter-Tracy only."""

    bessel_roots: NumberArray[TwoDimensions]
    """Shape `(n_aquifers, AQUIFER_BESSEL_SERIES_TERMS)`. Carter-Tracy, `bounded=True` rows only."""

    pd_coefficients: NumberArray[TwoDimensions]
    """Shape `(n_aquifers, AQUIFER_BESSEL_SERIES_TERMS)`. Carter-Tracy, `bounded=True` rows only."""

    pd_prime_coefficients: NumberArray[TwoDimensions]
    """Shape `(n_aquifers, AQUIFER_BESSEL_SERIES_TERMS)`. Carter-Tracy, `bounded=True` rows only."""

    linear_coefficients: NumberArray[OneDimension]
    """Shape `(n_aquifers,)`. Carter-Tracy, `bounded=True` rows only."""

    constant_coefficients: NumberArray[OneDimension]
    """Shape `(n_aquifers,)`. Carter-Tracy, `bounded=True` rows only."""

    # Fetkovich fields

    productivity_indices: NumberArray[OneDimension]
    """Shape `(n_aquifers,)`. Fetkovich only."""

    encroachable_waters: NumberArray[OneDimension]
    """Shape `(n_aquifers,)`. Fetkovich only."""

    def region_rows(self, *, row: Integer) -> range:
        """
        One aquifer's own face-row range, without the caller needing to
        know about `region_offsets`.

        :param row: The aquifer's row.
        :returns: `range(start, end)` over this aquifer's face rows.
        """
        return range(self.region_offsets[row], self.region_offsets[row + 1])


class CompiledBoundaryConditions(typing.NamedTuple):
    """
    Every boundary condition in a model, compiled to flat arrays.

    `static_pressure_values`/`static_flux_values`/`static_is_dirichlet` are filled
    once, here, from every `ConstantFluxBoundary`/`ConstantPressureBoundary` region.

    This is genuinely static, since neither depends on anything that changes during a run.
    Every other face position is zero/`False` here and gets filled in by
    `compute_boundary_cache` every call, from `productivity_indices` or `aquifers`.
    """

    n_boundary_faces: Integer
    """Total boundary face count - `len(Grid.boundary_face_indices)`."""

    static_pressure_values: NumberArray[OneDimension]
    """Shape `(n_boundary_faces,)`. Every `ConstantPressureBoundary` face's value; `0.0` elsewhere."""

    static_flux_values: NumberArray[OneDimension]
    """Shape `(n_boundary_faces,)`. Every `ConstantFluxBoundary` face's value; `0.0` elsewhere."""

    static_is_dirichlet: BooleanArray[OneDimension]
    """Shape `(n_boundary_faces,)`. `True` only at a `ConstantPressureBoundary` face."""

    productivity_indices: CompiledProductivityIndices
    """Every `ProductivityIndexBoundary` region, empty if there are none."""

    productivity_index_names: tuple[str, ...]
    """Each `productivity_indices` region's own name, same row order. Reporting only."""

    aquifers: CompiledAquifers
    """Every analytic aquifer region, empty if there are none."""

    aquifer_names: tuple[str, ...]
    """Each `aquifers` region's own name, same row order. Reporting and `decompile_aquifer` only."""


def resolve_owner_cells(
    reservoir: Reservoir, face_positions: IntArray[OneDimension]
) -> IntArray[OneDimension]:
    """
    Resolves boundary face positions to their owner cells, once, at compile time.

    :param reservoir: Supplies `.grid`.
    :param face_positions: Positions into `Grid.boundary_face_indices`.
    :returns: Matching owner cell indices, same order and shape.
    """
    grid = reservoir.grid
    global_face_indices = grid.boundary_face_indices[face_positions]
    return typing.cast(IntArray[OneDimension], grid.face_cell_indices[global_face_indices, 0])


def compile_boundary_conditions(
    boundary_conditions: BoundaryConditions,
    reservoir: Reservoir,
    dtype: npt.DTypeLike = None,
) -> CompiledBoundaryConditions:
    """
    Compiles a rich `BoundaryConditions` into flat, array-based
    `CompiledBoundaryConditions`.

    Processes `boundary_conditions.regions` in list order, matching the
    "later region wins" convention `BoundaryConditions` itself documents.
    A region assigns its own faces regardless of kind, so a
    `ConstantPressureBoundary` region appearing after an aquifer region
    on the same faces correctly overrides it, and vice versa.

    :param boundary_conditions: The rich boundary condition set to compile.
    :param reservoir: Supplies `.grid`, for resolving owner cells once.
    :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
    :returns: `CompiledBoundaryConditions`.
    """
    resolved_dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    n_boundary_faces = len(reservoir.grid.boundary_face_indices)

    static_pressure_values = np.zeros(n_boundary_faces, dtype=resolved_dtype)
    static_flux_values = np.zeros(n_boundary_faces, dtype=resolved_dtype)
    static_is_dirichlet = np.zeros(n_boundary_faces, dtype=np.bool_)

    pi_regions: list[tuple[str, IntArray[OneDimension], Number, Number]] = []
    aquifer_regions: list[
        tuple[str, IntArray[OneDimension], CarterTracyAquifer | FetkovichAquifer]
    ] = []
    face_pi_row: dict[int, int] = {}
    face_aquifer_row: dict[int, int] = {}

    for region in boundary_conditions.regions:
        face_positions = typing.cast(
            IntArray[OneDimension], np.asarray(region.face_positions, dtype=np.int64)
        )
        if face_positions.shape[0] == 0:
            continue
        condition = region.condition

        # Clear any earlier assignment at these exact faces first, so a
        # later region of any kind correctly overrides an earlier one of
        # any (possibly different) kind.
        static_pressure_values[face_positions] = 0.0
        static_flux_values[face_positions] = 0.0
        static_is_dirichlet[face_positions] = False
        for position in face_positions.tolist():
            face_pi_row.pop(position, None)
            face_aquifer_row.pop(position, None)

        if isinstance(condition, ConstantPressureBoundary):
            static_pressure_values[face_positions] = condition.pressure
            static_is_dirichlet[face_positions] = True
        elif isinstance(condition, ConstantFluxBoundary):
            static_flux_values[face_positions] = condition.flux
        elif isinstance(condition, (CarterTracyAquifer, FetkovichAquifer)):
            row = len(aquifer_regions)
            aquifer_regions.append((region.name, face_positions, condition))
            for position in face_positions.tolist():
                face_aquifer_row[position] = row
        else:
            # ProductivityIndexBoundary, or any other FLUX/PRESSURE kind
            # not covered above - resolved generically via its own
            # `pressure_boundary`/`productivity_index` fields.
            pi_condition = typing.cast(ProductivityIndexBoundary, condition)
            row = len(pi_regions)
            pi_regions.append((
                region.name,
                face_positions,
                pi_condition.pressure_boundary,
                pi_condition.productivity_index,
            ))
            for position in face_positions.tolist():
                face_pi_row[position] = row

    # A region overridden by a later, different-kind region on some of its
    # own faces should keep only the faces it still actually owns.
    live_pi_faces = set(face_pi_row)
    live_aquifer_faces = set(face_aquifer_row)

    # Productivity index table
    pi_owner_cells: list[Integer] = []
    pi_face_positions: list[Integer] = []
    pi_row_of_face: list[Integer] = []
    pi_pressure_boundary: list[Number] = []
    pi_productivity_index: list[Number] = []
    pi_names: list[str] = []
    for original_row, (name, face_positions, pressure_boundary, productivity_index) in enumerate(
        pi_regions
    ):
        kept_positions = typing.cast(
            IntArray[OneDimension],
            np.array(
                [
                    position
                    for position in face_positions.tolist()
                    if position in live_pi_faces and face_pi_row[position] == original_row
                ],
                dtype=np.int64,
            ),
        )
        if kept_positions.shape[0] == 0:
            continue
        owner_cells = resolve_owner_cells(reservoir, kept_positions)
        new_row = len(pi_names)
        pi_owner_cells.extend(owner_cells.tolist())
        pi_face_positions.extend(kept_positions.tolist())
        pi_row_of_face.extend([new_row] * kept_positions.shape[0])
        pi_pressure_boundary.append(pressure_boundary)
        pi_productivity_index.append(productivity_index)
        pi_names.append(name)

    productivity_indices = CompiledProductivityIndices(
        owner_cells=typing.cast(
            IntArray[OneDimension], np.asarray(pi_owner_cells, dtype=np.int64)
        ),
        face_positions=typing.cast(
            IntArray[OneDimension], np.asarray(pi_face_positions, dtype=np.int64)
        ),
        row_of_face=typing.cast(
            IntArray[OneDimension], np.asarray(pi_row_of_face, dtype=np.int64)
        ),
        pressure_boundary=typing.cast(
            NumberArray[OneDimension], np.asarray(pi_pressure_boundary, dtype=resolved_dtype)
        ),
        productivity_index=typing.cast(
            NumberArray[OneDimension], np.asarray(pi_productivity_index, dtype=resolved_dtype)
        ),
    )

    # Aquifer table
    n_terms = c.AQUIFER_BESSEL_SERIES_TERMS
    aquifer_region_offsets = [0]
    aquifer_owner_cells: list[Integer] = []
    aquifer_face_positions: list[Integer] = []
    aquifer_names: list[str] = []
    aquifer_kinds: list[Integer] = []
    aquifer_initial_pressures: list[Number] = []
    aquifer_aquifer_constants: list[Number] = []
    aquifer_dimensionless_time_scales: list[Number] = []
    aquifer_bounded: list[bool] = []
    aquifer_dimensionless_radius_ratios: list[Number] = []
    aquifer_bessel_roots: list[NumberArray[OneDimension]] = []
    aquifer_pd_coefficients: list[NumberArray[OneDimension]] = []
    aquifer_pd_prime_coefficients: list[NumberArray[OneDimension]] = []
    aquifer_linear_coefficients: list[Number] = []
    aquifer_constant_coefficients: list[Number] = []
    aquifer_productivity_indices: list[Number] = []
    aquifer_encroachable_waters: list[Number] = []

    for original_row, (name, face_positions, condition) in enumerate(aquifer_regions):
        kept_positions = typing.cast(
            IntArray[OneDimension],
            np.array(
                [
                    position
                    for position in face_positions.tolist()
                    if position in live_aquifer_faces
                    and face_aquifer_row[position] == original_row
                ],
                dtype=np.int64,
            ),
        )
        if kept_positions.shape[0] == 0:
            continue
        owner_cells = resolve_owner_cells(reservoir, kept_positions)
        aquifer_owner_cells.extend(owner_cells.tolist())
        aquifer_face_positions.extend(kept_positions.tolist())
        aquifer_region_offsets.append(len(aquifer_owner_cells))
        aquifer_names.append(name)
        aquifer_initial_pressures.append(condition.initial_pressure)

        if isinstance(condition, CarterTracyAquifer):
            aquifer_kinds.append(AquiferKind.CARTER_TRACY)
            aquifer_aquifer_constants.append(condition.resolved_aquifer_constant)
            if condition.hydraulic_diffusivity is not None:
                assert condition.inner_radius is not None
                dt_scale = condition.hydraulic_diffusivity / (condition.inner_radius**2)
            elif condition.dimensionless_time_scale is not None:
                dt_scale = condition.dimensionless_time_scale
            else:
                dt_scale = 1.0
            aquifer_dimensionless_time_scales.append(dt_scale)
            aquifer_bounded.append(condition.bounded_aquifer)
            aquifer_dimensionless_radius_ratios.append(
                condition.resolved_dimensionless_radius_ratio
            )
            if condition.bounded_aquifer:
                aquifer_bessel_roots.append(condition.bessel_roots)
                aquifer_pd_coefficients.append(condition.pd_coefficients)
                aquifer_pd_prime_coefficients.append(condition.pd_prime_coefficients)
            else:
                aquifer_bessel_roots.append(np.zeros(n_terms, dtype=resolved_dtype))  # type: ignore[arg-type]
                aquifer_pd_coefficients.append(np.zeros(n_terms, dtype=resolved_dtype))  # type: ignore[arg-type]
                aquifer_pd_prime_coefficients.append(np.zeros(n_terms, dtype=resolved_dtype))  # type: ignore[arg-type]
            aquifer_linear_coefficients.append(condition.linear_coefficient)
            aquifer_constant_coefficients.append(condition.constant_coefficient)
            aquifer_productivity_indices.append(np.nan)
            aquifer_encroachable_waters.append(np.nan)
        else:
            assert isinstance(condition, FetkovichAquifer)
            aquifer_kinds.append(AquiferKind.FETKOVICH)
            aquifer_aquifer_constants.append(np.nan)
            aquifer_dimensionless_time_scales.append(np.nan)
            aquifer_bounded.append(False)
            aquifer_dimensionless_radius_ratios.append(np.nan)
            aquifer_bessel_roots.append(np.zeros(n_terms, dtype=resolved_dtype))  # type: ignore[arg-type]
            aquifer_pd_coefficients.append(np.zeros(n_terms, dtype=resolved_dtype))  # type: ignore[arg-type]
            aquifer_pd_prime_coefficients.append(np.zeros(n_terms, dtype=resolved_dtype))  # type: ignore[arg-type]
            aquifer_linear_coefficients.append(0.0)
            aquifer_constant_coefficients.append(0.0)
            aquifer_productivity_indices.append(condition.productivity_index)
            aquifer_encroachable_waters.append(condition.encroachable_water)

    aquifers = CompiledAquifers(
        kinds=np.asarray(aquifer_kinds, dtype=np.int32),  # type: ignore[arg-type]
        region_offsets=np.asarray(aquifer_region_offsets, dtype=np.int64),  # type: ignore[arg-type]
        owner_cells=np.asarray(aquifer_owner_cells, dtype=np.int64),  # type: ignore[arg-type]
        face_positions=np.asarray(aquifer_face_positions, dtype=np.int64),  # type: ignore[arg-type]
        initial_pressures=np.asarray(aquifer_initial_pressures, dtype=resolved_dtype),  # type: ignore[arg-type]
        aquifer_constants=np.asarray(aquifer_aquifer_constants, dtype=resolved_dtype),  # type: ignore[arg-type]
        dimensionless_time_scales=np.asarray(  # type: ignore[arg-type]
            aquifer_dimensionless_time_scales, dtype=resolved_dtype
        ),
        bounded=np.asarray(aquifer_bounded, dtype=np.bool_),  # type: ignore[arg-type]
        dimensionless_radius_ratios=np.asarray(  # type: ignore[arg-type]
            aquifer_dimensionless_radius_ratios, dtype=resolved_dtype
        ),
        bessel_roots=(  # type: ignore[arg-type]
            np.stack(aquifer_bessel_roots).astype(resolved_dtype)
            if aquifer_bessel_roots
            else np.zeros((0, n_terms), dtype=resolved_dtype)
        ),
        pd_coefficients=(  # type: ignore[arg-type]
            np.stack(aquifer_pd_coefficients).astype(resolved_dtype)
            if aquifer_pd_coefficients
            else np.zeros((0, n_terms), dtype=resolved_dtype)
        ),
        pd_prime_coefficients=(  # type: ignore[arg-type]
            np.stack(aquifer_pd_prime_coefficients).astype(resolved_dtype)
            if aquifer_pd_prime_coefficients
            else np.zeros((0, n_terms), dtype=resolved_dtype)
        ),
        linear_coefficients=np.asarray(aquifer_linear_coefficients, dtype=resolved_dtype),  # type: ignore[arg-type]
        constant_coefficients=np.asarray(aquifer_constant_coefficients, dtype=resolved_dtype),  # type: ignore[arg-type]
        productivity_indices=np.asarray(aquifer_productivity_indices, dtype=resolved_dtype),  # type: ignore[arg-type]
        encroachable_waters=np.asarray(aquifer_encroachable_waters, dtype=resolved_dtype),  # type: ignore[arg-type]
    )
    return CompiledBoundaryConditions(
        n_boundary_faces=n_boundary_faces,
        static_pressure_values=static_pressure_values,
        static_flux_values=static_flux_values,
        static_is_dirichlet=static_is_dirichlet,
        productivity_indices=productivity_indices,
        productivity_index_names=tuple(pi_names),
        aquifers=aquifers,
        aquifer_names=tuple(aquifer_names),
    )
