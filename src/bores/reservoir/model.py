"""Reservoir characterization and an assembly of grid, rock, faults, and regions."""

import typing

import numpy as np
from typing_extensions import Self

from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.reservoir.faults import Fault, apply_faults
from bores.reservoir.regions import Regions
from bores.reservoir.rock import Rock
from bores.reservoir.transmissibility import (
    ConnectionTransmissibilities,
    compute_connection_transmissibilities,
)
from bores.types import CellArray, Number, UnitSystem
from bores.units import (
    UnitConversionTable,
    build_unit_conversion_table,
)

__all__ = ["Reservoir"]


def validate_rock(rock: Rock, n_cells: int) -> Rock:
    """
    Verify that all per-cell arrays in `rock` have length `n_cells`.

    :param rock: `Rock` to validate.
    :param n_cells: Expected array length.
    :raises ValidationError: On any mismatch.
    """
    checks = {
        "rock.porosity": rock.porosity,
        "rock.absolute_permeability.x": rock.absolute_permeability.x,
        "rock.absolute_permeability.y": rock.absolute_permeability.y,
        "rock.absolute_permeability.z": rock.absolute_permeability.z,
        "rock.net_to_gross": rock.net_to_gross,
        "rock.connate_water_saturation": rock.connate_water_saturation,
        "rock.irreducible_water_saturation": rock.irreducible_water_saturation,
        "rock.residual_oil_saturation_water": (rock.residual_oil_saturation_water),
        "rock.residual_oil_saturation_gas": (rock.residual_oil_saturation_gas),
        "rock.residual_gas_saturation": rock.residual_gas_saturation,
    }
    for name, array in checks.items():
        if array.shape != (n_cells,):
            raise ValidationError(f"`{name}` has shape {array.shape}; expected ({n_cells},).")
    return rock


def validate_regions(regions: Regions, n_cells: int) -> Regions:
    """
    Verify that every region array in `regions` has length `n_cells` and holds region
    numbers of at least 1.

    :param regions: `Regions` to validate.
    :param n_cells: Expected array length.
    :raises ValidationError: On any mismatch or region number below 1.
    """
    for name in (
        "pvt_region",
        "saturation_region",
        "imbibition_region",
        "equilibrium_region",
        "rock_region",
        "fluid_in_place_region",
    ):
        array = getattr(regions, name)
        if array is None:
            continue
        if array.shape != (n_cells,):
            raise ValidationError(
                f"`regions.{name}` has shape {array.shape}; expected ({n_cells},)."
            )
        if array.size > 0 and int(array.min()) < 1:
            raise ValidationError(f"`regions.{name}` holds region numbers below 1.")
    return regions


class Reservoir:
    """
    Reservoir characterization.

    Binds a polyhedral `Grid` to per-cell rock, faults, regions, boundary conditions,
    etc. On construction all property groups are normalised to the
    declared `unit_system` (defaults to the grid's own unit system).
    """

    __slots__ = (
        "_transmissibilities",
        "faults",
        "grid",
        "regions",
        "rock",
        "unit_system",
    )

    def __init__(
        self,
        grid: Grid,
        rock: Rock,
        regions: Regions | None = None,
        faults: typing.Collection[Fault] | None = None,
        unit_system: UnitSystem | None = None,
    ) -> None:
        """
        Initialize the reservoir model.

        :param grid: Fully constructed `bores.grids.base.Grid`.
        :param rock: Static petrophysical properties. Array lengths must equal `grid.n_cells`.
        :param regions: Optional per-cell region assignments. Array lengths must equal `grid.n_cells`.
        :param faults: Optional collection of `Fault` objects to apply to the grid.
        :param unit_system: Target unit system for all property groups. When `None`, defaults
            to `grid.unit_system`.
        :raises ValidationError: If any array length in `rock` or `regions` does not match
            `grid.n_cells`, or a region number is below 1.
        """
        n_cells = grid.n_cells
        validate_rock(rock, n_cells)
        if regions is not None:
            validate_regions(regions, n_cells)

        target_unit_system = unit_system if unit_system is not None else grid.unit_system
        unit_conversion_table = build_unit_conversion_table()
        if target_unit_system != grid.unit_system:
            # Normalise grid to the target unit system.
            grid = grid.convert(target_unit_system, table=unit_conversion_table)

        # Normalise property groups to the target unit system.
        rock = rock.convert(target_unit_system, table=unit_conversion_table)
        if faults is not None:
            grid = apply_faults(grid, *faults)

        self.grid = grid
        """The unstructured polyhedral grid with all geometry and topology."""

        self.rock = rock
        """Static petrophysical properties in `unit_system`."""

        self.regions = regions
        """Per-cell region assignments metadata."""

        self.faults = faults
        """Optional collection of `Fault` objects applied to the grid."""

        self.unit_system = target_unit_system
        """
        Unit system in which all property groups are expressed.

        Matches `grid.unit_system` by construction.
        """
        self._transmissibilities: ConnectionTransmissibilities | None = None

    @property
    def n_cells(self) -> int:
        """Total number of active cells in the grid."""
        return self.grid.n_cells

    @property
    def n_faces(self) -> int:
        """Total number of faces (boundary + interior) in the grid."""
        return self.grid.n_faces

    @property
    def n_nnc(self) -> int:
        """Total number of non-neighbour connections in the grid."""
        return self.grid.n_nnc

    @property
    def n_interior_faces(self) -> int:
        """Number of interior faces (shared between two active cells)."""
        return self.grid.n_interior_faces

    @property
    def n_boundary_faces(self) -> int:
        """Number of boundary faces (one side is the exterior domain)."""
        return self.grid.n_boundary_faces

    @property
    def depth(self) -> CellArray:
        """
        Shape `(n_cells,)` - depth of each cell centroid (positive downward).

        Alias for `grid.cell_center_depths`. Units follow `unit_system`.
        """
        return self.grid.cell_center_depths  # type: ignore[return-value]

    @property
    def elevation(self) -> CellArray:
        """
        Shape `(n_cells,)` - elevation of each cell centroid (positive upward).

        Alias for `grid.cell_center_elevations`. Units follow `unit_system`.
        """
        return self.grid.cell_center_elevations  # type: ignore[return-value]

    @property
    def pore_volumes(self) -> CellArray:
        """
        Shape `(n_cells,)` - bulk pore volume of each active cell.

        Computed as `φ x NTG x Vcell`. This is the pore volume at the rock's reference
        pressure. Use `compute_pore_volumes` for the pore volume at another pressure.

        Units: ft³ (FIELD), m³ (METRIC / SI), cm³ (LAB).
        """
        assert self.grid.cell_volumes is not None, (
            "`grid.cell_volumes` is None; the grid was constructed without "
            "pre-computed volumes and the divergence-theorem computation failed."
        )
        return (  # type: ignore[return-value]
            self.rock.porosity * self.rock.net_to_gross * self.grid.cell_volumes
        )

    def compute_pore_volumes(self, pressure: CellArray) -> CellArray:
        """
        Computes pore volume of each cell at `pressure`, with rock compaction.

        Returns `pore_volumes` times the rock's pore volume multiplier.
        Equal to `pore_volumes` when the rock has no compressibility data.

        :param pressure: Shape `(n_cells,)` cell pressures in `unit_system`.
        :returns: Pore volumes in the same units as `pore_volumes`.
        """
        multiplier = self.rock.get_pore_volume_multiplier(
            pressure=pressure,
            rock_region=self.regions.rock_region if self.regions is not None else None,
            dtype=self.pore_volumes.dtype,
        )
        return typing.cast(CellArray, self.pore_volumes * multiplier)

    @property
    def hydrocarbon_pore_volumes(self) -> CellArray:
        """
        Shape `(n_cells,)` - hydrocarbon pore volume of each active cell.

        Computed as `pore_volumes x (1 - swc)` where swc is the connate
        water saturation from `rock`.

        Units: same as `pore_volumes`.
        """
        return (  # type: ignore[return-value]
            self.pore_volumes * (1.0 - self.rock.connate_water_saturation)
        )

    @property
    def transmissibilities(self) -> ConnectionTransmissibilities:
        """
        TPFA connection transmissibilities, computed on first access and
        then cached for subsequent access.

        Returns a `ConnectionTransmissibilities` named tuple with:

        - `interior`      - shape `(n_interior,)` harmonic-mean T.
        - `boundary`      - shape `(n_boundary,)` owner half-T.
        - `nnc`           - shape `(n_nnc,)` or `None`.
        - `unit_system`   - the unit system the values are in.

        To force recomputation (e.g. after updating `rock`), call
        `invalidate_transmissibilities()` first.
        """
        if self._transmissibilities is None:
            transmissibilities = compute_connection_transmissibilities(
                grid=self.grid, rock=self.rock, unit_system=self.unit_system
            )
            assert transmissibilities.unit_system == self.unit_system
            self._transmissibilities = transmissibilities
        return self._transmissibilities

    def invalidate_transmissibilities(self) -> None:
        """
        Clear the cached transmissibilities.

        Call this whenever `rock.absolute_permeability` or `rock.net_to_gross`
        is updated (e.g. history-matching) so the next access to
        `transmissibilities` triggers a fresh computation.
        """
        self._transmissibilities = None

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `Reservoir` with all property groups rescaled to `target`.

        For a true coordinate reprojection (e.g. ft -> m for all vertex
        positions), use a grid factory or IO utility that rebuilds the grid.

        :param target: Desired `UnitSystem`.
        :returns: New `Reservoir` in `target` units.
        """
        if target == self.unit_system:
            return self

        table = table or build_unit_conversion_table()
        new_model = self.__class__(
            grid=self.grid.convert(target, table=table),
            rock=self.rock.convert(target, table=table),
            regions=self.regions,
            unit_system=target,
        )
        new_model.faults = self.faults
        # Transmissibility cache is invalidated automatically since rock was
        # converted and the new model starts with a clean cache.
        return new_model

    def get_face_transmissibility(self, face_index: int) -> Number:
        """
        Return the transmissibility (or half-transmissibility for boundary) of a single face.

        :param face_index: Index into `grid.face_cell_indices`.
        :returns: Geometric transmissibility `K·A/d` (mD·ft in FIELD etc.), without the Darcy
            unit constant.
        :raises KeyError: If `face_index` is not a valid face.
        """
        position = np.flatnonzero(self.grid.interior_face_indices == face_index)
        if position.size > 0:
            return typing.cast(Number, self.transmissibilities.interior[position[0]])

        position = np.flatnonzero(self.grid.boundary_face_indices == face_index)
        if position.size > 0:
            return typing.cast(Number, self.transmissibilities.boundary[position[0]])
        raise KeyError(f"Face {face_index} not found in interior or boundary faces of this grid.")

    def summary(self) -> dict[str, typing.Any]:
        """
        Return a lightweight diagnostic summary dictionary.

        Includes cell counts, pore-volume totals, pressure statistics, and
        saturation range checks. Useful for logging and quick sanity checks.

        :returns: Dictionary of scalar statistics.
        """
        return {
            "n_cells": self.n_cells,
            "n_faces": self.n_faces,
            "n_interior_faces": self.n_interior_faces,
            "n_boundary_faces": self.n_boundary_faces,
            "n_nnc": self.grid.n_nnc,
            "n_faults": len(self.faults) if self.faults else 0,
            "unit_system": self.unit_system.value,
            "total_pore_volume": float(self.pore_volumes.sum()),
            "has_transmissibility_multipliers": (self.grid.has_transmissibility_multipliers),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_cells={self.n_cells}, "
            f"n_faces={self.n_faces}, "
            f"n_interior={self.n_interior_faces}, "
            f"n_boundary={self.n_boundary_faces}, "
            f"n_nnc={self.n_nnc}, "
            f"n_faults={len(self.faults) if self.faults else 0}, "
            f"unit_system={self.unit_system.value!r}, "
            f")"
        )
