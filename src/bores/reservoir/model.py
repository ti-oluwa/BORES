"""Reservoir characterization and an assembly of grid, rock, faults, and regions."""

import typing

from typing_extensions import Self

from bores.constants import (
    UnitConversionTable,
    build_unit_conversion_table,
    get_conversion_factors,
)
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.reservoir.faults import Fault, apply_faults
from bores.reservoir.regions import Regions
from bores.reservoir.rock import Rock
from bores.reservoir.transmissibility import (
    ConnectionTransmissibilities,
    compute_connection_transmissibilities,
    get_face_transmissibility_map,
)
from bores.serde.base import Serializable
from bores.types import CellArray, Number, UnitSystem

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
        "rock.residual_oil_saturation_water_flood": (rock.residual_oil_saturation_water_flood),
        "rock.residual_oil_saturation_gas_flood": (rock.residual_oil_saturation_gas_flood),
        "rock.residual_gas_saturation": rock.residual_gas_saturation,
    }
    for name, array in checks.items():
        if array.shape != (n_cells,):
            raise ValidationError(f"`{name}` has shape {array.shape}; expected ({n_cells},).")
    return rock


class Reservoir(
    Serializable,
    fields={
        "grid": Grid,
        "rock": Rock,
        "regions": Regions | None,
        "unit_system": UnitSystem | None,
    },
):
    """
    Reservoir characterization.

    Binds a polyhedral `Grid` to per-cell rock, faults, regions, boundary conditions,
    etc. On construction all property groups are normalised to the
    declared `unit_system` (defaults to the grid's own unit system).
    """

    __slots__ = (
        "_face_transmissibility_map",
        "_transmissibilities",
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
        :param unit_system: Target unit system for all property groups. When `None`, defaults
            to `grid.unit_system`.
        :raises ValidationError: If any array length in `rock`, `state`, or `hysteresis` does not
            match `grid.n_cells`, or if `datum_depth` is negative.
        """
        n_cells = grid.n_cells
        validate_rock(rock, n_cells)

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

        self.unit_system = target_unit_system
        """
        Unit system in which all property groups are expressed.

        Matches `grid.unit_system` by construction.
        """
        self._transmissibilities: ConnectionTransmissibilities | None = None
        self._face_transmissibility_map: dict[int, Number] | None = None

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

        Computed as `φ x NTG x Vcell`.

        Units: ft³ (FIELD), m³ (METRIC / SI), cm³ (LAB).
        """
        assert self.grid.cell_volumes is not None, (
            "`grid.cell_volumes` is None; the grid was constructed without "
            "pre-computed volumes and the divergence-theorem computation failed."
        )
        return (  # type: ignore[return-value]
            self.rock.porosity * self.rock.net_to_gross * self.grid.cell_volumes
        )

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
        - `interior_map`  - global face index for each interior entry.
        - `boundary_map`  - global face index for each boundary entry.

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

    @property
    def face_transmissibility_map(self) -> dict[int, Number]:
        if self._face_transmissibility_map is None:
            self._face_transmissibility_map = get_face_transmissibility_map(
                self.grid, self.transmissibilities
            )
        return self._face_transmissibility_map

    def invalidate_transmissibilities(self) -> None:
        """
        Clear the cached transmissibilities.

        Call this whenever `rock.absolute_permeability` or `rock.net_to_gross`
        is updated (e.g. history-matching) so the next access to
        `transmissibilities` triggers a fresh computation.
        """
        self._transmissibilities = None
        self._face_transmissibility_map = None

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
        get_conversion_factors(self.unit_system, target, table=table)
        new_model = self.__class__(
            grid=self.grid.convert(target, table=table),
            rock=self.rock.convert(target, table=table),
            regions=self.regions,
            unit_system=target,
        )
        # Transmissibility cache is invalidated automatically since rock was
        # converted and the new model starts with a clean cache.
        return new_model

    def get_face_transmissibility(self, face_index: int) -> Number:
        """
        Return the transmissibility (or half-transmissibility for boundary) of a single face.

        :param face_index: Index into `grid.face_cell_indices`.
        :returns: Transmissibility in grid units (mD·ft in FIELD etc.).
        :raises KeyError: If `face_index` is not a valid face.
        """
        t = self.face_transmissibility_map.get(face_index)
        if t is None:
            raise KeyError(
                f"Face {face_index} not found in interior or boundary faces of this grid."
            )
        return t

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
            f"unit_system={self.unit_system.value!r}, "
            f")"
        )
