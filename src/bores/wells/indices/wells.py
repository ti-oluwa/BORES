"""
Well index (connection factor) computation.

Resolves each open perforation to a connection factor (index) per cell it
intersects. The pipeline is three steps.

First, an effective (direction-aware, geometric-mean) permeability is
derived first, then an effective drainage radius from that permeability
and the local cell geometry, then the well index itself from the standard
Peaceman equation.

Cells that aren't locally Cartesian-like fall back to an isotropic equivalent-radius
formulation instead, since Peaceman's anisotropic radius formula assumes a clean
local (dx, dy, dz) frame that a distorted or unstructured cell doesn't have.
"""

import typing

import attrs
import numba
import numpy as np
from typing_extensions import Self

from bores.constants import get_conversion_factors
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.serde.base import Serializable
from bores.types import (
    Integer,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
    Orientation,
    UnitConversionTable,
    UnitSystem,
)
from bores.utils import scale
from bores.wells.base import AnyPerforation, MDPerforation, Perforation, Wells
from bores.wells.indices.perforations import (
    PerforationIndex,
    resolve_md_perforations_indices,
    resolve_perforations_indices,
)

__all__ = [
    "WellIndex",
    "build_wells_indices",
    "compute_2D_effective_drainage_radius",
    "compute_3D_effective_drainage_radius",
    "compute_effective_permeability_for_well",
    "compute_equivalent_radius_well_index",
    "compute_peaceman_well_index",
    "is_locally_cartesian",
    "resolve_connection_factor",
    "resolve_well_index_direction",
]


@attrs.frozen(kw_only=True, slots=True)
class WellIndex(Serializable):
    """Computed well indices for all perforations of a single well."""

    well_name: str
    """Name of the well these indices belong to."""

    perforations: tuple[PerforationIndex, ...] = attrs.field(converter=tuple)
    """Resolved connection factor for each of this well's open perforations."""

    total_well_index: Number
    """Sum of every perforation's well index."""

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system these indices are expressed in."""

    def __attrs_post_init__(self) -> None:
        mismatched = [
            perforation
            for perforation in self.perforations
            if perforation.unit_system != self.unit_system
        ]
        if mismatched:
            raise ValidationError(
                f"All `perforations` must share this WellIndex's unit_system "
                f"({self.unit_system.value}); found {mismatched[0].unit_system.value}."
            )

    def get_allocation_fraction(self, perforation: PerforationIndex) -> Number:
        """
        Gets the fraction of the well's total connection factor a given
        perforation contributes.

        :param perforation: One of this well's perforations.
        :returns: `perforation.well_index / total_well_index`, or `1.0` if
            `total_well_index` isn't positive.
        :raises ValidationError: If `perforation.well_index` hasn't been resolved.
        """
        if self.total_well_index <= 0:
            return 1.0
        if perforation.well_index is None:
            raise ValidationError(
                "`perforation.well_index` is None. This `PerforationIndex` "
                "hasn't been resolved by `build_wells_indices` yet."
            )
        return perforation.well_index / self.total_well_index

    def __iter__(self) -> typing.Iterator[PerforationIndex]:
        return iter(self.perforations)

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Returns a  new `WellIndex` in the *target* unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New `WellIndex` with every `PerforationIndex` and
            `total_well_index` converted to target.
        """
        if target == self.unit_system:
            return self
        factors = get_conversion_factors(self.unit_system, target, table=table)
        index_factor = factors["permeability"] * factors["length"]
        return attrs.evolve(
            self,
            perforations=tuple(
                perforation.convert(target, table=table) for perforation in self.perforations
            ),
            total_well_index=scale(self.total_well_index, index_factor),
            unit_system=target,
        )


def resolve_well_index_direction(
    perforation: Perforation, grid: Grid, cell_index: Integer
) -> Orientation:
    """
    Resolve which axis a Peaceman-style well index should treat as the
    "along-wellbore" direction for `perforation` in `cell_index`.

    Resolution order: `perforation.direction` if set (and not `UNSET`);
    otherwise the cell's geometrically thinnest axis. The well is assumed
    to pierce perpendicular to the layering, which for a typical grid is
    Z, but isn't guaranteed to be for a rotated or irregularly-shaped cell.

    :param perforation: The perforation being resolved.
    :param grid: Grid the cell belongs to.
    :param cell_index: Cell the perforation resolved into.
    :returns: `Orientation.X`, `Orientation.Y`, or `Orientation.Z`.
    """
    if perforation.direction is not None and perforation.direction is not Orientation.UNSET:
        return perforation.direction

    lengths = {
        Orientation.X: grid.cell_length_x[cell_index],
        Orientation.Y: grid.cell_length_y[cell_index],
        Orientation.Z: grid.cell_length_z[cell_index],
    }
    return min(lengths, key=lengths.get)  # type: ignore[arg-type]


def is_locally_cartesian(grid: Grid, cell_index: Integer, tolerance: Number = 0.05) -> bool:
    """
    Decide whether a cell is "Cartesian-like enough" for Peaceman's
    formula to be valid, vs. requiring the equivalent-radius fallback.

    Compares the cell's true volume against the volume of its own AABB; if
    the ratio is within `tolerance` of 1.0, the cell is box-like enough for
    Peaceman.

    :param grid: Grid providing `cell_volumes`, `cell_length_x/y/z`.
    :param cell_index: Cell to classify.
    :param tolerance: Allowed fractional deviation from a perfect box.
    :returns: `True` if Peaceman should be used, `False` for the fallback.
    """
    aabb_volume = (
        grid.cell_length_x[cell_index]
        * grid.cell_length_y[cell_index]
        * grid.cell_length_z[cell_index]
    )
    if aabb_volume <= 0.0:
        return False

    assert grid.cell_volumes is not None
    ratio = grid.cell_volumes[cell_index] / aabb_volume
    return abs(ratio - 1.0) <= tolerance


@numba.njit(cache=True)
def compute_peaceman_well_index(
    permeability: Number,
    interval_thickness: Number,
    wellbore_radius: Number,
    effective_drainage_radius: Number,
    skin_factor: Number = 0.0,
    net_to_gross: Number = 1.0,
    regime_constant: Number = -3 / 4,
) -> Number:
    """
    Compute the well index for a given well using the Peaceman equation.

    W = (k * h * N/G) / (ln(re/rw) + C + s)

    :param permeability: Effective permeability at the cell (mD).
    :param interval_thickness: Completion length within the cell (ft).
    :param wellbore_radius: Radius of the wellbore (ft).
    :param effective_drainage_radius: Effective drainage radius (ft).
    :param skin_factor: Skin factor for the well (dimensionless).
    :param net_to_gross: Net-to-gross ratio of the reservoir interval.
    :param regime_constant: 0 for steady, -3/4 for pseudo steady, 1/2 for transient.
    :return: The well index (mD*ft).
    """
    return (permeability * interval_thickness * net_to_gross) / (
        np.log(effective_drainage_radius / wellbore_radius) + regime_constant + skin_factor
    )


@numba.njit(cache=True)
def compute_3D_effective_drainage_radius(
    interval_thickness: tuple[Number, Number, Number],
    permeability: tuple[Number, Number, Number],
    well_orientation: Orientation,
) -> Number:
    """
    Compute the effective drainage radius for a well in a 3D reservoir
    model using Peaceman's (1983) anisotropic effective drainage radius
    formula.

    :param interval_thickness: Tuple of cell dimensions (dx, dy, dz) in ft.
    :param permeability: Tuple of permeabilities (kx, ky, kz) in mD.
    :param well_orientation: Wellbore axis.
    :return: Effective drainage (Peaceman) radius in ft, or 0.0 if either
        perpendicular permeability is zero.
    """
    dx, dy, dz = interval_thickness
    kx, ky, kz = permeability

    if well_orientation == Orientation.X:
        if ky <= 0.0 or kz <= 0.0:
            return 0.0
        r1, r2 = ky / kz, kz / ky
        numerator = np.sqrt(r1) * dy**2 + np.sqrt(r2) * dz**2
        denominator = r1**0.25 + r2**0.25
    elif well_orientation == Orientation.Y:
        if kx <= 0.0 or kz <= 0.0:
            return 0.0
        r1, r2 = kx / kz, kz / kx
        numerator = np.sqrt(r1) * dx**2 + np.sqrt(r2) * dz**2
        denominator = r1**0.25 + r2**0.25
    elif well_orientation == Orientation.Z:
        if kx <= 0.0 or ky <= 0.0:
            return 0.0
        r1, r2 = ky / kx, kx / ky
        numerator = np.sqrt(r1) * dx**2 + np.sqrt(r2) * dy**2
        denominator = r1**0.25 + r2**0.25
    else:
        raise ValidationError("Invalid well orientation")

    return 0.28 * np.sqrt(numerator / denominator)


@numba.njit(cache=True)
def compute_2D_effective_drainage_radius(
    interval_thickness: tuple[Number, Number],
    permeability: tuple[Number, Number],
) -> Number:
    """
    Compute the effective drainage radius for a well in a 2D reservoir
    model using Peaceman's (1983) anisotropic formula.

    :param interval_thickness: Tuple of cell dimensions (dx, dy) in ft.
    :param permeability: Tuple of permeabilities (kx, ky) in mD.
    :return: Effective drainage (Peaceman) radius in ft, or 0.0 if either
        permeability is zero or negative.
    """
    kx, ky = permeability
    if kx <= 0.0 or ky <= 0.0:
        return 0.0

    dx, dy = interval_thickness[0], interval_thickness[1]
    r1, r2 = ky / kx, kx / ky
    numerator = np.sqrt(r1) * dx**2 + np.sqrt(r2) * dy**2
    denominator = r1**0.25 + r2**0.25
    return 0.28 * np.sqrt(numerator / denominator)


@numba.njit(cache=True, inline="always")
def compute_geometric_mean(values: typing.Sequence[Number]) -> Number:
    """Geometric mean of `values`, clamping any negative input to zero."""
    product = 1.0
    n = 0
    for v in values:
        product *= max(v, 0.0)
        n += 1
    if n == 0:
        raise ValidationError("No permeability values provided")
    return product ** (1.0 / n)


@numba.njit(cache=True)
def compute_effective_permeability_for_well(
    permeability: tuple[Number, Number, Number], orientation: Orientation
) -> Number:
    """
    Compute `k_eff` for Peaceman WI using geometric mean of the two
    permeabilities perpendicular to the well axis.

    :param permeability: `(kx, ky, kz)`.
    :param orientation: `Orientation.X`/`Y`/`Z`.
    :return: Effective (geometric-mean) permeability.
    """
    kx, ky, kz = permeability
    if orientation == Orientation.Z:
        return np.sqrt(max(kx, 0.0) * max(ky, 0.0))
    elif orientation == Orientation.X:
        return np.sqrt(max(ky, 0.0) * max(kz, 0.0))
    elif orientation == Orientation.Y:
        return np.sqrt(max(kx, 0.0) * max(kz, 0.0))
    return compute_geometric_mean((kx, ky, kz))


@numba.njit(cache=True)
def compute_equivalent_radius_well_index(
    permeability: Number,
    cell_volume: Number,
    completion_length: Number,
    wellbore_radius: Number,
    skin: Number,
) -> Number:
    """
    Well index for a cell where `is_locally_cartesian` returned `False`.

    Isotropic equivalent-radius formulation: `r_0 = 0.14 * sqrt(cell_volume
    / completion_length)`, the volume-per-unit-length analogue of
    Peaceman's `r_0 = 0.14 * sqrt(dx**2 + dy**2)`, valid without assuming a
    clean local (dx, dy) pair.

    :param permeability: Isotropic (or geometric-mean) permeability (mD).
    :param cell_volume: `grid.cell_volumes[cell_index]`.
    :param completion_length: Perforation length within this cell (ft).
    :param wellbore_radius: Perforation or Well wellbore radius (ft).
    :param skin: Perforation skin factor.
    :returns: Well index (mD*ft).
    """
    r0 = 0.14 * np.sqrt(cell_volume / completion_length)
    return 2.0 * np.pi * permeability * completion_length / (np.log(r0 / wellbore_radius) + skin)


def resolve_connection_factor(
    perforation: AnyPerforation,
    grid: Grid,
    cell_index: Integer,
    partial_penetration_fraction: Number,
    wellbore_radius: Number,
    permeabilities: typing.Mapping[Orientation, Number],
    regime_constant: float = -3 / 4,
    net_to_gross: float = 1.0,
) -> Number:
    """
    Resolves the connection factor for one (perforation, cell) pair.

    Resolution order: `perforation.connection_factor_override` if set;
    otherwise Peaceman (`compute_peaceman_well_index`) if `is_locally_cartesian`;
    otherwise the equivalent-radius fallback (`compute_equivalent_radius_well_index`).

    :param perforation: Source perforation (for skin, override).
    :param grid: Grid providing geometry.
    :param cell_index: Resolved cell.
    :param partial_penetration_fraction: From the matching `PerforationIndex`.
    :param wellbore_radius: Already-resolved radius (perforation override or
        well default).
    :param permeabilities: Per-axis permeability at `cell_index`, keyed by
        `Orientation.X`/`Y`/`Z`.
    :param regime_constant: Forwarded to `compute_peaceman_well_index`.
    :param net_to_gross: Forwarded to `compute_peaceman_well_index`.
    :returns: Connection factor / well index for this (perforation, cell) pair.
    """
    if perforation.connection_factor_override is not None:
        return perforation.connection_factor_override

    kx = permeabilities[Orientation.X]
    ky = permeabilities[Orientation.Y]
    kz = permeabilities[Orientation.Z]

    if isinstance(perforation, MDPerforation):
        # No discrete axis to run Peaceman against so we always use isotropic
        # equivalent-radius, using the geometric-mean permeability and this
        # connection's true (MD-fraction-scaled) length within the cell.
        effective_permeability = compute_geometric_mean((kx, ky, kz))
        completion_length = partial_penetration_fraction * perforation.length
        assert grid.cell_volumes is not None
        well_index = compute_equivalent_radius_well_index(
            permeability=effective_permeability,
            cell_volume=grid.cell_volumes[cell_index],
            completion_length=completion_length,
            wellbore_radius=wellbore_radius,
            skin=perforation.skin,
        )
        if perforation.connection_factor_multiplier is not None:
            well_index *= perforation.connection_factor_multiplier
        return well_index

    # Perforation (TVD, axis-aligned).
    direction = resolve_well_index_direction(perforation, grid, cell_index)
    cell_axes_lengths = {
        Orientation.X: grid.cell_length_x[cell_index],
        Orientation.Y: grid.cell_length_y[cell_index],
        Orientation.Z: grid.cell_length_z[cell_index],
    }
    completion_length = partial_penetration_fraction * cell_axes_lengths[direction]
    effective_permeability = compute_effective_permeability_for_well(
        permeability=(kx, ky, kz), orientation=direction
    )

    if is_locally_cartesian(grid, cell_index):
        effective_drainage_radius = compute_3D_effective_drainage_radius(
            interval_thickness=(
                grid.cell_length_x[cell_index],
                grid.cell_length_y[cell_index],
                grid.cell_length_z[cell_index],
            ),
            permeability=(kx, ky, kz),
            well_orientation=direction,
        )
        well_index = compute_peaceman_well_index(
            permeability=effective_permeability,
            interval_thickness=completion_length,
            wellbore_radius=wellbore_radius,
            effective_drainage_radius=effective_drainage_radius,
            skin_factor=perforation.skin,
            net_to_gross=net_to_gross,
            regime_constant=regime_constant,
        )
        if perforation.connection_factor_multiplier is not None:
            well_index *= perforation.connection_factor_multiplier
        return well_index

    assert grid.cell_volumes is not None
    well_index = compute_equivalent_radius_well_index(
        permeability=effective_permeability,
        cell_volume=grid.cell_volumes[cell_index],
        completion_length=completion_length,
        wellbore_radius=wellbore_radius,
        skin=perforation.skin,
    )
    if perforation.connection_factor_multiplier is not None:
        well_index *= perforation.connection_factor_multiplier
    return well_index


def build_wells_indices(
    grid: Grid,
    wells: Wells,
    permeabilities: typing.Mapping[Orientation, NumberArray[OneDimension]],
    regime_constant: float = -3 / 4,
    net_to_gross: NumberOrArray[OneDimension] = 1.0,
    **resolve_kwargs: typing.Any,
) -> dict[str, WellIndex]:
    """
    Resolve every well in `wells` against `grid` and compute connection
    factors for every open perforation.

    :param grid: Grid to resolve against.
    :param wells: `Wells` container (all wells to resolve).
    :param permeabilities: Per-axis permeability arrays, shape `(n_cells,)`
        each, keyed by `Orientation.X`/`Y`/`Z`.
    :param resolve_kwargs: Passed through to `resolve_perforations_indices`.
    :returns: Mapping from well name to `WellIndex`.
    :raises ValidationError: Propagated from `resolve_perforations_indices` for any
        well with a dangling completion.
    """
    if grid.unit_system != wells.unit_system:
        raise ValidationError(
            f"Grid `unit_system` ({grid.unit_system.value}) != Wells "
            f"`unit_system` ({wells.unit_system.value})."
        )

    result: dict[str, WellIndex] = {}
    for name in wells:
        well = wells[name]
        if well.trajectory is None:
            perforation_indices = resolve_perforations_indices(
                grid=grid, well=well, **resolve_kwargs
            )
        else:
            perforation_indices = resolve_md_perforations_indices(
                grid=grid, well=well, **resolve_kwargs
            )

        resolved: list[PerforationIndex] = []
        total_well_index = 0.0
        for perforation_index in perforation_indices:
            perforation = perforation_index.perforation
            cell_idx = perforation_index.cell_index
            wellbore_radius = perforation.wellbore_radius
            cell_permeabilities = {axis: array[cell_idx] for axis, array in permeabilities.items()}
            cell_net_to_gross = (
                net_to_gross[cell_idx] if isinstance(net_to_gross, np.ndarray) else 1.0
            )
            well_index = resolve_connection_factor(
                perforation=perforation,
                grid=grid,
                cell_index=cell_idx,
                partial_penetration_fraction=perforation_index.partial_penetration_fraction,
                wellbore_radius=wellbore_radius,
                permeabilities=cell_permeabilities,
                regime_constant=regime_constant,
                net_to_gross=cell_net_to_gross,
            )
            conductivity = None
            if perforation.connection_factor_override is None:
                # Isolates the Peaceman numerator (k*h*N/G) by evaluating the
                # same connection a second time with skin shifted by +1 and
                # solving the two-point system: 1/numerator = 1/WI(skin+1) - 1/WI(skin).
                # This works for any starting skin, including zero, and never
                # touches well_index itself.
                perturbed_well_index = resolve_connection_factor(
                    perforation=attrs.evolve(perforation, skin=perforation.skin + 1.0),
                    grid=grid,
                    cell_index=cell_idx,
                    partial_penetration_fraction=perforation_index.partial_penetration_fraction,
                    wellbore_radius=wellbore_radius,
                    permeabilities=cell_permeabilities,
                    regime_constant=regime_constant,
                    net_to_gross=cell_net_to_gross,
                )
                inverse_difference = (1.0 / perturbed_well_index) - (1.0 / well_index)
                if inverse_difference != 0:
                    conductivity = 1.0 / inverse_difference
            resolved.append(
                attrs.evolve(perforation_index, well_index=well_index, conductivity=conductivity)
            )
            total_well_index += well_index

        result[well.name] = WellIndex(
            well_name=well.name,
            perforations=tuple(resolved),
            total_well_index=total_well_index,
            unit_system=wells.unit_system,
        )
    return result
