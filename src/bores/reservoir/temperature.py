import typing

import attrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.constants import UnitConversionTable, get_conversion_factors
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.serde.stores import StoreSerializable
from bores.types import (
    CellArray,
    IntCellArray,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
    UnitSystem,
)
from bores.utils import scale, scale_and_offset

__all__ = ["Temperature", "TemperatureGradient", "TemperatureTable"]


@attrs.frozen(slots=True)
class TemperatureGradient(StoreSerializable):
    """
    A linear temperature-depth gradient for a single PVT region.

    Temperature at a given depth is computed as:

        T(d) = reference_temperature + gradient * (depth - reference_depth)

    where positive `gradient` means temperature increases with depth
    (the physically normal case in a geothermal context).
    """

    reference_temperature: Number
    """Temperature at `reference_depth` in `unit_system` temperature units."""

    reference_depth: Number
    """Datum depth in `unit_system` length units (positive downward)."""

    gradient: Number
    """
    Temperature change per unit depth (temperature/length in `unit_system` 
    e.g. °F/ft in FIELD, °C/m in METRIC). Positive = warmer with depth.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system for all dimensional parameters."""

    def at_depth(self, depth: NumberOrArray[OneDimension]) -> NumberOrArray[OneDimension]:
        """
        Return temperature at the given depth or array of depths.

        :param depth: Depth(s) in `unit_system` length units (positive downward).
        :returns: Temperature value(s) in `unit_system` temperature units.
        """
        return typing.cast(
            NumberOrArray[OneDimension],
            self.reference_temperature + self.gradient * (depth - self.reference_depth),
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `TemperatureGradient` with all dimensional parameters
        rescaled to *target*.

        The `gradient` (temperature/length) is rescaled by the temperature
        factor divided by the length factor. `reference_temperature` uses
        the temperature factor and offset. `reference_depth` uses the
        length factor.

        :param target: Target `UnitSystem`.
        :param table: Optional custom conversion table.
        :returns: New `TemperatureGradient` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        return attrs.evolve(
            self,
            reference_temperature=scale_and_offset(
                self.reference_temperature,
                factor=factors["temperature"],
                offset=factors["temperature_offset"],
            ),
            reference_depth=scale(self.reference_depth, factors["length"]),
            gradient=scale(self.gradient, factors["temperature"] / factors["length"]),
            unit_system=target,
        )


@attrs.frozen(slots=True)
class TemperatureTable(StoreSerializable):
    """
    A depth-indexed temperature lookup table for a single PVT / equilibration
    region, corresponding to one table from the Eclipse `TEMPVD` keyword.

    Temperature at a cell centroid depth is linearly interpolated from the
    `(depth, temperature)` pairs. Values above the shallowest entry or
    below the deepest entry are clamped to the respective endpoint. No
    extrapolation is performed, consistent with Eclipse behaviour.
    """

    depths: NumberArray[OneDimension]
    """1-D strictly increasing depth array in `unit_system` length units."""

    temperatures: NumberArray[OneDimension]
    """
    1-D temperature array in `unit_system` temperature units.
    Must have the same length as `depths`.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system for all dimensional parameters."""

    def __attrs_post_init__(self) -> None:
        if self.depths.ndim != 1:
            raise ValidationError("`depths` must be a 1-D array.")
        if self.temperatures.ndim != 1:
            raise ValidationError("`temperatures` must be a 1-D array.")
        if len(self.depths) != len(self.temperatures):
            raise ValidationError(
                f"`depths` length {len(self.depths)} must match "
                f"`temperatures` length {len(self.temperatures)}."
            )
        if len(self.depths) < 2:
            raise ValidationError(
                "`TemperatureTable` requires at least 2 (depth, temperature) pairs."
            )
        if not np.all(np.diff(self.depths) > 0):
            raise ValidationError("`depths` must be strictly increasing.")

    def at_depth(self, depth: NumberOrArray[OneDimension]) -> NumberOrArray[OneDimension]:
        """
        Return temperature at the given depth or array of depths.

        Linearly interpolates within the table bounds; clamps to endpoint
        values outside the range (no extrapolation).

        :param depth: Scalar or shape `(n_cells,)` depth array in
            `unit_system` length units.
        :returns: Temperature value(s) in `unit_system` temperature units.
        """
        is_scalar = np.isscalar(depth)
        depth_arr = np.atleast_1d(depth)
        result = np.interp(depth_arr, self.depths, self.temperatures)
        return (
            typing.cast(Number, result[0])
            if is_scalar
            else typing.cast(NumberArray[OneDimension], result)
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `TemperatureTable` with depths and temperatures
        rescaled to *target*.

        :param target: Target `UnitSystem`.
        :param table: Optional custom conversion table.
        :returns: New `TemperatureTable` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        return attrs.evolve(
            self,
            depths=scale(self.depths, factors["length"]),  # type: ignore[arg-type]
            temperatures=scale_and_offset(
                self.temperatures,
                factor=factors["temperature"],
                offset=factors["temperature_offset"],
            ),  # type: ignore[arg-type]
            unit_system=target,
        )


TemperatureSpec = Number | TemperatureGradient | TemperatureTable


@attrs.frozen(slots=True)
class Temperature(StoreSerializable):
    """
    Reservoir temperature specification supporting per-region scalars,
    gradients, and depth-indexed tables.

    Represents either a single uniform reservoir temperature (`default`)
    or a mapping of 1-based PVT/EQL region indices
    (depending on what its is regioned by) to one of:

    - A scalar `Number` - uniform temperature for that region.
    - A `TemperatureGradient` - linear temperature-depth profile.
    - A `TemperatureTable` - `TEMPVD`-style depth-indexed table.

    A special key `-1` in `regions` stores the fallback value used for
    cells whose region index is not explicitly listed.

    `as_cell_array` broadcasts the per-region specification to a per-cell
    array using a depth array and a region-index array. Use `convert`
    to produce a copy in a different `UnitSystem`.

    **Construction**:

    The simplest case - one temperature for all cells:

    ```python
    temperature = Temperature(default=200.0, unit_system=UnitSystem.FIELD)
    ```

    Per-region scalars:
    ```python
    temperature = Temperature(
        regions={1: 180.0, 2: 210.0},
        unit_system=UnitSystem.FIELD,
    )
    ```

    Mixed - some regions use a gradient, one a table:

    ```python
    temperature = Temperature(
        default=200.0,
        regions={
            1: TemperatureGradient(reference_temperature=180.0, ...),
            2: TemperatureTable(depths=..., temperatures=...),
        },
        unit_system=UnitSystem.FIELD,
    )
    ```
    """

    default: TemperatureSpec | None = None
    """
    Fallback temperature specification applied to cells whose region index
    is absent from `regions`, or when `regions` is `None`. Must be
    supplied when `regions` is `None` or empty.
    """

    regions: dict[int, TemperatureSpec] | None = None
    """
    Mapping from 1-based PVT/EQL region index to a temperature specification.
    Key `-1` is reserved as the internal fallback; it is set automatically
    from `default` on initialization and must not be supplied
    directly by the caller.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system for all temperature and depth values stored here."""

    def __attrs_post_init__(self) -> None:
        if self.default is None and not self.regions:
            raise ValidationError("Either `default` or `regions` must be provided.")

        regions: dict[int, TemperatureSpec] = dict(self.regions or {})

        # Resolve the fallback (-1 key)
        if -1 not in regions:
            if self.default is not None:
                regions[-1] = self.default
            else:
                # No explicit default - use the first region's spec as fallback
                first = next(iter(regions.values()))
                regions[-1] = first

        object.__setattr__(self, "regions", regions)

    def region(self, num: int) -> TemperatureSpec:
        """
        Return the temperature specification for the given 1-based region index.

        Falls back to the default (key `-1`) when `num` is not in
        `regions`.

        :param num: 1-based PVT/EQL region index.
        :returns: `Number`, `TemperatureGradient`, or `TemperatureTable`.
        """
        assert self.regions is not None
        if num in self.regions:
            return self.regions[num]
        return self.regions[-1]

    def as_cell_array(
        self,
        x_regions: IntCellArray,
        cell_depths: CellArray,
        dtype: npt.DTypeLike = None,
    ) -> CellArray:
        """
        Broadcast per-region temperature specifications to a per-cell array.

        For each unique region index in `region`:

        - Scalar `Number` - fills all cells in the region with that value.
        - `TemperatureGradient` - evaluates `gradient.at_depth(depth)`
          per cell.
        - `TemperatureTable` - interpolates `table.at_depth(depth)` per
          cell.

        Cells whose region index is absent from `regions` use the fallback
        at key `-1`.

        :param x_regions: Shape `(n_cells,)` int array of 1-based PVT/EQL region
            indices.
        :param cell_depths: Shape `(n_cells,)` depth array in
            `unit_system` length units (positive downward). Required for
            `TemperatureGradient` and `TemperatureTable` specs; ignored
            (but still required) for scalar specs.
        :param dtype: Output dtype; defaults to `get_dtype()`.
        :returns: Shape `(n_cells,)` `CellArray` of temperature values in
            `unit_system` temperature units.
        """
        assert self.regions is not None
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        n_cells = x_regions.size
        default_spec = self.regions[-1]

        # Pre-fill with default
        if isinstance(default_spec, (TemperatureGradient, TemperatureTable)):
            out = default_spec.at_depth(cell_depths).astype(dtype, copy=False)  # type: ignore[union-attr]
        else:
            out = np.full(n_cells, default_spec, dtype=dtype)

        # Overwrite cells belonging to explicitly listed regions
        for region_idx, spec in self.regions.items():
            if region_idx == -1:
                continue

            xnum = region_idx + 1
            mask = x_regions == xnum
            if not np.any(mask):
                continue
            if isinstance(spec, (TemperatureGradient, TemperatureTable)):
                out[mask] = spec.at_depth(cell_depths[mask]).astype(dtype, copy=False)  # type: ignore[arg-type, union-attr]
            else:
                out[mask] = dtype.type(spec)  # type: ignore[arg-type]

        return typing.cast(CellArray, out)

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a copy with all temperature specifications converted to *target*.

        Scalar values are rescaled using the temperature factor and offset.
        `TemperatureGradient` and `TemperatureTable` instances delegate
        to their own `convert` methods.

        :param target: Target `UnitSystem`.
        :param table: Optional custom conversion table.
        :returns: New `Temperature` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        factor = factors["temperature"]
        offset = factors["temperature_offset"]

        def _convert_spec(spec: TemperatureSpec) -> TemperatureSpec:
            if isinstance(spec, (TemperatureGradient, TemperatureTable)):
                return spec.convert(target, table=table)
            return scale_and_offset(spec, factor=factor, offset=offset)

        new_default = _convert_spec(self.default) if self.default is not None else None
        new_regions: dict[int, TemperatureSpec] | None = None
        if self.regions is not None:
            new_regions = {k: _convert_spec(v) for k, v in self.regions.items()}

        return attrs.evolve(self, default=new_default, regions=new_regions, unit_system=target)

    @classmethod
    def from_deck(cls, deck_file: DeckFile, *, dtype: npt.DTypeLike = None) -> Self:
        """
        Build a `Temperature` from a parsed `DeckFile`.

        Keyword priority (highest to lowest):

        1. `TEMPVD`: depth-indexed table, one per PVT/EQL region.
           Each table becomes a `TemperatureTable` keyed by its 1-based
           PVT/EQL region index. When `EQLNUM` and `PVTNUM`
           differ in the deck this may require the caller to remap keys;
           the method assumes each `TEMPVD` table index maps to the same-
           numbered PVT/EQL region.
        2. `RTEMP`: single scalar applied uniformly as the `default`.
        3. Neither present: returns `None`; caller is responsible for
           supplying a default temperature.

        :param deck_file: Parsed `DeckFile` containing SOLUTION-section
            keywords.
        :returns: `Temperature` or raises `ValidationError` when
            no temperature keyword is found.
        :raises ValidationError: If neither `TEMPVD` nor `RTEMP` is
            present in the deck.
        """
        unit_system = deck_file.unit_system
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()

        tempvd_records: list[list[dict[str, typing.Any]]] | None = deck_file.get(
            "TEMPVD"
        ) or deck_file.get("RTEMPVD")
        rtemp_records: list[list[dict[str, typing.Any]]] | None = deck_file.get("RTEMP")

        # `TEMPVD`: one `TemperatureTable` per equilibration or PVT region,
        # depending on whether it is regioned by `PVTNUM` or `EQLNUM`
        if tempvd_records:
            regions: dict[int, TemperatureSpec] = {}
            for region_idx, rows in enumerate(tempvd_records):
                if not rows:
                    continue

                region_num = region_idx + 1  # 1-based
                depths = np.array([row["depth"] for row in rows], dtype=dtype)
                temperatures = np.array([row["temperature"] for row in rows], dtype=dtype)
                if not np.all(np.diff(depths) > 0):
                    raise ValidationError(
                        f"`TEMPVD` region {region_num}: `depth` values must be strictly increasing."
                    )
                regions[region_num] = TemperatureTable(
                    depths=typing.cast(NumberArray[OneDimension], depths),
                    temperatures=typing.cast(NumberArray[OneDimension], temperatures),
                    unit_system=unit_system,
                )

            if not regions:
                raise ValidationError("`TEMPVD` keyword is present but contains no valid rows.")
            return cls(regions=regions, unit_system=unit_system)

        # RTEMP: single scalar default
        if rtemp_records and rtemp_records[0]:
            temperature: Number = float(rtemp_records[0][0]["temperature"])
            return cls(default=temperature, unit_system=unit_system)

        raise ValidationError(
            "No temperature keyword found in the DeckFile. "
            "Expected `TEMPVD` (depth-indexed table) or `RTEMP` (single scalar). "
            f"Supply a temperature explicitly via `{cls.__name__}(default=...)`."
        )
